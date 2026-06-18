"""
Real-time TCN inference with rtmlib skeleton extraction.

rtmlib Wholebody → 133 keypoints → face 제거 (65 joints) or body 17 → 프레임 버퍼 → TCN 추론

65-joint layout (face 제거):
    0-16:  Body (COCO 17)
   17-22:  Feet (6)
   23-43:  Left Hand (21)
   44-64:  Right Hand (21)

Usage:
    # 웹캠
    python src/inference_tcn.py

    # 비디오 파일
    python src/inference_tcn.py --source video.mp4

    # 이미지 (프레임 반복하여 시퀀스 생성)
    python src/inference_tcn.py --source image.jpg
"""

import sys
import os
import argparse
import time
from collections import deque

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import cv2
import torch

from models.tcn import TCN
from utils.track_state import TrackState, TrackManager, match_centers_to_tracks
from runtime.realtime_inference_common import (
    format_track_state_line,
    get_shoulder_center,
    init_pose_tracker,
    open_video_capture,
    parse_video_source,
    update_track_from_keypoints,
    warmup_capture,
)
from runtime.realtime_viz import draw_basic_hud, draw_prob_bars, draw_track_label
from runtime.tcn_runtime import (
    extract_skeleton,
    normalize_frame,
    predict_tcn,
    predict_tcn_batch,
)


# ===== 클래스별 표시 색상 (BGR) =====
CLASS_COLORS = {
    "idle":             (128, 128, 128),  # gray
    "waving":           (0, 165, 255),    # orange
    "hands_up_single":  (0, 0, 255),      # red
    "hands_up_both":    (0, 100, 255),    # dark orange
    "pointing":         (0, 255, 0),      # green
    "stop":             (255, 0, 0),      # blue
    "unknown":          (80, 80, 80),     # dark gray
}


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time TCN action recognition")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: webcam index (0,1,...) or file path")
    parser.add_argument("--model", type=str, default=None,
                        help="TCN checkpoint (default: models/best_tcn_xsub.pth)")
    parser.add_argument("--rtmlib_path", type=str,
                        default="rtmlib")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["balanced", "performance", "lightweight"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--kpt_thr", type=float, default=0.4,
                        help="Keypoint threshold for skeleton drawing")
    parser.add_argument("--buf_size", type=int, default=60,
                        help="Frame buffer size for TCN (default: 60)")
    parser.add_argument("--infer_every", type=int, default=5,
                        help="Run TCN every N frames (default: 5)")
    parser.add_argument("--max_tracks_infer", type=int, default=8,
                        help="Max number of tracks to run TCN on per inference step")
    parser.add_argument("--det_frequency", type=int, default=1,
                        help="Pose detector frequency for rtmlib PoseTracker (lower is more stable)")
    parser.add_argument("--track_match_dist", type=float, default=120.0,
                        help="Pixel distance threshold for local track matching")
    parser.add_argument("--grace_frames", type=int, default=15,
                        help="Keep disconnected tracks reconnectable for N frames")
    parser.add_argument("--ttl_frames", type=int, default=45,
                        help="Remove track if missing for more than N frames")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────

def load_tcn(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = TCN(
        input_dim=checkpoint.get("input_dim", 130),
        num_classes=checkpoint.get("num_classes", 6),
        hidden_dims=checkpoint.get("hidden_dims", [64, 128, 128, 256]),
        kernel_size=checkpoint.get("kernel_size", 5),
        dropout=checkpoint.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    info = {
        "num_classes": checkpoint.get("num_classes", 6),
        "num_joints": checkpoint.get("num_joints", 65),
        "input_dim": checkpoint.get("input_dim", 130),
        "max_frames": checkpoint.get("max_frames", 120),
        "epoch": checkpoint.get("epoch", "?"),
        "val_acc": checkpoint.get("val_acc", "?"),
    }
    return model, info


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = args.model or os.path.join(ROOT_DIR, "models", "best_tcn_xsub.pth")

    # ===== Load TCN model =====
    print(f"Loading TCN model: {model_path}")
    model, model_info = load_tcn(model_path, device)
    max_frames = model_info["max_frames"]
    num_classes = model_info["num_classes"]
    num_joints = model_info["num_joints"]
    print(f"  Classes: {num_classes}, Joints: {num_joints}, Max frames: {max_frames}")
    print(f"  Trained epoch: {model_info['epoch']}, Val acc: {model_info['val_acc']}")

    # ===== Init rtmlib =====
    sys.path.insert(0, args.rtmlib_path)
    from rtmlib import draw_skeleton

    print(f"Initializing PoseTracker (mode: {args.mode})...")
    wholebody = init_pose_tracker(
        rtmlib_path=args.rtmlib_path,
        mode=args.mode,
        device=args.device,
        backend="onnxruntime",
        det_frequency=args.det_frequency,
        tracking=False,
        return_track_ids=False,
    )

    # ===== Video source =====
    source, is_camera, is_image = parse_video_source(args.source)

    # ===== Image mode =====
    if is_image:
        print(f"Image mode: {source}")
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"Error: Cannot read image: {source}")
            return

        # 이미지 → 동일 프레임을 반복하여 시퀀스 생성
        buf = deque(maxlen=args.buf_size)
        keypoints, scores = wholebody(frame)

        if keypoints is None or len(keypoints) == 0:
            print("No person detected.")
            return

        kp_body, sc_body = extract_skeleton(keypoints, scores, num_joints=num_joints, person_idx=0)
        kp_norm = normalize_frame(kp_body, sc_body)

        # 동일 프레임으로 버퍼 채우기
        for _ in range(args.buf_size):
            buf.append(kp_norm)

        class_name, confidence, probs_dict = predict_tcn(
            model, buf, max_frames, num_classes, device, num_joints=num_joints
        )

        # Draw
        img_show = frame.copy()
        img_show = draw_skeleton(img_show, keypoints, scores,
                                 openpose_skeleton=False, kpt_thr=args.kpt_thr)
        center = get_shoulder_center(keypoints[0])
        label_pos = (center[0], center[1] - 50)
        draw_track_label(
            img_show,
            label_pos,
            class_name,
            confidence,
            CLASS_COLORS,
            extra_lines=[f"buf: {len(buf)}/{args.buf_size}"],
        )

        print(f"\nResult: {class_name} ({confidence:.1%})")
        for c, p in probs_dict.items():
            print(f"  {c}: {p:.4f}")

        cv2.imshow("TCN Action Recognition", img_show)
        print("\nPress any key to quit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ===== Video / Camera mode =====
    cap = open_video_capture(source, is_camera)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        return

    if is_camera:
        print("Warming up camera...")
        warmup_capture(cap, num_frames=5)

    # Track-wise frame buffers and states
    tracks = TrackManager()  # tid -> TrackState
    # Current target
    active_target_tid = None

    frame_count = 0
    next_tid = 0
    fps_list = []
    fail_count = 0
    tcn_ms = 0.0
    RECONNECT_FRAMES = max(0, int(args.grace_frames))
    TTL_FRAMES = max(RECONNECT_FRAMES + 1, int(args.ttl_frames))

    print(f"\nStarted! Buffer size: {args.buf_size}, "
          f"Infer every: {args.infer_every} frames")
    print("Press 'q' to quit, 'r' to reset tracks, 'i' to print track states.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            fail_count += 1
            if is_camera and fail_count < 30:
                time.sleep(0.1)
                continue
            break
        fail_count = 0
        frame_count += 1

        t_start = time.time()

        # ---- Pose detection ----
        result = wholebody(frame)
        
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            keypoints, scores = result[0], result[1]
        else:
            keypoints, scores = None, None
        
        t_pose = time.time()

        img_show = frame.copy()
        tid_to_detidx = {}  # reset for this frame

        if keypoints is not None and scores is not None and len(keypoints) > 0:
            # Draw skeleton
            img_show = draw_skeleton(img_show, keypoints, scores,
                                     openpose_skeleton=False, kpt_thr=args.kpt_thr)

            # 각 detection을 track에 업데이트
            n_det = min(len(keypoints), len(scores))
            det_centers = []
            for det_idx in range(n_det):
                det_centers.append(get_shoulder_center(keypoints[det_idx]))

            det_to_tid, next_tid = match_centers_to_tracks(
                det_centers, tracks, frame_count, RECONNECT_FRAMES, args.track_match_dist, next_tid
            )

            visible_tids = set()
            for det_idx in range(n_det):
                tid = int(det_to_tid[det_idx])
                tid_to_detidx[tid] = det_idx
                visible_tids.add(tid)
                
                # Extract & normalize skeleton
                kp_body, sc_body = extract_skeleton(keypoints, scores, num_joints=num_joints, person_idx=det_idx)
                if kp_body is not None:
                    kp_norm = normalize_frame(kp_body, sc_body)
                    
                    # Track이 없으면 생성
                    if tid not in tracks:
                        tracks[tid] = TrackState(buf_size=args.buf_size)
                    
                    # 버퍼에 추가 및 상태 업데이트
                    tracks[tid].buffer.append(kp_norm)
                    kp_px = keypoints[det_idx][:17]  # 17 body joints
                    update_track_from_keypoints(tracks[tid], kp_px, det_centers[det_idx], frame_count)

            # ---- TCN inference (every N frames) ----
            if frame_count % args.infer_every == 0:
                t_tcn_start = time.time()
                
                # 최근 관측된 track 우선으로 배치 추론
                infer_candidates = [
                    (tid, tr) for tid, tr in tracks.items()
                    if (frame_count - tr.last_seen_frame) <= 1 and len(tr.buffer) >= 5
                ]
                infer_candidates.sort(key=lambda x: x[1].bbox_area, reverse=True)
                infer_candidates = infer_candidates[:args.max_tracks_infer]

                pred_map = predict_tcn_batch(
                    model, infer_candidates, max_frames, num_classes, device,
                    num_joints=num_joints
                )
                for tid, (pred_cls, pred_conf, pred_probs) in pred_map.items():
                    tracks[tid].pred_cls = pred_cls
                    tracks[tid].pred_conf = pred_conf
                    tracks[tid].pred_probs = pred_probs
                
                tcn_ms = (time.time() - t_tcn_start) * 1000

            # 이번 프레임에 보이지 않은 track만 miss count 증가
            for tid, track in tracks.items():
                if tid not in visible_tids:
                    track.missing_frames = frame_count - track.last_seen_frame

            # ---- hands_up 후보 선택 ----
            hands_up_candidates = []
            for tid, track in tracks.items():
                # 화면에 있는 사람만 현재 target 후보로 사용
                if tid not in tid_to_detidx:
                    continue
                if track.pred_cls in ["hands_up_single", "hands_up_both"] and track.pred_conf > 0.7:
                    hands_up_candidates.append((tid, track))
            
            # 우선순위: hands_up_both > hands_up_single, bbox 크기 큰 순
            hands_up_candidates.sort(
                key=lambda x: (
                    x[1].pred_cls == "hands_up_both",  # hands_up_both가 True = 1
                    x[1].bbox_area
                ),
                reverse=True
            )
            
            # 최우선 타겟 선택
            if hands_up_candidates:
                active_target_tid = hands_up_candidates[0][0]
            else:
                # 기존 타겟이 잠깐 가려진 경우 grace 동안 유지
                if (
                    active_target_tid is not None
                    and active_target_tid in tracks
                    and tracks[active_target_tid].missing_frames <= RECONNECT_FRAMES
                ):
                    pass
                else:
                    active_target_tid = None

            # Draw labels for all tracks
            for tid, track in tracks.items():
                if tid in tid_to_detidx:
                    det_idx = tid_to_detidx[tid]
                    center = get_shoulder_center(keypoints[det_idx])
                    label_pos = (center[0], center[1] - 50)
                    
                    # Draw box around person if it's the target
                    if tid == active_target_tid and det_idx < len(keypoints):
                        kp_px = keypoints[det_idx][:17]
                        x_min = int(kp_px[:, 0].min())
                        x_max = int(kp_px[:, 0].max())
                        y_min = int(kp_px[:, 1].min())
                        y_max = int(kp_px[:, 1].max())
                        cv2.rectangle(img_show, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    
                    prefix = f"TID:{tid}"
                    if tid == active_target_tid:
                        prefix += " [TARGET]"
                    st = tracks.get_track_state(tid)
                    if st is not None and st.get("last_center") is not None:
                        cx, cy = st["last_center"]
                        extra = f"xy=({cx:.1f},{cy:.1f})"
                        cls_name = st["pred_cls"]
                        conf = st["pred_conf"]
                        buf_len = st["buffer_len"]
                    else:
                        extra = None
                        cls_name = track.pred_cls
                        conf = track.pred_conf
                        buf_len = len(track.buffer)
                    extra_lines = [f"buf: {buf_len}/{args.buf_size}"]
                    if extra:
                        extra_lines.append(extra)
                    draw_track_label(
                        img_show,
                        label_pos,
                        cls_name,
                        conf,
                        CLASS_COLORS,
                        prefix=prefix,
                        extra_lines=extra_lines,
                    )

        # ---- Clean up old tracks ----
        dead_tids = [tid for tid, track in tracks.items()
                     if frame_count - track.last_seen_frame > TTL_FRAMES]
        for tid in dead_tids:
            del tracks[tid]
            if active_target_tid == tid:
                active_target_tid = None

        t_end = time.time()

        # FPS
        fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        pose_ms = (t_pose - t_start) * 1000
        
        # HUD: 모든 track의 class 확률
        all_probs = {}
        if tracks:
            # 현재 활성 track들의 probs를 취합 (첫 track 기준으로 표시)
            first_track = next(iter(tracks.values()))
            all_probs = first_track.pred_probs
        
        total_buf_len = sum(len(t.buffer) for t in tracks.values())
        total_buf_cap = args.buf_size * len(tracks)
        draw_basic_hud(
            img_show,
            avg_fps,
            f"Pose: {pose_ms:.0f}ms  TCN: {tcn_ms:.0f}ms",
            extra_lines=[f"Buffer: {total_buf_len}/{total_buf_cap}"],
        )
        draw_prob_bars(img_show, all_probs, CLASS_COLORS, start_y=115, row_h=22, label_max_chars=12)

        cv2.imshow("TCN Action Recognition", img_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracks.clear()
            active_target_tid = None
            print("[Reset] All tracks cleared.")
        elif key == ord('i'):
            all_states = tracks.get_all_states()
            if not all_states:
                print("[Tracks] none")
            else:
                print("[Tracks]")
                for tid, st in sorted(all_states.items(), key=lambda x: x[0]):
                    print(format_track_state_line(tid, st, include_buffer=True))

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
