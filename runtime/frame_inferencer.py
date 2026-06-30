import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from runtime.mlp_runtime import predict_mlp_frames
from runtime.realtime_inference_common import (
    get_shoulder_center,
    init_pose_tracker,
    update_track_from_keypoints,
)
from runtime.realtime_viz import draw_basic_hud, draw_prob_bars, draw_track_label
from runtime.tcn_runtime import extract_skeleton, normalize_frame, predict_tcn_batch
from utils.track_state import TrackManager, TrackState, match_centers_to_tracks


@dataclass
class DetectionResult:
    track_id: int
    det_index: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    center_xy: Optional[Tuple[float, float]]
    bbox_xyxy: Optional[Tuple[float, float, float, float]]
    skeleton_xy: List[List[float]]
    keypoint_scores: List[float]

    def to_dict(self):
        return {
            "track_id": int(self.track_id),
            "det_index": int(self.det_index),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "probabilities": dict(self.probabilities),
            "center_xy": None if self.center_xy is None else [float(self.center_xy[0]), float(self.center_xy[1])],
            "bbox_xyxy": None if self.bbox_xyxy is None else [float(v) for v in self.bbox_xyxy],
            "skeleton_xy": [[float(x), float(y)] for x, y in self.skeleton_xy],
            "keypoint_scores": [float(v) for v in self.keypoint_scores],
        }


@dataclass
class FrameInferenceResult:
    detections: List[DetectionResult]
    debug_image: Optional[np.ndarray]
    keypoints: Optional[np.ndarray]
    scores: Optional[np.ndarray]

    def to_dict(self):
        return {
            "detections": [d.to_dict() for d in self.detections],
            "keypoints": None if self.keypoints is None else self.keypoints.tolist(),
            "scores": None if self.scores is None else self.scores.tolist(),
            "debug_image": self.debug_image,
        }


class MLPFrameInferencer:
    def __init__(
        self,
        model,
        device,
        rtmlib_path="rtmlib",
        mode="balanced",
        pose_device="cuda",
        backend="onnxruntime",
        det_frequency=1,
        max_persons_infer=8,
        track_match_dist=120.0,
        track_ttl=20,
        kpt_thr=0.4,
        class_colors=None,
    ):
        self.model = model
        self.device = device
        self.max_persons_infer = max_persons_infer
        self.track_match_dist = float(track_match_dist)
        self.track_ttl = max(1, int(track_ttl))
        self.kpt_thr = float(kpt_thr)
        self.class_colors = class_colors or {"unknown": (80, 80, 80)}

        self.tracker = init_pose_tracker(
            rtmlib_path=rtmlib_path,
            mode=mode,
            device=pose_device,
            backend=backend,
            det_frequency=det_frequency,
            tracking=False,
            return_track_ids=False,
            max_bboxes=self.max_persons_infer,
        )

        self.frame_count = 0
        self.next_tid = 0
        self.tracks = TrackManager()

    def infer(self, frame, return_debug_image=True):
        from rtmlib import draw_skeleton

        self.frame_count += 1
        keypoints, scores = self.tracker(frame)
        detections = []

        debug_image = frame.copy() if return_debug_image else None
        if keypoints is None or scores is None or len(keypoints) == 0:
            if return_debug_image:
                draw_basic_hud(debug_image, 0.0, "No person detected")
            return FrameInferenceResult(detections, debug_image, keypoints, scores)

        n_det = min(len(keypoints), len(scores), self.max_persons_infer)
        det_centers = [get_shoulder_center(keypoints[det_idx]) for det_idx in range(n_det)]
        det_to_tid, self.next_tid = match_centers_to_tracks(
            det_centers,
            self.tracks,
            self.frame_count,
            self.track_ttl,
            self.track_match_dist,
            self.next_tid,
        )

        results = predict_mlp_frames(
            self.model,
            keypoints,
            scores,
            self.device,
            max_persons=self.max_persons_infer,
        )

        visible_tids = set()
        for r in results:
            pidx = r["person_idx"]
            if pidx >= len(det_to_tid):
                continue
            tid = int(det_to_tid[pidx])
            visible_tids.add(tid)
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(buf_size=1)

            center = get_shoulder_center(keypoints[pidx])
            kp_px = keypoints[pidx][:17]
            update_track_from_keypoints(self.tracks[tid], kp_px, center, self.frame_count)
            self.tracks[tid].pred_cls = r["class"]
            self.tracks[tid].pred_conf = r["confidence"]
            self.tracks[tid].pred_probs = r["probabilities"]

            det = DetectionResult(
                track_id=tid,
                det_index=pidx,
                class_name=r["class"],
                confidence=float(r["confidence"]),
                probabilities=dict(r["probabilities"]),
                center_xy=center,
                bbox_xyxy=self.tracks[tid].last_bbox,
                skeleton_xy=keypoints[pidx].astype(np.float32).tolist(),
                keypoint_scores=scores[pidx].astype(np.float32).tolist(),
            )
            detections.append(det)

            if return_debug_image:
                draw_track_label(
                    debug_image,
                    (center[0], center[1] - 60),
                    det.class_name,
                    det.confidence,
                    self.class_colors,
                    prefix=f"TID:{tid}",
                    font_scale=0.8,
                )

        for tid, tr in self.tracks.items():
            if tid not in visible_tids:
                tr.missing_frames = self.frame_count - tr.last_seen_frame

        dead_tids = [tid for tid, tr in self.tracks.items() if tr.missing_frames > self.track_ttl]
        for tid in dead_tids:
            del self.tracks[tid]

        if return_debug_image:
            debug_image = draw_skeleton(
                debug_image,
                keypoints,
                scores,
                openpose_skeleton=False,
                kpt_thr=self.kpt_thr,
            )
            best_probs = {}
            if detections:
                best = max(detections, key=lambda x: x.confidence)
                best_probs = best.probabilities
            draw_basic_hud(debug_image, 0.0, f"Detections: {len(detections)}")
            draw_prob_bars(debug_image, best_probs, self.class_colors, start_y=90, row_h=24, label_max_chars=10)

        return FrameInferenceResult(detections, debug_image, keypoints, scores)


class TCNFrameInferencer:
    def __init__(
        self,
        model,
        device,
        num_classes,
        max_frames,
        num_joints,
        rtmlib_path="rtmlib",
        mode="balanced",
        pose_device="cuda",
        backend="onnxruntime",
        det_frequency=1,
        buf_size=60,
        infer_every=5,
        max_tracks_infer=8,
        track_match_dist=120.0,
        grace_frames=15,
        ttl_frames=45,
        kpt_thr=0.4,
        window_sec=2.0,
        class_colors=None,
    ):
        self.model = model
        self.device = device
        self.num_classes = int(num_classes)
        self.max_frames = int(max_frames)
        self.num_joints = int(num_joints)
        self.buf_size = max(5, int(buf_size))
        self.infer_every = max(1, int(infer_every))
        self.max_tracks_infer = max(1, int(max_tracks_infer))
        self.track_match_dist = float(track_match_dist)
        self.grace_frames = max(0, int(grace_frames))
        self.ttl_frames = max(self.grace_frames + 1, int(ttl_frames))
        self.kpt_thr = float(kpt_thr)
        # window_sec>0 이면 시간 기반 버퍼 활성화(fps 무관). None/0 이면 기존 프레임 기반.
        self.window_sec = float(window_sec) if window_sec else None
        self.class_colors = class_colors or {"unknown": (80, 80, 80)}

        self.tracker = init_pose_tracker(
            rtmlib_path=rtmlib_path,
            mode=mode,
            device=pose_device,
            backend=backend,
            det_frequency=det_frequency,
            tracking=False,
            return_track_ids=False,
            max_bboxes=self.max_tracks_infer,
        )

        self.frame_count = 0
        self.next_tid = 0
        self.tracks = TrackManager()

    def infer(self, frame, return_debug_image=True, timestamp=None):
        from rtmlib import draw_skeleton

        if timestamp is None:
            timestamp = time.time()
        self.frame_count += 1
        keypoints, scores = self.tracker(frame)
        detections = []
        debug_image = frame.copy() if return_debug_image else None

        if keypoints is None or scores is None or len(keypoints) == 0:
            if return_debug_image:
                draw_basic_hud(debug_image, 0.0, "No person detected")
            return FrameInferenceResult(detections, debug_image, keypoints, scores)

        n_det = min(len(keypoints), len(scores))
        det_centers = [get_shoulder_center(keypoints[det_idx]) for det_idx in range(n_det)]
        det_to_tid, self.next_tid = match_centers_to_tracks(
            det_centers,
            self.tracks,
            self.frame_count,
            self.grace_frames,
            self.track_match_dist,
            self.next_tid,
        )

        tid_to_detidx = {}
        visible_tids = set()
        for det_idx in range(n_det):
            tid = int(det_to_tid[det_idx])
            tid_to_detidx[tid] = det_idx
            visible_tids.add(tid)

            kp_body, sc_body = extract_skeleton(
                keypoints, scores, num_joints=self.num_joints, person_idx=det_idx
            )
            if kp_body is None:
                continue
            kp_norm = normalize_frame(kp_body, sc_body)

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(buf_size=self.buf_size)
            self.tracks[tid].push(kp_norm, timestamp)

            kp_px = keypoints[det_idx][:17]
            update_track_from_keypoints(self.tracks[tid], kp_px, det_centers[det_idx], self.frame_count)

        if self.frame_count % self.infer_every == 0:
            infer_candidates = [
                (tid, tr)
                for tid, tr in self.tracks.items()
                if (self.frame_count - tr.last_seen_frame) <= 1 and len(tr.buffer) >= 5
            ]
            infer_candidates.sort(key=lambda x: x[1].bbox_area, reverse=True)
            infer_candidates = infer_candidates[: self.max_tracks_infer]

            pred_map = predict_tcn_batch(
                self.model,
                infer_candidates,
                self.max_frames,
                self.num_classes,
                self.device,
                num_joints=self.num_joints,
                window_sec=self.window_sec,
            )
            for tid, (pred_cls, pred_conf, pred_probs) in pred_map.items():
                self.tracks[tid].pred_cls = pred_cls
                self.tracks[tid].pred_conf = pred_conf
                self.tracks[tid].pred_probs = pred_probs

        for tid, track in self.tracks.items():
            if tid not in visible_tids:
                track.missing_frames = self.frame_count - track.last_seen_frame

        dead_tids = [
            tid
            for tid, track in self.tracks.items()
            if (self.frame_count - track.last_seen_frame) > self.ttl_frames
        ]
        for tid in dead_tids:
            del self.tracks[tid]

        for tid, det_idx in tid_to_detidx.items():
            track = self.tracks.get_track_state(tid)
            if track is None:
                continue
            center = track["last_center"]
            det = DetectionResult(
                track_id=tid,
                det_index=det_idx,
                class_name=track["pred_cls"],
                confidence=float(track["pred_conf"]),
                probabilities=dict(track["pred_probs"]),
                center_xy=None if center is None else (float(center[0]), float(center[1])),
                bbox_xyxy=track["last_bbox"],
                skeleton_xy=keypoints[det_idx].astype(np.float32).tolist(),
                keypoint_scores=scores[det_idx].astype(np.float32).tolist(),
            )
            detections.append(det)

            if return_debug_image and center is not None:
                draw_track_label(
                    debug_image,
                    (center[0], center[1] - 50),
                    det.class_name,
                    det.confidence,
                    self.class_colors,
                    prefix=f"TID:{tid}",
                    extra_lines=[f"buf: {track['buffer_len']}/{self.buf_size}"],
                    font_scale=0.8,
                )

        if return_debug_image:
            debug_image = draw_skeleton(
                debug_image,
                keypoints,
                scores,
                openpose_skeleton=False,
                kpt_thr=self.kpt_thr,
            )
            best_probs = {}
            if detections:
                best = max(detections, key=lambda x: x.confidence)
                best_probs = best.probabilities
            draw_basic_hud(debug_image, 0.0, f"Detections: {len(detections)}")
            draw_prob_bars(debug_image, best_probs, self.class_colors, start_y=90, row_h=22, label_max_chars=12)

        return FrameInferenceResult(detections, debug_image, keypoints, scores)

