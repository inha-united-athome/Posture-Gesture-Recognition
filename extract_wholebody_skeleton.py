"""
Extract wholebody skeletons (body+feet+hands, no face) from custom video clips.

Processes all .avi videos in data/dataset/ subfolders and creates a pkl
with the same format as NTU subset.

65-joint skeleton layout (RTMPose 133에서 face 68개 제거):
    0-16:  Body (COCO 17)
   17-22:  Feet (6)
   23-43:  Left Hand (21)
   44-64:  Right Hand (21)

Class mapping (from folder name):
    idle          → 0
    wave          → 1
    hand_up       → 2  (hands_up_single)
    hand_up_both  → 3  (hands_up_both)
    pointing      → 4
    stop          → 5

PKL format (same as NTU):
    annotations[i]:
        - "frame_dir": str
        - "label": int (0~5)
        - "total_frames": int
        - "keypoint": np.ndarray (1, T, 65, 2)
        - "keypoint_score": np.ndarray (1, T, 65)

Usage:
    python extract_wholebody_skeleton.py
    python extract_wholebody_skeleton.py --out wholebody_6class.pkl
    python extract_wholebody_skeleton.py --device cpu --backend opencv
"""

import os
import sys
import argparse
import pickle
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


# ===== Class mapping (folder name → label) =====
FOLDER_TO_LABEL = {
    "idle": 0,
    "wave": 1,
    "hand_up": 2,
    "hand_up_both": 3,
    "pointing": 4,
    "stop": 5,
}

LABEL_TO_NAME = {
    0: "idle",
    1: "waving",
    2: "hands_up_single",
    3: "hands_up_both",
    4: "pointing",
    5: "stop",
}

# RTMPose 133 → 65 joints (face 제거)
# body(0-16) + feet(17-22) + left_hand(91-111) + right_hand(112-132)
KEEP_IDXS = list(range(0, 23)) + list(range(91, 133))  # 65 indices
NUM_JOINTS = 65


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract wholebody skeletons (65 joints, no face) from custom videos"
    )
    parser.add_argument("--rtmlib_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtmlib"))
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset"))
    parser.add_argument("--out", type=str, default="wholebody_6class.pkl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="onnxruntime")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--person_select", type=str, default="largest",
                        choices=["first", "largest", "center"],
                        help="Person to use when multiple are detected per frame: "
                             "'first' = keypoints[0] (tracker order), "
                             "'largest' = biggest bbox (recommended for crowd clips), "
                             "'center' = closest to frame center")
    parser.add_argument("--score_thr", type=float, default=0.3,
                        help="Per-keypoint score threshold used when measuring bbox "
                             "for --person_select largest/center")
    return parser.parse_args()


def init_pose_tracker(rtmlib_path, device, backend):
    sys.path.insert(0, rtmlib_path)
    from rtmlib import PoseTracker, Wholebody

    tracker = PoseTracker(
        Wholebody,
        det_frequency=7,
        to_openpose=False,
        mode='performance',
        backend=backend,
        device=device,
    )
    return tracker


def _select_person_idx(keypoints, scores, mode, score_thr, frame_wh):
    """
    Choose which detected person to use for this frame.

    keypoints: (N, 133, 2) — N detected persons
    scores:    (N, 133)    — raw rtmlib logits (sigmoid applied for thresholding)
    mode:      "first" | "largest" | "center"
    Returns: index in [0, N)
    """
    N = len(keypoints)
    if N == 1 or mode == "first":
        return 0

    # sigmoid for thresholding only
    sc_prob = 1.0 / (1.0 + np.exp(-scores.astype(np.float32)))

    best_idx = 0
    best_score = -np.inf
    fw, fh = frame_wh
    cx_frame, cy_frame = fw / 2.0, fh / 2.0

    for i in range(N):
        valid = sc_prob[i] > score_thr
        if valid.sum() < 4:
            # Too few confident joints — fall back to all
            kp_i = keypoints[i]
        else:
            kp_i = keypoints[i][valid]

        x_min, y_min = kp_i.min(axis=0)
        x_max, y_max = kp_i.max(axis=0)
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)

        if mode == "largest":
            metric = w * h
        else:  # "center"
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            dist = ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5
            metric = -dist  # smaller distance = better

        if metric > best_score:
            best_score = metric
            best_idx = i

    return best_idx


def extract_video_skeleton(video_path, tracker, person_select="first", score_thr=0.3):
    """
    Extract 65-joint keypoints (no face) from every frame of a video.

    Returns:
        keypoint:       (1, T, 65, 2) or None
        keypoint_score: (1, T, 65) or None
        total_frames:   int
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    all_kp = []
    all_sc = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = tracker(frame)

        if keypoints is None or len(keypoints) == 0:
            # No person detected → zeros
            kp_frame = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            sc_frame = np.zeros((NUM_JOINTS,), dtype=np.float32)
        else:
            kps = np.asarray(keypoints, dtype=np.float32)   # (N, 133, 2)
            scs = np.asarray(scores, dtype=np.float32)       # (N, 133)
            pidx = _select_person_idx(kps, scs, person_select, score_thr, (fw, fh))

            kp_133 = kps[pidx]   # (133, 2)
            sc_133 = scs[pidx]   # (133,)

            kp_frame = kp_133[KEEP_IDXS]   # (65, 2)
            sc_frame = sc_133[KEEP_IDXS]   # (65,)

            # rtmlib raw logits → sigmoid
            sc_frame = 1.0 / (1.0 + np.exp(-sc_frame))

        all_kp.append(kp_frame)
        all_sc.append(sc_frame)

    cap.release()

    if len(all_kp) == 0:
        return None, None, 0

    # Stack: (T, 65, 2) and (T, 65)
    kp_array = np.stack(all_kp, axis=0)   # (T, 65, 2)
    sc_array = np.stack(all_sc, axis=0)   # (T, 65)

    # Add person dimension: (1, T, 65, 2) and (1, T, 65)
    kp_array = kp_array[np.newaxis]
    sc_array = sc_array[np.newaxis]

    return kp_array, sc_array, kp_array.shape[1]


def collect_videos(data_dir):
    """
    Recursively collect all .avi/.mp4 videos from data/dataset/ subfolders.
    Map top-level folder name to class label.

    Returns:
        list of (video_path, label, frame_dir_name)
    """
    videos = []

    for action_folder in sorted(os.listdir(data_dir)):
        action_path = os.path.join(data_dir, action_folder)
        if not os.path.isdir(action_path):
            continue

        if action_folder not in FOLDER_TO_LABEL:
            print(f"  [WARN] Unknown folder '{action_folder}', skipping")
            continue

        label = FOLDER_TO_LABEL[action_folder]

        # Recursively find all video files
        video_files = []
        for ext in ["*.avi", "*.mp4"]:
            video_files.extend(glob(os.path.join(action_path, "**", ext), recursive=True))
        video_files = sorted(video_files)

        for vpath in video_files:
            # Create a unique frame_dir name from relative path
            rel = os.path.relpath(vpath, data_dir)
            # e.g. "hand_up/left_hand_up/clip_000001.avi" → "hand_up__left_hand_up__clip_000001"
            frame_dir = os.path.splitext(rel)[0].replace(os.sep, "__")
            videos.append((vpath, label, frame_dir))

        print(f"  {action_folder} (label={label}): {len(video_files)} videos")

    return videos


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"{'='*60}")
    print(f"  Wholebody Skeleton Extraction (65 joints, no face)")
    print(f"{'='*60}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output:     {args.out}")
    print(f"  Device:     {args.device}")
    print(f"  Backend:    {args.backend}")
    print(f"  Person sel: {args.person_select} (score_thr={args.score_thr})")
    print()

    # ===== Collect videos =====
    print("Scanning videos...")
    video_list = collect_videos(args.data_dir)
    print(f"\nTotal: {len(video_list)} videos\n")

    if len(video_list) == 0:
        print("[ERROR] No videos found!")
        return

    # ===== Init pose tracker =====
    print("Initializing pose tracker...")
    tracker = init_pose_tracker(args.rtmlib_path, args.device, args.backend)

    # ===== Extract skeletons =====
    annotations = []
    label_counts = {i: 0 for i in range(len(LABEL_TO_NAME))}

    for video_path, label, frame_dir in tqdm(video_list, desc="Extracting"):
        kp, sc, total_frames = extract_video_skeleton(
            video_path, tracker,
            person_select=args.person_select,
            score_thr=args.score_thr,
        )

        if kp is None or total_frames < 3:
            tqdm.write(f"  [SKIP] {frame_dir}: too few frames ({total_frames})")
            continue

        ann = {
            "frame_dir": frame_dir,
            "label": label,
            "total_frames": total_frames,
            "keypoint": kp,            # (1, T, 65, 2)
            "keypoint_score": sc,      # (1, T, 65)
            "img_shape": (480, 640),
            "original_shape": (480, 640),
            "num_joints": NUM_JOINTS,
        }
        annotations.append(ann)
        label_counts[label] += 1

    print(f"\nExtracted {len(annotations)} clips:")
    for label_id, count in sorted(label_counts.items()):
        name = LABEL_TO_NAME.get(label_id, f"class_{label_id}")
        print(f"  {label_id}: {name:20s} — {count} clips")

    # ===== Per-class stratified train/val split =====
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for i, ann in enumerate(annotations):
        label_to_indices[ann["label"]].append(i)

    train_indices = []
    val_indices = []

    for label_id, indices in sorted(label_to_indices.items()):
        np.random.shuffle(indices)
        n_train = int(len(indices) * args.train_ratio)
        train_indices.extend(sorted(indices[:n_train]))
        val_indices.extend(sorted(indices[n_train:]))

    split = {
        "xsub_train": train_indices,
        "xsub_val": val_indices,
        "xset_train": train_indices,
        "xset_val": val_indices,
    }

    print(f"\nSplit: Train={len(train_indices)}, Val={len(val_indices)}")

    # ===== Save pkl =====
    out = {
        "annotations": annotations,
        "split": split,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Saved: {args.out}")
    print(f"  Joints per frame: {NUM_JOINTS} (body17 + feet6 + hands42)")
    print(f"  PKL keypoint shape: (1, T, {NUM_JOINTS}, 2)")


if __name__ == "__main__":
    main()
