"""
Extract pose skeletons from custom video clips with an Ultralytics YOLO pose model.

The output PKL uses the same structure as extract_wholebody_skeleton.py:
    annotations[i]["keypoint"]       -> (1, T, V, 2)
    annotations[i]["keypoint_score"] -> (1, T, V)

This lets the same TCN training code compare RTMLib wholebody skeletons against
YOLO pose skeletons. The number of joints V is inferred from the model output.

Usage:
    python extract_yolo_pose_skeleton.py --weights yolo26-pose.pt --out yolo26_pose_8class.pkl
    python extract_yolo_pose_skeleton.py --weights yolo11n-pose.pt --out yolo_pose_8class.pkl
"""

import argparse
import os
import pickle
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from extract_wholebody_skeleton import FOLDER_TO_LABEL, LABEL_TO_NAME


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT_DIR, "runtime", "ultralytics"))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Extract YOLO pose skeleton PKL from videos")
    parser.add_argument("--weights", type=str, required=True,
                        help="Ultralytics YOLO pose weights, e.g. yolo26-pose.pt")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset"))
    parser.add_argument("--out", type=str, default="yolo_pose_8class.pkl")
    parser.add_argument("--device", type=str, default="0",
                        help="Ultralytics device: 0, cpu, cuda:0, etc.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--person_select", type=str, default="largest",
                        choices=["first", "largest", "center"])
    return parser.parse_args()


def collect_videos(data_dir):
    videos = []
    for action_folder in sorted(os.listdir(data_dir)):
        action_path = os.path.join(data_dir, action_folder)
        if not os.path.isdir(action_path):
            continue
        if action_folder not in FOLDER_TO_LABEL:
            print(f"  [WARN] Unknown folder '{action_folder}', skipping")
            continue

        label = FOLDER_TO_LABEL[action_folder]
        video_files = []
        for ext in ["*.avi", "*.mp4", "*.mov", "*.mkv"]:
            video_files.extend(glob(os.path.join(action_path, "**", ext), recursive=True))
        video_files = sorted(video_files)

        for vpath in video_files:
            rel = os.path.relpath(vpath, data_dir)
            frame_dir = os.path.splitext(rel)[0].replace(os.sep, "__")
            videos.append((vpath, label, frame_dir))

        print(f"  {action_folder} (label={label}): {len(video_files)} videos")
    return videos


def _select_person(result, frame_wh, mode):
    if result.keypoints is None or result.keypoints.xy is None:
        return None

    xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
    if xy.shape[0] == 0:
        return None

    conf = result.keypoints.conf
    if conf is None:
        score = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        score = conf.detach().cpu().numpy().astype(np.float32)

    if xy.shape[0] == 1 or mode == "first":
        return xy[0], score[0]

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    fw, fh = frame_wh
    frame_center = np.array([fw / 2.0, fh / 2.0], dtype=np.float32)

    if mode == "largest":
        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        idx = int(np.argmax(areas))
    else:
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
        dists = np.linalg.norm(centers - frame_center[None], axis=1)
        idx = int(np.argmin(dists))

    return xy[idx], score[idx]


def extract_video_skeleton(video_path, model, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0, (480, 640), 0

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    all_kp = []
    all_sc = []
    num_joints = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        selected = _select_person(results[0], (fw, fh), args.person_select)

        if selected is None:
            if num_joints is None:
                continue
            kp_frame = np.zeros((num_joints, 2), dtype=np.float32)
            sc_frame = np.zeros((num_joints,), dtype=np.float32)
        else:
            kp_frame, sc_frame = selected
            num_joints = kp_frame.shape[0]

        all_kp.append(kp_frame)
        all_sc.append(sc_frame)

    cap.release()

    if len(all_kp) == 0 or num_joints is None:
        return None, None, 0, (fh, fw), 0

    kp_array = np.stack(all_kp, axis=0)[np.newaxis].astype(np.float32)
    sc_array = np.stack(all_sc, axis=0)[np.newaxis].astype(np.float32)
    return kp_array, sc_array, kp_array.shape[1], (fh, fw), num_joints


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  YOLO Pose Skeleton Extraction")
    print("=" * 60)
    print(f"  Weights:    {args.weights}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output:     {args.out}")
    print(f"  Device:     {args.device}")
    print(f"  Person sel: {args.person_select}")
    print()

    video_list = collect_videos(args.data_dir)
    print(f"\nTotal: {len(video_list)} videos\n")
    if not video_list:
        print("[ERROR] No videos found")
        return

    model = YOLO(args.weights)

    annotations = []
    label_counts = {i: 0 for i in LABEL_TO_NAME}
    detected_joints = None

    for video_path, label, frame_dir in tqdm(video_list, desc="Extracting"):
        kp, sc, total_frames, img_shape, num_joints = extract_video_skeleton(video_path, model, args)
        if kp is None or total_frames < 3:
            tqdm.write(f"  [SKIP] {frame_dir}: too few detected frames ({total_frames})")
            continue

        detected_joints = detected_joints or num_joints
        ann = {
            "frame_dir": frame_dir,
            "label": label,
            "total_frames": total_frames,
            "keypoint": kp,
            "keypoint_score": sc,
            "img_shape": img_shape,
            "original_shape": img_shape,
            "num_joints": num_joints,
            "pose_backend": "ultralytics_yolo",
            "pose_weights": os.path.basename(args.weights),
        }
        annotations.append(ann)
        label_counts[label] += 1

    print(f"\nExtracted {len(annotations)} clips:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  {label_id}: {LABEL_TO_NAME[label_id]:16s} {count} clips")

    label_to_indices = defaultdict(list)
    for i, ann in enumerate(annotations):
        label_to_indices[ann["label"]].append(i)

    train_indices = []
    val_indices = []
    for _, indices in sorted(label_to_indices.items()):
        indices = list(indices)
        rng.shuffle(indices)
        n_train = int(len(indices) * args.train_ratio)
        train_indices.extend(sorted(indices[:n_train]))
        val_indices.extend(sorted(indices[n_train:]))

    split = {
        "xsub_train": train_indices,
        "xsub_val": val_indices,
        "xset_train": train_indices,
        "xset_val": val_indices,
    }

    out = {
        "annotations": annotations,
        "split": split,
        "meta": {
            "pose_backend": "ultralytics_yolo",
            "pose_weights": os.path.basename(args.weights),
            "num_joints": detected_joints,
            "label_to_name": LABEL_TO_NAME,
        },
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved: {args.out}")
    print(f"  Joints per frame: {detected_joints}")
    print(f"  Split: Train={len(train_indices)}, Val={len(val_indices)}")


if __name__ == "__main__":
    main()
