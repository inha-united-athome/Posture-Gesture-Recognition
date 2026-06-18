"""
8-class pkl → 6-class pkl 변환 스크립트.

left/right를 하나의 클래스로 합칩니다:
  left_hand_up(2) + right_hand_up(3) → hand_up(2)
  left_pointing(5) + right_pointing(6) → pointing(4)

8-class 레이블 → 6-class 레이블:
  0 idle          → 0 idle
  1 wave          → 1 wave
  2 left_hand_up  → 2 hand_up
  3 right_hand_up → 2 hand_up
  4 hand_up_both  → 3 hand_up_both
  5 left_pointing → 4 pointing
  6 right_pointing→ 4 pointing
  7 stop          → 5 stop

Usage:
    python convert_8to6class.py
    python convert_8to6class.py --input wholebody_8class.pkl --output wholebody_6class_new.pkl
    python convert_8to6class.py --all
"""

import pickle
import argparse
import copy
from collections import Counter

LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5}

CLASS_NAMES = {
    0: "idle",
    1: "wave",
    2: "hand_up",
    3: "hand_up_both",
    4: "pointing",
    5: "stop",
}

ALL_PKLS = [
    ("wholebody_8class.pkl",                         "wholebody_6class_merged.pkl"),
    ("wholebody_8class_performance.pkl",             "wholebody_6class_performance_merged.pkl"),
    ("wholebody3d_8class.pkl",                       "wholebody3d_6class_merged.pkl"),
    ("wholebody3d_8class_aligned_to_rtmw2d.pkl",     "wholebody3d_6class_aligned_merged.pkl"),
    ("rtmo_l_8class.pkl",                            "rtmo_l_6class_merged.pkl"),
    ("rtmo_m_8class.pkl",                            "rtmo_m_6class_merged.pkl"),
    ("yolo26l_pose_8class.pkl",                      "yolo26l_pose_6class_merged.pkl"),
    ("yolo26x_pose_8class.pkl",                      "yolo26x_pose_6class_merged.pkl"),
    ("yolo26l_pose_8class_pointing_good.pkl",        "yolo26l_pose_6class_pointing_good_merged.pkl"),
    ("yolo26x_pose_8class_pointing_good.pkl",        "yolo26x_pose_6class_pointing_good_merged.pkl"),
    ("yolo26m_pose_2d_from_motionbert_xy_8class.pkl","yolo26m_pose_2d_from_motionbert_6class_merged.pkl"),
    ("yolo26l_motionbert3d_from_2d_8class.pkl",      "yolo26l_motionbert3d_from_2d_6class_merged.pkl"),
    ("yolo26m_motionbert3d_8class.pkl",              "yolo26m_motionbert3d_6class_merged.pkl"),
]


def convert(input_path: str, output_path: str):
    print(f"Loading: {input_path}")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]
    split = data["split"]

    # 레이블 리매핑 (deep copy로 원본 보호)
    new_annotations = []
    for ann in annotations:
        new_ann = copy.copy(ann)
        old_label = ann["label"]
        new_label = LABEL_REMAP.get(old_label)
        if new_label is None:
            print(f"  Warning: unknown label {old_label}, skipping sample {ann.get('frame_dir','?')}")
            continue
        new_ann["label"] = new_label
        new_annotations.append(new_ann)

    # split 인덱스는 annotation 순서 기준 — 삭제된 샘플이 없으면 그대로 유지
    # (unknown label 스킵이 없는 한 인덱스 동일)
    new_data = {"annotations": new_annotations, "split": split}

    # 통계 출력
    print(f"  Converted {len(new_annotations)} samples")
    for split_name, indices in split.items():
        valid_indices = [i for i in indices if i < len(new_annotations)]
        counts = Counter(new_annotations[i]["label"] for i in valid_indices)
        print(f"  {split_name} ({len(valid_indices)} samples):")
        for cls_id in sorted(counts):
            print(f"    {cls_id}: {CLASS_NAMES[cls_id]:15s} — {counts[cls_id]}")

    with open(output_path, "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {output_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="8-class → 6-class pkl 변환")
    parser.add_argument("--input",  type=str, default=None, help="입력 pkl 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 pkl 경로")
    parser.add_argument("--all",    action="store_true",    help="모든 8class pkl 일괄 변환")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        import os
        for src, dst in ALL_PKLS:
            if os.path.exists(src):
                convert(src, dst)
            else:
                print(f"  Skip (not found): {src}")
    elif args.input and args.output:
        convert(args.input, args.output)
    else:
        # 기본: 가장 흔히 쓰는 wholebody_8class.pkl 하나만 변환
        convert("wholebody_8class.pkl", "wholebody_6class_merged.pkl")


if __name__ == "__main__":
    main()
