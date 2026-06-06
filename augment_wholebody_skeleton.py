"""
Offline augmentation for wholebody skeleton PKL files.

Input/output format matches extract_wholebody_skeleton.py:
    annotations[i]["keypoint"]       -> (M, T, V, 2)
    annotations[i]["keypoint_score"] -> (M, T, V)

The script adds augmented copies to the train split by default. Each copy applies
x/y scaling plus an SO(3)-style yaw rotation around the image z-axis
(2D image-plane rotation). A transformed clip is discarded if any confident
keypoint leaves the image bounds.

Usage:
    python augment_wholebody_skeleton.py --in_pkl wholebody_6class.pkl --out wholebody_6class_aug.pkl
    python augment_wholebody_skeleton.py --copies 3 --x-scales 0.9 1.0 1.1 --y-scales 0.9 1.0 1.1
"""

import argparse
import copy
import os
import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm


DEFAULT_SCALES = (0.9, 1.0, 1.1)
DEFAULT_YAW_DEGREES = (-15.0, -10.0, 10.0, 15.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment wholebody skeleton PKL with scale + yaw transforms"
    )
    parser.add_argument("--in_pkl", type=str, default="wholebody_6class.pkl")
    parser.add_argument("--out", type=str, default="wholebody_6class_aug.pkl")
    parser.add_argument("--copies", type=int, default=2,
                        help="Number of accepted augmented copies to try per source sample")
    parser.add_argument("--scales", type=float, nargs="+", default=None,
                        help="Legacy uniform scale list. If set, used for both x/y scales.")
    parser.add_argument("--x-scales", dest="x_scales", type=float, nargs="+",
                        default=list(DEFAULT_SCALES))
    parser.add_argument("--y-scales", dest="y_scales", type=float, nargs="+",
                        default=list(DEFAULT_SCALES))
    parser.add_argument("--yaw-degrees", dest="yaw_degrees", type=float, nargs="+",
                        default=list(DEFAULT_YAW_DEGREES))
    parser.add_argument("--score_thr", type=float, default=0.3,
                        help="Only keypoints above this score are used for bounds checks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-prefixes", type=str, nargs="+",
                        default=["xsub", "xset"],
                        help="Split prefixes whose *_train entries receive augmented samples")
    parser.add_argument("--augment-val", action="store_true",
                        help="Also augment *_val splits. Usually leave this off.")
    return parser.parse_args()


def _get_img_hw(annotation):
    shape = annotation.get("img_shape") or annotation.get("original_shape") or (480, 640)
    if len(shape) < 2:
        return 480, 640
    return int(shape[0]), int(shape[1])


def _valid_mask(keypoint, score, score_thr):
    finite = np.isfinite(keypoint).all(axis=-1)
    positive = np.any(keypoint != 0.0, axis=-1)
    confident = score >= score_thr
    return finite & positive & confident


def _frame_centers(keypoint, score, score_thr):
    """
    Return one transform center per person/frame as (M, T, 1, 2).
    Shoulder midpoint is preferred; otherwise confident keypoint mean is used.
    """
    m, t, _, _ = keypoint.shape
    centers = np.zeros((m, t, 1, 2), dtype=np.float32)
    mask = _valid_mask(keypoint, score, score_thr)

    left_shoulder = 5
    right_shoulder = 6

    for person_idx in range(m):
        for frame_idx in range(t):
            frame_kp = keypoint[person_idx, frame_idx]
            frame_mask = mask[person_idx, frame_idx]

            if (
                frame_mask[left_shoulder]
                and frame_mask[right_shoulder]
            ):
                center = (frame_kp[left_shoulder] + frame_kp[right_shoulder]) * 0.5
            elif frame_mask.any():
                center = frame_kp[frame_mask].mean(axis=0)
            else:
                center = np.array([0.0, 0.0], dtype=np.float32)

            centers[person_idx, frame_idx, 0] = center

    return centers


def transform_keypoints(keypoint, score, scale_x, scale_y, yaw_degrees, score_thr):
    """
    Apply x/y scale and image-plane yaw rotation around per-frame centers.
    """
    kp = keypoint.astype(np.float32, copy=True)
    centers = _frame_centers(kp, score, score_thr)
    theta = np.deg2rad(yaw_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    centered = kp - centers
    scaled = centered * np.array([float(scale_x), float(scale_y)], dtype=np.float32)
    transformed = scaled @ rot.T
    transformed = transformed + centers

    valid = _valid_mask(kp, score, score_thr)
    transformed[~valid] = kp[~valid]
    return transformed.astype(np.float32)


def inside_image(keypoint, score, height, width, score_thr):
    mask = _valid_mask(keypoint, score, score_thr)
    if not mask.any():
        return False

    pts = keypoint[mask]
    x_ok = (pts[:, 0] >= 0.0) & (pts[:, 0] < float(width))
    y_ok = (pts[:, 1] >= 0.0) & (pts[:, 1] < float(height))
    return bool(np.all(x_ok & y_ok))


def build_augmented_annotation(annotation, scale_x, scale_y, yaw_degrees, score_thr, suffix):
    aug = copy.deepcopy(annotation)
    kp = aug["keypoint"].astype(np.float32, copy=True)
    score = aug["keypoint_score"].astype(np.float32, copy=False)
    height, width = _get_img_hw(aug)

    aug_kp = transform_keypoints(kp, score, scale_x, scale_y, yaw_degrees, score_thr)
    if not inside_image(aug_kp, score, height, width, score_thr):
        return None

    aug["keypoint"] = aug_kp
    aug["frame_dir"] = f"{annotation['frame_dir']}__aug_{suffix}"
    aug["augmented"] = True
    aug["augment_params"] = {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "yaw_degrees": float(yaw_degrees),
        "score_thr": float(score_thr),
    }
    return aug


def _target_indices_for_split(data, split_name):
    split = data.get("split", {})
    return list(split.get(split_name, []))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    in_path = os.path.abspath(args.in_pkl)
    out_path = os.path.abspath(args.out)

    with open(in_path, "rb") as f:
        data = pickle.load(f)

    x_scales = args.scales if args.scales is not None else args.x_scales
    y_scales = args.scales if args.scales is not None else args.y_scales

    annotations = list(data["annotations"])
    original_count = len(annotations)
    split = copy.deepcopy(data.get("split", {}))

    train_keys = [f"{prefix}_train" for prefix in args.split_prefixes]
    target_indices = sorted({
        idx
        for key in train_keys
        for idx in _target_indices_for_split(data, key)
    })

    if args.augment_val:
        val_keys = [f"{prefix}_val" for prefix in args.split_prefixes]
        target_indices = sorted(set(target_indices) | {
            idx
            for key in val_keys
            for idx in _target_indices_for_split(data, key)
        })

    if not target_indices:
        target_indices = list(range(original_count))

    added_by_source = {}
    rejected = 0

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Original annotations: {original_count}")
    print(f"Augmenting source samples: {len(target_indices)}")
    print(f"Copies per source: {args.copies}")

    for source_idx in tqdm(target_indices, desc="Augmenting"):
        source = annotations[source_idx]
        accepted = []
        attempts = 0
        max_attempts = max(args.copies * 20, 20)

        while len(accepted) < args.copies and attempts < max_attempts:
            attempts += 1
            scale_x = float(rng.choice(x_scales))
            scale_y = float(rng.choice(y_scales))
            yaw = float(rng.choice(args.yaw_degrees))
            suffix = f"sx{scale_x:.3f}_sy{scale_y:.3f}_y{yaw:+.1f}_{len(accepted)}".replace(".", "p")

            aug = build_augmented_annotation(
                source,
                scale_x=scale_x,
                scale_y=scale_y,
                yaw_degrees=yaw,
                score_thr=args.score_thr,
                suffix=suffix,
            )
            if aug is None:
                rejected += 1
                continue

            accepted.append(len(annotations))
            annotations.append(aug)

        added_by_source[source_idx] = accepted

    # Add accepted augmented samples only to splits that contained their source.
    for split_key, indices in list(split.items()):
        if split_key.endswith("_val") and not args.augment_val:
            continue

        expanded = list(indices)
        for source_idx in indices:
            expanded.extend(added_by_source.get(source_idx, []))
        split[split_key] = expanded

    out_data = copy.deepcopy(data)
    out_data["annotations"] = annotations
    out_data["split"] = split
    out_data["augmentation"] = {
        "source": os.path.basename(in_path),
        "copies": args.copies,
        "x_scales": [float(x) for x in x_scales],
        "y_scales": [float(x) for x in y_scales],
        "yaw_degrees": [float(x) for x in args.yaw_degrees],
        "score_thr": float(args.score_thr),
        "augmented_val": bool(args.augment_val),
    }

    with open(out_path, "wb") as f:
        pickle.dump(out_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    labels_before = Counter(ann["label"] for ann in data["annotations"])
    labels_after = Counter(ann["label"] for ann in annotations)

    print("\nDone")
    print(f"  Added annotations: {len(annotations) - original_count}")
    print(f"  Rejected transforms: {rejected}")
    print(f"  Label counts before: {dict(sorted(labels_before.items()))}")
    print(f"  Label counts after:  {dict(sorted(labels_after.items()))}")
    for key in sorted(split):
        print(f"  Split {key}: {len(data['split'].get(key, []))} -> {len(split[key])}")


if __name__ == "__main__":
    main()
