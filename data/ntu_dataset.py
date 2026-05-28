"""
NTU RGB+D Skeleton Dataset for Temporal Convolutional Network (TCN).

PKL format (from pyskl/mmaction2):
    data["annotations"]: list of dicts
        - "frame_dir": str (e.g. "S001C001P001R001A024")
        - "label": int (remapped 0..K-1)
        - "total_frames": int
        - "keypoint": np.ndarray (M, T, V, C)  # M=persons, T=frames, V=17 joints, C=2 (x,y)
        - "keypoint_score": np.ndarray (M, T, V)
    data["split"]: dict
        - "xsub_train": list of int (annotation indices)
        - "xsub_val": list of int
        - "xview_train": list of int
        - "xview_val": list of int
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.skeleton_ops import get_swap_pairs


# ===== Final label mapping (6-class merged dataset) =====
# 0: idle             — NTU (reading, writing, phone_call, playing_phone, typing)
# 1: waving           — NTU A023 hand_waving
# 2: hands_up_single  — 커스텀 수집 (한손 들기)
# 3: hands_up_both    — NTU A095 capitulate (양손 들기)
# 4: pointing         — NTU A031 pointing
# 5: stop             — 커스텀 수집 (정지 제스처)
NTU_ACTION_NAMES = {
    0: "idle",
    1: "waving",
    2: "hands_up_single",
    3: "hands_up_both",
    4: "pointing",
    5: "stop",
}


class NTUSkeletonDataset(Dataset):
    """
    NTU Skeleton dataset for TCN.

    Each sample returns:
        keypoint:       (C_in, T_fixed)   — 정규화된 skeleton features
        keypoint_score: (V, T_fixed)      — confidence scores
        label:          int
    """

    def __init__(
        self,
        pkl_path: str,
        split: str = "xsub_train",
        max_frames: int = 120,
        person_idx: int = 0,
        augment: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            pkl_path: subset pkl 파일 경로
            split: "xsub_train", "xsub_val", "xview_train", "xview_val"
            max_frames: 시퀀스 최대 길이 (패딩/크롭)
            person_idx: 사용할 사람 인덱스 (0 = 첫 번째 사람)
            augment: 데이터 증강 여부
            normalize: 어깨 기반 정규화 여부
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        annotations = data["annotations"]
        split_indices = data["split"][split]

        self.samples = [annotations[i] for i in split_indices]
        self.max_frames = max_frames
        self.person_idx = person_idx
        self.augment = augment
        self.normalize = normalize

        # Detect num_joints from data
        sample0 = self.samples[0]
        self._num_joints = sample0["keypoint"].shape[2]  # V dimension

        # Stats
        labels = [s["label"] for s in self.samples]
        self.label_counts = Counter(labels)
        self.num_classes = len(self.label_counts)

        print(f"[{split}] Loaded {len(self.samples)} samples, "
              f"{self.num_classes} classes, "
              f"joints: {self._num_joints}, "
              f"distribution: {dict(self.label_counts)}")

    @property
    def num_joints(self):
        return self._num_joints

    def __len__(self):
        return len(self.samples)

    def _normalize_skeleton(self, keypoint, keypoint_score):
        """
        어깨 중심 정규화.
        keypoint: (T, V, C)  — V=17, C=2
        keypoint_score: (T, V)

        Returns:
            normalized keypoint: (T, V, C)
        """
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6

        kp = keypoint.copy()
        sc = keypoint_score.copy()

        # 각 프레임별로 정규화
        for t in range(kp.shape[0]):
            ls = kp[t, LEFT_SHOULDER]
            rs = kp[t, RIGHT_SHOULDER]

            # 어깨 중점 기준 이동
            center = (ls + rs) / 2.0
            kp[t] -= center

            # 어깨 거리 기준 스케일링
            scale = np.linalg.norm(ls - rs)
            if scale > 1e-6:
                kp[t] /= scale

            # 낮은 confidence keypoint 처리
            low_conf = sc[t] < 0.3
            kp[t, low_conf] = 0.0

        return kp

    def _pad_or_crop(self, keypoint, keypoint_score):
        """
        시퀀스를 max_frames로 맞춤.
        keypoint: (T, V, C)
        keypoint_score: (T, V)

        Returns:
            padded keypoint: (max_frames, V, C)
            padded score: (max_frames, V)
            mask: (max_frames,)  — 1 for valid, 0 for padding
        """
        T = keypoint.shape[0]

        if T >= self.max_frames:
            # Uniform sampling (temporal downsampling)
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            kp_out = keypoint[indices]
            sc_out = keypoint_score[indices]
            mask = np.ones(self.max_frames, dtype=np.float32)
        else:
            # Zero padding
            V, C = keypoint.shape[1], keypoint.shape[2]
            kp_out = np.zeros((self.max_frames, V, C), dtype=np.float32)
            sc_out = np.zeros((self.max_frames, V), dtype=np.float32)
            mask = np.zeros(self.max_frames, dtype=np.float32)

            kp_out[:T] = keypoint
            sc_out[:T] = keypoint_score
            mask[:T] = 1.0

        return kp_out, sc_out, mask

    def _augment(self, keypoint):
        """
        Data augmentation for skeleton sequences.
        keypoint: (T, V, C)
        """
        kp = keypoint.copy()
        V = kp.shape[1]

        # 1) Random horizontal flip (x축 반전)
        if np.random.random() < 0.5:
            kp[:, :, 0] *= -1
            # Swap left/right joint pairs (body / feet / hands)
            swap_pairs = get_swap_pairs(V)
            for i, j in swap_pairs:
                kp[:, [i, j]] = kp[:, [j, i]]

        # 2) Random temporal shift (시간축 이동)
        if np.random.random() < 0.3:
            shift = np.random.randint(-3, 4)
            kp = np.roll(kp, shift, axis=0)

        # 3) Random noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, kp.shape).astype(np.float32)
            kp += noise

        # 4) Random scale
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            kp *= scale

        return kp

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        keypoint = sample["keypoint"]           # (M, T, V, C)
        keypoint_score = sample["keypoint_score"]  # (M, T, V)

        # 사용할 사람 선택
        M = keypoint.shape[0]
        pidx = min(self.person_idx, M - 1)
        kp = keypoint[pidx].astype(np.float32)          # (T, V, C)
        sc = keypoint_score[pidx].astype(np.float32)     # (T, V)

        # 정규화
        if self.normalize:
            kp = self._normalize_skeleton(kp, sc)

        # 증강
        if self.augment:
            kp = self._augment(kp)

        # 패딩/크롭
        kp, sc, mask = self._pad_or_crop(kp, sc)

        # (T, V, C) → (V*C, T) for TCN input  (channel-first)
        T, V, C = kp.shape
        features = kp.reshape(T, V * C).T  # (V*C, T) = (34, T)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    def get_class_weights(self):
        """역빈도 기반 클래스 가중치."""
        total = sum(self.label_counts.values())
        num_cls = self.num_classes
        max_label = max(self.label_counts.keys())
        weights = []
        for i in range(max_label + 1):
            count = self.label_counts.get(i, 0)
            if count > 0:
                weights.append(total / (num_cls * count))
            else:
                weights.append(1.0)
        return torch.FloatTensor(weights)

    def get_action_name(self, label_idx):
        return NTU_ACTION_NAMES.get(label_idx, f"class_{label_idx}")
