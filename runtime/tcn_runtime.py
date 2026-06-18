import numpy as np
import torch

from data.ntu_dataset import NTU_ACTION_NAMES


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST, _R_WRIST = 9, 10
_KEEP_IDXS_65 = list(range(0, 23)) + list(range(91, 133))


def extract_skeleton(keypoints, scores, num_joints=65, person_idx=0):
    if keypoints is None or len(keypoints) == 0:
        return None, None

    pidx = min(person_idx, len(keypoints) - 1)
    if num_joints == 65:
        kp = keypoints[pidx][_KEEP_IDXS_65].copy()
        sc = scores[pidx][_KEEP_IDXS_65].copy()
    else:
        kp = keypoints[pidx][:17].copy()
        sc = scores[pidx][:17].copy()
    return kp, sc


def normalize_frame(kp, sc, min_score=0.3):
    kp = kp.copy()
    center = (kp[LEFT_SHOULDER] + kp[RIGHT_SHOULDER]) / 2.0
    kp -= center

    scale = np.linalg.norm(kp[LEFT_SHOULDER] - kp[RIGHT_SHOULDER])
    if scale > 1e-6:
        kp /= scale

    kp[sc < min_score] = 0.0
    return kp


def buffer_to_tensor(frame_buffer, max_frames, device, num_joints=65):
    t_len = len(frame_buffer)
    input_dim = num_joints * 2
    arr = np.stack(list(frame_buffer), axis=0)

    if t_len >= max_frames:
        indices = np.linspace(0, t_len - 1, max_frames, dtype=int)
        arr = arr[indices]
        mask = np.ones(max_frames, dtype=np.float32)
    else:
        padded = np.zeros((max_frames, num_joints, 2), dtype=np.float32)
        padded[:t_len] = arr
        arr = padded
        mask = np.zeros(max_frames, dtype=np.float32)
        mask[:t_len] = 1.0

    features = arr.reshape(max_frames, input_dim).T
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    return features, mask


def buffers_to_batch_tensors(frame_buffers, max_frames, device, num_joints=65):
    feature_list = []
    mask_list = []
    for frame_buffer in frame_buffers:
        feat, mask = buffer_to_tensor(frame_buffer, max_frames, device, num_joints)
        feature_list.append(feat)
        mask_list.append(mask)
    return torch.cat(feature_list, dim=0), torch.cat(mask_list, dim=0)


def compute_pose_features(frame_buffer):
    if len(frame_buffer) < 5:
        return {"wrist_above": False, "n_hands_up": 0, "temporal_std": 1.0}

    recent = list(frame_buffer)[-30:]
    arr = np.stack(recent, axis=0)
    last = arr[-1]

    l_up = last[_L_WRIST, 1] < last[_L_SHOULDER, 1] - 0.3
    r_up = last[_R_WRIST, 1] < last[_R_SHOULDER, 1] - 0.3
    n_hands_up = int(l_up) + int(r_up)

    l_std = np.std(arr[:, _L_WRIST, :], axis=0).mean()
    r_std = np.std(arr[:, _R_WRIST, :], axis=0).mean()
    temporal_std = max(l_std, r_std)

    return {"wrist_above": (n_hands_up >= 1), "n_hands_up": n_hands_up, "temporal_std": temporal_std}


def adjust_probs(probs_dict, pose_features):
    if not pose_features["wrist_above"]:
        return probs_dict

    temporal_std = pose_features["temporal_std"]
    n_hands = pose_features["n_hands_up"]
    static_thr = 0.15
    dynamic_thr = 0.25
    adjusted = dict(probs_dict)

    if temporal_std < static_thr:
        boost_cls = "hands_up_single" if n_hands == 1 else "hands_up_both"
        transfer = adjusted.get("waving", 0.0) * 0.6
        adjusted["waving"] = adjusted.get("waving", 0.0) - transfer
        adjusted[boost_cls] = adjusted.get(boost_cls, 0.0) + transfer
    elif temporal_std > dynamic_thr:
        transfer = adjusted.get("hands_up_single", 0.0) * 0.3
        adjusted["hands_up_single"] = adjusted.get("hands_up_single", 0.0) - transfer
        adjusted["waving"] = adjusted.get("waving", 0.0) + transfer

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted


@torch.no_grad()
def predict_tcn(model, frame_buffer, max_frames, num_classes, device, num_joints=65):
    if len(frame_buffer) < 5:
        return "unknown", 0.0, {}

    features, mask = buffer_to_tensor(frame_buffer, max_frames, device, num_joints)
    logits = model(features, mask)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    probs_dict = {NTU_ACTION_NAMES.get(i, f"class_{i}"): probs[i].item() for i in range(num_classes)}

    pose_features = compute_pose_features(frame_buffer)
    probs_dict = adjust_probs(probs_dict, pose_features)
    pred_cls = max(probs_dict, key=probs_dict.get)
    return pred_cls, probs_dict[pred_cls], probs_dict


@torch.no_grad()
def predict_tcn_batch(model, track_items, max_frames, num_classes, device, num_joints=65):
    valid = [(tid, tr) for tid, tr in track_items if len(tr.buffer) >= 5]
    if not valid:
        return {}

    buffers = [tr.buffer for _, tr in valid]
    features, mask = buffers_to_batch_tensors(buffers, max_frames, device, num_joints)
    logits = model(features, mask)
    probs_batch = torch.softmax(logits, dim=1).detach().cpu().numpy()

    out = {}
    for i, (tid, track) in enumerate(valid):
        probs = probs_batch[i]
        probs_dict = {NTU_ACTION_NAMES.get(c, f"class_{c}"): float(probs[c]) for c in range(num_classes)}
        probs_dict = adjust_probs(probs_dict, compute_pose_features(track.buffer))
        pred_cls = max(probs_dict, key=probs_dict.get)
        out[tid] = (pred_cls, probs_dict[pred_cls], probs_dict)
    return out

