from collections import deque
import numpy as np


class TrackState:
    """Per-person state container shared by inference pipelines."""

    def __init__(self, buf_size=0):
        maxlen = max(1, int(buf_size))
        self.buffer = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)  # buffer와 1:1 대응하는 프레임 캡처 시각(초)
        self.pred_cls = "unknown"
        self.pred_conf = 0.0
        self.pred_probs = {}
        self.last_seen_frame = 0
        self.last_center = None
        self.missing_frames = 0
        self.last_depth_z = None
        self.last_bbox = None  # (x_min, y_min, x_max, y_max)
        self.bbox_wh = None    # (width, height)
        self.bbox_area = 0.0

    def push(self, kp, timestamp):
        """프레임 keypoint와 캡처 시각을 함께 버퍼에 추가 (시간 기반 리샘플용).

        buffer.append만 쓰면 timestamps가 비어 시간 리샘플이 비활성화되니,
        실시간 경로에서는 항상 이 메서드로 추가한다.
        """
        self.buffer.append(kp)
        self.timestamps.append(float(timestamp))

    def to_dict(self):
        return {
            "pred_cls": self.pred_cls,
            "pred_conf": float(self.pred_conf),
            "pred_probs": dict(self.pred_probs),
            "last_seen_frame": int(self.last_seen_frame),
            "last_center": tuple(self.last_center) if self.last_center is not None else None,
            "missing_frames": int(self.missing_frames),
            "buffer_len": len(self.buffer),
            "last_bbox": tuple(self.last_bbox) if self.last_bbox is not None else None,
            "bbox_wh": tuple(self.bbox_wh) if self.bbox_wh is not None else None,
            "bbox_area": float(self.bbox_area),
        }


class TrackManager:
    """TrackState container + query API."""

    def __init__(self):
        self._tracks = {}

    def __contains__(self, tid):
        return tid in self._tracks

    def __len__(self):
        return len(self._tracks)

    def __iter__(self):
        return iter(self._tracks)

    def __getitem__(self, tid):
        return self._tracks[tid]

    def __setitem__(self, tid, state):
        self._tracks[tid] = state

    def __delitem__(self, tid):
        del self._tracks[tid]

    def clear(self):
        self._tracks.clear()

    def keys(self):
        return self._tracks.keys()

    def values(self):
        return self._tracks.values()

    def items(self):
        return self._tracks.items()

    def get_track_state(self, tid):
        st = self._tracks.get(tid)
        return None if st is None else st.to_dict()

    def get_all_states(self):
        return {tid: st.to_dict() for tid, st in self._tracks.items()}


def match_centers_to_tracks(det_centers, tracks, frame_count, ttl_frames, dist_thr, next_tid):
    """
    Greedy center-distance matching between detections and existing tracks.
    Returns:
        det_to_tid: list[int] length=len(det_centers)
        next_tid: updated next_tid
    """
    n_det = len(det_centers)
    det_to_tid = [-1] * n_det
    if n_det == 0:
        return det_to_tid, next_tid

    active_tids = [
        tid for tid, tr in tracks.items()
        if (frame_count - tr.last_seen_frame) <= ttl_frames and tr.last_center is not None
    ]
    unmatched_dets = set(range(n_det))
    unmatched_tids = set(active_tids)

    while unmatched_dets and unmatched_tids:
        best = None
        for d in unmatched_dets:
            cx, cy = det_centers[d]
            for tid in unmatched_tids:
                tx, ty = tracks[tid].last_center
                dist = float(np.hypot(cx - tx, cy - ty))
                if best is None or dist < best[0]:
                    best = (dist, d, tid)

        if best is None or best[0] > dist_thr:
            break

        _, d, tid = best
        det_to_tid[d] = tid
        unmatched_dets.remove(d)
        unmatched_tids.remove(tid)

    for d in unmatched_dets:
        det_to_tid[d] = next_tid
        next_tid += 1

    return det_to_tid, next_tid
