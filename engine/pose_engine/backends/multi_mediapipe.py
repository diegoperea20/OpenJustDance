"""Multi-person backend based on per-crop MediaPipe (HOG + tracker)."""

from __future__ import annotations

import cv2
import numpy as np

from engine.pose_engine.backends.base import PoseBackend
from engine.pose_engine.backends.mediapipe import MediaPipeBackend
from engine.pose_engine.detector import detect_persons
from engine.pose_engine.skeleton import Pose
from engine.tracking.tracker import MultiPersonTracker


class MultiMediapipeBackend(PoseBackend):
    """Detect up to N people: HOG -> NMS -> tracker -> crop -> MediaPipe per person.

    Uses one internal MediaPipeBackend per crop (reuses a single instance for efficiency).
    """

    name = "multi-mediapipe"
    num_keypoints = 33

    def __init__(self, max_persons: int = 4, work_width: int = 640, **kwargs):
        self._inner = MediaPipeBackend(**kwargs)
        self._tracker = MultiPersonTracker(iou_thresh=0.15, max_lost=18)
        self.max_persons = max_persons
        self.work_width = work_width

    def detect(self, frame: np.ndarray) -> Pose:
        # legacy compat: return the first pose if any
        multi = self.detect_multi(frame)
        if not multi:
            return Pose()
        return multi[0][1]

    def detect_multi(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], Pose]]:
        dets = detect_persons(frame, work_width=self.work_width)
        # sort by x and limit
        bboxes = [(x, y, w, h) for x, y, w, h, _ in dets[:self.max_persons]]
        tracked = self._tracker.update(bboxes)
        # tracked is {track_id: bbox} of actives; but we need to map current detections
        # If there are detections, use tracked directly
        out: list[tuple[tuple[int, int, int, int], Pose]] = []
        h_img, w_img = frame.shape[:2]
        for tid, bbox in tracked.items():
            x, y, wb, hb = bbox
            # pad 12%
            pad_x = int(wb * 0.12); pad_y = int(hb * 0.12)
            x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
            x1 = min(w_img, x + wb + pad_x); y1 = min(h_img, y + hb + pad_y)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            pose = self._inner.detect(crop)
            if pose.is_empty():
                continue
            # convert crop coords (0..1) to frame coords (0..1) for consistent drawing?
            # Left in crop 0..1 space for a normalizer expecting crop 0..1; but better to map to frame?
            # For multi, the pose stays in crop space (0..1 of the crop) but the bbox is also attached for draw.
            # For scoring the normalized pose is used either way.
            # To keep the bbox for the caller, return the expanded original bbox
            out.append(((x0, y0, x1 - x0, y1 - y0), pose))
            if len(out) >= self.max_persons:
                break
        return out

    def close(self) -> None:
        self._inner.close()
