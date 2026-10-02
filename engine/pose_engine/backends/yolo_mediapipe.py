"""Multi-person backend: YOLO detects/crops, MediaPipe estimates one pose per person.

Architecture (YOLO-crop pattern + per-ID estimator):
1. YOLO-Pose (`yolo26n-pose.pt`) tracks (ByteTrack) and delivers
   stable `(track_id, bbox)` + YOLO keypoints as fallback.
2. ONE MediaPipe Pose instance is kept per `track_id` (video mode,
   with its own temporal state). Each frame the bbox is cropped with
   padding and run through its dedicated estimator.
3. Crop keypoints (0..1 of the crop) map to GLOBAL 0..1 coords
   of the frame, like YOLO, so T-pose/drawing/scoring agree.
4. When MediaPipe fails on a crop (heavy occlusion), the fallback is
   that same box's YOLO keypoints (there is always a pose per dancer
   while YOLO sees the box).

Advantage over direct YOLO-pose: 33 MediaPipe landmarks per person
(consistent with normalizer/comparator/T-pose) with per-person temporal
tracking over centered crops; the track_id -> dancer_id assignment never
permutes because each estimator belongs to a single track.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.pose_engine.backends.base import PoseBackend
from engine.pose_engine.backends.mediapipe import MediaPipeBackend
from engine.pose_engine.backends.yolo_pose import YoloPoseBackend
from engine.pose_engine.skeleton import Joint, Pose

Bbox = Tuple[int, int, int, int]


def _center(b: Bbox) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def _dist(a: Bbox, b: Bbox) -> float:
    """Center distance normalized by size (pattern compareDist)."""
    ax, ay = _center(a)
    bx, by = _center(b)
    diag = max(1.0, float(a[2] + a[3] + b[2] + b[3]) / 2.0)
    return (abs(ax - bx) + abs(ay - by)) / diag


class YoloMediapipeBackend(PoseBackend):
    """YOLO tracking + one dedicated MediaPipe Pose per tracked person."""

    name = "yolo-mediapipe"
    num_keypoints = 33

    def __init__(
        self,
        model: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        max_persons: int = 4,
        tracker: str | None = "botsort-reid",
        device: str | None = "auto",
        pad: float = 0.12,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        # 0 = lite: same quality on YOLO-centered crops and ~2x
        # faster (measured: 13/13 visible joints same as complexity 1).
        model_complexity: int = 0,
        expire_frames: int = 30,
        mp_factory=None,
        **kwargs,
    ):
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_persons = int(max_persons)
        self.pad = float(pad)
        self._mp_kwargs = {
            "min_detection_confidence": float(min_detection_confidence),
            "min_tracking_confidence": float(min_tracking_confidence),
            "model_complexity": int(model_complexity),
        }
        self._mp_factory = mp_factory  # injection for tests
        self.expire_frames = int(expire_frames)
        self._yolo = YoloPoseBackend(model=model, conf=conf, iou=iou, max_persons=max_persons,
                                      tracker=tracker, device=device)
        self.model_name = getattr(self._yolo, "model_name", model)
        # track_id -> {"mp": MediaPipeBackend, "bbox": Bbox, "missed": int}
        self._estimators: Dict[int, dict] = {}
        self._next_neg = -1  # temporary ids for trackless detections

    # -- per-person estimator management -------------------------------
    def _get_estimator(self, track_id: int) -> MediaPipeBackend:
        ent = self._estimators.get(track_id)
        if ent is None:
            mp = self._mp_factory() if self._mp_factory else MediaPipeBackend(**self._mp_kwargs)
            ent = {"mp": mp, "bbox": None, "missed": 0}
            self._estimators[track_id] = ent
        return ent["mp"]

    def _match_or_create(self, tid: Optional[int], bbox: Bbox) -> int:
        """Resolves the estimator for a detection.

        With a known track_id its estimator is reused (example pattern:
        the estimator belongs to the object). Without track_id the closest
        bbox estimator is searched (compareDist); when none is near, a
        temporary one is created. Returns the estimator key to use.
        """
        if tid is not None and tid in self._estimators:
            return tid
        if tid is not None:
            return tid  # _get_estimator will create it
        best, best_d = None, None
        for key, ent in self._estimators.items():
            prev = ent.get("bbox")
            if prev is None:
                continue
            dd = _dist(prev, bbox)
            if best_d is None or dd < best_d:
                best, best_d = key, dd
        if best is not None and best_d is not None and best_d <= 1.0:
            return best
        key = self._next_neg
        self._next_neg -= 1
        return key

    def _expire(self, seen: set) -> None:
        """Forgets estimators of tracks unseen for many frames."""
        for key in list(self._estimators.keys()):
            ent = self._estimators[key]
            if key in seen:
                ent["missed"] = 0
                continue
            ent["missed"] = int(ent.get("missed", 0)) + 1
            # temporaries (trackless) expire faster
            ttl = 5 if key < 0 else self.expire_frames
            if ent["missed"] > ttl:
                try:
                    ent["mp"].close()
                except Exception:
                    pass
                del self._estimators[key]

    # -- API PoseBackend ---------------------------------------------------
    def _crop(self, frame: np.ndarray, bbox: Bbox):
        h, w = frame.shape[:2]
        x, y, wb, hb = bbox
        px, py = int(wb * self.pad), int(hb * self.pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + wb + px), min(h, y + hb + py)
        if x1 <= x0 or y1 <= y0:
            return None, None
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None, None
        return crop, (x0, y0, x1 - x0, y1 - y0)

    @staticmethod
    def _to_global(pose_crop: Pose, crop_box: Bbox, w_img: int, h_img: int) -> Pose:
        """Maps crop-coord pose (0..1) to global coords (0..1)."""
        x0, y0, cw, ch = crop_box
        out = Pose()
        for pname, j in pose_crop.joints.items():
            gx = (x0 + j.x * cw) / float(w_img) if w_img else j.x
            gy = (y0 + j.y * ch) / float(h_img) if h_img else j.y
            out.joints[pname] = Joint(
                x=max(0.0, min(1.0, gx)),
                y=max(0.0, min(1.0, gy)),
                z=j.z,
                visibility=j.visibility,
            )
        return out

    def track_multi(self, frame: np.ndarray) -> List[Tuple[Optional[int], Bbox, Pose]]:
        """Returns [(track_id|None, bbox, pose_global)] (UNsorted).

        One entry per YOLO box (up to max_persons): the dedicated
        estimator's MediaPipe pose, or YOLO-keypoint fallback when MediaPipe fails.
        """
        h_img, w_img = frame.shape[:2]
        try:
            yolo_dets = self._yolo.track_multi(frame)
        except Exception:
            return []
        if not yolo_dets:
            self._expire(set())
            return []
        out: List[Tuple[Optional[int], Bbox, Pose]] = []
        seen: set = set()
        for (tid, bbox, yolo_pose) in yolo_dets[: self.max_persons]:
            key = self._match_or_create(tid, bbox)
            mp = self._get_estimator(key)
            pose_global: Optional[Pose] = None
            crop, crop_box = self._crop(frame, bbox)
            if crop is not None:
                try:
                    raw_crop = mp.detect(crop)
                except Exception:
                    raw_crop = Pose()
                if not raw_crop.is_empty():
                    pose_global = self._to_global(raw_crop, crop_box, w_img, h_img)
            if (pose_global is None or pose_global.is_empty()) and not yolo_pose.is_empty():
                # fallback: same-box YOLO keypoints (always a dancer)
                pose_global = yolo_pose
            if pose_global is None or pose_global.is_empty():
                continue
            seen.add(key)
            self._estimators[key]["bbox"] = bbox
            out.append((tid, bbox, pose_global))
            if len(out) >= self.max_persons:
                break
        self._expire(seen)
        return out

    def detect_multi(self, frame: np.ndarray) -> List[Tuple[Bbox, Pose]]:
        # No tracking (single isolated frame): delegates to direct YOLO-pose.
        try:
            return self._yolo.detect_multi(frame)
        except Exception:
            return []

    def detect(self, frame: np.ndarray) -> Pose:
        multi = self.detect_multi(frame)
        if not multi:
            return Pose()
        multi_sorted = sorted(multi, key=lambda x: x[0][2] * x[0][3], reverse=True)
        return multi_sorted[0][1]

    def reset_track(self) -> None:
        """Resets YOLO tracking and closes per-person estimators."""
        try:
            self._yolo.reset_track()
        except Exception:
            pass
        for ent in list(self._estimators.values()):
            try:
                ent["mp"].close()
            except Exception:
                pass
        self._estimators.clear()
        self._next_neg = -1

    def close(self) -> None:
        for ent in list(self._estimators.values()):
            try:
                ent["mp"].close()
            except Exception:
                pass
        self._estimators.clear()
        try:
            self._yolo.close()
        except Exception:
            pass
