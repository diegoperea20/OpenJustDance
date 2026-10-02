"""YOLO-Pose backend for multi-person (4 dancers) using Ultralytics.

Uses YOLO26n-pose / YOLO11n-pose / YOLOv8n-pose depending on availability.
Detects up to N people in a single forward pass, mapping COCO-17 -> SEMANTIC_JOINTS.

Tracking (model.track, persist=True) with appearance BoT-SORT + ReID
(engine/pose_engine/trackers/botsort-reid.yaml) or tuned ByteTrack:
IDs survive overlaps where one dancer covers another. See
https://docs.ultralytics.com/modes/track
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

from engine.pose_engine.backends.base import PoseBackend
from engine.pose_engine.skeleton import Joint, Pose

# COCO 17 -> SEMANTIC (head = nose)
COCO_TO_SEMANTIC = {
    0: "head",           # nose
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# For fallback when the model is missing
CANDIDATE_MODELS = [
    "yolo26n-pose.pt",   # requested by user (may not exist)
    "yolo11n-pose.pt",
    "yolov8n-pose.pt",
    "yolo11s-pose.pt",
]

# Ultralytics trackers (model.track). Own names -> repo yamls
# (engine/pose_engine/trackers/); "botsort"/"bytetrack" -> stock built-in.
TRACKER_FILES = {
    "botsort-reid": "botsort-reid.yaml",  # BoT-SORT + appearance ReID (default anti-overlap)
    "botsort-reid-gmc": "botsort-reid-gmc.yaml",  # + GMC when the camera moves (slow on CPU)
    "bytetrack-dance": "bytetrack-dance.yaml",  # tuned ByteTrack, no ReID (fast)
}


def pick_device(want: str | None = "auto") -> str:
    """Picks the inference device: "auto" -> cuda when torch sees it, else cpu.

    Accepts "auto"/"cpu"/"cuda"/"0"...; any other value is returned as-is.
    """
    if want in (None, "", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return want


def half_for_device(device: str | None) -> bool:
    """FP16 on CUDA only (on CPU half is slower or unsupported)."""
    try:
        return str(device).strip().lower().startswith("cuda")
    except Exception:
        return False


def silence_onnx_cuda_noise() -> None:
    """Lowers the ONNX Runtime logger to FATAL.

    ReID (yolo26n-reid.onnx) runs on CPU on most PCs because the pip
    onnxruntime CUDA EP demands exact cuDNN/CUDA builds almost nobody has;
    without this, every session prints an error block even though the CPU
    fallback works fine. Real failures still raise a Python exception.
    """
    try:
        import onnxruntime as ort

        ort.set_default_logger_severity(4)
    except Exception:
        pass


def resolve_tracker_cfg(name: str | None) -> str | None:
    """Resolves a tracker name to a yaml/word understood by model.track.

    - "botsort-reid"/"bytetrack-dance" -> tuned repo yaml (absolute path).
    - "botsort"/"bytetrack" -> ultralytics built-in ("botsort.yaml"...).
    - existing path -> as-is. None/empty -> None (ultralytics default).
    """
    if not name:
        return None
    if name in TRACKER_FILES:
        p = Path(__file__).resolve().parents[1] / "trackers" / TRACKER_FILES[name]
        if p.exists():
            return str(p)
        # when the yaml is missing, fall back to the equivalent built-in
        return "botsort.yaml" if "botsort" in name else "bytetrack.yaml"
    p = Path(name)
    if p.exists() or (len(p.suffix) and p.suffix.lower() in (".yaml", ".yml")):
        return name
    if name in ("botsort", "bytetrack"):
        return f"{name}.yaml"
    return name


class YoloPoseBackend(PoseBackend):
    """YOLO Pose multi-person.

    Lazy ultralytics model load. Without GPU it runs on CPU.
    Supports detect() (one pose) and detect_multi() (up to N).
    """

    name = "yolo-pose"
    num_keypoints = 17

    def __init__(self, model: str | None = None, conf: float = 0.4, iou: float = 0.5, max_persons: int = 4,
                 min_box_w: int = 20, min_box_h: int = 40, min_kp_conf: float = 0.15,
                 min_joints: int = 4, tracker: str | None = "botsort-reid",
                 device: str | None = "auto", half: bool | None = None, **kwargs):
        # resolve model
        self.model_name = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_persons = int(max_persons)
        self.min_box_w = int(min_box_w)
        self.min_box_h = int(min_box_h)
        self.min_kp_conf = float(min_kp_conf)
        self.min_joints = int(min_joints)
        self.tracker = tracker  # requested name ("botsort-reid", "bytetrack", ...)
        self.tracker_cfg: str | None = None  # resolved and verified yaml (warmup)
        self.tracker_error: str | None = None
        self.device = pick_device(device)  # "cuda" when a GPU exists, else "cpu"
        # FP16 on CUDA; FP32 on CPU. None = automatic per device.
        # Passed as `quantize` (ultralytics>=8.3 API): the old `half=True`
        # recreates the predictor+tracker on EVERY call (no ID persistence
        # and the ReID session re-created per frame). `quantize` must be the
        # SAME in warmup and every inference to reuse the predictor.
        self.half = bool(half) if half is not None else half_for_device(self.device)
        self.quantize = 16 if self.half else None
        self.tracker_device = self.device  # real tracker device after warmup
        silence_onnx_cuda_noise()  # ReID falls back to CPU without dumping the CUDA-EP block
        self._model = None
        self._load_error: str | None = None
        self._try_load()
        self._warmup_tracker()

    def health_check(self) -> None:
        """One strict real inference (no swallowed errors).

        Raises RuntimeError when the model did not load or the device
        underperforms (e.g. requested "cuda" without usable CUDA). Used by the
        camera thread for the cuda -> cpu cascade.
        """
        if self._model is None:
            raise RuntimeError(self._load_error or "modelo YOLO no cargado")
        probe = np.zeros((320, 320, 3), dtype=np.uint8)
        self._model(probe, verbose=False, device=self.device, quantize=self.quantize)

    def _try_load(self):
        from ultralytics import YOLO
        # Prefer the repo-local .pt (yolo26n-pose.pt) before downloading.
        # That way aespavideo.mp4 uses the requested model with no network.
        repo_root = Path(__file__).resolve().parents[3]
        local_hits: list[str] = []
        for root in [Path.cwd(), repo_root]:
            try:
                for pt in root.glob("yolo*-pose.pt"):
                    s = str(pt)
                    if s not in local_hits:
                        local_hits.append(s)
            except Exception:
                continue
        # Order: explicit model > repo-local yolo26n > other locals > remote candidates.
        candidates: list[str] = []
        if self.model_name:
            candidates.append(self.model_name)
        preferred = str(repo_root / "yolo26n-pose.pt")
        for c in [preferred] + sorted(local_hits):
            if c not in candidates:
                # only existing paths or known model names
                if Path(c).exists() or Path(c).name in CANDIDATE_MODELS:
                    candidates.append(c)
        for c in CANDIDATE_MODELS:
            if c not in candidates and Path(c).name not in [Path(x).name for x in candidates]:
                # only as a last resort (may imply a download)
                if Path(c).exists():
                    candidates.append(c)
        # when nothing is local, allow downloading the first candidate
        if not candidates:
            candidates = list(CANDIDATE_MODELS)
        # filter None
        candidates = [c for c in candidates if c]
        last_err = None
        for m in candidates:
            try:
                self._model = YOLO(m)
                self.model_name = m
                return
            except Exception as e:
                last_err = str(e)
                continue
        self._load_error = last_err or "No se pudo cargar modelo YOLO-Pose"
        raise RuntimeError(self._load_error)

    def _warmup_tracker(self) -> None:
        """Verifies the requested tracker with 1 black frame (fail-fast with fallback).

        Chain: requested -> bytetrack-dance -> built-in bytetrack -> default
        ultralytics (None). Warmup also triggers the ReID model download
        once at startup (not mid-video) and leaves the tracker
        clean via reset_track().
        """
        if self._model is None:
            return
        probe = np.zeros((320, 320, 3), dtype=np.uint8)
        chain: list[str | None] = [self.tracker, "bytetrack-dance", "bytetrack", None]
        seen: set[str] = set()
        last_err: str | None = None
        for name in chain:
            key = name or "__default__"
            if key in seen:
                continue
            seen.add(key)
            cfg = resolve_tracker_cfg(name) if name else None
            # no requested tracker (None): the ultralytics default always "works"
            if name is None:
                self.tracker_cfg = None
                self.tracker_device = self.device
                return
            try:
                self._model.track(probe, persist=True, tracker=cfg, conf=self.conf,
                                  iou=self.iou, verbose=False, max_det=self.max_persons,
                                  device=self.device, quantize=self.quantize)
                self.tracker_cfg = cfg
                self.tracker = name
                self.tracker_device = self.device
                try:
                    self.reset_track()
                except Exception:
                    pass
                return
            except Exception as e:
                last_err = str(e)[:300]
                continue
        # last resort: default without yaml (should not fail)
        self.tracker_cfg = None
        self.tracker_device = self.device
        self.tracker_error = last_err

    def _coco_to_pose(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray, w: int, h: int) -> Pose:
        """kpts_xy: (17,2) in pixels, kpts_conf: (17,) 0..1.

        Keeps every joint with conf >= min_kp_conf with
        visibility=conf (the normalizer decides with its own threshold).
        """
        pose = Pose()
        for coco_idx, sem_name in COCO_TO_SEMANTIC.items():
            if coco_idx >= len(kpts_xy):
                continue
            x_pix, y_pix = float(kpts_xy[coco_idx][0]), float(kpts_xy[coco_idx][1])
            conf = float(kpts_conf[coco_idx]) if kpts_conf is not None else 1.0
            if conf < self.min_kp_conf:
                continue
            # normalize 0..1 like mediapipe, clamp to frame
            x = x_pix / float(w) if w else 0.0
            y = y_pix / float(h) if h else 0.0
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            # z unavailable in YOLO, 0. visibility = YOLO conf.
            pose.joints[sem_name] = Joint(x=x, y=y, z=0.0, visibility=conf)
        return pose

    def detect(self, frame: np.ndarray) -> Pose:
        multi = self.detect_multi(frame)
        if not multi:
            return Pose()
        # return the largest area/conf pose (first)
        # sort by bbox area
        multi_sorted = sorted(multi, key=lambda x: x[0][2]*x[0][3], reverse=True)
        return multi_sorted[0][1]

    def _parse_result(self, r, w: int, h: int, with_ids: bool = False):
        """Parses one ultralytics result into a detection list.

        With with_ids=True returns [(track_id|None, bbox, pose)], else
        [(bbox, pose)]. Filters tiny boxes and empty poses.
        """
        out = []
        if r.boxes is None or r.keypoints is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else np.array(r.boxes.xyxy)
        ids = None
        if with_ids and getattr(r.boxes, 'id', None) is not None:
            try:
                ids = r.boxes.id.cpu().numpy().astype(int).tolist()
            except Exception:
                try:
                    ids = [int(v) for v in np.array(r.boxes.id).tolist()]
                except Exception:
                    ids = None
        try:
            kpts_xy = r.keypoints.xy.cpu().numpy() if hasattr(r.keypoints.xy, 'cpu') else np.array(r.keypoints.xy)
            if hasattr(r.keypoints, 'conf') and r.keypoints.conf is not None:
                kpts_conf = r.keypoints.conf.cpu().numpy() if hasattr(r.keypoints.conf, 'cpu') else np.array(r.keypoints.conf)
            else:
                kpts_conf = r.keypoints.data.cpu().numpy()[:,:,2] if hasattr(r.keypoints.data, 'cpu') else np.array(r.keypoints.data)[:,:,2]
        except Exception:
            return out
        if len(boxes) == 0 or len(kpts_xy) == 0:
            return out
        for i in range(min(len(boxes), len(kpts_xy), len(ids) if ids is not None else len(boxes))):
            x1, y1, x2, y2 = boxes[i]
            x, y, wb, hb = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            # filter tiny boxes (far dancers: 20x40 minimum).
            # also clip to frame to avoid degenerate bboxes.
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            wb = max(0, min(wb, w - x))
            hb = max(0, min(hb, h - y))
            if wb < self.min_box_w or hb < self.min_box_h:
                continue
            conf_i = kpts_conf[i] if i < len(kpts_conf) else None
            pose = self._coco_to_pose(kpts_xy[i], conf_i, w, h)
            if pose.is_empty() or len(pose.joints) < self.min_joints:
                continue
            if with_ids:
                tid = int(ids[i]) if ids is not None and i < len(ids) else None
                out.append((tid, (x, y, wb, hb), pose))
            else:
                out.append(((x, y, wb, hb), pose))
        return out

    def detect_multi(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], Pose]]:
        if self._model is None:
            return []
        h, w = frame.shape[:2]
        try:
            # verbose=False to avoid spamming
            results = self._model(frame, conf=self.conf, iou=self.iou, verbose=False,
                                  max_det=self.max_persons, device=self.device,
                                  quantize=self.quantize)
        except Exception:
            return []
        if not results or len(results) == 0:
            return []
        dets = self._parse_result(results[0], w, h, with_ids=False)
        # sort by x for a stable dancer_id assignment
        dets.sort(key=lambda x: x[0][0])
        # cap
        if len(dets) > self.max_persons:
            dets = dets[:self.max_persons]
        return dets

    def track_multi(self, frame: np.ndarray, imgsz: int | None = None) -> list[tuple[int | None, tuple[int, int, int, int], Pose]]:
        """Persistent tracking (BoT-SORT+ReID or ByteTrack): stable IDs across frames.

        Matches the documented pattern:
            model.track(frame, persist=True, tracker="botsort.yaml", conf=...)
        Must be called with sequential frames from the same source and the same
        backend instance. Returns [(track_id|None, bbox, pose)] UNsorted
        (tracker order is not spatial). `track_id` may be
        None on the first frames after a jump (tracker not initialized yet);
        the caller (TrackRegistry) links by motion+appearance+pose.
        Optional `imgsz` (e.g. 960 on ingest for far dancers).
        """
        if self._model is None:
            return []
        h, w = frame.shape[:2]
        try:
            kwargs = dict(persist=True, conf=self.conf, iou=self.iou,
                          verbose=False, max_det=self.max_persons,
                          device=self.device, quantize=self.quantize)
            if imgsz is not None:
                kwargs["imgsz"] = int(imgsz)
            if self.tracker_cfg is not None:
                kwargs["tracker"] = self.tracker_cfg
            results = self._model.track(frame, **kwargs)
        except Exception:
            return []
        if not results or len(results) == 0:
            return []
        dets = self._parse_result(results[0], w, h, with_ids=True)
        # keep by area (largest = real people)
        if len(dets) > self.max_persons:
            dets = sorted(dets, key=lambda t: t[1][2] * t[1][3], reverse=True)[:self.max_persons]
        return dets

    def reset_track(self) -> None:
        """Resets tracker state (IDs restart at 1). Call on video/session change."""
        try:
            predictor = getattr(self._model, 'predictor', None)
            trackers = getattr(predictor, 'trackers', None) if predictor is not None else None
            for t in (trackers or []):
                try:
                    t.reset()
                except Exception:
                    continue
        except Exception:
            pass

    def close(self) -> None:
        # ultralytics needs no explicit close
        pass
