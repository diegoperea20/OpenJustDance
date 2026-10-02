"""Extract choreography from a video: detect poses, normalize and generate song.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.normalize_engine import BodyNormalizer, PoseSmoother
from engine.pose_engine import create_backend
from engine.pose_engine.multi import (
    MultiBackendSet,
    StableSlotMapper,
    TrackRegistry,
    detect_multi_poses,
    get_multi_backends,
    track_multi_poses,
)
from engine.pose_engine.skeleton import Pose
from engine.tracking.tracker import MultiPersonTracker
from shared.dance_format.models import DancerPose, FramePose, Song


class CancelledError(Exception):
    """Extraction was cancelled by the user (via on_progress/should_cancel)."""


def _cancel_requested(on_progress_result=None, should_cancel=None) -> bool:
    """True if abort is needed: on_progress returned True or should_cancel() is True."""
    if on_progress_result is True:
        return True
    if should_cancel is not None:
        try:
            return bool(should_cancel())
        except Exception:
            return False
    return False


@dataclass
class ExtractOptions:
    video: Path
    title: str = ""
    artist: str = ""
    fps: Optional[float] = None  # Sampling FPS (defaults to the video's)
    max_width: int = 1280  # Max long side for downscale (supports vertical/horizontal)
    backend: str = "auto"
    start_time: float = 0.0
    end_time: Optional[float] = None
    rotation: int = 0  # 0, 90, 180, 270 - manual rotation (0=auto)
    multi: bool = False  # if True, extract up to 4 dancers per frame (legacy flag)
    multi_max: int = 4
    auto_multi: bool = True  # if True, auto-detect >1 person without needing a checkbox
    use_yolo: bool = True  # YOLO-Pose first (4 dancers in 1 forward); HOG+MP as fallback
    yolo_conf: float = 0.4
    yolo_iou: float = 0.6  # High NMS: do not merge overlapping dancer boxes
    estimator: str = "yolo"  # "yolo" = direct pure YOLO keypoints (recommended multi); "mp" = YOLO boxes+IDs + one MediaPipe per person
    tracker: str | None = "botsort-reid"  # BoT-SORT+ReID (anti-overlap), "bytetrack"/"bytetrack-dance", "botsort"
    device: str | None = "auto"  # "auto" = cuda if GPU present, else cpu; "cpu"/"cuda" force it
    preview_path: Optional[Path] = None  # if set, save annotated video in result.plot() style
    dump_poses_path: Optional[Path] = None  # if set, dump raw YOLO poses [{frame, persons:[...]}]
    debug_track_path: Optional[Path] = None  # if set, dump per-frame tracking decisions (jsonl)
    # --- Ingest with declared dancer count (1-4, trackingdancers style) ---
    num_dancers: int = 1  # declared by the user in ImportDialog/CLI; hard cap MAX_DANCERS
    yolo_model: Optional[str] = None  # None = auto (multi ingest pins yolo11m-pose.pt)
    face_id: bool = False  # True = mandatory InsightFace FaceID (multi); False = tracker only
    face_thresh: float = 0.38  # FaceID gallery cosine threshold
    face_check_every: int = 6  # periodic re-check per track_id
    face_det_size: int = 320  # face detector resolution over crop
    kp_thresh: float = 0.4  # minimum keypoint confidence for drawing/exporting
    smooth_alpha: float = 0.6  # EMA per persistent ID
    imgsz: int = 960  # YOLO ingest inference size
    poses_jsonl_path: Optional[Path] = None  # export trackingdancers-compatible {frame,id,box,kpts}


def _downscale(frame, max_width: int):
    """Downscale keeping aspect ratio. max_width is the long side (supports vertical)."""
    import cv2

    h, w = frame.shape[:2]
    longest = max(h, w)
    if max_width and longest > max_width:
        scale = max_width / float(longest)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def _apply_rotation(frame, rotation: int):
    import cv2

    rotation = int(rotation) % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _video_info(video: Path):
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = (count / fps) if fps else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, count, duration, width, height


def _get_or_create(store: dict, key: int, factory):
    inst = store.get(key)
    if inst is None:
        inst = factory()
        store[key] = inst
    return inst


def _normalize_and_pack(tracked: list, normalizers, smoothers, raw_smoothers,
                        vis_threshold: float = 0.4):
    """Normalize/smooth [(dancer_id, bbox, pose_global)] into list[DancerPose].

    vis_threshold: BodyNormalizer threshold per dancer. Pure YOLO uses 0.15
    (0.2-0.4 confidences under occlusions); live MediaPipe keeps 0.4.
    """
    dancers: list[DancerPose] = []
    for (slot, _bbox, pose_yolo) in tracked:
        if pose_yolo.is_empty():
            continue
        norm = _get_or_create(normalizers, slot, lambda: BodyNormalizer(visibility_threshold=vis_threshold))
        sm = _get_or_create(smoothers, slot, lambda: PoseSmoother(alpha=0.5))
        rsm = _get_or_create(raw_smoothers, slot, lambda: PoseSmoother(alpha=0.5))
        normalized = norm.normalize(pose_yolo)
        if normalized.is_empty():
            continue
        smoothed = sm.smooth(normalized)
        raw_pose = rsm.smooth(pose_yolo)
        dancers.append(DancerPose(dancer_id=slot, joints=smoothed.to_dict(), raw=raw_pose.to_dict()))
    dancers.sort(key=lambda d: d.dancer_id)
    return dancers


def _extract_single_frame_multi(frame, backends: MultiBackendSet, tracker, slot_mapper: StableSlotMapper,
                                registry, normalizers, smoothers, raw_smoothers, max_persons=4,
                                vis_threshold: float = 0.4):
    """Persistent-tracking multi helper (see `engine.pose_engine.multi`).

    1. `track_multi_poses` (YOLO ByteTrack + TrackRegistry): stable IDs across
       frames, permanent dancer_id per person. Poses already come in
       GLOBAL 0..1 coords.
    2. Fallback `detect_multi_poses` (sensitive HOG + MediaPipe) only if the track
       sees nobody in this frame (total occlusion / YOLO missing).
    Returns list[DancerPose] (empty if no detection).
    """
    if backends.yolo_backend is not None and registry is not None:
        tracked = track_multi_poses(frame, backends, registry, max_persons=max_persons)
        if tracked:
            return _normalize_and_pack(tracked, normalizers, smoothers, raw_smoothers, vis_threshold)
    # Fallback without tracking (single isolated frame): order by x as slots.
    dets = detect_multi_poses(frame, backends, tracker=tracker, max_persons=max_persons)
    if not dets:
        return []
    boxes = [b for b, _ in dets]
    slot_of = slot_mapper.assign(boxes)
    bbox_to_slot = {b: s for s, b in slot_of.items()}
    tracked = [(bbox_to_slot[b], b, p) for b, p in dets if b in bbox_to_slot]
    return _normalize_and_pack(tracked, normalizers, smoothers, raw_smoothers, vis_threshold)


COCO_TO_SEMANTIC_17 = {
    0: "head", 5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}
COCO_SKELETON_17 = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _coco_kpts_to_pose(kpts: "object", w: int, h: int, min_conf: float = 0.15) -> "Pose":
    """Convert (17,3) pixels [x,y,conf] to semantic Pose 0..1."""
    import numpy as _np

    from engine.pose_engine.skeleton import Joint as _Joint
    from engine.pose_engine.skeleton import Pose as _Pose

    pose = _Pose()
    arr = _np.asarray(kpts, dtype=float)
    for coco_idx, sem_name in COCO_TO_SEMANTIC_17.items():
        if coco_idx >= len(arr):
            continue
        x_pix, y_pix, conf = float(arr[coco_idx][0]), float(arr[coco_idx][1]), float(arr[coco_idx][2])
        if conf < min_conf:
            continue
        x = max(0.0, min(1.0, x_pix / float(w))) if w else 0.0
        y = max(0.0, min(1.0, y_pix / float(h))) if h else 0.0
        pose.joints[sem_name] = _Joint(x=x, y=y, z=0.0, visibility=conf)
    return pose


def _iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def _kpts_for_track(t, box, det_boxes, kpts_all):
    """Track keypoints: by det_ind if valid, else by best IoU>0.5."""
    import numpy as _np

    if len(kpts_all) == 0 or len(det_boxes) == 0:
        return None
    det_ind = int(t[7]) if len(t) > 7 else -1
    if 0 <= det_ind < len(kpts_all):
        return _np.asarray(kpts_all[det_ind], dtype=float)
    ious = [_iou_xyxy(box, d) for d in det_boxes]
    j = int(_np.argmax(ious))
    return _np.asarray(kpts_all[j], dtype=float) if ious[j] > 0.5 else None


def _extract_declared_single(options: ExtractOptions, on_progress=None,
                             should_cancel=None) -> Song:
    """Declared mono ingest: MediaPipe full-frame only, no YOLO or tracker.

    Cooperative cancellation: if ``on_progress`` returns True or
    ``should_cancel()`` returns True, resources are released and
    :class:`CancelledError` is raised without writing partial results.
    """
    import cv2 as _cv2

    from engine.normalize_engine import BodyNormalizer as _Norm
    from engine.normalize_engine import PoseSmoother as _Sm
    from engine.pose_engine import create_backend as _create_backend

    video = Path(options.video)
    video_fps, frame_count, duration, _, _ = _video_info(video)
    target_fps = options.fps or min(video_fps, 30.0)
    step = max(1, round(video_fps / target_fps))
    effective_fps = video_fps / step
    start = options.start_time
    end = options.end_time if options.end_time is not None else duration
    duration = max(0.0, end - start)
    try:
        backend = _create_backend("mediapipe-cpu")
    except Exception as exc:
        raise RuntimeError(f"Backend MediaPipe no disponible: {exc}") from exc
    normalizer = _Norm()
    smoother = _Sm(alpha=0.5)
    raw_smoother = _Sm(alpha=0.5)
    cap = _cv2.VideoCapture(str(video))
    if not cap.isOpened():
        try:
            backend.close()
        except Exception:
            pass
        raise ValueError(f"No se pudo abrir el video: {video}")
    preview_writer = None
    if options.preview_path is not None:
        Path(options.preview_path).parent.mkdir(parents=True, exist_ok=True)
        preview_writer = _cv2.VideoWriter(str(options.preview_path),
                                          _cv2.VideoWriter_fourcc(*"mp4v"),
                                          max(1.0, float(effective_fps)), (640, 480))
        preview_writer = None  # reopened with real size on first frame
    poses: list[FramePose] = []
    idx = 0
    frame_no = 0
    total_frames = int(frame_count)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / video_fps
            idx += 1
            if t < start:
                continue
            if t > end:
                break
            if (idx - 1) % step != 0:
                continue
            if _cancel_requested(should_cancel=should_cancel):
                raise CancelledError("Extracción cancelada por el usuario.")
            if options.rotation:
                frame = _apply_rotation(frame, options.rotation)
            small = _downscale(frame, options.max_width)
            t_round = round(t - start, 3)
            frame_no += 1
            raw = backend.detect(small)
            if raw.is_empty():
                poses.append(FramePose(time=t_round, joints={}, raw={}))
            else:
                normalized = normalizer.normalize(raw)
                if normalized.is_empty():
                    poses.append(FramePose(time=t_round, joints={}, raw={}))
                else:
                    smoothed = smoother.smooth(normalized)
                    raw_pose = raw_smoother.smooth(raw)
                    poses.append(FramePose(time=t_round, joints=smoothed.to_dict(),
                                           raw=raw_pose.to_dict()))
            if options.preview_path is not None and preview_writer is None:
                try:
                    h0, w0 = small.shape[:2]
                    preview_writer = _cv2.VideoWriter(
                        str(options.preview_path), _cv2.VideoWriter_fourcc(*"mp4v"),
                        max(1.0, float(effective_fps)), (w0, h0))
                except Exception:
                    preview_writer = None
            if options.preview_path is not None and preview_writer is not None:
                try:
                    from engine.pose_engine.skeleton import draw_pose as _draw
                    annot = small.copy()
                    if not raw.is_empty():
                        _draw(annot, raw, color=(0, 255, 0), thickness=3)
                    _cv2.putText(annot, f"Frame: {frame_no}", (10, annot.shape[0] - 12),
                                  _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    preview_writer.write(annot)
                except Exception:
                    pass
            if on_progress and total_frames > 0:
                _cancel = on_progress(idx, total_frames)
                if _cancel_requested(_cancel, should_cancel):
                    raise CancelledError("Extracción cancelada por el usuario.")
    finally:
        cap.release()
        if preview_writer is not None:
            try:
                preview_writer.release()
            except Exception:
                pass
        try:
            backend.close()
        except Exception:
            pass
    title = options.title or video.stem
    return Song(title=title, artist=options.artist or "Desconocido",
                fps=round(effective_fps, 3), duration=round(duration, 3),
                format="openjustdance-1", poses=poses, num_dancers=1)


def _extract_declared_multi(options: ExtractOptions, on_progress=None,
                            should_cancel=None) -> Song:
    """Declared multi ingest in trackingdancers/projectv.py style.

    Fixed: yolo11m-pose.pt (Ultralytics internal tracking, no boxmot package)
    + mandatory InsightFace FaceID + EMA per persistent ID + export
    .poses.jsonl {frame,id,box,kpts COCO-17 in pixels}.
    Returns Song in openjustdance-2 format with num_dancers=stable observed.

    Cooperative cancellation: see :class:`CancelledError`.
    """
    import json as _json

    import cv2 as _cv2
    import numpy as _np

    from engine.normalize_engine import BodyNormalizer as _Norm
    from engine.normalize_engine import PoseSmoother as _Sm
    from engine.pose_engine.backends.yolo_pose import YoloPoseBackend
    from engine.pose_engine.faceid import FaceGallery
    from engine.pose_engine.skeleton import Joint as _Joint
    from engine.pose_engine.skeleton import Pose as _Pose
    from engine.pose_engine.skeleton import dancer_color_bgr as _dancer_color

    num = max(1, min(4, int(getattr(options, "num_dancers", 1) or 1)))
    det_conf = float(getattr(options, "yolo_conf", 0.25) or 0.25)
    imgsz = int(getattr(options, "imgsz", 960) or 960)
    kp_thresh = float(getattr(options, "kp_thresh", 0.4) or 0.4)
    smooth_alpha = float(getattr(options, "smooth_alpha", 0.6) or 0.6)
    yolo_iou = float(getattr(options, "yolo_iou", 0.6) or 0.6)
    face_thresh = float(getattr(options, "face_thresh", 0.38) or 0.38)
    face_every = int(getattr(options, "face_check_every", 6) or 6)
    face_det = int(getattr(options, "face_det_size", 320) or 320)

    # Mandatory FaceID: fails here with install message if missing.
    gallery = FaceGallery(max_dancers=num, match_thresh=face_thresh,
                          check_every=face_every, det_size=(face_det, face_det))

    # Ultralytics internal tracking (repo botsort-reid.yaml): avoids the
    # direct `boxmot` package (v25 breaks with `lap` without lapjv in this env).
    model_name = getattr(options, "yolo_model", None) or "yolo11m-pose.pt"
    backend = YoloPoseBackend(model=model_name, conf=det_conf, iou=yolo_iou,
                              max_persons=num, min_kp_conf=0.15, min_joints=4,
                              tracker="botsort-reid",
                              device=getattr(options, "device", "auto"))
    backend.reset_track()

    video = Path(options.video)
    video_fps, frame_count, duration, _, _ = _video_info(video)
    target_fps = options.fps or min(video_fps, 30.0)
    step = max(1, round(video_fps / target_fps))
    effective_fps = video_fps / step
    start = options.start_time
    end = options.end_time if options.end_time is not None else duration
    duration = max(0.0, end - start)

    cap = _cv2.VideoCapture(str(video))
    if not cap.isOpened():
        try:
            backend.close()
        except Exception:
            pass
        raise ValueError(f"No se pudo abrir el video: {video}")
    if options.preview_path is not None:
        Path(options.preview_path).parent.mkdir(parents=True, exist_ok=True)
    preview_writer = None  # opened with the real size of the 1st sampled frame
    sem_smooth: dict = {}  # pid -> {joint: (x, y)} EMA in 0..1 coords
    norms: dict = {}
    sms: dict = {}
    rsms: dict = {}

    def smooth_pose(pid, pose: _Pose) -> _Pose:
        """EMA per persistent ID over 0..1 coords (visible joints only)."""
        prev = sem_smooth.get(pid)
        if prev is None:
            sem_smooth[pid] = {n: (float(j.x), float(j.y)) for n, j in pose.joints.items()}
            return pose
        out = _Pose()
        for n, j in pose.joints.items():
            if n in prev and float(j.visibility) >= kp_thresh:
                px, py = prev[n]
                nx = smooth_alpha * float(j.x) + (1 - smooth_alpha) * px
                ny = smooth_alpha * float(j.y) + (1 - smooth_alpha) * py
                prev[n] = (nx, ny)
                out.joints[n] = _Joint(x=nx, y=ny, z=j.z, visibility=j.visibility)
            else:
                prev[n] = (float(j.x), float(j.y))
                out.joints[n] = _Joint(x=j.x, y=j.y, z=j.z, visibility=j.visibility)
        return out

    def coco17_pixels(pose: _Pose, w: int, h: int) -> list:
        """Rebuild COCO-17 in pixels from the 0..1 semantic pose."""
        out = []
        for i in range(17):
            sem = COCO_TO_SEMANTIC_17.get(i)
            j = pose.joints.get(sem) if sem else None
            if j is None:
                out.append([0.0, 0.0, 0.0])
            else:
                out.append([round(float(j.x) * w, 1), round(float(j.y) * h, 1),
                            round(float(j.visibility), 2)])
        return out

    poses: list[FramePose] = []
    jsonl_lines: list = []
    frame_no = 0
    idx = 0
    total_frames = int(frame_count)
    pid_to_slot: dict = {}
    neg_id = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / video_fps
            idx += 1
            if t < start:
                continue
            if t > end:
                break
            if (idx - 1) % step != 0:
                continue
            if _cancel_requested(should_cancel=should_cancel):
                raise CancelledError("Extracción cancelada por el usuario.")
            if options.rotation:
                frame = _apply_rotation(frame, options.rotation)
            H, W = frame.shape[:2]
            if options.preview_path is not None and preview_writer is None:
                preview_writer = _cv2.VideoWriter(
                    str(options.preview_path), _cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, float(effective_fps)), (W, H))
            t_round = round(t - start, 3)
            frame_no += 1
            try:
                dets = backend.track_multi(frame, imgsz=imgsz)
            except Exception:
                dets = []
            dancers: list[DancerPose] = []
            if dets:
                for (tid, bbox, pose) in dets:
                    if pose.is_empty():
                        continue
                    if tid is None:
                        tid = neg_id
                        neg_id -= 1
                    x, y, wb, hb = bbox
                    box = (float(x), float(y), float(x + wb), float(y + hb))
                    head = pose.joints.get("head")
                    face_visible = bool(head is not None and float(head.visibility) >= 0.5)
                    pid = gallery.resolve_identity(tid, box, frame, frame_no, face_visible)
                    if isinstance(pid, str):  # tmp_ without face yet: does not pollute gallery
                        continue
                    pose_s = smooth_pose(pid, pose)
                    if pid not in pid_to_slot:
                        if len(pid_to_slot) >= num:
                            continue
                        pid_to_slot[pid] = len(pid_to_slot)
                    slot = pid_to_slot[pid]
                    norm = norms.get(slot)
                    if norm is None:
                        norm = _Norm(visibility_threshold=0.15)
                        norms[slot] = norm
                    sm = sms.get(slot)
                    if sm is None:
                        sm = _Sm(alpha=0.5)
                        sms[slot] = sm
                    rsm = rsms.get(slot)
                    if rsm is None:
                        rsm = _Sm(alpha=0.5)
                        rsms[slot] = rsm
                    normalized = norm.normalize(pose_s)
                    if normalized.is_empty():
                        continue
                    smoothed = sm.smooth(normalized)
                    raw_pose = rsm.smooth(pose_s)
                    dancers.append(DancerPose(dancer_id=slot, joints=smoothed.to_dict(),
                                              raw=raw_pose.to_dict()))
                    jsonl_lines.append({"frame": frame_no, "id": int(pid),
                                        "box": [round(v, 1) for v in box],
                                        "kpts": coco17_pixels(pose_s, W, H)})
                    if preview_writer is not None:
                        # Color ALWAYS by dancer_id (single process palette).
                        col = _dancer_color(slot)
                        _cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                       (int(box[2]), int(box[3])), col, 2)
                        _cv2.putText(frame, f"D{slot + 1}|ID{pid}", (int(box[0]), int(box[1]) - 8),
                                      _cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                        for a, b in COCO_SKELETON_17:
                            sa, sb = COCO_TO_SEMANTIC_17.get(a), COCO_TO_SEMANTIC_17.get(b)
                            if sa is None or sb is None:
                                continue
                            ja, jb = pose_s.joints.get(sa), pose_s.joints.get(sb)
                            if ja is None or jb is None:
                                continue
                            if float(ja.visibility) < kp_thresh or float(jb.visibility) < kp_thresh:
                                continue
                            _cv2.line(frame, (int(ja.x * W), int(ja.y * H)),
                                      (int(jb.x * W), int(jb.y * H)), col, 2, _cv2.LINE_AA)
            dancers.sort(key=lambda d: d.dancer_id)
            if preview_writer is not None:
                preview_writer.write(frame)
            if not dancers:
                poses.append(FramePose(time=t_round, joints={}, raw={}, dancers=[]))
            else:
                first = dancers[0]
                poses.append(FramePose(time=t_round, joints=first.joints, raw=first.raw,
                                       dancers=dancers))
            if on_progress and total_frames > 0:
                _cancel = on_progress(idx, total_frames)
                if _cancel_requested(_cancel, should_cancel):
                    raise CancelledError("Extracción cancelada por el usuario.")
    finally:
        cap.release()
        if preview_writer is not None:
            try:
                preview_writer.release()
            except Exception:
                pass
        try:
            backend.close()
        except Exception:
            pass
    if options.poses_jsonl_path is not None:
        try:
            Path(options.poses_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
            with open(options.poses_jsonl_path, "w", encoding="utf-8") as _fh:
                for _ln in jsonl_lines:
                    _fh.write(_json.dumps(_ln) + "\n")
        except Exception:
            pass
    observed = len(gallery.stable_ids())
    title = options.title or video.stem
    # Same 1-4 pipeline: with 1 stable dancer save single format
    # (DancerSelect auto-skip) while keeping the dancers list.
    fmt = "openjustdance-2" if max(1, observed or 1) > 1 else "openjustdance-1"
    return Song(title=title, artist=options.artist or "Desconocido",
                fps=round(effective_fps, 3), duration=round(duration, 3),
                format=fmt, poses=poses,
                num_dancers=max(1, observed or 1))


def extract(options: ExtractOptions, on_progress=None, should_cancel=None) -> Song:
    """Process the video and return a Song with normalized poses.

    If options.multi=True or auto_multi detects >1 person, extract up to multi_max dancers
    per frame using HOG+tracker+per-crop MediaPipe. Without checkbox: auto_multi avoids misses.

    Cooperative cancellation: if ``on_progress`` returns True or
    ``should_cancel()`` returns True, :class:`CancelledError` is raised.
    """
    import cv2
    # --- Declared path: dancer count wins (robust, trackingdancers style) ---
    # 1 or more (up to 4): ALL go through the same projectv.py process
    # (YOLO11m + tracking + FaceID). No separate MediaPipe path anymore.
    try:
        _nd = int(getattr(options, "num_dancers", 1) or 1)
    except Exception:
        _nd = 1
    if 1 <= _nd <= 4:
        _mm = getattr(options, "multi_max", 4) or 4
        options.multi_max = max(_nd, min(4, int(_mm)))
        return _extract_declared_multi(options, on_progress=on_progress,
                                       should_cancel=should_cancel)

    video = Path(options.video)
    video_fps, frame_count, duration, _, _ = _video_info(video)

    target_fps = options.fps or min(video_fps, 30.0)
    step = max(1, round(video_fps / target_fps))
    effective_fps = video_fps / step

    start = options.start_time
    end = options.end_time if options.end_time is not None else duration
    duration = max(0.0, end - start)

    # Per-person backend (always MediaPipe unless the user asks for another non-YOLO one).
    # Pure YOLO-Pose goes separately as the multi primary (1 forward, N people,
    # as in the user example: model(frame, conf) -> keypoints per person).
    estimator = getattr(options, "estimator", "yolo") or "yolo"
    # Normalizer threshold: YOLO gives visibility=conf (0.2-0.4 under occlusions),
    # needs 0.15; live MediaPipe keeps 0.4.
    vis_threshold = 0.15 if estimator == "yolo" else 0.4
    _mp_name = options.backend if options.backend not in (None, "", "auto", "yolo-pose", "yolo26n-pose", "yolov8n-pose", "yolo26n") else "mediapipe-cpu"
    try:
        backend = create_backend(_mp_name)
    except Exception:
        backend = create_backend("mediapipe-cpu")
    normalizer = BodyNormalizer()
    smoother = PoseSmoother(alpha=0.5)
    raw_smoother = PoseSmoother(alpha=0.5)
    # YOLO-first: loaded ONCE at startup (not gated by HOG). On failure
    # (no torch/model), continue with sensitive HOG + MediaPipe.
    backends: MultiBackendSet = get_multi_backends(
        mp_backend=backend,
        yolo=bool(options.use_yolo),
        max_persons=options.multi_max,
        yolo_conf=options.yolo_conf,
        yolo_iou=options.yolo_iou,
        estimator=getattr(options, "estimator", "yolo"),
        tracker=getattr(options, "tracker", "botsort-reid"),
        device=getattr(options, "device", "auto"),
    )

    # multi always on when auto_multi or --multi (YOLO decides how many there are;
    # no longer disabled after 60 frames: cost is 1 YOLO forward per frame).
    use_multi = bool(options.multi)
    auto_active = bool(options.auto_multi and not use_multi)
    do_multi_global = bool(use_multi or auto_active or options.use_yolo)
    tracker = MultiPersonTracker(iou_thresh=0.15, max_lost=18) if do_multi_global else None
    slot_mapper = StableSlotMapper(max_slots=options.multi_max)
    registry = TrackRegistry(max_slots=options.multi_max)
    # Clean YOLO tracker state (IDs from 1 for this video) and registry.
    registry.reset()
    slot_mapper.reset()
    try:
        if backends.yolo_backend is not None:
            backends.yolo_backend.reset_track()
    except Exception:
        pass
    multi_norms: dict[int, BodyNormalizer] = {}
    multi_sms: dict[int, PoseSmoother] = {}
    multi_rsms: dict[int, PoseSmoother] = {}
    multi_hits = 0
    yolo_hits = 0
    # Debug dump in user-example style: [{frame, persons:[{person, keypoints:[{x,y,confidence}]}]}]
    dump_frames: list = []
    debug_lines: list = []
    frame_no = 0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        backend.close()
        raise ValueError(f"No se pudo abrir el video: {video}")

    # Optional annotated preview (example result.plot() style).
    preview_writer = None
    preview_size = None
    try:
        if options.preview_path is not None:
            Path(options.preview_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # opened on the first sampled frame (known size after downscale)
            preview_size = (fourcc, effective_fps)
    except Exception:
        preview_writer = None

    poses: list[FramePose] = []
    total_frames = int(frame_count)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / video_fps
            idx += 1
            if t < start:
                continue
            if t > end:
                break
            if (idx - 1) % step != 0:
                continue
            if _cancel_requested(should_cancel=should_cancel):
                raise CancelledError("Extracción cancelada por el usuario.")

            if options.rotation:
                frame = _apply_rotation(frame, options.rotation)
            small = _downscale(frame, options.max_width)

            t_round = round(t - start, 3)
            frame_no += 1

            # YOLO-first multi mode on each sampled frame.
            do_multi = do_multi_global
            if do_multi:
                if tracker is None:
                    tracker = MultiPersonTracker(iou_thresh=0.15, max_lost=18)
                dancers = _extract_single_frame_multi(small, backends, tracker, slot_mapper, registry, multi_norms, multi_sms, multi_rsms, max_persons=options.multi_max, vis_threshold=vis_threshold)
                # Registry tracking info (tid+bbox per dancer, via, events).
                try:
                    dbg = registry.debug_info or {}
                except Exception:
                    dbg = {}
                tid_of = {int(md): (mt, mb) for md, mt, mb in dbg.get("matches", [])}
                # Raw dump for debugging (global 0..1 coords + visibility as confidence).
                if options.dump_poses_path is not None:
                    persons = []
                    for d in dancers:
                        kps = []
                        for _name, arr in d.raw.items():
                            x, y = float(arr[0]), float(arr[1])
                            conf = float(arr[3]) if len(arr) > 3 else 1.0
                            kps.append({"joint": _name, "x": x, "y": y, "confidence": conf})
                        _tid, _bb = tid_of.get(int(d.dancer_id), (None, None))
                        persons.append({"person": int(d.dancer_id) + 1, "track_id": _tid,
                                        "bbox": _bb, "keypoints": kps})
                    dump_frames.append({"frame": frame_no, "persons": persons})
                # Tracking decision log (jsonl): tids, via, events.
                if options.debug_track_path is not None:
                    debug_lines.append({"frame": frame_no, "t": t_round,
                                        "via": dbg.get("via"),
                                        "matches": [{"dancer": int(md), "tid": mt, "bbox": mb}
                                                    for md, mt, mb in dbg.get("matches", [])],
                                        "events": list(dbg.get("events", []))})
                # Annotated preview: draw each dancer with its skeleton.
                if options.preview_path is not None:
                    try:
                        from engine.pose_engine.skeleton import Pose as _Pose, Joint as _Joint, draw_pose as _draw
                        if preview_writer is None:
                            h0, w0 = small.shape[:2]
                            preview_writer = cv2.VideoWriter(str(options.preview_path), preview_size[0], max(1.0, float(effective_fps)), (w0, h0))
                        annot = small.copy()
                        palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                        for di, d in enumerate(dancers):
                            # Color and index ALWAYS by dancer_id: with 3 visible
                            # D4 stays yellow (it used to take D3's red).
                            _col = palette[int(d.dancer_id) % len(palette)]
                            _p = _Pose.from_dict(d.raw)
                            _draw(annot, _p, color=_col, thickness=3)
                            _tid, _bb = tid_of.get(int(d.dancer_id), (None, None))
                            _tag = f"D{int(d.dancer_id) + 1}·T{_tid if _tid is not None else '?'}"
                            cv2.putText(annot, _tag, (10, 30 + int(d.dancer_id) * 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, _col, 2)
                        cv2.putText(annot, f"Frame: {frame_no}", (10, annot.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        preview_writer.write(annot)
                    except Exception:
                        pass
                if len(dancers) >= 2:
                    multi_hits += 1
                if backends.yolo_backend is not None and len(dancers) >= 1:
                    yolo_hits += 1
                if not dancers:
                    poses.append(FramePose(time=t_round, joints={}, raw={}, dancers=[]))
                else:
                    first = dancers[0]
                    poses.append(FramePose(time=t_round, joints=first.joints, raw=first.raw, dancers=dancers))
            else:
                raw = backend.detect(small)
                if raw.is_empty():
                    poses.append(FramePose(time=t_round, joints={}, raw={}))
                else:
                    normalized = normalizer.normalize(raw)
                    if normalized.is_empty():
                        poses.append(FramePose(time=t_round, joints={}, raw={}))
                    else:
                        smoothed = smoother.smooth(normalized)
                        raw_pose = raw_smoother.smooth(raw)
                        poses.append(
                            FramePose(
                                time=t_round,
                                joints=smoothed.to_dict(),
                                raw=raw_pose.to_dict(),
                            )
                        )

            if on_progress and total_frames > 0:
                _cancel = on_progress(idx, total_frames)
                if _cancel_requested(_cancel, should_cancel):
                    raise CancelledError("Extracción cancelada por el usuario.")
    finally:
        cap.release()
        try:
            if preview_writer is not None:
                preview_writer.release()
        except Exception:
            pass
        try:
            backend.close()
        except Exception:
            pass
        if backends.yolo_backend is not None:
            try:
                backends.yolo_backend.close()
            except Exception:
                pass
        if options.dump_poses_path is not None:
            try:
                import json as _json
                Path(options.dump_poses_path).parent.mkdir(parents=True, exist_ok=True)
                Path(options.dump_poses_path).write_text(_json.dumps(dump_frames, indent=2), encoding="utf-8")
            except Exception:
                pass
        if options.debug_track_path is not None:
            try:
                import json as _json2
                Path(options.debug_track_path).parent.mkdir(parents=True, exist_ok=True)
                with open(options.debug_track_path, "w", encoding="utf-8") as _fh:
                    for _ln in debug_lines:
                        _fh.write(_json2.dumps(_ln) + "\n")
            except Exception:
                pass

    title = options.title or video.stem
    # robust num_dancers inference: max alone is not enough (mono false positives give 3), use mode and threshold
    from collections import Counter
    cnt = Counter()
    for p in poses:
        if p.dancers:
            cnt[len(p.dancers)] += 1
        elif p.joints:
            cnt[1] += 1
        else:
            cnt[0] += 1
    # total with pose
    total_pose = sum(v for k,v in cnt.items() if k>0)
    if total_pose == 0:
        nd = 1
    else:
        # most frequent
        most_common_k, most_common_c = cnt.most_common(1)[0]
        if most_common_k == 0:
            # find next non-zero
            for k,c in cnt.most_common():
                if k>0:
                    most_common_k, most_common_c = k,c
                    break
        # if the most frequent is 1 and holds the majority, it is mono even with false 2-3 spikes
        # for aespavideo: 4:31, 3:23, 2:6 -> most 4 (31) >15% and > any other, 4 is picked
        # for videotest1short: 1:37, 2:19, 3:4 -> most 1 (37) -> mono
        if most_common_k == 1 and most_common_c > total_pose*0.4:
            nd = 1
        elif most_common_k >= 2 and most_common_c >= total_pose*0.15:
            nd = most_common_k
        else:
            # fallback to max but with threshold: only if it appears in at least 20% of frames
            max_k = max((k for k in cnt if k>0), default=1)
            if cnt[max_k] >= total_pose*0.2:
                nd = max_k
            else:
                nd = 1
        # if multi was never seen (multi_hits==0) force 1 (truly mono video)
        if multi_hits == 0:
            nd = 1
    nd = min(max(1, nd), options.multi_max)
    # If mono (nd==1) but some frames hold false 2-3 dancers, collapse to legacy single to avoid confusing the UI/"overlay"
    if nd == 1:
        for p in poses:
            if p.dancers:
                # keep only the first dancer as legacy
                first = p.dancers[0] if p.dancers else None
                if first is not None:
                    p.joints = dict(first.joints)
                    p.raw = dict(first.raw)
                p.dancers = []
    fmt = "openjustdance-2" if nd > 1 else "openjustdance-1"
    return Song(
        title=title,
        artist=options.artist or "Desconocido",
        fps=round(effective_fps, 3),
        duration=round(duration, 3),
        format=fmt,
        poses=poses,
        num_dancers=nd,
    )


def extract_audio(video: Path, out_mp3: Path, ffmpeg: str = "ffmpeg") -> None:
    """Extract video audio to MP3 using ffmpeg."""
    import subprocess

    out_mp3 = Path(out_mp3)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(out_mp3),
        ],
        check=True,
        capture_output=True,
    )


def make_cover(video: Path, out_png: Path, rotation: int = 0) -> bool:
    """Save the first video frame as PNG cover."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    if rotation:
        frame = _apply_rotation(frame, rotation)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)
    return True
