"""Camera capture threads to avoid blocking the UI (QThread)."""

from __future__ import annotations

import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from engine.normalize_engine import BodyNormalizer, PoseSmoother
from engine.pose_engine import Camera, create_backend, draw_pose
from engine.pose_engine.multi import (
    StableSlotMapper,
    TrackRegistry,
    detect_multi_poses,
    get_multi_backends,
    track_multi_poses,
)
from engine.pose_engine.skeleton import Pose
from engine.score_engine import compare_space_prescale, to_compare_space


def create_live_yolo_backend(device: str | None = "auto"):
    """Creates the live YOLO backend with cuda -> cpu cascade.

    `device`: "auto" (CUDA when a GPU is present, else CPU), "cuda" or "cpu".
    Returns (backend | None, warning | None): when the requested device resolves
    to CUDA but inference underperforms (no usable CUDA, OOM, broken EP),
    it retries on CPU before giving up. The warning describes the fallback.
    """
    from engine.pose_engine import create_backend as _create_yolo

    params = dict(model="yolo11n-pose.pt", max_persons=4, conf=0.4,
                  iou=0.6, tracker="botsort-reid")
    try:
        backend = _create_yolo("yolo-pose", device=device, **params)
    except Exception as exc:
        return None, f"no se pudo crear YOLO ({exc})"
    if backend is None:
        return None, "backend YOLO no disponible"
    try:
        backend.health_check()
        return backend, None
    except Exception as exc:
        resolved = str(getattr(backend, "device", "") or "")
        try:
            backend.close()
        except Exception:
            pass
        if resolved.strip().lower().startswith("cuda"):
            try:
                backend_cpu = _create_yolo("yolo-pose", device="cpu", **params)
                backend_cpu.health_check()
                return backend_cpu, (
                    "CUDA no rindió "
                    f"({str(exc)[:120]}); usando CPU")
            except Exception as exc2:
                return None, f"YOLO sin CUDA ni CPU ({str(exc2)[:120]})"
        return None, f"YOLO no rinde en {resolved or device} ({str(exc)[:120]})"


def live_backend_label(backend) -> str:
    """'model · device' label for display (e.g. 'yolo11n-pose · cuda')."""
    try:
        name = getattr(backend, "model_name", None) or "yolo11n-pose"
        if isinstance(name, str) and name.endswith(".pt"):
            name = name[:-3]
        device = getattr(backend, "device", "") or ""
        return f"{name} · {device}" if device else str(name)
    except Exception:
        return "yolo11n-pose"


def open_camera_with_fallback(index: int, width: int, height: int):
    """Opens camera `index`; when it fails and index is not 0, tries 0.

    Returns (camera | None, warning | None, error | None):
    - direct success: (camera, None, None)
    - fallback success: (camera0, warning_msg, None)
    - total failure: (None, None, error_msg)
    """
    camera = Camera(index, width, height)
    if camera.open():
        return camera, None, None
    if index != 0:
        try:
            camera.release()
        except Exception:
            pass
        fallback = Camera(0, width, height)
        if fallback.open():
            return fallback, f"La camara {index} no abrio; usando camara 0.", None
        return None, None, fallback.last_error or "No se pudo abrir la camara"
    return None, None, camera.last_error or "No se pudo abrir la camara"


class TrackedPerson:
    """Tracked person with poses."""
    def __init__(self, track_id: int, bbox, pose: Pose, raw: Pose, norm_pose: Pose):
        self.track_id = track_id
        self.bbox = bbox  # x,y,w,h
        self.pose = pose  # raw alias (image 0..1)
        self.raw = raw
        self.norm_pose = norm_pose  # normalized
        self.timestamp = 0.0


class SharedState:
    """Thread-safe shared state between the camera thread and the UI.

    Keeps legacy compat: pose/raw_pose (first track) + multi tracks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pose = None  # Normalized pose of track 0 (compat)
        self.raw_pose = None  # Raw pose of track 0
        self.timestamp = 0.0
        self.brightness = 0.0
        self.fps = 0.0
        self.tracks: dict[int, TrackedPerson] = {}  # multi
        self._bboxes: dict[int, tuple[int, int, int, int]] = {}
        self.frame_size: tuple[int, int] | None = None  # (w, h) of the processed frame
        self.reference_aspect: float | None = None  # Reference video W/H (set by GameView)

    def update(self, pose, raw_pose, timestamp, brightness) -> None:
        with self._lock:
            self.pose = pose
            self.raw_pose = raw_pose
            self.timestamp = timestamp
            self.brightness = brightness

    def update_multi(self, tracks: dict[int, TrackedPerson], timestamp, brightness) -> None:
        with self._lock:
            self.tracks = dict(tracks)
            # compat: first track as legacy
            if tracks:
                first = sorted(tracks.items(), key=lambda kv: kv[0])[0][1]
                self.pose = first.norm_pose
                self.raw_pose = first.raw
                self.timestamp = timestamp
                self.brightness = brightness
            else:
                self.pose = None
                self.raw_pose = None
                self.timestamp = timestamp
                self.brightness = brightness

    def snapshot(self):
        with self._lock:
            return self.pose, self.raw_pose, self.timestamp, self.brightness

    def snapshot_multi(self):
        with self._lock:
            return dict(self.tracks), self.timestamp, self.brightness

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self.fps = fps

    def set_frame_size(self, w: int, h: int) -> None:
        """Frame size over which the pose was estimated (0..1 coords)."""
        try:
            w, h = int(w), int(h)
            if w > 0 and h > 0:
                with self._lock:
                    self.frame_size = (w, h)
        except Exception:
            pass

    def frame_aspect(self) -> float | None:
        """Live frame W/H aspect (None when still unknown)."""
        with self._lock:
            fs = self.frame_size
        if not fs:
            return None
        try:
            w, h = fs
            if w > 0 and h > 0:
                return float(w) / float(h)
        except Exception:
            pass
        return None

    def set_reference_aspect(self, aspect: float | None) -> None:
        """Sets the reference video W/H aspect (None = legacy)."""
        try:
            aspect = float(aspect) if aspect is not None else None
            if aspect is not None and not (aspect > 0):
                aspect = None
        except Exception:
            aspect = None
        with self._lock:
            self.reference_aspect = aspect

    def get_reference_aspect(self) -> float | None:
        with self._lock:
            return self.reference_aspect


class CameraThread(QThread):
    """Capture loop: camera -> detection -> overlay -> QImage + multi poses."""

    frame_ready = Signal(QImage)
    stats_ready = Signal(float, float, str)  # fps, brillo, backend
    failed = Signal(str)
    device_warning = Signal(str)  # cuda->cpu fallback or other non-fatal warning
    tracks_ready = Signal(object)  # dict track_id -> bbox+pose

    MAX_CONSECUTIVE_READ_FAILURES = 60
    DARK_BRIGHTNESS = 20.0
    WORK_WIDTH = 640

    def __init__(self, settings, shared: SharedState, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.shared = shared
        self._stop = threading.Event()
        self._backend = None
        self._camera = None
        self.enable_multi = False  # enabled from Lobby in multi mode; mono compat by default
        self._expected_players: int | None = None
        self._song_num_dancers: int | None = None  # song dancer count (wins over lobby)
        self._frame_std = 0.0  # for gray diagnostics
        self._reset_tracking_requested = False

    def set_multi(self, enabled: bool) -> None:
        self.enable_multi = bool(enabled)

    def set_expected_players(self, n: int | None) -> None:
        try:
            self._expected_players = int(n) if n is not None else None
        except Exception:
            self._expected_players = None

    def set_song_num_dancers(self, n: int | None) -> None:
        """Sets the song dancer count (1-4). Informational fallback only.

        The camera follows PLAYERS (expected_players), not dancers: 1 player
        in front of the webcam -> YOLO with max 1 person even when the video has 4.
        Call on song change + request_tracking_reset().
        """
        try:
            self._song_num_dancers = max(1, min(4, int(n))) if n is not None else None
        except Exception:
            self._song_num_dancers = None

    def effective_dancers(self) -> int:
        """Number of PEOPLE to track on webcam (players first)."""
        for cand in (self._expected_players, self._song_num_dancers):
            try:
                if cand is not None and int(cand) >= 1:
                    return max(1, min(4, int(cand)))
            except Exception:
                continue
        return 4 if self.enable_multi else 1

    def request_tracking_reset(self) -> None:
        """Requests a tracking ID reset (runs inside the thread)."""
        self._reset_tracking_requested = True

    def _is_gray_frame(self, frame) -> bool:
        try:
            import numpy as np
            std = float(np.std(frame))
            mean = float(np.mean(frame))
            self._frame_std = std
            return (std < 8.0 and 115.0 < mean < 140.0)
        except Exception:
            return False

    def run(self) -> None:
        import cv2
        import numpy as np

        try:
            self._backend = create_backend(self.settings.backend)
        except Exception as exc:
            self.failed.emit(f"No se pudo crear el backend de pose: {exc}")
            return

        camera, fb_warning, fb_error = open_camera_with_fallback(
            self.settings.camera_index,
            self.settings.camera_width,
            self.settings.camera_height,
        )
        if camera is None:
            self.failed.emit(fb_error or "No se pudo abrir la camara")
            self._backend.close()
            return
        if fb_warning:
            try:
                self.device_warning.emit(fb_warning)
            except Exception:
                pass
        self._camera = camera

        normalizer = BodyNormalizer()
        smoother = PoseSmoother(self.settings.smooth_alpha)
        # per-track normalizers/smoothers for multi (key = stable track_id)
        multi_norms: dict[int, BodyNormalizer] = {}
        multi_smooth: dict[int, PoseSmoother] = {}
        # Fixed router: 1 dancer = MediaPipe only; >1 = YOLO11n-pose only.
        # The live YOLO backend is ALWAYS yolo11n-pose.pt (lightweight) with
        # direct keypoints (no per-person MediaPipe).
        # Loaded once (not per frame) for multi lobby/game.
        # Device from settings (auto/cuda/cpu) with cuda -> cpu cascade.
        _yolo_live, _yolo_warning = create_live_yolo_backend(
            getattr(self.settings, "device", "auto"))
        if _yolo_warning:
            try:
                self.device_warning.emit(f"Inferencia: {_yolo_warning}")
            except Exception:
                pass
        try:
            multi_backends = get_multi_backends(mp_backend=self._backend, yolo=False)
            if _yolo_live is not None:
                multi_backends.yolo_backend = _yolo_live
        except Exception:
            multi_backends = get_multi_backends(mp_backend=self._backend, yolo=False)
        slot_mapper = StableSlotMapper(max_slots=4)
        track_registry = TrackRegistry(max_slots=4)
        fps = 0.0
        mirror = self.settings.mirror
        read_failures = 0
        gray_streak = 0
        boosted = False
        orig_exposure = None
        orig_brightness = None
        orig_contrast = None
        orig_auto_exp = None
        PLAYER_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 200, 0), (255, 0, 255), (0, 255, 255)]

        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                frame = camera.read()
                if frame is None:
                    if self._stop.is_set():
                        break
                    read_failures += 1
                    if read_failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
                        self.failed.emit(
                            "La cámara no entrega frames. Revisa que no esté en uso "
                            "por otra app o desconéctala/conéctala."
                        )
                        return
                    time.sleep(0.03)
                    continue
                read_failures = 0

                if mirror:
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                if w > self.WORK_WIDTH:
                    scale = self.WORK_WIDTH / w
                    frame = cv2.resize(frame, (self.WORK_WIDTH, int(h * scale)))
                # Seal the processed frame size: raw poses are in
                # 0..1 of THIS frame; its aspect fixes the comparison.
                # k pre-scales x into comparison space (1.0 = legacy).
                try:
                    fh, fw = frame.shape[:2]
                    self.shared.set_frame_size(fw, fh)
                    _k = compare_space_prescale(
                        self.shared.get_reference_aspect(),
                        (float(fw) / float(fh)) if fh else None,
                    )
                except Exception:
                    _k = 1.0

                brightness = float(np.mean(frame))
                is_gray = self._is_gray_frame(frame)
                if is_gray:
                    gray_streak += 1
                else:
                    gray_streak = max(0, gray_streak - 1)

                # Detect uniform gray (typical of badly fixed manual exposure on integrated cams).
                # When it persists >1.2s (~15 frames at 12 effective fps) revert to auto-exposure.
                if gray_streak >= 15 and boosted:
                    try:
                        if orig_auto_exp is not None:
                            camera._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, orig_auto_exp)
                        else:
                            camera._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                        if orig_brightness is not None:
                            camera._cap.set(cv2.CAP_PROP_BRIGHTNESS, orig_brightness)
                        if orig_contrast is not None:
                            camera._cap.set(cv2.CAP_PROP_CONTRAST, orig_contrast)
                        if orig_exposure is not None:
                            camera._cap.set(cv2.CAP_PROP_EXPOSURE, orig_exposure)
                    except Exception:
                        pass
                    boosted = False
                    gray_streak = 0

                # When the image arrives almost black, try auto-exposure instead of fixing a dark manual one.
                # Only in non-multi mode or extreme darkness, reversibly (no EXPOSURE=-6 leaving gray on integrated cams).
                if not boosted and brightness < self.DARK_BRIGHTNESS and not is_gray:
                    # save originals for revert
                    try:
                        orig_brightness = camera._cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        orig_contrast = camera._cap.get(cv2.CAP_PROP_CONTRAST)
                        orig_exposure = camera._cap.get(cv2.CAP_PROP_EXPOSURE)
                        orig_auto_exp = camera._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    except Exception:
                        pass
                    boosted = True
                    try:
                        # prefer auto-exposure; on integrated cams 0.75 = auto
                        camera._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                        # slight boost without closing the shutter
                        # do not touch manual EXPOSURE (-6 caused gray)
                    except Exception:
                        pass
                    # schedule revert in 3s when it does not improve
                    # handled in loop via gray_streak / brightness

                if self.enable_multi or self._song_num_dancers is not None:
                    # SINGLE live pipeline: always YOLO11n-pose for 1..N
                    # people (same estimation as the extracted reference).
                    # MediaPipe ONLY when YOLO failed to load (missing model).
                    _yolo_ok = multi_backends.yolo_backend is not None
                    if not _yolo_ok:
                        # Fallback without YOLO (missing model): mono MediaPipe.
                        raw = self._backend.detect(frame)
                        # draw single pose only (no multi colors)
                        if not raw.is_empty():
                            draw_pose(frame, raw)
                            normalized = normalizer.normalize(to_compare_space(raw, _k))
                            smoothed = smoother.smooth(normalized) if not normalized.is_empty() else normalized
                            self.shared.update(smoothed, raw, time.perf_counter(), brightness)
                            tp = TrackedPerson(track_id=1, bbox=(0, 0, w, h), pose=raw, raw=raw, norm_pose=smoothed)
                            tp.timestamp = time.perf_counter()
                            self.shared.update_multi({1: tp}, time.perf_counter(), brightness)
                        else:
                            self.shared.update(None, None, time.perf_counter(), brightness)
                            self.shared.update_multi({}, time.perf_counter(), brightness)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h_r, w_r, _ = rgb.shape
                        qimg = QImage(rgb.data, w_r, h_r, w_r * 3, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(qimg)
                    else:
                        # YOLO11n-pose + TrackRegistry, stable IDs.
                        # 1 player -> max 1 person (largest); N -> N.
                        # No HOG+MediaPipe fallback: YOLO only live.
                        if self._reset_tracking_requested:
                            self._reset_tracking_requested = False
                            track_registry.reset()
                            slot_mapper.reset()
                            try:
                                if multi_backends.yolo_backend is not None:
                                    multi_backends.yolo_backend.reset_track()
                            except Exception:
                                pass
                            multi_norms.clear()
                            multi_smooth.clear()
                        max_p = self.effective_dancers()
                        try:
                            if multi_backends.yolo_backend is not None:
                                multi_backends.yolo_backend.max_persons = max_p
                        except Exception:
                            pass
                        tracked: list = []
                        try:
                            tracked = track_multi_poses(frame, multi_backends, track_registry, max_persons=max_p)
                        except Exception:
                            tracked = []
                        dets: list = []  # no HOG/MediaPipe fallback live
                        bbox_to_tid: dict = {}
                        if tracked:
                            for (slot, bbox, _raw) in tracked:
                                bbox_to_tid[bbox] = slot + 1  # track_id 1-based permanente
                        else:
                            slot_mapper.reset()
                        framed: list = [(bbox, raw) for (_s, bbox, raw) in tracked] if tracked else dets
                        tracks: dict[int, TrackedPerson] = {}
                        for (bbox, raw) in framed:
                            tid = bbox_to_tid.get(bbox)
                            if tid is None:
                                continue
                            x0, y0, wb, hb = bbox
                            color_idx = (tid - 1) % len(PLAYER_COLORS)
                            cv2.rectangle(frame, (x0, y0), (x0 + wb, y0 + hb), PLAYER_COLORS[color_idx], 2)
                            cv2.putText(frame, f"ID {tid}", (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PLAYER_COLORS[color_idx], 2)
                            norm = multi_norms.get(tid)
                            if norm is None:
                                norm = BodyNormalizer()
                                multi_norms[tid] = norm
                            sm = multi_smooth.get(tid)
                            if sm is None:
                                sm = PoseSmoother(self.settings.smooth_alpha)
                                multi_smooth[tid] = sm
                            normalized = norm.normalize(to_compare_space(raw, _k))
                            smoothed = sm.smooth(normalized) if not normalized.is_empty() else normalized
                            tp = TrackedPerson(track_id=tid, bbox=(x0, y0, wb, hb), pose=raw, raw=raw, norm_pose=smoothed)
                            tp.timestamp = time.perf_counter()
                            tracks[tid] = tp
                            Hh, Ww = frame.shape[:2]
                            for name, joint in raw.joints.items():
                                if joint.visibility < 0.3:
                                    continue
                                fx = int(joint.x * Ww)
                                fy = int(joint.y * Hh)
                                cv2.circle(frame, (fx, fy), 3, (255, 255, 255), -1)
                                cv2.circle(frame, (fx, fy), 3, PLAYER_COLORS[color_idx], 1)
                            from engine.pose_engine.skeleton import POSE_CONNECTIONS
                            for a, b_ in POSE_CONNECTIONS:
                                ja = raw.get(a); jb = raw.get(b_)
                                if ja is None or jb is None or ja.visibility < 0.3 or jb.visibility < 0.3:
                                    continue
                                pa = (int(ja.x * Ww), int(ja.y * Hh))
                                pb = (int(jb.x * Ww), int(jb.y * Hh))
                                cv2.line(frame, pa, pb, PLAYER_COLORS[color_idx], 2)
                        # YOLO only live: when it sees nobody, empty (no synthetics).
                        if not tracks:
                            self.shared.update(None, None, time.perf_counter(), brightness)
                        else:
                            # when there are multi tracks, update legacy with first track for snapshot()
                            first = sorted(tracks.items(), key=lambda kv: kv[0])[0][1]
                            self.shared.update(first.norm_pose, first.raw, time.perf_counter(), brightness)
                        self.shared.update_multi(tracks, time.perf_counter(), brightness)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h_r, w_r, _ = rgb.shape
                        qimg = QImage(rgb.data, w_r, h_r, w_r * 3, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(qimg)
                        try:
                            self.tracks_ready.emit(dict(tracks))
                        except Exception:
                            pass
                else:
                    raw = self._backend.detect(frame)
                    draw_pose(frame, raw)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = rgb.shape
                    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(qimg)
                    normalized = normalizer.normalize(to_compare_space(raw, _k))
                    smoothed = smoother.smooth(normalized)
                    self.shared.update(smoothed, raw, time.perf_counter(), brightness)
                    # also update multi with 1 dummy track for compat
                    if not raw.is_empty():
                        tp = TrackedPerson(track_id=1, bbox=(0, 0, w, h), pose=raw, raw=raw, norm_pose=smoothed)
                        self.shared.update_multi({1: tp}, time.perf_counter(), brightness)
                    else:
                        self.shared.update_multi({}, time.perf_counter(), brightness)

                dt = time.perf_counter() - t0
                if dt > 0:
                    fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps else 1.0 / dt
                self.shared.set_fps(fps)
                try:
                    _yb = multi_backends.yolo_backend
                    _live_name = live_backend_label(_yb) if _yb is not None else self._backend.name
                except Exception:
                    _live_name = self._backend.name
                self.stats_ready.emit(fps, brightness, _live_name)
        except Exception as exc:  # pragma: no cover - defensivo
            self.failed.emit(f"Error en el pipeline de cámara: {exc}\n{traceback.format_exc()}")
        finally:
            camera.release()
            try:
                yb = getattr(locals().get("multi_backends", None), "yolo_backend", None)
                if yb is not None:
                    yb.close()
            except Exception:
                pass
            self._backend.close()
            self._backend = None
            self._camera = None

    def stop(self) -> None:
        self._stop.set()
        # Release the camera immediately: when read() is blocked (e.g. a
        # webcam delivering a black/broken image), the event alone is not enough
        # to finish the thread and the camera would stay on.
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self.isRunning() and not self._stop.is_set()
