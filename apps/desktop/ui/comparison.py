"""Widgets for pose comparison on the results screen."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QImage
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from engine.pose_engine.skeleton import POSE_CONNECTIONS, Pose

from apps.desktop.i18n import normalize_lang, tr


class SkeletonWidget(QWidget):
    """Draws a skeleton from a Pose (normalized or raw 0..1)."""

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._pose: Optional[Pose] = None
        self._is_normalized = True
        self._title = title
        self._lang = "en"
        self._line_color = QColor("#00e676")
        self._dot_color = QColor("#00b0ff")
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0f0f0f; border-radius:12px; border:1px solid #2c2c2c;")

    def set_pose(self, pose: Optional[Pose], is_normalized: bool = True) -> None:
        self._pose = pose
        self._is_normalized = is_normalized
        self.update()

    def set_colors(self, line: str = "#00e676", dot: str = "#00b0ff") -> None:
        """Skeleton color (e.g. the followed dancer's palette color)."""
        self._line_color = QColor(line)
        self._dot_color = QColor(dot)
        self.update()

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def _display_flip(self) -> float:
        """Detects whether the pose has flipped Y (head y < hip y => negative up)
        and returns -1 to fix it, 1 when Y is already positive up."""
        if self._pose is None or self._pose.is_empty():
            return 1.0
        # use head vs hip center
        head = self._pose.get("head")
        lh = self._pose.get("left_hip")
        rh = self._pose.get("right_hip")
        if head is None:
            # fallback to shoulders
            head = self._pose.get("left_shoulder") or self._pose.get("right_shoulder")
        hip_y = None
        if lh and rh:
            hip_y = (lh.y + rh.y) / 2.0
        elif lh:
            hip_y = lh.y
        elif rh:
            hip_y = rh.y
        if head is None or hip_y is None:
            return 1.0
        # In correct normalized space (Y+ up) head.y > hip_y (hip at 0, head +2..3)
        # In old files (Y- up) head.y < hip_y (head -3, hip 0)
        # Raw also has head.y < hip_y but is_normalized=False never reaches here
        if head.y < hip_y:
            return -1.0
        return 1.0

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f0f0f"))

        if self._title:
            painter.setPen(QColor("#888"))
            painter.drawText(8, 18, self._title)

        if self._pose is None or self._pose.is_empty():
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignCenter, tr(self._lang, "comp.no_pose"))
            return

        w = self.width()
        h = self.height()
        flip = self._display_flip() if self._is_normalized else 1.0

        # To avoid clipping (head/feet outside) compute a scale fitting the bounding box
        scale = min(w, h) * 0.32
        cx = w / 2.0
        cy = h / 2.0 + 14
        if self._is_normalized and self._pose is not None:
            # bounding-box auto-fit (same logic for user and reference when normalized)
            xs = []
            ys = []
            for j in self._pose.joints.values():
                if j.visibility < 0.3:
                    continue
                xs.append(j.x)
                ys.append(j.y * flip)
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bw = max_x - min_x
                bh = max_y - min_y
                if bw > 1e-6 and bh > 1e-6:
                    # leave a 12% margin and cap max scale so small poses are not blown up
                    sx = (w * 0.86) / bw
                    sy = (h * 0.78) / bh
                    fitted = min(sx, sy)
                    # never grow beyond the fixed scale, only shrink when clipping
                    scale = min(scale, fitted)
                    # center on the bounding-box center, not the (0,0) origin, so a cropped pose stays whole
                    # but keep the hip origin near the visual center: average origin and bbox center
                    # so it does not drift too far
                    bx = (min_x + max_x) / 2.0
                    by = (min_y + max_y) / 2.0
                    # blended center: 70% origin (0,0) + 30% bbox center -> like the reference but unclipped
                    cx = w / 2.0 - bx * scale * 0.30
                    cy = h / 2.0 + 14 - by * scale * 0.30

        def pt(name: str):
            joint = self._pose.get(name)
            if joint is None or joint.visibility < 0.3:
                return None
            if self._is_normalized:
                y_corr = joint.y * flip
                x = cx + joint.x * scale
                y = cy - y_corr * scale
                return (int(x), int(y))
            else:
                return (int(joint.x * w), int(joint.y * h))

        # connections
        pen_line = QPen(self._line_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_line)
        for a, b in POSE_CONNECTIONS:
            pa, pb = pt(a), pt(b)
            if pa is not None and pb is not None:
                painter.drawLine(pa[0], pa[1], pb[0], pb[1])

        # joints
        for name in self._pose.joints:
            p = pt(name)
            if p is None:
                continue
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(self._dot_color)
            painter.drawEllipse(p[0] - 4, p[1] - 4, 8, 8)


class VideoSkeletonWidget(QWidget):
    """Shows the song video frame + skeleton on top.

    Supports two modes:
    - normalized (is_normalized=True): centered skeleton with the same scale as SkeletonWidget (fair comparison).
    - raw (is_normalized=False): skeleton aligned to the video (0..1).
    By default the comparison uses normalized so proportions match.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pose: Optional[Pose] = None
        self._is_normalized = True
        self._lang = "en"
        self._qimg: Optional[QImage] = None
        self._video_path: Optional[Path] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_t: Optional[float] = None  # time of the frame in _qimg
        self._line_color = QColor("#00e676")
        self._dot_color = QColor("#00b0ff")
        self._ref_items: list = []  # [(dancer_id, Pose raw 0..1)] all
        self._selected_id: int = 0
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#000; border-radius:12px; border:1px solid #2c2c2c;")

    def set_colors(self, line: str = "#00e676", dot: str = "#00b0ff") -> None:
        """Skeleton color (e.g. the followed dancer's palette color)."""
        self._line_color = QColor(line)
        self._dot_color = QColor(dot)
        self.update()

    def set_video(self, video_path: Optional[Path]) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_path = Path(video_path) if video_path else None
        self._qimg = None
        self._last_t = None
        if self._video_path and self._video_path.exists():
            self._cap = cv2.VideoCapture(str(self._video_path))
        self.update()

    def set_frame_at(self, time_s: float) -> None:
        """EXACT video frame at time_s (no keyframe error).

        `CAP_PROP_POS_MSEC` lands on the previous keyframe (seconds of error
        in mp4): instead of keeping that frame, step back ~0.6s and read
        forward to the requested time. Small advances (50ms play)
        reuse the sequential position and are cheap.
        """
        if self._cap is None or not self._cap.isOpened():
            self._qimg = None
            self._last_t = None
            self.update()
            return
        try:
            t = max(0.0, float(time_s))
            # cheap sequential advance when asking slightly ahead
            use_seq = (self._last_t is not None and self._qimg is not None
                       and 0.0 <= t - self._last_t <= 0.25)
            if not use_seq:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t - 0.6)) * 1000.0)
            frame = None
            pos_ms = 0.0
            for _ in range(60):  # at most ~2s of video @30fps (bounded: never hang the UI)
                ok, fr = self._cap.read()
                if not ok or fr is None:
                    break
                try:
                    pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                except Exception:
                    pos_ms = t
                frame = fr
                if pos_ms >= t - 0.03:
                    break
            if frame is None:
                self._qimg = None
                self._last_t = None
                self.update()
                return
            self._last_t = pos_ms
            h0, w0 = frame.shape[:2]
            longest = max(h0, w0)
            if longest > 960:
                s = 960.0 / float(longest)
                frame = cv2.resize(frame, (int(w0 * s), int(h0 * s)),
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            self._qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self.update()
        except Exception:
            self._qimg = None
            self._last_t = None
            self.update()

    def set_pose(self, pose: Optional[Pose], is_normalized: bool = True) -> None:
        self._pose = pose
        self._is_normalized = is_normalized
        self.update()

    def set_ref_poses(self, items: list, selected_id: int = 0) -> None:
        """All raw reference poses (one per dancer) + selected.

        items: [(dancer_id, Pose in video 0..1 coords)]. Each is drawn
        with the process palette color; the selected one thicker +
        D{n} label. This matches the screen to preview_pose.mp4.
        """
        self._ref_items = list(items or [])
        try:
            self._selected_id = int(selected_id)
        except Exception:
            self._selected_id = 0
        self.update()

    @staticmethod
    def _dancer_qcolor(dancer_id: int) -> QColor:
        try:
            from engine.pose_engine.skeleton import dancer_color_bgr

            b, g, r = dancer_color_bgr(dancer_id)
            return QColor(r, g, b)
        except Exception:
            return QColor("#00e676")

    def close_video(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _display_flip(self) -> float:
        if self._pose is None or self._pose.is_empty() or not self._is_normalized:
            return 1.0
        head = self._pose.get("head")
        lh = self._pose.get("left_hip")
        rh = self._pose.get("right_hip")
        if head is None:
            head = self._pose.get("left_shoulder") or self._pose.get("right_shoulder")
        hip_y = None
        if lh and rh:
            hip_y = (lh.y + rh.y) / 2.0
        elif lh:
            hip_y = lh.y
        elif rh:
            hip_y = rh.y
        if head is None or hip_y is None:
            return 1.0
        if head.y < hip_y:
            return -1.0
        return 1.0

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#000"))

        w = self.width()
        h = self.height()

        has_video = self._qimg is not None and not self._qimg.isNull()
        # Reference back to video-aligned RAW mode (as before) - no alpha
        if self._is_normalized:
            # The normalized mode stays available, but the current comparison uses raw;
            # this branch is kept in case it is used elsewhere
            if has_video:
                pix = QPixmap.fromImage(self._qimg).scaled(
                    w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                x = (w - pix.width()) // 2
                y = (h - pix.height()) // 2
                painter.setOpacity(0.45)
                painter.drawPixmap(x, y, pix)
                painter.setOpacity(1.0)
            if self._pose is None or self._pose.is_empty():
                if not has_video:
                    painter.setPen(QColor("#666"))
                    painter.drawText(self.rect(), Qt.AlignCenter, tr(self._lang, "comp.no_video"))
                return
            flip = self._display_flip()
            scale = min(w, h) * 0.32
            cx = w / 2.0
            cy = h / 2.0 + 18

            def pt_norm(name: str):
                joint = self._pose.get(name)
                if joint is None or joint.visibility < 0.3:
                    return None
                y_corr = joint.y * flip
                x = cx + joint.x * scale
                y = cy - y_corr * scale
                return (int(x), int(y))

            pen = QPen(self._line_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            for a, b in POSE_CONNECTIONS:
                pa, pb = pt_norm(a), pt_norm(b)
                if pa and pb:
                    painter.drawLine(pa[0], pa[1], pb[0], pb[1])
            for n in self._pose.joints:
                p = pt_norm(n)
                if p:
                    painter.setPen(QPen(QColor("#ffffff"), 1))
                    painter.setBrush(self._dot_color)
                    painter.drawEllipse(p[0] - 4, p[1] - 4, 8, 8)
            return

        # RAW MODE (reference): full-size video + aligned skeletons.
        # Draw ALL dancers with their process color and highlight the selected one.
        if has_video:
            pix = QPixmap.fromImage(self._qimg).scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (w - pix.width()) // 2
            y = (h - pix.height()) // 2
            painter.drawPixmap(x, y, pix)
            img_w, img_h = self._qimg.width(), self._qimg.height()
            scale_v = min(w / img_w, h / img_h)
            disp_w = int(img_w * scale_v)
            disp_h = int(img_h * scale_v)
            off_x = (w - disp_w) // 2
            off_y = (h - disp_h) // 2

            def pt_raw(pose, name: str):
                joint = pose.get(name) if pose else None
                if joint is None or joint.visibility < 0.3:
                    return None
                return (int(off_x + joint.x * disp_w), int(off_y + joint.y * disp_h))

            items = list(self._ref_items) if self._ref_items else (
                [(self._selected_id, self._pose)] if self._pose and not self._pose.is_empty() else []
            )
            # non-selected first, selected last (on top)
            items = sorted(items, key=lambda it: 0 if it[0] != self._selected_id else 1)
            for did, pose in items:
                if pose is None or pose.is_empty():
                    continue
                sel = (did == self._selected_id)
                col = self._dancer_qcolor(did)
                pen = QPen(col, 3 if sel else 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                for a, b in POSE_CONNECTIONS:
                    pa, pb = pt_raw(pose, a), pt_raw(pose, b)
                    if pa and pb:
                        painter.drawLine(pa[0], pa[1], pb[0], pb[1])
                for name in pose.joints:
                    p = pt_raw(pose, name)
                    if p is None:
                        continue
                    painter.setPen(QPen(QColor("#ffffff"), 1))
                    painter.setBrush(col)
                    _r = 4 if sel else 3
                    painter.drawEllipse(p[0] - _r, p[1] - _r, 2 * _r, 2 * _r)
                if sel:
                    head = pose.get("head")
                    hp = pt_raw(pose, "head") if head else None
                    if hp:
                        painter.setPen(QPen(col, 2))
                        painter.drawText(hp[0] + 8, max(12, hp[1] - 8), f"D{did + 1}")
            return

        if self._pose is None or self._pose.is_empty():
            painter.setPen(QColor("#666"))
            painter.drawText(self.rect(), Qt.AlignCenter, tr(self._lang, "comp.no_video"))
            return
        # fallback without video but with raw pose: center
        painter.setPen(QColor("#666"))
        painter.drawText(self.rect(), Qt.AlignCenter, tr(self._lang, "comp.no_video"))


class PoseComparisonWidget(QWidget):
    """Side-by-side comparison: left recorded player pose, right reference video + extracted pose."""

    def __init__(self, parent=None, language: str = "en") -> None:
        super().__init__(parent)
        self.song = None
        self.recording: list[dict] = []
        self.times: list[float] = []
        self._lang = normalize_lang(language)

        self.left_canvas = SkeletonWidget(tr("en", "comp.your_pose"))
        self.left_canvas.setMinimumSize(200, 200)
        self.left_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_canvas = VideoSkeletonWidget()
        self.right_canvas.setMinimumSize(200, 200)
        self.right_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_label = QLabel(tr("en", "comp.ref_label"))
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setObjectName("micro")
        self.left_title = QLabel(tr("en", "comp.your_pose"))
        self.left_title.setAlignment(Qt.AlignCenter)
        self.left_title.setObjectName("micro")
        self.right_title = QLabel(tr("en", "comp.ref_title"))
        self.right_title.setAlignment(Qt.AlignCenter)
        self.right_title.setObjectName("micro")

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setObjectName("mono")
        self.rating_label = QLabel("—")
        self.rating_label.setAlignment(Qt.AlignCenter)
        self.rating_label.setObjectName("rating")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self._on_slider)
        # While dragging, random seeks on a long mp4 (230 MB/4K) are the
        # most expensive: update skeletons instantly but the video only on
        # release (same quality, no intermediate seeks hanging the UI).
        self._slider_held = False
        try:
            self.slider.sliderPressed.connect(self._on_slider_pressed)
            self.slider.sliderReleased.connect(self._on_slider_released)
            self.slider.sliderMoved.connect(self._on_slider_drag)
        except Exception:
            pass
        self._sync = None  # Cached SongSynchronizer (do not rebuild per update)
        self._sync_key: int | None = None

        self.btn_play = QPushButton(tr(self._lang, "comp.play"))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setFixedWidth(110)
        try:
            from apps.desktop.ui.icons import set_button_icon as _set_icon

            _set_icon(self.btn_play, "play", 16)
        except Exception:
            pass

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(50)
        self._play_timer.timeout.connect(self._tick_play)
        self._playing = False

        top = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(self.left_title)
        left_box.addWidget(self.left_canvas, 1)
        right_box = QVBoxLayout()
        right_box.addWidget(self.right_title)
        right_box.addWidget(self.right_canvas, 1)
        right_box.addWidget(self.right_label)
        top.addLayout(left_box, 1)
        top.addLayout(right_box, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider, 1)
        controls_info = QHBoxLayout()
        controls_info.addWidget(self.time_label)
        controls_info.addStretch()
        controls_info.addWidget(self.rating_label)

        lay = QVBoxLayout(self)
        lay.addLayout(top, 1)
        lay.addLayout(controls)
        lay.addLayout(controls_info)

    def set_data(self, song, recording: list[dict], dancer_id: int = 0) -> None:
        self.song = song
        self.recording = recording or []
        self.times = [r.get("time", 0.0) for r in self.recording] if self.recording else []
        try:
            self._dancer_id = max(0, int(dancer_id))
        except Exception:
            self._dancer_id = 0
        # Label with the followed reference dancer
        try:
            self.right_label.setText(tr(self._lang, "comp.ref_dancer", n=self._dancer_id + 1))
        except Exception:
            pass
        # Reference skeleton with the SAME process color
        # (single palette per dancer_id: preview, cards and comparison).
        try:
            from engine.pose_engine.skeleton import dancer_color_bgr

            b, g, r = dancer_color_bgr(self._dancer_id)
            self.right_canvas.set_colors(line=f"#{r:02x}{g:02x}{b:02x}", dot="#ffffff")
        except Exception:
            pass
        # video path
        video_path = None
        if song is not None and getattr(song, "path", None):
            p = Path(song.path) / "song.mp4"
            if p.exists():
                video_path = p
        self.right_canvas.set_video(video_path)
        # Cached synchronizer: previously 2 were built per update
        # (rebuilding the 7043-time list + bisects on every tick).
        try:
            from engine.timing_engine import SongSynchronizer

            self._sync = SongSynchronizer(song) if song is not None else None
            self._sync_key = id(song)
        except Exception:
            self._sync = None
            self._sync_key = None
        # set up slider
        duration = song.duration if song and song.duration else (self.times[-1] if self.times else 0)
        self.slider.setRange(0, max(1, int(duration * 1000)))
        self.slider.setValue(0)
        self._update_for_time(0.0)
        if not self.recording:
            self.left_canvas.set_pose(None)
            self.rating_label.setText(tr(self._lang, "comp.no_recording"))

    def _on_slider(self, val_ms: int) -> None:
        t = val_ms / 1000.0
        # valueChanged also fires while dragging: in that case the
        # drag already updated the skeletons; only the video is missing on release.
        self._update_for_time(t, with_video=not self._slider_held)

    def _on_slider_pressed(self) -> None:
        self._slider_held = True

    def _on_slider_drag(self, val_ms: int) -> None:
        self._slider_held = True
        self._update_for_time(val_ms / 1000.0, with_video=False)

    def _on_slider_released(self) -> None:
        self._slider_held = False
        try:
            self._update_for_time(self.slider.value() / 1000.0, with_video=True)
        except Exception:
            pass

    def _update_for_time(self, t: float, with_video: bool = True) -> None:
        # find closest player pose and reference pose
        player_pose = None
        ref_pose = None
        ref_raw_pose = None
        rating = "—"
        if self.recording:
            idx = bisect.bisect_left(self.times, t)
            if 0 <= idx < len(self.recording):
                # pick the closest
                if idx > 0 and (t - self.times[idx - 1] < self.times[idx] - t):
                    idx -= 1
                rec = self.recording[idx]
                # player pose (normalized dict)
                pd = rec.get("player")
                if pd:
                    try:
                        player_pose = Pose.from_dict(pd) if isinstance(pd, dict) else pd
                    except Exception:
                        player_pose = None
                rating = rec.get("rating", "—")
                # ref pose for the left skeleton? No, left is player; right needs ref
                # ref may come from the recording ref field or directly from song.poses
            else:
                if self.recording:
                    rec = self.recording[-1]
                    pd = rec.get("player")
                    if pd:
                        try:
                            player_pose = Pose.from_dict(pd) if isinstance(pd, dict) else pd
                        except Exception:
                            player_pose = None

        # ref pose from the song's FOLLOWED DANCER (not always 1):
        # same lookup as live scoring (dancer_pose_at), with legacy
        # fallback to joints/raw when the song has no dancer list.
        # A single frame_at per update (before: 2 synchronizers + 3 lookups).
        did = getattr(self, "_dancer_id", 0)
        try:
            if self._sync is None or self._sync_key != id(self.song):
                from engine.timing_engine import SongSynchronizer

                self._sync = SongSynchronizer(self.song) if self.song is not None else None
                self._sync_key = id(self.song)
            sync = self._sync
        except Exception:
            sync = None
        fp = None
        if sync is not None and self.song and self.song.poses:
            try:
                fp = sync.frame_at(t)
            except Exception:
                fp = None
            dpose = fp.get_dancer(did) if fp is not None else None
            try:
                if dpose is not None and dpose.joints:
                    ref_pose = Pose.from_dict(dpose.joints)
                if dpose is not None and dpose.raw:
                    ref_raw_pose = Pose.from_dict(dpose.raw)
            except Exception:
                pass
            if (ref_pose is None or ref_pose.is_empty()) and (ref_raw_pose is None or ref_raw_pose.is_empty()) and fp is not None:
                try:
                    if fp.joints:
                        ref_pose = Pose.from_dict(fp.joints)
                    if fp.raw:
                        ref_raw_pose = Pose.from_dict(fp.raw)
                except Exception:
                    pass

        # Left: normalized user pose, centered and auto-fitted so it never clips
        # Right: ALL raw reference poses aligned to the video with their
        # process color (same as preview_pose.mp4) + selected highlighted.
        self.left_canvas.set_pose(player_pose, is_normalized=True)
        items: list = []
        try:
            _fp = fp
            if _fp is not None and _fp.dancers:
                for _dd in _fp.dancers:
                    if _dd.raw:
                        try:
                            _rp = Pose.from_dict(_dd.raw)
                        except Exception:
                            _rp = None
                        if _rp is not None and not _rp.is_empty():
                            items.append((_dd.dancer_id, _rp))
            elif _fp is not None and _fp.raw:
                try:
                    _rp0 = Pose.from_dict(_fp.raw)
                except Exception:
                    _rp0 = None
                if _rp0 is not None and not _rp0.is_empty():
                    items.append((did, _rp0))
        except Exception:
            items = []
        if items:
            self.right_canvas.set_ref_poses(items, did)
        if ref_raw_pose is not None and not ref_raw_pose.is_empty():
            self.right_canvas.set_pose(ref_raw_pose, is_normalized=False)
        elif ref_pose is not None and not ref_pose.is_empty():
            self.right_canvas.set_pose(ref_pose, is_normalized=True)
        else:
            self.right_canvas.set_pose(None)
        # update right video frame (same file quality; skipped while
        # dragging the slider to avoid chaining random seeks that
        # hang the UI on long mp4s; the exact frame arrives on release).
        if with_video and self.song and getattr(self.song, "path", None):
            try:
                self.right_canvas.set_frame_at(t)
            except Exception:
                pass

        # labels
        dur = self.song.duration if self.song else (self.times[-1] if self.times else 0)
        def fmt(s):
            s = max(0, int(s))
            m, sec = divmod(s, 60)
            return f"{m}:{sec:02d}"
        self.time_label.setText(f"{fmt(t)} / {fmt(dur)}")
        # color rating
        colors = {"Perfect": "#00e676", "Great": "#76ff03", "Good": "#ffd600", "Ok": "#ff9100", "Miss": "#ff5252", "Neutral": "#888", "—": "#888"}
        col = colors.get(rating, "#888")
        self.rating_label.setText(rating)
        self.rating_label.setStyleSheet(f"QLabel#rating {{ color: {col}; }}")

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.left_title.setText(tr(code, "comp.your_pose"))
        self.right_title.setText(tr(code, "comp.ref_title"))
        try:
            did = getattr(self, "_dancer_id", 0)
            self.right_label.setText(tr(code, "comp.ref_dancer", n=int(did) + 1))
        except Exception:
            self.right_label.setText(tr(code, "comp.ref_label"))
        self._set_play_icon(bool(getattr(self, "_playing", False)))
        try:
            self.left_canvas._lang = code
            self.right_canvas._lang = code
            self.left_canvas.update()
            self.right_canvas.update()
        except Exception:
            pass

    def _set_play_icon(self, playing: bool) -> None:
        self.btn_play.setText(tr(self._lang, "comp.pause") if playing else tr(self._lang, "comp.play"))
        try:
            from apps.desktop.ui.icons import set_button_icon as _set_icon

            _set_icon(self.btn_play, "pause" if playing else "play", 16)
        except Exception:
            pass

    def _toggle_play(self) -> None:
        if self._playing:
            self._play_timer.stop()
            self._set_play_icon(False)
            self._playing = False
        else:
            # when at the end, restart
            if self.slider.value() >= self.slider.maximum() - 5:
                self.slider.setValue(0)
            self._play_timer.start()
            self._set_play_icon(True)
            self._playing = True

    def _tick_play(self) -> None:
        val = self.slider.value() + 50  # 50ms step
        if val >= self.slider.maximum():
            val = self.slider.maximum()
            self.slider.setValue(val)
            self._play_timer.stop()
            self._set_play_icon(False)
            self._playing = False
            return
        self.slider.setValue(val)

    def stop(self) -> None:
        self._play_timer.stop()
        self._playing = False
        self._set_play_icon(False)
        self.right_canvas.close_video()

    def showEvent(self, event) -> None:
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.stop()
        super().hideEvent(event)
