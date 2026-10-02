"""Reference dancer selection per player (multi).

Each dancer shows a card with its reference image (cover crop with its
drawn skeleton), so the player sees whom they follow.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QFrame,
)

from apps.desktop.i18n import normalize_lang, tr
from apps.desktop.ui.icons import set_button_icon
from apps.desktop.ui.responsive import FlowLayout, scrollable
from apps.desktop.ui.theme import COLORS
from engine.pose_engine.skeleton import POSE_CONNECTIONS, dancer_color_bgr


# --- "Best photo" scoring per dancer (runtime, no files) ---
# Port of the overlap/visibility/sharpness/size/edge/face criterion to
# existing song.mp4 + song.poses. Face = `head` visibility
# (no extra InsightFace). All with cv2/numpy, no heavy dependencies.
SNAPSHOT_W_OVERLAP = 0.30
SNAPSHOT_W_VISIBILITY = 0.20
SNAPSHOT_W_SHARPNESS = 0.15
SNAPSHOT_W_SIZE = 0.15
SNAPSHOT_W_EDGE = 0.10
SNAPSHOT_W_FACE = 0.10
SNAPSHOT_SHARPNESS_REF = 300.0  # Laplacian var. already counting as "sharp"
SNAPSHOT_TARGET_H_RATIO = 0.45  # box_h/H already counting as "close" (horiz.)
SNAPSHOT_TARGET_H_RATIO_VERT = 0.50  # same on vertical 9:16
SNAPSHOT_TARGET_W_RATIO_VERT = 0.60  # box_w/W already counting as "close"
SNAPSHOT_EDGE_MARGIN_PX = 4
SNAPSHOT_MARGIN_X = 0.08  # side padding of the final crop
SNAPSHOT_MARGIN_Y = 0.12  # vertical padding (more head/feet room)
SNAPSHOT_MAX_CANDIDATES = 16  # cap per card (grid performance)
SNAPSHOT_VIS_THRESH = 0.4
SNAPSHOT_MIN_JOINTS = 4

_CARD_CACHE: dict = {}  # (str(song.path), dancer_id, size) -> QPixmap


def _is_portrait_wh(w: int, h: int) -> bool:
    try:
        return float(h) > float(w) * 1.2
    except Exception:
        return False


def _video_dims(video_path: Path):
    """(W, H) of the video or (None, None)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w or None, h or None
    except Exception:
        return None, None


def _song_is_vertical(song) -> bool:
    """True when the song is vertical (9:16): checks cover.png then video."""
    try:
        song_dir = Path(song.path) if getattr(song, "path", None) else None
        if song_dir is not None:
            cover = song_dir / "cover.png"
            if cover.exists():
                try:
                    import cv2

                    img = cv2.imread(str(cover))
                    if img is not None and img.size:
                        h, w = img.shape[:2]
                        return _is_portrait_wh(w, h)
                except Exception:
                    pass
            for name in ("song.mp4", "preview_pose.mp4"):
                v = song_dir / name
                if v.exists():
                    w, h = _video_dims(v)
                    if w and h:
                        return _is_portrait_wh(w, h)
    except Exception:
        pass
    return False


def _joints_points(joints, w: int, h: int, vis_thresh: float = SNAPSHOT_VIS_THRESH):
    """Visible points [(x, y)] in pixels + head vis (0..1)."""
    pts = []
    head_vis = 0.0
    try:
        for name, arr in (joints or {}).items():
            try:
                x, y = float(arr[0]) * w, float(arr[1]) * h
                v = float(arr[3]) if len(arr) > 3 else 1.0
            except Exception:
                continue
            if name == "head":
                try:
                    head_vis = max(head_vis, float(v))
                except Exception:
                    pass
            if v < vis_thresh:
                continue
            pts.append((x, y))
    except Exception:
        pass
    return pts, head_vis


def _box_iou(a, b) -> float:
    try:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / (ua + 1e-6)
    except Exception:
        return 0.0


def _laplacian_score(gray_crop) -> float:
    try:
        import cv2

        if gray_crop is None or gray_crop.size == 0:
            return 0.0
        var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
        return min(1.0, var / SNAPSHOT_SHARPNESS_REF)
    except Exception:
        return 0.0


def _score_candidate(frame, joints, other_joints_list, w: int, h: int,
                     vertical: bool = False) -> float:
    """0..1 score of this frame as the dancer card."""
    try:
        import cv2

        pts, head_vis = _joints_points(joints, w, h)
        if len(pts) < 3:
            return -1.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        box = (x0, y0, x1, y1)
        box_w, box_h = max(1.0, x1 - x0), max(1.0, y1 - y0)

        # overlap: 1 - max IoU against other dancers of the same FramePose
        overlap = 0.0
        for oj in other_joints_list or []:
            opts, _ = _joints_points(oj, w, h)
            if len(opts) < 3:
                continue
            oxs = [p[0] for p in opts]
            oys = [p[1] for p in opts]
            obox = (min(oxs), min(oys), max(oxs), max(oys))
            overlap = max(overlap, _box_iou(box, obox))
        overlap_score = 1.0 - min(1.0, overlap)

        # visibility: fraction of joints with enough conf
        try:
            vis_vals = [float(a[3]) if len(a) > 3 else 1.0
                        for a in (joints or {}).values()]
            visibility = (sum(1 for v in vis_vals if v >= SNAPSHOT_VIS_THRESH)
                          / max(1, len(vis_vals)))
        except Exception:
            visibility = 0.0

        # sharpness over the box crop
        mx1, my1 = max(0, int(x0)), max(0, int(y0))
        mx2, my2 = min(w, int(x1)), min(h, int(y1))
        crop = frame[my1:my2, mx1:mx2]
        if crop.size == 0:
            return -1.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = _laplacian_score(gray)

        # size: near/large scores higher (on vertical also look at width)
        if vertical:
            size_score = max(min(1.0, (box_h / max(1, h)) / SNAPSHOT_TARGET_H_RATIO_VERT),
                             min(1.0, (box_w / max(1, w)) / SNAPSHOT_TARGET_W_RATIO_VERT))
        else:
            size_score = min(1.0, (box_h / max(1, h)) / SNAPSHOT_TARGET_H_RATIO)

        # edge: penalize when touching the border (on vertical, side edges x2)
        touches_x = (x0 <= SNAPSHOT_EDGE_MARGIN_PX
                     or x1 >= w - SNAPSHOT_EDGE_MARGIN_PX)
        touches_y = (y0 <= SNAPSHOT_EDGE_MARGIN_PX
                     or y1 >= h - SNAPSHOT_EDGE_MARGIN_PX)
        if vertical:
            edge_score = 1.0
            if touches_x:
                edge_score -= 0.6
            if touches_y:
                edge_score -= 0.3
            edge_score = max(0.2, edge_score)
        else:
            edge_score = 0.4 if (touches_x or touches_y) else 1.0

        face_score = min(1.0, max(0.0, float(head_vis)))

        return (SNAPSHOT_W_OVERLAP * overlap_score
                + SNAPSHOT_W_VISIBILITY * visibility
                + SNAPSHOT_W_SHARPNESS * sharpness
                + SNAPSHOT_W_SIZE * size_score
                + SNAPSHOT_W_EDGE * edge_score
                + SNAPSHOT_W_FACE * face_score)
    except Exception:
        return -1.0


def _collect_candidates(song, dancer_id: int, max_n: int = SNAPSHOT_MAX_CANDIDATES):
    """[(FramePose, joints)] list uniform in time with a valid pose."""
    poses = list(getattr(song, "poses", []) or [])
    if not poses:
        return []
    n = len(poses)
    step = max(1, n // max(1, int(max_n)))
    cands = []
    for i in range(0, n, step):
        try:
            d = poses[i].get_dancer(dancer_id)
        except Exception:
            continue
        if d is None:
            continue
        joints = d.raw or d.joints
        if not joints:
            continue
        try:
            nvis = sum(1 for a in joints.values()
                       if (float(a[3]) if len(a) > 3 else 1.0) >= SNAPSHOT_VIS_THRESH)
        except Exception:
            nvis = 0
        if nvis < SNAPSHOT_MIN_JOINTS:
            continue
        others = []
        try:
            for dd in (poses[i].dancers or []):
                if dd.dancer_id != dancer_id and (dd.raw or dd.joints):
                    others.append(dd.raw or dd.joints)
        except Exception:
            others = []
        cands.append((poses[i], joints, others))
        if len(cands) >= max_n:
            break
    # ensure the center (usually a representative pose) when missing
    try:
        mid_fp, mid_j = _sample_dancer_pose(song, dancer_id)
        if mid_fp is not None and mid_j and not any(fp.time == mid_fp.time for fp, _, _ in cands):
            others = []
            try:
                for dd in (mid_fp.dancers or []):
                    if dd.dancer_id != dancer_id and (dd.raw or dd.joints):
                        others.append(dd.raw or dd.joints)
            except Exception:
                pass
            cands.append((mid_fp, mid_j, others))
    except Exception:
        pass
    cands.sort(key=lambda c: float(c[0].time))
    return cands[:max_n]


def _read_frames_for_times(video_path: Path, times: list[float]):
    """{time: frame} with a single open (exact seek by time)."""
    import cv2

    out: dict = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out
    try:
        for t in sorted(set(float(x) for x in times)):
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 0.6) * 1000.0)
                frame = None
                for _ in range(90):
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    try:
                        pos = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    except Exception:
                        pos = t
                    frame = fr
                    if pos >= t - 0.03:
                        break
                if frame is not None:
                    out[t] = frame
            except Exception:
                continue
    finally:
        cap.release()
    return out


def _maybe_rotate_to_cover(frame, video_path: Path, song_dir: Path | None):
    """Rotates the frame to match the cover.png orientation.

    song.mp4 is copied unrotated but cover.png is saved rotated
    (make_cover with rotation); when their orientations differ, the import
    used 90/270 and poses live in rotated space -> rotate 90 CW.
    UI-only heuristic (no metadata in song.json).
    """
    if frame is None or song_dir is None:
        return frame
    try:
        import cv2

        cover = song_dir / "cover.png"
        if not cover.exists():
            return frame
        cimg = cv2.imread(str(cover))
        if cimg is None or not cimg.size:
            return frame
        ch, cw = cimg.shape[:2]
        fh, fw = frame.shape[:2]
        if _is_portrait_wh(cw, ch) != _is_portrait_wh(fw, fh):
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    except Exception:
        pass
    return frame


def _sample_dancer_pose(song, dancer_id: int):
    """(FramePose, joints) of the dancer on a middle frame where they appear.

    Robust: searches from the center outward for the first frame with joints.
    """
    poses = getattr(song, "poses", []) or []
    if not poses:
        return None, None
    mid = len(poses) // 2
    order = sorted(range(len(poses)), key=lambda i: abs(i - mid))
    for i in order:
        try:
            d = poses[i].get_dancer(dancer_id)
        except Exception:
            continue
        if d is None:
            continue
        joints = d.raw or d.joints
        if joints:
            return poses[i], joints
    return None, None


def _read_frame_at(video_path: Path, time_s: float):
    """Video frame at time_s with exact read (keyframe tolerant).

    Steps back ~0.6s and reads forward to the requested time, like
    the comparison, so image and joints come from the SAME instant.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        t = max(0.0, float(time_s))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 0.6) * 1000.0)
        frame = None
        for _ in range(90):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            try:
                pos = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            except Exception:
                pos = t
            frame = fr
            if pos >= t - 0.03:
                break
        return frame
    except Exception:
        return None
    finally:
        cap.release()


def _render_card(base, joints, dancer_id: int, size: int, vertical: bool):
    """Crops to the dancer, resizes and draws box+skeleton. Returns QPixmap."""
    import cv2

    H, W = base.shape[:2]
    pts, _ = _joints_points(joints, W, H, vis_thresh=0.3)
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if vertical:
        # Portrait crop: do not force square (cuts head/feet on 9:16).
        cw = bw * (1.0 + SNAPSHOT_MARGIN_X * 2.0)
        ch = bh * (1.0 + SNAPSHOT_MARGIN_Y * 2.0)
        cw = min(max(cw, 32.0), float(W))
        ch = min(max(ch, 32.0), float(H))
        rx0 = max(0.0, min(float(W) - cw, cx - cw / 2.0))
        ry0 = max(0.0, min(float(H) - ch, cy - ch / 2.0))
        crop = base[int(ry0):int(ry0 + ch), int(rx0):int(rx0 + cw)]
        if crop.size == 0:
            return None
        tw, th = size, int(round(size * 1.25))
        crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        scx, scy, ox, oy = tw / cw, th / ch, rx0, ry0
    else:
        pad = 0.18
        side = max(bw, bh) * (1.0 + pad * 2.0)
        side = min(side, float(min(W, H)))
        side = max(side, 32.0)
        rx0 = max(0.0, min(float(W) - side, cx - side / 2.0))
        ry0 = max(0.0, min(float(H) - side, cy - side / 2.0))
        crop = base[int(ry0):int(ry0 + side), int(rx0):int(rx0 + side)]
        if crop.size == 0:
            return None
        tw = th = size
        crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        scx = scy = float(tw) / float(side)
        ox, oy = rx0, ry0
        cw = ch = side

    def _pt(arr):
        return (max(0, min(tw - 1, int((float(arr[0]) * W - ox) * scx))),
                max(0, min(th - 1, int((float(arr[1]) * H - oy) * scy))))

    color = dancer_color_bgr(dancer_id)
    # Exact box of the dancer + D{n} label in its color.
    _cv_box0 = (max(0, min(tw - 1, int((min(xs) - ox) * scx))),
                max(0, min(th - 1, int((min(ys) - oy) * scy))))
    _cv_box1 = (max(0, min(tw - 1, int((max(xs) - ox) * scx))),
                max(0, min(th - 1, int((max(ys) - oy) * scy))))
    cv2.rectangle(crop, _cv_box0, _cv_box1, color, 2)
    cv2.putText(crop, f"D{int(dancer_id) + 1}", (_cv_box0[0], max(12, _cv_box0[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for a, b in POSE_CONNECTIONS:
        ja, jb = joints.get(a), joints.get(b)
        if ja is None or jb is None:
            continue
        try:
            va = float(ja[3]) if len(ja) > 3 else 1.0
            vb = float(jb[3]) if len(jb) > 3 else 1.0
        except Exception:
            continue
        if va < 0.3 or vb < 0.3:
            continue
        cv2.line(crop, _pt(ja), _pt(jb), color, 2, cv2.LINE_AA)
    for arr in joints.values():
        try:
            v = float(arr[3]) if len(arr) > 3 else 1.0
        except Exception:
            continue
        if v < 0.3:
            continue
        c = _pt(arr)
        cv2.circle(crop, c, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(crop, c, 3, color, 1, cv2.LINE_AA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy())


def dancer_card_pixmap(song, dancer_id: int, size: int = 200):
    """QPixmap with the dancer's BEST photo + skeleton in its color.

    Scores up to SNAPSHOT_MAX_CANDIDATES uniform frames with the
    overlap/visibility/sharpness/size/edge/face(head) score and keeps the
    best (on vertical uses a portrait crop + adapted score). Image and joints
    come from the SAME instant. Fallback: middle frame then cover.png.
    Color = the single process palette.
    """
    try:
        import cv2
    except Exception:
        return None
    try:
        key = (str(getattr(song, "path", "") or ""), int(dancer_id), int(size))
        cached = _CARD_CACHE.get(key)
        if cached is not None and not cached.isNull():
            return cached
        song_dir = Path(song.path) if getattr(song, "path", None) else None
        video = song_dir / "song.mp4" if song_dir is not None else None
        if video is None or not video.exists():
            prev = song_dir / "preview_pose.mp4" if song_dir is not None else None
            video = prev if (prev is not None and prev.exists()) else None
        # --- multi-candidate path (video available) ---
        if video is not None:
            cands = _collect_candidates(song, dancer_id)
            if cands:
                frames = _read_frames_for_times(video, [float(fp.time) for fp, _, _ in cands])
                best = None  # (score, frame_rot, joints)
                for fp, joints, others in cands:
                    fr = frames.get(float(fp.time))
                    if fr is None:
                        continue
                    fr = _maybe_rotate_to_cover(fr, video, song_dir)
                    H, W = fr.shape[:2]
                    vertical = _is_portrait_wh(W, H)
                    # downscale only for the Laplacian on very large frames
                    probe = fr
                    try:
                        longest = max(H, W)
                        if longest > 960:
                            s = 960.0 / float(longest)
                            probe = cv2.resize(fr, (int(W * s), int(H * s)),
                                               interpolation=cv2.INTER_AREA)
                            pH, pW = probe.shape[:2]
                        else:
                            pH, pW = H, W
                    except Exception:
                        probe, pH, pW = fr, H, W
                    sc = _score_candidate(probe, joints, others, pW, pH, vertical)
                    if sc < 0:
                        continue
                    if best is None or sc > best[0]:
                        best = (sc, fr, joints)
                if best is not None:
                    _, bfr, bjoints = best
                    BH, BW = bfr.shape[:2]
                    pix = _render_card(bfr, bjoints, dancer_id, size,
                                       _is_portrait_wh(BW, BH))
                    if pix is not None and not pix.isNull():
                        _CARD_CACHE[key] = pix
                        return pix
            # fallback single: frame central
            pose_fp, joints = _sample_dancer_pose(song, dancer_id)
            if joints is not None and pose_fp is not None:
                base = _read_frame_at(video, float(pose_fp.time))
                base = _maybe_rotate_to_cover(base, video, song_dir)
                if base is not None:
                    BH, BW = base.shape[:2]
                    pix = _render_card(base, joints, dancer_id, size,
                                       _is_portrait_wh(BW, BH))
                    if pix is not None and not pix.isNull():
                        _CARD_CACHE[key] = pix
                        return pix
        if song_dir is not None:
            # Fallback: cover.png IS the first frame -> pose at t=0.
            cover = song_dir / "cover.png"
            if cover.exists():
                base = cv2.imread(str(cover))
                poses = getattr(song, "poses", []) or []
                joints = None
                if poses:
                    try:
                        d0 = poses[0].get_dancer(dancer_id)
                    except Exception:
                        d0 = None
                    if d0 is not None:
                        joints = d0.raw or d0.joints
                if not joints:
                    _, joints = _sample_dancer_pose(song, dancer_id)
                if base is not None and joints:
                    BH, BW = base.shape[:2]
                    pix = _render_card(base, joints, dancer_id, size,
                                       _is_portrait_wh(BW, BH))
                    if pix is not None and not pix.isNull():
                        _CARD_CACHE[key] = pix
                        return pix
        return None
    except Exception:
        return None


class DancerSelectView(QWidget):
    back_requested = Signal()
    finished = Signal(object)  # PlayerManager

    def __init__(self, player_manager, parent=None) -> None:
        super().__init__(parent)
        self.player_manager = player_manager
        self.song = None
        self._lang = "en"
        self._current_idx = 0  # index of the player currently choosing

        self.title = QLabel("")
        self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignCenter)

        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)

        self.flow = FlowLayout(margin=4, hspacing=10, vspacing=10, center=True)
        self.grid_widget = QWidget()
        self.grid_widget.setLayout(self.flow)
        grid_scroll = scrollable(self.grid_widget, self)
        grid_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setAlignment(Qt.AlignCenter)

        self.summary = QLabel("")
        self.summary.setObjectName("ok")
        self.summary.setAlignment(Qt.AlignCenter)
        self.summary.setWordWrap(True)

        self.auto_btn = QPushButton("")
        set_button_icon(self.auto_btn, "shuffle", 16)
        self.auto_btn.clicked.connect(self._auto_assign)
        self.confirm_btn = QPushButton("")
        self.confirm_btn.setObjectName("primary")
        set_button_icon(self.confirm_btn, "play", 18, COLORS["ink"])
        self.confirm_btn.clicked.connect(self._confirm)
        self.back_btn = QPushButton("")
        set_button_icon(self.back_btn, "arrow-left", 16)
        self.back_btn.clicked.connect(self.back_requested)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(self.title)
        lay.addWidget(self.hint)
        lay.addWidget(grid_scroll, 1)
        lay.addWidget(self.status)
        lay.addWidget(self.summary)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.auto_btn)
        btn_row.addWidget(self.confirm_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._buttons: list[QPushButton] = []
        self._cards: list[QFrame] = []
        self.retranslate("en")

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.auto_btn.setText(tr(code, "dancer.auto"))
        self.auto_btn.setToolTip(tr(code, "dancer.auto_tooltip"))
        self.confirm_btn.setText(tr(code, "dancer.confirm"))
        self.back_btn.setText(tr(code, "dancer.back"))
        if self.song is not None:
            try:
                n = self.song.num_dancers if self.song else 1
                self.hint.setText(tr(code, "dancer.hint", n=n, plural="es" if (code == "es" and n != 1) else ("s" if code != "es" and n != 1 else "")))
            except Exception:
                pass
        try:
            self._refresh()
        except Exception:
            pass

    def begin(self, song) -> None:
        self.song = song
        self._current_idx = 0
        # when only 1 dancer with repeats allowed, auto-assign to all and skip? Still let them choose
        self._build_grid()
        self._refresh()

    def _build_grid(self) -> None:
        # clear
        self.flow.clear()
        self._buttons.clear()
        self._cards.clear()
        n = self.song.num_dancers if self.song else 1
        try:
            vertical = _song_is_vertical(self.song)
        except Exception:
            vertical = False
        card_w, card_h = (210, 310) if vertical else (210, 270)
        img_w, img_h = (190, 240) if vertical else (190, 190)
        for dancer_id in range(n):
            card = QFrame()
            card.setObjectName("card")
            card.setMinimumSize(card_w, card_h)
            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(8, 8, 8, 8)
            vbox.setSpacing(6)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setFixedSize(img_w, img_h)
            img_label.setStyleSheet("background:#000; border-radius:8px;")
            pix = dancer_card_pixmap(self.song, dancer_id, size=200)
            if pix is not None and not pix.isNull():
                img_label.setPixmap(pix.scaled(img_w, img_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img_label.setText(f"D{dancer_id + 1}")
                img_label.setStyleSheet("background:#000; border-radius:8px; font-size:48px; font-weight:800; color:#555;")
            btn = QPushButton(tr(self._lang, "dancer.card", n=dancer_id + 1))
            set_button_icon(btn, "person-standing", 16)
            btn.clicked.connect(lambda checked=False, did=dancer_id: self._pick(did))
            vbox.addWidget(img_label, 0, Qt.AlignCenter)
            vbox.addWidget(btn)
            self._buttons.append(btn)
            self._cards.append(card)
            self.flow.addWidget(card)
        # centered grid hint
        try:
            plural = "es" if (self._lang == "es" and n != 1) else ("s" if self._lang != "es" and n != 1 else "")
        except Exception:
            plural = ""
        self.hint.setText(tr(self._lang, "dancer.hint", n=n, plural=plural))

    def _current_player(self):
        if not self.player_manager.players:
            return None
        if self._current_idx >= len(self.player_manager.players):
            return None
        return self.player_manager.players[self._current_idx]

    def _refresh(self) -> None:
        p = self._current_player()
        code = self._lang
        # Pose-only: do not block on missing video, only logical assignment
        if p is None:
            # all assigned
            self.title.setText(tr(code, "dancer.ready_title"))
            self.status.setText(tr(code, "dancer.ready_status"))
            self.summary.setText(self._summary_text())
            self.confirm_btn.setEnabled(True)
            # disable dancer buttons but keep auto visible
            for b in self._buttons:
                b.setEnabled(False)
            self.auto_btn.setEnabled(False)
            return
        self.title.setText(tr(code, "dancer.pick_title", pid=p.player_id))
        try:
            n_dancers = self.song.num_dancers if self.song else 1
        except Exception:
            n_dancers = 1
        self.status.setText(tr(code, "dancer.pick_status", pid=p.player_id, tid=p.camera_track_id, n=n_dancers))
        self.summary.setText(self._summary_text())
        # allow confirm only when everyone is ready, but auto always available
        self.confirm_btn.setEnabled(self._all_assigned())
        self.auto_btn.setEnabled(True)
        for idx, (card, b) in enumerate(zip(self._cards, self._buttons)):
            # highlight the card when this player already holds that dancer
            if p.reference_dancer_id == idx:
                card.setObjectName("cardHi")
                b.setText(tr(code, "dancer.card_chosen", n=idx + 1))
            else:
                card.setObjectName("card")
                b.setText(tr(code, "dancer.card", n=idx + 1))
            card.style().unpolish(card)
            card.style().polish(card)
            b.setEnabled(True)

    def _pick(self, dancer_id: int) -> None:
        p = self._current_player()
        if p is None:
            return
        self.player_manager.assign_dancer(p.player_id, dancer_id)
        self.status.setText(tr(self._lang, "dancer.picked_status", pid=p.player_id, did=dancer_id + 1))
        # advance to the next player who has not chosen yet
        self._advance()
        self._refresh()

    def _all_assigned(self) -> bool:
        return all(pl.reference_dancer_id is not None for pl in self.player_manager.players)

    def _auto_assign(self) -> None:
        n = self.song.num_dancers if self.song else 1
        n = max(1, int(n))
        for idx, pl in enumerate(self.player_manager.players):
            did = idx % n
            self.player_manager.assign_dancer(pl.player_id, did)
        self._current_idx = len(self.player_manager.players)
        self._refresh()

    def _advance(self) -> None:
        # find the next player without an assigned dancer; allow re-picking via the summary
        for i, pl in enumerate(self.player_manager.players):
            if pl.reference_dancer_id is None:
                self._current_idx = i
                return
        # everyone assigned
        self._current_idx = len(self.player_manager.players)

    def _summary_text(self) -> str:
        if not self.player_manager.players:
            return ""
        code = self._lang
        lines = []
        for p in self.player_manager.players:
            if p.reference_dancer_id is not None:
                lines.append(tr(code, "dancer.summary_line", pid=p.player_id, tid=p.camera_track_id, did=p.reference_dancer_id + 1))
            else:
                who = tr(code, "dancer.pick_title", pid=p.player_id).split(" — ")[0]
                lines.append(f"{who} (Track {p.camera_track_id}) → " + tr(code, "dancer.summary_none"))
        return "\n".join(lines)

    def _confirm(self) -> None:
        # validate everyone has a dancer
        for p in self.player_manager.players:
            if p.reference_dancer_id is None:
                self.status.setText(tr(self._lang, "dancer.missing", pid=p.player_id))
                return
        self.finished.emit(self.player_manager)
