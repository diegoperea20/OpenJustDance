"""Regression: picking D1..D4 must compare against that single dancer (id2=dancer2).

Covers:
- dancer_pose_at() returns DISTINCT poses per dancer (scoring core).
- The results comparison uses the chosen dancer (previously always 1).
- DancerSelect builds an image card per dancer when a cover exists.
- assign_dancer persists the chosen dancer until play.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from engine.timing_engine import SongSynchronizer
from shared.dance_format.models import DancerPose, FramePose, Song


def _synth_song(n: int = 4) -> Song:
    poses = []
    for i in range(5):
        t = round(i * 0.033, 3)
        dancers = []
        for did in range(n):
            # each dancer with head at a distinct x -> distinguishable
            joints = {
                "head": [0.1 * (did + 1), 2.0, 0.0, 1.0],
                "left_shoulder": [-0.2, 1.0, 0.0, 1.0],
                "right_shoulder": [0.2, 1.0, 0.0, 1.0],
                "left_hip": [-0.1, 0.0, 0.0, 1.0],
                "right_hip": [0.1, 0.0, 0.0, 1.0],
            }
            dancers.append(DancerPose(dancer_id=did, joints=dict(joints),
                                      raw=dict(joints)))
        first = dancers[0]
        poses.append(FramePose(time=t, joints=dict(first.joints),
                               raw=dict(first.raw), dancers=dancers))
    return Song(title="t", artist="t", fps=30.0, duration=0.2,
                format="openjustdance-2", poses=poses, num_dancers=n)


def test_reference_distinct_per_dancer():
    song = _synth_song(4)
    sync = SongSynchronizer(song)
    heads = []
    for did in range(4):
        dp = sync.dancer_pose_at(0.05, did)
        assert dp is not None, f"dancer {did} sin pose"
        assert dp.dancer_id == did, "id único debe mapear a su dancer"
        heads.append(dp.joints["head"][0])
    assert len(set(heads)) == 4, f"D1..D4 deben diferir, salieron {heads}"


def test_legacy_fallback_single():
    song = _synth_song(1)
    song.poses[0].dancers = []  # legacy without list
    sync = SongSynchronizer(song)
    assert sync.dancer_pose_at(0.05, 0) is not None
    assert sync.dancer_pose_at(0.05, 2) is None  # no list means no dancer 2


def test_assign_dancer_persists():
    from engine.tracking.player_manager import PlayerManager

    pm = PlayerManager()
    pm.set_expected(1)
    p = pm.claim(1)
    assert pm.assign_dancer(p.player_id, 2)
    assert pm.get_by_player(p.player_id).reference_dancer_id == 2
    # the game uses that id for the reference (never reset to 0)
    assert pm.get_by_player(p.player_id).reference_dancer_id != 0


def test_comparison_uses_chosen_dancer():
    from PySide6.QtWidgets import QApplication

    from apps.desktop.ui.comparison import PoseComparisonWidget

    app = QApplication.instance() or QApplication([])
    w = PoseComparisonWidget()
    song = _synth_song(4)
    rec = [{"time": 0.05, "player": None, "rating": "—"}]
    w.set_data(song, rec, dancer_id=2)
    assert w._dancer_id == 2
    assert "3" in w.right_label.text()  # Bailarín 3
    w._update_for_time(0.05)
    rp = w.right_canvas._pose
    assert rp is not None and not rp.is_empty()
    # must be dancer 3's pose (head x=0.3), not dancer 1's (0.1)
    assert abs(rp.get("head").x - 0.3) < 1e-6


def test_palette_unique_per_dancer():
    from engine.pose_engine.skeleton import dancer_color_bgr

    colors = [dancer_color_bgr(i) for i in range(4)]
    assert len(set(colors)) == 4, "cada dancer un color único del proceso"
    # the comparison uses the same palette (BGR -> hex)
    from apps.desktop.ui.comparison import PoseComparisonWidget

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    w = PoseComparisonWidget()
    w.set_data(_synth_song(4), [{"time": 0.05, "player": None, "rating": "—"}],
               dancer_id=1)
    b, g, r = dancer_color_bgr(1)
    assert w.right_canvas._line_color.name() == f"#{r:02x}{g:02x}{b:02x}"


def test_comparison_frame_exact():
    """The video frame must be from the requested instant (no keyframe error)."""
    from PySide6.QtWidgets import QApplication

    from apps.desktop.ui.comparison import VideoSkeletonWidget

    app = QApplication.instance() or QApplication([])
    video = Path("songs/aespavideo/song.mp4")
    if not video.exists():
        return
    w = VideoSkeletonWidget()
    w.set_video(video)
    for t in (1.0, 5.0, 10.0):
        w.set_frame_at(t)
        assert w._qimg is not None and not w._qimg.isNull()
        assert w._last_t is not None and abs(w._last_t - t) < 0.3, (
            f"frame en {w._last_t}, pedido {t}: desync tipo keyframe")


def test_comparison_draws_all_dancers_selected_highlight():
    from PySide6.QtWidgets import QApplication

    from apps.desktop.ui.comparison import PoseComparisonWidget

    app = QApplication.instance() or QApplication([])
    w = PoseComparisonWidget()
    song = _synth_song(4)
    rec = [{"time": 0.05, "player": None, "rating": "—"}]
    w.set_data(song, rec, dancer_id=1)
    w._update_for_time(0.05)
    items = w.right_canvas._ref_items
    assert len(items) == 4, f"debe dibujar los 4 dancers, hay {len(items)}"
    assert w.right_canvas._selected_id == 1
    assert sorted(d for d, _ in items) == [0, 1, 2, 3]


def test_dancer_card_pixmap_real_song():
    from PySide6.QtWidgets import QApplication

    from apps.desktop.ui.dancer_select_view import dancer_card_pixmap
    from shared.dance_format.loader import load_song

    app = QApplication.instance() or QApplication([])
    song_path = Path("songs/aespavideo/song.json")
    if not song_path.exists():
        return  # no test song available, skipped
    song = load_song(song_path)
    song.path = song_path.parent
    for did in range(min(4, song.num_dancers)):
        pix = dancer_card_pixmap(song, did)
        assert pix is not None and not pix.isNull(), f"sin tarjeta para dancer {did}"
