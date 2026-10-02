"""Vertical (9:16) vs horizontal camera comparison: comparison space.

Without correction, the SAME body gives ~0.65 error (= permanent Miss) from
the aspect baked in by the normalizer alone. Pre-scaling x into comparison
space (k = live/ref) before normalizing must give ~0 error
(exact: includes hip de-rotation).
"""
import json
from pathlib import Path

import pytest

from apps.desktop.controller import GameController
from apps.desktop.threads import SharedState
from engine.normalize_engine import BodyNormalizer
from engine.pose_engine.skeleton import Joint, Pose
from engine.score_engine import (
    PoseComparator,
    Scorer,
    compare_space_prescale,
    to_compare_space,
)
from shared.dance_format.loader import load_song, song_reference_aspect, video_aspect

REPO = Path(__file__).resolve().parents[1]
# Vertical reference clip (720x1280) + its extracted song. Both are local
# untracked media: skip instead of failing when they are absent.
VERT_VIDEO = REPO / "README-assets" / "newjeansshort.mp4"  # 720x1280
VERT_SONG_DIR = REPO / "songs" / "videotest1short"
VERT_SONG = VERT_SONG_DIR / "song.json"

needs_media = pytest.mark.skipif(
    not (VERT_VIDEO.is_file() and VERT_SONG.is_file()),
    reason="local untracked media missing (newjeansshort.mp4 / songs/videotest1short)",
)

# Vertical reference 720x1280; simulated live 640x480 with the SAME body.
W_V, H_V = 720, 1280
W_L, H_L = 640, 480
AR_V = W_V / H_V
AR_L = W_L / H_L


def _load_ref():
    d = json.loads(VERT_SONG.read_text(encoding="utf-8"))
    ref = next(p for p in d["poses"] if p.get("joints") and len(p["joints"]) > 10)
    return Pose.from_dict(ref["joints"]), Pose.from_dict(ref["raw"])


def _live_raw_same_body(raw: Pose, w: int, h: int) -> Pose:
    """Same body in pixels, centered in a w×h frame (0..1 coords)."""
    pix = {n: (j.x * W_V, j.y * H_V, j.visibility) for n, j in raw.joints.items()}
    cx = sum(p[0] for p in pix.values()) / len(pix)
    cy = sum(p[1] for p in pix.values()) / len(pix)
    return Pose(joints={
        n: Joint(x=(x - cx + w / 2) / w, y=(y - cy + h / 2) / h, z=0.0, visibility=v)
        for n, (x, y, v) in pix.items()
    })


@needs_media
def test_bug_sin_correccion_mismo_cuerpo_da_miss():
    ref_n, raw = _load_ref()
    live_n = BodyNormalizer().normalize(_live_raw_same_body(raw, W_L, H_L))
    err = PoseComparator().compare(live_n, ref_n)["error"]
    assert err is not None and err > 0.3, f"se esperaba Miss sin corrección, error={err}"


@needs_media
def test_espacio_comparacion_mismo_cuerpo_da_perfect():
    # What the camera thread does: pre-scale live raw and normalize.
    ref_n, raw = _load_ref()
    k = compare_space_prescale(AR_V, AR_L)
    assert k != 1.0
    live_n = BodyNormalizer().normalize(
        to_compare_space(_live_raw_same_body(raw, W_L, H_L), k))
    ctrl = GameController(PoseComparator(), Scorer())
    res = ctrl.evaluate(live_n, ref_n)
    assert res["rating"] == "Perfect", res
    assert res["error"] < 0.05, res["error"]


@needs_media
def test_horizontal_inmutable_bit_a_bit():
    # k=1 with equal aspect: the legacy pipeline does not change a single bit.
    _, raw = _load_ref()
    live_raw = _live_raw_same_body(raw, W_L, H_L)
    k = compare_space_prescale(AR_L, AR_L)
    assert k == 1.0
    a = BodyNormalizer().normalize(live_raw)
    b = BodyNormalizer().normalize(to_compare_space(live_raw, k))
    assert to_compare_space(live_raw, k) is live_raw  # passthrough
    for name, j in a.joints.items():
        o = b.joints[name]
        assert (j.x, j.y, j.z) == (o.x, o.y, o.z)


def test_prescale_guards():
    assert compare_space_prescale(None, 1.7) == 1.0
    assert compare_space_prescale(0.56, None) == 1.0
    assert compare_space_prescale(0, 1.7) == 1.0
    assert compare_space_prescale(-1, 1.7) == 1.0
    assert compare_space_prescale("basura", 1.7) == 1.0
    assert compare_space_prescale(0.01, 1.7) == 1.0  # out of range
    assert compare_space_prescale(0.5625, 1.3333) > 2.0  # real vertical case
    assert to_compare_space(Pose(), 2.0).is_empty()
    assert to_compare_space(None, 2.0) is None


@needs_media
def test_video_aspect_real():
    assert abs(video_aspect(VERT_VIDEO) - 0.5625) < 0.01
    assert video_aspect(REPO / "no_existe.mp4") is None


@needs_media
def test_song_reference_aspect_fallback_probe():
    # new song.json stores the field; old ones fall back to probing song.mp4.
    # find_songs() sets song.path; simulated the same way here.
    song = load_song(VERT_SONG)
    song.path = VERT_SONG.parent
    assert abs(song.reference_aspect - 0.5625) < 0.01
    assert abs(song_reference_aspect(song) - 0.5625) < 0.01

    legacy = load_song(VERT_SONG)
    legacy.path = VERT_SONG.parent
    legacy.reference_aspect = None  # old song.json without the field
    assert abs(song_reference_aspect(legacy) - 0.5625) < 0.01

    song2 = load_song(VERT_SONG)  # no path -> None (legacy, no correction)
    song2.path = None
    song2.reference_aspect = None
    assert song_reference_aspect(song2) is None


def test_song_reference_aspect_roundtrip():
    from shared.dance_format.models import Song

    s = Song(title="t", artist="a", fps=30.0, duration=1.0, reference_aspect=0.5625)
    s2 = Song.from_dict(s.to_dict())
    assert abs(s2.reference_aspect - 0.5625) < 1e-9
    s3 = Song.from_dict({"title": "x"})  # old songs
    assert s3.reference_aspect is None


def test_shared_state_aspects():
    st = SharedState()
    assert st.frame_aspect() is None
    assert st.get_reference_aspect() is None
    st.set_frame_size(640, 480)
    assert abs(st.frame_aspect() - 640 / 480) < 1e-9
    st.set_frame_size(0, 0)  # invalid: keeps the previous one
    assert abs(st.frame_aspect() - 640 / 480) < 1e-9
    st.set_reference_aspect(0.5625)
    assert abs(st.get_reference_aspect() - 0.5625) < 1e-9
    st.set_reference_aspect(None)
    assert st.get_reference_aspect() is None
