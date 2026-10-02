"""Anti-occlusion: crossing, temporary cover and ByteTrack ID-switch (no network or models)."""

import numpy as np

from engine.pose_engine.skeleton import Joint, Pose
from engine.pose_engine.multi import TrackRegistry


def _pose(seed: float, vis: float = 0.9):
    # distinct poses per dancer (seed shifts x): simulates different bodies
    p = Pose()
    pts = {"head": (0.50 + seed, 0.20), "left_shoulder": (0.45 + seed, 0.30),
           "right_shoulder": (0.55 + seed, 0.30), "left_hip": (0.46 + seed, 0.55),
           "right_hip": (0.54 + seed, 0.55), "left_knee": (0.46 + seed, 0.75),
           "right_knee": (0.54 + seed, 0.75)}
    for name, (x, y) in pts.items():
        p.joints[name] = Joint(x=x, y=y, z=0.0, visibility=vis)
    return p


def _mk(tid, x, seed=0.0, w=60, h=200):
    return (tid, (x, 50, w, h), _pose(seed))


def test_cruce_gradual_mantiene_identidad():
    # A (tid 1) walks left to right and B (tid 2) the reverse, step by
    # step (like real video). Even when the x-order flips halfway, each
    # dancer_id must follow its person (velocity + pose continuity).
    reg = TrackRegistry(max_slots=2)
    xa, xb = 10, 300
    reg.update([_mk(1, xa, seed=0.0), _mk(2, xb, seed=0.2)])
    for _ in range(8):
        xa += 36  # A -> right
        xb -= 36  # B -> left
        r = reg.update([_mk(1, xa, seed=0.0), _mk(2, xb, seed=0.2)])
        got = {d: b for d, b, _ in r}
        assert got[0][0] == xa and got[1][0] == xb
    # at the end A is on the right and B on the left, no permutation
    assert xa > xb
    got = {d: b for d, b, _ in r}
    assert got[0][0] == xa and got[1][0] == xb


def test_merge_split_con_tids_cambiados():
    # Crossing with temporary merge: YOLO sees 1 box for 2 frames (A covers B) and
    # on split the tids arrive swapped. Distinct pose per dancer.
    reg = TrackRegistry(max_slots=2)
    reg.update([_mk(1, 10, seed=0.0), _mk(2, 300, seed=0.2)])
    # gradual approach
    reg.update([_mk(1, 100, seed=0.0), _mk(2, 220, seed=0.2)])
    # merge: only A visible in the middle (wide box)
    for _ in range(2):
        r = reg.update([_mk(1, 150, seed=0.0, w=120)])
        assert [d for d, _, _ in r] == [0]
    assert 1 in reg._st  # B survives as occluded
    # split already crossed, with swapped tids: A (seed 0) on the right
    # with tid 2, B (seed 0.2) on the left with tid 1
    r = reg.update([_mk(1, 100, seed=0.2), _mk(2, 220, seed=0.0)])
    got = {d: (b[0], b[2]) for d, b, _ in r}
    assert sorted(d for d, _, _ in r) == [0, 1]
    assert got[0][0] == 220 and got[1][0] == 100


def test_id_switch_tracker_se_corrige():
    # ByteTrack swaps IDs on occlusion (tid 1 <-> tid 2), but the
    # poses/bodies stay put: the registry must absorb the change
    # without permuting dancer_ids (thanks to pose+position continuity).
    reg = TrackRegistry(max_slots=2)
    reg.update([_mk(1, 10, seed=0.0), _mk(2, 300, seed=0.2)])
    # next frame: crossed IDs, same bboxes and poses
    r = reg.update([_mk(2, 10, seed=0.0), _mk(1, 300, seed=0.2)])
    got = {d: b for d, b, _ in r}
    assert got[0][0] == 10 and got[1][0] == 300
    # and the binding absorbed the new tids (stable third frame)
    r2 = reg.update([_mk(2, 12, seed=0.0), _mk(1, 302, seed=0.2)])
    got2 = {d: b for d, b, _ in r2}
    assert got2[0][0] == 12 and got2[1][0] == 302


def test_ocluido_conserva_slot_y_vuelve():
    # B (tid 2) stays covered by A: YOLO sees 1 detection only (A's, large).
    # B must keep dancer 1 and recover it on reappearance, no new slot.
    reg = TrackRegistry(max_slots=2)
    reg.update([_mk(1, 10, seed=0.0), _mk(2, 300, seed=0.2)])
    # 6 frames with A only (B covered): B must not expire (occ_ttl=45)
    for i in range(6):
        r = reg.update([_mk(1, 10 + i, seed=0.0, w=120, h=220)])
        assert [d for d, _, _ in r] == [0]
    assert 1 in reg._st  # B still alive as occluded
    # B reappears with a NEW track_id (ByteTrack recreates it): re-ID by pose
    r2 = reg.update([_mk(1, 10, seed=0.0), _mk(9, 300, seed=0.2)])
    assert sorted(d for d, _, _ in r2) == [0, 1]
    got = {d: b for d, b, _ in r2}
    assert got[1][0] == 300


def test_override_no_reescribe_memoria():
    # An override (global) match updates bbox/binding but freezes
    # velocity/appearance/pose: when the override was a ghost, clean
    # memory allows correction on following frames (anti lock-in).
    reg = TrackRegistry(max_slots=2)
    reg.update([_mk(1, 10, seed=0.0), _mk(2, 300, seed=0.2)])
    # merge: 1 det only (A in the middle) -> d0 takes it (legit direct match)
    reg.update([_mk(1, 150, seed=0.0, w=120)])
    pose0 = dict(reg._st[0]["pose"])
    vel0 = reg._st[0]["vel"]
    # split with swapped tids: enters via override (see test_merge_split)
    r = reg.update([_mk(1, 100, seed=0.2), _mk(2, 220, seed=0.0)])
    got = {d: b for d, b, _ in r}
    # the split with swapped tids enters via override (global wins with margin)
    assert got.get(0, (None,))[0] == 220
    # ...and d0 memory must stay intact (frozen, anti lock-in)
    assert reg._st[0]["vel"] == vel0
    assert reg._st[0]["pose"] == pose0
    assert any("frozen d0" in e for e in reg.debug_info.get("events", []))


def _hist_at(idx: int):
    import numpy as np

    h = np.zeros(64, dtype=np.float64)
    h[idx % 64] = 1.0
    return h


def _mkapp(tid, x, seed=0.0, app=None, w=60, h=200):
    from engine.pose_engine.multi import _pose_snapshot  # noqa

    tid_, bbox, pose = _mk(tid, x, seed, w, h)
    return (tid_, bbox, pose, app)


def test_rebind_une_ocluido_con_tid_persistente():
    # The tracker switches ID on the same person (flip-flop) and live memory
    # got poisoned (a ghost override rewrote it): Hungarian can no longer
    # match (cost > gate), but the ANCHOR can. The persistent tid
    # must re-bind via rebind in ~6 frames instead of leaking "drop w/o slot".
    red, blue = _hist_at(5), _hist_at(40)
    reg = TrackRegistry(max_slots=2)
    reg.update([_mkapp(1, 10, seed=0.0, app=blue), _mkapp(2, 300, seed=0.2, app=red)])
    # B disappears (A visible only): B occluded-retained
    for _ in range(3):
        reg.update([_mkapp(1, 10, seed=0.0, app=blue)])
    assert 1 in reg._st
    assert "_anchor_pose" in reg._st[1]
    # Poison B's live memory (another person's pose + blue appearance):
    # Hungarian will score high, the anchor (seed 0.2 + red) stays exact.
    reg._st[1]["pose"] = {k: (x + 0.7, y) for k, (x, y) in reg._st[1]["pose"].items()}
    reg._st[1]["app"] = blue
    # B reappears with a NEW tid in place, persistent
    rebound = None
    for i in range(8):
        r = reg.update([_mkapp(1, 10, seed=0.0, app=blue),
                        _mkapp(9, 300, seed=0.2, app=red)])
        if any(d == 1 for d, _, _ in r):
            rebound = i
            break
    assert rebound is not None and rebound <= 6
    got = {d: b for d, b, _ in r}
    assert got[1][0] == 300
    assert any("rebind d1" in e for e in reg.debug_info.get("events", []))


def test_anchor_se_restaura_en_revive():
    reg = TrackRegistry(max_slots=1, max_missed=2)
    reg.update([_mk(1, 10, seed=0.0)])
    assert "_anchor_pose" in reg._st[0]
    for _ in range(4):
        reg.update([])
    assert reg._st == {}  # expired -> gallery
    assert len(reg._lost) == 1
    assert "anchor_pose" in reg._lost[0]
    reg.update([_mk(9, 12, seed=0.0)])
    assert 0 in reg._st
    assert "_anchor_pose" in reg._st[0]


def test_frames_vacios_envejecen_y_expiran():
    reg = TrackRegistry(max_slots=2, max_missed=3)
    reg.update([_mk(1, 10), _mk(2, 300)])
    for _ in range(5):
        assert reg.update([]) == []
    assert reg._st == {}
    assert reg.track_to_dancer == {}


def test_apariencia_rompe_empate_espacial():
    # Two equidistant detections of two dancers: appearance (real frame)
    # decides. Dancer 0 = red torso, dancer 1 = blue torso.
    reg = TrackRegistry(max_slots=2)
    frame0 = np.zeros((300, 400, 3), dtype=np.uint8)
    # paint torsos: bbox0 x=10 (red torso BGR 0,0,255), bbox1 x=200 (blue 255,0,0)
    frame0[90:170, 19:61] = (0, 0, 255)
    frame0[90:170, 209:251] = (255, 0, 0)
    dets0 = [(1, (10, 50, 60, 200), _pose(0.0)), (2, (200, 50, 60, 200), _pose(0.0))]
    assert [d for d, _, _ in reg.update(dets0, frame=frame0)] == [0, 1]
    # next frame: both moved toward the center but torsos stay
    # pure (red left, blue right) and tids arrive crossed.
    # bbox A (95,...) -> torso x 104..146 (pintado rojo 104..140)
    # bbox B (135,...) -> torso x 144..186 (pintado azul 144..186)
    frame1 = np.zeros((300, 400, 3), dtype=np.uint8)
    frame1[90:170, 104:141] = (0, 0, 255)    # red
    frame1[90:170, 144:187] = (255, 0, 0)    # blue
    dets1 = [(2, (95, 50, 60, 200), _pose(0.0)), (1, (135, 50, 60, 200), _pose(0.0))]
    got = {d: b for d, b, _ in reg.update(dets1, frame=frame1)}
    # the red torso must stay dancer 0 even with tid 2
    assert got[0][0] == 95 and got[1][0] == 135
