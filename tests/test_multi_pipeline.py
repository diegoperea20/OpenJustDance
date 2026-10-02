"""YOLO-first multi pipeline and stable slots tests (no network or GUI)."""

from engine.pose_engine.multi import StableSlotMapper, dedup_boxes
from engine.pose_engine.skeleton import Joint, Pose


def test_dedup_fusiona_duplicado_hog():
    # Multi-scale HOG duplicates winStride 4,4 with IoU ~0.34 (aespavideo f90).
    a = (520, 292, 222, 428)
    b = (544, 464, 128, 256)
    assert len(dedup_boxes([a, b], iou_thresh=0.30)) == 1
    # Two distinct people are not merged.
    c = (100, 100, 120, 300)
    d = (400, 100, 120, 300)
    assert len(dedup_boxes([c, d], iou_thresh=0.30)) == 2


def test_slots_estables_ante_cruce():
    m = StableSlotMapper(max_slots=4)
    first = m.assign([(100, 0, 50, 100), (300, 0, 50, 100)])
    assert first[0][0] < first[1][0]
    # Small jitter: slots must not swap.
    second = m.assign([(102, 0, 50, 100), (298, 0, 50, 100)])
    assert second[0][0] < 200 and second[1][0] > 200


def test_detect_multi_yolo_first_usa_yolo():
    import numpy as np
    from engine.pose_engine.multi import detect_multi_poses, MultiBackendSet

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    made = []

    class FakeYolo:
        def detect_multi(self, f):
            made.append(True)
            p = Pose(joints={"head": Joint(0.5, 0.2, 0.0, 0.9)})
            return [((10, 10, 50, 100), p), ((200, 10, 50, 100), p)]

    class FakeMp:
        def detect(self, f):
            raise AssertionError("no debe usarse el fallback si YOLO da 2")

    out = detect_multi_poses(frame, MultiBackendSet(mp_backend=FakeMp(), yolo_backend=FakeYolo()), tracker=None)
    assert made and len(out) == 2
    # sorted by x
    assert out[0][0][0] < out[1][0][0]


def test_track_registry_no_permuta_al_perder_uno():
    # 4 tracks -> one drops (3) -> returns: existing dancers do not change.
    from engine.pose_engine.multi import TrackRegistry

    def _mk(tid, x):
        return (tid, (x, 10, 50, 100), Pose(joints={"head": Joint(0.5, 0.2, 0.0, 0.9)}))

    reg = TrackRegistry(max_slots=4)
    r1 = reg.update([_mk(1, 10), _mk(2, 200), _mk(3, 400), _mk(4, 600)])
    assert [d for d, _, _ in r1] == [0, 1, 2, 3]
    r2 = reg.update([_mk(1, 12), _mk(2, 202), _mk(4, 602)])      # track 3 occluded
    assert [(d) for d, _, _ in r2] == [0, 1, 3]
    r3 = reg.update([_mk(1, 14), _mk(2, 204), _mk(3, 402), _mk(4, 604)])
    assert [d for d, _, _ in r3] == [0, 1, 2, 3]


def test_track_registry_sin_ids_hereda_espacial():
    # Tracker still initializing (None ids): inherit by proximity, do not create new slots.
    from engine.pose_engine.multi import TrackRegistry

    reg = TrackRegistry(max_slots=4)
    p = Pose(joints={"head": Joint(0.5, 0.2, 0.0, 0.9)})
    reg.update([(1, (10, 10, 50, 100), p), (2, (300, 10, 50, 100), p)])
    r = reg.update([(None, (12, 10, 50, 100), p), (None, (302, 10, 50, 100), p)])
    assert sorted(d for d, _, _ in r) == [0, 1]


def test_track_multi_poses_usa_track_con_ids():
    import numpy as np
    from engine.pose_engine.multi import MultiBackendSet, TrackRegistry, track_multi_poses

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    p = Pose(joints={"head": Joint(0.5, 0.2, 0.0, 0.9)})

    class FakeYoloTrack:
        def __init__(self):
            self.calls = 0

        def track_multi(self, f):
            self.calls += 1
            # tracker order intentionally non-spatial (id 2 first)
            return [(2, (300, 10, 50, 100), p), (1, (10, 10, 50, 100), p)]

    yb = FakeYoloTrack()
    reg = TrackRegistry(max_slots=4)
    out = track_multi_poses(frame, MultiBackendSet(mp_backend=None, yolo_backend=yb), reg)
    assert yb.calls == 1
    # dancer 0 = track 1 (left), dancer 1 = track 2, even when the tracker returns them reversed
    assert [(d) for d, _, _ in out] == [0, 1]
    # second frame with only track 2: keeps dancer 1 (no reindex to 0)
    out2 = track_multi_poses(frame, MultiBackendSet(mp_backend=None, yolo_backend=FakeYoloTrack2(p)), reg)
    assert [d for d, _, _ in out2] == [1]


class FakeYoloTrack2:
    def __init__(self, p):
        self.p = p

    def track_multi(self, f):
        return [(2, (302, 10, 50, 100), self.p)]


def test_track_registry_expira_slots_muertos():
    # Without expiry, old IDs exhaust slots and new people are lost.
    from engine.pose_engine.multi import TrackRegistry

    def _mk(tid, x):
        return (tid, (x, 10, 50, 100), Pose(joints={"head": Joint(0.5, 0.2, 0.0, 0.9)}))

    reg = TrackRegistry(max_slots=2, max_missed=3)
    assert [d for d, _, _ in reg.update([_mk(1, 10), _mk(2, 300)])] == [0, 1]
    # track 1 disappears for 4 frames -> slot freed
    for _ in range(4):
        reg.update([_mk(2, 300)])
    assert 1 not in reg.track_to_dancer
    # new track takes the free slot
    assert [d for d, _, _ in reg.update([_mk(2, 300), _mk(3, 10)])] == [0, 1]
    # empty frames also age
    for _ in range(5):
        reg.update([])
    assert reg.track_to_dancer == {}


def test_detect_multi_fallback_sin_yolo():
    import numpy as np
    from engine.pose_engine.multi import detect_multi_poses, MultiBackendSet

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeMp:
        def detect(self, crop):
            return Pose(joints={"head": Joint(0.5, 0.5, 0.0, 0.9)})

    import engine.pose_engine.multi as M

    orig = M.detect_persons if hasattr(M, "detect_persons") else None
    # Local monkeypatch: detect_persons is imported inside the function,
    # so we patch engine.pose_engine.detector.detect_persons.
    import engine.pose_engine.detector as D

    old = D.detect_persons
    D.detect_persons = lambda *a, **k: [(10, 10, 60, 120, 1.0)]
    try:
        out = detect_multi_poses(frame, MultiBackendSet(mp_backend=FakeMp(), yolo_backend=None), tracker=None)
    finally:
        D.detect_persons = old
    assert len(out) == 1
    # global 0..1 coords
    _, pose = out[0]
    j = pose.get("head")
    assert j is not None and 0.0 <= j.x <= 1.0 and 0.0 <= j.y <= 1.0
