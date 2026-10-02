"""Pure-YOLO pipeline tests with 4 dancers (aespavideo.mp4). No network or models."""

import numpy as np

from engine.pose_engine.skeleton import Joint, Pose
from engine.pose_engine.multi import TrackRegistry, track_multi_poses, MultiBackendSet
from tools.pose_extractor.extract import ExtractOptions, _normalize_and_pack


def _pose(x=0.5, y=0.4, vis=0.35):
    # 6 joints with typical YOLO conf 0.35 (below the old 0.4 normalizer threshold)
    p = Pose()
    for name, dx, dy in [("head", 0, -0.2), ("left_shoulder", -0.05, 0),
                         ("right_shoulder", 0.05, 0), ("left_hip", -0.04, 0.25),
                         ("right_hip", 0.04, 0.25), ("left_knee", -0.04, 0.45)]:
        p.joints[name] = Joint(x=x + dx, y=y + dy, z=0.0, visibility=vis)
    return p


class FakeYolo4:
    def __init__(self):
        self.calls = 0

    def track_multi(self, frame):
        self.calls += 1
        p = _pose()
        return [(1, (10, 10, 80, 200), p), (2, (200, 10, 80, 200), p),
                (3, (390, 10, 80, 200), p), (4, (580, 10, 80, 200), p)]

    def reset_track(self):
        pass

    def close(self):
        pass


def test_track_4_dancers_ids_estables():
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    reg = TrackRegistry(max_slots=4)
    yb = FakeYolo4()
    out = track_multi_poses(frame, MultiBackendSet(mp_backend=None, yolo_backend=yb), reg)
    assert [d for d, _, _ in out] == [0, 1, 2, 3]
    # track 3 drops: the rest keep their dancer_id
    p = _pose()

    class Fake3(FakeYolo4):
        def track_multi(self, frame):
            return [(1, (10, 10, 80, 200), p), (2, (200, 10, 80, 200), p),
                    (4, (580, 10, 80, 200), p)]

    out2 = track_multi_poses(frame, MultiBackendSet(mp_backend=None, yolo_backend=Fake3()), reg)
    assert [d for d, _, _ in out2] == [0, 1, 3]
    # 3 returns: recovers dancer 2
    out3 = track_multi_poses(frame, MultiBackendSet(mp_backend=None, yolo_backend=yb), reg)
    assert [d for d, _, _ in out3] == [0, 1, 2, 3]


def test_normalize_yolo_tolera_conf_baja():
    tracked = [(0, (10, 10, 80, 200), _pose(vis=0.3)),
               (1, (200, 10, 80, 200), _pose(vis=0.3))]
    dancers = _normalize_and_pack(tracked, {}, {}, {}, vis_threshold=0.15)
    assert len(dancers) == 2
    # with the old 0.4 threshold they would be lost (documents the fixed bug)
    lost = _normalize_and_pack(tracked, {}, {}, {}, vis_threshold=0.4)
    assert len(lost) == 0


def test_extract_options_defaults_yolo():
    import inspect
    sig = inspect.signature(ExtractOptions)
    assert sig.parameters["estimator"].default == "yolo"
    assert sig.parameters["yolo_conf"].default == 0.4
