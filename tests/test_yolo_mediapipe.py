"""YOLO-boxes + per-person MediaPipe backend tests (with fakes, no models)."""

import numpy as np

from engine.pose_engine.skeleton import Joint, Pose
from engine.pose_engine.backends.yolo_mediapipe import YoloMediapipeBackend


def _pose_at(x, y, vis=0.9):
    p = Pose()
    p.joints["head"] = Joint(x=x, y=y, z=0.0, visibility=vis)
    p.joints["left_shoulder"] = Joint(x=x - 0.05, y=y + 0.1, z=0.0, visibility=vis)
    return p


class FakeMP:
    """Per-person estimator: returns a crop-centered pose."""

    instances = []

    def __init__(self):
        self.calls = 0
        FakeMP.instances.append(self)

    def detect(self, crop):
        self.calls += 1
        h, w = crop.shape[:2]
        # fixed point at the crop center
        return _pose_at(0.5, 0.4)

    def close(self):
        pass


class FakeYolo:
    """Fake YOLO: 2 boxes with stable track ids + fallback poses."""

    def __init__(self):
        self.n = 0

    def track_multi(self, frame):
        return [
            (1, (100, 100, 200, 400), _pose_at(0.2, 0.3, vis=0.8)),
            (2, (400, 100, 200, 400), _pose_at(0.7, 0.3, vis=0.8)),
        ]

    def detect_multi(self, frame):
        return []

    def reset_track(self):
        pass

    def close(self):
        pass


def _backend(yolo=None):
    FakeMP.instances.clear()
    b = YoloMediapipeBackend.__new__(YoloMediapipeBackend)
    # manual init without a real YOLO
    from engine.pose_engine.backends.yolo_mediapipe import YoloMediapipeBackend as B
    B.__init__(b, conf=0.25, max_persons=4, mp_factory=FakeMP, model_complexity=1)
    b._yolo = yolo or FakeYolo()
    return b


def test_un_estimador_por_track_y_coords_globales():
    b = _backend()
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    out = b.track_multi(frame)
    assert len(out) == 2
    tids = sorted(t for t, _, _ in out)
    assert tids == [1, 2]
    # one dedicated estimator per track
    assert len(FakeMP.instances) == 2
    # second frame reuses (creates no more)
    b.track_multi(frame)
    assert len(FakeMP.instances) == 2
    # global coords: track 1 head ~ its box center
    for tid, bbox, pose in out:
        head = pose.get("head")
        assert head is not None
        x, y, wb, hb = bbox
        if tid == 1:
            assert abs(head.x - (x + 0.5 * (wb * 1.24)) / 800.0) < 0.05
    b.close()


def test_fallback_yolo_si_mediapipe_falla():
    class EmptyMP(FakeMP):
        def detect(self, crop):
            return Pose()

    b = _backend()
    b._mp_factory = EmptyMP
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    out = b.track_multi(frame)
    # even when MediaPipe fails, every dancer has a pose via YOLO fallback
    assert len(out) == 2
    for _, _, pose in out:
        assert not pose.is_empty()
    b.close()


def test_sin_track_id_matchea_por_cercania():
    class NoIdYolo(FakeYolo):
        def track_multi(self, frame):
            return [
                (None, (102, 100, 200, 400), _pose_at(0.2, 0.3)),
                (None, (402, 100, 200, 400), _pose_at(0.7, 0.3)),
            ]

    b = _backend(yolo=NoIdYolo())
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    out = b.track_multi(frame)
    assert len(out) == 2
    # reuses nearby estimators instead of creating 2 new ones per frame
    n1 = len(FakeMP.instances)
    b.track_multi(frame)
    assert len(FakeMP.instances) == n1
    b.close()


def test_expira_estimadores_viejos():
    b = _backend()
    b.expire_frames = 2
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    b.track_multi(frame)
    assert len(b._estimators) == 2
    # YOLO stops seeing everyone -> expire after ttl
    b._yolo = FakeYoloEmpty()
    for _ in range(4):
        b.track_multi(frame)
    assert len(b._estimators) == 0
    b.close()


class FakeYoloEmpty:
    def track_multi(self, frame):
        return []

    def detect_multi(self, frame):
        return []

    def reset_track(self):
        pass

    def close(self):
        pass
