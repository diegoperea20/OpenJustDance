"""ByteTrack/BoT-SORT trackers: config resolution and warmup with fallback (no network)."""

from pathlib import Path

import numpy as np

from engine.pose_engine.backends.yolo_pose import YoloPoseBackend, resolve_tracker_cfg
from tools.pose_extractor.extract import ExtractOptions


def _tracker_dir() -> Path:
    return Path("engine/pose_engine/trackers".replace("/", __import__("os").sep))


def test_resolve_tuneados_del_repo():
    cfg = resolve_tracker_cfg("botsort-reid")
    assert cfg is not None and cfg.endswith("botsort-reid.yaml")
    assert Path(cfg).exists()
    cfg2 = resolve_tracker_cfg("bytetrack-dance")
    assert cfg2 is not None and cfg2.endswith("bytetrack-dance.yaml")
    assert Path(cfg2).exists()
    cfg3 = resolve_tracker_cfg("botsort-reid-gmc")
    assert cfg3 is not None and cfg3.endswith("botsort-reid-gmc.yaml")
    assert Path(cfg3).exists()


def test_resolve_builtins_y_none():
    assert resolve_tracker_cfg("botsort") == "botsort.yaml"
    assert resolve_tracker_cfg("bytetrack") == "bytetrack.yaml"
    assert resolve_tracker_cfg(None) is None
    assert resolve_tracker_cfg("") is None


def test_yamls_parsean_con_claves_esperadas():
    import yaml

    d = _tracker_dir()
    bt = yaml.safe_load((d / "botsort-reid.yaml").read_text(encoding="utf-8"))
    assert bt["tracker_type"] == "botsort"
    assert bt["with_reid"] is True
    assert bt["model"] == "yolo26n-reid.onnx"
    assert int(bt["track_buffer"]) >= 60
    yb = yaml.safe_load((d / "bytetrack-dance.yaml").read_text(encoding="utf-8"))
    assert yb["tracker_type"] == "bytetrack"
    assert int(yb["track_buffer"]) >= 60
    gmc = yaml.safe_load((d / "botsort-reid-gmc.yaml").read_text(encoding="utf-8"))
    assert gmc["tracker_type"] == "botsort"
    assert gmc["with_reid"] is True
    assert gmc["gmc_method"] == "sparseOptFlow"
    assert bt["gmc_method"] == "none"


class _StubModel:
    """Fake YOLO: fails on 'broken' cfgs, records track() kwargs."""

    def __init__(self, fail_cfgs=()):
        self.fail_cfgs = set(fail_cfgs)
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        cfg = kwargs.get("tracker")
        if cfg in self.fail_cfgs:
            raise RuntimeError(f"tracker roto: {cfg}")
        class _R:
            boxes = None
            keypoints = None
        return [_R()]


def _backend_stub(tracker, fail_cfgs=()):
    b = YoloPoseBackend.__new__(YoloPoseBackend)
    b.conf, b.iou, b.max_persons = 0.4, 0.5, 4
    b.min_box_w, b.min_box_h, b.min_kp_conf, b.min_joints = 20, 40, 0.15, 4
    b.tracker, b.tracker_cfg, b.tracker_error = tracker, None, None
    b.device = "cpu"
    b.quantize = None  # FP32 on CPU (mirrors YoloPoseBackend.__init__)
    b.tracker_device = "cpu"
    b._model = _StubModel(fail_cfgs=fail_cfgs)
    b._load_error = None
    return b


def test_warmup_usa_pedido_si_funciona():
    b = _backend_stub("bytetrack")
    b._warmup_tracker()
    assert b.tracker_cfg == "bytetrack.yaml"
    assert b.tracker == "bytetrack"
    assert b._model.calls and b._model.calls[0].get("tracker") == "bytetrack.yaml"


def test_warmup_fallback_si_reid_falla():
    # broken botsort-reid (e.g. no network to download ReID) -> bytetrack-dance.
    b = _backend_stub("botsort-reid",
                      fail_cfgs={resolve_tracker_cfg("botsort-reid")})
    b._warmup_tracker()
    assert b.tracker_cfg is not None and b.tracker_cfg.endswith("bytetrack-dance.yaml")
    assert b.tracker == "bytetrack-dance"


def test_warmup_cae_a_default_si_todo_falla():
    b = _backend_stub("botsort-reid", fail_cfgs="ALL")
    # fails everything yaml: stub fails on any non-None cfg
    orig_track = b._model.track

    def _track(frame, **kwargs):
        if kwargs.get("tracker") is not None:
            raise RuntimeError("todo roto")
        return orig_track(frame, **kwargs)

    b._model.track = _track
    b._warmup_tracker()
    assert b.tracker_cfg is None  # ultralytics default


def test_track_multi_pasa_tracker_resuelto():
    b = _backend_stub("bytetrack")
    b._warmup_tracker()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert b.track_multi(frame) == []  # stub with no detections
    # every real call carries persist=True + tracker
    assert b._model.calls
    for c in b._model.calls:
        assert c.get("persist") is True
    assert any(c.get("tracker") == "bytetrack.yaml" for c in b._model.calls)


def test_extract_options_default_tracker():
    import inspect
    sig = inspect.signature(ExtractOptions)
    assert sig.parameters["tracker"].default == "botsort-reid"


def test_pick_device():
    from engine.pose_engine.backends.yolo_pose import pick_device

    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda") == "cuda"
    assert pick_device("auto") in ("cpu", "cuda")
    assert pick_device(None) in ("cpu", "cuda")


def test_track_multi_pasa_device():
    b = _backend_stub("bytetrack")
    b._warmup_tracker()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    b.track_multi(frame)
    assert b._model.calls
    assert all(c.get("device") == "cpu" for c in b._model.calls)
    # constant quantize on every call: when it changes, ultralytics
    # recreates the predictor+tracker per frame (no ID persistence).
    assert {c.get("quantize") for c in b._model.calls} == {None}
