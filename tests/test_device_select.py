"""Inference device: auto->cuda/cpu, cuda->cpu cascade, honest reporting."""

import sys

from engine.pose_engine.backends.yolo_pose import half_for_device, pick_device


def _no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


def _yes_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)


def test_pick_device_auto_usa_cuda_si_hay(monkeypatch):
    _yes_cuda(monkeypatch)
    assert pick_device("auto") == "cuda"
    assert pick_device(None) == "cuda"
    assert pick_device("") == "cuda"


def test_pick_device_auto_cae_a_cpu_sin_gpu(monkeypatch):
    _no_cuda(monkeypatch)
    assert pick_device("auto") == "cpu"


def test_pick_device_explicito_pasa_tal_cual(monkeypatch):
    _no_cuda(monkeypatch)
    assert pick_device("cpu") == "cpu"
    # explicit "cuda" without GPU is honored here; the CPU cascade happens in
    # create_live_yolo_backend after health_check.
    assert pick_device("cuda") == "cuda"


def test_pick_device_sin_torch_cae_a_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert pick_device("auto") == "cpu"


def test_half_for_device():
    assert half_for_device("cuda") is True
    assert half_for_device("cuda:0") is True
    assert half_for_device("cpu") is False
    assert half_for_device("auto") is False
    assert half_for_device(None) is False


def test_settings_device_default_y_normalizacion(monkeypatch, tmp_path):
    from apps.desktop.settings import Settings

    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert Settings().device == "auto"
    s = Settings(device="CUDA")
    s.save()
    assert Settings.load().device == "cuda"
    s2 = Settings(device="tpu")
    s2.save()
    assert Settings.load().device == "auto"


def test_gpu_summary_honesto(monkeypatch):
    import engine.pose_engine.hardware as hw

    monkeypatch.setattr(hw, "detect_gpus", lambda: ["RTX 5050"])
    monkeypatch.setattr(hw, "torch_cuda_available", lambda: True)
    assert hw.gpu_summary() == "GPU: RTX 5050 | inferencia: cuda (auto)"
    assert hw.gpu_summary("yolo11n-pose · cuda") == (
        "GPU: RTX 5050 | inferencia: yolo11n-pose · cuda")
    monkeypatch.setattr(hw, "detect_gpus", lambda: [])
    monkeypatch.setattr(hw, "torch_cuda_available", lambda: False)
    assert hw.gpu_summary() == "GPU: no detectada | inferencia: CPU"


class _FakeBackend:
    def __init__(self, device, healthy=True):
        self.device = device
        self.model_name = "yolo11n-pose.pt"
        self._healthy = healthy
        self.closed = False

    def health_check(self):
        if not self._healthy:
            raise RuntimeError("CUDA roto")

    def close(self):
        self.closed = True


def test_live_yolo_ok_sin_warning(monkeypatch):
    import engine.pose_engine as ep
    from apps.desktop.threads import create_live_yolo_backend

    seen = []
    monkeypatch.setattr(ep, "create_backend",
                        lambda name, **kw: seen.append(kw.get("device")) or _FakeBackend("cuda"))
    be, warning = create_live_yolo_backend("auto")
    assert warning is None
    assert be.device == "cuda"
    assert seen == ["auto"]


def test_live_yolo_cascada_cuda_a_cpu(monkeypatch):
    import engine.pose_engine as ep
    from apps.desktop.threads import create_live_yolo_backend

    seen = []

    def factory(name, **kw):
        seen.append(kw.get("device"))
        dev = "cuda" if kw.get("device") == "auto" else kw.get("device")
        return _FakeBackend(dev, healthy=(dev == "cpu"))

    monkeypatch.setattr(ep, "create_backend", factory)
    be, warning = create_live_yolo_backend("auto")
    assert be is not None and be.device == "cpu"
    assert warning is not None and "CPU" in warning
    assert seen == ["auto", "cpu"]


def test_live_yolo_todo_falla_da_none_y_motivo(monkeypatch):
    import engine.pose_engine as ep
    from apps.desktop.threads import create_live_yolo_backend

    calls = []

    def factory(name, **kw):
        calls.append(kw.get("device"))
        return _FakeBackend(kw.get("device"), healthy=False)

    monkeypatch.setattr(ep, "create_backend", factory)
    be, warning = create_live_yolo_backend("cuda")
    assert be is None and warning
    assert calls == ["cuda", "cpu"]


def test_live_yolo_create_lanza_da_none(monkeypatch):
    import engine.pose_engine as ep
    from apps.desktop.threads import create_live_yolo_backend

    def boom(name, **kw):
        raise RuntimeError("sin pesos")

    monkeypatch.setattr(ep, "create_backend", boom)
    be, warning = create_live_yolo_backend("auto")
    assert be is None and "sin pesos" in warning


def test_live_backend_label():
    from apps.desktop.threads import live_backend_label

    assert live_backend_label(_FakeBackend("cuda")) == "yolo11n-pose · cuda"
    assert live_backend_label(None) == "yolo11n-pose"
