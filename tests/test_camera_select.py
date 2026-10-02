"""Camera selector in Settings: enumeration, combo, and fallback to camera 0."""

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeCap:
    """Fake VideoCapture: only OPEN_IDX indices open."""

    OPEN_IDX = {0, 2}

    def __init__(self, index, api=None):
        self._index = index
        self._opened = index in self.OPEN_IDX

    def isOpened(self):
        return self._opened

    def read(self):
        import numpy as np

        if not self._opened:
            return False, None
        return True, np.zeros((480, 640, 3), dtype="uint8")

    def get(self, _prop):
        return 640.0 if _prop == 3 else 480.0  # CAP_PROP_FRAME_WIDTH/HEIGHT

    def release(self):
        self._opened = False


def _install_fake_cv2(monkeypatch):
    fake = types.ModuleType("cv2")
    fake.CAP_DSHOW = 700
    fake.CAP_MSMF = 1400
    fake.CAP_PROP_FRAME_WIDTH = 3
    fake.CAP_PROP_FRAME_HEIGHT = 4
    fake.VideoCapture = lambda index, api=None: _FakeCap(index, api)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


def test_list_available_cameras_detecta_0_y_2(monkeypatch):
    _install_fake_cv2(monkeypatch)
    from engine.pose_engine.camera import list_available_cameras

    found = list_available_cameras(max_index=4)
    assert [i for i, _label in found] == [0, 2]
    assert all(label.startswith(f"Camara {i}") for i, label in found)


def test_list_available_cameras_sin_opencv(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from engine.pose_engine.camera import list_available_cameras

    assert list_available_cameras() == []


def test_open_camera_with_fallback_usa_0_si_wanted_falla(monkeypatch):
    import apps.desktop.threads as th

    calls = []

    class _FakeCam:
        def __init__(self, index, w, h):
            self.index = index
            calls.append(index)
            self.last_error = "No se pudo abrir la camara"

        def open(self):
            return self.index == 0

        def release(self):
            pass

    monkeypatch.setattr(th, "Camera", _FakeCam)
    cam, warning, error = th.open_camera_with_fallback(2, 640, 480)
    assert cam is not None and cam.index == 0
    assert error is None
    assert warning is not None and "2" in warning and "0" in warning
    assert calls == [2, 0]


def test_open_camera_with_fallback_directo_sin_warning(monkeypatch):
    import apps.desktop.threads as th

    class _FakeCam:
        def __init__(self, index, w, h):
            self.index = index
            self.last_error = None

        def open(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(th, "Camera", _FakeCam)
    cam, warning, error = th.open_camera_with_fallback(1, 640, 480)
    assert cam.index == 1 and warning is None and error is None


def test_open_camera_with_fallback_todo_falla(monkeypatch):
    import apps.desktop.threads as th

    class _FakeCam:
        def __init__(self, index, w, h):
            self.index = index
            self.last_error = f"falla {index}"

        def open(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(th, "Camera", _FakeCam)
    cam, warning, error = th.open_camera_with_fallback(3, 640, 480)
    assert cam is None and warning is None and error


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_settings_view_camera_combo_save_y_fill(monkeypatch, tmp_path):
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import SettingsView

    monkeypatch.setenv("APPDATA", str(tmp_path))
    _app()
    s = Settings(camera_index=1)
    view = SettingsView(s)
    try:
        view._fill_cameras([(0, "Camara 0"), (1, "Camara 1")], 1)
        assert view.camera_combo.findData(1) >= 0
        assert view.selected_camera_index() == 1
        # switch to 0 and save persists
        view.camera_combo.setCurrentIndex(view.camera_combo.findData(0))
        view._save()
        assert Settings.load().camera_index == 0
    finally:
        view._stop_scan()


def test_settings_view_preserva_guardada_no_detectada(monkeypatch, tmp_path):
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import SettingsView

    monkeypatch.setenv("APPDATA", str(tmp_path))
    _app()
    view = SettingsView(Settings(camera_index=3))
    try:
        view._fill_cameras([], 3)
        # even when undetected, the saved value stays selectable
        assert view.camera_combo.findData(3) >= 0
        assert view.selected_camera_index() == 3
        view._save()
        assert Settings.load().camera_index == 3
    finally:
        view._stop_scan()


def test_refresh_cameras_async_rellena_sin_bloquear(monkeypatch, tmp_path):
    """Scan runs in the background: the button disables and on completion
    the combo fills and the button re-enables."""
    from PySide6.QtCore import QEventLoop, QTimer

    import engine.pose_engine as ep
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import SettingsView

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(ep, "list_available_cameras",
                        lambda *a, **k: [(0, "Camara 0"), (1, "Camara 1")])
    _app()
    view = SettingsView(Settings(camera_index=1))
    try:
        view._refresh_cameras()
        assert not view.camera_refresh_btn.isEnabled()  # scanning
        loop = QEventLoop()
        try:
            view._scan_thread.done.connect(loop.quit)
        except Exception:
            pass
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        assert view.camera_refresh_btn.isEnabled()
        assert view.camera_combo.findData(1) >= 0
        assert view.selected_camera_index() == 1
    finally:
        view._stop_scan()


def test_device_warning_fallback_se_traduce():
    from apps.desktop.ui.views import MainWindow

    assert MainWindow._tr_device_msg(
        "La camara 2 no abrio; usando camara 0.", "en"
    ) == "Camera 2 did not open; using camera 0."
    assert MainWindow._tr_device_msg(
        "La camara 2 no abrio; usando camara 0.", "es"
    ) == "La camara 2 no abrio; usando camara 0."
