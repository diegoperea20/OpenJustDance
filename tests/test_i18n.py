"""EN/ES language: key parity, persistence, and hot switching."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_key_parity_en_es():
    from apps.desktop.i18n import STRINGS

    assert set(STRINGS.keys()) == {"en", "es"}
    assert set(STRINGS["en"].keys()) == set(STRINGS["es"].keys())
    for key, text in STRINGS["en"].items():
        assert isinstance(text, str) and text, f"empty en:{key}"
        assert isinstance(STRINGS["es"][key], str) and STRINGS["es"][key], f"empty es:{key}"


def test_tr_fallback_and_format():
    from apps.desktop.i18n import tr

    assert tr("en", "menu.play") == "  Play"
    assert tr("es", "menu.play") == "  Jugar"
    # unknown language -> default English
    assert tr("fr", "menu.play") == "  Play"
    assert tr(None, "menu.play") == "  Play"
    # missing key -> the key
    assert tr("en", "no.such.key") == "no.such.key"
    # format with kwargs
    assert tr("en", "join.players", n=1, exp=4) == "Players: 1 / 4"
    assert tr("es", "join.players", n=1, exp=4) == "Jugadores: 1 / 4"


def test_settings_language_default_and_roundtrip(tmp_path, monkeypatch):
    from apps.desktop.settings import Settings

    monkeypatch.setenv("APPDATA", str(tmp_path))
    s = Settings()
    assert s.language == "en"  # English by default
    s.language = "es"
    s.save()
    assert Settings.load().language == "es"
    # invalid value on disk -> English
    import json

    path = Settings._config_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["language"] = "xx"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert Settings.load().language == "en"


def test_developer_mode_default_and_roundtrip(tmp_path, monkeypatch):
    from apps.desktop.settings import Settings

    monkeypatch.setenv("APPDATA", str(tmp_path))
    # Default: PiP layout (small camera overlay, big reference video)
    assert Settings().developer_mode is False
    s = Settings()
    s.developer_mode = True
    s.save()
    assert Settings.load().developer_mode is True


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_settings_view_developer_mode_checkbox(tmp_path, monkeypatch):
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import SettingsView

    monkeypatch.setenv("APPDATA", str(tmp_path))
    _app()
    view = SettingsView(Settings())
    assert view.dev_mode.isChecked() is False  # unchecked by default
    assert view.lbl_dev_mode.text() == "Developer game mode view"
    view.dev_mode.setChecked(True)
    view._save()
    assert Settings.load().developer_mode is True
    # retranslate keeps the checkbox value
    view.retranslate("es")
    assert view.lbl_dev_mode.text() == "Vista de desarrollador"
    assert view.dev_mode.isChecked() is True


def test_game_view_pip_and_legacy_layout():
    from apps.desktop.settings import Settings
    from apps.desktop.threads import SharedState
    from apps.desktop.ui.views import GameView

    _app()
    view = GameView(SharedState())
    # default = PiP: camera label floats over the reference video
    assert view._developer_mode is False
    assert view.camera_label.parent() is view.video_card
    assert view.camera_label.maximumWidth() <= 320
    # legacy developer view: camera back in the stacked container
    view.apply_layout_mode(True)
    assert view._developer_mode is True
    assert view.camera_label.parent() is view.cam_container
    assert view.camera_label.maximumWidth() > 320
    # back to PiP
    view.apply_layout_mode(False)
    assert view.camera_label.parent() is view.video_card
    del view


def test_settings_view_language_combo_and_save(tmp_path, monkeypatch):
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import SettingsView

    monkeypatch.setenv("APPDATA", str(tmp_path))
    _app()
    view = SettingsView(Settings())
    # combo with en/es, English by default
    assert view.language.count() == 2
    assert view.language.currentData() == "en"
    assert view.title_label.text() == "Settings"
    # saving in Spanish persists
    idx = view.language.findData("es")
    view.language.setCurrentIndex(idx)
    view._save()
    assert Settings.load().language == "es"


def test_main_window_apply_language_hot_switch():
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import MainWindow

    _app()
    win = MainWindow(Settings())
    try:
        win.apply_language("en")
        assert win.menu_view.play.text() == "  Play"
        assert win.menu_view.settings_btn.text() == "  Settings"
        win.apply_language("es")
        assert win.menu_view.play.text() == "  Jugar"
        assert win.menu_view.settings_btn.text() == "  Ajustes"
        assert win.calibration_view.title_label.text() == "Calibración"
        assert win.settings_view.title_label.text() == "Ajustes"
        # and back to English
        win.apply_language("en")
        assert win.menu_view.play.text() == "  Play"
        assert win.calibration_view.title_label.text() == "Calibration"
    finally:
        try:
            win.stop_camera()
        except Exception:
            pass
        win.close()


def test_worker_and_camera_msg_mapping():
    from apps.desktop.settings import Settings
    from apps.desktop.ui.views import SongSelectView

    _app()
    view = SongSelectView(Settings())
    view.settings.language = "en"
    assert view._tr_worker_msg("Extrayendo poses...") == "Extracting poses..."
    assert view._tr_worker_msg("Listo") == "Done"
    assert (
        view._tr_worker_msg("Se detectaron 2 estables (declarados 4)")
        == "Detected 2 stable (declared 4)"
    )
    view.settings.language = "es"
    assert view._tr_worker_msg("Extrayendo poses...") == "Extrayendo poses..."
