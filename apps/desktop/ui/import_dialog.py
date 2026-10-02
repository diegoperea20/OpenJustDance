"""Dialog to configure importing a video as a song."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from apps.desktop.i18n import normalize_lang, tr
from shared.dance_format.loader import find_songs


class ImportDialog(QDialog):
    def __init__(self, video: Path, songs_dir: Path, parent=None, language: str = "en") -> None:
        super().__init__(parent)
        self._lang = normalize_lang(language)
        _t = lambda key, **kw: tr(self._lang, key, **kw)
        self.setWindowTitle(_t("imp.title"))
        self.setModal(True)
        self.resize(420, 320)
        self.video = Path(video)
        self.songs_dir = Path(songs_dir)

        title = QLabel(f"Video: <b>{self.video.name}</b>")
        title.setWordWrap(True)
        title.setObjectName("hint")
        # Try to detect dimensions for the portrait/landscape hint
        self._video_hint = ""
        try:
            import cv2

            cap = cv2.VideoCapture(str(self.video))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            orient = "vertical (9:16)" if h > w else "horizontal (16:9)"
            self._video_hint = f"{w}x{h} · {orient}"
        except Exception:
            pass
        hint_label = QLabel(self._video_hint)
        hint_label.setObjectName("mono")
        hint_label.setVisible(bool(self._video_hint))

        form = QFormLayout()
        self.title_edit = QLineEdit(self.video.stem.replace("_", " ").strip())
        self.title_edit.setPlaceholderText(_t("imp.title_ph"))
        form.addRow(_t("imp.title_label"), self.title_edit)

        self.artist_edit = QLineEdit(_t("songs.unknown_artist"))
        self.artist_edit.setPlaceholderText(_t("imp.artist_ph"))
        form.addRow(_t("imp.artist_label"), self.artist_edit)

        self.dir_edit = QLineEdit(self.video.stem)
        self.dir_edit.setPlaceholderText(_t("imp.folder_ph"))
        form.addRow(_t("imp.folder_label"), self.dir_edit)
        self.dir_warning = QLabel("")
        self.dir_warning.setObjectName("warn")
        self.dir_warning.hide()
        form.addRow("", self.dir_warning)
        self.dir_edit.textChanged.connect(self._check_dir)

        self.rotation_combo = QComboBox()
        self.rotation_combo.addItem(_t("imp.rot_auto"), 0)
        self.rotation_combo.addItem(_t("imp.rot_0"), 0)
        self.rotation_combo.addItem(_t("imp.rot_90"), 90)
        self.rotation_combo.addItem(_t("imp.rot_180"), 180)
        self.rotation_combo.addItem(_t("imp.rot_270"), 270)
        form.addRow(_t("imp.rotation_label"), self.rotation_combo)
        rot_hint = QLabel(_t("imp.rot_hint"))
        rot_hint.setWordWrap(True)
        rot_hint.setObjectName("micro")
        form.addRow("", rot_hint)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0, 60)
        self.fps_spin.setValue(0)
        self.fps_spin.setSpecialValueText(_t("imp.fps_auto"))
        self.fps_spin.setSingleStep(1)
        form.addRow(_t("imp.fps_label"), self.fps_spin)

        self.num_dancers_spin = QSpinBox()
        self.num_dancers_spin.setRange(1, 4)
        self.num_dancers_spin.setValue(1)
        self.num_dancers_spin.setSingleStep(1)
        form.addRow(_t("imp.dancers_label"), self.num_dancers_spin)
        dancers_hint = QLabel(_t("imp.dancers_hint"))
        dancers_hint.setWordWrap(True)
        dancers_hint.setObjectName("micro")
        form.addRow("", dancers_hint)

        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(320, 1920)
        self.max_width_spin.setValue(1280)
        self.max_width_spin.setSingleStep(160)
        form.addRow(_t("imp.side_label"), self.max_width_spin)
        dim_hint = QLabel(_t("imp.dim_hint"))
        dim_hint.setWordWrap(True)
        dim_hint.setObjectName("micro")
        form.addRow("", dim_hint)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText(_t("imp.import_btn"))
        self.buttons.button(QDialogButtonBox.Cancel).setText(_t("imp.cancel_btn"))
        try:
            from apps.desktop.ui.icons import set_button_icon as _set_icon

            _set_icon(self.buttons.button(QDialogButtonBox.Ok), "upload", 16)
            _set_icon(self.buttons.button(QDialogButtonBox.Cancel), "x", 16)
        except Exception:
            pass
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(hint_label)
        lay.addLayout(form)
        lay.addWidget(self.buttons)
        self._check_dir()

    def _check_dir(self) -> None:
        slug = self.dir_edit.text().strip()
        if not slug:
            self.dir_warning.hide()
            return
        # sanitizar preview
        import re, unicodedata

        s = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
        if not s:
            s = "cancion"
        dest = self.songs_dir / s
        if dest.exists():
            self.dir_warning.setText(tr(self._lang, "imp.folder_exists", s=s))
            self.dir_warning.show()
        else:
            self.dir_warning.hide()

    def _on_accept(self) -> None:
        if not self.title_edit.text().strip():
            self.title_edit.setFocus()
            return
        self.accept()

    def get_data(self) -> dict:
        fps_val = self.fps_spin.value()
        fps = None if fps_val == 0 else float(fps_val)
        return {
            "title": self.title_edit.text().strip() or self.video.stem,
            "artist": self.artist_edit.text().strip() or tr(self._lang, "songs.unknown_artist"),
            "slug": self.dir_edit.text().strip() or self.video.stem,
            "fps": fps,
            "max_width": int(self.max_width_spin.value()),
            "rotation": int(self.rotation_combo.currentData()),
            "num_dancers": int(self.num_dancers_spin.value()),
        }
