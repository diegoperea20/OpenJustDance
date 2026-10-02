"""Main window and app views (menu, songs, calibration, game, results, settings)."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from engine.pose_engine import Pose
from engine.pose_engine.hardware import gpu_summary
from engine.score_engine import PoseComparator, Scorer
from engine.timing_engine import SongSynchronizer
from shared.dance_format import find_songs

from apps.desktop import settings as app_settings
from apps.desktop.controller import GameController
from apps.desktop.i18n import count_labels as _count_labels
from apps.desktop.i18n import normalize_lang, tr
from apps.desktop.settings import Settings
from apps.desktop.threads import CameraThread, SharedState
from apps.desktop.ui.icons import set_button_icon
from apps.desktop.ui.theme import APP_QSS, COLORS, load_fonts
from apps.desktop.ui.widgets import RatingBadge, StatBox
from engine.tracking.player_manager import PlayerManager

# Global Neon Stage style (tokens in apps/desktop/ui/theme.py).
APP_STYLE = APP_QSS


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes}:{sec:02d}"


class _CameraScanWorker(QThread):
    """Background camera probe to avoid freezing the UI.

    Opening missing indexes with DSHOW/MSMF takes ~0.2-0.5s per
    attempt; on the UI thread that froze Settings for several seconds.
    """

    done = Signal(object)  # list[tuple[int, str]]

    def run(self) -> None:
        try:
            from engine.pose_engine import list_available_cameras as _list_cams
            found = _list_cams()
        except Exception:
            found = []
        try:
            self.done.emit(list(found))
        except Exception:
            pass


class MenuView(QWidget):
    play_requested = Signal()
    settings_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lang = "en"
        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        title = QLabel()
        try:
            from apps.desktop.ui.icons import brand_pixmap as _brand_pix

            _pm = _brand_pix("open-just-dance-logo", 500)
            if _pm is None or _pm.isNull():
                raise ValueError("brand logo not available")
            logo.setPixmap(_pm)
            logo.setFixedWidth(500)
            logo.setScaledContents(True)
            # The logo already includes the name: hide the text title.
            title.hide()
        except Exception:
            try:
                from apps.desktop.ui.icons import pixmap as _icon_pix

                logo.setPixmap(_icon_pix("sparkles", 44, COLORS["accent"]))
            except Exception:
                pass
            title.setText("OPEN JUST DANCE")
            title.setObjectName("display")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 60px;")
        self.subtitle = QLabel("Dance with your webcam. The AI scores you in real time.")
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setWordWrap(True)

        self.play = QPushButton("  Play")
        self.play.setObjectName("primary")
        self.play.setMinimumWidth(280)
        self.play.setMinimumHeight(48)
        set_button_icon(self.play, "play", 20, COLORS["ink"])
        self.play.clicked.connect(self.play_requested)
        self.settings_btn = QPushButton("  Settings")
        self.settings_btn.setMinimumWidth(280)
        set_button_icon(self.settings_btn, "settings", 18)
        self.settings_btn.clicked.connect(self.settings_requested)
        self.quit_btn = QPushButton("  Quit")
        self.quit_btn.setObjectName("ghost")
        self.quit_btn.setMinimumWidth(280)
        set_button_icon(self.quit_btn, "x", 18)
        self.quit_btn.clicked.connect(self.quit_requested)

        self.footer = QLabel("Webcam + pose AI  ·  1–4 players")
        self.footer.setObjectName("micro")
        self.footer.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addStretch(2)
        layout.addWidget(logo, 0, Qt.AlignCenter)
        layout.addWidget(title, 0, Qt.AlignCenter)
        layout.addWidget(self.subtitle)
        layout.addSpacing(24)
        layout.addWidget(self.play, 0, Qt.AlignCenter)
        layout.addWidget(self.settings_btn, 0, Qt.AlignCenter)
        layout.addWidget(self.quit_btn, 0, Qt.AlignCenter)
        layout.addStretch(3)
        layout.addWidget(self.footer)

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.subtitle.setText(tr(code, "menu.subtitle"))
        self.play.setText(tr(code, "menu.play"))
        self.settings_btn.setText(tr(code, "menu.settings"))
        self.quit_btn.setText(tr(code, "menu.quit"))
        self.footer.setText(tr(code, "menu.footer"))


class SongSelectView(QWidget):
    back_requested = Signal()
    play_requested = Signal(object)  # Song

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.songs = []
        self._import_thread = None
        self._progress = None
        self._progress_bar = None
        self._progress_dark_text = False
        self._import_cancelled = False
        self.setAcceptDrops(True)

        def _t(key: str, **kw) -> str:
            return tr(getattr(self.settings, "language", "en"), key, **kw)

        self.title_label = QLabel(_t("songs.title"))
        self.title_label.setObjectName("title")

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("search")
        self.search_edit.setPlaceholderText(_t("songs.search_placeholder"))
        self.search_edit.setClearButtonEnabled(True)
        try:
            from apps.desktop.ui.icons import icon as _icon

            self.search_edit.addAction(_icon("search", 16, COLORS["text2"]),
                                       QLineEdit.LeadingPosition)
        except Exception:
            pass
        self.search_edit.textChanged.connect(self._on_search)
        self.search_edit.returnPressed.connect(self._play)
        self.count_label = QLabel("")
        self.count_label.setObjectName("micro")

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_select)
        self.list_widget.itemDoubleClicked.connect(lambda _: self._play())
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        try:
            from PySide6.QtWidgets import QAbstractItemView as _QAV

            self.list_widget.setVerticalScrollMode(_QAV.ScrollMode.ScrollPerPixel)
        except Exception:
            pass

        self.cover = QLabel()
        self.cover.setMinimumSize(180, 180)
        self.cover.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cover.setAlignment(Qt.AlignCenter)
        self.cover.setObjectName("coverArt")
        self.cover.setStyleSheet(
            f"background: {COLORS['bg2']}; border: 1px solid {COLORS['line']};"
            " border-radius: 12px;"
        )
        try:
            from apps.desktop.ui.icons import pixmap as _icon_pix

            self._cover_placeholder = _icon_pix("music", 56, COLORS["text2"])
            self.cover.setPixmap(self._cover_placeholder)
        except Exception:
            self.cover.setText(tr(getattr(self.settings, "language", "en"), "songs.sin_cover"))

        self.info = QLabel("")
        self.info.setObjectName("hint")
        self.info.setWordWrap(True)

        self.play_btn = QPushButton("")
        self.play_btn.setObjectName("primary")
        set_button_icon(self.play_btn, "play", 18, COLORS["ink"])
        self.play_btn.clicked.connect(self._play)
        self.import_btn = QPushButton("")
        set_button_icon(self.import_btn, "upload", 18)
        self.import_btn.clicked.connect(self._import_video)
        self.edit_btn = QPushButton("")
        set_button_icon(self.edit_btn, "pencil", 16)
        self.edit_btn.clicked.connect(self._edit_song)
        self.delete_btn = QPushButton("")
        self.delete_btn.setObjectName("danger")
        set_button_icon(self.delete_btn, "trash-2", 16)
        self.delete_btn.clicked.connect(self._delete_song)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.edit_btn)
        edit_row.addWidget(self.delete_btn)
        self.back_btn = QPushButton("")
        set_button_icon(self.back_btn, "arrow-left", 16)
        self.back_btn.clicked.connect(self.back_requested)
        # Clickable drag&drop area: opens the same dialog as Import video.
        self.drop_zone = QFrame()
        self.drop_zone.setObjectName("dropZone")
        self.drop_zone.setMinimumHeight(150)
        self.drop_zone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drop_zone.setCursor(Qt.PointingHandCursor)
        self.drop_zone.setToolTip("")
        drop_lay = QVBoxLayout(self.drop_zone)
        drop_lay.setContentsMargins(12, 14, 12, 14)
        drop_lay.setSpacing(6)
        drop_lay.setAlignment(Qt.AlignCenter)
        self.drop_icon = QLabel()
        self.drop_icon.setAlignment(Qt.AlignCenter)
        try:
            from apps.desktop.ui.icons import pixmap as _icon_pix

            self.drop_icon.setPixmap(_icon_pix("folder", 48, COLORS["text2"]))
        except Exception:
            self.drop_icon.setText("📁")
        self.drop_title = QLabel("")
        self.drop_title.setObjectName("hint")
        self.drop_title.setAlignment(Qt.AlignCenter)
        self.drop_title.setWordWrap(True)
        drop_sub = QLabel("mp4 · mov · avi · mkv · webm")
        drop_sub.setObjectName("micro")
        drop_sub.setAlignment(Qt.AlignCenter)
        # Children do not intercept clicks: the whole area opens the dialog.
        for _w in (self.drop_icon, self.drop_title, drop_sub):
            _w.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        drop_lay.addWidget(self.drop_icon, 0, Qt.AlignCenter)
        drop_lay.addWidget(self.drop_title, 0, Qt.AlignCenter)
        drop_lay.addWidget(drop_sub, 0, Qt.AlignCenter)
        self.drop_zone.mousePressEvent = lambda _ev: self._import_video()

        right = QVBoxLayout()
        right.addWidget(self.cover, 1)
        right.addWidget(self.info)
        right.addWidget(self.import_btn)
        right.addWidget(self.play_btn)
        right.addLayout(edit_row)
        right.addWidget(self.back_btn)
        right.addWidget(self.drop_zone)
        right_pane = QWidget()
        right_pane.setLayout(right)
        right_pane.setMinimumWidth(240)

        left = QVBoxLayout()
        left.addWidget(self.title_label)
        left.addWidget(self.search_edit)
        left.addWidget(self.count_label)
        left.addWidget(self.list_widget, 1)
        left_pane = QWidget()
        left_pane.setLayout(left)
        left_pane.setMinimumWidth(280)

        from PySide6.QtWidgets import QSplitter

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(splitter)
        self.retranslate()
        self.refresh()

    def _t(self, key: str, **kw) -> str:
        return tr(getattr(self.settings, "language", "en"), key, **kw)

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            try:
                self.settings.language = normalize_lang(lang)
            except Exception:
                pass
        self.title_label.setText(self._t("songs.title"))
        self.search_edit.setPlaceholderText(self._t("songs.search_placeholder"))
        self.play_btn.setText(self._t("songs.play"))
        self.import_btn.setText(self._t("songs.import"))
        self.import_btn.setToolTip(self._t("songs.import_tooltip"))
        self.edit_btn.setText(self._t("songs.edit"))
        self.edit_btn.setToolTip(self._t("songs.edit_tooltip"))
        self.delete_btn.setText(self._t("songs.delete"))
        self.delete_btn.setToolTip(self._t("songs.delete_tooltip"))
        self.back_btn.setText(self._t("songs.back"))
        self.drop_zone.setToolTip(self._t("songs.drop_tooltip"))
        self.drop_title.setText(self._t("songs.drop_title"))
        self._rebuild_list()

    @staticmethod
    def _norm(text: str) -> str:
        import unicodedata

        try:
            t = unicodedata.normalize("NFKD", str(text or ""))
            return "".join(c for c in t if not unicodedata.combining(c)).casefold()
        except Exception:
            return str(text or "").lower()

    def refresh(self) -> None:
        self.songs = find_songs(self.settings.resolve_songs_dir())
        self._rebuild_list()

    def _on_search(self, _text: str = "") -> None:
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        """Rebuilds the list applying the search filter.

        Items store the Song in Qt.UserRole: selection/play/edit
        never depend on the index, so filtering breaks nothing.
        """
        query = self._norm(self.search_edit.text()) if hasattr(self, "search_edit") else ""
        if not self.songs:
            self.list_widget.clear()
            hint = QListWidgetItem(self._t("songs.no_songs"))
            hint.setFlags(hint.flags() & ~Qt.ItemIsSelectable)
            self.list_widget.addItem(hint)
            self.count_label.setText("")
            self._set_song_buttons(False)
            self.info.setText(self._t("songs.select_hint"))
            return
        shown = [
            s for s in self.songs
            if not query or query in self._norm(s.title) or query in self._norm(s.artist)
        ]
        self.list_widget.blockSignals(True)
        try:
            self.list_widget.clear()
            for song in shown:
                nd = getattr(song, "num_dancers", 1)
                suffix = self._t("songs.dancers_suffix", n=nd) if nd > 1 else ""
                item = QListWidgetItem(f"{song.title} — {song.artist}{suffix}")
                item.setData(Qt.UserRole, song)
                self.list_widget.addItem(item)
            if not shown:
                empty = QListWidgetItem(self._t("songs.no_results", query=self.search_edit.text().strip()))
                empty.setFlags(empty.flags() & ~Qt.ItemIsSelectable)
                self.list_widget.addItem(empty)
        finally:
            self.list_widget.blockSignals(False)
        total = len(self.songs)
        if total == 1:
            count_txt = self._t("songs.count_one", shown=len(shown), total=total)
        else:
            count_txt = self._t("songs.count_many", shown=len(shown), total=total)
        if query:
            count_txt += self._t("songs.count_filter", query=self.search_edit.text().strip())
        self.count_label.setText(count_txt)
        if shown:
            self.list_widget.setCurrentRow(0)
        # _on_select keeps info/buttons consistent in all cases
        self._on_select(self.list_widget.currentRow())

    def _set_song_buttons(self, enabled: bool) -> None:
        self.play_btn.setEnabled(enabled)
        self.edit_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)

    def _on_select(self, _row: int = -1) -> None:
        item = self.list_widget.currentItem()
        song = item.data(Qt.UserRole) if item is not None else None
        if song is None:
            self._set_song_buttons(False)
            return
        self._set_song_buttons(True)
        nd = getattr(song, "num_dancers", 1)
        if nd > 1:
            dancers_txt = self._t("songs.info_dancers", n=nd)
        else:
            dancers_txt = ""
        self.info.setText(
            self._t(
                "songs.info_line",
                title=song.title,
                artist=song.artist,
                duration=format_time(song.duration),
                fps=f"{song.fps:.0f}",
                dancers=dancers_txt,
                poses=len(song.poses),
            )
        )
        cover_path = Path(song.path) / "cover.png"
        if cover_path.exists():
            side = max(180, min(self.cover.width(), self.cover.height()) or 220)
            pix = QPixmap(str(cover_path)).scaled(
                side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.cover.setPixmap(pix)
        else:
            self.cover.setPixmap(getattr(self, "_cover_placeholder", QPixmap()))

    def _play(self) -> None:
        song = self._current_song()
        if song is not None:
            self.play_requested.emit(song)

    def _current_song(self):
        item = self.list_widget.currentItem()
        if item is None:
            return None
        try:
            return item.data(Qt.UserRole)
        except Exception:
            return None

    def _select_song_by_path(self, path) -> None:
        """Selects the visible list song with that path (or nothing)."""
        try:
            want = str(path)
        except Exception:
            return
        for i in range(self.list_widget.count()):
            try:
                s = self.list_widget.item(i).data(Qt.UserRole)
            except Exception:
                s = None
            if s is not None and str(getattr(s, "path", "")) == want:
                self.list_widget.setCurrentRow(i)
                return

    def _edit_song(self) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        song = self._current_song()
        if song is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self._t("songs.edit_title"))
        dlg.resize(360, 160)
        form = QFormLayout(dlg)
        title_edit = QLineEdit(song.title)
        title_edit.setPlaceholderText(self._t("songs.edit_title_ph"))
        form.addRow(self._t("songs.edit_title_label"), title_edit)
        artist_edit = QLineEdit(song.artist)
        artist_edit.setPlaceholderText(self._t("songs.edit_artist_ph"))
        form.addRow(self._t("songs.edit_artist_label"), artist_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(self._t("songs.save"))
        buttons.button(QDialogButtonBox.Cancel).setText(self._t("songs.cancel"))
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        new_title = title_edit.text().strip()
        if not new_title:
            QMessageBox.warning(self, self._t("songs.edit_title"), self._t("songs.empty_title"))
            return
        new_artist = artist_edit.text().strip() or self._t("songs.unknown_artist")
        try:
            from shared.dance_format.loader import save_song

            song.title = new_title
            song.artist = new_artist
            save_song(song, Path(song.path) / "song.json")
        except Exception as exc:
            QMessageBox.warning(self, self._t("songs.edit_title"), self._t("songs.save_error", err=exc))
            return
        keep_path = song.path
        self.refresh()
        self._select_song_by_path(keep_path)

    def _delete_song(self) -> None:
        import shutil

        song = self._current_song()
        if song is None:
            return
        ans = QMessageBox.question(
            self,
            self._t("songs.delete_title"),
            self._t("songs.delete_confirm", title=song.title, artist=song.artist, path=song.path),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        # Safety: only delete inside the songs folder.
        try:
            songs_dir = self.settings.resolve_songs_dir().resolve()
            target = Path(song.path).resolve()
            if target == songs_dir or songs_dir not in target.parents:
                raise ValueError(f"Ruta fuera de canciones: {target}")
        except Exception as exc:
            QMessageBox.warning(self, self._t("songs.delete_title"), self._t("songs.delete_blocked", err=exc))
            return
        try:
            shutil.rmtree(str(target))
        except Exception as exc:
            QMessageBox.warning(self, self._t("songs.delete_title"), self._t("songs.delete_error", err=exc))
            return
        self.refresh()
        QMessageBox.information(self, self._t("songs.deleted_title"), self._t("songs.deleted_msg", title=song.title))

    # ---------- importar video ----------
    def _import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("songs.import_dialog_title"),
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;Todos (*.*)",
        )
        if not path:
            return
        self._start_import(Path(path))

    def _start_import(self, video: Path) -> None:
        from apps.desktop.ui.import_dialog import ImportDialog
        from apps.desktop.workers.import_process import VideoImportProcess
        from apps.desktop.workers.import_worker import slugify, unique_song_dir

        songs_dir = self.settings.resolve_songs_dir()
        dlg = ImportDialog(video, songs_dir, self, language=getattr(self.settings, "language", "en"))
        if dlg.exec() != QDialog.Accepted:
            return
        data = dlg.get_data()
        slug = slugify(data["slug"])
        out_dir = unique_song_dir(songs_dir, slug)

        # Avoid simultaneous double import
        if self._import_thread is not None and self._import_thread.isRunning():
            QMessageBox.warning(self, self._t("songs.import_progress_title"), self._t("songs.import_in_progress"))
            return

        self._progress = QProgressDialog(self._t("songs.importing", name=video.name), self._t("songs.cancel"), 0, 100, self)
        self._progress.setWindowTitle(self._t("songs.import_progress_title"))
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel_import)
        # Only here: adaptive text inside the bar. At first the text
        # sits on the dark background (white) and when the blue chunk reaches it
        # switches to black. A single color has no contrast in both cases.
        self._progress_bar = None
        self._progress_dark_text = False
        try:
            _bar = self._progress.findChild(QProgressBar)
            if _bar is not None:
                _bar.setStyleSheet(f"QProgressBar {{ color: {COLORS['text1']}; }}")
                self._progress_bar = _bar
        except Exception:
            pass
        self._progress.show()
        self.import_btn.setEnabled(False)
        self._import_cancelled = False

        # Extraction runs in a CHILD PROCESS (not a thread): cancel is
        # safe and instant (flag + OS kill) and never hangs the UI.
        self._import_thread = VideoImportProcess(
            video=video,
            out_dir=out_dir,
            title=data["title"],
            artist=data["artist"],
            fps=data["fps"],
            max_width=data["max_width"],
            rotation=data["rotation"],
            num_dancers=int(data.get("num_dancers", 1) or 1),
            parent=self,
        )
        self._pending_import = {"video": video, "rotation": int(data.get("rotation", 0) or 0),
                                "out_dir": str(out_dir)}
        self._import_thread.progress.connect(self._on_import_progress)
        self._import_thread.status.connect(self._on_import_status)
        self._import_thread.succeeded.connect(self._on_import_ok)
        self._import_thread.failed.connect(self._on_import_failed)
        self._import_thread.cancelled.connect(self._on_import_cancelled)
        self._import_thread.needs_decision.connect(self._on_import_decision)
        self._import_thread.finished.connect(self._on_import_finished)
        self._import_thread.start()

    def _on_import_progress(self, cur: int, total: int) -> None:
        if self._progress is not None and total > 0:
            pct = int(100 * cur / total)
            self._progress.setValue(pct)
            # The % is centered: below 50% it sits on dark background (white),
            # at >=50% on the blue chunk (black). Only repaint on crossing.
            try:
                want_dark = pct >= 50
                if want_dark != self._progress_dark_text and self._progress_bar is not None:
                    color = COLORS["ink"] if want_dark else COLORS["text1"]
                    self._progress_bar.setStyleSheet(f"QProgressBar {{ color: {color}; }}")
                    self._progress_dark_text = want_dark
            except Exception:
                pass

    def _on_import_status(self, msg: str) -> None:
        if self._progress is not None:
            self._progress.setLabelText(self._tr_worker_msg(msg))

    def _tr_worker_msg(self, msg: str) -> str:
        """Translates child-process messages (always in Spanish) to the current language."""
        import re

        code = normalize_lang(getattr(self.settings, "language", "en"))
        if code == "es":
            return msg
        s = str(msg or "")
        if s == "Extrayendo poses...":
            return tr(code, "import.status_extracting")
        if s == "Guardando canción...":
            return tr(code, "import.status_saving")
        if s == "Listo":
            return tr(code, "import.status_done")
        if s == "Cancelando... (terminando)":
            return tr(code, "import.cancel_finishing")
        m = re.fullmatch(r"Se detectaron (\d+) estables \(declarados (\d+)\)", s)
        if m:
            return tr(code, "import.status_detected", o=m.group(1), d=m.group(2))
        m = re.fullmatch(r"No existe el video: (.*)", s, re.DOTALL)
        if m:
            return tr(code, "import.err_no_video", path=m.group(1))
        if s == "No se detectó ninguna pose en el video.":
            return tr(code, "import.err_no_poses")
        if s.startswith("El video no contiene poses detectables."):
            return tr(code, "import.err_no_usable")
        m = re.fullmatch(r"No se pudo crear (.*): (.*)", s, re.DOTALL)
        if m:
            return tr(code, "import.err_create_dir", path=m.group(1), err=m.group(2))
        m = re.fullmatch(r"No se pudo escribir el payload: (.*)", s, re.DOTALL)
        if m:
            return tr(code, "import.err_payload", err=m.group(1))
        if s == "No se pudo lanzar el proceso de importación.":
            return tr(code, "import.err_launch")
        m = re.fullmatch(r"No se pudo lanzar el proceso de importación \(ver (.*)\)\.", s, re.DOTALL)
        if m:
            return tr(code, "import.err_launch_log", log=m.group(1))
        m = re.fullmatch(r"Error al importar: (.*)", s, re.DOTALL)
        if m:
            return tr(code, "import.err_generic", err=m.group(1))
        return msg

    def _on_import_ok(self, song) -> None:
        if getattr(self, "_import_cancelled", False):
            return
        if self._progress is not None:
            self._progress.setValue(100)
        self.refresh()
        # select the new song (by path; when the filter hides it there is no selection)
        self._select_song_by_path(song.path)
        QMessageBox.information(
            self,
            self._t("songs.imported_title"),
            self._t("songs.imported_msg", title=song.title, poses=len(song.poses), path=song.path),
        )

    def _on_import_failed(self, msg: str) -> None:
        if getattr(self, "_import_cancelled", False):
            return
        QMessageBox.warning(self, self._t("songs.import_error"), self._tr_worker_msg(msg))

    def _on_import_decision(self, song, out_dir, declared: int, observed: int) -> None:
        if getattr(self, "_import_cancelled", False):
            return
        from pathlib import Path as _Path

        from apps.desktop.workers.import_worker import save_song_dir
        # Close progress to show the decision dialog.
        if self._progress is not None:
            self._progress.close()
            self._progress = None
            self._progress_bar = None
        self.import_btn.setEnabled(True)
        box = QMessageBox(self)
        box.setWindowTitle(self._t("songs.decision_title"))
        box.setText(self._t("songs.decision_text", declared=declared, observed=observed))
        keep_btn = box.addButton(self._t("songs.decision_keep", observed=observed), QMessageBox.AcceptRole)
        box.addButton(self._t("songs.decision_cancel"), QMessageBox.RejectRole)
        # Capture BEFORE exec: _on_import_finished (thread finished)
        # may run during the nested loop and clear _pending_import.
        pending = getattr(self, "_pending_import", {}) or {}
        pending = dict(pending)
        box.exec()
        if getattr(self, "_import_cancelled", False):
            return
        if box.clickedButton() is keep_btn:
            try:
                save_song_dir(song, _Path(out_dir), _Path(pending.get("video", "")),
                              rotation=int(pending.get("rotation", 0) or 0))
            except Exception as exc:
                QMessageBox.warning(self, self._t("songs.import_error"), self._t("songs.decision_save_error", err=exc))
                return
            self._on_import_ok(song)
        else:
            import shutil as _shutil
            try:
                _shutil.rmtree(str(out_dir), ignore_errors=True)
            except Exception:
                pass
            QMessageBox.information(self, self._t("songs.upload_cancelled_title"),
                                    self._t("songs.upload_cancelled_msg"))
        self._import_thread = None
        self._pending_import = {}

    def _on_import_finished(self) -> None:
        # When cancelled, clean partial leftovers (half-written preview/poses).
        if getattr(self, "_import_cancelled", False):
            pending = getattr(self, "_pending_import", {}) or {}
            out_dir = pending.get("out_dir")
            if out_dir:
                import shutil as _shutil
                try:
                    _shutil.rmtree(str(out_dir), ignore_errors=True)
                except Exception:
                    pass
        if self._progress is not None:
            self._progress.close()
            self._progress = None
            self._progress_bar = None
        self.import_btn.setEnabled(True)
        self._import_thread = None
        self._import_cancelled = False
        self._pending_import = {}

    def _on_import_cancelled(self) -> None:
        # The worker aborted cooperatively: notify without an error popup.
        # The actual close and cleanup are done by _on_import_finished.
        if self._progress is not None:
            self._progress.setLabelText(self._t("songs.cancelled_cleaning"))

    def _cancel_import(self) -> None:
        # Safe and non-blocking: the worker runs in a child process.
        # cancel() sets the cooperative flag and, when the child does not exit in 3s
        # (e.g. stuck loading models), OS-kills it, which cannot
        # hang the UI. NEVER use QThread.terminate() here.
        thread = self._import_thread
        if thread is None or not thread.isRunning():
            self._on_import_finished()
            return
        self._import_cancelled = True
        try:
            thread.cancel()
        except Exception:
            pass
        if self._progress is not None:
            self._progress.setLabelText(self._t("songs.canceling"))
            try:
                self._progress.setCancelButtonText(self._t("songs.canceling"))
                btn = self._progress.findChild(QPushButton)
                if btn is not None:
                    btn.setEnabled(False)
            except Exception:
                pass

    def _set_drop_highlight(self, on: bool) -> None:
        zone = getattr(self, "drop_zone", None)
        if zone is None:
            return
        zone.setProperty("dragOver", "true" if on else "false")
        zone.style().unpolish(zone)
        zone.style().polish(zone)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                    event.acceptProposedAction()
                    self._set_drop_highlight(True)
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._set_drop_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        self._set_drop_highlight(False)
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                self._start_import(p)
                event.acceptProposedAction()
                return


class CalibrationView(QWidget):
    calibration_finished = Signal()
    restart_camera_requested = Signal()

    def __init__(self, shared: SharedState, parent=None) -> None:
        super().__init__(parent)
        self.shared = shared
        self._lang = "en"
        self._samples = []
        self._stable_since = None
        self._countdown_value = 0
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._check_timer = QTimer(self)
        self._check_timer.setInterval(100)
        self._check_timer.timeout.connect(self._check)
        self._raw_ok = False

        self.title_label = QLabel(tr("en", "calib.title"))
        self.title_label.setObjectName("title")
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setObjectName("video")

        self.status_label = QLabel("")
        self.status_label.setObjectName("mono")
        self.dark_warning = QLabel("")
        self.dark_warning.setObjectName("warn")
        self.dark_warning.setWordWrap(True)
        self.dark_warning.hide()

        self.restart_btn = QPushButton("")
        set_button_icon(self.restart_btn, "rotate-ccw", 16)
        self.restart_btn.setFixedWidth(220)
        self.restart_btn.clicked.connect(self.restart_camera_requested)

        self.steps = {
            "body": QLabel(""),
            "full": QLabel(""),
            "light": QLabel(""),
            "stable": QLabel(""),
        }
        for lbl in self.steps.values():
            lbl.setObjectName("hint")
            lbl.setWordWrap(True)

        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setObjectName("countdown")

        left = QVBoxLayout()
        left.addWidget(self.camera_label, 1)
        left.addWidget(self.status_label)
        left.addWidget(self.dark_warning)
        steps_card = QFrame()
        steps_card.setObjectName("card")
        steps_layout = QVBoxLayout(steps_card)
        for lbl in self.steps.values():
            steps_layout.addWidget(lbl)
        left.addWidget(steps_card)
        left.addWidget(self.restart_btn, 0, Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addLayout(left)
        layout.addWidget(self.countdown_label, 0, Qt.AlignCenter)
        layout.addStretch(1)
        self.retranslate("en")

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.title_label.setText(tr(code, "calib.title"))
        self.status_label.setText(tr(code, "calib.connecting"))
        self.dark_warning.setText(tr(code, "calib.dark_warning"))
        self.restart_btn.setText(tr(code, "calib.restart"))
        self.steps["body"].setText(tr(code, "calib.step_search"))
        self.steps["full"].setText(tr(code, "calib.step_full"))
        self.steps["light"].setText(tr(code, "calib.step_light"))
        self.steps["stable"].setText(tr(code, "calib.step_stable"))

    def update_stats(self, fps: float, brightness: float, backend: str) -> None:
        # Note: brightness <25 on integrated cams is usually a closed shutter; not a pose error
        self.status_label.setText(
            tr(self._lang, "calib.status", fps=f"{fps:5.1f}", bright=f"{brightness:3.0f}", backend=backend)
        )

    def begin(self) -> None:
        self._samples = []
        self._stable_since = None
        self._countdown_value = 0
        self.countdown_label.setText("")
        self._reset_steps()
        self._check_timer.start()

    def _reset_steps(self) -> None:
        for lbl in self.steps.values():
            lbl.setObjectName("hint")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def update_frame(self, qimg) -> None:
        # Integrated laptop cam: when the QImage arrives as uniform gray from manual exposure,
        # the thread already reverts to auto; meanwhile do not freeze the UI.
        try:
            pix = QPixmap.fromImage(qimg).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
            self.camera_label.setPixmap(pix)
        except Exception:
            pass

    def _set_step(self, key: str, ok: bool, text: str) -> None:
        # SVG icon with color already applied: raw SVG uses currentColor and
        # Qt would paint it black in <img> (no contrast on dark background).
        try:
            from apps.desktop.ui.icons import tinted_path as _tinted

            name = "check-circle" if ok else "circle"
            color = COLORS["success"] if ok else COLORS["text2"]
            src = _tinted(name, color)
            lbl = self.steps[key]
            if src:
                lbl.setText(f'<img src="{src}" width="16" height="16"/> '
                            f'<span style="color:{color};">{text}</span>')
            else:
                lbl.setText(f'<span style="color:{color};">{text}</span>')
        except Exception:
            lbl = self.steps[key]
            lbl.setText(text)

    def _check(self) -> None:
        # In calibration we prefer showing pose only (without demanding a perfect image):
        # when there is a valid pose even with momentarily gray brightness, do not block.
        pose, raw, _ts, brightness = self.shared.snapshot()
        # multi fallback: when there is no mono raw but there are tracks, use the first track
        if raw is None:
            try:
                tracks, _, _ = self.shared.snapshot_multi()
                if tracks:
                    first = sorted(tracks.items(), key=lambda kv: kv[0])[0][1]
                    pose, raw = first.norm_pose, first.raw
            except Exception:
                pass

        if raw is None:
            # No camera frame has arrived yet.
            self.steps["body"].setText(tr(self._lang, "calib.wait_camera"))
            self.steps["body"].setStyleSheet("font-size: 16px; color: #ffb300;")
            self._samples = []
            self._stable_since = None
            return

        # Integrated laptop cams may show 1-2 gray frames at start; do not spam the warning.
        # Only show when persisting >1s gray? For now threshold 25 is only a visual hint.
        self.dark_warning.setVisible(brightness < 25.0)

        body_ok = not raw.is_empty()
        self._set_step("body", body_ok, tr(self._lang, "calib.step_body"))
        if not body_ok:
            self.steps["body"].setText(tr(self._lang, "calib.step_search"))
            self._samples = []
            self._stable_since = None
            return

        needed = [
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "left_wrist", "right_wrist", "left_ankle", "right_ankle",
        ]
        full_ok = all(raw.visible(n, 0.5) for n in needed)
        self._set_step("full", full_ok, tr(self._lang, "calib.step_full"))

        # Pose-only: in calibration lighting does not block when a body is detected.
        # The indicator stays but never gates the countdown (integrated laptop cam under average light).
        light_ok = brightness >= 40.0
        self._set_step("light", light_ok, tr(self._lang, "calib.step_light"))
        # When there is a body and stable tracking, light is not blocking (informational only)
        light_blocking = False

        stable_ok = False
        head = pose.get("head") if (pose is not None and not pose.is_empty()) else None
        if head is not None:
            self._samples.append((head.x, head.y))
            if len(self._samples) > 20:
                self._samples.pop(0)
            if len(self._samples) >= 10:
                n = len(self._samples)
                mx = sum(s[0] for s in self._samples) / n
                my = sum(s[1] for s in self._samples) / n
                var = sum((s[0] - mx) ** 2 + (s[1] - my) ** 2 for s in self._samples) / n
                stable_ok = var < 0.002
        self._set_step("stable", stable_ok, tr(self._lang, "calib.step_stable"))

        # Pose-only: do not gate on light when tracking is good (integrated laptop cam)
        calib_ok = all([body_ok, full_ok, stable_ok]) and (light_ok or not light_blocking)
        if calib_ok:
            if self._stable_since is None:
                self._stable_since = time.monotonic()
            elif time.monotonic() - self._stable_since >= 2.5:
                self._start_countdown()
        else:
            self._stable_since = None

    def _start_countdown(self) -> None:
        self._check_timer.stop()
        self._countdown_value = 4
        self._tick_countdown()
        self._countdown_timer.start(1000)

    def _tick_countdown(self) -> None:
        self._countdown_value -= 1
        if self._countdown_value <= 0:
            self._countdown_timer.stop()
            self.countdown_label.setText("")
            self.calibration_finished.emit()
            return
        self.countdown_label.setText(str(self._countdown_value))


class GameView(QWidget):
    game_finished = Signal(object)  # {summary, recording, song}

    def __init__(self, shared: SharedState, parent=None) -> None:
        super().__init__(parent)
        self.shared = shared
        self.song = None
        self._lang = "en"
        self.controller = None
        self.synchronizer = None
        self.player = None
        self.audio_output = None
        self.video_sink = None
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.recording: list[dict] = []  # history for the result comparison
        self._max_recording = 5000  # bound memory
        self._ref_aspect: float | None = None  # reference video W/H
        # --- robustness for long/heavy videos (e.g. 4K 235s): never block the UI ---
        self._tick_busy = False  # reentrancy guard: ticks never pile up
        self._finished = False  # idempotent _finish (error + EndOfMedia)
        self._last_video_wall = 0.0  # last painted video frame (throttle)
        self._last_cam_wall = 0.0  # last painted camera frame (throttle)
        self._min_frame_interval = 0.066  # ~15 fps of painting is enough in game
        self._last_ref_obj_id: int | None = None  # ref pose cache per FramePose
        self._last_ref_pose = None
        self._last_rec_t = -1.0  # last (decimated) recording
        self._rec_interval = 0.10  # record at 10 Hz (comparison needs no 20 Hz)
        self._last_pos_ms = -1  # stuck-player watchdog
        self._last_advance_wall = 0.0
        self._stall_warned = False

        self.video_widget = QLabel()
        self.video_widget.setMinimumSize(320, 240)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setAlignment(Qt.AlignCenter)
        self.video_widget.setObjectName("video")
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setObjectName("video")

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("mono")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setObjectName("spotlight")
        self.progress.setTextVisible(False)

        self.video_card = QFrame()
        self.video_card.setObjectName("card")
        vlayout = QVBoxLayout(self.video_card)
        vlayout.addWidget(self.video_widget)

        self.cam_container = QWidget()
        self._cam_layout = QVBoxLayout(self.cam_container)
        self._cam_layout.setContentsMargins(0, 0, 0, 0)
        self._cam_layout.addWidget(self.camera_label)

        bottom = QVBoxLayout()
        bottom.addWidget(self.time_label)
        bottom.addWidget(self.progress)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_card, 2)
        layout.addWidget(self.cam_container, 2)
        layout.addLayout(bottom)

        # PiP (picture-in-picture) game layout: large reference video with
        # the player camera as a small overlay at the bottom-left.
        # Developer mode keeps the legacy stacked layout.
        self._developer_mode = False
        self._pip_ratio = 0.25
        self._pip_margin = 12
        try:
            self.video_card.installEventFilter(self)
        except Exception:
            pass
        self.apply_layout_mode(False)

    def apply_layout_mode(self, developer_mode: bool) -> None:
        """Switches between legacy (True) and PiP (False) camera layouts."""
        self._developer_mode = bool(developer_mode)
        try:
            if self._developer_mode:
                self._layout_legacy()
            else:
                self._layout_pip()
        except Exception as exc:
            print(f"[gameView layout] {exc}")

    def _layout_legacy(self) -> None:
        """Legacy developer view: reference video + camera stacked."""
        try:
            self._cam_layout.removeWidget(self.camera_label)
        except Exception:
            pass
        try:
            self.camera_label.setParent(self.cam_container)
        except Exception:
            pass
        if self._cam_layout.indexOf(self.camera_label) < 0:
            self._cam_layout.addWidget(self.camera_label)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setMaximumSize(16777215, 16777215)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.show()
        try:
            self.cam_container.show()
        except Exception:
            pass

    def _layout_pip(self) -> None:
        """Default game view: large reference + small camera overlay."""
        try:
            self._cam_layout.removeWidget(self.camera_label)
        except Exception:
            pass
        try:
            self.cam_container.hide()
        except Exception:
            pass
        try:
            self.camera_label.setParent(self.video_card)
        except Exception:
            pass
        self.camera_label.setMinimumSize(160, 120)
        self.camera_label.setMaximumSize(320, 240)
        self.camera_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.camera_label.show()
        try:
            self.camera_label.raise_()
        except Exception:
            pass
        self._position_pip()

    def _position_pip(self) -> None:
        """Places the camera overlay at the bottom-left of the video card."""
        try:
            stage = self.video_card
            w = int(stage.width() * self._pip_ratio)
            w = max(160, min(320, w))
            h = max(120, min(240, int(w * 3 / 4)))
            m = int(self._pip_margin)
            x = m
            y = max(m, stage.height() - h - m)
            self.camera_label.setFixedSize(w, h)
            self.camera_label.move(x, y)
        except Exception:
            pass

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        try:
            from PySide6.QtCore import QEvent as _QEvent

            if obj is getattr(self, "video_card", None) and event.type() == _QEvent.Resize:
                if not bool(getattr(self, "_developer_mode", False)):
                    self._position_pip()
        except Exception:
            pass
        try:
            return super().eventFilter(obj, event)
        except Exception:
            return False

    def begin(self, song, player_manager=None) -> None:
        self.song = song
        self.player_manager = player_manager
        settings = Settings.load()
        try:
            self.apply_layout_mode(bool(getattr(settings, "developer_mode", False)))
        except Exception:
            pass
        # multi mode when there is a player_manager with players
        self.is_multi = bool(player_manager and getattr(player_manager, "players", []))
        if self.is_multi:
            # per-player synchronizers/controllers already come in player_manager, but build a local list
            self.synchronizer = SongSynchronizer(song)
            self.multi_controllers: dict[int, GameController] = {}
            for p in player_manager.players:
                if p.controller is None:
                    p.controller = GameController(PoseComparator(weights=settings.weights, angle_weight=settings.angle_weight), Scorer(thresholds=settings.thresholds()))
                self.multi_controllers[p.player_id] = p.controller
                # ensure an assigned dancer
                if p.reference_dancer_id is None:
                    p.reference_dancer_id = 0
            self.controller = None
            # adapt UI for multi: show one row per player
            self._setup_multi_ui()
        else:
            self.is_multi = False
            self.controller = GameController(
                PoseComparator(
                    weights=settings.weights,
                    angle_weight=settings.angle_weight,
                ),
                Scorer(thresholds=settings.thresholds()),
            )
            self.synchronizer = SongSynchronizer(song)
        self.recording = []
        # reset robustness state per match
        self._tick_busy = False
        self._finished = False
        self._last_video_wall = 0.0
        self._last_cam_wall = 0.0
        self._last_ref_obj_id = None
        self._last_ref_pose = None
        self._last_rec_t = -1.0
        self._last_pos_ms = -1
        self._last_advance_wall = time.monotonic()
        self._stall_warned = False
        # Publish the reference video aspect: the camera thread
        # pre-scales into comparison space before normalizing, so
        # comparison works on vertical (9:16) against a horizontal camera.
        # Without this, vertical gave error ~0.65 = permanent Miss.
        try:
            from shared.dance_format.loader import song_reference_aspect

            self._ref_aspect = song_reference_aspect(song)
        except Exception:
            self._ref_aspect = None
        try:
            self.shared.set_reference_aspect(self._ref_aspect)
        except Exception:
            pass
        self._setup_player(song)
        self.timer.start()

    def _setup_multi_ui(self) -> None:
        # When already built, do not duplicate
        if hasattr(self, "_multi_boxes_created") and self._multi_boxes_created:
            return
        # Create extra multi labels next to the compact top row
        # Simplified: reuse the existing boxes but switch titles for multi
        try:
            self.multi_container = QFrame()
            self.multi_container.setObjectName("card")
            self.multi_layout = QVBoxLayout(self.multi_container)
            self.player_rows: list[tuple] = []
            for p in self.player_manager.players:
                row = QHBoxLayout()
                lbl = QLabel(tr(self._lang, "game.player_row", pid=p.player_id, tid=p.camera_track_id, did=(p.reference_dancer_id or 0) + 1))
                lbl.setObjectName("mono")
                score = StatBox(f"P{p.player_id}")
                combo = StatBox(f"C{p.player_id}")
                acc = StatBox(f"%{p.player_id}")
                rating = RatingBadge()
                row.addWidget(lbl)
                row.addWidget(score)
                row.addWidget(combo)
                row.addWidget(acc)
                row.addWidget(rating)
                self.multi_layout.addLayout(row)
                self.player_rows.append((p.player_id, score, combo, acc, rating))
            # insert below the camera_label
            # find layout; we have self already built; insert after
            self.layout().addWidget(self.multi_container)
            self._multi_boxes_created = True
        except Exception as exc:
            print(f"[gameView multi ui] {exc}")

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        # Refresh multi row labels when they already exist.
        try:
            rows = getattr(self, "player_rows", []) or []
            pm = getattr(self, "player_manager", None)
            if pm is not None:
                for i, row in enumerate(rows):
                    pid = row[0]
                    try:
                        p = pm.get_by_player(pid)
                        if p is not None:
                            # the QLabel is the first item of the row layout
                            lay = self.multi_layout.itemAt(i)
                            if lay is not None:
                                item = lay.itemAt(0)
                                w = item.widget() if item is not None else None
                                if w is not None:
                                    w.setText(tr(self._lang, "game.player_row", pid=p.player_id, tid=p.camera_track_id, did=(p.reference_dancer_id or 0) + 1))
                    except Exception:
                        pass
        except Exception:
            pass

    def _setup_player(self, song) -> None:
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.player.setAudioOutput(self.audio_output)
        self.video_sink = QVideoSink(self)
        self.player.setVideoSink(self.video_sink)
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)

        song_dir = Path(song.path) if getattr(song, "path", None) else None
        media = None
        if song_dir:
            cand = song_dir / "song.mp4"
            if cand.exists():
                media = cand
            else:
                cand2 = song_dir / "song.mp3"
                if cand2.exists():
                    media = cand2
        if media is not None:
            self.player.setSource(QUrl.fromLocalFile(str(media)))
        else:
            # no media: do not play, but keep the timer for tests
            pass
        self.player.positionChanged.connect(self._on_position)
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.errorOccurred.connect(self._on_player_error)
        self.player.play()

    def _on_video_frame(self, frame: QVideoFrame) -> None:
        # Throttle BEFORE toImage(): decoding a 4K frame to QImage costs
        # tens of ms; at 30 fps it would saturate the UI thread and the app "hangs".
        # Painting at ~15 fps is indistinguishable in game and keeps the UI alive.
        try:
            now = time.monotonic()
            if now - self._last_video_wall < self._min_frame_interval:
                return
            self._last_video_wall = now
            qimg = frame.toImage()
            if qimg.isNull():
                return
            target = self.video_widget.size()
            if target.width() > 0 and target.height() > 0 and (
                qimg.width() > target.width() or qimg.height() > target.height()
            ):
                # Scale the QImage (cheap) before converting to QPixmap:
                # avoids an intermediate 3840x2160 pixmap (~33 MB).
                qimg = qimg.scaled(target, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.video_widget.setPixmap(QPixmap.fromImage(qimg))
        except Exception as exc:
            print(f"[video] frame skip: {exc}")

    def update_frame(self, qimg) -> None:
        # Same throttle for the camera: the thread may emit faster than
        # the UI can paint (especially with YOLO + per-frame overlay).
        try:
            if qimg is None or qimg.isNull():
                return
            now = time.monotonic()
            if now - self._last_cam_wall < self._min_frame_interval:
                return
            self._last_cam_wall = now
            target = self.camera_label.size()
            if target.width() > 0 and target.height() > 0 and (
                qimg.width() > target.width() or qimg.height() > target.height()
            ):
                qimg = qimg.scaled(target, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def _on_position(self, position_ms: int) -> None:
        try:
            if self.player is None:
                return
            duration = self.player.duration()
        except Exception:
            return
        if duration > 0:
            self.time_label.setText(
                f"{format_time(position_ms / 1000)} / {format_time(duration / 1000)}"
            )
            self.progress.setValue(int(100 * position_ms / duration))

    def _on_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._finish()

    def _on_player_error(self, error, error_string: str) -> None:
        # Previously it only printed and the game hung on a broken video.
        # Now: visible warning + clean close with partial results.
        if error == QMediaPlayer.Error.NoError:
            return
        print(f"[video] error: {error_string}")
        try:
            from PySide6.QtCore import QTimer as _QTimer

            def _show() -> None:
                try:
                    QMessageBox.warning(
                        self,
                        tr(self._lang, "game.video_error_title"),
                        tr(self._lang, "game.video_error_msg", err=error_string),
                    )
                except Exception:
                    pass
                self._finish()
            _QTimer.singleShot(0, _show)
        except Exception:
            try:
                self._finish()
            except Exception:
                pass

    def _tick(self) -> None:
        if self.player is None or self.synchronizer is None:
            return
        if self._tick_busy:
            # The previous tick is still processing (heavy video): skip instead of piling up.
            return
        self._tick_busy = True
        try:
            try:
                pos_ms = int(self.player.position())
            except Exception:
                return
            t = pos_ms / 1000.0
            self._watchdog_player(pos_ms)
            if getattr(self, "is_multi", False) and getattr(self, "player_manager", None):
                self._tick_multi(t)
                return
            pose, raw, _ts, _brightness = self.shared.snapshot()
            ref = self.synchronizer.pose_at(t)
            # Cache per FramePose: pose_at returns the same object while t
            # falls in its interval; avoids Pose.from_dict + flip on every tick.
            if ref is None:
                ref_pose = None
            elif id(ref) == self._last_ref_obj_id:
                ref_pose = self._last_ref_pose
            else:
                ref_pose = Pose.from_dict(ref.joints) if ref else None
                ref_pose = self._ensure_y_up(ref_pose)
                self._last_ref_obj_id = id(ref)
                self._last_ref_pose = ref_pose
            # ref_raw_pose is unused in game and comparison (the
            # comparison re-reads the reference from song): do not build it.
            pose = self._ensure_y_up(pose)
            result = self.controller.evaluate(pose, ref_pose)

            # Decimated 10 Hz recording, minimal: the comparison only uses
            # player+rating (it re-reads the reference from song). Saving
            # duplicated raw/ref per tick was the biggest memory/GC churn
            # on long songs (5000 entries x 4 pose dicts).
            try:
                # -1e-6: 0.3-0.2 gives 0.0999... in float and would skip 1 of every ~10
                if (t - self._last_rec_t) >= self._rec_interval - 1e-6 and len(self.recording) < self._max_recording:
                    self._last_rec_t = t
                    entry = {
                        "time": float(t),
                        "player": pose.to_dict() if pose and not pose.is_empty() else None,
                        "rating": result.get("rating", "—"),
                        "neutral": bool(result.get("neutral", False)),
                        "error": result.get("error"),
                        "tracking": bool(result.get("tracking", False)),
                    }
                    self.recording.append(entry)
            except Exception:
                pass
            # No live stats row (combo/accuracy/rating): the controller
            # keeps accumulating for the results screen.
        finally:
            self._tick_busy = False

    @staticmethod
    def _ensure_y_up(p: Pose | None) -> Pose | None:
        # Orientation fix: the old reference pose (song.json)
        # may have negative Y up (head y = -3) while the
        # current BodyNormalizer gives positive Y up (head y = +3).
        if p is None or p.is_empty():
            return p
        from engine.pose_engine.skeleton import Joint as _Joint

        head = p.get("head")
        lh = p.get("left_hip")
        rh = p.get("right_hip")
        if head is None:
            head = p.get("left_shoulder") or p.get("right_shoulder")
        hip_y = None
        if lh and rh:
            hip_y = (lh.y + rh.y) / 2.0
        elif lh:
            hip_y = lh.y
        elif rh:
            hip_y = rh.y
        if head is None or hip_y is None:
            return p
        if head.y < hip_y:  # negative Y up -> flip
            return Pose(joints={name: _Joint(x=j.x, y=-j.y, z=j.z, visibility=j.visibility) for name, j in p.joints.items()})
        return p

    def _watchdog_player(self, pos_ms: int) -> None:
        # When the player stops advancing (codec stuck on heavy 4K),
        # the app previously looked hung with no message. Now it warns once and
        # offers to finish with partial results instead of hanging.
        try:
            if pos_ms != self._last_pos_ms:
                self._last_pos_ms = pos_ms
                self._last_advance_wall = time.monotonic()
                return
            try:
                duration = int(self.player.duration())
            except Exception:
                return
            if duration <= 0 or pos_ms >= duration - 500:
                return  # at the end not advancing is normal
            try:
                from PySide6.QtMultimedia import QMediaPlayer as _QMP

                if self.player.playbackState() != _QMP.PlaybackState.PlayingState:
                    return  # paused by the user: not a stall
            except Exception:
                pass  # test players without playbackState: keep watching
            if time.monotonic() - self._last_advance_wall < 5.0:
                return
            if self._stall_warned:
                return
            self._stall_warned = True
            print(f"[video] stall: sin avance {pos_ms}/{duration} ms")
            from PySide6.QtCore import QTimer as _QTimer

            def _ask() -> None:
                try:
                    ans = QMessageBox.warning(
                        self,
                        tr(self._lang, "game.stall_title"),
                        tr(self._lang, "game.stall_msg"),
                        QMessageBox.Ok | QMessageBox.Close,
                        QMessageBox.Ok,
                    )
                    if ans == QMessageBox.Close:
                        self._finish()
                    else:
                        self._last_advance_wall = time.monotonic()
                        self._stall_warned = False
                except Exception:
                    pass
            _QTimer.singleShot(0, _ask)
        except Exception:
            pass

    def _tick_multi(self, t: float) -> None:
        tracks, _, _ = self.shared.snapshot_multi()
        # mark LOST/ACTIVE without reassigning (§9)
        self.player_manager.update_tracks(list(tracks.keys()))
        # update rows
        for row in getattr(self, "player_rows", []):
            pid, score_box, combo_box, acc_box, badge = row
            p = self.player_manager.get_by_player(pid)
            if p is None:
                continue
            tp = tracks.get(p.camera_track_id) if p.camera_track_id else None
            if p.state == "LOST" or tp is None:
                badge.set_rating("LOST")
                acc_box.set_value("—")
                # update label
                continue
            dancer_id = p.reference_dancer_id if p.reference_dancer_id is not None else 0
            frame = self.synchronizer.frame_at(t)
            dpose = frame.get_dancer(dancer_id) if frame else None
            ref_joints = dpose.joints if dpose else {}
            ref_pose = Pose.from_dict(ref_joints) if ref_joints else None
            player_pose = self._ensure_y_up(tp.norm_pose) if tp and tp.norm_pose else None
            ref_pose = self._ensure_y_up(ref_pose)
            ctrl = self.multi_controllers.get(pid)
            if ctrl is None:
                continue
            result = ctrl.evaluate(player_pose, ref_pose)
            if result.get("neutral"):
                badge.set_rating("—")
                continue
            score_box.set_value(str(result["score"]))
            combo_box.set_value(str(result["combo"]))
            acc_box.set_value(f"{result['frame_score']:.0f}%")
            badge.set_rating(result["rating"])
            # decimated record by first player (the comparison re-reads the
            # reference from song: do not duplicate raw/ref here either).
            try:
                if (pid == self.player_manager.players[0].player_id
                        and (t - self._last_rec_t) >= self._rec_interval - 1e-6
                        and len(self.recording) < self._max_recording):
                    self._last_rec_t = t
                    entry = {"time": float(t), "player": player_pose.to_dict() if player_pose and not player_pose.is_empty() else None,
                             "rating": result.get("rating","—"),
                             "neutral": bool(result.get("neutral",False)), "error": result.get("error"), "tracking": bool(result.get("tracking",False)), "player_id": pid,
                             "dancer_id": int(dancer_id)}
                    self.recording.append(entry)
            except Exception:
                pass
        # single global mirror in multi mode: do not update single boxes

    def _finish(self) -> None:
        # Idempotent: video error + EndOfMedia may arrive together; the
        # second _finish previously re-emitted results and re-navigated.
        if self._finished:
            return
        self._finished = True
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.player is not None:
                self.player.stop()
        except Exception:
            pass
        if getattr(self, "is_multi", False):
            summaries = {}
            for p in self.player_manager.players:
                ctrl = self.multi_controllers.get(p.player_id)
                if ctrl:
                    summaries[p.player_id] = ctrl.summary()
            # aggregate for the primary payload (first player)
            first = self.player_manager.players[0].player_id if self.player_manager.players else None
            summary = summaries.get(first, {"score":0,"max_combo":0,"frames":0,"accuracy":0,"rating_counts":{"Perfect":0,"Great":0,"Good":0,"Ok":0,"Miss":0}}) if first else {"score":0,"max_combo":0,"frames":0,"accuracy":0,"rating_counts":{"Perfect":0,"Great":0,"Good":0,"Miss":0}}
            summary["multi"] = summaries
            payload = {"summary": summary, "recording": list(self.recording), "song": self.song, "player_manager": self.player_manager.to_list()}
            self.game_finished.emit(payload)
            return
        summary = self.controller.summary()
        payload = {"summary": summary, "recording": list(self.recording), "song": self.song}
        self.game_finished.emit(payload)


class ResultView(QWidget):
    menu_requested = Signal()
    replay_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lang = "en"
        self._last_payload = None
        self.title_label = QLabel(tr("en", "result.title"))
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.score_label = QLabel("0")
        self.score_label.setObjectName("score")
        self.score_label.setAlignment(Qt.AlignCenter)

        stars_row = QHBoxLayout()
        stars_row.addStretch()
        self._star_labels: list[QLabel] = []
        for _ in range(5):
            s = QLabel()
            s.setAlignment(Qt.AlignCenter)
            s.setFixedSize(40, 40)
            self._star_labels.append(s)
            stars_row.addWidget(s)
        stars_row.addStretch()
        self._set_stars(0)

        self.details = QLabel("")
        self.details.setObjectName("hint")
        self.details.setAlignment(Qt.AlignCenter)

        # Pose comparison (new)
        from apps.desktop.ui.comparison import PoseComparisonWidget

        self.comparison = PoseComparisonWidget(language="en")
        self.comparison.setVisible(False)
        comp_card = QFrame()
        comp_card.setObjectName("card")
        comp_layout = QVBoxLayout(comp_card)
        self.comp_title = QLabel(tr("en", "result.compare_title"))
        self.comp_title.setObjectName("subtitle")
        self.comp_title.setAlignment(Qt.AlignCenter)
        self.comp_hint = QLabel(tr("en", "result.compare_hint"))
        self.comp_hint.setWordWrap(True)
        self.comp_hint.setObjectName("micro")
        self.comp_hint.setAlignment(Qt.AlignCenter)
        comp_layout.addWidget(self.comp_title)
        comp_layout.addWidget(self.comp_hint)
        comp_layout.addWidget(self.comparison, 1)

        self.replay_btn = QPushButton(tr("en", "result.replay"))
        self.replay_btn.setObjectName("primary")
        set_button_icon(self.replay_btn, "play", 18, COLORS["ink"])
        self.replay_btn.clicked.connect(self.replay_requested)
        self.menu_btn = QPushButton(tr("en", "result.menu"))
        set_button_icon(self.menu_btn, "arrow-left", 16)
        self.menu_btn.clicked.connect(self.menu_requested)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self.title_label)
        layout.addSpacing(8)
        layout.addWidget(self.score_label)
        layout.addLayout(stars_row)
        layout.addWidget(self.details)
        layout.addWidget(comp_card, 3)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.replay_btn)
        btn_row.addWidget(self.menu_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addSpacing(8)

        from apps.desktop.ui.responsive import scrollable as _scrollable

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(body, self))

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.title_label.setText(tr(code, "result.title"))
        self.comp_title.setText(tr(code, "result.compare_title"))
        self.comp_hint.setText(tr(code, "result.compare_hint"))
        self.replay_btn.setText(tr(code, "result.replay"))
        self.menu_btn.setText(tr(code, "result.menu"))
        try:
            self.comparison.retranslate(code)
        except Exception:
            pass
        if self._last_payload is not None:
            try:
                self.show_summary(self._last_payload)
            except Exception:
                pass

    def _set_stars(self, n: int) -> None:
        try:
            from apps.desktop.ui.icons import pixmap as _icon_pix
        except Exception:
            _icon_pix = None
        for i, lbl in enumerate(self._star_labels):
            on = i < max(0, min(5, int(n)))
            if _icon_pix is None:
                lbl.setText("★" if on else "☆")
                continue
            # All filled: earned in gold, pending in dim gray.
            lbl.setPixmap(_icon_pix("star-filled", 32, "#FFD700" if on else COLORS["text3"]))

    @staticmethod
    def _stars(accuracy: float) -> int:
        acc = max(0.0, min(100.0, accuracy))
        if acc >= 80:
            return 5
        if acc >= 60:
            return 4
        if acc >= 40:
            return 3
        if acc >= 20:
            return 2
        return 1

    def show_summary(self, summary: dict) -> None:
        try:
            self._last_payload = summary
        except Exception:
            pass
        recording = None
        song = None
        payload_summary = summary
        multi = None
        if isinstance(summary, dict) and "summary" in summary and "recording" in summary:
            payload_summary = summary["summary"]
            recording = summary.get("recording")
            song = summary.get("song")
            multi = payload_summary.get("multi") if isinstance(payload_summary, dict) else None
        counts = payload_summary.get("rating_counts", {"Perfect":0,"Great":0,"Good":0,"Ok":0,"Miss":0})
        # PLAYER -> followed Dancer mapping (comes in the game payload).
        dancer_of: dict = {}
        try:
            for pent in (summary.get("player_manager") or []):
                dancer_of[int(pent.get("playerId"))] = pent.get("referenceDancerId")
        except Exception:
            dancer_of = {}
        code = self._lang
        if multi and isinstance(multi, dict):
            lines = []
            for pid, s in sorted(multi.items()):
                c = s.get("rating_counts", counts)
                _did = dancer_of.get(pid)
                _dtag = tr(code, "result.player_dtag", did=_did + 1) if isinstance(_did, int) else ""
                try:
                    acc = float(s.get('accuracy', 0))
                except Exception:
                    acc = 0
                lines.append(tr(code, "result.player_line", pid=pid, dtag=_dtag, score=s.get('score',0), combo=s.get('max_combo',0), acc=f"{acc:.1f}", p=c.get('Perfect',0), g=c.get('Great',0), go=c.get('Good',0), o=c.get('Ok',0), m=c.get('Miss',0)))
            try:
                first_pid = list(multi.keys())[0]
            except Exception:
                first_pid = 0
            try:
                acc0 = float(payload_summary.get('accuracy',0))
            except Exception:
                acc0 = 0
            detail = "\n".join(lines) + "\n\n" + tr(code, "result.total_line", pid=first_pid, combo=payload_summary.get('max_combo',0), acc=f"{acc0:.1f}")
        else:
            try:
                acc = float(payload_summary.get('accuracy',0))
            except Exception:
                acc = 0
            detail = (
                tr(code, "result.combo_max", combo=payload_summary.get('max_combo',0)) + "\n"
                + tr(code, "result.accuracy", acc=f"{acc:.1f}") + "\n"
                + tr(code, "result.counts", p=counts.get('Perfect',0), g=counts.get('Great',0), go=counts.get('Good',0), o=counts.get('Ok',0), m=counts.get('Miss',0))
            )
        stars = self._stars(float(payload_summary.get("accuracy",0)))
        self.score_label.setText(str(payload_summary.get("score",0)))
        self._set_stars(stars)
        self.details.setText(detail)

        # update the comparison with the dancer followed by the first player
        # (previously it always showed Dancer 1 even when another was chosen).
        try:
            if recording is not None and song is not None:
                _did = 0
                try:
                    if dancer_of:
                        _v = list(dancer_of.values())[0]
                        _did = int(_v) if _v is not None else 0
                    elif recording and isinstance(recording[0], dict) and recording[0].get("dancer_id") is not None:
                        _did = int(recording[0]["dancer_id"])
                except Exception:
                    _did = 0
                self.comparison.set_data(song, recording, dancer_id=_did)
                self.comparison.setVisible(True)
            elif recording is not None:
                # when recording arrives without song, try hiding
                self.comparison.setVisible(False)
            else:
                # no recording data (old tests) -> hide comparison
                self.comparison.setVisible(False)
        except Exception as exc:
            print(f"[result] comparativa error: {exc}")
            self.comparison.setVisible(False)


class SettingsView(QWidget):
    back_requested = Signal()
    saved = Signal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings

        self.title_label = QLabel("")
        self.title_label.setObjectName("title")

        from apps.desktop.i18n import difficulty_labels as _diff_labels

        form = QFormLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(220)
        self.camera_refresh_btn = QPushButton("")
        set_button_icon(self.camera_refresh_btn, "rotate-ccw", 16)
        self.camera_refresh_btn.clicked.connect(self._refresh_cameras)
        self._scan_thread: _CameraScanWorker | None = None
        self._scan_wanted: int = 0
        _cam_row = QHBoxLayout()
        _cam_row.addWidget(self.camera_combo, 1)
        _cam_row.addWidget(self.camera_refresh_btn, 0)
        _cam_widget = QWidget()
        _cam_widget.setLayout(_cam_row)
        _cam_widget.setContentsMargins(0, 0, 0, 0)
        self.lbl_camera = QLabel("")
        form.addRow(self.lbl_camera, _cam_widget)

        self.mirror = QComboBox()
        self.mirror.setCurrentIndex(0 if settings.mirror else 1)
        self.lbl_mirror = QLabel("")
        form.addRow(self.lbl_mirror, self.mirror)

        self.backend = QComboBox()
        self.backend.addItem("auto", "auto")
        self.backend.addItem("mediapipe-cpu", "mediapipe-cpu")
        idx = self.backend.findData(settings.backend)
        self.backend.setCurrentIndex(max(0, idx))
        self.lbl_backend = QLabel("")
        form.addRow(self.lbl_backend, self.backend)

        self.device = QComboBox()
        idx = self.device.findData(getattr(settings, "device", "auto"))
        self.device.setCurrentIndex(max(0, idx))
        self.lbl_device = QLabel("")
        form.addRow(self.lbl_device, self.device)

        self.language = QComboBox()
        self.lbl_language = QLabel("")
        form.addRow(self.lbl_language, self.language)

        self.difficulty = QComboBox()
        self.difficulty.currentIndexChanged.connect(self._on_difficulty)
        self.lbl_difficulty = QLabel("")
        form.addRow(self.lbl_difficulty, self.difficulty)

        self.perfect = self._spin_threshold(settings.perfect_threshold)
        self.great = self._spin_threshold(settings.great_threshold)
        self.good = self._spin_threshold(settings.good_threshold)
        self.ok = self._spin_threshold(settings.ok_threshold)
        self.lbl_perfect = QLabel("")
        self.lbl_great = QLabel("")
        self.lbl_good = QLabel("")
        self.lbl_ok = QLabel("")
        form.addRow(self.lbl_perfect, self.perfect)
        form.addRow(self.lbl_great, self.great)
        form.addRow(self.lbl_good, self.good)
        form.addRow(self.lbl_ok, self.ok)

        self.angle_weight = QDoubleSpinBox()
        self.angle_weight.setRange(0.0, 1.0)
        self.angle_weight.setSingleStep(0.05)
        self.angle_weight.setValue(settings.angle_weight)
        self.lbl_angle = QLabel("")
        form.addRow(self.lbl_angle, self.angle_weight)

        self.smooth_alpha = QDoubleSpinBox()
        self.smooth_alpha.setRange(0.0, 1.0)
        self.smooth_alpha.setSingleStep(0.05)
        self.smooth_alpha.setValue(settings.smooth_alpha)
        self.lbl_smooth = QLabel("")
        form.addRow(self.lbl_smooth, self.smooth_alpha)

        self.dev_mode = QCheckBox()
        self.dev_mode.setChecked(bool(getattr(settings, "developer_mode", False)))
        self.lbl_dev_mode = QLabel("")
        form.addRow(self.lbl_dev_mode, self.dev_mode)

        self._on_difficulty()

        self.gpu_label = QLabel(gpu_summary())
        self.gpu_label.setObjectName("micro")
        self.gpu_label.setWordWrap(True)
        self.lbl_hardware = QLabel("")
        form.addRow(self.lbl_hardware, self.gpu_label)

        self.save_btn = QPushButton("")
        self.save_btn.setObjectName("primary")
        set_button_icon(self.save_btn, "check", 18, COLORS["ink"])
        self.save_btn.clicked.connect(self._save)
        self.back_btn = QPushButton("")
        set_button_icon(self.back_btn, "arrow-left", 16)
        self.back_btn.clicked.connect(self.back_requested)

        buttons = QHBoxLayout()
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.back_btn)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.addLayout(form)

        from apps.desktop.ui.responsive import scrollable as _scrollable2

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(self.title_label)
        layout.addWidget(_scrollable2(card, self), 1)
        layout.addLayout(buttons)
        self.retranslate()
        self.sync_from_settings()

    def _t(self, key: str, **kw) -> str:
        return tr(getattr(self.settings, "language", "en"), key, **kw)

    def retranslate(self, lang: str | None = None) -> None:
        """Retranslates labels; preserves current field values."""
        if lang is not None:
            try:
                self.settings.language = normalize_lang(lang)
            except Exception:
                pass
        code = normalize_lang(getattr(self.settings, "language", "en"))
        from apps.desktop.i18n import difficulty_labels as _diff_labels

        # Combos with translated texts: rebuild preserving selection.
        cur_mirror = self.mirror.currentData() if self.mirror.count() else True
        self.mirror.blockSignals(True)
        try:
            self.mirror.clear()
            self.mirror.addItem(tr(code, "settings.mirror_on"), True)
            self.mirror.addItem(tr(code, "settings.mirror_off"), False)
            self.mirror.setCurrentIndex(0 if cur_mirror else 1)
        finally:
            self.mirror.blockSignals(False)
        cur_lang = normalize_lang(self.language.currentData()) if self.language.count() else code
        if self.language.count() == 0:
            cur_lang = code
        self.language.blockSignals(True)
        try:
            self.language.clear()
            self.language.addItem(tr(code, "settings.lang_en"), "en")
            self.language.addItem(tr(code, "settings.lang_es"), "es")
            idx = self.language.findData(cur_lang)
            self.language.setCurrentIndex(max(0, idx))
        finally:
            self.language.blockSignals(False)
        cur_diff = self.difficulty.currentData() if self.difficulty.count() else getattr(self.settings, "difficulty", "medium")
        self.difficulty.blockSignals(True)
        try:
            self.difficulty.clear()
            for key, label in _diff_labels(code).items():
                self.difficulty.addItem(label, key)
            idx = self.difficulty.findData(cur_diff)
            self.difficulty.setCurrentIndex(max(0, idx))
        finally:
            self.difficulty.blockSignals(False)
        cur_dev = self.device.currentData() if self.device.count() else getattr(self.settings, "device", "auto")
        self.device.blockSignals(True)
        try:
            self.device.clear()
            self.device.addItem(tr(code, "settings.device_auto"), "auto")
            self.device.addItem(tr(code, "settings.device_cuda"), "cuda")
            self.device.addItem(tr(code, "settings.device_cpu"), "cpu")
            idx = self.device.findData(cur_dev)
            self.device.setCurrentIndex(max(0, idx))
        finally:
            self.device.blockSignals(False)
        self.title_label.setText(tr(code, "settings.title"))
        self.lbl_camera.setText(tr(code, "settings.camera"))
        try:
            self.camera_refresh_btn.setText(tr(code, "settings.camera_refresh"))
            self.camera_refresh_btn.setToolTip(tr(code, "settings.camera_refresh"))
        except Exception:
            pass
        # Retranslate special combo items (without re-probing cameras).
        try:
            for i in range(self.camera_combo.count()):
                data = self.camera_combo.itemData(i)
                if data is None:
                    self.camera_combo.setItemText(i, tr(code, "settings.camera_none"))
                elif isinstance(data, int) and self.camera_combo.itemText(i).endswith(")"):
                    # "(saved)" item or one with resolution: only retranslate the tagged one.
                    txt = self.camera_combo.itemText(i)
                    if "guardada" in txt or "saved" in txt:
                        self.camera_combo.setItemText(
                            i, tr(code, "settings.camera_saved_missing", n=data))
        except Exception:
            pass
        self.lbl_mirror.setText(tr(code, "settings.mirror"))
        self.lbl_backend.setText(tr(code, "settings.backend"))
        self.lbl_device.setText(tr(code, "settings.device"))
        self.lbl_language.setText(tr(code, "settings.language"))
        self.lbl_difficulty.setText(tr(code, "settings.difficulty"))
        self.lbl_perfect.setText(tr(code, "settings.th_perfect"))
        self.lbl_great.setText(tr(code, "settings.th_great"))
        self.lbl_good.setText(tr(code, "settings.th_good"))
        self.lbl_ok.setText(tr(code, "settings.th_ok"))
        self.lbl_angle.setText(tr(code, "settings.angle"))
        self.lbl_smooth.setText(tr(code, "settings.smooth"))
        self.lbl_dev_mode.setText(tr(code, "settings.dev_mode"))
        try:
            self.dev_mode.setToolTip(tr(code, "settings.dev_mode_hint"))
            self.lbl_dev_mode.setToolTip(tr(code, "settings.dev_mode_hint"))
        except Exception:
            pass
        self.lbl_hardware.setText(tr(code, "settings.hardware"))
        self.save_btn.setText(tr(code, "settings.save"))
        self.back_btn.setText(tr(code, "settings.back"))

    def sync_from_settings(self) -> None:
        """Refreshes field values from self.settings (when returning to Settings)."""
        s = self.settings
        try:
            saved_idx = int(getattr(s, "camera_index", 0) or 0)
        except Exception:
            saved_idx = 0
        try:
            # No hardware probing here (fast): ensure the saved
            # value exists in the combo. Real enumeration happens
            # on entering Settings or via the Refresh button.
            if self.camera_combo.count() == 0:
                self.camera_combo.addItem(
                    self._t("settings.camera_saved_missing", n=saved_idx), saved_idx)
            if self.camera_combo.findData(saved_idx) < 0:
                # Saved index not detected (camera unplugged):
                # still offered so the value is not lost.
                self.camera_combo.addItem(
                    self._t("settings.camera_saved_missing", n=saved_idx), saved_idx)
            self.camera_combo.setCurrentIndex(
                max(0, self.camera_combo.findData(saved_idx)))
        except Exception:
            pass
        try:
            self.mirror.setCurrentIndex(0 if s.mirror else 1)
        except Exception:
            pass
        try:
            idx = self.backend.findData(s.backend)
            self.backend.setCurrentIndex(max(0, idx))
        except Exception:
            pass
        try:
            idx = self.device.findData(getattr(s, "device", "auto"))
            self.device.setCurrentIndex(max(0, idx))
        except Exception:
            pass
        try:
            self.gpu_label.setText(gpu_summary())
        except Exception:
            pass
        try:
            idx = self.language.findData(normalize_lang(getattr(s, "language", "en")))
            self.language.setCurrentIndex(max(0, idx))
        except Exception:
            pass
        try:
            idx = self.difficulty.findData(s.difficulty)
            self.difficulty.setCurrentIndex(max(0, idx))
        except Exception:
            pass
        self._on_difficulty()
        try:
            self.smooth_alpha.setValue(float(s.smooth_alpha))
        except Exception:
            pass
        try:
            self.dev_mode.setChecked(bool(getattr(s, "developer_mode", False)))
        except Exception:
            pass

    def selected_camera_index(self) -> int:
        """Selected camera index in the combo (fallback to saved)."""
        try:
            data = self.camera_combo.currentData()
            if isinstance(data, int) and data >= 0:
                return data
        except Exception:
            pass
        try:
            return int(getattr(self.settings, "camera_index", 0) or 0)
        except Exception:
            return 0

    def _refresh_cameras(self) -> None:
        """Starts the background camera scan (does not block the UI).

        While scanning, the button stays disabled with a
        "Searching..." text. On finish, `_on_scan_finished` fills the combo.
        When a scan is already running, does nothing.
        """
        try:
            if self._scan_thread is not None and self._scan_thread.isRunning():
                return
        except Exception:
            pass
        try:
            self._scan_wanted = self.selected_camera_index()
        except Exception:
            self._scan_wanted = 0
        try:
            self.camera_refresh_btn.setEnabled(False)
            self.camera_refresh_btn.setText(self._t("settings.camera_searching"))
        except Exception:
            pass
        try:
            worker = _CameraScanWorker(self)
        except Exception:
            self._fill_cameras([], self._scan_wanted)
            return
        self._scan_thread = worker
        try:
            worker.done.connect(self._on_scan_finished)
            worker.finished.connect(worker.deleteLater)
        except Exception:
            pass
        worker.start()

    def _on_scan_finished(self, found) -> None:
        """Fills the combo with the background scan result."""
        self._scan_thread = None
        try:
            items = [(int(i), str(lbl)) for (i, lbl) in list(found or [])]
        except Exception:
            items = []
        try:
            wanted = int(self._scan_wanted)
        except Exception:
            wanted = 0
        self._fill_cameras(items, wanted)
        try:
            self.camera_refresh_btn.setEnabled(True)
            self.camera_refresh_btn.setText(self._t("settings.camera_refresh"))
        except Exception:
            pass

    def _stop_scan(self) -> None:
        """Waits for the running scan (when closing the app)."""
        try:
            th = self._scan_thread
            if th is not None and th.isRunning() and not th.wait(10000):
                try:
                    th.terminate()
                    th.wait(2000)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._scan_thread = None

    def _fill_cameras(self, found: list, wanted: int) -> None:
        """Fills the combo (synchronous; used by the scan slot).

        Preserves the `wanted` selection even when that camera is no
        longer connected ("saved" item).
        """
        try:
            wanted = int(wanted)
        except Exception:
            wanted = 0
        self.camera_combo.blockSignals(True)
        try:
            self.camera_combo.clear()
            for index, label in found:
                self.camera_combo.addItem(str(label), int(index))
            if not found:
                self.camera_combo.addItem(self._t("settings.camera_none"), None)
            # Preserve selection even when no longer connected.
            if self.camera_combo.findData(wanted) < 0 and wanted is not None:
                try:
                    w = int(wanted)
                    if w >= 0:
                        self.camera_combo.addItem(
                            self._t("settings.camera_saved_missing", n=w), w)
                except Exception:
                    pass
            idx = self.camera_combo.findData(wanted)
            self.camera_combo.setCurrentIndex(max(0, idx))
        finally:
            self.camera_combo.blockSignals(False)

    @staticmethod
    def _spin_threshold(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        return spin

    def _on_difficulty(self) -> None:
        preset = app_settings.DIFFICULTY_PRESETS.get(self.difficulty.currentData())
        if preset is None:
            return
        self.perfect.setValue(preset["perfect_threshold"])
        self.great.setValue(preset["great_threshold"])
        self.good.setValue(preset["good_threshold"])
        self.ok.setValue(preset["ok_threshold"])
        self.angle_weight.setValue(preset["angle_weight"])

    def _save(self) -> None:
        s = self.settings
        s.camera_index = self.selected_camera_index()
        s.mirror = self.mirror.currentData()
        s.backend = self.backend.currentData()
        s.device = self.device.currentData() or "auto"
        try:
            s.language = normalize_lang(self.language.currentData())
        except Exception:
            pass
        s.difficulty = self.difficulty.currentData()
        s.perfect_threshold = self.perfect.value()
        s.great_threshold = self.great.value()
        s.good_threshold = self.good.value()
        s.ok_threshold = self.ok.value()
        s.angle_weight = self.angle_weight.value()
        s.smooth_alpha = self.smooth_alpha.value()
        try:
            s.developer_mode = bool(self.dev_mode.isChecked())
        except Exception:
            pass
        s.save()
        self.saved.emit()


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or Settings.load()
        self.shared = SharedState()
        self.camera_thread: CameraThread | None = None
        self.current_song = None
        # player manager for multi (factory with current settings)
        def _cmp_factory():
            return PoseComparator(weights=self.settings.weights, angle_weight=self.settings.angle_weight)
        def _scorer_factory():
            return Scorer(thresholds=self.settings.thresholds())
        self.player_manager = PlayerManager(comparator_factory=_cmp_factory, scorer_factory=_scorer_factory)

        self.setWindowTitle("Open Just Dance")
        try:
            from apps.desktop.ui.icons import brand_icon as _brand_icon

            _app_icon = _brand_icon()
            if not _app_icon.isNull():
                self.setWindowIcon(_app_icon)
        except Exception:
            pass
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)
        load_fonts()
        self.setStyleSheet(APP_STYLE)

        # lazy import to avoid a cycle
        from apps.desktop.ui.join_view import JoinView
        from apps.desktop.ui.dancer_select_view import DancerSelectView
        from apps.desktop.ui.player_count_view import PlayerCountView
        self.join_view = JoinView(self.shared, self.player_manager, self)
        self.dancer_view = DancerSelectView(self.player_manager, self)
        self.player_count_view = PlayerCountView(self)
        self.calibration_view = CalibrationView(self.shared, self)
        self.game_view = GameView(self.shared, self)
        self.stack = QStackedWidget()
        self.menu_view = MenuView(self)
        self.song_view = SongSelectView(self.settings, self)
        self.result_view = ResultView(self)
        self.settings_view = SettingsView(self.settings, self)

        for view in (
            self.menu_view,
            self.song_view,
            self.player_count_view,
            self.join_view,
            self.dancer_view,
            self.calibration_view,
            self.game_view,
            self.result_view,
            self.settings_view,
        ):
            self.stack.addWidget(view)
        self.setCentralWidget(self.stack)

        self.menu_view.play_requested.connect(self._go_songs)
        self.menu_view.settings_requested.connect(self._go_settings)
        self.menu_view.quit_requested.connect(self.close)
        self.song_view.back_requested.connect(self._go_menu)
        self.song_view.play_requested.connect(self._go_player_count)
        self.player_count_view.count_selected.connect(self._on_player_count)
        self.player_count_view.back_requested.connect(self._go_songs)
        self.join_view.join_finished.connect(self._on_join_finished)
        self.join_view.back_requested.connect(self._go_player_count_back)
        self.dancer_view.finished.connect(self._on_dancer_finished)
        self.dancer_view.back_requested.connect(self._go_lobby)
        self.calibration_view.calibration_finished.connect(self._start_game)
        self.calibration_view.restart_camera_requested.connect(self.restart_camera)
        self.game_view.game_finished.connect(self._show_results)
        self.result_view.menu_requested.connect(self._go_menu)
        self.result_view.replay_requested.connect(lambda: self._go_player_count(self.current_song))
        self.settings_view.back_requested.connect(self._go_menu)
        self.settings_view.saved.connect(self._on_settings_saved)

        self.statusBar().showMessage(gpu_summary())
        self.stack.setCurrentWidget(self.menu_view)
        self.apply_language()

    def apply_language(self, lang: str | None = None) -> None:
        """Retranslates every view to the current language (hot switch)."""
        code = normalize_lang(lang if lang is not None else getattr(self.settings, "language", "en"))
        try:
            self.settings.language = code
        except Exception:
            pass
        for view in (
            self.menu_view,
            self.song_view,
            self.player_count_view,
            self.join_view,
            self.dancer_view,
            self.calibration_view,
            self.game_view,
            self.result_view,
            self.settings_view,
        ):
            try:
                view.retranslate(code)
            except Exception:
                pass

    # ---------- navigation ----------
    def _go_menu(self) -> None:
        try:
            self.result_view.comparison.stop()
        except Exception:
            pass
        try:
            self.join_view.stop()
        except Exception:
            pass
        self.stop_camera()
        self.stack.setCurrentWidget(self.menu_view)

    def _go_songs(self) -> None:
        try:
            self.join_view.stop()
        except Exception:
            pass
        self.song_view.refresh()
        self.stack.setCurrentWidget(self.song_view)

    def _go_settings(self) -> None:
        try:
            self.settings_view.sync_from_settings()
            self.settings_view.retranslate()
            # Enumerate cameras on entry (only when the combo has no
            # real results yet: avoids freezing on every visit).
            try:
                if self.settings_view.camera_combo.count() <= 1:
                    self.settings_view._refresh_cameras()
            except Exception:
                pass
        except Exception:
            pass
        self.stack.setCurrentWidget(self.settings_view)

    def _go_player_count(self, song) -> None:
        self.current_song = song
        self.player_count_view.begin(song)
        self.stack.setCurrentWidget(self.player_count_view)

    def _on_player_count(self, n: int) -> None:
        # set expectation 1..4 and move to lobby
        try:
            self.player_manager.set_expected(int(n))
        except Exception:
            pass
        self._start_lobby(self.current_song)

    def _go_player_count_back(self) -> None:
        # back from lobby to selector without losing the song
        self.player_count_view.begin(self.current_song)
        self.stack.setCurrentWidget(self.player_count_view)

    def _start_lobby(self, song) -> None:
        # compat: when arriving directly (old replay) with no expectation, go to selector instead? default to 1
        if song is not None:
            self.current_song = song
        if getattr(self.player_manager, "expected_players", None) is None:
            # without expectation (legacy/test path), go to selector
            self._go_player_count(self.current_song)
            return
        self.ensure_camera()
        # SINGLE YOLO11n pipeline in lobby/calibration/game (also 1P):
        # avoids mid-session backend switches and keeps IDs stable.
        # 1 player -> max 1 person via expected_players (no multi colors).
        exp = getattr(self.player_manager, "expected_players", None)
        use_multi = True
        if self.camera_thread:
            try:
                self.camera_thread.set_multi(use_multi)
                self.camera_thread.set_expected_players(exp)
                # Clean tracking IDs on lobby entry (new multi session).
                self.camera_thread.request_tracking_reset()
            except Exception:
                pass
        self.join_view.begin()
        self.stack.setCurrentWidget(self.join_view)

    def _go_lobby(self) -> None:
        self.stack.setCurrentWidget(self.join_view)
        self.join_view.begin()

    def _on_join_finished(self, pm) -> None:
        self.player_manager = pm
        # Pose-only: any player may pick any dancer (repeats allowed).
        # When the song has 1 dancer, auto-assign and skip selection.
        nd = getattr(self.current_song, "num_dancers", 1)
        auto_skip = False
        if nd == 1:
            for p in self.player_manager.players:
                if p.reference_dancer_id is None:
                    p.reference_dancer_id = 0
                    p.state = "READY"
            auto_skip = True
        elif nd > 1:
            # when everyone already has an assigned dancer (replay), skip too
            if all(p.reference_dancer_id is not None for p in self.player_manager.players):
                auto_skip = True
        if auto_skip:
            self._start_calibration_direct()
        else:
            self.dancer_view.begin(self.current_song)
            self.stack.setCurrentWidget(self.dancer_view)

    def _on_dancer_finished(self, pm) -> None:
        self.player_manager = pm
        self._start_calibration_direct()

    def _start_calibration_direct(self) -> None:
        # recalibrate _go: calibration needs ensure_camera already active
        self.ensure_camera()
        # Calibration with the same lobby YOLO pipeline (also 1P).
        exp = getattr(self.player_manager, "expected_players", None)
        if self.camera_thread:
            try:
                use_multi = True
                self.camera_thread.set_multi(use_multi)
                self.camera_thread.set_expected_players(exp)
            except Exception:
                pass
        self.calibration_view.begin()
        self.stack.setCurrentWidget(self.calibration_view)

    def _start_calibration(self, song) -> None:
        # legacy compat: redirects to lobby
        self._start_lobby(song)

    def _start_game(self) -> None:
        # game: always YOLO11n (same as lobby/calibration);
        # person count = expected player count.
        exp = getattr(self.player_manager, "expected_players", None)
        if self.camera_thread:
            try:
                use_multi = True
                self.camera_thread.set_multi(use_multi)
                self.camera_thread.set_expected_players(exp)
                self.camera_thread.set_song_num_dancers(
                    getattr(self.current_song, "num_dancers", None))
                self.camera_thread.request_tracking_reset()
            except Exception:
                pass
        # pass player_manager to game_view
        try:
            self.game_view.begin(self.current_song, self.player_manager)
        except TypeError:
            self.game_view.begin(self.current_song)
        self.stack.setCurrentWidget(self.game_view)

    def _show_results(self, payload: dict) -> None:
        # payload may be {summary, recording, song} (new) or an old summary (tests)
        self.result_view.show_summary(payload)
        self.stack.setCurrentWidget(self.result_view)

    def _on_settings_saved(self) -> None:
        old_index = getattr(self.settings, "camera_index", 0)
        self.settings = Settings.load()
        # When the camera was running on another index, restart it
        # to apply the new one (in Settings it is usually stopped and
        # will apply on the next ensure_camera).
        try:
            new_index = getattr(self.settings, "camera_index", 0)
            if (new_index != old_index and self.camera_thread is not None
                    and self.camera_thread.isRunning()):
                self.restart_camera()
        except Exception:
            pass
        # SongSelectView shares the settings object: update the language there too.
        try:
            self.song_view.settings = self.settings
        except Exception:
            pass
        try:
            self.settings_view.settings = self.settings
        except Exception:
            pass
        self.apply_language()
        code = normalize_lang(getattr(self.settings, "language", "en"))
        self.statusBar().showMessage(tr(code, "main.settings_saved") + gpu_summary(), 4000)
        self._go_menu()

    # ---------- camera ----------
    def _wire_camera(self, thread: CameraThread) -> None:
        thread.frame_ready.connect(self._on_frame)
        thread.stats_ready.connect(self._on_camera_stats)
        thread.failed.connect(self._on_camera_failed)
        try:
            thread.device_warning.connect(self._on_device_warning)
        except Exception:
            pass

    def _unwire_camera(self, thread: CameraThread) -> None:
        try:
            thread.frame_ready.disconnect()
            thread.stats_ready.disconnect()
            thread.failed.disconnect()
        except RuntimeError:
            pass
        try:
            thread.device_warning.disconnect()
        except Exception:
            pass

    def ensure_camera(self) -> None:
        if self.camera_thread is None:
            self.camera_thread = CameraThread(self.settings, self.shared, self)
            self._wire_camera(self.camera_thread)
            self._apply_saved_multi_config(self.camera_thread, None)
            self.camera_thread.start()
        elif not self.camera_thread.isRunning():
            old = self.camera_thread
            saved = self._snapshot_multi_config(old)
            self._unwire_camera(old)
            self.camera_thread = CameraThread(self.settings, self.shared, self)
            self._wire_camera(self.camera_thread)
            self._apply_saved_multi_config(self.camera_thread, saved)
            self.camera_thread.start()

    def _on_camera_stats(self, fps: float, brightness: float, backend: str) -> None:
        self.calibration_view.update_stats(fps, brightness, backend)
        try:
            self.join_view.update_stats(fps, brightness, backend)
        except Exception:
            pass

    def stop_camera(self) -> None:
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()
            if not self.camera_thread.wait(5000):
                # Last resort: a broken webcam may keep read() blocked.
                # terminate() is unsafe in general, but here it beats
                # leaving the camera on forever.
                try:
                    self.camera_thread.terminate()
                    self.camera_thread.wait(2000)
                except RuntimeError:
                    pass

    def _snapshot_multi_config(self, thread) -> dict | None:
        """Captures enable_multi/expected/song from the old thread (without touching it)."""
        if thread is None:
            return None
        try:
            multi = bool(getattr(thread, "enable_multi", False))
        except Exception:
            multi = False
        try:
            exp = getattr(thread, "_expected_players", None)
            exp = int(exp) if exp is not None else None
        except Exception:
            exp = None
        try:
            song = getattr(thread, "_song_num_dancers", None)
            song = int(song) if song is not None else None
        except Exception:
            song = None
        return {"enable_multi": multi, "expected": exp, "song": song}

    def _fallback_multi_config(self) -> tuple:
        """Falls back to player_manager/current_song when the old thread had no data."""
        try:
            exp = getattr(getattr(self, "player_manager", None), "expected_players", None)
            exp = int(exp) if exp is not None else None
        except Exception:
            exp = None
        try:
            song = getattr(getattr(self, "current_song", None), "num_dancers", None)
            song = int(song) if song is not None else None
        except Exception:
            song = None
        return exp, song

    def _apply_saved_multi_config(self, thread, saved: dict | None) -> None:
        """Re-applies the YOLO pipeline to the new thread after recreating it.

        Without this, the new thread is born with enable_multi=False and falls
        into the mono MediaPipe branch even when it used to run YOLO.
        """
        if thread is None:
            return
        fb_exp, fb_song = self._fallback_multi_config()
        if saved is None:
            exp, song = fb_exp, fb_song
            # No previous thread: YOLO only with multi context (lobby/game).
            multi = True if (exp is not None or song is not None) else None
        else:
            exp = saved.get("expected") if saved.get("expected") is not None else fb_exp
            song = saved.get("song") if saved.get("song") is not None else fb_song
            multi = bool(saved.get("enable_multi", False))
            # With known players/song, the pipeline must be YOLO
            # (condition in CameraThread.run: enable_multi or song is not None;
            # with expected but multi=False it would fall to MediaPipe).
            if exp is not None or song is not None:
                multi = True
        try:
            if multi is not None:
                thread.set_multi(multi)
            if exp is not None:
                thread.set_expected_players(exp)
            if song is not None:
                thread.set_song_num_dancers(song)
            thread.request_tracking_reset()
        except Exception:
            pass

    def restart_camera(self) -> None:
        # Preserve the YOLO pipeline: the new thread is born with
        # enable_multi=False and would fall to MediaPipe unless re-applied.
        saved = self._snapshot_multi_config(self.camera_thread)
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.stop_camera()
            self.camera_thread = None
        self.ensure_camera()
        if saved is not None and self.camera_thread is not None:
            self._apply_saved_multi_config(self.camera_thread, saved)

    def _on_frame(self, qimg) -> None:
        current = self.stack.currentWidget()
        if current is self.calibration_view:
            self.calibration_view.update_frame(qimg)
        elif current is self.join_view:
            self.join_view.update_frame(qimg)
        elif current is self.game_view:
            self.game_view.update_frame(qimg)
        elif current is self.player_count_view:
            # no video needed, but a future preview could be shown here
            pass

    def _on_device_warning(self, message: str) -> None:
        code = normalize_lang(getattr(self.settings, "language", "en"))
        shown = self._tr_device_msg(message, code)
        self.statusBar().showMessage(str(shown or ""), 6000)

    @staticmethod
    def _tr_device_msg(message: str, code: str) -> str:
        import re

        if normalize_lang(code) == "es":
            return message
        m = re.fullmatch(r"La camara (\d+) no abrio; usando camara 0\.", str(message or ""))
        if m:
            return tr(code, "settings.camera_fallback", wanted=m.group(1))
        return message

    def _on_camera_failed(self, message: str) -> None:
        code = normalize_lang(getattr(self.settings, "language", "en"))
        shown = self._tr_camera_msg(message, code)
        self.statusBar().showMessage(tr(code, "main.camera_title") + ": " + shown, 6000)
        QMessageBox.warning(self, tr(code, "main.camera_title"), shown)

    @staticmethod
    def _tr_camera_msg(message: str, code: str) -> str:
        import re

        if normalize_lang(code) == "es":
            return message
        s = str(message or "")
        m = re.fullmatch(r"No se pudo crear el backend de pose: (.*)", s, re.DOTALL)
        if m:
            return tr(code, "camera.err_backend", err=m.group(1))
        if s == "No se pudo abrir la camara":
            return tr(code, "camera.err_open")
        if s.startswith("La cámara no entrega frames."):
            return tr(code, "camera.err_no_frames")
        m = re.fullmatch(r"Error en el pipeline de cámara: (.*)", s, re.DOTALL)
        if m:
            return tr(code, "camera.err_pipeline", err=m.group(1))
        return message

    def closeEvent(self, event) -> None:
        try:
            self.settings_view._stop_scan()
        except Exception:
            pass
        self.stop_camera()
        event.accept()
