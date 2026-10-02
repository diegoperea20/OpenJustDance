"""Player-count selection before the lobby (1..4)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.i18n import count_labels as _count_labels
from apps.desktop.i18n import normalize_lang, tr
from apps.desktop.ui.icons import set_button_icon
from apps.desktop.ui.responsive import scrollable
from apps.desktop.ui.theme import COLORS


class PlayerCountView(QWidget):
    count_selected = Signal(int)  # 1..4
    back_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.song = None
        self._selected = 1
        self._lang = "en"

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(24, 24, 24, 24)

        self.title_label = QLabel("")
        self.title_label.setObjectName("display")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 40px;")

        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignCenter)

        self.song_info = QLabel("")
        self.song_info.setObjectName("mono")
        self.song_info.setAlignment(Qt.AlignCenter)
        self.song_info.setWordWrap(True)

        # Cards 1..4 in a centered row (4x120px fit from 900px up;
        # a FlowLayout here collapsed the width due to the side stretches).
        self.buttons: list[QPushButton] = []
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        cards_row.addStretch()
        for n in range(1, 5):
            btn = QPushButton(f"{n}\n")
            btn.setObjectName("countCard")
            btn.setCheckable(True)
            btn.setMinimumSize(120, 120)
            btn.setMaximumSize(160, 160)
            btn.clicked.connect(lambda checked=False, v=n: self._pick(v))
            self.buttons.append(btn)
            cards_row.addWidget(btn)
        cards_row.addStretch()

        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setAlignment(Qt.AlignCenter)

        self.confirm_btn = QPushButton("")
        self.confirm_btn.setObjectName("primary")
        set_button_icon(self.confirm_btn, "arrow-right", 18, COLORS["ink"])
        self.confirm_btn.setMinimumHeight(44)
        self.confirm_btn.clicked.connect(self._confirm)
        self.back_btn = QPushButton("")
        set_button_icon(self.back_btn, "arrow-left", 16)
        self.back_btn.clicked.connect(self.back_requested)

        lay.addStretch(1)
        lay.addWidget(self.title_label)
        lay.addSpacing(8)
        lay.addWidget(self.hint)
        lay.addWidget(self.song_info)
        lay.addSpacing(18)
        lay.addLayout(cards_row)
        lay.addSpacing(12)
        lay.addWidget(self.status)
        lay.addSpacing(18)
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()
        btn_row2.addWidget(self.back_btn)
        btn_row2.addWidget(self.confirm_btn)
        btn_row2.addStretch()
        lay.addLayout(btn_row2)
        lay.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrollable(content, self))

        self.retranslate("en")
        self._refresh()

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.title_label.setText(tr(code, "pcount.title"))
        self.hint.setText(tr(code, "pcount.hint"))
        labels = _count_labels(code)
        for n, btn in enumerate(self.buttons, start=1):
            btn.setText(f"{n}\n{labels[n]}")
        self.confirm_btn.setText(tr(code, "pcount.confirm"))
        self.back_btn.setText(tr(code, "pcount.back"))
        self._refresh()
        if self.song is not None:
            try:
                self.begin(self.song)
            except Exception:
                pass

    def begin(self, song) -> None:
        self.song = song
        nd = getattr(song, "num_dancers", 1) if song else 1
        title = getattr(song, "title", "") if song else ""
        artist = getattr(song, "artist", "") if song else ""
        code = self._lang
        if nd > 1:
            self.song_info.setText(tr(code, "pcount.song_multi", title=title, artist=artist, n=nd))
        else:
            self.song_info.setText(tr(code, "pcount.song_single", title=title, artist=artist))
        if self._selected < 1 or self._selected > 4:
            self._selected = 1
        self._refresh()

    def _pick(self, n: int) -> None:
        self._selected = int(n)
        self._refresh()

    def _refresh(self) -> None:
        for i, btn in enumerate(self.buttons, start=1):
            btn.setChecked(i == self._selected)
            btn.setProperty("checked", "true" if i == self._selected else "false")
            # Reinforce styling so the checked state shows without relying only on dynamic QSS.
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        nd = getattr(self.song, "num_dancers", 1) if self.song else 1
        code = self._lang
        if nd == 1 and self._selected > 1:
            self.status.setText(tr(code, "pcount.status_same", n=self._selected))
        elif nd > 1:
            if self._selected > 1:
                self.status.setText(tr(code, "pcount.status_each_many", n=self._selected, nd=nd))
            else:
                self.status.setText(tr(code, "pcount.status_each_one", n=self._selected, nd=nd))
        else:
            if self._selected > 1:
                self.status.setText(tr(code, "pcount.status_n_many", n=self._selected))
            else:
                self.status.setText(tr(code, "pcount.status_n_one", n=self._selected))

    def _confirm(self) -> None:
        self.count_selected.emit(int(self._selected))
