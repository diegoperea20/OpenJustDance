"""Reusable UI widgets (Neon Stage)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from apps.desktop.ui.theme import COLORS

RATING_COLORS = {
    "Perfect": "#E040FB",  # signature: reward magenta
    "Great": "#00E676",
    "Good": "#FFD600",
    "Ok": "#FF9100",
    "Miss": "#FF5252",
}


class RatingBadge(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__("-", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(48)
        self.setObjectName("rating")
        self.set_rating("")

    def set_rating(self, rating: str) -> None:
        color = RATING_COLORS.get(rating, COLORS["text3"])
        self.setText(rating if rating else "-")
        # Dynamic color over the theme base style (#rating).
        self.setStyleSheet(
            f"QLabel#rating {{ color: {color}; }}"
            + ("QLabel#rating { border: 1px solid #E040FB; }" if rating == "Perfect" else "")
        )


class StatBox(QWidget):
    """Box with value and title labels."""

    def __init__(self, title: str, value: str = "-", parent=None) -> None:
        super().__init__(parent)
        self.value_label = QLabel(value)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setObjectName("display")
        self.value_label.setStyleSheet("font-size: 30px; color: #ffffff;")
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("micro")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.value_label)
        layout.addWidget(title_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
