"""Responsive helpers for the desktop app (Qt, not CSS).

- ``FlowLayout``: wrapping layout (player/dancer cards).
- ``scrollable()``: wraps tall content in a stretchy QScrollArea.
- ``video_label()``: expansive-policy video QLabel (no fixed sizes).
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QScrollArea, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Fluid wrapping layout with centered rows.

    Each row is centered horizontally within the available width (case:
    dancer/player cards) instead of sticking to the left. With
    4 cards in a wide window they stay centered; on wrap (narrow
    window), each partial row stays centered too.
    """

    def __init__(self, parent=None, margin: int = 0, hspacing: int = 12,
                 vspacing: int = 12, center: bool = True) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h = hspacing
        self._v = vspacing
        self._center = bool(center)
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def clear(self) -> None:
        while self.count():
            item = self.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()

    def _do_layout(self, rect: QRect, test: bool) -> int:
        m = self.contentsMargins()
        avail = max(0, rect.width() - m.left() - m.right())
        # Pass 1: group items into rows with the same wrap rule.
        rows: list = []  # each row: [(item, w, h)]
        cur: list = []
        cur_w = 0
        for item in self._items:
            hint = item.sizeHint()
            w = max(hint.width(), item.minimumSize().width())
            h = max(hint.height(), item.minimumSize().height())
            if cur and cur_w + self._h + w > avail:
                rows.append(cur)
                cur = []
                cur_w = 0
            if cur:
                cur_w += self._h
            cur.append((item, w, h))
            cur_w += w
        if cur:
            rows.append(cur)
        # Pass 2: place each row centered within the available width.
        y = rect.y() + m.top()
        for i, row in enumerate(rows):
            row_w = sum(w for _, w, _ in row) + self._h * (len(row) - 1)
            line_h = max(h for _, _, h in row)
            x = rect.x() + m.left()
            if self._center and row_w < avail:
                x += (avail - row_w) // 2
            if not test:
                for item, w, h in row:
                    item.setGeometry(QRect(QPoint(x, y), QSize(w, h)))
                    x += w + self._h
            y += line_h
            if i < len(rows) - 1:
                y += self._v
        if not rows:
            return m.top() + m.bottom()
        return y - rect.y() + m.bottom()


def scrollable(content: QWidget, parent=None) -> QScrollArea:
    """Wrap ``content`` in a resizable QScrollArea (vertical responsive)."""
    area = QScrollArea(parent)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(content)
    content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    return area


def margins() -> QMargins:
    return QMargins(16, 16, 16, 16)
