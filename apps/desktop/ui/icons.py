"""Central SVG icon loader (vendored Lucide, no dependencies).

SHARP rendering: each icon is rasterized with ``QSvgRenderer`` at the exact
requested size (with x3 supersampling for smooth HiDPI edges) instead of
loading a generic QPixmap and rescaling it (which left them blurry).

Automatic contrast:
- Dark surfaces (plain buttons, backgrounds) → WHITE icon ``#F2F2F5``.
- Blue ``#primary`` buttons → BLACK icon ``#0A0A0A`` (pass color).

Usage:
    from apps.desktop.ui.icons import icon, set_button_icon

    set_button_icon(btn, "search")                    # dark background → white
    set_button_icon(btn, "play", color=COLORS["ink"])  # blue background → black
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"

#: Brand SVGs (logo + app icon) in the repo-root ``assets/``.
_BRAND_DIR = Path(__file__).resolve().parents[3] / "assets"

#: Supersampling: rasterize at size x RATIO and tag the DPR, so Qt
#: draws the icon at its logical size but with x3 detail (sharp).
RENDER_RATIO = 3

#: Default white: contrasts over the app's dark backgrounds.
LIGHT = "#F2F2F5"
#: Black for icons on the blue accent ( #primary buttons).
INK = "#0A0A0A"

DEFAULT_SIZE = 18

_svg_cache: dict[str, str] = {}
_pix_cache: dict[tuple[str, int, str], QPixmap] = {}
_icon_cache: dict[tuple[str, int, str], QIcon] = {}
_tinted_files: dict[tuple[str, str], str] = {}


def available(name: str) -> bool:
    return (_ICON_DIR / f"{name}.svg").exists()


def names() -> list[str]:
    try:
        return sorted(p.stem for p in _ICON_DIR.glob("*.svg"))
    except OSError:
        return []


def _svg_text(name: str) -> str | None:
    if name in _svg_cache:
        return _svg_cache[name]
    try:
        text = (_ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    except OSError:
        return None
    _svg_cache[name] = text
    return text


def _paint(name: str, size: int, color: str) -> QPixmap | None:
    """Rasterize the SVG to ``size`` logical px with color applied and x3 detail."""
    src = _svg_text(name)
    if not src:
        return None
    # currentColor → exact color BEFORE rasterizing: crisp edges,
    # without the blur of the SourceIn trick over an already rasterized pixmap.
    data = src.replace("currentColor", color)
    renderer = QSvgRenderer(bytearray(data, "utf-8"))
    if not renderer.isValid():
        return None
    px = max(1, int(size) * RENDER_RATIO)
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(float(RENDER_RATIO))
    return pm


def pixmap(name: str, size: int = DEFAULT_SIZE, color: str | None = None) -> QPixmap:
    """Sharp QPixmap of the icon.

    ``color=None`` → white (contrast over dark backgrounds).
    Pass ``INK``/``COLORS["ink"]`` for icons on a blue background.
    """
    key = (name, int(size), color or "")
    cached = _pix_cache.get(key)
    if cached is not None and not cached.isNull():
        return cached
    pm = _paint(name, int(size), color or LIGHT)
    if pm is None or pm.isNull():
        pm = QPixmap(max(1, int(size)), max(1, int(size)))
        pm.fill(Qt.transparent)
    _pix_cache[key] = pm
    return pm


def icon(name: str, size: int = DEFAULT_SIZE, color: str | None = None) -> QIcon:
    """Cached QIcon for buttons and lists."""
    key = (name, int(size), color or "")
    cached = _icon_cache.get(key)
    if cached is not None and not cached.isNull():
        return cached
    ic = QIcon(pixmap(name, size, color))
    _icon_cache[key] = ic
    return ic


def set_button_icon(btn, name: str, size: int = DEFAULT_SIZE, color: str | None = None) -> None:
    """Assign an SVG icon to a QPushButton/QToolButton (leaves text untouched).

    Without ``color`` the icon renders white (dark backgrounds). On
    ``#primary`` (blue background) buttons pass ``color=COLORS["ink"]`` (black).
    """
    try:
        btn.setIcon(icon(name, size=size, color=color))
        btn.setIconSize(QSize(int(size), int(size)))
    except Exception:
        pass


#: Brand logo width in the menu (logical px).
BRAND_LOGO_WIDTH = 500


def _brand_path(name: str) -> Path | None:
    """Path to ``assets/<name>.svg`` if it exists, else ``None``."""
    try:
        path = _BRAND_DIR / f"{name}.svg"
    except Exception:
        return None
    return path if path.exists() else None


def brand_pixmap(name: str, width: int = BRAND_LOGO_WIDTH) -> QPixmap | None:
    """Rasterize a brand SVG (``assets/``) to ``width`` px wide.

    Keeps the ``viewBox`` aspect ratio and uses x3 supersampling
    (same as the Lucide icons). Returns ``None`` when the SVG is
    missing or invalid (the caller must use a fallback).
    """
    if width is None or int(width) <= 0:
        return None
    path = _brand_path(name)
    if path is None:
        return None
    try:
        renderer = QSvgRenderer(str(path))
        if not renderer.isValid():
            return None
        view = renderer.viewBoxF()
        if not view.isValid() or view.width() <= 0:
            size = renderer.defaultSize()
            if size.width() <= 0 or size.height() <= 0:
                return None
            aspect = size.height() / size.width()
        else:
            aspect = view.height() / view.width()
        h = max(1, round(int(width) * aspect))
        px_w, px_h = max(1, int(width) * RENDER_RATIO), max(1, h * RENDER_RATIO)
        img = QImage(px_w, px_h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        painter = QPainter(img)
        try:
            renderer.render(painter)
        finally:
            painter.end()
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(float(RENDER_RATIO))
        return pm
    except Exception:
        return None


def brand_icon(name: str = "open-just-dance-icon",
               sizes: tuple[int, ...] = (16, 32, 48, 64, 128, 256)) -> QIcon:
    """Multi-size QIcon for the window/app from ``assets/<name>.svg``.

    When the SVG is unavailable, returns an empty QIcon.
    """
    ic = QIcon()
    for size in sizes:
        pm = brand_pixmap(name, width=int(size))
        if pm is not None and not pm.isNull():
            ic.addPixmap(pm)
    return ic


def tinted_path(name: str, color: str) -> str:
    """Path to a copy of the SVG with the color already applied.

    For places where QIcon/QPixmap cannot be used (e.g. ``<img>``
    inside a QLabel's rich text): the source SVG uses
    ``currentColor`` and Qt would paint it black (no contrast).
    Returns ``""`` when the icon does not exist.
    """
    key = (name, color)
    cached = _tinted_files.get(key)
    if cached and Path(cached).exists():
        return cached
    src = _svg_text(name)
    if not src:
        return ""
    try:
        from PySide6.QtCore import QDir

        tmp = Path(QDir.tempPath()) / "openjustdance-icons"
        tmp.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in color)
        dest = tmp / f"{name}-{safe}.svg"
        if not dest.exists():
            dest.write_text(src.replace("currentColor", color), encoding="utf-8")
        _tinted_files[key] = str(dest)
        return str(dest)
    except OSError:
        return ""
