"""Neon Stage design system (tokens + QSS + helpers).

Palette: dark graphite room + cyan spotlight + magenta only for rewards
(the "signature": game progress and Perfect badge with cyan-to-magenta gradient).

Usage:
    from apps.desktop.ui.theme import APP_QSS, COLORS, FONTS, load_fonts

    app.setStyleSheet(APP_QSS)  # or window.setStyleSheet(APP_QSS)
"""

from __future__ import annotations

COLORS = {
    "bg0": "#0B0B10",      # app background
    "bg1": "#14141C",      # cards / surfaces
    "bg2": "#1D1D28",      # hover / raised
    "bg3": "#23232E",      # list item hover
    "line": "#2A2A38",     # borders
    "text1": "#F2F2F5",    # primary
    "text2": "#A6A6B8",    # secondary
    "text3": "#6B6B80",    # tertiary / hints
    "accent": "#00B0FF",   # cyan spotlight
    "accent_hi": "#33C2FF",
    "accent_dim": "#00344D",  # selection background
    "accent2": "#E040FB",  # magenta, rewards ONLY
    "success": "#00E676",
    "warn": "#FFB300",
    "danger": "#FF5252",
    "ink": "#0A0A0A",      # text on accent
}

FONTS = {
    # NORMAL width in all roles: condensed faces (Barlow Condensed,
    # Arial Narrow) stretched glyphs vertically. Segoe UI is the
    # native Windows face with normal proportions; Inter as an option.
    "display": "'Segoe UI', 'Inter', sans-serif",
    "body": "'Segoe UI', 'Inter', sans-serif",
    "mono": "'Cascadia Mono', 'JetBrains Mono', Consolas, monospace",
}

RADIUS = {"sm": 6, "md": 8, "lg": 12, "xl": 16}

APP_QSS = f"""
/* ===== Neon Stage: base ===== */
QMainWindow, QWidget {{ background: {COLORS['bg0']}; color: {COLORS['text1']};
    font-family: {FONTS['body']}; font-size: 14px; }}
QWidget#stage {{ background: {COLORS['bg0']}; }}
QLabel {{ background: transparent; }}

/* ===== Títulos ===== */
QLabel#display {{ font-family: {FONTS['display']}; font-weight: 800;
    letter-spacing: 2px; color: {COLORS['accent']}; }}
QLabel#title {{ font-size: 26px; font-weight: 700; color: {COLORS['text1']}; }}
QLabel#subtitle {{ font-size: 14px; color: {COLORS['text2']}; }}
QLabel#hint {{ font-size: 12px; color: {COLORS['text2']}; }}
QLabel#micro {{ font-size: 11px; color: {COLORS['text3']}; }}
QLabel#mono {{ font-family: {FONTS['mono']}; font-size: 12px; color: {COLORS['text2']}; }}

/* ===== Botones ===== */
QPushButton {{ background: {COLORS['bg2']}; border: 1px solid {COLORS['line']};
    border-radius: {RADIUS['md']}px; padding: 10px 22px; font-weight: 600;
    color: {COLORS['text1']}; }}
QPushButton:hover {{ background: {COLORS['bg3']}; border-color: {COLORS['accent']}; }}
QPushButton:focus {{ border: 1px solid {COLORS['accent']}; outline: none; }}
QPushButton:pressed {{ background: {COLORS['bg1']}; }}
QPushButton:disabled {{ color: {COLORS['text3']}; background: {COLORS['bg1']};
    border-color: {COLORS['line']}; }}
QPushButton#primary {{ background: {COLORS['accent']}; color: {COLORS['ink']};
    border: 1px solid {COLORS['accent']}; }}
QPushButton#primary:hover {{ background: {COLORS['accent_hi']};
    border-color: {COLORS['accent_hi']}; }}
QPushButton#primary:disabled {{ background: {COLORS['bg2']}; color: {COLORS['text3']};
    border-color: {COLORS['line']}; }}
QPushButton#danger {{ background: #2A1215; border: 1px solid #7A2020;
    color: #FF8A80; }}
QPushButton#danger:hover {{ background: #3A1818; border-color: {COLORS['danger']}; }}
QPushButton#ghost {{ background: transparent; border: 1px solid {COLORS['line']}; }}
QPushButton#countCard {{ background: {COLORS['bg1']}; border: 2px solid {COLORS['line']};
    border-radius: {RADIUS['xl']}px; font-family: {FONTS['display']};
    font-size: 42px; font-weight: 800; }}
QPushButton#countCard[checked="true"] {{ background: {COLORS['accent_dim']};
    border: 2px solid {COLORS['accent']}; color: {COLORS['accent']}; }}

/* ===== Tarjetas ===== */
QFrame#card {{ background: {COLORS['bg1']}; border: 1px solid {COLORS['line']};
    border-radius: {RADIUS['lg']}px; }}
QFrame#dropZone {{ background: {COLORS['bg1']}; border: 2px dashed {COLORS['line']};
    border-radius: {RADIUS['lg']}px; padding: 10px; }}
QFrame#dropZone:hover {{ border-color: {COLORS['accent']}; background: {COLORS['bg2']}; }}
QFrame#dropZone[dragOver="true"] {{ background: {COLORS['accent_dim']};
    border: 2px dashed {COLORS['accent']}; }}
QFrame#cardHi {{ background: {COLORS['accent_dim']};
    border: 2px solid {COLORS['success']}; border-radius: 10px; }}
QFrame#video {{ background: #000000; border: 1px solid {COLORS['line']};
    border-radius: {RADIUS['lg']}px; }}

/* ===== Listas ===== */
QListWidget {{ background: {COLORS['bg1']}; border: 1px solid {COLORS['line']};
    border-radius: {RADIUS['lg']}px; padding: 6px; outline: none; }}
QListWidget::item {{ padding: 10px 12px; border-radius: {RADIUS['md']}px;
    margin: 2px 4px 2px 2px; }}
QListWidget::item:hover:!selected {{ background: {COLORS['bg3']}; }}
QListWidget::item:selected {{ background: {COLORS['accent']}; color: {COLORS['ink']};
    font-weight: 600; }}

/* ===== Scrollbars neon (finas, sin flechas) ===== */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px 4px 0px; }}
QScrollBar::handle:vertical {{ background: #33333F; min-height: 36px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {COLORS['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0px 4px 2px 4px; }}
QScrollBar::handle:horizontal {{ background: #33333F; min-width: 36px; border-radius: 5px; }}
QScrollBar::handle:horizontal:hover {{ background: {COLORS['accent']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

/* ===== Progreso: firma cyan→magenta solo en juego ===== */
QProgressBar {{ border: 1px solid {COLORS['line']}; border-radius: 6px;
    background: {COLORS['bg1']}; text-align: center; color: {COLORS['text2']}; }}
QProgressBar::chunk {{ background: {COLORS['accent']}; border-radius: 5px; }}
QProgressBar#spotlight::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 {COLORS['accent2']}); border-radius: 5px; }}

/* ===== Inputs ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {COLORS['bg1']};
    border: 1px solid {COLORS['line']}; border-radius: {RADIUS['sm']}px; padding: 8px 10px;
    selection-background-color: {COLORS['accent']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {COLORS['accent']}; background: {COLORS['bg2']}; }}
QLineEdit#search {{ border-radius: 10px; padding: 9px 12px; }}

/* ===== Slider (comparativa) ===== */
QSlider::groove:horizontal {{ height: 6px; background: {COLORS['bg2']};
    border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; margin: -6px 0; border-radius: 8px;
    background: {COLORS['accent']}; }}
QSlider::handle:horizontal:hover {{ background: {COLORS['accent_hi']}; }}

/* ===== Avisos ===== */
QLabel#warn {{ color: {COLORS['warn']}; background: #2A1E00;
    border-radius: 6px; padding: 6px; font-weight: 600; }}
QLabel#dangerText {{ color: #FF8A80; }}
QLabel#ok {{ color: {COLORS['success']}; font-weight: 600; }}

/* ===== Rating badge ===== */
QLabel#rating {{ font-family: {FONTS['display']}; font-size: 30px; font-weight: 800;
    border-radius: 8px; background: rgba(0,0,0,0.35); padding: 4px 16px; }}
QLabel#score {{ font-family: {FONTS['display']}; font-size: 72px; font-weight: 800;
    color: {COLORS['accent']}; }}
QLabel#stars {{ font-size: 40px; color: #FFD700; }}
QLabel#countdown {{ font-family: {FONTS['display']}; font-size: 120px;
    font-weight: 800; color: {COLORS['accent']}; }}

/* ===== StatusBar / dialogs ===== */
QStatusBar {{ background: {COLORS['bg1']}; color: {COLORS['text2']}; }}
QProgressDialog {{ background: {COLORS['bg1']}; }}
QSplitter::handle {{ background: {COLORS['line']}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QToolTip {{ background: {COLORS['bg2']}; color: {COLORS['text1']};
    border: 1px solid {COLORS['line']}; padding: 4px; }}
"""


def load_fonts() -> None:
    """Register optional fonts next to assets (if present).

    The theme uses the system Segoe UI by default; if
    ``assets/fonts/Inter-*.ttf`` is added it is used as a fallback.
    Never blocks or downloads anything.
    """
    try:
        from PySide6.QtGui import QFontDatabase

        base = Path(__file__).resolve().parent / "assets" / "fonts"
        if base.is_dir():
            for ttf in base.glob("*.ttf"):
                QFontDatabase.addApplicationFont(str(ttf))
    except Exception:
        pass
