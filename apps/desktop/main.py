"""Desktop app entry point.

Usage:
    uv run openjustdance
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication

    from apps.desktop.ui.views import MainWindow

    app = QApplication(sys.argv if argv is None else list(argv))
    app.setApplicationName("Open Just Dance")
    app.setOrganizationName("openjustdance")
    try:
        from apps.desktop.ui.icons import brand_icon

        _icon = brand_icon()
        if not _icon.isNull():
            app.setWindowIcon(_icon)
    except Exception:
        pass

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
