"""Webcam capture with OpenCV."""

from __future__ import annotations

import platform
from typing import Optional

import numpy as np


class Camera:
    """Wrapper over cv2.VideoCapture with backend fallbacks for Windows."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap = None
        self._last_error: Optional[str] = None

    def open(self) -> bool:
        try:
            import cv2
        except ImportError:
            self._last_error = "opencv-python no esta instalado"
            return False

        kwargs = {}
        if platform.system() == "Windows":
            kwargs["apiPreference"] = cv2.CAP_DSHOW

        cap = cv2.VideoCapture(self.index, **kwargs)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            # last resort with MSMF (integrated laptop cams usually prefer it)
            try:
                cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
            except Exception:
                pass
        if not cap.isOpened():
            self._last_error = f"No se pudo abrir la camara {self.index}"
            return False

        if self.width and self.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # verify negotiation: if it stays 0x0 or too small, revert to native
            try:
                aw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                ah = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if (aw < 160 or ah < 120):
                    # let the driver use native resolution
                    pass
            except Exception:
                pass

        self._cap = cap
        return True

    def probe_gray(self, frame) -> bool:
        """Detect a uniform gray frame (typical of badly fixed manual exposure on integrated cams)."""
        try:
            import numpy as np
            if frame is None or frame.size == 0:
                return False
            # mean ~128 and very low std => gray
            std = float(np.std(frame))
            mean = float(np.mean(frame))
            return (std < 8.0 and 115.0 < mean < 140.0)
        except Exception:
            return False

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class _SilenceOpenCVLogs:
    """Silence videoio WARN/ERROR output while probing cameras.

    OpenCV dumps console messages like "VIDEOIO(DSHOW): backend is
    generally available..." for every missing index. The level is set
    to SILENT and the previous one is restored on exit (context manager).
    """

    def __init__(self) -> None:
        self._cv2 = None
        self._set_level = None
        self._prev = None

    def __enter__(self) -> "_SilenceOpenCVLogs":
        try:
            import cv2 as _cv2
        except ImportError:
            return self
        self._cv2 = _cv2
        # OpenCV 4.x exposes setLogLevel at top level (cv2.setLogLevel);
        # on other builds it lives in cv2.utils.logging.
        get_level = getattr(_cv2, "getLogLevel", None)
        set_level = getattr(_cv2, "setLogLevel", None)
        silent = getattr(_cv2, "LOG_LEVEL_SILENT", 0)
        if get_level is None or set_level is None:
            try:
                import cv2.utils.logging as _log
                get_level, set_level = _log.getLogLevel, _log.setLogLevel
                silent = _log.LOG_LEVEL_SILENT
            except Exception:
                return self
        self._set_level = set_level
        try:
            self._prev = get_level()
            set_level(silent)
        except Exception:
            self._prev = None
        return self

    def __exit__(self, *exc) -> None:
        if self._set_level is not None and self._prev is not None:
            try:
                self._set_level(self._prev)
            except Exception:
                pass


def list_available_cameras(max_index: int = 5) -> list[tuple[int, str]]:
    """Enumerate available cameras by probing indices 0..max_index-1.

    Returns a list of (index, label). The label includes the negotiated
    resolution when readable (e.g. "Cámara 1 (1280x720)").
    Returns [] if OpenCV is not installed or there are no cameras.

    Silences OpenCV logs during probing and skips the
    "default" attempt on Windows (there it equals MSMF: it was a redundant open
    per missing index).
    """
    try:
        import cv2
    except ImportError:
        return []
    try:
        max_index = max(1, min(10, int(max_index)))
    except Exception:
        max_index = 5
    is_windows = platform.system() == "Windows"
    found: list[tuple[int, str]] = []
    with _SilenceOpenCVLogs():
        for index in range(max_index):
            label: Optional[str] = None
            # Same order as Camera.open(): DSHOW first on Windows and
            # MSMF as fallback. On Windows the "default" backend is
            # MSMF, so probing it separately doubled the time per
            # missing index.
            attempts = []
            if is_windows:
                try:
                    attempts.append(cv2.CAP_DSHOW)
                except Exception:
                    pass
                try:
                    attempts.append(cv2.CAP_MSMF)
                except Exception:
                    pass
            else:
                attempts.append(None)  # default
            for api in attempts:
                cap = None
                try:
                    if api is None:
                        cap = cv2.VideoCapture(index)
                    else:
                        cap = cv2.VideoCapture(index, api)
                    if cap is None or not cap.isOpened():
                        continue
                    # Confirm it delivers at least one frame.
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    try:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    except Exception:
                        w, h = 0, 0
                    if w > 0 and h > 0:
                        label = f"Camara {index} ({w}x{h})"
                    else:
                        label = f"Camara {index}"
                    break
                except Exception:
                    continue
                finally:
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
            if label is not None:
                found.append((index, label))
    return found
