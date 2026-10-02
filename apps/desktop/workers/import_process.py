"""Video import in a CHILD PROCESS (QProcess + file IPC).

Why a process instead of QThread: extraction (InsightFace/YOLO model
loading, inference, ONNX) can block for minutes inside native code where
cooperative cancellation never arrives, and ``QThread.terminate()`` on
Windows is ``TerminateThread``: when the thread dies holding the GIL it
freezes the WHOLE process (the UI "hangs" forever). Killing a child
process with ``QProcess.kill()`` is instant, safe and cannot hang the UI.

Protocol (see tools/pose_extractor/proc_entry.py):
* parent writes ``payload.json`` and launches ``<python> -m proc_entry payload``.
* child rewrites ``progress.json`` (atomic); the parent polls it at ~8Hz.
* cancel: the parent creates ``cancel.flag`` (fast cooperative path) and, when
  the child is still alive after 3s, ``kill()`` (OS guarantee). The UI answers
  instantly in both cases.
* success: the child leaves ``song.json``; the parent loads it with load_song
  and follows the normal flow (needs_decision / save_song_dir / succeeded).

The signal interface is identical to VideoImportThread so the view needs
no logic change: progress/status/succeeded/failed/cancelled/needs_decision
+ finished on exit (like QThread.finished).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from apps.desktop.workers.import_worker import save_song_dir
from shared.dance_format.loader import load_song

# Grace window for cooperative exit before the guaranteed kill.
KILL_GRACE_MS = 3000
POLL_MS = 120


class VideoImportProcess(QObject):
    """Runs extraction in a child process with guaranteed cancellation."""

    progress = Signal(int, int)  # current, total
    status = Signal(str)
    succeeded = Signal(object)  # Song
    failed = Signal(str)
    cancelled = Signal()  # user cancelled: do not save, show no error
    needs_decision = Signal(object, object, int, int)  # song, out_dir, declared, observed
    finished = Signal()  # always on exit (success, failure or cancellation)

    def __init__(
        self,
        video: Path,
        out_dir: Path,
        title: str,
        artist: str,
        fps: float | None = None,
        max_width: int = 1280,
        rotation: int = 0,
        num_dancers: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video = Path(video)
        self.out_dir = Path(out_dir)
        self.title = title
        self.artist = artist
        self.fps = fps
        self.max_width = max_width
        self.rotation = rotation
        self.num_dancers = max(1, min(4, int(num_dancers or 1)))
        self.payload_path = self.out_dir / "payload.json"
        self.progress_path = self.out_dir / "progress.json"
        self.cancel_path = self.out_dir / "cancel.flag"
        self.result_path = self.out_dir / "song.json"
        self.log_path = self.out_dir / "import.log"
        self._proc: QProcess | None = None
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._poll_progress)
        self._cancelling = False
        self._done = False
        self._last_msg = ""

    # ---------------- View-facing API ----------------
    def start(self) -> None:
        self._cancelling = False
        self._done = False
        self._last_msg = ""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.failed.emit(f"No se pudo crear {self.out_dir}: {exc}")
            self.finished.emit()
            return
        for stale in (self.cancel_path, self.progress_path,
                      self.progress_path.with_name(self.progress_path.name + ".tmp")):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
        payload = {
            "video": str(self.video),
            "out_dir": str(self.out_dir),
            "title": self.title,
            "artist": self.artist,
            "fps": self.fps,
            "max_width": self.max_width,
            "rotation": self.rotation,
            "num_dancers": self.num_dancers,
            "progress_path": str(self.progress_path),
            "cancel_path": str(self.cancel_path),
            "result_path": str(self.result_path),
        }
        try:
            self.payload_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            self.failed.emit(f"No se pudo escribir el payload: {exc}")
            self.finished.emit()
            return
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        try:
            self._proc.setStandardOutputFile(str(self.log_path))
        except Exception:
            pass
        self._proc.finished.connect(self._on_proc_finished)
        self._proc.errorOccurred.connect(self._on_proc_error)
        self.status.emit("Extrayendo poses...")
        self._proc.start(sys.executable, ["-m", "tools.pose_extractor.proc_entry",
                                          str(self.payload_path)])
        if not self._proc.waitForStarted(15000):
            self.failed.emit("No se pudo lanzar el proceso de importación.")
            self._finish_done()
            return
        self._poll.start()

    def cancel(self) -> None:
        """Requests cancellation. Returns instantly; the UI never blocks."""
        if self._done:
            return
        self._cancelling = True
        try:
            self.cancel_path.touch(exist_ok=True)
        except OSError:
            pass
        self.status.emit("Cancelando... (terminando)")
        # Guarantee: when the child does not exit cooperatively (e.g. stuck
        # loading models or inside a native call), OS kill().
        QTimer.singleShot(KILL_GRACE_MS, self._kill_if_running)

    def isRunning(self) -> bool:  # noqa: N802 (API Qt)
        proc = self._proc
        return bool(proc is not None and proc.state() == QProcess.Running)

    # ---------------- internals ----------------
    def _kill_if_running(self) -> None:
        proc = self._proc
        if proc is not None and proc.state() == QProcess.Running:
            print("[import] el hijo no salió en 3s tras cancelar; kill()")
            try:
                proc.kill()
                proc.waitForFinished(5000)
            except Exception:
                pass

    def _poll_progress(self) -> None:
        try:
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except Exception:
            return
        try:
            cur, total = int(data.get("i", 0)), int(data.get("total", 0))
        except Exception:
            return
        if total > 0:
            self.progress.emit(cur, total)
        if self._cancelling:
            return  # do not overwrite "Cancelling..." with stale child messages
        msg = str(data.get("msg", "") or "")
        if msg and msg != self._last_msg:
            self._last_msg = msg
            self.status.emit(msg)

    def _on_proc_error(self, err) -> None:
        if self._done:
            return
        # Only FailedToStart is a launch failure. Crashed/Timedout after
        # a cancellation kill() is resolved by _on_proc_finished.
        if err == QProcess.FailedToStart:
            if self._proc is not None and self._proc.state() == QProcess.NotRunning:
                self._fail("No se pudo lanzar el proceso de importación "
                           f"(ver {self.log_path}).")

    def _on_proc_finished(self, exit_code: int, _exit_status) -> None:
        if self._done:
            return
        self._poll.stop()
        if self._cancelling or int(exit_code) == 3:
            self._cleanup_partial()
            self.cancelled.emit()
            self._finish_done()
            return
        if int(exit_code) != 0:
            tail = self._log_tail()
            self._fail(f"La importación falló (código {int(exit_code)})."
                       + (f"\n{tail}" if tail else ""))
            return
        try:
            song = load_song(self.result_path)
        except Exception as exc:
            self._fail(f"No se pudo leer el resultado: {exc}")
            return
        if not song.poses:
            self._fail("No se detectó ninguna pose en el video.")
            return
        non_empty = sum(1 for p in song.poses if (p.joints or p.dancers))
        if non_empty == 0:
            self._fail("El video no contiene poses detectables. Prueba con otro "
                       "video con la persona visible y buena iluminación.")
            return
        observed = int(getattr(song, "num_dancers", 1) or 1)
        declared = int(self.num_dancers)
        if declared > 1 and observed < declared:
            self.status.emit(f"Se detectaron {observed} estables (declarados {declared})")
            self.needs_decision.emit(song, self.out_dir, declared, observed)
            self._finish_done(keep_pending=True)
            return
        self.status.emit("Guardando canción...")
        try:
            save_song_dir(song, self.out_dir, self.video, rotation=self.rotation)
        except Exception as exc:
            self._fail(f"No se pudo guardar: {exc}")
            return
        self._clean_transients()
        self.status.emit("Listo")
        self.succeeded.emit(song)
        self._finish_done()

    # ---------------- utilities ----------------
    def _fail(self, msg: str) -> None:
        self.failed.emit(msg)
        self._finish_done()

    def _finish_done(self, keep_pending: bool = False) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._poll.stop()
        except Exception:
            pass
        if not keep_pending:
            pass
        self.finished.emit()

    def _cleanup_partial(self) -> None:
        try:
            self._poll.stop()
        except Exception:
            pass
        # Close the log handle QProcess keeps open; on Windows
        # an open file blocks deleting the directory.
        proc = self._proc
        if proc is not None:
            try:
                proc.close()
            except Exception:
                pass
        self._rmtree_retry(0)

    def _rmtree_retry(self, attempt: int) -> None:
        try:
            shutil.rmtree(str(self.out_dir), ignore_errors=True)
        except Exception:
            pass
        # Async retry (no UI block): the OS may take a while to
        # release the freshly killed child's locks.
        if self.out_dir.exists() and attempt < 10:
            QTimer.singleShot(300, lambda: self._rmtree_retry(attempt + 1))

    def _clean_transients(self) -> None:
        for p in (self.payload_path, self.progress_path,
                  self.progress_path.with_name(self.progress_path.name + ".tmp"),
                  self.cancel_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def _log_tail(self, n: int = 6) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(l for l in lines[-n:] if l.strip())
        except OSError:
            return ""
