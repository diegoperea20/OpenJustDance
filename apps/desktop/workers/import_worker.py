"""Worker importing a video as a song without blocking the UI."""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from shared.dance_format.loader import save_song
from tools.pose_extractor.extract import (
    CancelledError,
    ExtractOptions,
    extract,
    extract_audio,
    make_cover,
)


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "cancion"


def unique_song_dir(base: Path, slug: str) -> Path:
    """Returns a unique directory inside base (slug, slug_2, ...)."""
    candidate = base / slug
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = base / f"{slug}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def save_song_dir(song, out_dir: Path, video: Path, rotation: int = 0) -> None:
    """Saves song.json + mp4 + cover + mp3 + poses.jsonl into out_dir."""
    from shared.dance_format.loader import save_song, video_aspect

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Seal the reference video aspect: required for the
    # aspect fix in comparisons (vertical vs horizontal camera).
    # Watch rotation: poses are extracted from the ROTATED frame; at 90°/270°
    # the baked aspect is the inverse of the original file.
    try:
        if getattr(song, "reference_aspect", None) is None:
            ar = video_aspect(video)
            if ar is not None and int(rotation or 0) % 180 != 0:
                ar = 1.0 / ar
            song.reference_aspect = ar
    except Exception:
        pass
    save_song(song, out_dir / "song.json")
    song.path = out_dir
    try:
        shutil.copy2(video, out_dir / "song.mp4")
    except OSError as exc:
        print(f"[import] aviso no se pudo copiar video: {exc}")
    try:
        make_cover(video, out_dir / "cover.png", rotation=rotation)
    except Exception as exc:
        print(f"[import] aviso caratula: {exc}")
    try:
        extract_audio(video, out_dir / "song.mp3")
    except Exception as exc:
        print(f"[import] aviso audio (ffmpeg?): {exc}")


class VideoImportThread(QThread):
    """Extracts poses and builds the song folder in the background."""

    progress = Signal(int, int)  # current, total
    status = Signal(str)
    succeeded = Signal(object)  # Song
    failed = Signal(str)
    cancelled = Signal()  # user cancelled: do not save, show no error
    needs_decision = Signal(object, object, int, int)  # song, out_dir, declared, observed

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
        self._cancel_flag = False

    def cancel(self) -> None:
        """Requests cooperative cancellation (non-blocking; the loop sees it on the next frame)."""
        self._cancel_flag = True
        try:
            self.requestInterruption()
        except Exception:
            pass

    def is_cancelled(self) -> bool:
        """True when the user requested cancellation (own flag + Qt flag)."""
        try:
            if self.isInterruptionRequested():
                return True
        except Exception:
            pass
        return bool(self._cancel_flag)

    def run(self) -> None:
        try:
            if not self.video.is_file():
                self.failed.emit(f"No existe el video: {self.video}")
                return

            self.status.emit("Extrayendo poses...")
            # Same 1-4 process (trackingdancers/projectv.py style):
            # always yolo11m-pose.pt + mandatory FaceID + poses.jsonl.
            options = ExtractOptions(
                video=self.video,
                title=self.title,
                artist=self.artist,
                fps=self.fps,
                max_width=self.max_width,
                rotation=self.rotation,
                num_dancers=self.num_dancers,
                multi_max=self.num_dancers,
                yolo_model="yolo11m-pose.pt",
                face_id=True,
                yolo_conf=0.25,
                imgsz=960,
                preview_path=(self.out_dir / "preview_pose.mp4"),
                dump_poses_path=(self.out_dir / "poses.json"),
                debug_track_path=(self.out_dir / "track_debug.jsonl"),
                poses_jsonl_path=(self.out_dir / "poses.jsonl"),
            )

            def on_progress(i: int, total: int) -> bool:
                self.progress.emit(i, total)
                # True = abort requested; extract turns it into CancelledError.
                return self.is_cancelled()

            try:
                song = extract(options, on_progress=on_progress,
                               should_cancel=self.is_cancelled)
            except CancelledError:
                self.cancelled.emit()
                return

            # Cancel may have arrived between the last frame and here: do not save.
            if self.is_cancelled():
                self.cancelled.emit()
                return

            if not song.poses:
                self.failed.emit("No se detectó ninguna pose en el video.")
                return

            # Stats for the warning
            non_empty = sum(1 for p in song.poses if (p.joints or p.dancers))
            if non_empty == 0:
                self.failed.emit(
                    "El video no contiene poses detectables. Prueba con otro video con la persona visible y buena iluminación."
                )
                return

            observed = int(getattr(song, "num_dancers", 1) or 1)
            declared = int(self.num_dancers)
            if declared > 1 and observed < declared:
                # Mismatch: do not save yet; the UI asks Keep/Cancel.
                self.status.emit(f"Se detectaron {observed} estables (declarados {declared})")
                self.needs_decision.emit(song, self.out_dir, declared, observed)
                return

            self.status.emit("Guardando canción...")
            save_song_dir(song, self.out_dir, self.video, rotation=self.rotation)

            self.status.emit("Listo")
            self.succeeded.emit(song)

        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.failed.emit(f"Error al importar: {exc}")
