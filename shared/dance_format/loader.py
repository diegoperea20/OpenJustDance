"""Song persistence and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from shared.dance_format.models import Song


def save_song(song: Song, path: Path) -> None:
    """Save the song as JSON (pretty) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(song.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_song(path: Path) -> Song:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Song.from_dict(data)


def video_aspect(video: Path) -> Optional[float]:
    """W/H aspect of a video (None if unreadable). Lazy cv2."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w / h
    except Exception:
        pass
    return None


def song_reference_aspect(song: Song) -> Optional[float]:
    """W/H aspect of a song's reference video.

    1) `reference_aspect` field (new songs).
    2) Probe of `song.mp4` next to `song.path` (old songs; assumes
       rotation 0/180 — with 90°/270° the true aspect is inverted, but old
       songs do not store the rotation used).
    3) None = unknown (no correction, legacy behavior).
    """
    try:
        ra = getattr(song, "reference_aspect", None)
        if ra is not None and float(ra) > 0:
            return float(ra)
    except (TypeError, ValueError):
        pass
    try:
        base = getattr(song, "path", None)
        if base is not None:
            cand = Path(base) / "song.mp4"
            if cand.is_file():
                return video_aspect(cand)
    except Exception:
        pass
    return None


def validate_song(song: Song) -> List[str]:
    """Return a list of errors (empty if the song is valid)."""
    errors: List[str] = []
    if not song.title:
        errors.append("title vacio")
    if song.duration <= 0 and not song.poses:
        errors.append("duracion invalida y sin poses")
    for i, pose in enumerate(song.poses):
        if pose.time < 0:
            errors.append(f"poses[{i}].time negativo")
        # empty joints = neutral frame (VideoTest with no detection) -> adds/subtracts no points
        # intentionally allowed, not a format error
        if not pose.joints:
            continue
    return errors


_SONG_CACHE: dict[str, tuple[float, int, Song]] = {}
"""Cache by (mtime, size): avoids re-parsing 50+ MB song.json on every
view refresh (it used to block the UI >1s per heavy song)."""


def find_songs(songs_dir: Path) -> List[Song]:
    """Scan `songs_dir` and load all valid songs."""
    songs: List[Song] = []
    songs_dir = Path(songs_dir)
    if not songs_dir.is_dir():
        return songs
    for song_dir in sorted(songs_dir.iterdir()):
        if not song_dir.is_dir():
            continue
        json_path = song_dir / "song.json"
        if not json_path.exists():
            continue
        try:
            stat = json_path.stat()
            key = str(json_path)
            cached = _SONG_CACHE.get(key)
            if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                song = cached[2]
                song.path = song_dir
                songs.append(song)
                continue
            song = load_song(json_path)
            song.path = song_dir
            # Do not cache corrupt/invalid songs; only ones that load cleanly.
            try:
                _SONG_CACHE[key] = (stat.st_mtime, stat.st_size, song)
            except Exception:
                pass
            songs.append(song)
        except Exception:
            continue
    return songs
