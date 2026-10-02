"""Dance format: defines the project song standard.

Song directory layout:

    songs/<nombre>/
        song.mp4       Choreography video
        song.mp3       Audio
        cover.png      Cover art
        song.json      Choreography (normalized poses + metadata)

song.json:

    {
        "title": "Dance Song",
        "artist": "Unknown",
        "fps": 30,
        "duration": 180.0,
        "format": "openjustdance-1",
        "joints_order": ["head", "left_shoulder", ...],
        "poses": [
            {"time": 0.0, "joints": {"head": [x, y, z, vis], ...}},
            ...
        ]
    }

Poses are normalized to canonical space (center = hips, scale =
shoulder width, hips aligned horizontally, y up), making the format
independent of the detection backend.
"""

from shared.dance_format.models import FramePose, Song
from shared.dance_format.loader import (
    find_songs,
    load_song,
    save_song,
    validate_song,
)

__all__ = ["FramePose", "Song", "save_song", "load_song", "validate_song", "find_songs"]
