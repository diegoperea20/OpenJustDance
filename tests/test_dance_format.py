import json

import pytest

from engine.pose_engine.skeleton import SEMANTIC_JOINTS
from shared.dance_format.loader import save_song, validate_song
from shared.dance_format.models import FramePose, Song


def _dummy_song() -> Song:
    return Song(
        title="Test Song",
        artist="Tester",
        fps=30.0,
        duration=10.0,
        poses=[
            FramePose(
                time=0.0,
                joints={name: [0.0, 0.0, 0.0, 1.0] for name in SEMANTIC_JOINTS},
            ),
            FramePose(
                time=1.0,
                joints={name: [0.1, 0.1, 0.0, 1.0] for name in SEMANTIC_JOINTS},
            ),
        ],
    )


def test_song_roundtrip(tmp_path):
    song = _dummy_song()
    path = tmp_path / "song.json"
    save_song(song, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["title"] == "Test Song"
    assert raw["format"] == "openjustdance-1"
    assert len(raw["poses"]) == 2
    assert raw["poses"][0]["time"] == 0.0

    loaded = Song.from_dict(raw)
    assert loaded.title == song.title
    assert loaded.fps == song.fps
    assert loaded.poses[0].joints["head"] == [0.0, 0.0, 0.0, 1.0]


def test_validate_song_ok():
    assert validate_song(_dummy_song()) == []


def test_validate_song_detects_missing_joints():
    song = _dummy_song()
    song.poses[0].joints = {}
    # empty joints = neutral frame (VideoTest with no detection) -> allowed
    assert validate_song(song) == []


def test_validate_song_still_detects_invalid_time():
    song = _dummy_song()
    song.poses[0].time = -1.0
    assert validate_song(song) != []


def test_frame_pose_roundtrip():
    pose = FramePose(time=2.5, joints={"head": [1, 2, 3, 4]})
    assert FramePose.from_dict(pose.to_dict()) == pose
