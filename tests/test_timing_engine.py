import pytest

from engine.timing_engine import SongSynchronizer
from shared.dance_format.models import FramePose, Song


def _song():
    return Song(
        title="t",
        artist="a",
        fps=30.0,
        duration=3.0,
        poses=[
            FramePose(time=0.0, joints={"head": [0, 0, 0, 1]}),
            FramePose(time=1.0, joints={"head": [1, 0, 0, 1]}),
            FramePose(time=2.0, joints={"head": [2, 0, 0, 1]}),
        ],
    )


def test_pose_before_first():
    sync = SongSynchronizer(_song())
    assert sync.pose_at(0.0).time == 0.0
    assert sync.pose_at(-5.0).time == 0.0


def test_pose_after_last():
    sync = SongSynchronizer(_song())
    assert sync.pose_at(100.0).time == 2.0


def test_nearest_by_time():
    sync = SongSynchronizer(_song())
    # between 0.0 and 1.0 -> closer to 1.0
    assert sync.pose_at(0.9).time == 1.0
    assert sync.pose_at(0.4).time == 0.0
    # on the boundary -> picks the lower index
    assert sync.pose_at(0.5).time == 0.0


def test_exact_time():
    sync = SongSynchronizer(_song())
    assert sync.pose_at(1.0).time == 1.0


def test_empty_song():
    sync = SongSynchronizer(Song(title="t", artist="a", fps=30.0, duration=0.0))
    assert sync.pose_at(0.0) is None
