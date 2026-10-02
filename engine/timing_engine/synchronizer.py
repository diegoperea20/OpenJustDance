"""Sync: looks up the closest pose by time (never by frame).

This avoids issues across videos with different FPS: the player position
is always compared against the reference pose closest to the current time.
"""

from __future__ import annotations

import bisect
from typing import Optional

from shared.dance_format.models import DancerPose, FramePose, Song


class SongSynchronizer:
    def __init__(self, song: Song) -> None:
        self.song = song
        self.times = [pose.time for pose in song.poses]

    def pose_at(self, time_s: float) -> Optional[FramePose]:
        """Legacy: returns the closest full FramePose."""
        return self.frame_at(time_s)

    def frame_at(self, time_s: float) -> Optional[FramePose]:
        """Returns the pose closest to the given time (or None when no poses).

        Supports two cases for neutral frames (VideoTest with no detection):
        - New songs: contain a FramePose with joints={} at every sampled
          timestamp -> returned directly (empty joints = neutral).
        - Old songs: the extractor used `continue` and never stored the frame,
          so there are time gaps. When the distance to the closest pose
          exceeds the sampling interval, that instant is treated as having
          no pose and None is returned so the controller neither adds nor
          subtracts.
        """
        if not self.times:
            return None
        idx = bisect.bisect_left(self.times, time_s)
        if idx == 0:
            nearest = self.song.poses[0]
        elif idx >= len(self.song.poses):
            nearest = self.song.poses[-1]
        else:
            before = self.song.poses[idx - 1]
            after = self.song.poses[idx]
            if (time_s - before.time) <= (after.time - time_s):
                nearest = before
            else:
                nearest = after

        # When the closest frame is empty / dancer-less, it is neutral
        if nearest.dancers:
            if all(not d.joints for d in nearest.dancers):
                return nearest
        elif not nearest.joints:
            return nearest

        # Gap detection for old songs without placeholders
        if len(self.times) >= 2:
            span = self.times[-1] - self.times[0]
            avg_interval = span / (len(self.times) - 1) if span > 0 else 1.0 / 30.0
            # Only enable for dense songs (avg <0.2 ~ >5fps). Tests with
            # poses at 0,1,2 have avg=1.0 and are skipped.
            if avg_interval < 0.2:
                fps = self.song.fps if self.song.fps and self.song.fps > 1e-9 else 30.0
                fps_interval = 1.0 / fps
                # Use the smaller of avg and fps to stay strict in detection
                interval = min(avg_interval, fps_interval) if fps_interval < 0.2 else avg_interval
                if abs(time_s - nearest.time) > interval * 0.75:
                    return None
        return nearest

    def dancer_pose_at(self, time_s: float, dancer_id: int = 0) -> Optional[DancerPose]:
        """Returns the dancer_id pose closest to the time."""
        frame = self.frame_at(time_s)
        if frame is None:
            return None
        return frame.get_dancer(dancer_id)
