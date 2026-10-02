"""Temporal pose smoothing (exponential moving average) to remove noise."""

from __future__ import annotations

import numpy as np

from engine.pose_engine.skeleton import Joint, Pose


class PoseSmoother:
    """Smooths a pose sequence with an exponential moving average (EMA)."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar entre 0 y 1")
        self.alpha = alpha
        self._state: dict[str, np.ndarray] = {}

    def smooth(self, pose: Pose) -> Pose:
        out = Pose()
        for name, joint in pose.joints.items():
            vec = np.array([joint.x, joint.y, joint.z], dtype=float)
            prev = self._state.get(name)
            if prev is None:
                smoothed = vec.copy()
            else:
                smoothed = self.alpha * vec + (1.0 - self.alpha) * prev
            self._state[name] = smoothed
            out.joints[name] = Joint(
                x=float(smoothed[0]),
                y=float(smoothed[1]),
                z=float(smoothed[2]),
                visibility=joint.visibility,
            )

        # When the body disappears, drop stale joint state.
        seen = set(pose.joints.keys())
        for name in list(self._state.keys()):
            if name not in seen:
                del self._state[name]
        return out

    def reset(self) -> None:
        self._state.clear()
