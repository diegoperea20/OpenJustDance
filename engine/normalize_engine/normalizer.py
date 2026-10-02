"""Body normalization into a shared reference frame.

Absolute positions are never compared. Every pose is mapped to canonical
space: the origin is the hip center, scale is shoulder width,
tilt is fixed by aligning hips to the horizontal
and the Y axis points up.
"""

from __future__ import annotations

import math
from typing import Optional

from engine.pose_engine.skeleton import Joint, Pose


class BodyNormalizer:
    """Normalizes a raw pose (0..1 image coords) to canonical space."""

    def __init__(self, visibility_threshold: float = 0.4, z_scale: bool = True) -> None:
        self.visibility_threshold = visibility_threshold
        self.z_scale = z_scale

    def _pair(self, pose: Pose, a: str, b: str) -> tuple[Optional[Joint], Optional[Joint]]:
        return pose.get(a), pose.get(b)

    def _visible(self, pose: Pose, name: str) -> bool:
        return pose.visible(name, self.visibility_threshold)

    def _center_and_scale(self, pose: Pose):
        """Returns (cx, cy, scale) or None when too little body is visible."""
        lh, rh = self._pair(pose, "left_hip", "right_hip")
        ls, rs = self._pair(pose, "left_shoulder", "right_shoulder")

        has_hips = self._visible(pose, "left_hip") and self._visible(pose, "right_hip")
        has_shoulders = self._visible(pose, "left_shoulder") and self._visible(
            pose, "right_shoulder"
        )
        if not (has_hips or has_shoulders):
            return None

        if has_hips:
            cx = (lh.x + rh.x) / 2.0
            cy = (lh.y + rh.y) / 2.0
        else:
            cx = (ls.x + rs.x) / 2.0
            cy = (ls.y + rs.y) / 2.0

        scale = 1.0
        if has_shoulders:
            scale = math.hypot(rs.x - ls.x, rs.y - ls.y)
        if scale < 1e-6:
            # Fallback: torso length.
            if has_hips and has_shoulders:
                mid_sx = (ls.x + rs.x) / 2.0
                mid_sy = (ls.y + rs.y) / 2.0
                scale = math.hypot(mid_sx - cx, mid_sy - cy) * 2.0
        if scale < 1e-6:
            return None
        return cx, cy, scale

    def _rotation_angle(self, pose: Pose) -> float:
        """Hip-line angle vs the horizontal (radians)."""
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        if lh is not None and rh is not None:
            return math.atan2(rh.y - lh.y, rh.x - lh.x)
        ls = pose.get("left_shoulder")
        rs = pose.get("right_shoulder")
        if ls is not None and rs is not None:
            return math.atan2(rs.y - ls.y, rs.x - ls.x)
        return 0.0

    def normalize(self, pose: Pose) -> Pose:
        """Returns a new pose in canonical space (or empty when no body)."""
        center = self._center_and_scale(pose)
        if center is None:
            return Pose()
        cx, cy, scale = center

        # Rotate in image space (y down) to align hips
        # with the horizontal; then flip Y into canonical space.
        theta = self._rotation_angle(pose)
        cos_t, sin_t = math.cos(-theta), math.sin(-theta)
        inv = 1.0 / scale

        out = Pose()
        for name, joint in pose.joints.items():
            dx = (joint.x - cx) * inv
            dy = (joint.y - cy) * inv
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            z = (joint.z * inv) if self.z_scale else joint.z
            out.joints[name] = Joint(
                x=float(rx), y=float(-ry), z=float(z), visibility=joint.visibility
            )
        return out
