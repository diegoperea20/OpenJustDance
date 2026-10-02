"""Normalized pose comparison.

Compares two bodies in canonical space combining xy position error
and joint-angle error. Absolute coordinates are never compared.

Default weights (plan): head 5%, shoulders 15%, elbows 15%, wrists 25%,
hips 10%, knees 15%, ankles 15%. Configurable.
"""

from __future__ import annotations

import math
from typing import Optional

from engine.pose_engine.skeleton import Joint, Pose

DEFAULT_WEIGHTS: dict[str, float] = {
    "head": 0.05,
    "left_shoulder": 0.075,
    "right_shoulder": 0.075,
    "left_elbow": 0.075,
    "right_elbow": 0.075,
    "left_wrist": 0.125,
    "right_wrist": 0.125,
    "left_hip": 0.05,
    "right_hip": 0.05,
    "left_knee": 0.075,
    "right_knee": 0.075,
    "left_ankle": 0.075,
    "right_ankle": 0.075,
}

# Triplets for articulated joint-angle error: (side, middle joint, before, after)
ANGLE_TRIPLETS = [
    ("left", "left_shoulder", "left_hip", "left_elbow"),
    ("left", "left_elbow", "left_shoulder", "left_wrist"),
    ("left", "left_hip", "left_shoulder", "left_knee"),
    ("left", "left_knee", "left_hip", "left_ankle"),
    ("right", "right_shoulder", "right_hip", "right_elbow"),
    ("right", "right_elbow", "right_shoulder", "right_wrist"),
    ("right", "right_hip", "right_shoulder", "right_knee"),
    ("right", "right_knee", "right_hip", "right_ankle"),
]


def _angle_2d(a: Joint, b: Joint, c: Joint) -> float:
    """Angle (degrees) at point b formed by vectors b->a and b->c."""
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cos))


class PoseComparator:
    """Compares a player pose against a normalized reference pose."""

    # Visibility below which a joint is garbage (YOLO emits 0,0,0 with
    # conf 0 under occlusions): always ignored, no ramp.
    VIS_FLOOR = 0.10
    # Minimum coverage (ramp sum ~= 1 solid joint) to report tracking.
    MIN_COVERAGE = 1.0

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        visibility_threshold: float = 0.4,
        angle_weight: float = 0.3,
        smooth_factor: float = 0.5,
        align_max_deg: float = 30.0,
    ) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.visibility_threshold = visibility_threshold
        self.angle_weight = max(0.0, min(1.0, angle_weight))
        self.position_weight = 1.0 - self.angle_weight
        self.smooth_factor = smooth_factor
        self.align_max_deg = max(0.0, float(align_max_deg))
        self._smoothed_error: Optional[float] = None

    def reset(self) -> None:
        self._smoothed_error = None

    def _vis_factor(self, visibility: float) -> float:
        """Smooth 0..1 weight by visibility (ramp from floor to threshold).

        Replaces the hard cutoff: joints with medium conf (typical in
        horizontal references with a small dancer) contribute
        proportionally instead of being dropped. With vis >= threshold it is 1.0
        (identical to the previous behavior).
        """
        try:
            v = float(visibility)
        except (TypeError, ValueError):
            return 0.0
        t = self.visibility_threshold
        if v >= t:
            return 1.0
        if t <= self.VIS_FLOOR or v < self.VIS_FLOOR:
            return 0.0
        return (v - self.VIS_FLOOR) / (t - self.VIS_FLOOR)

    @staticmethod
    def _best_rotation(player_xy, ref_xy, weights) -> float:
        """Angle (rad) aligning player->reference (weighted 2D Kabsch).

        Absorbs the de-rotation residue that adds noise on BOTH sides
        (hip theta with few pixels + baked-in aspect). It only
        fixes global rotation+translation: a different pose still
        yields high error after alignment.
        """
        sw = 0.0
        pCx = pCy = rCx = rCy = 0.0
        for w, (px, py), (rx, ry) in zip(weights, player_xy, ref_xy):
            sw += w
            pCx += w * px
            pCy += w * py
            rCx += w * rx
            rCy += w * ry
        if sw <= 1e-12:
            return 0.0
        pCx /= sw
        pCy /= sw
        rCx /= sw
        rCy /= sw
        dot = cross = 0.0
        for w, (px, py), (rx, ry) in zip(weights, player_xy, ref_xy):
            dxp, dyp = px - pCx, py - pCy
            dxr, dyr = rx - rCx, ry - rCy
            dot += w * (dxp * dxr + dyp * dyr)
            cross += w * (dxp * dyr - dyp * dxr)
        if abs(dot) < 1e-12 and abs(cross) < 1e-12:
            return 0.0
        return math.atan2(cross, dot)

    def _position_error(self, player: Pose, reference: Pose) -> tuple[Optional[float], dict, float]:
        names: list[str] = []
        ws: list[float] = []
        p_xy: list[tuple[float, float]] = []
        r_xy: list[tuple[float, float]] = []
        coverage = 0.0
        for name, weight in self.weights.items():
            p = player.get(name)
            r = reference.get(name)
            if p is None or r is None:
                continue
            factor = min(self._vis_factor(p.visibility), self._vis_factor(r.visibility))
            if factor <= 0.0:
                continue
            coverage += factor
            names.append(name)
            ws.append(weight * factor)
            p_xy.append((p.x, p.y))
            r_xy.append((r.x, r.y))
        if not names or coverage < self.MIN_COVERAGE:
            return None, {}, 0.0
        # Rigid alignment (bounded translation + rotation) before measuring:
        # center centroids and fix residual spin. Without it, noise
        # from theta in noisy references lands fully in the error.
        total_w = math.fsum(ws)
        if total_w <= 1e-12:
            return None, {}, 0.0
        pCx = math.fsum(w * x for w, (x, _y) in zip(ws, p_xy)) / total_w
        pCy = math.fsum(w * y for w, (_x, y) in zip(ws, p_xy)) / total_w
        rCx = math.fsum(w * x for w, (x, _y) in zip(ws, r_xy)) / total_w
        rCy = math.fsum(w * y for w, (_x, y) in zip(ws, r_xy)) / total_w
        angle = self._best_rotation(p_xy, r_xy, ws) if len(names) >= 2 else 0.0
        if abs(angle) > math.radians(self.align_max_deg):
            # Large spin = genuinely different orientation: do not forgive.
            angle = 0.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        acc = 0.0
        joint_errors: dict[str, float] = {}
        for name, w, (px, py), (rx, ry) in zip(names, ws, p_xy, r_xy):
            # Player rotated about its centroid vs reference shifted
            # to the same centroid (global rotation + translation out).
            qx = (px - pCx) * cos_a - (py - pCy) * sin_a
            qy = (px - pCx) * sin_a + (py - pCy) * cos_a
            err = math.hypot(qx - (rx - rCx), qy - (ry - rCy))
            acc += w * err
            joint_errors[name] = err
        return acc / total_w, joint_errors, math.degrees(angle)

    def _angle_error(self, player: Pose, reference: Pose) -> Optional[float]:
        total_weight = 0.0
        acc = 0.0
        for _side, mid, before, after in ANGLE_TRIPLETS:
            p_mid = player.get(mid)
            r_mid = reference.get(mid)
            p_before = player.get(before)
            r_before = reference.get(before)
            p_after = player.get(after)
            r_after = reference.get(after)
            if (
                p_mid is None
                or r_mid is None
                or p_before is None
                or r_before is None
                or p_after is None
                or r_after is None
            ):
                continue
            # Smooth weight by visibility (triplet minimum): with
            # medium-conf joints the triplet contributes proportionally
            # instead of being dropped entirely.
            factor = min(
                self._vis_factor(p_mid.visibility),
                self._vis_factor(r_mid.visibility),
                self._vis_factor(p_before.visibility),
                self._vis_factor(r_before.visibility),
                self._vis_factor(p_after.visibility),
                self._vis_factor(r_after.visibility),
            )
            if factor <= 0.0:
                continue
            a_p = _angle_2d(p_before, p_mid, p_after)
            a_r = _angle_2d(r_before, r_mid, r_after)
            diff = abs(a_p - a_r) / 180.0
            weight = self.weights.get(mid, 0.0) * factor
            acc += weight * diff
            total_weight += weight
        if total_weight <= 1e-9:
            return None
        return acc / total_weight

    def compare(self, player: Pose, reference: Pose) -> dict:
        """Returns {error, position_error, angle_error, joint_errors, tracking, align_angle_deg}."""
        pos_err, joint_errors, align_deg = self._position_error(player, reference)
        if pos_err is None:
            self._smoothed_error = None
            return {
                "error": None,
                "position_error": None,
                "angle_error": None,
                "joint_errors": {},
                "tracking": False,
                "align_angle_deg": 0.0,
            }

        angle_err = self._angle_error(player, reference)
        error = (
            self.position_weight * pos_err + self.angle_weight * angle_err
            if angle_err is not None
            else pos_err
        )

        if self.smooth_factor > 0.0:
            prev = self._smoothed_error
            if prev is None:
                error = error
            else:
                error = self.smooth_factor * error + (1.0 - self.smooth_factor) * prev
        self._smoothed_error = error

        return {
            "error": error,
            "position_error": pos_err,
            "angle_error": angle_err,
            "joint_errors": joint_errors,
            "tracking": True,
            "align_angle_deg": align_deg,
        }


# ---------------------------------------------------------------------------
# Aspect-invariant comparison space (vertical 9:16 vs 16:9 camera).
#
# BodyNormalizer centers on hips and scales by shoulder width measured in
# 0..1 coordinates. On x the aspect cancels out, but on y a W/H factor
# stays baked in (and hip de-rotation is also computed distorted):
# the same physical pose yields error ~0.65 (= permanent Miss) when comparing
# a vertical reference against a horizontal camera.
#
# Exact fix: pre-scale x of the RAW pose (0..1) by k = live/ref BEFORE
# normalizing, on BOTH sides. Center, scale and de-rotation are then
# computed in the same space and comparison is exact (rotation included)
# with smoothing intact. With equal aspect k = 1 (identity: 100%
# backwards compatible, bit for bit).
# ---------------------------------------------------------------------------

def compare_space_prescale(ref_aspect: float | None, live_aspect: float | None) -> float:
    """Factor k to pre-scale x before normalizing (k = live/ref).

    Returns 1.0 when either aspect is unknown or invalid (legacy).
    Clamped to [0.2, 5.0]: outside that the probe is garbage and scaling
    would do more harm than no correction.
    """
    try:
        ra, la = float(ref_aspect), float(live_aspect)
    except (TypeError, ValueError):
        return 1.0
    if not (ra > 0 and la > 0):
        return 1.0
    k = la / ra
    if not (0.2 <= k <= 5.0):
        return 1.0
    return k


def to_compare_space(player_raw, factor: float):
    """Copy of the raw pose with pre-scaled x (passthrough when factor == 1).

    `player_raw` is a Pose in frame 0..1 coords; Pose is imported here to
    avoid cycles (this module already imports it above).
    """
    if player_raw is None or player_raw.is_empty():
        return player_raw
    try:
        f = float(factor)
    except (TypeError, ValueError):
        return player_raw
    if f == 1.0:
        return player_raw
    out = Pose()
    for name, joint in player_raw.joints.items():
        out.joints[name] = Joint(x=joint.x * f, y=joint.y, z=joint.z,
                                 visibility=joint.visibility)
    return out
