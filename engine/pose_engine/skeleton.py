from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Indices of the 33 MediaPipe Pose landmarks (in official order).
MP_LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# Semantic joints used by the song format and scoring.
SEMANTIC_JOINTS = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# MediaPipe index -> semantic name mapping.
MP_TO_SEMANTIC = {
    0: "head",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

# Connections for drawing the skeleton (pairs of semantic names).
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# SINGLE palette per dancer_id (BGR for cv2). Used by the extractor preview
# (preview_pose.mp4), the DancerSelect cards and the comparison view, so
# D2 is always the same color across the whole pipeline.
DANCER_PALETTE_BGR = (
    (0, 255, 0),    # D1 green
    (255, 0, 0),    # D2 blue
    (0, 0, 255),    # D3 red
    (255, 255, 0),  # D4 cyan
)


def dancer_color_bgr(dancer_id: int):
    """BGR palette color for a dancer_id (robust to odd ids)."""
    try:
        idx = int(dancer_id) % len(DANCER_PALETTE_BGR)
    except Exception:
        idx = 0
    return DANCER_PALETTE_BGR[idx]


@dataclass
class Joint:
    """A keypoint with (x, y, z) coordinates and visibility."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 1.0

    def to_array(self) -> list[float]:
        return [self.x, self.y, self.z, self.visibility]

    @classmethod
    def from_array(cls, arr) -> "Joint":
        x, y, z, vis = (float(v) for v in arr)
        return cls(x=x, y=y, z=z, visibility=vis)


@dataclass
class Pose:
    """A human body as a semantic-name -> Joint mapping."""

    joints: dict[str, Joint] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Joint]:
        return self.joints.get(name)

    def visible(self, name: str, threshold: float = 0.5) -> bool:
        joint = self.joints.get(name)
        return joint is not None and joint.visibility >= threshold

    def is_empty(self) -> bool:
        return not self.joints

    def to_dict(self) -> dict[str, list[float]]:
        return {name: joint.to_array() for name, joint in self.joints.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "Pose":
        return cls(joints={name: Joint.from_array(arr) for name, arr in data.items()})


def draw_pose(frame, pose: Pose, color=(0, 255, 128), thickness: int = 3) -> None:
    """Draw the skeleton on the frame (normalized 0..1 image coordinates).

    Assumes joint.x / joint.y are normalized to the image size
    (as returned by MediaPipe) and that y points downward.
    """
    import cv2

    h, w = frame.shape[:2]

    def pt(name):
        joint = pose.get(name)
        if joint is None or joint.visibility < 0.3:
            return None
        return (int(joint.x * w), int(joint.y * h))

    for a, b in POSE_CONNECTIONS:
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, color, thickness)

    for name, joint in pose.joints.items():
        p = pt(name)
        if p is not None:
            radius = max(2, thickness - 1)
            cv2.circle(frame, p, radius, (255, 255, 255), -1)
            cv2.circle(frame, p, radius, color, 1)
