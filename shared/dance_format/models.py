"""Song format models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.pose_engine.skeleton import SEMANTIC_JOINTS

FORMAT_VERSION = "openjustdance-1"
FORMAT_VERSION_MULTI = "openjustdance-2"


@dataclass
class DancerPose:
    """Pose of a single dancer inside a multi frame."""

    dancer_id: int = 0
    joints: Dict[str, List[float]] = field(default_factory=dict)
    raw: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"dancer_id": self.dancer_id, "joints": self.joints, "raw": self.raw}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DancerPose":
        return cls(
            dancer_id=int(data.get("dancer_id", 0)),
            joints=data.get("joints", {}),
            raw=data.get("raw", {}),
        )


@dataclass
class FramePose:
    """A reference pose at a point in time.

    Supports two modes:
    * legacy (openjustdance-1): direct joints/raw (single dancer)
    * multi  (openjustdance-2): dancers list with 0..N poses (multi-dancer)
    """

    time: float
    joints: Dict[str, List[float]] = field(default_factory=dict)
    raw: Dict[str, List[float]] = field(default_factory=dict)
    # new multi field; when set, it takes priority over joints/raw
    dancers: List[DancerPose] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"time": self.time, "joints": self.joints, "raw": self.raw}
        if self.dancers:
            data["dancers"] = [d.to_dict() for d in self.dancers]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FramePose":
        dancers_data = data.get("dancers")
        dancers: List[DancerPose] = []
        if isinstance(dancers_data, list) and dancers_data:
            for d in dancers_data:
                try:
                    dancers.append(DancerPose.from_dict(d))
                except Exception:
                    continue
        return cls(
            time=float(data["time"]),
            joints=data.get("joints", {}),
            raw=data.get("raw", {}),
            dancers=dancers,
        )

    def get_dancer(self, dancer_id: int) -> Optional[DancerPose]:
        """Return the pose for dancer_id or None. Legacy fallback when dancers is empty."""
        if self.dancers:
            for d in self.dancers:
                if d.dancer_id == dancer_id:
                    return d
            return None
        # legacy: dancer 0 only
        if dancer_id == 0:
            return DancerPose(dancer_id=0, joints=self.joints, raw=self.raw)
        return None

    def num_dancers(self) -> int:
        if self.dancers:
            return len(self.dancers)
        # legacy: 1 if there are joints, 0 if neutral empty
        return 1 if self.joints else 0

    def is_empty_for(self, dancer_id: int) -> bool:
        d = self.get_dancer(dancer_id)
        if d is None:
            return True
        return not d.joints


@dataclass
class Song:
    """A playable song."""

    title: str
    artist: str
    fps: float
    duration: float
    format: str = FORMAT_VERSION
    joints_order: List[str] = field(default_factory=lambda: list(SEMANTIC_JOINTS))
    poses: List[FramePose] = field(default_factory=list)
    path: Optional[Path] = None
    # new: number of reference dancers (1 = legacy)
    num_dancers: int = 1
    # new: W/H aspect of the reference video (0.5625 = vertical 9:16,
    # 1.78 = horizontal 16:9). None = unknown (old songs: inferred
    # from song.mp4). Required to compare vertical vs horizontal
    # camera (the normalizer bakes the aspect into y).
    reference_aspect: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "title": self.title,
            "artist": self.artist,
            "fps": self.fps,
            "duration": self.duration,
            "format": self.format,
            "joints_order": self.joints_order,
            "poses": [pose.to_dict() for pose in self.poses],
        }
        if self.num_dancers != 1:
            data["num_dancers"] = self.num_dancers
        if self.reference_aspect is not None:
            try:
                data["reference_aspect"] = float(self.reference_aspect)
            except (TypeError, ValueError):
                pass
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Song":
        poses = [FramePose.from_dict(p) for p in data.get("poses", [])]
        # infer num_dancers when missing from JSON
        nd = data.get("num_dancers")
        if nd is None:
            # max dancers per frame
            nd = 1
            for p in poses:
                if p.dancers:
                    nd = max(nd, len(p.dancers))
                elif p.joints:
                    nd = max(nd, 1)
            # when the format is explicitly multi
            if data.get("format") == FORMAT_VERSION_MULTI:
                nd = max(nd, 2)
        try:
            ra = data.get("reference_aspect")
            ra = float(ra) if ra is not None else None
            if ra is not None and not (0.2 <= ra <= 5.0):
                ra = None
        except (TypeError, ValueError):
            ra = None
        return cls(
            title=data.get("title", "Sin titulo"),
            artist=data.get("artist", "Desconocido"),
            fps=float(data.get("fps", 30.0)),
            duration=float(data.get("duration", 0.0)),
            format=data.get("format", FORMAT_VERSION),
            joints_order=data.get("joints_order", list(SEMANTIC_JOINTS)),
            poses=poses,
            num_dancers=int(nd) if int(nd) > 0 else 1,
            reference_aspect=ra,
        )

    def max_dancers(self) -> int:
        return self.num_dancers
