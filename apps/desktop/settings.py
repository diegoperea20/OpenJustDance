"""Persistent app configuration."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from engine.score_engine import DEFAULT_WEIGHTS


def default_songs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "songs"


# Accuracy thresholds per difficulty level. Higher value = more forgiving.
DIFFICULTY_PRESETS: dict[str, dict[str, float]] = {
    "easy": {
        "perfect_threshold": 0.20,
        "great_threshold": 0.32,
        "good_threshold": 0.48,
        "ok_threshold": 0.62,
        "angle_weight": 0.0,
    },
    "medium": {
        "perfect_threshold": 0.15,
        "great_threshold": 0.28,
        "good_threshold": 0.42,
        "ok_threshold": 0.55,
        "angle_weight": 0.15,
    },
    "hard": {
        "perfect_threshold": 0.06,
        "great_threshold": 0.12,
        "good_threshold": 0.22,
        "ok_threshold": 0.35,
        "angle_weight": 0.30,
    },
}

DIFFICULTY_LABELS: dict[str, str] = {
    "easy": "Fácil",
    "medium": "Medio",
    "hard": "Difícil",
}


@dataclass
class Settings:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    mirror: bool = True
    backend: str = "auto"
    # Live YOLO inference device: "auto" (CUDA if a GPU is present, else
    # CPU), "cuda" or "cpu". MediaPipe is always CPU on Windows (pip).
    device: str = "auto"
    language: str = "en"
    perfect_threshold: float = 0.05
    great_threshold: float = 0.10
    good_threshold: float = 0.20
    ok_threshold: float = 0.30
    angle_weight: float = 0.30
    smooth_alpha: float = 0.5
    difficulty: str = "medium"
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    songs_dir: str = ""
    # Developer game mode view: when True the GameView keeps the legacy
    # side-by-side camera layout; when False (default) the player camera
    # is a small overlay at the bottom-left over a large reference video.
    developer_mode: bool = False

    def thresholds(self) -> dict[str, float]:
        return {
            "Perfect": self.perfect_threshold,
            "Great": self.great_threshold,
            "Good": self.good_threshold,
            "Ok": self.ok_threshold,
        }

    def apply_difficulty(self, difficulty: str) -> None:
        """Apply the difficulty preset values to the thresholds."""
        preset = DIFFICULTY_PRESETS.get(difficulty)
        if preset is None:
            return
        self.difficulty = difficulty
        self.perfect_threshold = preset["perfect_threshold"]
        self.great_threshold = preset["great_threshold"]
        self.good_threshold = preset["good_threshold"]
        self.ok_threshold = preset["ok_threshold"]
        self.angle_weight = preset["angle_weight"]

    def resolve_songs_dir(self) -> Path:
        if self.songs_dir:
            return Path(self.songs_dir)
        env = os.environ.get("OPENJUSTDANCE_SONGS")
        if env:
            return Path(env)
        return default_songs_dir()

    @staticmethod
    def _config_path() -> Path:
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "openjustdance" / "settings.json"

    def save(self) -> None:
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        path = cls._config_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in known}
            # Language: only en/es, English by default.
            lang = str(filtered.get("language", "en") or "en").strip().lower()
            filtered["language"] = lang if lang in ("en", "es") else "en"
            # Device: only auto/cuda/cpu, auto by default.
            dev = str(filtered.get("device", "auto") or "auto").strip().lower()
            filtered["device"] = dev if dev in ("auto", "cuda", "cpu") else "auto"
            # Developer mode: strict bool, default False (PiP layout).
            try:
                filtered["developer_mode"] = bool(filtered.get("developer_mode", False))
            except Exception:
                filtered["developer_mode"] = False
            return cls(**filtered)
        except Exception:
            return cls()
