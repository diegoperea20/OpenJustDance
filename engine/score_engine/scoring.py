"""Score and rating generation from comparison error.

Default thresholds (plan): Perfect < 5%, Great < 10%, Good < 20%,
Ok < 30%, Miss otherwise. Configurable.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_THRESHOLDS: dict[str, float] = {
    "Perfect": 0.05,
    "Great": 0.10,
    "Good": 0.20,
    "Ok": 0.30,
}

RATING_POINTS: dict[str, int] = {
    "Perfect": 10,
    "Great": 8,
    "Good": 6,
    "Ok": 4,
    "Miss": 0,
}


class Scorer:
    """Maps a normalized error to a per-frame rating and points."""

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        max_score: float = 100.0,
    ) -> None:
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.max_score = max_score

    def rate(self, error: Optional[float]) -> str:
        if error is None:
            return "Miss"
        ordered = sorted(self.thresholds.items(), key=lambda kv: kv[1])
        for rating, threshold in ordered:
            if error <= threshold:
                return rating
        return "Miss"

    def score(self, error: Optional[float]) -> float:
        """Per-frame score in [0, max_score]. None (no tracking) = 0."""
        if error is None:
            return 0.0
        return max(0.0, self.max_score * (1.0 - min(error, 1.0)))

    def points(self, rating: str) -> int:
        return RATING_POINTS.get(rating, 0)
