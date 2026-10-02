"""Game controller: scoring, combo and state (pure, testable logic)."""

from __future__ import annotations

from typing import Optional

from engine.pose_engine.skeleton import Pose
from engine.score_engine import PoseComparator, Scorer


class GameController:
    """Accumulates score and combo from pose comparison."""

    RATINGS = ["Perfect", "Great", "Good", "Ok", "Miss"]

    def __init__(self, comparator: PoseComparator, scorer: Scorer) -> None:
        self.comparator = comparator
        self.scorer = scorer
        self.reset()

    def reset(self) -> None:
        self.combo = 0
        self.max_combo = 0
        self.total_points = 0
        self.frames = 0
        self.accuracy_sum = 0.0
        self.rating_counts = {rating: 0 for rating in self.RATINGS}
        self.comparator.reset()

    def evaluate(self, player: Optional[Pose], reference: Optional[Pose]) -> dict:
        """Compare the player pose against the reference and update state.

        If the reference has no pose (VideoTest with no detection -> empty joints /
        Pose.is_empty()), the frame is neutral: it adds/subtracts no points and leaves combo untouched.
        """
        # Neutral frame: reference with no detection -> neither reward nor penalize
        if reference is None or reference.is_empty():
            return {
                "score": self.total_points,
                "frame_score": 0,
                "rating": "Neutral",
                "combo": self.combo,
                "max_combo": self.max_combo,
                "error": None,
                "tracking": False,
                "neutral": True,
            }
        result = self.comparator.compare(player, reference) if player else {
            "error": None,
            "tracking": False,
        }
        error = result["error"]
        rating = self.scorer.rate(error)
        points = self.scorer.points(rating)
        frame_score = self.scorer.score(error)

        if rating == "Miss":
            self.combo = 0
        else:
            self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)

        self.total_points += points
        self.frames += 1
        self.accuracy_sum += frame_score
        self.rating_counts[rating] += 1

        return {
            "score": self.total_points,
            "frame_score": int(round(frame_score)),
            "rating": rating,
            "combo": self.combo,
            "max_combo": self.max_combo,
            "error": error,
            "tracking": result.get("tracking", False),
        }

    def summary(self) -> dict:
        """Final summary for the results screen."""
        accuracy = (self.accuracy_sum / self.frames) if self.frames else 0.0
        return {
            "score": self.total_points,
            "max_combo": self.max_combo,
            "frames": self.frames,
            "accuracy": accuracy,
            "rating_counts": dict(self.rating_counts),
        }
