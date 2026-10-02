from abc import ABC, abstractmethod

import numpy as np

from engine.pose_engine.skeleton import Pose


class PoseBackend(ABC):
    """Common interface for detecting a pose from a BGR frame."""

    name: str = "base"
    # Number of keypoints returned by the backend (33 MediaPipe, 17 COCO...).
    num_keypoints: int = 0

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Pose:
        """Return a Pose (empty if no body is detected).

        x/y coordinates are normalized 0..1 to the image size
        (y downwards), z is the backend's raw relative depth.
        """

    def detect_multi(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], Pose]]:
        """Optional: detect multiple poses. Default uses the single detect.

        Return a list of (bbox x,y,w,h in frame coords, Pose). Full-frame bbox if only one.
        """
        pose = self.detect(frame)
        if pose.is_empty():
            return []
        h, w = frame.shape[:2]
        return [((0, 0, w, h), pose)]

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
