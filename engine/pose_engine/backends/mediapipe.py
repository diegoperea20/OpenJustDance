from __future__ import annotations

from typing import Optional

import numpy as np

from engine.pose_engine.backends.base import PoseBackend
from engine.pose_engine.skeleton import MP_TO_SEMANTIC, Joint, Pose


class MediaPipeBackend(PoseBackend):
    """Pose detection with MediaPipe Pose (classic solutions API).

    Runs on CPU. It is the default backend and works on all
    supported platforms (Windows, Linux, macOS).
    """

    name = "mediapipe-cpu"
    num_keypoints = 33

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Pose:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            results = self._pose.process(rgb)
        finally:
            rgb.flags.writeable = True

        if not results.pose_landmarks:
            return Pose()

        pose = Pose()
        landmarks = results.pose_landmarks.landmark
        for idx, name in MP_TO_SEMANTIC.items():
            lm = landmarks[idx]
            pose.joints[name] = Joint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
        return pose

    def close(self) -> None:
        self._pose.close()


__all__ = ["MediaPipeBackend"]
