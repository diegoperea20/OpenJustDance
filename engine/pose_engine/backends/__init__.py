"""Pose detection backends.

Each backend implements the same `PoseBackend` interface. The
`mediapipe-cpu` backend is the only one enabled by default (pip MediaPipe has no
GPU support on Windows). The architecture allows adding a GPU backend (e.g.
ONNX Runtime with CUDA/DirectML) without touching the rest of the code.
"""

from engine.pose_engine.backends.base import PoseBackend
from engine.pose_engine.backends.mediapipe import MediaPipeBackend
from engine.pose_engine.backends.multi_mediapipe import MultiMediapipeBackend
try:
    from engine.pose_engine.backends.yolo_pose import YoloPoseBackend
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False
    YoloPoseBackend = None  # type: ignore
try:
    from engine.pose_engine.backends.yolo_mediapipe import YoloMediapipeBackend
    _HAS_YOLO_MP = bool(_HAS_YOLO)
except Exception:
    _HAS_YOLO_MP = False
    YoloMediapipeBackend = None  # type: ignore

BACKENDS: dict[str, type[PoseBackend]] = {
    "mediapipe-cpu": MediaPipeBackend,
    "multi-mediapipe": MultiMediapipeBackend,
}
if _HAS_YOLO and YoloPoseBackend is not None:
    BACKENDS["yolo-pose"] = YoloPoseBackend
    BACKENDS["yolo26n-pose"] = YoloPoseBackend
    BACKENDS["yolov8n-pose"] = YoloPoseBackend
    BACKENDS["yolo26n"] = YoloPoseBackend
if _HAS_YOLO_MP and YoloMediapipeBackend is not None:
    BACKENDS["yolo-mediapipe"] = YoloMediapipeBackend


def create_backend(name: str = "auto", **kwargs) -> PoseBackend:
    """Create a backend by name. `auto` selects the best available."""
    if name in (None, "", "auto"):
        name = "mediapipe-cpu"
    if name not in BACKENDS:
        raise ValueError(
            f"Backend desconocido: {name!r}. Disponibles: {sorted(BACKENDS)}"
        )
    return BACKENDS[name](**kwargs)


__all__ = ["PoseBackend", "MediaPipeBackend", "BACKENDS", "create_backend"]
if _HAS_YOLO_MP:
    __all__ += ["YoloMediapipeBackend"]
