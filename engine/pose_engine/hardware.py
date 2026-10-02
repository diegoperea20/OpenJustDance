"""Hardware (GPU) detection for UI info and logs.

Note: MediaPipe (pip) has no GPU support on Windows, so that backend
runs on CPU. YOLO (ultralytics/torch) does use CUDA when available:
see `pick_device` in backends/yolo_pose.py. This module detects and reports
the available GPU and the real inference device for the UI.
"""

from __future__ import annotations

import platform
import shutil
import subprocess


def detect_gpus() -> list[str]:
    """Return the names of detected NVIDIA GPUs via nvidia-smi."""
    if platform.system() != "Windows" and shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def opencv_cuda_available() -> bool:
    """Does the installed OpenCV build support CUDA?"""
    try:
        import cv2

        return bool(cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:
        return False


def torch_cuda_available() -> bool:
    """Does torch see a usable CUDA GPU (the one YOLO uses)?"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_cuda_name() -> str | None:
    """Name of the GPU seen by torch, or None if there is no CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def gpu_info(inference: str | None = None) -> dict:
    """Summary of detected hardware (for the status bar / settings).

    `inference`: label of the real live-inference device
    (e.g. "yolo11n-pose · cuda"); if None, the detected
    capability is reported (cuda if torch sees it, else CPU).
    """
    return {
        "gpus": detect_gpus(),
        "opencv_cuda": opencv_cuda_available(),
        "torch_cuda": torch_cuda_available(),
        "torch_cuda_name": torch_cuda_name(),
        "inference": inference
        if inference is not None
        else ("cuda (auto)" if torch_cuda_available() else "CPU"),
    }


def gpu_summary(inference: str | None = None) -> str:
    """Short description: e.g. 'GPU: NVIDIA GeForce RTX 5050 | inferencia: yolo11n-pose · cuda'."""
    gpus = detect_gpus()
    base = f"GPU: {', '.join(gpus)}" if gpus else "GPU: no detectada"
    return f"{base} | inferencia: {gpu_info(inference).get('inference')}"
