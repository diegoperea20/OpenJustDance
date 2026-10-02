# Contributing

## Workflow

1. `uv sync` 
2. Create a branch, make focused changes, run `uv run pytest`.

3. Open a PR describing behavior change + tests. No secret commits.

Before committing, inspect `git status`, `git diff`, `git log --oneline -10`; stage only intended files. Never commit weights, videos, or songs (see below).

## Hard conventions (from AGENTS.md)

- **Engine has no Qt**: `engine/`, `shared/`, `tools/` import only numpy/scipy/opencv/mediapipe (+ `torch`/`ultralytics`/`boxmot`/`insightface` in YOLO/FaceID paths). Qt lives exclusively in `apps/`.
- **New package?** Regenerate the editable install so `uv run` sees it: `uv sync --reinstall-package openjustdance`. Console scripts live in `[project.scripts]` (`openjustdance`, `pose-extractor`, `engine-server`).
- **Backends**: implement `PoseBackend` (`engine/pose_engine/backends/base.py`), register in `BACKENDS`. Registration depends only on the import succeeding; weight existence is checked inside `yolo_pose.py`. `create_backend("auto")` = `mediapipe-cpu`, not YOLO.
- **Multi pipeline is unified**: YOLO-Pose first, HOG fallback only when YOLO sees nobody (`engine/pose_engine/multi.py`). Do not reintroduce divergent single-sided gates like `multi_hits >= 3`. `StableSlotMapper` = Hungarian by distance; do not re-sort by x every frame.
- **Compare by time, not frames**: use `SongSynchronizer.frame_at()` / `dancer_pose_at()` per `dancer_id`. Empty `joints={}` = neutral.
- **FaceID is mandatory in UI** (`proc_entry.py` always `face_id=True`), opt-in in CLI (`--face-id`). Keep thresholds centralized (`0.38 / 6 / 320`).
- **Palette**: use `DANCER_PALETTE_BGR`, labels `D{n}|ID{n}` (no `·` glyph in OpenCV Hershey).

## Dependencies to know

- `numpy>=1.24.0` (no upper bound), `mediapipe==0.10.14` (classic `mp.solutions.pose` API), `torch>=2.0` + `torchvision` (CUDA index `pytorch-cu128`), `ultralytics>=8.3`, `boxmot`, `insightface>=2.0`, `onnxruntime-gpu==1.22.0`.
- Known doc quirks (as-is, do not "fix" unasked): stale comment suggesting manual `uv pip install insightface onnxruntime-gpu` although they are in main deps; stray top-level `dev = [pytest, black, ruff]` block that is not a valid dependency-group (only `pytest` is installed via `uv sync --group dev`); `uv.lock` and `*.onnx` are not ignored.
- GPU auto-detection lives in `engine/pose_engine/hardware.py` and applies to YOLO (`--device auto|cpu|cuda`, Settings `device`), not MediaPipe. Do not try `delegate=GPU` for MediaPipe on Windows.

## What not to commit

`.gitignore` covers `.venv/`, `__pycache__/`, `.pytest_cache/`, `.coverage`, `.vscode/`, `.idea/`, `.cache/`, `*.log`, `/songs`, `*.mp4`, `*.pt`. Local weights (`yolo11m-pose.pt`, `yolo11n-pose.pt`, `yolo26n-pose.pt`, `yolo26n-reid.onnx`) and test videos (`aespanova.mp4`, `aespavideo.mp4`, `single.mp4`, `videotest*.mp4`) stay untracked. Note `*.onnx` and `uv.lock` are currently tracked/untracked as-is.

## Tests

`tests/` uses pytest (18 files). Controller tests (`apps/desktop/controller.py`) are pure logic and do not need PySide6. YOLO/weight/video tests guard with skip when files are missing. See [DEVELOPMENT.md](DEVELOPMENT.md).
