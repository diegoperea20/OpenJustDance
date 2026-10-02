# Privacy & Performance

## Privacy: local-only by design

- Camera frames, videos, songs, and FaceID embeddings never leave your machine. No upload, no cloud inference, no telemetry.
- Face gallery (`FaceGallery`, InsightFace `buffalo_l`) runs locally (CUDA provider if `cuda_ep_usable()`, else CPU) and stores cosine embeddings only in memory for the import/play session.
- Songs live as plain folders on disk (`songs/`, `cover.png`, `song.mp4/mp3/json`). Delete a song = delete its folder (`SongSelect → Delete`).
- Settings live at `%APPDATA%/openjustdance/settings.json`. Override the library path with `$OPENJUSTDANCE_SONGS`.

## Performance

### Which model runs where

| Stage | Model | Size | Why |
|---|---|---|---|
| Ingest (offline) | `yolo11m-pose.pt` | `imgsz 960` | Accurate multi-dancer extraction |
| Live webcam | `yolo11n-pose.pt` | 640 px working width | Lightweight real-time |
| Alt backend | `yolo26n-pose.pt` | alias `yolo26n-pose` | Optional |
| ReID appearance | `yolo26n-reid.onnx` | via `botsort-reid*.yaml` | Anti-overlap identity (usually CPU fallback) |
| Face | `buffalo_l` (InsightFace) | `det-size 320` | Identity gallery, re-check every 6 frames |

### GPU vs CPU (Windows)

- **MediaPipe** (`==0.10.14`, pip): no GPU on Windows — always CPU. Do not attempt `delegate=GPU`.
- **YOLO** (`ultralytics`/`torch`, `pytorch-cu128` index): CUDA when `torch.cuda.is_available()`. `pick_device("auto")` → `cuda` else `cpu`; FP16 only on `cuda*`. Force with `--device cpu|cuda` (CLI) or Settings `device`.
- **ONNX** (ReID/FaceID): mini-session probe `cuda_ep_usable()`; falls back to CPU silently (ORT logger set to FATAL).
- Inspect with `engine/pose_engine/hardware.py`: `detect_gpus()` (`nvidia-smi`), `torch_cuda_available()/torch_cuda_name()`, `opencv_cuda_available()`, `gpu_info()/gpu_summary()` (shown in Settings → Hardware).

### Trackers

| Choice | Speed | When to use |
|---|---|---|
| `botsort-reid` (default) | balanced | General multi-dancer, overlap-resistant |
| `botsort-reid-gmc` | slow | Only when the reference camera itself moves (GMC) |
| `bytetrack-dance` / `bytetrack` | fast | Crowded or low-end GPU/CPU, no ReID |
| `botsort` (stock yaml) | medium | Fallback without repo tuning |

Plain `botsort`/`bytetrack` have no repo yaml and resolve to Ultralytics stock configs.

### Tuning knobs

- `--yolo-conf 0.4` (CLI) / `0.25` (UI import): lower (`0.1–0.3`) keeps occluded dancers but adds false positives; higher (`0.5–0.6`) is cleaner for solos.
- `--yolo-iou 0.6`: keeps overlapping dancers separate (do not lower for duets/crews).
- `--imgsz 960` ingest vs 640 live: raise only if small dancers are missed; costs VRAM/time linearly.
- `--face-check-every 6` / `--face-thresh 0.38`: more frequent checks = stabler IDs, slower ingest.
- `--smooth-alpha 0.6` ingest / `0.5` live smoothing: higher = smoother but laggier.
- Normalizer threshold `0.15` (YOLO ingest) vs `0.4` (MediaPipe/live): lower keeps low-conf joints under occlusion.
- Game tick `50 ms`, recording decimated `10 Hz` (max `5000` frames), paint throttled `~15 fps` + stall watchdog — tuned for integrated GPUs.

### Storage

Weights and videos are untracked (`*.pt`, `*.mp4` ignored; note `*.onnx` and `uv.lock` currently are not). Keep `yolo11m-pose.pt` for ingest and `yolo11n-pose.pt` for play side by side in the repo root; they are never committed.
