# Development

Setup, test, and troubleshooting guide. Conventions: [CONTRIBUTING.md](CONTRIBUTING.md). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

## Environment

- OS: Windows, shell PowerShell 5.1. Package manager **uv**. Env in `.venv/`. Python 3.11 (`.python-version`).
- Flat layout: top-level packages `engine`, `shared`, `tools`, `apps`, `services` (`[tool.setuptools.packages.find]`).
- Heavy deps in main `dependencies`: `torch>=2.0`, `torchvision>=0.18` (CUDA index `pytorch-cu128`), `ultralytics>=8.3`, `boxmot`, `insightface>=2.0`, `onnxruntime-gpu==1.22.0`.
- Dev group `[dependency-groups] dev` = `pytest>=8.0` only.

## Commands

```powershell
uv sync                          # install (uses pyproject.toml)
uv sync --extra server           # + engine-server deps (aiohttp)
uv sync --extra gpu              # redundant: torch/torchvision already in main deps
uv run pytest                    # all tests
uv run pytest tests/test_controller.py -v   # single file
uv run openjustdance             # desktop app (PySide6)
uv run pose-extractor video.mp4 -o songs/my_song [--title X] [--artist Y]
uv run engine-server --port 8765 # optional WebSocket server (needs --extra server)
```

Headless UI test:

```powershell
$env:QT_QPA_PLATFORM="offscreen"
uv run pytest
```

When adding a new package (e.g. `engine/new_module/`):

```powershell
uv sync --reinstall-package openjustdance
```

## Tests (18 files)

Pure / fakes / mocked — no weights or video needed:

`test_camera_select`, `test_compare_horizontal`, `test_controller` (pure logic, no PySide6), `test_dance_format`, `test_device_select` (monkeypatches torch/hw), `test_i18n`, `test_ingest_dancers` (dummy paths, parser defaults), `test_multi_pipeline` (`FakeYolo/FakeMp`), `test_normalize_engine`, `test_score_engine`, `test_timing_engine`, `test_tpose`, `test_track_occlusion`, `test_tracker_cfg` (asserts committed yamls + `yolo26n-reid.onnx` string + fake YOLO), `test_yolo_mediapipe` (fakes), `test_yolo_multi4` (fakes, `ExtractOptions` defaults).

Need local untracked media/song (skip if missing, never commit `*.pt/*.mp4`):

- `test_aspect_compare.py` — `README-assets/newjeansshort.mp4` (720×1280) + `songs/videotest1short/song.json`. Guarded by `pytest.mark.skipif` when the local media is absent.
- `test_dancer_select_ref.py` — synthetic `Song`, except 2 tests guarded by `exists()` on `songs/aespavideo/song.mp4|song.json`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| YOLO falls back to CPU with warning | `health_check` CUDA → CPU cascade. Check `torch.cuda.is_available()`, `hardware.gpu_summary()`, NVIDIA driver / CUDA build. |
| ONNX CUDA noise in logs | Expected: `silence_onnx_cuda_noise()` sets ORT logger to FATAL; `yolo26n-reid.onnx` usually runs CPU fallback. Probe via `faceid.cuda_ep_usable()`. |
| Camera gray / dark | `probe_gray()` (`std<8` + mean `115–140` = manual gray exposure), dark `<20` triggers exposure handling. Try another `camera_index`, DS → MSMF fallback is automatic. |
| Import fails with FaceID error | CLI needs `--face-id` + installed `insightface`. UI always requires it — install main deps via `uv sync`. |
| Comparison scrubber lands on wrong frame | `CAP_PROP_POS_MSEC` lands on keyframes — the widget seeks ~0.6 s back and reads forward (by design). |
| `uv run` does not see a new package | Run `uv sync --reinstall-package openjustdance`. |
| PySide6 import fails in CI/headless | Set `$env:QT_QPA_PLATFORM="offscreen"` before import. |
| `black`/`ruff` missing after sync | Expected: stray top-level `dev = [...]` block is not a valid group; only `pytest` is in `[dependency-groups] dev`. |
| No poses extracted | Check `--yolo-conf` (`0.1–0.6`; UI forces `0.25`), `--yolo-iou 0.6`, `--imgsz`, and that weights exist locally. |

Songs dir resolution: Settings `songs_dir` → `$OPENJUSTDANCE_SONGS` → repo `songs/`. Config at `%APPDATA%/openjustdance/settings.json`.
