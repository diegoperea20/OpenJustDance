# AGENTS.md

Guide for AI agents (and developers) working on **openjustdance**.

## Environment

- OS: Windows, shell: **PowerShell 5.1**.
- Package manager: **uv**. Env lives in `.venv/`. Python 3.11
  (`.python-version`).
- Flat layout: top-level packages are `engine`, `shared`,
  `tools`, `apps`, `services` (declared in `[tool.setuptools.packages.find]`).
- Heavy deps are currently in main `dependencies` (not optional):
  `torch>=2.0`, `torchvision>=0.18` (CUDA index `pytorch-cu128` in
  `[tool.uv]`), `ultralytics>=8.3` (YOLO-Pose + trackers),
  `boxmot>=25.0.0`, `insightface>=2.0`, `onnxruntime-gpu==1.22.0`.
  `engine/pose_engine/backends/__init__.py` still keeps `try/except`
  guards around YOLO imports as a fallback (HOG+MediaPipe), but with a
  normal `uv sync` YOLO is expected to be present.
  Note: `pyproject.toml` still contains a stale comment suggesting a manual
  `uv pip install insightface onnxruntime-gpu` for FaceID — it contradicts
  the main dependencies list above; documented as-is, do not "fix" without
  being asked.
- Dev group: `[dependency-groups] dev` only contains `pytest>=8.0`.
  There is a stray top-level `dev = [pytest, black, ruff]` block in
  `pyproject.toml` that is NOT a valid dependency-group, so `black`/`ruff`
  are not installed via `uv sync --group dev`. Documented as-is.
- Local weights in repo root (do NOT commit, untracked via `*.pt`):
  `yolo11m-pose.pt` (ingest, accurate), `yolo11n-pose.pt` (live webcam,
  lightweight), `yolo26n-pose.pt`, `yolo26n-reid.onnx` (BoT-SORT ReID
  appearance model). Note: `*.onnx` is NOT in `.gitignore`, unlike `*.pt`.
  Local test videos are also uncommitted (covered by generic `*.mp4` rule:
  `aespanova.mp4`, `aespavideo.mp4`, `single.mp4`, `videotest*.mp4`).

## Commands

```powershell
uv sync                          # install dependencies (uses pyproject.toml)
uv sync --extra server           # adds engine_server deps (aiohttp)
uv sync --extra gpu              # redundant: torch/torchvision already in main deps (index pytorch-cu128)
uv run pytest                    # run all tests
uv run openjustdance             # launch desktop app (PySide6)
uv run pose-extractor video.mp4 -o songs/my_song [--title X] [--artist Y]
uv run engine-server --port 8765 # optional WebSocket server (requires --extra server)
# Multi-dancer extractor (YOLO-first, see Conventions). Full realistic example:
uv run pose-extractor aespavideo.mp4 -o songs/aespa --title "AESPA" --artist "AESPA" --estimator yolo --num-dancers 4 --yolo-model yolo11m-pose.pt --yolo-conf 0.4 --yolo-iou 0.6 --imgsz 960 --tracker botsort-reid --face-id --preview --dump-poses --poses-jsonl songs/aespa/poses.jsonl
```

## Conventions

- **The engine does NOT depend on PySide6.** `engine/`, `shared/` and `tools/`
  only import numpy/scipy/opencv/mediapipe (+ `torch`/`ultralytics`/
  `boxmot`/`insightface` in YOLO/FaceID paths). Qt dependencies live
  exclusively in `apps/`.
- When **adding a new package** (e.g. `engine/new_module/`), regenerate the
  editable install so `uv run` sees it:
  `uv sync --reinstall-package openjustdance`.
- **Console scripts** are defined in `[project.scripts]` (pyproject.toml):
  `openjustdance`, `pose-extractor`, `engine-server`.
- Dependencies: `numpy>=1.24.0` with NO upper bound (the old `<2.0` pin is
  gone). MediaPipe is pinned at `==0.10.14` (classic `mp.solutions.pose` API).
- **MediaPipe has no GPU support on Windows** (pip). Its inference is CPU. Do
  not try `delegate=GPU` from the tasks API on Windows. GPU auto-detection
  lives in `engine/pose_engine/hardware.py` (`gpu_info`/`detect_gpus`) and
  applies to YOLO (`--device auto|cpu|cuda`, Settings `device`), not to
  MediaPipe. To add a GPU backend, implement the `PoseBackend` interface
  (`engine/pose_engine/backends/base.py`) and register it in `BACKENDS`.
- Pose backends (`engine/pose_engine/backends/`, `BACKENDS` in `__init__.py`):
  `mediapipe-cpu` (always), `multi-mediapipe`, `yolo-pose` (+ aliases
  `yolo26n-pose`/`yolov8n-pose`/`yolo26n`), `yolo-mediapipe` (YOLO boxes+IDs +
  one MediaPipe per person). Registration here only depends on the
  `import` succeeding (`_HAS_YOLO`/`_HAS_YOLO_MP`); weight existence is
  checked inside `yolo_pose.py`, not in `__init__.py`.
  `create_backend("auto")` defaults to `mediapipe-cpu`, not YOLO.
  `PoseBackend.detect_multi()` is optional (default wraps `detect()` in 1
  full-frame bbox).
- Unified multi-person pipeline in `engine/pose_engine/multi.py`: **YOLO-Pose
  first** (1 forward, N people, global 0..1 coords, `track_multi_poses` with
  `TrackRegistry`), fallback to **sensitive HOG + tracker + MediaPipe per crop**
  (`detect_multi_poses`, `detector.py` + `tracking/tracker.py`) only when YOLO
  sees nobody. `StableSlotMapper` assigns stable `dancer_id` 0..N-1
  (Hungarian by distance, do not re-sort by x every frame). Used both by the
  offline extractor and live `CameraThread` (logic was unified before: do not
  reintroduce divergent single-sided gates like `multi_hits>=3`).
  Real models: ingest/multi = `yolo11m-pose.pt`, live webcam = `yolo11n-pose.pt`.
- Persistent YOLO trackers (`engine/pose_engine/trackers/*.yaml`, `--tracker`
  flag): only 3 yaml files exist on disk (`botsort-reid` = default,
  BoT-SORT+ReID anti-overlap; `botsort-reid-gmc` = +GMC when the camera moves,
  slow; `bytetrack-dance`), but `--tracker` offers 5 choices (plain `botsort`
  and `bytetrack` have no own yaml). ReID uses local `yolo26n-reid.onnx`.
- Extractor (`tools/pose_extractor/cli.py`, `extract.py`, `proc_entry.py`):
  `--estimator yolo|mp` (`yolo` = direct YOLO keypoints, recommended for multi;
  `mp` = YOLO boxes + MediaPipe per person; default `yolo`), `--yolo-conf`
  default `0.4` in CLI (`0.1-0.6`; UI import child `proc_entry.py` forces
  `0.25`), `--yolo-iou` default `0.6` (do not merge overlapping dancers),
  `--device auto|cpu|cuda` (default `auto`), `--yolo-model` (default `None`
  → multi ingest pins `yolo11m-pose.pt`), `--num-dancers 1-4` (default `1`;
  single and multi share one YOLO11m+tracking+FaceID process),
  `--face-id` (optional in CLI) + `--face-thresh 0.38`
  `--face-check-every 6` `--face-det-size 320`, `--kp-thresh 0.4`,
  `--smooth-alpha 0.6` (EMA per persistent ID), `--imgsz 960`,
  `--backend/--fps/--max-width/--start/--end/--title/--artist/--ffmpeg`,
  `--multi/--multi-max` (legacy; `auto_multi=True` in `ExtractOptions` already
  detects >1 person alone), `--no-yolo` (HOG+MediaPipe only),
  `--preview` → `preview_pose.mp4` (palette + `D{n}|ID{n}` labels; `·` is
  avoided because OpenCV Hershey has no such glyph), `--dump-poses` →
  `poses.json`, `--debug-track` → `track_debug.jsonl`, `--poses-jsonl` →
  `poses.jsonl` (`{frame,id,box,kpts17}` trackingdancers-style sidecar).
  `ExtractOptions` also has UI-only `rotation` (0/90/180/270).
  Per-dancer normalizer threshold: `0.15` for pure YOLO (conf 0.2-0.4 under
  occlusions), `0.4` for MediaPipe/live.
- FaceID (`engine/pose_engine/faceid.py`, NOT optional in UI): `FaceGallery`
  (`buffalo_l`, cosine `match_thresh=0.38`, re-check every 6 frames, face
  gate by `head >= 0.5`, EMA per persistent ID, `cuda_ep_usable()` via a mini
  ONNX `CUDAExecutionProvider` session, `tmp_{track_id}` identity when no
  face). CLI: opt-in via `--face-id` (fails with a clear message when
  `insightface` is missing). UI import (`proc_entry.py`): always
  `face_id=True` + `yolo11m-pose.pt` + `poses.jsonl` + preview/dump/debug.
- Song format is `openjustdance-1` (mono) / `openjustdance-2` (multi)
  (`shared/dance_format/models.py`). `FramePose.dancers: list[DancerPose]`
  takes priority over legacy `joints`/`raw`; `Song.num_dancers` (1 = legacy;
  `models.from_dict` infers by per-frame max, while the mode+thresholds
  heuristic lives in `extract.py` post-processing, not in the model). Poses
  are stored **normalized** (center = hips, scale = shoulder width, y up).
  Comparison is ALWAYS by time (`SongSynchronizer.frame_at()` /
  `dancer_pose_at()` per `dancer_id`), never by frame number. Frames with
  `joints={}` are neutral (neither add nor subtract). If `nd==1`, collapse
  bogus `dancers` into legacy single. `poses.jsonl`/`.poses.jsonl` is a
  lateral export (`ExtractOptions.poses_jsonl_path`, UI always
  `out/poses.jsonl`), NOT a field of `song.json` (`Song.to_dict` only
  serializes inline `poses:[...]`). `Song.reference_aspect` (video W/H sealed
  at import; fallback probe of `song.mp4`): the normalizer bakes aspect into
  y, so `CameraThread` pre-scales live raw x by k=`compare_space_prescale(ref,
  live)` before normalizing (`SharedState.reference_aspect` is set by
  `GameView.begin`). Without this, vertical (9:16) vs horizontal camera gives
  ~0.65 error = permanent Miss. `DANCER_PALETTE_BGR` in
  `engine/pose_engine/skeleton.py` is the single palette (D1 green, D2 blue,
  D3 red, D4 cyan) used by extractor preview (`D{n}|ID{n}`), cards and
  comparison.
- Camera (`engine/pose_engine/camera.py`, `apps/desktop/threads.py`):
  DS → MSMF open fallback on integrated cams, negotiated-resolution check,
  `probe_gray()` (std<8 + mean 115-140 = manual gray exposure). `SharedState`
  keeps legacy `pose/raw` (first track) + `snapshot_multi()`;
  `set_song_num_dancers(n)` + `request_tracking_reset()` on song change;
  `effective_dancers()` — the song wins over the lobby. In `CalibrationView`
  light is informational, not blocking when body+tracking exist.
- Multi UI flow (`apps/desktop/ui/`): `Menu → SongSelect ([N dancers],
  search bar, drag&drop import, edit/delete song) → PlayerCountView (1..4,
  sets expected_players) → JoinView (T-pose claim, PlayerManager) →
  DancerSelectView (any dancer repeatable; real photo cards cropped from
  song.mp4 at the pose instant + exact box + D{n} label + skeleton in process
  color; auto-skip when num_dancers==1) → Calibration → GameView
  (begin(song, player_manager), single or _tick_multi per player) →
  ResultView (per-JUG summary JUG n → Dancer N + comparison.py user vs
  reference with dancer_pose_at(t, chosen) + legacy fallback; missing dancer
  frames are neutral)`. `ImportDialog` (+ `workers/import_worker.py`,
  `import_process.py`, `proc_entry.py` QProcess child with atomic JSON
  progress, exit codes 0/1/2/3 incl. `CancelledError`/cancel-by-file) imports
  a local video as a song. Comparison widget (`comparison.py`,
  `VideoSkeletonWidget.set_frame_at`) seeks ~0.6s back and reads forward to t
  because CAP_PROP_POS_MSEC lands on keyframes; draws ALL dancers in their
  color with the chosen one highlighted. Theming/icons/i18n: `ui/theme.py`,
  `ui/icons.py` + `ui/assets/icons/*.svg` (~35 icons), `ui/responsive.py`,
  `ui/widgets.py`, `apps/desktop/i18n.py` (STRINGS en/es, tr(),
  difficulty/count labels). `PlayerManager`
  (`engine/tracking/player_manager.py`): maps TrackID↔PlayerID↔DancerID, no
  auto-reassignment (lost = LOST), `claim` requires a free TrackID.
- Settings (`apps/desktop/settings.py`): `camera_index/width/height, mirror,
  backend, device auto|cuda|cpu, language en|es (default en), difficulty
  presets easy/medium/hard + Perfect/Great/Good/Ok thresholds, angle_weight,
  smooth_alpha, weights, songs_dir (+OPENJUSTDANCE_SONGS env)`.
- `tests/` use pytest. Controller tests (`apps/desktop/controller.py`) are
  pure logic and do not require PySide6. Full suite (18 files):
  `test_aspect_compare`, `test_camera_select`, `test_compare_horizontal`,
  `test_controller`, `test_dance_format`, `test_dancer_select_ref`,
  `test_device_select`, `test_i18n`, `test_ingest_dancers`,
  `test_multi_pipeline`, `test_normalize_engine`, `test_score_engine`,
  `test_timing_engine`, `test_tpose`, `test_track_occlusion`,
  `test_tracker_cfg`, `test_yolo_mediapipe`, `test_yolo_multi4` (YOLO/weight/
  video ones need local untracked files; never commit them).
- Headless UI test: `$env:QT_QPA_PLATFORM="offscreen"` before importing PySide6.
- No secret commits. `.gitignore` covers `.venv/`, `__pycache__/`,
  `.pytest_cache/`, `.coverage`, `.vscode/`, `.idea/`, `.cache/`, `*.log`,
  `/songs`, `*.mp4`, `*.pt`. Note: `uv.lock` and `*.onnx` are currently NOT
  ignored (documented as-is).

## Structure

```
apps/desktop/        PySide6 UI: main.py, settings.py (camera/device/language/difficulty/songs_dir), controller.py, threads.py (multi CameraThread + SharedState), i18n.py (en/es), workers/import_worker.py+import_process.py (async cancelable import), ui/views.py, ui/player_count_view.py, ui/join_view.py, ui/dancer_select_view.py (real photo cards), ui/comparison.py (exact set_frame_at + palette), ui/import_dialog.py (num_dancers spin + cancel), ui/theme.py+icons.py+responsive.py+widgets.py, ui/assets/icons/*.svg
engine/pose_engine/  camera.py, detector.py (HOG+NMS+T-pose), multi.py (YOLO-first pipeline + TrackRegistry/StableSlotMapper), faceid.py (FaceGallery buffalo_l), hardware.py (gpu_info/detect_gpus), skeleton.py (DANCER_PALETTE_BGR), backends/ (base.py, mediapipe.py, multi_mediapipe.py, yolo_pose.py, yolo_mediapipe.py), trackers/ (botsort-reid.yaml, botsort-reid-gmc.yaml, bytetrack-dance.yaml)
engine/tracking/     player_manager.py (TrackID↔PlayerID↔DancerID), tracker.py
engine/normalize_engine/  normalizer.py, smoother.py
engine/score_engine/      comparator.py (compare_space_prescale), scoring.py
engine/timing_engine/     synchronizer.py (frame_at/dancer_pose_at)
services/engine_server/   server.py (optional, aiohttp)
shared/dance_format/      models.py (openjustdance-1/2, DancerPose, num_dancers, reference_aspect), loader.py
tools/pose_extractor/     extract.py (ExtractOptions multi/YOLO + preview/dump/debug/jsonl), cli.py, proc_entry.py (UI import child: yolo11m + face_id + jsonl always)
tests/                    pytest (18 files: aspect_compare, camera_select, compare_horizontal, controller, dance_format, dancer_select_ref, device_select, i18n, ingest_dancers, multi_pipeline, normalize_engine, score_engine, timing_engine, tpose, track_occlusion, tracker_cfg, yolo_mediapipe, yolo_multi4)
songs/                    generated songs, ignored at root (/songs): song.json, song.mp4, song.mp3, cover.png + optional preview_pose.mp4/poses.json/poses.jsonl/track_debug.jsonl
```
