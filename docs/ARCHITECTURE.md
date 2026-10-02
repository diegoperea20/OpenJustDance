# Architecture

How OpenJustDance is put together: offline ingest, live play, and the engines in between.

## Big picture

```
video.mp4
  └─ tools/pose_extractor (offline, yolo11m, imgsz 960)
       └─ songs/<name>/song.json + song.mp4/mp3/cover.png (+ poses.jsonl)
            └─ apps/desktop (live, yolo11n, 640px)
                 ├─ engine/pose_engine   detection + tracking + FaceID
                 ├─ engine/normalize_engine  hip-center / shoulder-width norm
                 ├─ engine/score_engine      compare + rate + points
                 ├─ engine/timing_engine     lookup by time
                 └─ engine/tracking          TrackID <-> PlayerID <-> DancerID
```

Two modes share the same concepts but use different models: accurate `yolo11m-pose.pt` for ingest, lightweight `yolo11n-pose.pt` for the webcam. The engine never depends on PySide6 — Qt lives exclusively in `apps/`.

Related: [SONG_FORMAT.md](SONG_FORMAT.md), [USAGE.md](USAGE.md), [PRIVACY_PERFORMANCE.md](PRIVACY_PERFORMANCE.md).

## Packages

| Path | Role |
|---|---|
| `engine/pose_engine/` | `camera.py`, `detector.py` (HOG+NMS+T-pose), `multi.py` (YOLO-first pipeline + `TrackRegistry`/`StableSlotMapper`), `faceid.py` (`FaceGallery`), `hardware.py` (`gpu_info`/`detect_gpus`), `skeleton.py` (`DANCER_PALETTE_BGR`, `SEMANTIC_JOINTS`), `backends/`, `trackers/*.yaml` |
| `engine/tracking/` | `player_manager.py` (no auto-reassignment; lost = LOST), `tracker.py` |
| `engine/normalize_engine/` | `normalizer.py` (`BodyNormalizer`), `smoother.py` (`PoseSmoother` EMA) |
| `engine/score_engine/` | `comparator.py` (`PoseComparator`, `compare_space_prescale`), `scoring.py` (`Scorer`) |
| `engine/timing_engine/` | `synchronizer.py` (`frame_at`, `dancer_pose_at` via bisection, by time) |
| `shared/dance_format/` | `models.py` (`openjustdance-1/2`, `DancerPose`, `num_dancers`, `reference_aspect`), `loader.py` |
| `tools/pose_extractor/` | `extract.py` (`ExtractOptions`), `cli.py`, `proc_entry.py` (UI import child) |
| `apps/desktop/` | `main.py`, `controller.py` (pure logic), `threads.py` (`CameraThread`+`SharedState`), `settings.py`, `i18n.py`, `workers/`, `ui/` |
| `services/engine_server/` | Optional aiohttp WebSocket server |

## Offline ingest pipeline

Single declared-count path (`extract()` → `_extract_declared_multi()` for `num_dancers` 1–4). All dancers share one `YOLO11m + tracking + FaceID` process.

1. `YoloPoseBackend(model=yolo11m-pose.pt, tracker=botsort-reid)` → `track_multi(frame, imgsz)` per sampled frame. Global `0..1` coords, persistent `track_id`.
2. `FaceGallery(buffalo_l).resolve_identity(tid, box, frame)` — `tmp_{tid}` until a face is seen; head gate `head >= 0.5`, re-check every N frames (`--face-check-every 6`).
3. `pid → slot (dancer_id 0..N-1)` capped at `num_dancers`; EMA `smooth_alpha` (default `0.6`) on `0..1` coords per persistent ID.
4. Per-slot `BodyNormalizer(visibility_threshold=0.15)` + `PoseSmoother` → normalized `joints` (center = hips, scale = shoulder width, y-up) and `raw` (`0..1`). Sorted by `dancer_id`.
5. Empty frame → `FramePose(time, joints={}, raw={}, dancers=[])` — neutral, never scored.
6. `Song.format = openjustdance-2 if observed_stable_ids > 1 else openjustdance-1`, `num_dancers = max(1, observed)`.

`proc_entry.py` (UI child, `python -m tools.pose_extractor.proc_entry <payload.json>`) forces `yolo11m-pose.pt + face_id=True + yolo_conf=0.25 + imgsz=960` and always writes `preview_pose.mp4 / poses.json / track_debug.jsonl / poses.jsonl`. IPC via atomic `progress_path` JSON `{i,total,msg}`, `cancel_path` flag, `result_path` song JSON. Exit codes: `0 ok / 3 cancelled / 2 bad payload / 1 error`.

## Live pipeline

`CameraThread` (`apps/desktop/threads.py`) + `SharedState`:

- Open with DirectShow → MediaFoundation fallback; negotiated-resolution check; `probe_gray()` (`std < 8` + mean `115–140` = manual gray exposure); dark `< 20` triggers exposure handling.
- Fixed working width `640`, optional mirror flip.
- Live YOLO always `yolo11n-pose.pt`, `conf 0.4`, `iou 0.6`, `botsort-reid`, `max 1..4` persons (`expected_players` sets the cap). MediaPipe is used only if YOLO is missing.
- `health_check` with `cuda → cpu` cascade + warning.
- `compare_space_prescale(ref, live)` before normalizing: `x *= k`, `k = live_aspect / ref_aspect` clamped to `0.2..5.0`. Without this, vertical (9:16) reference vs horizontal camera gives ~0.65 permanent error.
- `SharedState` keeps legacy `pose/raw` (first track, compat) + `tracks{tid: TrackedPerson(bbox, raw, norm)}` + `fps/brightness/frame_size/reference_aspect`. `set_song_num_dancers(n)` + `request_tracking_reset()` on song change; `effective_dancers()` — the song wins over the lobby.

## Pose backends

Interface `engine/pose_engine/backends/base.py: PoseBackend`: `detect(frame) -> Pose` (`0..1` x/y, y-down), optional `detect_multi(frame) -> list[(x,y,w,h,Pose)]` (default wraps `detect()` in a full-frame bbox), `close()`.

| Name | Class | Notes |
|---|---|---|
| `mediapipe-cpu` | `MediaPipeBackend` | Always available. `create_backend("auto")` defaults here, not YOLO |
| `multi-mediapipe` | `MultiMediapipeBackend` | One MediaPipe per person |
| `yolo-pose` (+ aliases `yolo26n-pose`, `yolov8n-pose`, `yolo26n`) | `YoloPoseBackend` | Registered only if the YOLO import succeeds (`_HAS_YOLO`) |
| `yolo-mediapipe` | `YoloMediapipeBackend` | YOLO boxes+IDs + one MediaPipe per person. Needs `_HAS_YOLO_MP` |

Weight existence is checked inside `yolo_pose.py`, not at registration. `pick_device(want="auto")` → `cuda` if `torch.cuda.is_available()` else `cpu`; FP16 only on `cuda*`. MediaPipe on Windows pip has no GPU support — always CPU.

## Tracking and identity

- **Unified multi-person logic** (`engine/pose_engine/multi.py`): YOLO-Pose first (1 forward, N people, global `0..1` coords, `track_multi_poses` with `TrackRegistry`); fallback to sensitive HOG + tracker + MediaPipe per crop (`detect_multi_poses`, `detector.py` + `tracking/tracker.py`) only when YOLO sees nobody.
- **`StableSlotMapper`**: Hungarian assignment by distance. Never re-sort by x every frame — slots stay stable through crossings.
- **Persistent YOLO trackers** (`engine/pose_engine/trackers/`): only 3 yamls on disk — `botsort-reid.yaml` (default, BoT-SORT + ReID anti-overlap, model `yolo26n-reid.onnx`), `botsort-reid-gmc.yaml` (+GMC when the camera moves, slow), `bytetrack-dance.yaml` (tuned ByteTrack, no ReID, fast). CLI offers 5 choices; plain `botsort`/`bytetrack` map to Ultralytics stock yamls.
- **FaceID** (`engine/pose_engine/faceid.py`, `FaceGallery`): `buffalo_l`, cosine `match_thresh=0.38`, re-check every 6 frames (`--face-check-every`), face gate `head >= 0.5`, EMA per persistent ID, `cuda_ep_usable()` mini-ONNX probe (`CUDAExecutionProvider` + `torch.cuda.is_available()`), `tmp_{track_id}` identity when no face. CLI opt-in via `--face-id`; UI import always on.
- **`PlayerManager`** (`engine/tracking/player_manager.py`): maps `TrackID ↔ PlayerID ↔ DancerID`, no auto-reassignment (lost = LOST), `claim` requires a free `TrackID`.

## Normalization, comparison, scoring, timing

- **`BodyNormalizer`**: origin = hip center (fallback shoulder center), scale = shoulder width (fallback torso × 2), de-rotate hips to horizontal, flip y-up. Threshold `0.15` for pure-YOLO ingest (conf `0.2–0.4` under occlusions), `0.4` for MediaPipe/live.
- **`PoseSmoother`**: per-joint EMA (`alpha` `0.5` normalizer / `0.6` ingest), drops stale joints.
- **`PoseComparator.compare(player, ref)`**: weighted position error (`DEFAULT_WEIGHTS`: head 5%, shoulders/elbows/knees/ankles 7.5% ×2, wrists 12.5% ×2, hips 5% ×2) with centroid + bounded Kabsch rotation (`align_max_deg=30°`) + 8 angle-triplet error; `error = 0.7 * pos + 0.3 * ang`. Visibility ramp `VIS_FLOOR=0.10 → threshold`, `MIN_COVERAGE=1.0`; `tracking=False → error=None`.
- **`Scorer`** (`engine/score_engine/scoring.py`): `Perfect ≤ 0.05, Great ≤ 0.10, Good ≤ 0.20, Ok ≤ 0.30, else Miss`; `score = max_score * (1 - min(err, 1))`, points `10/8/6/4/0`. `GameController.evaluate(player, ref)`: empty ref = Neutral (no points/combo change), else `compare → rate → points`; Miss resets combo.
- **`SongSynchronizer`** (`engine/timing_engine/synchronizer.py`): `frame_at(t)` / `dancer_pose_at(t, id)` — nearest pose **by time** (bisection), never by frame index. Gaps/empty = neutral.
- **Single palette**: `DANCER_PALETTE_BGR` in `engine/pose_engine/skeleton.py` (D1 green, D2 blue, D3 red, D4 cyan) used by extractor preview (`D{n}|ID{n}`), cards, and comparison.
