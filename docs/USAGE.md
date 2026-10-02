# Usage

How to play, import videos, and use the CLI. For the file spec see [SONG_FORMAT.md](SONG_FORMAT.md); for internals see [ARCHITECTURE.md](ARCHITECTURE.md).

## Play a song

1. `uv run openjustdance`
2. `Play → SongSelect` — pick a song (`[N dancers]` badge), or import one below.
3. `PlayerCount` — choose 1–4 players.
4. `Join` — hold a T-pose until the bar fills (or `Add manually`).
5. `DancerSelect` — each player picks any reference dancer (repeats allowed). Auto-skipped for 1-dancer songs.
6. `Calibration` — stand still until the countdown.
7. `Game` — follow the large reference video; your camera is the small overlay. Ratings appear live (`Perfect / Great / Good / Ok / Miss`).
8. `Result` — score, stars, per-player breakdown, comparison scrubber.

Tips: mirror is on by default; if scores look permanently low with a vertical reference video, check that `reference_aspect` was sealed (import path does it automatically).

## Import from the UI

`SongSelect → Import video…` (or drop a video onto the drop zone):

- Fields: `Title* / Artist / Folder slug / Rotation Auto/0/90/180/270 / FPS Auto (max 30) / Nº dancers 1–4 / Long side 320..1920 (default 1280)`. The dialog shows a `WxH · vertical/horizontal` hint.
- Backend: `QProcess python -m tools.pose_extractor.proc_entry payload.json`; progress polls `progress.json ~8 Hz`; cancel via `cancel.flag` + `kill()` after 3 s; partial dir cleaned on cancel; `import.log` tail shown on `exit != 0`.
- Always uses `yolo11m-pose.pt + FaceID + poses.jsonl` (`conf 0.25, imgsz 960`, preview/dump/debug on).
- If `declared > observed` you get a decision: `Keep {observed} / Cancel`. Otherwise it saves `song.json + song.mp4 + cover.png + song.mp3 + poses.jsonl` and seals `reference_aspect` (rotation inverted at `90/270`).

## Import from the CLI

Basic (single dancer):

```powershell
uv run pose-extractor video.mp4 -o songs/my_song --title "Title" --artist "Artist"
```

Realistic multi-dancer (YOLO-first, recommended):

```powershell
uv run pose-extractor aespavideo.mp4 -o songs/aespa --title "AESPA" --artist "AESPA" `
  --estimator yolo --num-dancers 4 --yolo-model yolo11m-pose.pt `
  --yolo-conf 0.4 --yolo-iou 0.6 --imgsz 960 `
  --tracker botsort-reid --face-id --preview --dump-poses --poses-jsonl songs/aespa/poses.jsonl
```

MediaPipe-per-person variant (YOLO boxes + MediaPipe keypoints):

```powershell
uv run pose-extractor video.mp4 -o songs/my_song --estimator mp --num-dancers 2 --face-id
```

### Full flag reference (`tools/pose_extractor/cli.py`)

| Flag | Default | Notes |
|---|---|---|
| `video` (positional) | — | Input `.mp4/.mov/...` |
| `-o / --output` | required | Song output dir |
| `--title / --artist` | `""` | Stored in `song.json` |
| `--fps` | video fps | Sampling fps, max 30 |
| `--max-width` | `1280` | Downscale long side for processing |
| `--backend` | `auto` | `auto\|mediapipe-cpu\|multi-mediapipe\|yolo-pose\|yolo26n-pose\|yolov8n-pose`. YOLO multi is primary unless `--no-yolo` |
| `--estimator` | `yolo` | `yolo` = direct YOLO keypoints (recommended multi); `mp` = YOLO boxes+IDs + 1 MediaPipe/person |
| `--tracker` | `botsort-reid` | `botsort-reid\|botsort-reid-gmc\|botsort\|bytetrack\|bytetrack-dance`. See [ARCHITECTURE.md](ARCHITECTURE.md) |
| `--device` | `auto` | `auto\|cpu\|cuda`. `auto` uses GPU if present |
| `--yolo-conf` | `0.4` | Range `0.1–0.6`. UI import child forces `0.25` |
| `--yolo-iou` | `0.6` | Do not merge overlapping dancers |
| `--yolo-model` | `None` | Defaults to `yolo11m-pose.pt` in multi ingest |
| `--imgsz` | `960` | YOLO ingest inference size |
| `--num-dancers` | `1` | `1–4`. Single and multi share one process |
| `--multi / --multi-max` | off / `4` | Legacy; `auto_multi=True` in `ExtractOptions` already detects >1 person |
| `--face-id` | off | Opt-in in CLI; always on in UI import. Fails clearly if `insightface` is missing |
| `--face-thresh` | `0.38` | Cosine gallery threshold |
| `--face-check-every` | `6` | Re-verify every N frames |
| `--face-det-size` | `320` | Face detector resolution over crop |
| `--kp-thresh` | `0.4` | Min keypoint confidence |
| `--smooth-alpha` | `0.6` | EMA per persistent ID |
| `--no-yolo` | off | HOG + MediaPipe only |
| `--start / --end` | `0.0` / end | Time slice in seconds |
| `--backend / --ffmpeg` | `auto` / `ffmpeg` | Detection backend / ffmpeg binary path |
| `--preview` | off | → `preview_pose.mp4` (palette + `D{n}\|ID{n}` labels) |
| `--dump-poses` | off | → `poses.json` |
| `--debug-track` | off | → `track_debug.jsonl` |
| `--poses-jsonl` | off | → `poses.jsonl` sidecar `{frame,id,box,kpts17}` |
| `ExtractOptions.rotation` | `0` | UI-only `0/90/180/270` |

Outputs in `out_dir/`: `song.json`, `song.mp4` (copy), `song.mp3` (`ffmpeg libmp3lame -q:a 2`), `cover.png` (first frame) + optionals above. UI-only `rotation` is applied at import.

## Settings

`%APPDATA%/openjustdance/settings.json`. Songs dir resolution: Settings `songs_dir` → `$OPENJUSTDANCE_SONGS` → repo `songs/`.

| Key | Values | Default |
|---|---|---|
| `camera_index/width/height` | camera picker | `0 / 640 / 480` |
| `mirror` | bool | `true` |
| `backend` | `auto\|mediapipe-cpu` | `auto` |
| `device` | `auto\|cuda\|cpu` | `auto` |
| `language` | `en\|es` | `en` |
| `difficulty` | `easy\|medium\|hard` | `medium` (auto-fills thresholds + angle weight) |
| `perfect/great/good/ok_threshold` | `0..1` spins | `0.05/0.10/0.20/0.30` |
| `angle_weight` | `0..1` | `0.30` |
| `smooth_alpha` | `0..1` | `0.5` |
| `developer_mode` | bool | `false` (PiP; `true` = side-by-side) |

Hardware line shows `gpu_summary()`.

## Engine server (optional)

```powershell
uv sync --extra server
uv run engine-server --port 8765
```

Requires the `server` extra (`aiohttp`). See `services/engine_server/server.py`.
