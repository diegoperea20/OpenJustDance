# Song format

`openjustdance-1` (mono) / `openjustdance-2` (multi). Code: `shared/dance_format/models.py`, `loader.py`.

## song.json

```json
{
  "title": "AESPA",
  "artist": "AESPA",
  "fps": 30.0,
  "duration": 42.5,
  "format": "openjustdance-2",
  "joints_order": ["nose", "..."],
  "num_dancers": 4,
  "reference_aspect": 0.5625,
  "poses": [
    {
      "time": 0.033,
      "joints": {"nose": [x, y, z, vis], "...": []},
      "raw": {"nose": [x, y, z, vis]},
      "dancers": [
        {"dancer_id": 0, "joints": {}, "raw": {}},
        {"dancer_id": 1, "joints": {}, "raw": {}}
      ]
    }
  ]
}
```

Rules:

- `openjustdance-1`: per-frame `joints`/`raw` (single dancer), `num_dancers = 1`.
- `openjustdance-2`: per-frame `dancers: list[DancerPose]` **takes priority**; `joints`/`raw` mirror `dancers[0]` for compat. `num_dancers > 1`.
- Each joint is `[x, y, z, vis]`. `joints` = normalized/canonical (center = hips, scale = shoulder width, y-up); `raw` = `0..1` full-frame coords.
- `FramePose.get_dancer(id)`, legacy fallback `id == 0`. If `nd == 1`, bogus `dancers` collapse into legacy single.
- `Song.from_dict` infers `num_dancers` by per-frame max (format `-2` forces `>= 2`); the mode+thresholds heuristic lives in `extract.py` post-processing, not in the model.
- `reference_aspect` (video W/H sealed at import; fallback probe of `song.mp4`): validated `0.2..5.0`. `loader.song_reference_aspect()`: field → probe `song.mp4` → `None` (legacy). `0.5625` = vertical 9:16, `1.78` = horizontal 16:9.
- `validate_song()`: only checks title/duration/negative time. `find_songs()` caches by `(mtime, size)`.

## Timing and neutrality

- Comparison is **always by time** (`SongSynchronizer.frame_at()` / `dancer_pose_at()` per `dancer_id`, bisection), never by frame number.
- Frames with `joints = {}` are **neutral**: neither add nor subtract. Missing-dancer frames in multi are neutral too. `GameController` returns Neutral (no combo change) on empty ref.
- Result comparison for player P uses `dancer_pose_at(t, chosen_dancer)` + legacy fallback.

## Normalization

The normalizer bakes aspect into y, so `CameraThread` pre-scales live raw x by `k = compare_space_prescale(ref, live)` before normalizing (`SharedState.reference_aspect` is set by `GameView.begin`). Without this, vertical vs horizontal gives ~0.65 error = permanent Miss.

- Origin: hip center (fallback shoulder center). Scale: shoulder width (fallback torso × 2). Hips de-rotated to horizontal, y flipped up.
- Per-dancer threshold: `0.15` for pure YOLO ingest (conf `0.2–0.4` under occlusions), `0.4` for MediaPipe/live.
- `poses.jsonl` / `.poses.jsonl` is a lateral export (`ExtractOptions.poses_jsonl_path`, UI always `out/poses.jsonl`), **not** a field of `song.json` (`Song.to_dict` only serializes inline `poses:[...]`).

## Sidecar files

A song folder holds more than `song.json` (see [USAGE.md](USAGE.md)):

| File | Producer | Content |
|---|---|---|
| `song.mp4` | copy of input | Reference video |
| `song.mp3` | `ffmpeg libmp3lame -q:a 2` | Audio fallback |
| `cover.png` | first frame | Thumbnail + dancer-card fallback |
| `preview_pose.mp4` | `--preview` | Palette skeleton + `D{n}\|ID{n}` labels (avoids `·` — OpenCV Hershey has no such glyph) |
| `poses.json` | `--dump-poses` | Raw `[{frame, persons:[{person, keypoints}]}]` |
| `poses.jsonl` | `--poses-jsonl` (UI always) | One JSON/line: `{frame (1-based sampled), id (stable FaceID), box [x1,y1,x2,y2] px, kpts [[x,y,conf]×17 COCO-17 px]}` trackingdancers-style |
| `track_debug.jsonl` | `--debug-track` | tids + per-frame association decisions |

## Dancer palette

Single source `DANCER_PALETTE_BGR` in `engine/pose_engine/skeleton.py`: D1 green, D2 blue, D3 red, D4 cyan. Used by extractor preview, cards, and comparison.
