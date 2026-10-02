# Design

UX flow, visual language, and interaction decisions behind the desktop game.

## Principles

- **The song wins**: `effective_dancers()` prefers the song's `num_dancers` over the lobby count. Dancer identity is stable (`StableSlotMapper`, Hungarian by distance) so players can trust colors and labels.
- **Any dancer, repeatable**: several players may follow the same reference dancer. No exclusive locking.
- **Forgiving calibration, strict scoring**: light level never blocks calibration; empty/missing frames are neutral (neither add nor subtract).
- **Same palette everywhere**: `DANCER_PALETTE_BGR` (D1 green, D2 blue, D3 red, D4 cyan) in extractor previews, dancer cards, live overlay, and comparison view.

Related: [ARCHITECTURE.md](ARCHITECTURE.md), [USAGE.md](USAGE.md).

## Flow

```
Menu → SongSelect ([N dancers], search, import)
  → PlayerCount (1..4: Solo/Duet/Trio/Crew)
  → Join (T-pose claim, PlayerManager)
  → DancerSelect (any repeatable; auto-skip when num_dancers == 1)
  → Calibration → Game (single or per-player tick) → Result (per-player summary)
```

`MainWindow` is a `QStackedWidget` (`apps/desktop/ui/views.py`). All views implement `retranslate(lang)`; `MainWindow.apply_language()` hot-switches `en/es` without restart.

## Views

### Menu

`Play / Settings / Quit` + subtitle/footer. `Play → SongSelect`, `Settings → SettingsView`.

### SongSelect

- Search (`title/artist`, accent-insensitive), count `{shown} of {total}`.
- List stores `Song` in `Qt.UserRole`; cover from `cover.png`; info line `title — artist, duration/fps/dancers/poses`.
- Actions: `Play / Edit / Delete / Back / Import video…` + clickable drop zone (`mp4/mov/avi/mkv/webm`, drag & drop or click).
- Edit: title*/artist dialog. Delete: whole-folder confirm.

### PlayerCount (1–4)

Cards `1..4` (`120 px`), song hint (`{n} dancers (repeats)` vs `1 dancer · same choreo`), status line, `Continue to lobby / Back`. Selection → `PlayerManager.set_expected(n)` → lobby.

### Join (lobby)

`T-pose to join`. Left: camera. Right: `Players n/exp`, list `Player pid → Track tid (state)`, per-track `T-pose ID…%` + bar.

- Constants: `HOLD 0.6 s, DROPOUT_TOL 0.35 s, MIN_SCORE 0.55` (`detector.t_pose_score`, scale-invariant).
- `Add manually` (first free track, no pose; mono `Track 1` fallback from `shared.snapshot()` when `exp == 1`), `Remove last`, `Start` (disabled until `is_expected_fulfilled()`), `Back`.
- Mono shows a single central track without multi colors.

### DancerSelect

- Real photo cards cropped from `song.mp4` at the pose instant: `_collect_candidates` (≤16 uniform) + `_score_candidate` (overlap 0.30 / visibility 0.20 / sharpness 0.15 / size 0.15 / edge 0.10 / face=`head` 0.10); exact `box + D{n}` + skeleton drawn in the dancer's process color. Portrait-aware crop, `cover.png`/mid-frame fallback, cached.
- `Auto 1:1` (`i % n`), `Confirm and play` (blocked until every player has `reference_dancer_id != None`), `Back to lobby`.
- Skipped when `num_dancers == 1` (`_on_join_finished` auto-assigns `0` → Calibration).

### Calibration

Shows `Camera fps · brightness / backend`, `dark_warning < 25`, `Restart camera`. Steps: `body` (any pose) → `full` (shoulders/hips/wrists/ankles `vis ≥ 0.5`) → `light ≥ 40` (informational, never gates) → `stable` (head var `< 0.002` over ≤20 samples). `OK 2.5 s → countdown 3..1 → calibration_finished`.

### Game

`begin(song, player_manager)`. Per-player `GameController(PoseComparator(weights, angle_weight), Scorer(thresholds))` from Settings; `SongSynchronizer`; `QMediaPlayer + QAudioOutput + QVideoSink` (`song.mp4` else `song.mp3`). Seals `shared.reference_aspect = song_reference_aspect(song)`.

- Layout: default PiP (large reference + `160..320 px` camera bottom-left); `developer_mode` = legacy stacked.
- Rows `Ppid [Track tid → Dancer did]` + `StatBox / RatingBadge`.
- Tick `50 ms` (`_tick` / `_tick_multi`), recording decimated `10 Hz`, max `5000` frames, re-entrancy guard + `~15 fps` paint throttle + stall watchdog.
- Emits `{summary, recording, song, player_manager}` → Result.

### Result

Score, stars by accuracy (`≥80: 5, ≥60: 4, ≥40: 3, ≥20: 2, else 1`), details: multi `Ppid → Dancer did: pts · combo · acc% · P/G/Go/O/M`, else `max_combo / avg_acc / counts`.

`PoseComparisonWidget` (`set_data(song, recording, dancer_id=first_player)`): left = recorded normalized, right = `song.mp4` exact seek (`-0.6 s` back then read forward to `t`, because `CAP_PROP_POS_MSEC` lands on keyframes) + ALL raw poses in palette with the chosen one highlighted; slider in ms + `Play 50 ms`. `Play again / Menu`.

## Color system — Neon Stage (`apps/desktop/ui/theme.py`)

Source of truth: `COLORS`, `FONTS`, `RADIUS`, `APP_QSS` in `apps/desktop/ui/theme.py`.
Concept: dark graphite room + cyan spotlight + magenta only for rewards (the "signature":
game progress and Perfect badge use a cyan-to-magenta gradient).

| Token | Hex | Usage |
|---|---|---|
| `bg0` | `#0B0B10` | App background, stage |
| `bg1` | `#14141C` | Cards / surfaces, lists, inputs, progress track |
| `bg2` | `#1D1D28` | Hover / raised, buttons, slider groove, tooltips |
| `bg3` | `#23232E` | List item hover |
| `line` | `#2A2A38` | Borders, splitter handles |
| `text1` | `#F2F2F5` | Primary text |
| `text2` | `#A6A6B8` | Secondary text, hints, progress text |
| `text3` | `#6B6B80` | Tertiary / hints, disabled text |
| `accent` | `#00B0FF` | Cyan spotlight: primary buttons, selection, titles (`#display`), focus rings, slider handle, countdown/score |
| `accent_hi` | `#33C2FF` | Primary hover, slider hover |
| `accent_dim` | `#00344D` | Selection background, `countCard[checked]`, `cardHi`, drop-zone drag-over |
| `accent2` | `#E040FB` | Magenta, rewards ONLY (`QProgressBar#spotlight::chunk` cyan→magenta gradient) |
| `success` | `#00E676` | OK label, `cardHi` border |
| `warn` | `#FFB300` | Warning label (`#warn` on `#2A1E00` chip) |
| `danger` | `#FF5252` | Danger border/hover; danger button (`#danger`: bg `#2A1215`, border `#7A2020`, text `#FF8A80`); `#dangerText` |
| `ink` | `#0A0A0A` | Text on accent (selected list item, primary button) |

Extras hardcoded in QSS (not tokens): scrollbar handle `#33333F` (hover → `accent`),
video frame `#000000`, stars `#FFD700`, rating badge overlay `rgba(0,0,0,0.35)`.

Fonts (`FONTS`): display + body = `'Segoe UI', 'Inter', sans-serif` — normal width on
purpose (condensed faces like Barlow Condensed / Arial Narrow stretch glyphs vertically);
mono = `'Cascadia Mono', 'JetBrains Mono', Consolas, monospace` for `#mono` labels.
`load_fonts()` optionally registers `assets/fonts/Inter-*.ttf` if present, never downloads.

Radius (`RADIUS`): `sm 6 / md 8 / lg 12 / xl 16` — buttons `md`, cards/video/lists `lg`,
count cards `xl`, inputs `sm`.

QSS roles: `QLabel#display/title/subtitle/hint/micro/mono`, `QPushButton#primary/danger/ghost/countCard`,
`QFrame#card/dropZone/cardHi/video`, `QProgressBar#spotlight`, `QLabel#warn/dangerText/ok/rating/score/stars/countdown`.
Apply with `app.setStyleSheet(APP_QSS)` (see `theme.py` usage header).

Dancer skeleton colors are separate from the UI theme: `DANCER_PALETTE_BGR`
(D1 green, D2 blue, D3 red, D4 cyan) — see [SONG_FORMAT.md](SONG_FORMAT.md).

## Comparison widget specifics

`VideoSkeletonWidget.set_frame_at` seeks ~0.6 s back and reads forward to `t` for frame accuracy, then draws every dancer in their palette color with the active dancer highlighted. This is why scrubbing feels exact even on keyframed MP4s.

## Theming, icons, i18n, responsiveness

- `ui/theme.py`, `ui/icons.py` + `ui/assets/icons/*.svg` (~35 icons), `ui/responsive.py`, `ui/widgets.py`.
- `apps/desktop/i18n.py`: `STRINGS` `en/es` (~120 keys: menu/songs/pcount/join/dancer/calib/game/result/comp/imp/settings/main/import/camera), `tr(lang, key)`, English fallback, difficulty/count labels.
- Worker/thread messages emitted in Spanish are mapped at display (`_tr_camera_msg` / `_tr_device_msg`).

## Camera UX notes

DS → MSMF open fallback on integrated cams, negotiated-resolution check, `probe_gray()` guidance, `SharedState.reference_aspect` set by `GameView.begin`. In `CalibrationView` low light is a warning, not a blocker, when body + tracking exist.
