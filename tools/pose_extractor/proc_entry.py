"""Child import process entry point.

Launched by the parent (VideoImportProcess via QProcess) as:
    <python> -m tools.pose_extractor.proc_entry <payload.json>

The JSON payload holds the extraction options and the file-based IPC paths
(progress + cancellation flag + result). Communication:

* progress: the child rewrites ``progress_path`` (atomic tmp+rename JSON)
  with {"i":..,"total":..,"msg":..}; the parent polls it with a QTimer.
* cancellation: the parent creates ``cancel_path``; the child checks it every
  frame via on_progress/should_cancel and aborts with CancelledError.
* result: the child saves ``result_path`` (song.json) with save_song.
* stdout/stderr: the parent redirects them to import.log.

Exit codes: 0 ok, 3 cancelled by the user, 2 invalid payload, 1 error.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_PAYLOAD = 2
EXIT_CANCELLED = 3


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return EXIT_OK if argv else EXIT_BAD_PAYLOAD
    payload_path = Path(argv[0])
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        video = Path(payload["video"])
        out_dir = Path(payload["out_dir"])
        progress_path = Path(payload["progress_path"])
        cancel_path = Path(payload["cancel_path"])
        result_path = Path(payload["result_path"])
    except Exception as exc:
        print(f"[proc_entry] payload inválido: {exc}", file=sys.stderr)
        return EXIT_BAD_PAYLOAD

    fps = payload.get("fps")
    if fps is not None:
        fps = float(fps)

    from shared.dance_format.loader import save_song
    from tools.pose_extractor.extract import CancelledError, ExtractOptions, extract

    def is_cancelled() -> bool:
        try:
            return cancel_path.exists()
        except OSError:
            return False

    last_write = 0.0

    def report(i: int, total: int, msg: str = "") -> None:
        nonlocal last_write
        now = time.monotonic()
        # Throttle: the parent polls at ~8Hz; writing more is wasteful.
        if now - last_write < 0.15 and i != total:
            return
        last_write = now
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        _atomic_write_json(progress_path, {"i": int(i), "total": int(total), "msg": msg})

    def on_progress(i: int, total: int):
        report(i, total)
        # True = abort (extract turns it into CancelledError).
        return is_cancelled()

    print(f"[proc_entry] extrayendo {video.name} ...", flush=True)
    try:
        if not video.is_file():
            print(f"[proc_entry] no existe el video: {video}", file=sys.stderr)
            return EXIT_FAILED
        options = ExtractOptions(
            video=video,
            title=str(payload.get("title", "")),
            artist=str(payload.get("artist", "")),
            fps=fps,
            max_width=int(payload.get("max_width", 1280)),
            rotation=int(payload.get("rotation", 0)),
            num_dancers=int(payload.get("num_dancers", 1) or 1),
            multi_max=int(payload.get("num_dancers", 1) or 1),
            yolo_model="yolo11m-pose.pt",
            face_id=True,
            yolo_conf=0.25,
            imgsz=960,
            preview_path=(out_dir / "preview_pose.mp4"),
            dump_poses_path=(out_dir / "poses.json"),
            debug_track_path=(out_dir / "track_debug.jsonl"),
            poses_jsonl_path=(out_dir / "poses.jsonl"),
        )
        report(0, 1, "Extrayendo poses...")
        song = extract(options, on_progress=on_progress, should_cancel=is_cancelled)
    except CancelledError:
        print("[proc_entry] cancelado por el usuario", flush=True)
        return EXIT_CANCELLED
    except Exception:
        traceback.print_exc()
        return EXIT_FAILED

    if is_cancelled():
        print("[proc_entry] cancelado al terminar", flush=True)
        return EXIT_CANCELLED
    try:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        save_song(song, result_path)
    except Exception:
        traceback.print_exc()
        return EXIT_FAILED
    report(1, 1, "Listo")
    print(f"[proc_entry] ok: {len(song.poses)} poses -> {result_path}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
