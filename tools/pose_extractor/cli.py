"""pose-extractor CLI.

Usage:
    uv run pose-extractor video.mp4 -o songs/my_song [--title X] [--artist Y]
        [--fps 30] [--max-width 1280] [--backend auto]

Generates in the output directory: song.mp4, song.mp3, cover.png and song.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from tools.pose_extractor.extract import (
    ExtractOptions,
    extract,
    extract_audio,
    make_cover,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose-extractor",
        description="Convierte un video de baile en una coreografia jugable (song.json).",
    )
    parser.add_argument("video", help="Ruta del video de entrada (.mp4, .mov, ...)")
    parser.add_argument(
        "-o", "--output", required=True, help="Directorio de salida de la cancion"
    )
    parser.add_argument("--title", default="", help="Titulo de la cancion")
    parser.add_argument("--artist", default="", help="Artista de la cancion")
    parser.add_argument(
        "--fps", type=float, default=None, help="FPS de muestreo (por defecto el del video, max 30)"
    )
    parser.add_argument(
        "--max-width", type=int, default=1280, help="Ancho maximo de procesamiento (downscale)"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "mediapipe-cpu", "multi-mediapipe", "yolo-pose", "yolo26n-pose", "yolov8n-pose"],
        help="Backend de deteccion por persona (YOLO multi se usa igual como primario salvo --no-yolo)",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.4, help="Confianza YOLO-Pose (0.1-0.6, default 0.4)")
    parser.add_argument("--yolo-iou", type=float, default=0.6, help="IoU NMS de YOLO-Pose (0.6 no fusiona dancers solapados)")
    parser.add_argument(
        "--tracker",
        default="botsort-reid",
        choices=["botsort-reid", "botsort-reid-gmc", "botsort", "bytetrack", "bytetrack-dance"],
        help="Tracker persistente: botsort-reid = BoT-SORT + ReID (default); botsort-reid-gmc = +GMC si la camara se mueve (lento); bytetrack = ByteTrack rapido",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device YOLO: auto usa GPU si existe (default), o fuerza cpu/cuda",
    )
    parser.add_argument(
        "--estimator",
        default="yolo",
        choices=["mp", "yolo"],
        help="yolo: keypoints YOLO directos puros, 1 forward N personas (recomendado multi); mp: YOLO boxes+IDs + un MediaPipe por persona",
    )
    parser.add_argument("--preview", action="store_true", help="Guarda video anotado preview_pose.mp4 con los esqueletos (estilo result.plot())")
    parser.add_argument("--dump-poses", action="store_true", help="Guarda poses crudas poses.json estilo [{frame, persons:[{person, keypoints}]}]")
    parser.add_argument("--debug-track", action="store_true", help="Guarda track_debug.jsonl con tids y decisiones de asociacion por frame")
    parser.add_argument("--no-yolo", action="store_true", help="Desactiva YOLO-Pose (solo HOG+MediaPipe)")
    parser.add_argument(
        "--start", type=float, default=0.0, help="Tiempo inicial en segundos"
    )
    parser.add_argument("--end", type=float, default=None, help="Tiempo final en segundos")
    parser.add_argument("--multi", action="store_true", help="Extrae multi-dancer (HOG+NMS per frame hasta 4)")
    parser.add_argument("--multi-max", type=int, default=4, help="Max dancers en modo multi (1-4)")
    parser.add_argument("--num-dancers", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Nº total de dancers del video (1-4). Todos usan el mismo proceso YOLO11m+tracking+FaceID")
    parser.add_argument("--yolo-model", default=None, help="Modelo YOLO ingesta (default yolo11m-pose.pt en multi)")
    parser.add_argument("--face-id", action="store_true", help="FaceID InsightFace obligatorio (multi)")
    parser.add_argument("--face-thresh", type=float, default=0.38, help="Umbral coseno galería FaceID")
    parser.add_argument("--face-check-every", type=int, default=6, help="Re-verificación facial cada N frames")
    parser.add_argument("--face-det-size", type=int, default=320, help="Resolución detector facial sobre crop")
    parser.add_argument("--kp-thresh", type=float, default=0.4, help="Confianza mínima keypoint")
    parser.add_argument("--smooth-alpha", type=float, default=0.6, help="EMA por ID persistente")
    parser.add_argument("--imgsz", type=int, default=960, help="Tamaño inferencia YOLO ingesta")
    parser.add_argument("--poses-jsonl", default=None, help="Ruta export .poses.jsonl estilo trackingdancers")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Ruta del binario ffmpeg")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    video = Path(args.video).resolve()
    if not video.is_file():
        print(f"ERROR: no existe el video {video}", file=sys.stderr)
        return 1

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    options = ExtractOptions(
        video=video,
        title=args.title,
        artist=args.artist,
        fps=args.fps,
        max_width=args.max_width,
        backend=args.backend,
        start_time=args.start,
        end_time=args.end,
        multi=bool(args.multi),
        multi_max=int(args.multi_max),
        use_yolo=not bool(args.no_yolo),
        yolo_conf=float(args.yolo_conf),
        yolo_iou=float(args.yolo_iou),
        estimator=str(args.estimator),
        tracker=str(args.tracker),
        device=str(args.device),
        preview_path=(out_dir / "preview_pose.mp4") if bool(args.preview) else None,
        dump_poses_path=(out_dir / "poses.json") if bool(args.dump_poses) else None,
        debug_track_path=(out_dir / "track_debug.jsonl") if bool(args.debug_track) else None,
        num_dancers=int(args.num_dancers),
        yolo_model=args.yolo_model,
        face_id=bool(args.face_id),
        face_thresh=float(args.face_thresh),
        face_check_every=int(args.face_check_every),
        face_det_size=int(args.face_det_size),
        kp_thresh=float(args.kp_thresh),
        smooth_alpha=float(args.smooth_alpha),
        imgsz=int(args.imgsz),
        poses_jsonl_path=(Path(args.poses_jsonl) if args.poses_jsonl else None),
    )

    t0 = time.perf_counter()
    print(f"[1/4] Extrayendo poses de {video.name} ...")
    print(f"      estimator={options.estimator} tracker={options.tracker} "
          f"yolo_conf={options.yolo_conf} num_dancers={options.num_dancers} device={options.device}")

    def on_progress(i: int, total: int) -> None:
        if total and i % max(1, total // 20) == 0:
            print(f"      {100 * i / total:5.1f}%", flush=True)

    try:
        song = extract(options, on_progress=on_progress)
    except Exception as exc:
        print(f"ERROR al extraer poses: {exc}", file=sys.stderr)
        return 1

    if not song.poses:
        print("ERROR: no se detecto ninguna pose en el video.", file=sys.stderr)
        return 1

    from shared.dance_format.loader import save_song, validate_song

    print(f"[2/4] Guardando song.json ({len(song.poses)} poses)")
    save_song(song, out_dir / "song.json")
    for err in validate_song(song):
        print(f"      aviso: {err}")

    print("[3/4] Copiando video y caratula")
    try:
        shutil.copy2(video, out_dir / "song.mp4")
    except OSError as exc:
        print(f"      aviso: no se pudo copiar el video: {exc}")

    cover_ok = make_cover(video, out_dir / "cover.png")
    if not cover_ok:
        print("      aviso: no se pudo generar la caratula")

    print("[4/4] Extrayendo audio (ffmpeg)")
    try:
        extract_audio(video, out_dir / "song.mp3", ffmpeg=args.ffmpeg)
    except Exception as exc:
        print(f"      aviso: no se pudo extraer el audio: {exc}")

    elapsed = time.perf_counter() - t0
    print(
        f"Listo en {elapsed:.1f}s -> {out_dir}\n"
        f"  '{song.title}' de {song.artist}, {song.duration:.1f}s, "
        f"{len(song.poses)} poses a {song.fps} fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
