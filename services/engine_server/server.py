"""Optional network layer for external clients (CLI, web, Godot, mobile).

Exposes the same engine (pose + score) over WebSocket. Not the path used
by the desktop app, which consumes the engine directly in-process.

Setup:
    uv sync --extra server

Usage:
    uv run engine-server --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine-server",
        description="Servidor WebSocket opcional del motor de pose.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--rate", type=float, default=15.0, help="FPS de envio de poses")
    return parser


def _require_aiohttp() -> None:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        raise SystemExit("aiohttp no esta instalado. Ejecuta: uv sync --extra server")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _require_aiohttp()

    from aiohttp import web

    from engine.normalize_engine import BodyNormalizer
    from engine.pose_engine import Camera, create_backend

    lock = threading.Lock()
    shared = {"pose": None, "fps": 0.0, "backend": None, "error": None}

    def camera_loop() -> None:
        backend = create_backend("auto")
        with lock:
            shared["backend"] = backend.name
        camera = Camera(args.camera)
        if not camera.open():
            with lock:
                shared["error"] = camera.last_error or "no camera"
            return
        normalizer = BodyNormalizer()
        fps = 0.0
        while True:
            t0 = time.perf_counter()
            frame = camera.read()
            if frame is None:
                time.sleep(0.03)
                continue
            raw = backend.detect(frame)
            normalized = normalizer.normalize(raw)
            with lock:
                shared["pose"] = normalized.to_dict() if not normalized.is_empty() else None
            dt = time.perf_counter() - t0
            fps = fps * 0.9 + (1.0 / dt) * 0.1 if dt else fps
            with lock:
                shared["fps"] = fps
            time.sleep(max(0.0, 1.0 / args.rate - dt))

    threading.Thread(target=camera_loop, daemon=True).start()

    async def info_handler(_request):
        with lock:
            return web.json_response(
                {
                    "service": "openjustdance-engine",
                    "backend": shared["backend"],
                    "camera_index": args.camera,
                }
            )

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            while not ws.closed:
                with lock:
                    pose = shared["pose"]
                    fps = shared["fps"]
                    backend = shared["backend"]
                await ws.send_str(
                    json.dumps(
                        {
                            "pose": pose,
                            "fps": fps,
                            "backend": backend,
                            "time": time.time(),
                        }
                    )
                )
                await asyncio.sleep(1.0 / args.rate)
        finally:
            await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", info_handler)
    app.router.add_get("/ws", ws_handler)
    print(f"Engine server en ws://{args.host}:{args.port}/ws")
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
