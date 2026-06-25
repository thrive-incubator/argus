#!/usr/bin/env python
"""Argus web backend: runs the live camera pipeline and streams every signal to a browser
over WebSocket, and serves the dashboard page over HTTP.

    PYTHONPATH=src ./venv/bin/python scripts/web_server.py
    # then open http://localhost:8000

Flags:
    --synthetic     use a synthetic camera (no webcam needed; for demo/testing)
    --http-port     static page port (default 8000)
    --ws-port       websocket port (default 8765)
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argus.bus.ws import WebSocketBridge
from argus.capture.clock import local_clock
from argus.core.pipeline import Pipeline
from argus.extractors import (
    BlinkExtractor,
    HrvExtractor,
    MotionExtractor,
    RespirationExtractor,
    RppgExtractor,
)
from argus.extractors.wave import BreathWaveExtractor, PulseWaveExtractor


class Broadcaster:
    """Fans JSON messages from the (threaded) pipeline out to all WS clients."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.clients: set = set()

    def publish_threadsafe(self, message: str) -> None:
        self.loop.call_soon_threadsafe(self._fanout, message)

    def _fanout(self, message: str) -> None:
        for ws in list(self.clients):
            self.loop.create_task(self._safe_send(ws, message))

    async def _safe_send(self, ws, message: str) -> None:
        try:
            await ws.send(message)
        except Exception:
            self.clients.discard(ws)


def build_pipeline(synthetic: bool, bridge: WebSocketBridge):
    fps = 30.0
    exts = [
        RppgExtractor(fps=fps),
        HrvExtractor(fps=fps, window_s=20.0, min_window_s=15.0),
        RespirationExtractor(fps=fps, window_s=20.0, min_window_s=15.0),
        MotionExtractor(fps=fps),
        BlinkExtractor(fps=fps, baseline_frames=60),
        PulseWaveExtractor(),
        BreathWaveExtractor(),
    ]
    if synthetic:
        from argus.backbone.face import SyntheticFaceBackbone
        from argus.backbone.pose import SyntheticPoseBackbone
        from argus.capture.frame_source import SyntheticCamera

        face = SyntheticFaceBackbone()

        class _Pose(SyntheticPoseBackbone):
            def __init__(self):
                import numpy as np
                super().__init__(chest_signal=lambda ts: np.sin(2 * 3.14159 * 0.25 * ts))

        cam = SyntheticCamera(width=320, height=240, fps=fps, hr_bpm=72.0)
        face_bb, pose_bb = face, _Pose()
    else:
        models = ROOT / "models"
        from argus.backbone.face import MediaPipeFaceBackbone
        from argus.backbone.pose import MediaPipePoseBackbone
        from argus.capture.frame_source import OpenCVCamera

        face_bb = MediaPipeFaceBackbone(str(models / "face_landmarker.task"))
        pose_bb = MediaPipePoseBackbone(str(models / "pose_landmarker.task"))
        cam = OpenCVCamera(0, 1280, 720, int(fps))

    pipe = Pipeline(extractors=exts, face_backbone=face_bb, pose_backbone=pose_bb,
                    emit_clock=local_clock)
    return pipe, cam, fps


def pipeline_thread(synthetic: bool, broadcaster: Broadcaster, stop: threading.Event):
    bridge = WebSocketBridge(transport=type("T", (), {"send": lambda self, t: None})())
    pipe, cam, fps = build_pipeline(synthetic, bridge)
    i = 0
    while not stop.is_set():
        frame, ok = cam.read()
        if not ok:
            break
        for r in pipe.process_frame(frame, local_clock(), i):
            broadcaster.publish_threadsafe(WebSocketBridge.to_json(r))
        i += 1
        if synthetic:  # pace the synthetic camera to ~real time
            stop.wait(1.0 / fps)
    cam.release()


def serve_http(port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "web"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def amain(args):
    broadcaster_holder = {}

    async def handler(ws):
        broadcaster_holder["b"].clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            broadcaster_holder["b"].clients.discard(ws)

    import websockets

    loop = asyncio.get_running_loop()
    broadcaster_holder["b"] = Broadcaster(loop)
    stop = threading.Event()

    serve_http(args.http_port)
    print(f"Dashboard:  http://localhost:{args.http_port}")
    print(f"WebSocket:  ws://localhost:{args.ws_port}")
    print("camera:", "SYNTHETIC" if args.synthetic else "REAL webcam (MediaPipe)")

    threading.Thread(target=pipeline_thread,
                     args=(args.synthetic, broadcaster_holder["b"], stop), daemon=True).start()

    async with websockets.serve(handler, "127.0.0.1", args.ws_port):
        try:
            await asyncio.Future()  # run forever
        finally:
            stop.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--ws-port", type=int, default=8765)
    args = ap.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
