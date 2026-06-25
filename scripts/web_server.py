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
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _quiet_native_logs(logname: str):
    """Silence noisy native libraries (MediaPipe glog, TF-Lite, HF, objc dylib warnings).

    Log levels are set BEFORE the heavy imports; the C-level stderr (fd 2) — where all that
    noise is written — is redirected to a log file, so the console shows only our own stdout.
    Pass --verbose to disable. Real errors still go to the log file.
    """
    if "--verbose" in sys.argv:
        return None
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import warnings

    warnings.filterwarnings("ignore")
    (ROOT / "logs").mkdir(exist_ok=True)
    logf = open(ROOT / "logs" / logname, "a", buffering=1)
    os.dup2(logf.fileno(), 2)  # native stderr noise -> log file (console stays clean)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    return logf


_LOGF = _quiet_native_logs("argus_web.log")
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
        cam = OpenCVCamera(0, 1280, 720, int(fps), mirror=True)  # selfie view (egocentric)

    pipe = Pipeline(extractors=exts, face_backbone=face_bb, pose_backbone=pose_bb,
                    emit_clock=local_clock)
    return pipe, cam, fps


def add_fast_perception(pipe, synthetic):
    """Add affect (HSEmotion) + gaze (iris) — both fast enough for the per-frame loop."""
    if synthetic:
        from argus.perception.affect import AffectEstimate, AffectExtractor, FakeEmotionEstimator
        import math
        # a gently varying fake affect so the demo dashboard moves
        pipe.add_extractor(AffectExtractor(FakeEmotionEstimator(
            AffectEstimate("happiness", 0.6, 0.3, 0.9)), live_hz=10.0))
        return
    try:
        from argus.perception.affect import AffectExtractor, HSEmotionEstimator
        pipe.add_extractor(AffectExtractor(HSEmotionEstimator(), live_hz=12.0))
        print("  affect: ON (HSEmotion)")
    except Exception as e:
        print(f"  affect: off ({e})")
    try:
        from argus.perception.gaze import GazeFeatureExtractor, IrisGazeExtractor
        pipe.add_extractor(IrisGazeExtractor(hz=10.0))
        pipe.add_extractor(GazeFeatureExtractor(hz=20.0))  # for /gaze.html calibration
        print("  gaze: ON (MediaPipe iris + screen-gaze features)")
    except Exception as e:
        print(f"  gaze: off ({e})")


class AuWorker(threading.Thread):
    """Runs py-feat AUs in the background (~1 Hz) on the latest frame and broadcasts them."""

    KEY_AUS = ("AU01", "AU04", "AU06", "AU07", "AU12", "AU15", "AU25")

    def __init__(self, latest, broadcaster, stop, synthetic):
        super().__init__(daemon=True)
        self.latest, self.broadcaster, self.stop, self.synthetic = latest, broadcaster, stop, synthetic
        self.estimator = None

    def run(self):
        if self.synthetic:
            return
        try:
            from argus.perception.au import PyFeatAuEstimator
            self.estimator = PyFeatAuEstimator()
            print("  action units: ON (py-feat, ~1 Hz background)")
        except Exception as e:
            print(f"  action units: off ({e})")
            return
        from argus.contracts import SignalRecord
        while not self.stop.is_set():
            frame = self.latest.get("frame")
            if frame is not None:
                try:
                    aus = self.estimator.estimate(frame)
                    for k in self.KEY_AUS:
                        if k in aus:
                            rec = SignalRecord(f"au_{k}", aus[k], 1.0, local_clock(),
                                               gate="unknown", meta={"research": True})
                            self.broadcaster.publish_threadsafe(WebSocketBridge.to_json(rec))
                except Exception:
                    pass
            self.stop.wait(1.0)


def pipeline_thread(synthetic: bool, broadcaster: Broadcaster, stop: threading.Event, state: dict):
    from collections import deque

    from argus.viz.overlay import draw_debug, encode_jpeg_b64

    pipe, cam, fps = build_pipeline(synthetic, None)
    add_fast_perception(pipe, synthetic)
    latest = {"frame": None}
    AuWorker(latest, broadcaster, stop, synthetic).start()
    hud: dict = {}
    pulse_hist: deque = deque(maxlen=120)
    breath_hist: deque = deque(maxlen=120)
    cam_every = 3  # broadcast the camera frame at ~fps/3 (~10 Hz) to bound bandwidth
    i = 0
    while not stop.is_set():
        frame, ok = cam.read()
        if not ok:
            break
        latest["frame"] = frame
        for r in pipe.process_frame(frame, local_clock(), i):
            broadcaster.publish_threadsafe(WebSocketBridge.to_json(r))
            if r.name == "hr":
                hud["hr"] = r.value
                hud["hr_sqi"] = r.sqi
            elif r.name == "resp":
                hud["resp"] = r.value
            elif r.name == "pulse_wave":
                pulse_hist.append(r.value)
            elif r.name == "breath_wave":
                breath_hist.append(r.value)
            elif r.name == "affect_valence":
                hud["emotion"] = r.meta.get("emotion")
            elif r.name == "gaze_zone":
                hud["gaze"] = (r.meta.get("zone") or {}).get("horizontal")
        if i % cam_every == 0 and pipe.last_ctx is not None:
            if state.get("overlay", True):
                img = draw_debug(frame, pipe.last_ctx, hud,
                                 {"pulse": list(pulse_hist), "breath": list(breath_hist)})
            else:
                img = frame  # raw feed, overlays off
            b64 = encode_jpeg_b64(img)
            if b64:
                broadcaster.publish_threadsafe(json.dumps({"name": "camera", "value": b64}))
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
    state = {"overlay": True}  # camera debug overlays on/off (toggled from the browser)

    async def handler(ws):
        broadcaster_holder["b"].clients.add(ws)
        try:
            async for raw in ws:  # accept commands from the browser (e.g. overlay toggle)
                try:
                    msg = json.loads(raw)
                    if msg.get("cmd") == "overlay":
                        state["overlay"] = bool(msg.get("on", True))
                except Exception:
                    pass
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
                     args=(args.synthetic, broadcaster_holder["b"], stop, state), daemon=True).start()

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
