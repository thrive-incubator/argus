#!/usr/bin/env python
"""Run Argus live on a real webcam — CAMERA ONLY.

Heart rate (and signal quality) are derived entirely from the camera via rPPG/POS. No
contact sensor is involved. A Polar H10 is OPTIONAL and only used as a ground-truth
*reference* for accuracy comparison (``--validate-against-polar``) — it is never required
for the camera to work.

Usage:
    PYTHONPATH=src ./venv/bin/python scripts/run_live.py
    PYTHONPATH=src ./venv/bin/python scripts/run_live.py --duration 120 --osc 127.0.0.1:7000
    PYTHONPATH=src ./venv/bin/python scripts/run_live.py --validate-against-polar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow running as `python scripts/run_live.py` without PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.backbone.cascade import CascadeFaceBackbone
from argus.bus.osc import OscBridge, UdpTransport
from argus.bus.outlet import InMemoryBus
from argus.capture.clock import local_clock
from argus.capture.frame_source import OpenCVCamera
from argus.core.pipeline import Pipeline
from argus.dashboard.render import Dashboard
from argus.extractors import RppgExtractor


def main() -> int:
    ap = argparse.ArgumentParser(description="Argus live (camera-only) heart rate")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    ap.add_argument("--window", type=float, default=10.0, help="rPPG window seconds (8-15)")
    ap.add_argument("--osc", default=None, help="stream HR to OSC, e.g. 127.0.0.1:7000")
    ap.add_argument("--validate-against-polar", action="store_true",
                    help="ALSO read a Polar H10 as a ground-truth reference (optional)")
    ap.add_argument("--polar-name", default="Polar H10")
    args = ap.parse_args()

    print("Opening camera... (look at the camera, hold reasonably still, good lighting)")
    cam = OpenCVCamera(args.camera_index, args.width, args.height, args.fps)
    face = CascadeFaceBackbone()
    ext = RppgExtractor(fps=float(args.fps), window_s=args.window,
                        min_window_s=max(8.0, args.window - 2), update_period_s=1.0)
    bus = InMemoryBus()
    dash = Dashboard(phase=0)
    pipe = Pipeline(extractors=[ext], face_backbone=face, bus=bus, dashboard=dash,
                    emit_clock=local_clock)

    osc = None
    if args.osc:
        host, port = args.osc.split(":")
        osc = OscBridge(UdpTransport(host, int(port)))
        print(f"Streaming HR to OSC {args.osc}")

    polar = None
    if args.validate_against_polar:
        from argus.groundtruth.ble import PolarThread

        polar = PolarThread(name_prefix=args.polar_name, clock=local_clock)
        polar.start()
        print(f"Connecting to '{args.polar_name}' as a ground-truth REFERENCE (optional)...")

    t0 = local_clock()
    last_print = 0.0
    frames = 0
    face_seen = 0
    try:
        while local_clock() - t0 < args.duration:
            frame, ok = cam.read()
            if not ok:
                print("camera read failed"); break
            ts = local_clock()
            records = pipe.process_frame(frame, ts, frames)
            frames += 1
            if pipe.face_backbone and ext._ring and len(ext._ring):
                face_seen += 1
            for r in records:
                if osc is not None and r.name == "hr":
                    osc.publish(r)

            now = time.time()
            if now - last_print >= 0.5:  # refresh ~2 Hz
                last_print = now
                elapsed = local_clock() - t0
                view = dash.render(now=local_clock()).get("hr")
                hr_txt = "acquiring..." if view is None else (
                    f"{view.value:.1f} bpm  q={view.sqi:.2f} {view.light}"
                )
                line = (f"\r[{elapsed:5.1f}s] frames={frames} "
                        f"face={'yes' if records or len(ext._ring) else '...'}  "
                        f"CAMERA HR: {hr_txt}")
                if polar is not None:
                    ref = polar.latest_hr
                    line += f"   | Polar(ref): {ref if ref is not None else polar.status}"
                sys.stdout.write(line + "   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        cam.release()
        if polar is not None:
            polar.stop()
        print("\n--- summary ---")
        print(f"frames processed: {frames}")
        print(f"HR updates emitted: {pipe.metrics.emits}")
        final = dash.render(now=local_clock()).get("hr")
        if final is not None:
            print(f"last camera HR: {final.value:.1f} bpm (quality {final.sqi:.2f})")
        else:
            print("no HR locked — try better/steadier lighting and hold still ~15 s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
