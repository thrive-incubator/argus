#!/usr/bin/env python
"""Run Argus live on a real webcam — ALL signals the environment can support.

Everything physiological comes from the CAMERA (rPPG/pose/face). No contact sensor is
needed; a Polar H10 is optional and only a ground-truth REFERENCE (--validate-against-polar).

Each live signal needs its own real input backend. This script auto-detects what is
available and activates every signal it can, printing exactly what is OFF and why:

  HR, HRV            -> camera + OpenCV face box           (works with just opencv)
  Respiration,Fidget -> MediaPipe Pose                     (--pose-model, needs mediapipe)
  Blink/PERCLOS      -> MediaPipe face mesh                (--face-model, needs mediapipe)
  Gaze zones         -> L2CS ONNX model                    (--gaze-model, needs onnxruntime)
  Affect (V/A)       -> HSEmotion ONNX model               (--affect-model)
  Action Units       -> LibreFace ONNX model               (--au-model)

Usage:
    PYTHONPATH=src ./venv/bin/python scripts/run_live.py
    PYTHONPATH=src ./venv/bin/python scripts/run_live.py --face-model models/face_landmarker.task \
        --pose-model models/pose_landmarker.task --affect-model models/hsemotion.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.bus.osc import OscBridge, UdpTransport
from argus.bus.outlet import InMemoryBus
from argus.capture.clock import local_clock
from argus.capture.frame_source import OpenCVCamera
from argus.core.pipeline import Pipeline
from argus.dashboard.render import Dashboard
from argus.extractors import (
    BlinkExtractor,
    HrvExtractor,
    MotionExtractor,
    RespirationExtractor,
    RppgExtractor,
)


def _try(fn, label, off):
    try:
        return fn()
    except Exception as e:  # importable-but-misconfigured, missing model, etc.
        off.append(f"{label}: {type(e).__name__}: {e}")
        return None


def build(args):
    active, off = [], []

    # --- Face backbone: MediaPipe mesh if a model is given, else OpenCV cascade ---
    face_bb = None
    if args.face_model:
        def mk_face():
            from argus.backbone.face import MediaPipeFaceBackbone
            return MediaPipeFaceBackbone(args.face_model)
        face_bb = _try(mk_face, "MediaPipe face mesh", off)
    face_kind = "mediapipe" if face_bb else "cascade"
    if face_bb is None:
        from argus.backbone.cascade import CascadeFaceBackbone
        face_bb = CascadeFaceBackbone()

    # --- Pose backbone (respiration, fidget) ---
    pose_bb = None
    if args.pose_model:
        def mk_pose():
            from argus.backbone.pose import MediaPipePoseBackbone
            return MediaPipePoseBackbone(args.pose_model)
        pose_bb = _try(mk_pose, "MediaPipe pose", off)
    else:
        off.append("Respiration/Fidget: need --pose-model (MediaPipe Pose)")

    extractors = []
    # HR + HRV always (camera rPPG)
    extractors.append(RppgExtractor(fps=float(args.fps), window_s=args.window,
                                    min_window_s=max(8.0, args.window - 2)))
    extractors.append(HrvExtractor(fps=float(args.fps), window_s=args.hrv_window,
                                   min_window_s=min(20.0, args.hrv_window)))
    active += ["hr", "hrv"]

    if pose_bb is not None:
        extractors += [RespirationExtractor(fps=float(args.fps)),
                       MotionExtractor(fps=float(args.fps))]
        active += ["resp", "fidget"]

    # Blink needs the full face mesh (EAR eye points); cascade box is insufficient
    if face_kind == "mediapipe":
        extractors.append(BlinkExtractor(fps=float(args.fps)))
        active.append("blink")
    else:
        off.append("Blink/PERCLOS: need --face-model (cascade box lacks eye landmarks)")

    # ONNX model-backed signals
    if args.gaze_model:
        def mk_gaze():
            from argus.perception.gaze import GazeExtractor, L2csGazeEstimator
            return GazeExtractor(L2csGazeEstimator(args.gaze_model), fps=float(args.fps))
        g = _try(mk_gaze, "Gaze (L2CS ONNX)", off)
        if g:
            extractors.append(g); active.append("gaze_zone")
    else:
        off.append("Gaze: need --gaze-model (L2CS ONNX) + onnxruntime")

    if args.affect_model:
        def mk_affect():
            from argus.perception.affect import AffectExtractor, HSEmotionEstimator
            return AffectExtractor(HSEmotionEstimator(args.affect_model), fps=float(args.fps))
        a = _try(mk_affect, "Affect (HSEmotion ONNX)", off)
        if a:
            extractors.append(a); active += ["affect_valence", "affect_arousal"]
    else:
        off.append("Affect: need --affect-model (HSEmotion ONNX) + onnxruntime")

    if args.au_model:
        def mk_au():
            from argus.perception.au import AuExtractor, LibreFaceAuEstimator
            return AuExtractor(LibreFaceAuEstimator(args.au_model))
        au = _try(mk_au, "Action Units (LibreFace ONNX)", off)
        if au:
            extractors.append(au); active.append("au_*")
    else:
        off.append("Action Units: need --au-model (LibreFace ONNX) + onnxruntime")

    return face_bb, face_kind, pose_bb, extractors, active, off


def main() -> int:
    ap = argparse.ArgumentParser(description="Argus live (camera-only) — all available signals")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--window", type=float, default=10.0, help="rPPG HR window s")
    ap.add_argument("--hrv-window", type=float, default=30.0, help="HRV window s")
    ap.add_argument("--face-model", default=None, help="MediaPipe face_landmarker.task")
    ap.add_argument("--pose-model", default=None, help="MediaPipe pose_landmarker.task")
    ap.add_argument("--gaze-model", default=None, help="L2CS ONNX")
    ap.add_argument("--affect-model", default=None, help="HSEmotion ONNX")
    ap.add_argument("--au-model", default=None, help="LibreFace ONNX")
    ap.add_argument("--osc", default=None, help="stream to OSC host:port")
    ap.add_argument("--validate-against-polar", action="store_true")
    ap.add_argument("--polar-name", default="Polar H10")
    args = ap.parse_args()

    face_bb, face_kind, pose_bb, extractors, active, off = build(args)

    print("=== Argus live ===")
    print(f"face backbone: {face_kind}")
    print(f"ACTIVE signals ({len(active)}): {', '.join(active)}")
    print("OFF (and why):")
    for line in off:
        print(f"  - {line}")
    print("Look at the camera; HR needs ~10 s to lock, HRV ~30 s.\n")

    cam = OpenCVCamera(args.camera_index, args.width, args.height, args.fps)
    bus = InMemoryBus()
    dash = Dashboard(phase=1)
    pipe = Pipeline(extractors=extractors, face_backbone=face_bb, pose_backbone=pose_bb,
                    bus=bus, dashboard=dash, emit_clock=local_clock)

    osc = OscBridge(UdpTransport(*([args.osc.split(":")[0], int(args.osc.split(":")[1])]))) \
        if args.osc else None

    polar = None
    if args.validate_against_polar:
        from argus.groundtruth.ble import PolarThread
        polar = PolarThread(name_prefix=args.polar_name, clock=local_clock)
        polar.start()

    t0 = local_clock()
    last = 0.0
    frames = 0
    try:
        while local_clock() - t0 < args.duration:
            ok_frame, ok = cam.read()
            if not ok:
                print("camera read failed"); break
            for r in pipe.process_frame(ok_frame, local_clock(), frames):
                if osc is not None:
                    osc.publish(r)
            frames += 1
            now = time.time()
            if now - last >= 0.5:
                last = now
                view = dash.render(now=local_clock())
                cells = []
                for name in active:
                    v = view.get(name if not name.endswith("*") else None)
                    if v is None:
                        cells.append(f"{name}=…")
                    else:
                        val = f"{v.value:.1f}" if isinstance(v.value, (int, float)) else v.value
                        cells.append(f"{name}={val}{v.light}")
                line = f"\r[{local_clock()-t0:5.1f}s] " + "  ".join(cells)
                if polar is not None:
                    line += f"  | Polar(ref)={polar.latest_hr or polar.status}"
                sys.stdout.write(line + "    "); sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cam.release()
        if polar is not None:
            polar.stop()
        print(f"\n\nframes={frames}  emits={pipe.metrics.emits}")
        for name, v in dash.render(now=local_clock()).items():
            val = f"{v.value:.1f}" if isinstance(v.value, (int, float)) else v.value
            print(f"  {name:16} = {val}  (quality {v.sqi:.2f}, {v.status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
