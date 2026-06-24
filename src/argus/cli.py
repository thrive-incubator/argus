"""``argus`` CLI (J3, NFR-4): run | record | report | fetch-models.

Uses synthetic sources by default so every subcommand runs headlessly; the real device
sources are swapped in via the same interfaces on hardware.
"""

from __future__ import annotations

import argparse

import numpy as np

from .backbone.face import SyntheticFaceBackbone
from .backbone.pose import SyntheticPoseBackbone
from .bus.outlet import InMemoryBus, StreamSpec
from .bus.recorder import Recorder
from .capture.frame_source import SyntheticCamera
from .core.models import MODEL_MANIFEST, fetch_models
from .core.pipeline import Pipeline
from .dashboard.render import Dashboard
from .extractors import RppgExtractor
from .validation.report import generate_report, write_report_html


def _build_pipeline(bus=None, recorder=None, dashboard=None) -> tuple[Pipeline, object]:
    fb = SyntheticFaceBackbone()
    ext = RppgExtractor(fps=30.0)
    pipe = Pipeline(extractors=[ext], face_backbone=fb, bus=bus, recorder=recorder,
                    dashboard=dashboard, emit_clock=lambda: 0.0)
    return pipe, fb


def cmd_run(args) -> int:
    bus = InMemoryBus()
    dash = Dashboard(phase=0)
    pipe, _ = _build_pipeline(bus=bus, dashboard=dash)
    pipe.run_source(SyntheticCamera(width=64, height=64, n_frames=args.frames), max_frames=args.frames)
    print(f"processed {pipe.metrics.frames} frames, emitted {pipe.metrics.emits} records")
    print(dash.to_text(now=args.frames / 30.0) or "(no signals yet)")
    return 0


def cmd_record(args) -> int:
    rec = Recorder()
    # also record a synthetic Polar reference stream alongside (B2.AC1)
    rec.declare(StreamSpec("polar_hr", channel_count=3, nominal_srate=1.0, unit="bpm"))
    pipe, _ = _build_pipeline(recorder=rec)
    pipe.run_source(SyntheticCamera(width=64, height=64, n_frames=args.frames), max_frames=args.frames)
    from .contracts import SignalRecord

    for i in range(5):
        rec.record(SignalRecord("polar_hr", 72.0 + 0.1 * i, 1.0, float(i), gate="good"))
    out = args.out or f"{args.session}.xdf"
    rec.write(out)
    print(f"wrote {out} with streams {rec.stream_names()}")
    return 0


def cmd_report(args) -> int:
    import pyxdf

    streams, _ = pyxdf.load_xdf(args.xdf)
    by_name = {s["info"]["name"][0]: s for s in streams}
    data = {}
    if "hr" in by_name and "polar_hr" in by_name:
        hr = by_name["hr"]["time_series"][:, 0]
        ref = by_name["polar_hr"]["time_series"][:, 0]
        n = min(len(hr), len(ref))
        if n >= 1:
            data["session"] = {"hr_measured": np.asarray(hr[:n]), "hr_ref": np.asarray(ref[:n])}
    report = generate_report(data or {"session": {}})
    write_report_html(report, args.out)
    print(f"wrote report {args.out}")
    return 0


def cmd_fetch_models(args) -> int:
    if args.dry_run:
        for a in MODEL_MANIFEST:
            print(f"{a.name}: {a.filename} ({a.license})")
        return 0
    paths = fetch_models(args.dest)  # pragma: no cover - network
    print(f"fetched {len(paths)} models")  # pragma: no cover
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="argus")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="launch live pipeline + dashboard + bus")
    pr.add_argument("--frames", type=int, default=360)
    pr.set_defaults(func=cmd_run)

    prec = sub.add_parser("record", help="run a validation session to a synchronized XDF")
    prec.add_argument("--session", required=True)
    prec.add_argument("--out", default=None)
    prec.add_argument("--frames", type=int, default=360)
    prec.set_defaults(func=cmd_record)

    prep = sub.add_parser("report", help="produce the validation report from an XDF")
    prep.add_argument("--xdf", required=True)
    prep.add_argument("--out", default="argus_report.html")
    prep.set_defaults(func=cmd_report)

    pf = sub.add_parser("fetch-models", help="download + verify model assets")
    pf.add_argument("--dest", default="models")
    pf.add_argument("--dry-run", action="store_true")
    pf.set_defaults(func=cmd_fetch_models)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
