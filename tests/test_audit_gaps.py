"""Self-audit gap closures: E1.AC2, F1.AC3, H1.AC3, J1.AC1, I2.AC1."""

import numpy as np
import pytest

from argus.backbone.face import SyntheticFaceBackbone
from argus.backbone.pose import SyntheticPoseBackbone
from argus.bus.outlet import InMemoryBus
from argus.contracts import FrameContext
from argus.capture.frame_source import SyntheticCamera
from argus.core.pipeline import Pipeline
from argus.core.threaded import ThreadedPipeline
from argus.dsp.blink import blink_f1
from argus.dsp.respiration import rppg_derived_rr
from argus.extractors import RppgExtractor
from argus.extractors.respiration_extractor import RespirationExtractor
from argus.validation.report import generate_report


# E1.AC2 — rPPG-derived RR cross-check computed + agreement reported.
def test_rppg_derived_rr_function():
    fps = 30.0
    t = np.arange(int(fps * 30)) / fps
    pulse = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm carrier
    am = 1.0 + 0.5 * np.sin(2 * np.pi * 0.25 * t)  # 15 brpm amplitude modulation
    bvp = pulse * am
    assert abs(rppg_derived_rr(bvp, fps) - 15.0) <= 3.0


def test_respiration_extractor_reports_cross_check():
    brpm = 15.0
    chest = lambda ts: np.sin(2 * np.pi * (brpm / 60.0) * ts)
    pose = SyntheticPoseBackbone(chest_signal=chest)
    fb = SyntheticFaceBackbone()
    ext = RespirationExtractor(fps=30.0, window_s=20.0, min_window_s=15.0, update_period_s=2.0)
    cam = SyntheticCamera(width=32, height=32, n_frames=int(30 * 22))
    last = None
    i = 0
    while True:
        frame, ok = cam.read()
        if not ok:
            break
        ts = (i + 1) / 30.0
        ctx = FrameContext(frame=frame, ts=ts, frame_id=i,
                           face=fb.process(frame, ts), pose=pose.process(frame, ts))
        for r in ext.consume(ctx):
            if r.name == "resp":
                last = r
        i += 1
    assert last is not None
    assert "rppg_rr" in last.meta and "resp_agreement_brpm" in last.meta
    assert np.isfinite(last.meta["rppg_rr"])  # secondary actually computed


# F1.AC3 — blink F1 ≥ 0.90 vs annotation with ±tol event matching.
def test_blink_f1_metric():
    annotated = [1.0, 2.0, 3.0, 4.0, 5.0]
    detected = [1.02, 1.98, 3.05, 4.01, 5.03]  # all within ±0.1 s
    res = blink_f1(detected, annotated, tol_s=0.1)
    assert res["f1"] >= 0.90 and res["tp"] == 5

    # a false positive + a miss lowers F1 below 1.0
    res2 = blink_f1([1.02, 1.98, 7.0], [1.0, 2.0, 3.0], tol_s=0.1)
    assert res2["fp"] == 1 and res2["fn"] == 1 and res2["f1"] < 0.9


# H1.AC3 — low-SQI HR record is emitted WITH a flag (not dropped).
def test_low_sqi_emitted_with_flag():
    # very noisy ROI -> low SNR -> low sqi, but still emitted
    ext = RppgExtractor(fps=30.0, window_s=10.0, min_window_s=8.0, update_period_s=1.0)
    rng = np.random.default_rng(0)
    recs = []
    for i in range(int(30 * 11)):
        frame = (rng.integers(0, 255, (16, 16, 3))).astype(np.uint8)  # pure noise, no pulse
        ts = (i + 1) / 30.0
        face = SyntheticFaceBackbone().process(np.full((4, 4, 3), 30, np.uint8), ts)
        recs.extend(ext.consume(FrameContext(frame=frame, ts=ts, frame_id=i, face=face)))
    hrs = [r for r in recs if r.name == "hr"]
    assert len(hrs) >= 1  # not dropped despite poor signal
    assert "low_sqi" in hrs[-1].meta  # carries the flag


# J1.AC1 — threaded pipeline: capture thread + worker (CV) + asyncio I/O edge.
def test_threaded_pipeline_runs_with_asyncio_edge():
    from argus.contracts import Extractor, SignalRecord

    class PerFrameExt(Extractor):  # emits each frame (windowing is tested elsewhere)
        name = "tick"

        def consume(self, ctx):
            return [SignalRecord("tick", 1.0, 1.0, ctx.ts, gate="good")]

    bus = InMemoryBus()
    pipe = Pipeline(extractors=[PerFrameExt()],
                    face_backbone=SyntheticFaceBackbone(), emit_clock=lambda: 0.0)
    tp = ThreadedPipeline(pipe, SyntheticCamera(width=32, height=32, n_frames=360), fps=30.0)
    tp.run(bus=bus, timeout_s=10.0)
    assert pipe.metrics.frames > 0  # frames processed across threads (CV hot path)
    assert tp.emitted >= 1  # records drained via the asyncio I/O edge
    assert "tick" in bus.stream_names()  # reached the bus through the I/O edge


# I2.AC1 — SNR included in the report.
def test_report_includes_snr():
    rep = generate_report({
        "rest": {"hr_measured": [70.0, 71, 72], "hr_ref": [70.0, 70, 71],
                 "snr_db": [8.0, 9.0, 7.5]},
    })
    assert "mean_snr_db" in rep["conditions"]["rest"]["hr"]
    assert rep["conditions"]["rest"]["hr"]["mean_snr_db"] == pytest.approx(8.1667, abs=0.01)
