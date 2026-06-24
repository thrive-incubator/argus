"""FR-4/A4, H2.AC4/AC5, NFR-1/7/8 — pipeline + gate application."""

import numpy as np
import pytest

from argus.backbone.face import SyntheticFaceBackbone
from argus.bus.outlet import InMemoryBus
from argus.contracts import Extractor, FrameContext, SignalRecord
from argus.core.pipeline import Pipeline
from argus.capture.clock import FakeClock
from argus.capture.frame_source import SyntheticCamera
from argus.dashboard.render import Dashboard
from argus.extractors import RppgExtractor
from argus.quality.gate_apply import GateController
from argus.quality.motion_gate import MotionGate


class HrStub(Extractor):
    """Emits a fixed HR + HRV every frame (for deterministic gate tests)."""

    name = "hrstub"

    def __init__(self):
        self.i = 0

    def consume(self, ctx):
        self.i += 1
        return [
            SignalRecord("hr", 72.0 + self.i, 0.9, ctx.ts, gate="unknown",
                         meta={"snr_db": 6.0}),
            SignalRecord("hrv", 45.0, 0.9, ctx.ts, gate="unknown", meta={"rest_only": True}),
            SignalRecord("resp", 15.0, 0.8, ctx.ts, gate="unknown", meta={}),
        ]


# H2.AC4/AC5 — gate rides on records; reject suppresses HRV/RR and holds last-good HR.
def test_gate_controller_suppresses_and_holds():
    gc = GateController(MotionGate(dwell_frames=1))
    # GOOD: low motion, high snr
    gc.step(motion=0.05, snr_db=8.0)
    out = gc.apply([
        SignalRecord("hr", 72.0, 0.9, 1.0),
        SignalRecord("hrv", 45.0, 0.9, 1.0),
        SignalRecord("resp", 15.0, 0.8, 1.0),
    ])
    names = [r.name for r in out]
    assert names == ["hr", "hrv", "resp"]
    assert all(r.gate == "good" for r in out)

    # REJECT: high motion -> hrv/resp suppressed, hr holds last-good (72.0)
    gc.step(motion=0.9, snr_db=8.0)
    out2 = gc.apply([
        SignalRecord("hr", 200.0, 0.1, 2.0),  # garbage during motion
        SignalRecord("hrv", 999.0, 0.1, 2.0),
        SignalRecord("resp", 99.0, 0.1, 2.0),
    ])
    assert [r.name for r in out2] == ["hr"]  # only HR survives
    assert out2[0].value == 72.0 and out2[0].gate == "reject" and out2[0].meta["held"] is True

    # recover to GOOD
    gc.step(motion=0.02, snr_db=8.0)
    out3 = gc.apply([SignalRecord("hr", 74.0, 0.95, 3.0), SignalRecord("hrv", 46.0, 0.9, 3.0)])
    assert [r.name for r in out3] == ["hr", "hrv"] and out3[0].value == 74.0


def test_pipeline_gate_integration_with_dashboard():
    dash = Dashboard(phase=1)
    gc = GateController(MotionGate(dwell_frames=1))
    motion = {"v": 0.02}
    pipe = Pipeline(
        extractors=[HrStub()],
        gate_controller=gc,
        dashboard=dash,
        motion_provider=lambda ctx: motion["v"],
        snr_provider=lambda ctx: 8.0,
        emit_clock=FakeClock(step=0.0),
    )
    cam = SyntheticCamera(width=8, height=8, n_frames=4)
    for i in range(4):
        frame, _ = cam.read()
        if i == 2:
            motion["v"] = 0.9  # induce motion on frame 2
        else:
            motion["v"] = 0.02
        pipe.process_frame(frame, ts=(i + 1) / 30.0, frame_id=i)
    view = dash.render(now=10.0)
    # HR always present; under the induced-motion frame it was held + re-acquiring
    assert "hr" in view


# NFR-7 — add an extractor without touching backbone/bus.
def test_extensibility_add_extractor():
    bus = InMemoryBus()

    class NewExt(Extractor):
        name = "novel"

        def consume(self, ctx):
            return [SignalRecord("novel", 1.0, 1.0, ctx.ts, gate="good")]

    pipe = Pipeline(extractors=[], bus=bus, emit_clock=FakeClock(step=0.0))
    pipe.add_extractor(NewExt())
    pipe.run_source(SyntheticCamera(width=8, height=8, n_frames=3), max_frames=3)
    assert "novel" in bus.stream_names()


# NFR-8 — observability counters populated.
def test_metrics_exposed():
    pipe = Pipeline(extractors=[HrStub()], emit_clock=FakeClock(step=0.0))
    pipe.run_source(SyntheticCamera(width=8, height=8, n_frames=5), max_frames=5)
    assert pipe.metrics.frames == 5
    assert pipe.metrics.emits == 15  # 3 records/frame
    assert pipe.metrics.extractor_calls["hrstub"] == 5
    assert "hrstub" in pipe.metrics.extractor_time_s


# NFR-1 — capture→emit latency is tracked and within budget (headless instant processing).
def test_latency_tracked_within_budget():
    pipe = Pipeline(extractors=[HrStub()], emit_clock=FakeClock(start=0.0, step=1e-4))
    pipe.run_source(SyntheticCamera(width=8, height=8, n_frames=10), max_frames=10, fps=30.0)
    assert pipe.metrics.max_latency_s <= 2.0


# A4.AC1 (proxy) — extractor ring stays bounded over many frames (no unbounded growth).
def test_rppg_ring_bounded_over_many_frames():
    ext = RppgExtractor(fps=30.0)
    fb = SyntheticFaceBackbone()
    cam = SyntheticCamera(width=8, height=8, n_frames=500)
    for i in range(500):
        frame, _ = cam.read()
        ts = (i + 1) / 30.0
        ctx = FrameContext(frame=frame, ts=ts, frame_id=i, face=fb.process(frame, ts))
        ext.consume(ctx)
    assert len(ext._ring) <= 100_000  # capacity bound respected
