"""Pipeline orchestrator (FR-4, A4, J1, NFR-1/2/7/8).

Wires one backbone pass per frame → decoupled extractors → gate → bus/recorder/dashboard.
Exposes observability counters and capture→emit latency. ``process_frame`` is the unit of
work; ``run_source`` drives a frame source synchronously (the threaded variant uses the same
``process_frame`` on a capture thread — see ``argus.capture.capture_thread``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..contracts import Extractor, FrameContext, SignalRecord


@dataclass
class Metrics:
    frames: int = 0
    emits: int = 0
    dropped_frames: int = 0
    extractor_time_s: dict[str, float] = field(default_factory=dict)
    extractor_calls: dict[str, int] = field(default_factory=dict)
    max_latency_s: float = 0.0

    def record_call(self, name: str, dt: float) -> None:
        self.extractor_time_s[name] = self.extractor_time_s.get(name, 0.0) + dt
        self.extractor_calls[name] = self.extractor_calls.get(name, 0) + 1


class Pipeline:
    def __init__(self, extractors: list[Extractor], face_backbone=None, pose_backbone=None,
                 gate_controller=None, bus=None, recorder=None, dashboard=None,
                 motion_provider=None, snr_provider=None, emit_clock=time.monotonic):
        self.extractors = list(extractors)
        self.face_backbone = face_backbone
        self.pose_backbone = pose_backbone
        self.gate_controller = gate_controller
        self.bus = bus
        self.recorder = recorder
        self.dashboard = dashboard
        self.motion_provider = motion_provider  # ctx -> motion magnitude
        self.snr_provider = snr_provider  # ctx -> snr dB (else from hr meta)
        self.emit_clock = emit_clock
        self.metrics = Metrics()

    def add_extractor(self, extractor: Extractor) -> None:
        """NFR-7 — add an extractor without touching the backbone or bus."""
        self.extractors.append(extractor)

    def process_frame(self, frame, ts: float, frame_id: int) -> list[SignalRecord]:
        face = self.face_backbone.process(frame, ts) if self.face_backbone else None
        pose = self.pose_backbone.process(frame, ts) if self.pose_backbone else None
        ctx = FrameContext(frame=frame, ts=ts, frame_id=frame_id, face=face, pose=pose)

        raw: list[SignalRecord] = []
        for ext in self.extractors:
            t0 = self.emit_clock()
            recs = ext.consume(ctx)
            self.metrics.record_call(ext.name, self.emit_clock() - t0)
            raw.extend(recs)

        # gate (rides on every record; suppresses HRV/RR; holds last-good HR)
        if self.gate_controller is not None:
            motion = self.motion_provider(ctx) if self.motion_provider else 0.0
            snr = self._snr(ctx, raw)
            self.gate_controller.step(motion, snr)
            records = self.gate_controller.apply(raw)
        else:
            records = raw

        now = self.emit_clock()
        for r in records:
            self.metrics.emits += 1
            self.metrics.max_latency_s = max(self.metrics.max_latency_s, now - r.ts)
            if self.bus is not None:
                self.bus.publish(r)
            if self.recorder is not None:
                self.recorder.record(r)
            if self.dashboard is not None:
                self.dashboard.update(r)
        self.metrics.frames += 1
        return records

    def _snr(self, ctx, raw) -> float:
        if self.snr_provider:
            return float(self.snr_provider(ctx))
        for r in raw:
            if r.name == "hr":
                return float(r.meta.get("snr_db", 6.0))
        return 6.0

    def run_source(self, source, max_frames: int | None = None, fps: float = 30.0) -> int:
        """Drain a frame source synchronously; returns frames processed."""
        i = 0
        while max_frames is None or i < max_frames:
            frame, ok = source.read()
            if not ok:
                break
            self.process_frame(frame, ts=(i + 1) / fps, frame_id=i)
            i += 1
        return i
