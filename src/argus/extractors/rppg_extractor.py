"""HR-via-POS extractor (FR-5, D2). Rolling 8–15 s window, ~1 Hz update."""

from __future__ import annotations

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.rppg import bandpass, hr_from_bvp, pos
from ..dsp.roi import roi_mean_rgb
from ..dsp.sqi import dehaan_snr, snr_to_sqi
from ._util import resample_uniform


class RppgExtractor(Extractor):
    name = "hr"

    def __init__(self, fps: float = 30.0, window_s: float = 10.0,
                 min_window_s: float = 8.0, update_period_s: float = 1.0,
                 gate_fn=None):
        self.fps = fps
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.update_period_s = update_period_s
        self.update_hz = update_period_s
        self.gate_fn = gate_fn  # callable(ctx) -> gate str; default "unknown" (P0 stub)
        self._ring = TimeSeriesRing()
        self._last_emit: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:  # A2.AC3 — no face, no emission
            return []
        rgb = roi_mean_rgb(ctx.frame, ctx.face.landmarks)
        self._ring.append(ctx.ts, rgb)

        if self._last_emit is not None and (ctx.ts - self._last_emit) < self.update_period_s:
            return []
        tw, vw = self._ring.window(self.window_s)
        if tw.size < int(self.fps * self.min_window_s):
            return []

        tu, vu = resample_uniform(tw, vw, self.fps)
        bvp = bandpass(pos(vu, self.fps), self.fps)
        hr = hr_from_bvp(bvp, self.fps)
        snr = dehaan_snr(bvp, self.fps, hr)
        self._last_emit = ctx.ts
        center = float((tw[0] + tw[-1]) / 2.0)
        gate = self.gate_fn(ctx) if self.gate_fn else "unknown"
        sqi = snr_to_sqi(snr)
        return [
            SignalRecord(
                name="hr",
                value=float(hr),
                sqi=sqi,
                ts=center,
                gate=gate,
                # low-SQI records are emitted WITH a flag, never silently dropped (H1.AC3)
                meta={"snr_db": snr, "method": "pos", "window_s": self.window_s,
                      "low_sqi": sqi < 0.3},
            )
        ]
