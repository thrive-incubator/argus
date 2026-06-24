"""Respiration extractor (FR-7, E1). Chest/shoulder displacement → RR (Indicative)."""

from __future__ import annotations

import numpy as np

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.respiration import RESP_BAND_HZ, respiration_rate
from ._util import resample_uniform

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


class RespirationExtractor(Extractor):
    name = "resp"

    def __init__(self, fps: float = 30.0, window_s: float = 20.0,
                 min_window_s: float = 15.0, update_period_s: float = 2.0):
        self.fps = fps
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.update_period_s = update_period_s
        self._ring = TimeSeriesRing()
        self._last_emit: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.pose is None:
            return []
        disp = float(ctx.pose.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean())
        visibility = float(ctx.pose.visibility[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean())
        self._ring.append(ctx.ts, disp)

        if self._last_emit is not None and (ctx.ts - self._last_emit) < self.update_period_s:
            return []
        tw, vw = self._ring.window(self.window_s)
        if tw.size < int(self.fps * self.min_window_s):
            return []
        self._last_emit = ctx.ts

        tu, vu = resample_uniform(tw, vw, self.fps)
        rr = respiration_rate(vu, self.fps)
        # SQI from in-band power fraction × pose visibility (E1.AC4)
        sqi = float(np.clip(self._band_fraction(vu) * visibility, 0.0, 1.0))
        center = float((tw[0] + tw[-1]) / 2.0)
        return [
            SignalRecord(
                name="resp",
                value=float(rr),
                sqi=sqi,
                ts=center,
                gate="unknown",
                meta={"indicative": True, "primary": "chest_motion", "band_hz": RESP_BAND_HZ},
            )
        ]

    def _band_fraction(self, x: np.ndarray) -> float:
        from scipy import signal

        x = x - x.mean()
        f, p = signal.periodogram(x, fs=self.fps)
        band = (f >= RESP_BAND_HZ[0]) & (f <= RESP_BAND_HZ[1])
        total = p.sum()
        return float(p[band].sum() / total) if total > 0 else 0.0
