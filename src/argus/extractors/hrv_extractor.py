"""HRV extractor (FR-6, D3). Upsample → peaks → IBI → SDNN/RMSSD, GOOD-fraction gated."""

from __future__ import annotations

import numpy as np

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.hrv import compute_hrv, detect_peaks, upsample_bvp
from ..dsp.rppg import bandpass, pos
from ..dsp.roi import roi_mean_rgb
from ._util import resample_uniform


class HrvExtractor(Extractor):
    name = "hrv"

    def __init__(self, fps: float = 30.0, window_s: float = 60.0,
                 min_window_s: float = 20.0, update_period_s: float = 5.0,
                 upsample_hz: float = 256.0, good_fraction_min: float = 0.80):
        self.fps = fps
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.update_period_s = update_period_s
        self.upsample_hz = upsample_hz
        self.good_fraction_min = good_fraction_min
        self._ring = TimeSeriesRing()
        self._last_emit: float | None = None
        self.insufficient = False  # observability: last window lacked clean data

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:
            return []
        self._ring.append(ctx.ts, roi_mean_rgb(ctx.frame, ctx.face.landmarks))

        if self._last_emit is not None and (ctx.ts - self._last_emit) < self.update_period_s:
            return []
        tw, vw = self._ring.window(self.window_s)
        if tw.size < int(self.fps * self.min_window_s):
            return []
        self._last_emit = ctx.ts

        tu, vu = resample_uniform(tw, vw, self.fps)
        bvp = bandpass(pos(vu, self.fps), self.fps)
        _, y_up, fs_up = upsample_bvp(bvp, self.fps, self.upsample_hz)  # D3.AC1
        peaks_t = detect_peaks(y_up, fs_up)
        ibi_ms = np.diff(peaks_t) * 1000.0
        if ibi_ms.size < 2:
            self.insufficient = True
            return []
        res = compute_hrv(ibi_ms, good_flags=None, good_fraction_min=self.good_fraction_min)
        if res is None:  # D3.AC3 — report insufficient, never a silent pass
            self.insufficient = True
            return []
        self.insufficient = False
        center = float((tw[0] + tw[-1]) / 2.0)
        return [
            SignalRecord(
                name="hrv",
                value=res.sdnn_ms,  # committed metric
                sqi=res.good_fraction,
                ts=center,
                gate="good",
                meta={
                    "sdnn_ms": res.sdnn_ms,
                    "rmssd_ms": res.rmssd_ms,
                    "n_beats": res.n_beats,
                    "good_fraction": res.good_fraction,
                    "rest_only": True,
                },
            )
        ]
