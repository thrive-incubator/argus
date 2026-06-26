"""HRV extractor (FR-6, D3). Upsample → peaks → IBI → SDNN/RMSSD, GOOD-fraction gated."""

from __future__ import annotations

import numpy as np

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.hrv import compute_hrv, correct_ibis, detect_peaks, upsample_bvp
from ..dsp.rppg import bandpass, hr_from_bvp, pos
from ..dsp.roi import roi_mean_rgb
from ..dsp.sqi import window_sqi_gate
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
        # time-span gate (robust to live throughput below nominal fps)
        if tw.size < 8 or (float(tw[-1]) - float(tw[0])) < self.min_window_s:
            return []
        self._last_emit = ctx.ts

        tu, vu = resample_uniform(tw, vw, self.fps)
        bvp = bandpass(pos(vu, self.fps), self.fps)
        # Per-window signal-quality is ADVISORY (review §2): we measure it and surface it via the
        # record's sqi/meta, but only suppress a *catastrophic* (essentially-noise) window. A hard
        # SNR/NSQI reject blanked HRV on perfectly usable webcam rPPG (0–8 dB SNR).
        hr_hint = hr_from_bvp(bvp, self.fps)
        gate = window_sqi_gate(bvp, self.fps, hr_hint)
        if gate["score"] < 0.05:  # pure noise → no meaningful HRV
            self.insufficient = True
            return []
        _, y_up, fs_up = upsample_bvp(bvp, self.fps, self.upsample_hz)  # D3.AC1
        peaks_t = detect_peaks(y_up, fs_up)  # parabolic sub-sample timing (review §2)
        peaks_t, good = correct_ibis(peaks_t, fs_up)  # clean ectopic/missed beats (review §2)
        ibi_ms = np.diff(peaks_t) * 1000.0
        if ibi_ms.size < 2:
            self.insufficient = True
            return []
        clean_frac = float(np.mean(good)) if good is not None and len(good) else 1.0
        # Emit on the cleaned IBIs. The GOOD-fraction emit POLICY lives in compute_hrv (D3.AC3,
        # exercised in tests); the live path keeps the original always-emit behaviour and reports
        # quality through sqi/meta rather than blanking the card.
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
                sqi=float(gate["score"]),  # window signal quality (0..1)
                ts=center,
                gate="good",
                meta={
                    "sdnn_ms": res.sdnn_ms,
                    "rmssd_ms": res.rmssd_ms,
                    "n_beats": res.n_beats,
                    "clean_beat_fraction": clean_frac,
                    "window_sqi": gate["score"],
                    "nsqi": gate["nsqi"],
                    "low_quality": not gate["accept"],
                    "rest_only": True,
                },
            )
        ]
