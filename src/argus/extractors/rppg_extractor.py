"""HR-via-POS extractor (FR-5, D2). Rolling 8–15 s window, ~1 Hz update."""

from __future__ import annotations

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.rppg import bandpass, hr_from_bvp, pos
from ..dsp.roi import STANDARD_PATCHES, roi_patch_stack
from ..dsp.sqi import dehaan_snr, fuse_patches_by_snr, snr_to_sqi
from ._util import resample_uniform

_N_PATCH = len(STANDARD_PATCHES)


class RppgExtractor(Extractor):
    name = "hr"

    def __init__(self, fps: float = 30.0, window_s: float = 10.0,
                 min_window_s: float = 8.0, update_period_s: float = 1.0,
                 gate_fn=None, per_patch: bool = True):
        self.fps = fps
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.update_period_s = update_period_s
        self.update_hz = update_period_s
        self.gate_fn = gate_fn  # callable(ctx) -> gate str; default "unknown" (P0 stub)
        self.per_patch = per_patch  # SNR-weighted multi-patch fusion (review §1)
        self._ring = TimeSeriesRing()
        self._last_emit: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:  # A2.AC3 — no face, no emission
            return []
        # Store ALL standard patches (glabella, forehead, cheeks) as a fixed-width vector so
        # the fusion can weight them by per-patch SNR at emit time (review §1).
        stack = roi_patch_stack(ctx.frame, ctx.face.landmarks)  # (n_patch, 3)
        self._ring.append(ctx.ts, stack.reshape(-1))

        if self._last_emit is not None and (ctx.ts - self._last_emit) < self.update_period_s:
            return []
        tw, vw = self._ring.window(self.window_s)
        # Gate on the window's actual time SPAN, not a frame count — so the estimate still
        # emits if live throughput dips below the nominal fps (resample fills the grid).
        if tw.size < 8 or (float(tw[-1]) - float(tw[0])) < self.min_window_s:
            return []

        tu, vu = resample_uniform(tw, vw, self.fps)  # (N, n_patch*3)
        patches = {lbl: vu[:, i * 3:(i + 1) * 3] for i, lbl in enumerate(STANDARD_PATCHES)}
        patch_info: dict = {}
        if self.per_patch:
            bvp, patch_info = fuse_patches_by_snr(patches, self.fps)
        else:
            mean_rgb = vu.reshape(len(vu), _N_PATCH, 3).mean(axis=1)
            bvp = bandpass(pos(mean_rgb, self.fps), self.fps)
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
                meta={"snr_db": snr, "method": "pos_multipatch" if self.per_patch else "pos",
                      "window_s": self.window_s, "low_sqi": sqi < 0.3,
                      "patch_weights": patch_info.get("weights", {}),
                      "patch_snr_db": patch_info.get("per_patch_snr_db", {})},
            )
        ]
