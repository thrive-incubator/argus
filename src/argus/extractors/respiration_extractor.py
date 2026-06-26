"""Respiration extractor (FR-7, E1). Chest/shoulder displacement → RR (Indicative)."""

from __future__ import annotations

import numpy as np

import numpy as np

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.respiration import (
    RESP_BAND_HZ,
    ChestFlowRespiration,
    resp_band_fraction,
    respiration_rate,
    rppg_derived_rr,
)
from ..dsp.rppg import bandpass, pos
from ..dsp.roi import roi_mean_rgb
from ._util import resample_uniform

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


class RespirationExtractor(Extractor):
    name = "resp"

    def __init__(self, fps: float = 30.0, window_s: float = 20.0,
                 min_window_s: float = 15.0, update_period_s: float = 2.0,
                 optical_flow: bool = False):
        self.fps = fps
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.update_period_s = update_period_s
        self.optical_flow = optical_flow  # chest-ROI Farnebäck flow primary (review §3)
        self._ring = TimeSeriesRing()
        self._roi_ring = TimeSeriesRing()  # for the secondary rPPG-RR cross-check
        self._flow = ChestFlowRespiration(fps, window_s, min_window_s) if optical_flow else None
        self._last_emit: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.pose is None:
            return []
        disp = float(ctx.pose.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean())
        visibility = float(ctx.pose.visibility[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean())
        self._ring.append(ctx.ts, disp)
        if ctx.face is not None:  # collect ROI for the rPPG-derived RR cross-check
            self._roi_ring.append(ctx.ts, roi_mean_rgb(ctx.frame, ctx.face.landmarks))
        if self._flow is not None and ctx.frame is not None:
            try:
                self._flow.update(ctx.ts, ctx.frame)
            except Exception:
                pass  # flow is best-effort; chest-motion remains the fallback

        if self._last_emit is not None and (ctx.ts - self._last_emit) < self.update_period_s:
            return []
        tw, vw = self._ring.window(self.window_s)
        # time-span gate (robust to live throughput below nominal fps)
        if tw.size < 8 or (float(tw[-1]) - float(tw[0])) < self.min_window_s:
            return []
        self._last_emit = ctx.ts

        tu, vu = resample_uniform(tw, vw, self.fps)
        motion_rr = respiration_rate(vu, self.fps)  # shoulder-landmark estimate
        motion_sqi = float(np.clip(resp_band_fraction(vu, self.fps) * visibility, 0.0, 1.0))

        # chest-ROI optical-flow estimate (review §3 primary when it actually sees motion)
        flow_rr = float("nan")
        flow_sqi = 0.0
        if self._flow is not None:
            est = self._flow.estimate()
            if est is not None:
                flow_rr, flow_sqi = est

        # pick the primary by SQI: optical flow wins when the camera sees real chest motion;
        # otherwise the shoulder-landmark signal carries (e.g. low-texture / static frames).
        if flow_sqi > motion_sqi:
            primary, rr, sqi = "optical_flow", flow_rr, flow_sqi
        else:
            primary, rr, sqi = "chest_motion", motion_rr, motion_sqi

        # secondary rPPG-derived RR cross-check + agreement (E1.AC2)
        rppg_rr = float("nan")
        agreement = float("nan")
        rt, rv = self._roi_ring.window(self.window_s)
        if rt.size >= 8 and (float(rt[-1]) - float(rt[0])) >= self.min_window_s:
            _, rvu = resample_uniform(rt, rv, self.fps)
            bvp = bandpass(pos(rvu, self.fps), self.fps)
            rppg_rr = rppg_derived_rr(bvp, self.fps)
            agreement = abs(rr - rppg_rr)

        center = float((tw[0] + tw[-1]) / 2.0)
        return [
            SignalRecord(
                name="resp",
                value=float(rr),
                sqi=float(sqi),
                ts=center,
                gate="unknown",
                meta={"indicative": True, "primary": primary, "band_hz": RESP_BAND_HZ,
                      "motion_rr": motion_rr, "motion_sqi": motion_sqi,
                      "flow_rr": flow_rr, "flow_sqi": flow_sqi,
                      "rppg_rr": rppg_rr, "resp_agreement_brpm": agreement},
            )
        ]
