"""Head/posture motion + fidget extractor (FR-12).

Posture from torso geometry; fidget index = aggregate landmark jitter energy over a short
window. Fidget uses lightly-filtered landmarks so the One-Euro pose filter doesn't suppress
the very high-frequency signal it measures (review item m2).
"""

from __future__ import annotations

import numpy as np

from ..capture.buffers import TimeSeriesRing
from ..contracts import Extractor, FrameContext, SignalRecord

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24


class MotionExtractor(Extractor):
    name = "fidget"

    def __init__(self, fps: float = 30.0, window_s: float = 5.0, update_period_s: float = 1.0):
        self.fps = fps
        self.window_s = window_s
        self.update_period_s = update_period_s
        self._ring = TimeSeriesRing()
        self._last_emit: float | None = None

    @staticmethod
    def _posture_lean(pose) -> float:
        """Forward/lateral lean: horizontal offset of shoulder midpoint vs hip midpoint."""
        sh = pose.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
        hp = pose.landmarks[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
        return float(sh[0] - hp[0])

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.pose is None:
            return []
        # Fidget needs the RAW (unfiltered) jitter — use image-space landmarks, which the
        # backbone does NOT One-Euro filter (the world landmarks are filtered, killing jitter).
        src = ctx.pose.image_landmarks if ctx.pose.image_landmarks is not None else ctx.pose.landmarks
        centroid = src[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
        self._ring.append(ctx.ts, centroid)

        if self._last_emit is not None and (ctx.ts - self._last_emit) < self.update_period_s:
            return []
        tw, vw = self._ring.window(self.window_s)
        if tw.size < 5:
            return []
        self._last_emit = ctx.ts

        vel = np.diff(vw, axis=0)
        fidget = float(np.sqrt((vel**2).sum(axis=1)).mean()) * 100.0  # scale to readable units
        lean = self._posture_lean(ctx.pose)
        return [
            SignalRecord("fidget", fidget, 1.0, ctx.ts, gate="unknown",
                         meta={"posture_lean": lean, "descriptive": True}),
            SignalRecord("posture_lean", lean, 1.0, ctx.ts, gate="unknown",
                         meta={"descriptive": True}),
        ]
