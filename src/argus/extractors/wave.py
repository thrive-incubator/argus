"""Per-frame waveform extractors for live visualization.

These emit one sample per frame (not windowed) so a web frontend can draw the *shape* of
the pulse and the breathing motion — the rPPG blood-volume wave and chest displacement —
rather than only the windowed rate numbers.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.roi import roi_mean_rgb

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


class PulseWaveExtractor(Extractor):
    """Per-frame detrended ROI green value — the visible rPPG pulse waveform."""

    name = "pulse_wave"

    def __init__(self, detrend_frames: int = 90):
        self._buf: deque[float] = deque(maxlen=detrend_frames)

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:
            return []
        green = float(roi_mean_rgb(ctx.frame, ctx.face.landmarks)[1])
        self._buf.append(green)
        detrended = green - float(np.mean(self._buf))
        return [SignalRecord("pulse_wave", detrended, 1.0, ctx.ts, gate="unknown")]


class BreathWaveExtractor(Extractor):
    """Per-frame detrended chest/shoulder displacement — the visible breathing waveform."""

    name = "breath_wave"

    def __init__(self, detrend_frames: int = 150):
        self._buf: deque[float] = deque(maxlen=detrend_frames)

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.pose is None:
            return []
        disp = float(ctx.pose.landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean())
        self._buf.append(disp)
        detrended = disp - float(np.mean(self._buf))
        return [SignalRecord("breath_wave", detrended, 1.0, ctx.ts, gate="unknown")]
