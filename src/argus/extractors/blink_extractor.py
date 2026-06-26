"""Blink/PERCLOS extractor (FR-8, F1).

EAR comes from the face landmarks via an injectable ``ear_fn`` (defaults to a landmark
EAR using the standard MediaPipe eye indices). Blink metrics are withheld until the
adaptive baseline is armed (F1.AC5).
"""

from __future__ import annotations

import numpy as np

from ..contracts import Extractor, FrameContext, SignalRecord
from ..dsp.blink import AdaptiveBlinkDetector, PerclosP80, eye_aspect_ratio, eye_openness

# MediaPipe FaceMesh right-eye 6-point ring (p1..p6) used for EAR.
RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
# ARKit/MediaPipe blendshape indices for eyeBlinkLeft / eyeBlinkRight (0 = open, 1 = closed).
EYE_BLINK_IDX = (9, 10)


def _default_ear(ctx: FrameContext) -> float | None:
    if ctx.face is None:
        return None
    pts = ctx.face.landmarks[list(RIGHT_EYE_IDX), :2]
    return eye_aspect_ratio(pts)


def _blink_blendshape(ctx: FrameContext) -> float | None:
    """Mean eyeBlink blendshape score (0..1) if the face backbone provides blendshapes."""
    face = getattr(ctx, "face", None)
    bs = getattr(face, "blendshapes", None) if face is not None else None
    if bs is None:
        return None
    bs = np.asarray(bs, dtype=float)
    if bs.shape[0] <= max(EYE_BLINK_IDX):
        return None
    return float(np.clip(bs[list(EYE_BLINK_IDX)].mean(), 0.0, 1.0))


class BlinkExtractor(Extractor):
    name = "blink"

    def __init__(self, fps: float = 30.0, baseline_frames: int = 300,
                 ratio: float = 0.6, min_frames: int = 2,
                 rate_window_s: float = 60.0, min_fps: float = 25.0, ear_fn=None,
                 perclos_window_s: float = 60.0, blink_max_s: float = 0.4,
                 blendshape_fn=None):
        self.fps = fps
        self.min_fps = min_fps
        self.rate_window_s = rate_window_s
        self.ear_fn = ear_fn or _default_ear
        self.blendshape_fn = blendshape_fn or _blink_blendshape
        self._detector = AdaptiveBlinkDetector(
            baseline_frames=baseline_frames, ratio=ratio, min_frames=min_frames
        )
        self._perclos = PerclosP80(fps=fps, window_s=perclos_window_s, blink_max_s=blink_max_s)
        self._blink_ts: list[float] = []
        self.fps_warning = fps < min_fps  # F1.AC5

    @property
    def armed(self) -> bool:
        return self._detector.armed

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        ear = self.ear_fn(ctx)
        if ear is None:
            return []
        completed = self._detector.update(float(ear))
        if not self._detector.armed:  # withhold until baseline armed (F1.AC5)
            return []
        out: list[SignalRecord] = []
        if completed:
            self._blink_ts.append(ctx.ts)
            out.append(
                SignalRecord("blink_event", 1.0, 1.0, ctx.ts, gate="unknown",
                             meta={"duration_frames": self._detector.min_frames})
            )
        # windowed blink rate (blinks/min)
        recent = [t for t in self._blink_ts if t >= ctx.ts - self.rate_window_s]
        span = max(self.rate_window_s, 1e-6)
        rate = len(recent) / span * 60.0
        out.append(
            SignalRecord("blink_rate", float(rate), 1.0, ctx.ts, gate="unknown",
                         meta={"fps_warning": self.fps_warning})
        )
        # graded eye-openness → true P80 PERCLOS with blink exclusion (review §4)
        baseline = self._detector._baseline or 1e-9
        openness = eye_openness(float(ear), baseline, self.blendshape_fn(ctx))
        self._perclos.update(ctx.ts, openness)
        out.append(
            SignalRecord("perclos", self._perclos.value(), 1.0, ctx.ts, gate="unknown",
                         meta={"openness": openness, "p80": True,
                              "window_s": self._perclos.window_s})
        )
        return out
