"""Shared per-frame motion / signal-quality index (algorithm-review §9, gap #7).

Non-task body motion is the #1 error source for rPPG, respiration, blink-P80 *and* fidget
simultaneously. A single shared per-frame motion index lets every signal weight/gate on the
same evidence (rPPG-vs-motion fusion weight, PERCLOS validity, separating fidget from tracking
noise) — more leverage than any single-algorithm swap.

``energy`` is the MEA frame-difference (0..1); ``quality`` maps it through ``exp(-k·energy)``
so 1.0 = perfectly still and it decays smoothly as the subject moves.
"""

from __future__ import annotations

import numpy as np

from ..dsp.motion import _to_gray, motion_energy


def motion_quality(energy: float, k: float = 25.0) -> float:
    """Map a 0..1 frame-difference motion energy to a 0..1 quality (1 = still)."""
    return float(np.exp(-k * max(float(energy), 0.0)))


class FrameMotionIndex:
    """Streaming shared motion index: feed frames, read ``.energy`` and ``.quality``."""

    def __init__(self, roi=None, k: float = 25.0):
        self.roi = roi
        self.k = k
        self._prev: np.ndarray | None = None
        self.energy: float = 0.0
        self.quality: float = 1.0

    def update(self, frame) -> float:
        gray = _to_gray(frame)
        if self._prev is None:
            self._prev = gray
            self.energy, self.quality = 0.0, 1.0
            return self.quality
        self.energy = motion_energy(self._prev, gray, self.roi)
        self.quality = motion_quality(self.energy, self.k)
        self._prev = gray
        return self.quality
