"""Blink detection (Eye Aspect Ratio) and PERCLOS (ADR-08, TECH §6.4).

EAR (Soukupová & Čech 2016) with a **per-session adaptive threshold** auto-calibrated
from the open-eye baseline; a blink is EAR below threshold for >= N consecutive frames.
PERCLOS (P80) = fraction of a window with the eye >= 80% closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR for one eye given 6 landmark points p1..p6 (shape (6, 2)).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).
    Points ordered: p1,p4 = horizontal corners; p2,p3,p5,p6 = vertical lids.
    """
    p = np.asarray(pts, dtype=float)
    if p.shape != (6, 2):
        raise ValueError("expected 6 (x, y) eye landmarks")
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    if c <= 1e-9:
        return 0.0
    return float((a + b) / (2.0 * c))


@dataclass
class AdaptiveBlinkDetector:
    """Streaming blink detector with a per-session adaptive threshold.

    Calibrates an open-eye baseline over the first ``baseline_frames`` *valid* frames,
    then flags a blink when EAR < ``ratio * baseline`` for >= ``min_frames`` in a row.
    Stays un-armed (returns no blinks) until enough baseline frames are seen — review
    item B-blink-baseline.
    """

    baseline_frames: int = 300  # ~10 s at 30 fps
    ratio: float = 0.6
    min_frames: int = 2
    _samples: list[float] = field(default_factory=list)
    _baseline: float | None = None
    _below: int = 0
    blink_count: int = 0

    @property
    def armed(self) -> bool:
        return self._baseline is not None

    @property
    def threshold(self) -> float | None:
        return None if self._baseline is None else self.ratio * self._baseline

    def update(self, ear: float) -> bool:
        """Feed one EAR sample. Returns True on the frame a blink completes."""
        if self._baseline is None:
            self._samples.append(ear)
            if len(self._samples) >= self.baseline_frames:
                self._baseline = float(np.median(self._samples))
            return False

        thr = self.ratio * self._baseline
        completed = False
        if ear < thr:
            self._below += 1
        else:
            if self._below >= self.min_frames:
                self.blink_count += 1
                completed = True
            self._below = 0
        return completed


def perclos(ear_series, baseline: float, closed_frac: float = 0.8) -> float:
    """PERCLOS-P80: fraction of frames with the eye at least 80% closed.

    "80% closed" means EAR <= (1 - closed_frac) * open-eye baseline.
    """
    ear = np.asarray(ear_series, dtype=float)
    if len(ear) == 0:
        return 0.0
    thr = (1.0 - closed_frac) * baseline
    return float(np.mean(ear <= thr))
