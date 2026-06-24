"""One-Euro filter (ADR-02, A3.AC2). Casiez, Roussel & Vogel (2012).

Adaptive low-pass: low jitter at rest, low lag during motion. Used to smooth pose
landmarks before motion features (and a lightly-filtered variant feeds the fidget index).
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar One-Euro filter. Call with monotonically increasing timestamps."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None or t <= self._t_prev:
            self._t_prev, self._x_prev, self._dx_prev = t, x, 0.0
            return x
        dt = t - self._t_prev
        dx = (x - self._x_prev) / dt
        edx = self._dx_prev + _alpha(self.dcutoff, dt) * (dx - self._dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        x_hat = self._x_prev + _alpha(cutoff, dt) * (x - self._x_prev)
        self._t_prev, self._x_prev, self._dx_prev = t, x_hat, edx
        return x_hat


def filter_series(times, values, min_cutoff=1.0, beta=0.007) -> np.ndarray:
    """Apply a One-Euro filter to a 1-D series; returns the filtered array."""
    f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    return np.array([f(float(t), float(v)) for t, v in zip(times, values)])


class LandmarkOneEuro:
    """One-Euro filter over an array of landmarks (K, D): one filter per coordinate."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[int, OneEuroFilter] = {}

    def __call__(self, t: float, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        flat = pts.ravel()
        out = np.empty_like(flat)
        for i, v in enumerate(flat):
            f = self._filters.get(i)
            if f is None:
                f = self._filters[i] = OneEuroFilter(self.min_cutoff, self.beta)
            out[i] = f(t, float(v))
        return out.reshape(pts.shape)
