"""Body-motion / restlessness metrics (FR-12; algorithm-review §5).

The review's verdict on fidget is **AUGMENT** variance-of-velocity, not replace it:

- **Motion Energy Analysis (MEA)** — frame-differencing pixel change within a body ROI
  (Ramseyer & Tschacher) — is the validated psychotherapy/mental-health standard and is
  robust to landmark jitter. Used as the primary restlessness scalar.
- **Movement smoothness** — Spectral Arc Length (SPARC, Balasubramanian 2015) and
  log-dimensionless jerk (LDLJ) — captures *character* (smooth vs jittery), which the
  synchrony literature warns a single energy scalar misses.
"""

from __future__ import annotations

import numpy as np


def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    f = np.asarray(frame)
    if f.ndim == 3:
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f


def motion_energy(prev_gray: np.ndarray, gray: np.ndarray, roi=None) -> float:
    """MEA-style motion energy: mean absolute frame difference over an ROI, normalised 0..1.

    ``roi`` is ``(x0, y0, x1, y1)`` in pixels, or None for the full frame.
    """
    p = np.asarray(prev_gray, dtype=np.float64)
    g = np.asarray(gray, dtype=np.float64)
    if roi is not None:
        x0, y0, x1, y1 = roi
        p, g = p[y0:y1, x0:x1], g[y0:y1, x0:x1]
    return float(np.abs(g - p).mean() / 255.0)


class FrameMotionEnergy:
    """Streaming MEA over a torso ROI: feed frames, read the mean motion energy."""

    def __init__(self, roi=None):
        self.roi = roi
        self._prev: np.ndarray | None = None
        self.last: float = 0.0

    def update(self, frame: np.ndarray) -> float:
        gray = _to_gray(frame)
        if self._prev is None:
            self._prev = gray
            self.last = 0.0
            return 0.0
        e = motion_energy(self._prev, gray, self.roi)
        self._prev = gray
        self.last = e
        return e


def spectral_arc_length(speed: np.ndarray, fs: float, fc: float = 10.0,
                        amp_th: float = 0.05, padlevel: int = 4) -> float:
    """Spectral Arc Length (SPARC) smoothness of a speed profile (Balasubramanian 2015).

    Less negative (closer to 0) = smoother; more negative = jerkier. Returns 0.0 for a
    degenerate (near-constant / too-short) profile.
    """
    v = np.asarray(speed, dtype=float)
    if v.size < 4 or not np.any(np.abs(v) > 1e-12):
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mf = np.abs(np.fft.fft(v, nfft))
    peak = mf.max()
    if peak <= 1e-12:
        return 0.0
    mf = mf / peak
    sel = f <= fc
    f_sel, mf_sel = f[sel], mf[sel]
    above = np.where(mf_sel >= amp_th)[0]
    if above.size < 2:
        return 0.0
    lo, hi = above[0], above[-1]
    f_sel, mf_sel = f_sel[lo:hi + 1], mf_sel[lo:hi + 1]
    span = f_sel[-1] - f_sel[0]
    if span <= 1e-12:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(mf_sel) ** 2)))


def log_dimensionless_jerk(speed: np.ndarray, fs: float) -> float:
    """Log dimensionless jerk (LDLJ) smoothness of a speed profile.

    Less negative (closer to 0) = smoother. Returns 0.0 for a degenerate profile.
    """
    v = np.asarray(speed, dtype=float)
    if v.size < 3:
        return 0.0
    dt = 1.0 / fs
    vpeak = float(np.max(np.abs(v)))
    if vpeak <= 1e-12:
        return 0.0
    jerk = np.diff(v, n=2) / (dt ** 2)  # d²(speed)/dt²
    duration = v.size * dt
    dlj = (duration ** 3 / (vpeak ** 2)) * float(np.sum(jerk ** 2)) * dt
    if dlj <= 1e-20:
        return 0.0
    return float(-np.log(dlj))
