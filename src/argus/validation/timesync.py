"""Time-sync utilities for camera-vs-H10 alignment (C2, FR-22, ADR-16).

- ``instantaneous_hr``: beat times → (t, hr) series.
- ``resample``: onto a common grid (default 4 Hz).
- ``cross_correlate_lag``: recover the residual lag between two HR series.
- ``beat_match``: pair camera/reference beats within a tolerance; report yield.
- ``dtw_distance``: a *reported* jitter-tolerant metric, NEVER used as the aligner (C2.AC3).
"""

from __future__ import annotations

import numpy as np


def instantaneous_hr(beat_times):
    """Return ``(t, hr_bpm)`` from beat timestamps (HR at the midpoint of each IBI)."""
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2:
        return np.array([]), np.array([])
    ibi = np.diff(bt)
    t = (bt[:-1] + bt[1:]) / 2.0
    return t, 60.0 / ibi


def resample(t, y, fs: float = 4.0):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return t, y
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, y)


def cross_correlate_lag(a, b, fs: float = 4.0) -> float:
    """Residual lag (seconds) of ``b`` relative to ``a`` via cross-correlation peak.

    Positive lag means ``b`` is delayed relative to ``a``.
    """
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-n + 1, n)
    # Positive return == b is delayed relative to a (the natural convention).
    return float(-lags[int(np.argmax(corr))] / fs)


def beat_match(cam_beats, ref_beats, tol_s: float = 0.075):
    """Match camera beats to reference beats within ``tol_s``.

    Returns ``(pairs, yield_fraction)`` where pairs is a list of (cam_t, ref_t).
    """
    cam = np.asarray(cam_beats, dtype=float)
    ref = np.asarray(ref_beats, dtype=float)
    pairs = []
    used = np.zeros(len(ref), dtype=bool)
    for c in cam:
        if ref.size == 0:
            break
        j = int(np.argmin(np.abs(ref - c)))
        if not used[j] and abs(ref[j] - c) <= tol_s:
            used[j] = True
            pairs.append((float(c), float(ref[j])))
    yield_frac = len(pairs) / len(cam) if len(cam) else 0.0
    return pairs, yield_frac


def dtw_distance(a, b) -> float:
    """Dynamic-time-warping distance — a REPORTED metric only, never the aligner (C2.AC3)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])
