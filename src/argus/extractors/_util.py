"""Shared extractor helpers."""

from __future__ import annotations

import numpy as np


def resample_uniform(t: np.ndarray, v: np.ndarray, fps: float):
    """Resample an unevenly-sampled (t, v) series onto a uniform ``fps`` grid.

    ``v`` may be 1-D or (N, C); returns ``(t_uniform, v_uniform)``.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < 2:
        return t, v
    n = int((t[-1] - t[0]) * fps) + 1
    tu = t[0] + np.arange(n) / fps
    if v.ndim == 1:
        vu = np.interp(tu, t, v)
    else:
        vu = np.stack([np.interp(tu, t, v[:, c]) for c in range(v.shape[1])], axis=1)
    return tu, vu
