"""Covariate layer (TECH §9 rev 2).

Lighting is a **relative brightness index** from mean luma — uncalibrated, explicitly
NOT lux (luma depends on exposure/gain/white-balance). Absolute lux is logged
separately from a free phone app per block (review item M4).
"""

from __future__ import annotations

import numpy as np

# BT.601 luma weights for an RGB frame.
_LUMA = np.array([0.299, 0.587, 0.114])


def brightness_index(frame: np.ndarray) -> float:
    """Mean luma normalised to 0..1 (assumes 8-bit RGB/BGR-ish channels)."""
    f = np.asarray(frame, dtype=float)
    if f.ndim == 3 and f.shape[-1] >= 3:
        luma = f[..., :3] @ _LUMA
    else:
        luma = f
    return float(np.clip(luma.mean() / 255.0, 0.0, 1.0))


def exposure_flags(frame: np.ndarray, under: float = 0.15, over: float = 0.85):
    """Return ``(underexposed, overexposed)`` from the brightness index."""
    bi = brightness_index(frame)
    return bi < under, bi > over
