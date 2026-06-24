"""Exposure→host latency calibration via an LED flash (A1.AC4).

A bright flash is emitted at a known ``flash_emit_ts``; the camera observes it some host
latency later. We detect the brightness step in the captured series and return the
constant offset ``observed_capture_ts - flash_emit_ts`` to subtract from frame timestamps.
"""

from __future__ import annotations

import numpy as np


def detect_flash_index(brightness: np.ndarray, rel: float = 0.5) -> int:
    """Index of the first frame whose brightness crosses ``rel`` of the way from the
    baseline mean to the series maximum — robust to small baseline variation and to a
    zero-variance baseline (the LED flash is a large step)."""
    b = np.asarray(brightness, dtype=float)
    if b.size < 3:
        raise ValueError("need >= 3 brightness samples")
    base = b[: max(3, b.size // 4)]
    span = b.max() - base.mean()
    if span <= 1e-9:
        raise ValueError("no flash detected (flat brightness)")
    thr = base.mean() + rel * span
    crossings = b > thr
    if not crossings.any():
        raise ValueError("no flash detected")
    return int(np.argmax(crossings))


def exposure_offset(
    brightness: np.ndarray, capture_ts: np.ndarray, flash_emit_ts: float
) -> float:
    """Return the constant exposure→host offset (seconds) to subtract from frame ts."""
    idx = detect_flash_index(brightness)
    observed = float(np.asarray(capture_ts, dtype=float)[idx])
    return observed - flash_emit_ts
