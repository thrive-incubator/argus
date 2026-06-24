"""HRV from a BVP waveform or IBI series (ADR-05, TECH §6.2).

Scope locked by ADR-05: **HR + SDNN committed, RMSSD indicative, LF/HF NOT produced.**
There is deliberately no frequency-domain HRV function in this module (review R12).

Pipeline order (TECH §6.2): cubic-spline upsample BVP to >=256 Hz before peak
detection (avoids 33 ms IBI quantisation) -> systolic peaks -> IBIs -> SDNN/RMSSD,
emitted only when >=80% of the window's beats are GOOD-gated/accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

HRV_UPSAMPLE_HZ = 256.0
GOOD_FRACTION_MIN = 0.80  # review item M9 / FEAT D3.AC3

# The frequency-domain band names we explicitly DO NOT compute (ADR-05).
EXCLUDED_HRV_METRICS = ("lf", "hf", "lf_hf", "lf/hf")


def upsample_bvp(bvp: np.ndarray, fps: float, target_hz: float = HRV_UPSAMPLE_HZ):
    """Cubic-spline upsample a BVP waveform to >= ``target_hz`` (TECH §6.2 step 1).

    Returns ``(t_up, y_up, fs_up)`` where ``fs_up >= target_hz``.
    """
    bvp = np.asarray(bvp, dtype=float)
    n = len(bvp)
    if n < 4:
        raise ValueError("need at least 4 samples to upsample")
    duration = (n - 1) / fps
    t = np.arange(n) / fps
    factor = int(np.ceil(target_hz / fps))
    fs_up = fps * factor
    n_up = int(round(duration * fs_up)) + 1
    t_up = np.arange(n_up) / fs_up
    y_up = CubicSpline(t, bvp)(t_up)
    return t_up, y_up, fs_up


def sdnn(nn_ms: np.ndarray) -> float:
    """Standard deviation of NN intervals (sample std, ddof=1), in ms."""
    nn = np.asarray(nn_ms, dtype=float)
    if len(nn) < 2:
        raise ValueError("need >= 2 NN intervals for SDNN")
    return float(np.std(nn, ddof=1))


def rmssd(nn_ms: np.ndarray) -> float:
    """Root mean square of successive NN differences, in ms (indicative)."""
    nn = np.asarray(nn_ms, dtype=float)
    if len(nn) < 2:
        raise ValueError("need >= 2 NN intervals for RMSSD")
    diffs = np.diff(nn)
    return float(np.sqrt(np.mean(diffs**2)))


def detect_peaks(y: np.ndarray, fs: float, hr_max_bpm: float = 240.0) -> np.ndarray:
    """Detect systolic peaks; return peak times (s). Min spacing set by HR_MAX."""
    y = np.asarray(y, dtype=float)
    min_dist = max(int(fs * 60.0 / hr_max_bpm), 1)
    idx, _ = signal.find_peaks(y, distance=min_dist)
    return idx / fs


@dataclass(frozen=True)
class HrvResult:
    sdnn_ms: float
    rmssd_ms: float
    n_beats: int
    good_fraction: float
    rest_only: bool = True


def compute_hrv(
    nn_ms: np.ndarray,
    good_flags: np.ndarray | None = None,
    good_fraction_min: float = GOOD_FRACTION_MIN,
):
    """Compute SDNN+RMSSD only if >= ``good_fraction_min`` of beats are GOOD.

    Args:
        nn_ms: NN/IBI intervals (ms).
        good_flags: optional bool array (per interval); None means all good.

    Returns:
        ``HrvResult`` if the GOOD-fraction policy is met, else ``None``
        ("insufficient clean data" — never a silent pass; TECH §6.2 step 6).
    """
    nn = np.asarray(nn_ms, dtype=float)
    if good_flags is None:
        good = np.ones(len(nn), dtype=bool)
    else:
        good = np.asarray(good_flags, dtype=bool)
    frac = float(good.mean()) if len(good) else 0.0
    if frac < good_fraction_min or len(nn) < 2:
        return None
    return HrvResult(
        sdnn_ms=sdnn(nn),
        rmssd_ms=rmssd(nn),
        n_beats=len(nn),
        good_fraction=frac,
    )
