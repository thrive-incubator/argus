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


def _parabolic_offsets(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sub-sample peak offsets via 3-point parabolic interpolation (review §2).

    For each integer peak index, fit a parabola to (i-1, i, i+1) and return the offset of
    the vertex in samples (clamped to ±0.5). Upsampling raises the grid but does NOT recover
    sub-sample peak *timing*, which is exactly what limits RMSSD — this does.
    """
    if idx.size == 0:
        return np.zeros(0)
    i = idx.astype(int)
    interior = (i > 0) & (i < len(y) - 1)
    ym1 = np.where(interior, y[np.clip(i - 1, 0, len(y) - 1)], 0.0)
    y0 = y[i]
    yp1 = np.where(interior, y[np.clip(i + 1, 0, len(y) - 1)], 0.0)
    denom = ym1 - 2.0 * y0 + yp1
    off = np.where(np.abs(denom) > 1e-12, 0.5 * (ym1 - yp1) / denom, 0.0)
    off = np.where(interior, off, 0.0)
    return np.clip(off, -0.5, 0.5)


def detect_peaks(
    y: np.ndarray, fs: float, hr_max_bpm: float = 240.0, interpolate: bool = True
) -> np.ndarray:
    """Detect systolic peaks; return peak times (s). Min spacing set by HR_MAX.

    With ``interpolate`` (default), peak times are refined to sub-sample precision via
    parabolic interpolation — important for RMSSD, which is dominated by sub-sample timing
    that cubic-spline upsampling alone does not recover (review §2).
    """
    y = np.asarray(y, dtype=float)
    min_dist = max(int(fs * 60.0 / hr_max_bpm), 1)
    idx, _ = signal.find_peaks(y, distance=min_dist)
    if interpolate and idx.size:
        return (idx + _parabolic_offsets(y, idx)) / fs
    return idx / fs


def correct_ibis(peaks_t: np.ndarray, fs: float = HRV_UPSAMPLE_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Correct ectopic/missed/extra beats in a peak-time series (review §2).

    Uses NeuroKit2 ``signal_fixpeaks`` (Kubios / Lipponen–Tarvainen method) when available;
    otherwise falls back to a median-absolute-deviation outlier rejection on the IBIs.

    Returns ``(corrected_peaks_t, good_mask)`` where ``good_mask`` (per IBI) marks intervals
    that survived correction — usable as ``good_flags`` for :func:`compute_hrv`.
    """
    peaks_t = np.asarray(peaks_t, dtype=float)
    if peaks_t.size < 3:
        return peaks_t, np.ones(max(peaks_t.size - 1, 0), dtype=bool)
    try:
        import neurokit2 as nk

        peaks_idx = np.round(peaks_t * fs).astype(int)
        out = nk.signal_fixpeaks(peaks_idx, sampling_rate=fs, method="Kubios", iterative=True)
        clean = out[1] if isinstance(out, tuple) else out
        clean = np.asarray(clean, dtype=float)
        corrected_t = clean / fs
        # mark original IBIs whose interval changed materially as "corrected" (not good)
        ibi_orig = np.diff(peaks_t)
        good = np.ones(len(ibi_orig), dtype=bool)
        med = np.median(ibi_orig)
        if med > 0:
            good = np.abs(ibi_orig - med) <= 0.30 * med  # Kubios-style ±30% gate
        return corrected_t, good
    except Exception:
        ibi = np.diff(peaks_t)
        med = np.median(ibi)
        mad = np.median(np.abs(ibi - med)) + 1e-9
        good = np.abs(ibi - med) <= 5.0 * mad
        return peaks_t, good


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
