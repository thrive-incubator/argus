"""Remote-PPG: the POS algorithm and HR-from-waveform (ADR-03, TECH §6.1).

POS = Plane-Orthogonal-to-Skin (Wang et al., IEEE TBME 2017). Deterministic, CPU,
no training — the live cardiac path for a clean single seated adult.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Physiological HR search band.
HR_MIN_BPM = 42.0
HR_MAX_BPM = 240.0
# rPPG band-pass support (TECH §6.1 / §7).
BANDPASS_HZ = (0.7, 4.0)


def pos(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Extract a blood-volume-pulse (BVP) waveform from an RGB trace via POS.

    Args:
        rgb: array shape (N, 3) of per-frame spatial-mean R, G, B over the ROI.
        fps: frame rate (samples/s) — accepted for API symmetry / future windowing.

    Returns:
        1-D BVP waveform of length N (zero-mean).
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape (N, 3)")
    n = rgb.shape[0]
    if n < 4:
        raise ValueError("need at least 4 samples for POS")

    # Temporal normalisation: divide each channel by its mean over the window.
    mean = rgb.mean(axis=0)
    mean[mean == 0] = 1e-9
    cn = (rgb / mean).T  # shape (3, N)

    # Projection onto the two POS chrominance planes.
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    s = projection @ cn  # shape (2, N)

    std1 = s[1].std()
    alpha = (s[0].std() / std1) if std1 > 1e-12 else 0.0
    h = s[0] + alpha * s[1]
    h = h - h.mean()
    return h


def bandpass(x: np.ndarray, fps: float, band=BANDPASS_HZ) -> np.ndarray:
    """Zero-phase Butterworth band-pass within the rPPG support band."""
    x = np.asarray(x, dtype=float)
    nyq = fps / 2.0
    lo = max(band[0] / nyq, 1e-4)
    hi = min(band[1] / nyq, 0.999)
    b, a = signal.butter(3, [lo, hi], btype="band")
    return signal.filtfilt(b, a, x)


def hr_from_bvp(bvp: np.ndarray, fps: float, nfft: int = 4096) -> float:
    """Estimate HR (bpm) as the PSD peak constrained to [HR_MIN, HR_MAX] bpm.

    Uses a zero-padded periodogram for fine frequency resolution (TECH §6.1).
    """
    bvp = np.asarray(bvp, dtype=float)
    bvp = bvp - bvp.mean()
    if len(bvp) < 4:
        raise ValueError("need at least 4 samples")
    nfft = max(nfft, len(bvp))
    freqs, psd = signal.periodogram(bvp, fs=fps, nfft=nfft, detrend="constant")
    f_lo, f_hi = HR_MIN_BPM / 60.0, HR_MAX_BPM / 60.0
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        raise ValueError("no spectral bins in the HR band")
    peak = freqs[mask][int(np.argmax(psd[mask]))]
    return float(peak * 60.0)


def estimate_hr(rgb: np.ndarray, fps: float) -> float:
    """Full live-path HR: POS -> band-pass -> PSD peak (bpm)."""
    bvp = bandpass(pos(rgb, fps), fps)
    return hr_from_bvp(bvp, fps)
