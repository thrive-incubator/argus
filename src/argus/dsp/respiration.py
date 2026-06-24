"""Respiration rate from chest/shoulder motion (ADR-07 rev 2, TECH §6.3).

Indicative signal only (no respiratory belt → no accuracy/MAE claim). The band-pass
low edge is widened to 0.08 Hz so a 6-brpm (0.1 Hz) paced block is not attenuated at
the filter edge (review item sM1).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

RESP_BAND_HZ = (0.08, 0.5)  # 4.8–30 brpm


def rppg_derived_rr(bvp: np.ndarray, fps: float) -> float:
    """Secondary respiration estimate from the rPPG amplitude envelope (E1.AC2).

    Respiration amplitude-modulates the pulse; the Hilbert envelope's dominant frequency in
    the respiration band gives a (weaker, secondary) RR cross-check.
    """
    from scipy.signal import hilbert

    x = np.asarray(bvp, dtype=float)
    env = np.abs(hilbert(x - x.mean()))
    return respiration_rate(env, fps)


def respiration_rate(displacement: np.ndarray, fps: float, nfft: int = 4096) -> float:
    """Estimate respiration rate (breaths/min) from a chest-displacement series.

    band-pass [0.08, 0.5] Hz -> periodogram peak -> brpm.
    """
    x = np.asarray(displacement, dtype=float)
    x = x - x.mean()
    if len(x) < 8:
        raise ValueError("need at least 8 samples")
    nyq = fps / 2.0
    lo = max(RESP_BAND_HZ[0] / nyq, 1e-4)
    hi = min(RESP_BAND_HZ[1] / nyq, 0.999)
    b, a = signal.butter(2, [lo, hi], btype="band")
    xf = signal.filtfilt(b, a, x)
    nfft = max(nfft, len(xf))
    freqs, psd = signal.periodogram(xf, fs=fps, nfft=nfft, detrend="constant")
    mask = (freqs >= RESP_BAND_HZ[0]) & (freqs <= RESP_BAND_HZ[1])
    peak = freqs[mask][int(np.argmax(psd[mask]))]
    return float(peak * 60.0)
