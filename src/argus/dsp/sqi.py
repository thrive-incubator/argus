"""Signal-quality indices for rPPG (ADR-06, TECH §7).

De Haan & Jeanne (2013) spectral SNR: in-band power (around the HR fundamental and
its first harmonic) divided by **out-of-band power only** — the signal bins are
excluded from the denominator, otherwise it is not the De Haan ratio (review item sM2).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .rppg import BANDPASS_HZ, HR_MAX_BPM, HR_MIN_BPM

_FUND_HALFWIDTH_BPM = 6.0  # ±6 bpm window around fundamental & first harmonic


def dehaan_snr(
    bvp: np.ndarray,
    fps: float,
    hr_bpm: float,
    halfwidth_bpm: float = _FUND_HALFWIDTH_BPM,
    nfft: int = 4096,
) -> float:
    """Return the De Haan spectral SNR in dB.

    Numerator: power within ±halfwidth of the fundamental and first harmonic.
    Denominator: all remaining power over the band-pass support (signal bins removed).
    """
    bvp = np.asarray(bvp, dtype=float)
    bvp = bvp - bvp.mean()
    nfft = max(nfft, len(bvp))
    freqs, psd = signal.periodogram(bvp, fs=fps, nfft=nfft, detrend="constant")

    f0 = hr_bpm / 60.0
    hw = halfwidth_bpm / 60.0
    support = (freqs >= BANDPASS_HZ[0]) & (freqs <= BANDPASS_HZ[1])

    signal_band = (
        ((freqs >= f0 - hw) & (freqs <= f0 + hw))
        | ((freqs >= 2 * f0 - hw) & (freqs <= 2 * f0 + hw))
    ) & support
    noise_band = support & ~signal_band

    sig_power = psd[signal_band].sum()
    noise_power = psd[noise_band].sum()
    if noise_power <= 1e-20:
        noise_power = 1e-20
    return float(10.0 * np.log10((sig_power + 1e-20) / noise_power))


def snr_to_sqi(snr_db: float, lo: float = -3.0, hi: float = 12.0) -> float:
    """Normalise an SNR (dB) to a 0..1 quality index for the ``sqi`` field."""
    return float(np.clip((snr_db - lo) / (hi - lo), 0.0, 1.0))
