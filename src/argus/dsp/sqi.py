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


def skewness_sqi(waveform: np.ndarray) -> float:
    """Skewness of the pulse waveform — a cheap shape-sanity SQI (Elgendi 2016)."""
    from scipy import stats

    return float(stats.skew(np.asarray(waveform, dtype=float)))


def perfusion_index(waveform: np.ndarray) -> float:
    """Perfusion index = AC amplitude / |DC| (pulsatile strength)."""
    w = np.asarray(waveform, dtype=float)
    dc = np.abs(w.mean())
    ac = w.max() - w.min()
    return float(ac / dc) if dc > 1e-12 else 0.0


def beat_template_correlations(waveform, fs, peak_idx):
    """Per-beat Pearson correlation against the average beat template (Orphanidou 2015).

    Beats are windowed around each detected peak using the median beat length.
    """
    w = np.asarray(waveform, dtype=float)
    peaks = np.asarray(peak_idx, dtype=int)
    if len(peaks) < 3:
        return np.array([])
    half = int(np.median(np.diff(peaks)) / 2)
    half = max(half, 2)
    beats = []
    for p in peaks:
        if p - half >= 0 and p + half < len(w):
            beats.append(w[p - half : p + half])
    if len(beats) < 3:
        return np.array([])
    beats = np.array(beats)
    template = beats.mean(axis=0)
    corrs = []
    for b in beats:
        if b.std() < 1e-9 or template.std() < 1e-9:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(b, template)[0, 1]))
    return np.array(corrs)


def orphanidou_bsqi(waveform, fs, peak_idx, threshold: float = 0.86):
    """Orphanidou beat-template SQI.

    Returns ``(fraction_accepted, accepted_mask, per_beat_corr)``. The 0.86 default is
    PROVISIONAL for rPPG and is meant to be calibrated against H10-confirmed beats
    (D3.AC2) — pass a calibrated ``threshold``.
    """
    corrs = beat_template_correlations(waveform, fs, peak_idx)
    if corrs.size == 0:
        return 0.0, np.array([], dtype=bool), corrs
    accepted = corrs >= threshold
    return float(accepted.mean()), accepted, corrs
