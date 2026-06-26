"""Signal-quality indices for rPPG (ADR-06, TECH §7).

De Haan & Jeanne (2013) spectral SNR: in-band power (around the HR fundamental and
its first harmonic) divided by **out-of-band power only** — the signal bins are
excluded from the denominator, otherwise it is not the De Haan ratio (review item sM2).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .rppg import BANDPASS_HZ, HR_MAX_BPM, HR_MIN_BPM, bandpass, hr_from_bvp, pos

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


def fuse_patches_by_snr(patch_series, fps: float):
    """Fuse per-patch RGB time-series into one BVP, weighting each patch by its De Haan SNR.

    Implements the review §1 recommendation ("per-patch SNR weighting + occlusion fallback"):
    each patch is run through POS independently, scored by spectral SNR, and the patches are
    combined as an SNR-weighted sum of their amplitude-normalised BVPs. A patch that is
    occluded/turned/motion-corrupted scores a low SNR and is automatically down-weighted —
    a soft generalisation of hard yaw-based patch dropping.

    Args:
        patch_series: ``{label: ndarray(N, 3)}`` (or a sequence of ``(N, 3)`` arrays) of
            per-frame mean RGB for each ROI patch over the analysis window.
        fps: frame rate of the (already uniformly resampled) series.

    Returns:
        ``(bvp, info)`` — the fused band-passed BVP and a dict with per-patch ``weights``,
        ``per_patch_hr``, ``per_patch_snr_db`` and a ``fallback`` flag (True when no patch
        produced a finite SNR and a plain mean was used instead).
    """
    if isinstance(patch_series, dict):
        labels = list(patch_series.keys())
        series = [np.asarray(patch_series[k], dtype=float) for k in labels]
    else:
        series = [np.asarray(s, dtype=float) for s in patch_series]
        labels = [str(i) for i in range(len(series))]

    bvps: list[np.ndarray | None] = []
    snrs: list[float] = []
    hrs: list[float] = []
    for s in series:
        try:
            bvp = bandpass(pos(s, fps), fps)
            hr = hr_from_bvp(bvp, fps)
            snr = dehaan_snr(bvp, fps, hr)
            std = float(bvp.std()) or 1.0
            bvps.append(bvp / std)
            snrs.append(float(snr))
            hrs.append(float(hr))
        except Exception:
            bvps.append(None)
            snrs.append(-np.inf)
            hrs.append(float("nan"))

    snr_arr = np.array(snrs, dtype=float)
    valid = np.isfinite(snr_arr)
    if not valid.any():  # occlusion fallback: plain mean of the raw patch series
        mean_rgb = np.mean(series, axis=0)
        try:
            fb = bandpass(pos(mean_rgb, fps), fps)
        except Exception:  # too short / degenerate even for the mean path
            fb = np.zeros(len(mean_rgb), dtype=float)
        return fb, {
            "weights": {}, "per_patch_hr": {}, "per_patch_snr_db": {}, "fallback": True,
        }

    # Positive weights anchored at the worst valid SNR (so the best patch dominates but the
    # others still contribute), normalised to sum to 1.
    floor = float(np.nanmin(snr_arr[valid]))
    w = np.where(valid, snr_arr - floor + 1e-3, 0.0)
    if w.sum() <= 0:
        w = valid.astype(float)
    w = w / w.sum()

    length = min(len(b) for b in bvps if b is not None)
    fused = np.zeros(length, dtype=float)
    for wi, b in zip(w, bvps):
        if b is not None:
            fused += wi * b[:length]

    info = {
        "weights": {labels[i]: float(w[i]) for i in range(len(labels))},
        "per_patch_hr": {labels[i]: hrs[i] for i in range(len(labels))},
        "per_patch_snr_db": {labels[i]: snrs[i] for i in range(len(labels))},
        "fallback": False,
    }
    return fused, info


def skewness_sqi(waveform: np.ndarray) -> float:
    """Skewness of the pulse waveform — a cheap shape-sanity SQI (Elgendi 2016)."""
    from scipy import stats

    return float(stats.skew(np.asarray(waveform, dtype=float)))


def nsqi(bvp: np.ndarray, fps: float, hr_bpm: float,
         halfwidth_bpm: float = _FUND_HALFWIDTH_BPM, nfft: int = 4096) -> float:
    """Normalised noise SQI: out-of-band power as a fraction of total band-pass power.

    Lower is cleaner. The rPPG-specific accept threshold from npj Biosensing (2024) is
    ``NSQI < 0.293`` (review §2). This complements the De Haan SNR (a ratio) with a bounded
    0..1 noise fraction that is easy to gate per window.
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
    total = float(psd[support].sum())
    if total <= 1e-20:
        return 1.0
    return float(psd[support & ~signal_band].sum() / total)


def window_sqi_gate(
    bvp: np.ndarray, fps: float, hr_bpm: float,
    snr_min_db: float = -1.0, nsqi_max: float = 0.293, abs_skew_max: float = 2.5,
) -> dict:
    """Per-window accept decision combining spectral SNR + NSQI + skewness sanity (review §2).

    Returns ``{accept, score, snr_db, nsqi, skewness}``. ``score`` is a 0..1 quality index
    (the SNR-mapped SQI attenuated by the NSQI noise fraction). Gating per window (not per
    recording) is the key point — a few bad windows otherwise destroy RMSSD.
    """
    snr = dehaan_snr(bvp, fps, hr_bpm)
    noise = nsqi(bvp, fps, hr_bpm)
    skew = abs(skewness_sqi(bvp))
    accept = (snr >= snr_min_db) and (noise <= nsqi_max) and (skew <= abs_skew_max)
    score = float(np.clip(snr_to_sqi(snr) * (1.0 - noise), 0.0, 1.0))
    return {"accept": bool(accept), "score": score, "snr_db": float(snr),
            "nsqi": float(noise), "skewness": float(skewness_sqi(bvp))}


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
