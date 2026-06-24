"""R7–R9: POS rPPG, HR-from-PSD, De Haan SNR on synthetic traces (TECH §14a CI set)."""

import numpy as np
import pytest

from argus.dsp.rppg import bandpass, estimate_hr, hr_from_bvp, pos
from argus.dsp.sqi import dehaan_snr, snr_to_sqi


def _synth_rgb(hr_bpm, fps=30.0, seconds=12.0, noise=0.01, seed=0):
    """Synthetic ROI RGB trace with a blood-volume pulse at hr_bpm."""
    rng = np.random.default_rng(seed)
    n = int(fps * seconds)
    t = np.arange(n) / fps
    f0 = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f0 * t)
    # Per-channel DC + pulse amplitude (green carries the strongest pulse).
    base = np.array([0.80, 0.50, 0.40])
    amp = np.array([0.010, 0.030, 0.005])
    rgb = base[None, :] + amp[None, :] * pulse[:, None]
    rgb += noise * rng.standard_normal(rgb.shape)
    return rgb, t


# R7 + R8
@pytest.mark.parametrize("hr", [54.0, 72.0, 96.0, 120.0])
def test_pos_recovers_injected_hr(hr):
    rgb, _ = _synth_rgb(hr)
    est = estimate_hr(rgb, fps=30.0)
    assert abs(est - hr) <= 5.0  # within EC13 tolerance


def test_pos_output_shape_and_zero_mean():
    rgb, _ = _synth_rgb(72.0)
    bvp = pos(rgb, fps=30.0)
    assert bvp.shape == (rgb.shape[0],)
    assert abs(float(bvp.mean())) < 1e-9


def test_hr_from_bvp_constrained_to_band():
    fps = 30.0
    t = np.arange(int(fps * 12)) / fps
    # A 10 Hz tone is far outside the HR band; the picked peak must stay <= 240 bpm.
    bvp = np.sin(2 * np.pi * 10.0 * t)
    hr = hr_from_bvp(bvp, fps)
    assert 42.0 <= hr <= 240.0


def test_pos_rejects_bad_shape():
    with pytest.raises(ValueError):
        pos(np.zeros((10, 2)), fps=30.0)


# R9 — De Haan SNR: clean pulse >> noisy; denominator excludes signal bins.
def test_dehaan_snr_clean_beats_noisy():
    rgb_clean, _ = _synth_rgb(72.0, noise=0.002, seed=1)
    rgb_noisy, _ = _synth_rgb(72.0, noise=0.05, seed=1)
    snr_clean = dehaan_snr(bandpass(pos(rgb_clean, 30.0), 30.0), 30.0, 72.0)
    snr_noisy = dehaan_snr(bandpass(pos(rgb_noisy, 30.0), 30.0), 30.0, 72.0)
    assert snr_clean > snr_noisy


def test_dehaan_snr_pure_tone_is_high():
    fps = 30.0
    t = np.arange(int(fps * 12)) / fps
    bvp = np.sin(2 * np.pi * 1.2 * t)  # exactly 72 bpm
    snr = dehaan_snr(bvp, fps, 72.0)
    assert snr > 8.0  # almost all power in-band (finite-window leakage caps it ~10 dB)
    sqi = snr_to_sqi(snr)
    assert 0.0 <= sqi <= 1.0
    assert sqi > 0.8  # high-quality signal maps near the top of the 0..1 range
