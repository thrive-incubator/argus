"""R10–R13: HRV (upsample, SDNN/RMSSD, no LF/HF, GOOD-fraction policy)."""

import numpy as np
import pytest

from argus.dsp import hrv as hrv_mod
from argus.dsp.hrv import (
    EXCLUDED_HRV_METRICS,
    compute_hrv,
    correct_ibis,
    detect_peaks,
    rmssd,
    sdnn,
    upsample_bvp,
)
from argus.dsp.sqi import nsqi, window_sqi_gate


# R10
def test_upsample_reaches_target_rate_and_preserves_signal():
    fps = 30.0
    t = np.arange(int(fps * 8)) / fps
    bvp = np.sin(2 * np.pi * 1.2 * t)
    t_up, y_up, fs_up = upsample_bvp(bvp, fps, target_hz=256.0)
    assert fs_up >= 256.0
    assert len(t_up) == len(y_up)
    # Spline reconstructs the known sinusoid at an arbitrary interior time.
    probe = 3.111
    assert y_up[np.argmin(np.abs(t_up - probe))] == pytest.approx(
        np.sin(2 * np.pi * 1.2 * probe), abs=0.02
    )


# R11 — match independent hand computation, not the implementation's own expression.
def test_sdnn_rmssd_match_reference_formulas():
    nn = np.array([800.0, 820.0, 790.0, 810.0, 805.0])
    expected_sdnn = float(np.sqrt(np.sum((nn - nn.mean()) ** 2) / (len(nn) - 1)))
    diffs = [nn[i + 1] - nn[i] for i in range(len(nn) - 1)]
    expected_rmssd = float(np.sqrt(sum(d * d for d in diffs) / len(diffs)))
    assert sdnn(nn) == pytest.approx(expected_sdnn)
    assert rmssd(nn) == pytest.approx(expected_rmssd)


def test_sdnn_constant_series_is_zero():
    assert sdnn(np.full(6, 800.0)) == pytest.approx(0.0)


def test_hrv_requires_two_intervals():
    with pytest.raises(ValueError):
        sdnn(np.array([800.0]))


# R12 — LF/HF / frequency-domain HRV is NOT produced anywhere.
def test_no_lf_hf_api_exists():
    public = set(dir(hrv_mod))
    for name in ("lf", "hf", "lf_hf", "lomb", "psd_hrv", "frequency_hrv"):
        assert name not in public
    assert EXCLUDED_HRV_METRICS  # documented exclusion list present


# R13 — GOOD-fraction emit policy.
def test_compute_hrv_emits_when_mostly_good():
    nn = np.array([800.0, 810.0, 795.0, 805.0, 800.0])
    flags = np.array([True, True, True, True, False])  # 80% good
    res = compute_hrv(nn, flags, good_fraction_min=0.80)
    assert res is not None
    assert res.good_fraction == pytest.approx(0.8)
    assert res.n_beats == 5
    assert res.rest_only is True


def test_compute_hrv_returns_none_when_too_few_good():
    nn = np.array([800.0, 810.0, 795.0, 805.0, 800.0])
    flags = np.array([True, False, False, False, False])  # 20% good
    assert compute_hrv(nn, flags, good_fraction_min=0.80) is None


# Review §2 — parabolic interpolation recovers a sub-sample peak the integer grid misses.
def test_parabolic_peak_subsample_timing():
    fs = 50.0
    # a peak whose true vertex sits between samples (asymmetric neighbours)
    y = np.array([0.0, 0.1, 0.4, 0.9, 0.85, 0.3, 0.0, 0.0])
    coarse = detect_peaks(y, fs, interpolate=False)
    fine = detect_peaks(y, fs, interpolate=True)
    assert coarse.size == 1 and fine.size == 1
    # integer grid puts the peak at index 3 → 0.06 s; parabola shifts it toward the heavier side
    assert fine[0] != coarse[0]
    assert abs(fine[0] - coarse[0]) <= 0.5 / fs + 1e-9


def test_correct_ibis_flags_ectopic_beat():
    fs = 256.0
    # regular ~1 s beats with one early (ectopic) beat injected
    peaks = np.array([0.0, 1.0, 2.0, 2.4, 3.4, 4.4])
    _, good = correct_ibis(peaks, fs)
    assert good.shape[0] == peaks.size - 1
    assert good.sum() < good.shape[0]  # at least one interval rejected


def test_nsqi_low_for_clean_high_for_noise():
    fps, hr = 30.0, 72.0
    t = np.arange(int(fps * 12)) / fps
    clean = np.sin(2 * np.pi * (hr / 60.0) * t)
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal(t.size)
    assert nsqi(clean, fps, hr) < 0.293       # passes the rPPG accept threshold
    assert nsqi(noisy, fps, hr) > nsqi(clean, fps, hr)


def test_window_sqi_gate_accepts_clean_rejects_noise():
    fps, hr = 30.0, 72.0
    t = np.arange(int(fps * 12)) / fps
    clean = np.sin(2 * np.pi * (hr / 60.0) * t)
    rng = np.random.default_rng(1)
    noisy = rng.standard_normal(t.size)
    assert window_sqi_gate(clean, fps, hr)["accept"] is True
    assert window_sqi_gate(noisy, fps, hr)["accept"] is False


# Regression: a usable-but-noisy webcam rPPG window (SNR ~4 dB) must NOT be suppressed by the
# HRV extractor's advisory gate (it scores above the 0.05 catastrophic floor) — the strict
# accept/NSQI thresholds previously blanked HRV entirely on real signals.
def test_window_sqi_usable_signal_above_suppress_floor():
    from argus.dsp.rppg import bandpass
    fps, hr = 30.0, 72.0
    t = np.arange(int(fps * 20)) / fps
    rng = np.random.default_rng(7)
    bvp = bandpass(np.sin(2 * np.pi * (hr / 60.0) * t) + 1.0 * rng.standard_normal(t.size), fps)
    g = window_sqi_gate(bvp, fps, hr)
    assert g["score"] > 0.05            # above the catastrophic floor → HRV still emits (advisory)
    # and pure noise stays below it → correctly suppressed
    assert window_sqi_gate(rng.standard_normal(t.size), fps, hr)["score"] < 0.05
