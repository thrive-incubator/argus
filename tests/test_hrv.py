"""R10–R13: HRV (upsample, SDNN/RMSSD, no LF/HF, GOOD-fraction policy)."""

import numpy as np
import pytest

from argus.dsp import hrv as hrv_mod
from argus.dsp.hrv import (
    EXCLUDED_HRV_METRICS,
    compute_hrv,
    rmssd,
    sdnn,
    upsample_bvp,
)


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
