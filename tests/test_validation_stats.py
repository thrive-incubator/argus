"""R19–R22: validation statistics (Bland-Altman, errors, Lin's CCC, EC13/CTA-2065)."""

import numpy as np
import pytest

from argus.validation.stats import (
    bland_altman,
    cta2065_pass,
    ec13_pass,
    lins_ccc,
    mae,
    mape,
    rmse,
)


# R20
def test_error_metrics_match_hand_computation():
    measured = np.array([70.0, 80.0, 90.0])
    reference = np.array([72.0, 78.0, 93.0])
    assert mae(measured, reference) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(measured, reference) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    expected_mape = np.mean([2 / 72, 2 / 78, 3 / 93]) * 100
    assert mape(measured, reference) == pytest.approx(expected_mape)


def test_mape_undefined_on_zero_reference():
    with pytest.raises(ValueError):
        mape([1.0], [0.0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


# R19
def test_bland_altman_bias_and_loa():
    measured = np.array([10.0, 12.0, 14.0, 16.0])
    reference = np.array([11.0, 11.0, 15.0, 15.0])
    ba = bland_altman(measured, reference)
    diff = measured - reference
    assert ba.bias == pytest.approx(diff.mean())
    sd = np.std(diff, ddof=1)
    assert ba.loa_lower == pytest.approx(ba.bias - 1.96 * sd)
    assert ba.loa_upper == pytest.approx(ba.bias + 1.96 * sd)


# R21
def test_lins_ccc_perfect_agreement_is_one():
    x = np.array([60.0, 70.0, 80.0, 90.0])
    assert lins_ccc(x, x) == pytest.approx(1.0)


def test_lins_ccc_penalises_offset():
    x = np.array([60.0, 70.0, 80.0, 90.0])
    # perfectly correlated but shifted by 10 -> Pearson 1.0 but CCC < 1
    assert lins_ccc(x, x + 10.0) < 0.95


# R22
def test_ec13_pass_within_tolerance():
    ref = np.array([70.0, 72.0, 68.0, 71.0])
    assert ec13_pass(ref + 3.0, ref) is True  # 3 bpm < max(5, 10%)
    assert ec13_pass(ref + 9.0, ref) is False  # 9 bpm > tolerance


def test_cta2065_threshold():
    ref = np.array([70.0, 72.0, 68.0])
    assert cta2065_pass(ref * 1.05, ref) is True  # 5% < 10%
    assert cta2065_pass(ref * 1.20, ref) is False  # 20% > 10%
