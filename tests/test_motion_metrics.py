"""Review §5 — MEA frame-differencing + movement-smoothness (SPARC / LDLJ)."""

import numpy as np

from argus.dsp.motion import (
    log_dimensionless_jerk,
    motion_energy,
    spectral_arc_length,
)


def test_motion_energy_zero_when_static_positive_when_moving():
    a = np.full((40, 40), 100, np.uint8)
    assert motion_energy(a, a) == 0.0
    rng = np.random.default_rng(0)
    b = (a.astype(int) + rng.integers(-40, 40, a.shape)).clip(0, 255).astype(np.uint8)
    assert motion_energy(a, b) > 0.0


def test_motion_energy_roi():
    a = np.zeros((40, 40), np.uint8)
    b = a.copy()
    b[0:10, 0:10] = 200  # change only the top-left
    assert motion_energy(a, b, roi=(20, 20, 40, 40)) == 0.0  # ROI sees no change
    assert motion_energy(a, b, roi=(0, 0, 10, 10)) > 0.0


def test_sparc_smoother_is_less_negative():
    fs = 30.0
    t = np.arange(int(fs * 5)) / fs
    smooth = np.abs(np.sin(2 * np.pi * 0.5 * t))          # one slow bell-shaped movement
    rng = np.random.default_rng(1)
    jerky = np.abs(np.sin(2 * np.pi * 0.5 * t)) + 0.5 * np.abs(rng.standard_normal(t.size))
    assert spectral_arc_length(smooth, fs) > spectral_arc_length(jerky, fs)


def test_ldlj_smoother_is_less_negative():
    fs = 30.0
    t = np.arange(int(fs * 5)) / fs
    smooth = np.abs(np.sin(2 * np.pi * 0.5 * t))
    rng = np.random.default_rng(2)
    jerky = smooth + 0.5 * np.abs(rng.standard_normal(t.size))
    assert log_dimensionless_jerk(smooth, fs) > log_dimensionless_jerk(jerky, fs)


def test_sparc_degenerate_returns_zero():
    assert spectral_arc_length(np.zeros(100), 30.0) == 0.0
    assert spectral_arc_length(np.ones(2), 30.0) == 0.0
