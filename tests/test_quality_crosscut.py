"""Algorithm-review §9 cross-cutting: fairness, shared motion index, accuracy ceilings."""

import numpy as np

from argus.quality.ceilings import SIGNAL_CEILINGS, ceiling_for
from argus.quality.fairness import (
    estimate_skin_tone,
    individual_typology_angle,
    stratified_error,
)
from argus.quality.motion_index import FrameMotionIndex, motion_quality


def test_skin_tone_light_vs_dark():
    light = estimate_skin_tone((240, 205, 185))
    dark = estimate_skin_tone((85, 60, 50))
    assert light.fitzpatrick <= 2 and "light" in light.label
    assert dark.fitzpatrick >= 5
    # lighter skin → higher ITA°
    assert individual_typology_angle((240, 205, 185)) > individual_typology_angle((85, 60, 50))
    assert 1 <= light.fitzpatrick <= 6 and 1 <= dark.fitzpatrick <= 6


def test_stratified_error_disparity():
    samples = [(2, 4.0), (2, 6.0), (6, 12.0), (6, 16.0)]  # ~2-3× worse for Fitz VI
    rep = stratified_error(samples)
    assert rep[2]["n"] == 2 and rep[2]["mae"] == 5.0
    assert rep[6]["mae"] == 14.0
    assert rep["_disparity"] == 14.0 / 5.0


def test_motion_quality_monotonic():
    assert motion_quality(0.0) == 1.0
    assert motion_quality(0.1) < motion_quality(0.02) < 1.0


def test_frame_motion_index_still_vs_moving():
    idx = FrameMotionIndex()
    a = np.full((40, 40), 120, np.uint8)
    assert idx.update(a) == 1.0          # first frame → still
    assert idx.update(a) == 1.0          # identical → still, quality 1
    rng = np.random.default_rng(0)
    b = (a.astype(int) + rng.integers(-60, 60, a.shape)).clip(0, 255).astype(np.uint8)
    q = idx.update(b)
    assert idx.energy > 0.0 and q < 1.0  # motion lowers quality


def test_ceilings_present_for_all_signals():
    for name in ("hr", "hrv", "resp", "blink", "perclos", "fidget", "posture", "gaze", "affect"):
        c = ceiling_for(name)
        assert c is not None and {"best", "typical", "note"} <= set(c)
    assert ceiling_for("does-not-exist") is None
    assert len(SIGNAL_CEILINGS) >= 9
