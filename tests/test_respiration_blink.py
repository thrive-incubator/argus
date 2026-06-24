"""R14–R16: respiration (Indicative) + blink/EAR/PERCLOS."""

import numpy as np
import pytest

from argus.dsp.blink import AdaptiveBlinkDetector, eye_aspect_ratio, perclos
from argus.dsp.respiration import respiration_rate


# R14 — recovers a known breathing rate (Indicative: a recovery check, not accuracy bar)
@pytest.mark.parametrize("brpm", [6.0, 12.0, 18.0])
def test_respiration_recovers_known_rate(brpm):
    fps = 30.0
    t = np.arange(int(fps * 40)) / fps
    f0 = brpm / 60.0
    disp = np.sin(2 * np.pi * f0 * t) + 0.05 * t  # breathing + slow drift
    est = respiration_rate(disp, fps)
    assert abs(est - brpm) <= 1.5


def test_respiration_needs_enough_samples():
    with pytest.raises(ValueError):
        respiration_rate(np.zeros(4), fps=30.0)


# R15 — EAR open vs closed, and adaptive blink detection.
def test_ear_open_greater_than_closed():
    open_eye = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]], float)
    closed_eye = np.array([[0, 0], [1, 0.1], [2, 0.1], [3, 0], [2, -0.1], [1, -0.1]], float)
    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed_eye)


def test_ear_rejects_bad_landmark_count():
    with pytest.raises(ValueError):
        eye_aspect_ratio(np.zeros((4, 2)))


def test_adaptive_detector_unarmed_until_baseline():
    det = AdaptiveBlinkDetector(baseline_frames=50, ratio=0.6, min_frames=2)
    for _ in range(49):
        det.update(0.30)
    assert not det.armed  # not enough baseline yet -> no threshold
    det.update(0.30)
    assert det.armed
    assert det.threshold == pytest.approx(0.18)


def test_adaptive_detector_counts_blinks():
    det = AdaptiveBlinkDetector(baseline_frames=30, ratio=0.6, min_frames=2)
    series = [0.30] * 30  # baseline (open)
    # three blinks: dip below 0.18 for 3 frames, separated by open frames
    for _ in range(3):
        series += [0.10, 0.10, 0.10, 0.30, 0.30]
    for ear in series:
        det.update(ear)
    assert det.blink_count == 3


# R16 — PERCLOS increases with induced eye closure.
def test_perclos_increases_with_closure():
    baseline = 0.30
    mostly_open = np.full(100, 0.30)
    half_closed = np.concatenate([np.full(50, 0.30), np.full(50, 0.03)])
    assert perclos(mostly_open, baseline) == pytest.approx(0.0)
    assert perclos(half_closed, baseline) == pytest.approx(0.5)
