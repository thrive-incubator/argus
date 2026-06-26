"""R14–R16: respiration (Indicative) + blink/EAR/PERCLOS."""

import numpy as np
import pytest

from argus.dsp.blink import AdaptiveBlinkDetector, eye_aspect_ratio, perclos
from argus.dsp.respiration import (
    ChestFlowRespiration,
    farneback_vertical_motion,
    resp_band_fraction,
    respiration_rate,
)


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


# Review §3 — chest-ROI optical flow recovers RR from a vertically translating texture.
def _striped(shift_px: float, h=80, w=80):
    """A horizontally-striped frame translated vertically by ``shift_px`` (sub-pixel)."""
    y = np.arange(h)[:, None].astype(np.float64)
    img = 128 + 100 * np.sin(2 * np.pi * (y - shift_px) / 10.0)
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8), w, axis=1).reshape(h, w)


def test_farneback_vertical_motion_sign():
    # a rich (non-periodic) texture so the flow is unambiguous
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (80, 80), dtype=np.uint8)
    base = np.asarray(base, dtype=np.uint8)
    a = base
    b = np.roll(base, 4, axis=0)  # whole texture moved down 4 px
    dy = farneback_vertical_motion(a, b)
    assert dy > 0.5  # detects downward motion


def test_chest_flow_respiration_recovers_rate():
    fps, brpm, secs = 30.0, 15.0, 22.0
    f0 = brpm / 60.0
    flow = ChestFlowRespiration(fps, window_s=20.0, min_window_s=15.0)
    est = None
    n = int(fps * secs)
    for i in range(n):
        t = i / fps
        shift = 3.0 * np.sin(2 * np.pi * f0 * t)  # breathing translation, ±3 px
        frame = _striped(shift)
        flow.update(t, frame, roi=(0, 0, 80, 80))
        est = flow.estimate()
    assert est is not None
    rr, sqi = est
    assert abs(rr - brpm) <= 2.0
    assert sqi > 0.2


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


# Review §4 — graded eye-openness fuses EAR with the eyeBlink blendshape.
def test_eye_openness_graded_and_fused():
    from argus.dsp.blink import eye_openness
    assert eye_openness(0.30, 0.30) == pytest.approx(1.0)        # fully open
    assert eye_openness(0.0, 0.30) == pytest.approx(0.0)         # fully closed
    # blendshape says closed (0.9) drags a high-EAR openness down
    fused = eye_openness(0.30, 0.30, blink_score=0.9)
    assert fused < 0.6 and fused == pytest.approx((1.0 + 0.1) / 2.0)


# Review §4 — P80 excludes short blinks but counts sustained eyelid droop.
def test_perclos_p80_excludes_blinks_counts_droop():
    from argus.dsp.blink import PerclosP80

    fps = 30.0
    # short blinks: 3-frame closures (~100 ms) every second → should NOT inflate P80
    blink = PerclosP80(fps=fps, window_s=30.0, blink_max_s=0.4)
    t = 0.0
    for _ in range(20):
        for _f in range(27):  # open
            blink.update(t, 1.0); t += 1 / fps
        for _f in range(3):   # brief blink (closed)
            blink.update(t, 0.0); t += 1 / fps
    assert blink.value() < 0.05  # blinks excluded

    # sustained droop: half the window eyes ≥80% closed for long stretches → high P80
    droop = PerclosP80(fps=fps, window_s=30.0, blink_max_s=0.4)
    t = 0.0
    for _f in range(int(fps * 15)):
        droop.update(t, 1.0); t += 1 / fps
    for _f in range(int(fps * 15)):
        droop.update(t, 0.0); t += 1 / fps  # 15 s sustained closure
    assert droop.value() > 0.4
