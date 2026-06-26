"""D1 (ROI) + FR-13/H1 (skewness, perfusion, Orphanidou bSQI)."""

import numpy as np
import pytest

from argus.dsp.roi import (
    STANDARD_PATCHES,
    active_patch_count,
    roi_mean_rgb,
    roi_patch_stack,
    sample_patch_mean,
)
from argus.dsp.sqi import (
    beat_template_correlations,
    fuse_patches_by_snr,
    orphanidou_bsqi,
    perfusion_index,
    skewness_sqi,
)


def _landmarks():
    lm = np.zeros((478, 3))
    lm[9] = [0.5, 0.22, 0]  # glabella (between the eyebrows)
    lm[151] = [0.5, 0.30, 0]  # forehead
    lm[50] = [0.40, 0.55, 0]  # left cheek
    lm[280] = [0.60, 0.55, 0]  # right cheek
    return lm


# D1.AC1 — multi-patch mean over glabella + forehead + cheeks.
def test_roi_multipatch_mean():
    frame = np.full((200, 200, 3), [100, 150, 50], np.uint8)
    mean = roi_mean_rgb(frame, _landmarks())
    assert mean == pytest.approx([100, 150, 50])


# D1.AC2 — yaw beyond cutoff drops a cheek patch; glabella+forehead+one cheek remain.
def test_roi_yaw_drops_cheek():
    assert active_patch_count(0.0) == 4  # glabella + forehead + both cheeks
    assert active_patch_count(40.0, yaw_cutoff=25.0) == 3  # right-turned drops left cheek
    assert active_patch_count(-40.0, yaw_cutoff=25.0) == 3
    frame = np.full((200, 200, 3), [10, 20, 30], np.uint8)
    mean = roi_mean_rgb(frame, _landmarks(), yaw_deg=40.0)
    assert mean == pytest.approx([10, 20, 30])  # still valid


def test_roi_facial_hair_upper_face_only():
    # facial hair drops both cheeks, leaving glabella + forehead.
    assert active_patch_count(0.0, facial_hair=True) == 2


def test_roi_patch_stack_shape_and_order():
    frame = np.full((120, 120, 3), [30, 40, 50], np.uint8)
    stack = roi_patch_stack(frame, _landmarks())
    assert stack.shape == (len(STANDARD_PATCHES), 3)
    assert stack == pytest.approx(np.full((4, 3), [30, 40, 50]))


# Review §1 — per-patch SNR fusion down-weights a corrupted patch and recovers HR.
def test_fuse_patches_by_snr_downweights_noise():
    fps, hr = 30.0, 72.0
    n = int(fps * 12)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * (hr / 60.0) * t)
    base = np.array([0.8, 0.5, 0.4])
    amp = np.array([0.01, 0.03, 0.005])
    rng = np.random.default_rng(0)
    clean = base + amp * pulse[:, None] + 0.002 * rng.standard_normal((n, 3))
    noisy = base + 0.20 * rng.standard_normal((n, 3))  # no pulse, heavy noise
    bvp, info = fuse_patches_by_snr({"glabella": clean, "left_cheek": noisy}, fps)
    assert info["fallback"] is False
    # the clean patch must carry the majority of the fused weight
    assert info["weights"]["glabella"] > info["weights"]["left_cheek"]
    from argus.dsp.rppg import hr_from_bvp
    assert abs(hr_from_bvp(bvp, fps) - hr) <= 5.0


def test_fuse_patches_fallback_when_no_patch_scores():
    # too-short series → POS raises for every patch → graceful mean fallback, no crash.
    fps = 30.0
    tiny = np.ones((3, 3))
    bvp, info = fuse_patches_by_snr({"a": tiny, "b": tiny}, fps)
    assert info["fallback"] is True
    assert len(bvp) == tiny.shape[0]


def test_fuse_patches_single_patch_is_finite():
    fps, hr = 30.0, 66.0
    n = int(fps * 12)
    t = np.arange(n) / fps
    rgb = np.array([0.8, 0.5, 0.4]) + np.array([0.01, 0.03, 0.005]) * np.sin(
        2 * np.pi * (hr / 60.0) * t)[:, None]
    bvp, info = fuse_patches_by_snr({"glabella": rgb}, fps)
    assert info["weights"]["glabella"] == pytest.approx(1.0)
    assert np.isfinite(bvp).all()


# D1.AC3 — landmark jitter doesn't spike the ROI mean (patch averaging) on a flat frame.
def test_roi_stable_under_landmark_jitter():
    frame = np.full((300, 300, 3), [80, 90, 100], np.uint8)
    rng = np.random.default_rng(0)
    means = []
    for _ in range(20):
        lm = _landmarks()
        lm[[151, 50, 280], :2] += rng.normal(0, 0.01, (3, 2))  # jitter landmarks
        means.append(roi_mean_rgb(frame, lm))
    means = np.array(means)
    assert means.std(axis=0).max() < 1.0  # essentially constant


def test_sample_patch_clips_to_frame():
    frame = np.full((50, 50, 3), 7, np.uint8)
    assert sample_patch_mean(frame, 0.0, 0.0) == pytest.approx([7, 7, 7])
    assert sample_patch_mean(frame, 1.0, 1.0) == pytest.approx([7, 7, 7])


# H1.AC2 — skewness, perfusion, Orphanidou bSQI.
def test_skewness_and_perfusion():
    fps = 256.0
    t = np.arange(int(fps * 4)) / fps
    clean = np.sin(2 * np.pi * 1.2 * t)
    assert abs(skewness_sqi(clean)) < 0.2  # sinusoid ~ symmetric
    assert perfusion_index(clean + 5.0) == pytest.approx(2.0 / 5.0, abs=0.05)


def test_orphanidou_bsqi_clean_vs_irregular():
    fps = 256.0
    t = np.arange(int(fps * 6)) / fps
    clean = np.sin(2 * np.pi * 1.2 * t)
    from argus.dsp.hrv import detect_peaks

    peaks = (detect_peaks(clean, fps) * fps).astype(int)
    frac, accepted, corrs = orphanidou_bsqi(clean, fps, peaks, threshold=0.86)
    assert frac > 0.8  # clean periodic beats accepted
    assert corrs.size >= 3

    rng = np.random.default_rng(1)
    noisy = rng.standard_normal(t.size)
    npks = (detect_peaks(noisy, fps) * fps).astype(int)
    nfrac, _, _ = orphanidou_bsqi(noisy, fps, npks, threshold=0.86)
    assert nfrac < frac  # irregular noise accepts fewer beats


def test_bsqi_too_few_beats_returns_empty():
    assert beat_template_correlations(np.zeros(10), 30.0, np.array([1, 2])).size == 0
