"""FR-2, FR-3, A2.AC1-3, A3.AC1-3 (backbone + One-Euro)."""

import numpy as np
import pytest

from argus.backbone.face import SyntheticFaceBackbone
from argus.backbone.oneeuro import OneEuroFilter, filter_series
from argus.backbone.pose import USES_HOLISTIC, SyntheticPoseBackbone
from argus.backbone.types import (
    N_BLENDSHAPES,
    N_FACE_LANDMARKS,
    N_IRIS,
    N_POSE_LANDMARKS,
    FaceResult,
)


# A2.AC1 — one pass produces 478 landmarks, 10 iris, 52 blendshapes, 4x4 head pose.
def test_face_backbone_shapes():
    bb = SyntheticFaceBackbone()
    frame = np.full((8, 8, 3), 50, np.uint8)
    res = bb.process(frame, ts=0.1)
    assert isinstance(res, FaceResult)
    assert res.landmarks.shape == (N_FACE_LANDMARKS, 3)
    assert res.iris.shape == (N_IRIS, 3)
    assert res.blendshapes.shape == (N_BLENDSHAPES,)
    assert res.head_pose.shape == (4, 4)
    assert bb.calls == 1  # exactly one pass


# A2.AC2 — monotonic timestamps enforced.
def test_face_backbone_requires_monotonic_ts():
    bb = SyntheticFaceBackbone()
    frame = np.full((4, 4, 3), 50, np.uint8)
    bb.process(frame, ts=1.0)
    with pytest.raises(ValueError):
        bb.process(frame, ts=1.0)  # not strictly increasing


# A2.AC3 — no face -> None.
def test_face_backbone_no_face_returns_none():
    bb = SyntheticFaceBackbone()
    black = np.zeros((4, 4, 3), np.uint8)
    assert bb.process(black, ts=0.5) is None


def test_face_result_rejects_wrong_shapes():
    with pytest.raises(AssertionError):
        FaceResult(
            landmarks=np.zeros((10, 3)),
            iris=np.zeros((N_IRIS, 3)),
            blendshapes=np.zeros(N_BLENDSHAPES),
            head_pose=np.eye(4),
            ts=0.0,
        )


# A3.AC1 — separate Pose task, Holistic not used.
def test_pose_is_separate_task_not_holistic():
    assert USES_HOLISTIC is False


# A3.AC3 — per-landmark visibility exposed; occluded lower body has low visibility.
def test_pose_visibility_exposed():
    bb = SyntheticPoseBackbone()
    res = bb.process(np.zeros((4, 4, 3), np.uint8), ts=0.0)
    assert res.landmarks.shape == (N_POSE_LANDMARKS, 3)
    assert res.visibility.shape == (N_POSE_LANDMARKS,)
    assert res.visibility[11] > 0.5 and res.visibility[27] < 0.5  # shoulder vs ankle


# A3.AC2 — One-Euro reduces jitter on a static-subject (constant + noise) clip.
def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n) / 30.0
    raw = 1.0 + 0.05 * rng.standard_normal(n)  # static value + jitter
    filt = filter_series(t, raw, min_cutoff=0.5, beta=0.001)
    # compare on the settled tail
    assert filt[50:].std() < raw[50:].std() * 0.6


def test_one_euro_tracks_a_ramp():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    out = [f(i / 30.0, float(i)) for i in range(100)]
    # follows the ramp with the inherent (bounded) low-pass lag
    assert out[-1] == pytest.approx(99.0, abs=4.0)
    assert out[-1] > out[0]  # actually tracks upward, not frozen
