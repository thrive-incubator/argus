"""H2.AC1/AC2 gate inputs (FM/FSM/solvePnP) + FR-15/FR-23/H3 covariates."""

import numpy as np
import pytest

import cv2

from argus.quality.covariates import (
    CovariateExtractor,
    Fitzpatrick,
    SessionCovariates,
    cheek_reflectance_estimate,
)
from argus.quality.gate_inputs import (
    CANONICAL_3D,
    fsm,
    gate_motion_magnitude,
    head_pose_angles,
    inter_ocular_distance,
    landmark_motion,
)
from argus.contracts import FrameContext


def _lms(scale=1.0, shift=0.0):
    lm = np.zeros((478, 3))
    lm[33] = [100, 200, 0]  # left eye outer
    lm[263] = [100 + 80 * scale, 200, 0]  # right eye outer (IOD ~ 80*scale)
    lm[:10] = (np.arange(30).reshape(10, 3) + shift) * scale
    return lm


# H2.AC2 — inter-ocular normalization.
def test_inter_ocular_distance_and_motion_normalization():
    lm = _lms()
    assert inter_ocular_distance(lm) == pytest.approx(80.0)
    prev = lm.copy()
    cur = lm.copy()
    cur[:, 0] += 8.0  # shift all landmarks 8 px in x
    fm_x, fm_y, fm = landmark_motion(prev, cur, inter_ocular_distance(lm))
    assert fm_x == pytest.approx(0.1)  # 8 px / 80 px IOD
    assert fm_y == pytest.approx(0.0)


def test_fsm_area_change():
    prev = np.zeros((478, 3))
    prev[:4, :2] = [[0, 0], [0, 10], [10, 0], [10, 10]]
    cur = prev.copy()
    cur[:4, :2] = [[0, 0], [0, 20], [10, 0], [10, 20]]  # height doubles -> area 100->200
    assert fsm(prev, cur) == pytest.approx(1.0)


# H2.AC1 — solvePnP head pose runs and yaw grows with rotation.
def test_head_pose_yaw_increases_with_rotation():
    cam = np.array([[640, 0, 320.0], [0, 640, 240.0], [0, 0, 1]], np.float64)
    tvec = np.array([[0.0], [0.0], [400.0]])

    def project(yaw_deg):
        rvec, _ = cv2.Rodrigues(
            cv2.Rodrigues(np.array([0.0, np.radians(yaw_deg), 0.0]))[0]
        )
        pts, _ = cv2.projectPoints(CANONICAL_3D, rvec, tvec, cam, None)
        return pts.reshape(-1, 2)

    yaw_small = abs(head_pose_angles(project(5.0), cam)[1])
    yaw_big = abs(head_pose_angles(project(30.0), cam)[1])
    assert yaw_big > yaw_small


def test_gate_motion_magnitude_weights_pitch():
    # equal-magnitude pitch vs yaw -> pitch contributes more (×2)
    m_pitch = gate_motion_magnitude(0, 0, pitch_deg=20, yaw_deg=0)
    m_yaw = gate_motion_magnitude(0, 0, pitch_deg=0, yaw_deg=20)
    assert m_pitch > m_yaw


# FR-23 / H3.AC2 — session covariates recorded.
def test_session_covariates():
    cov = SessionCovariates(fitzpatrick=Fitzpatrick.IV, eyewear=True, facial_hair=False)
    assert cov.fitzpatrick == 4 and cov.eyewear and not cov.facial_hair


def test_cheek_reflectance_descriptive():
    bright = np.full((8, 8, 3), 255, np.uint8)
    assert cheek_reflectance_estimate(bright) == pytest.approx(1.0)


# FR-15 / H3.AC1 — covariate streams (lighting, presence).
def test_covariate_extractor_streams():
    ext = CovariateExtractor(lux_provider=lambda: 300.0)
    frame = np.full((8, 8, 3), 128, np.uint8)

    class DummyFace:
        pass

    recs = ext.consume(FrameContext(frame=frame, ts=1.0, frame_id=0, face=DummyFace()))
    names = {r.name for r in recs}
    assert "lighting_index" in names and "face_presence" in names
    lighting = next(r for r in recs if r.name == "lighting_index")
    assert lighting.meta["not_lux"] is True and lighting.meta["measured_lux"] == 300.0
    presence = next(r for r in recs if r.name == "face_presence")
    assert presence.value == 1.0


def test_covariate_face_absent_presence_zero():
    ext = CovariateExtractor()
    recs = ext.consume(FrameContext(frame=np.zeros((4, 4, 3), np.uint8), ts=1.0,
                                    frame_id=0, face=None))
    presence = next(r for r in recs if r.name == "face_presence")
    assert presence.value == 0.0
