"""Motion-gate input features computed from FaceMesh landmarks (H2.AC1).

- ``FM_X``/``FM_Y``: mean inter-frame landmark displacement, inter-ocular normalised (H2.AC2).
- ``FSM``: ROI-area change (≈ z-motion).
- head pose via ``cv2.solvePnP`` on canonical landmarks; pitch (nodding) weighted ×2.
"""

from __future__ import annotations

import cv2
import numpy as np

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# A small canonical 3D face model (arbitrary mm units) for solvePnP.
CANONICAL_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -63.0, -12.0],  # chin
        [-43.0, 32.0, -26.0],  # left eye outer
        [43.0, 32.0, -26.0],  # right eye outer
        [-28.0, -28.0, -22.0],  # left mouth
        [28.0, -28.0, -22.0],  # right mouth
    ],
    dtype=np.float64,
)


def inter_ocular_distance(landmarks: np.ndarray) -> float:
    lm = np.asarray(landmarks, dtype=float)
    d = np.linalg.norm(lm[LEFT_EYE_OUTER, :2] - lm[RIGHT_EYE_OUTER, :2])
    return float(max(d, 1e-9))


def landmark_motion(prev: np.ndarray, cur: np.ndarray, iod: float):
    """Return ``(fm_x, fm_y, fm)`` — mean per-axis and total displacement / IOD."""
    prev = np.asarray(prev, dtype=float)[:, :2]
    cur = np.asarray(cur, dtype=float)[:, :2]
    d = cur - prev
    fm_x = float(np.abs(d[:, 0]).mean()) / iod
    fm_y = float(np.abs(d[:, 1]).mean()) / iod
    fm = float(np.linalg.norm(d, axis=1).mean()) / iod
    return fm_x, fm_y, fm


def roi_area(landmarks: np.ndarray) -> float:
    """Bounding-box area of the face region (proxy for scale / z-motion)."""
    lm = np.asarray(landmarks, dtype=float)[:, :2]
    span = lm.max(axis=0) - lm.min(axis=0)
    return float(span[0] * span[1])


def fsm(prev_lms: np.ndarray, cur_lms: np.ndarray) -> float:
    """Fractional ROI-area change between frames."""
    a0 = roi_area(prev_lms)
    a1 = roi_area(cur_lms)
    return float(abs(a1 - a0) / a0) if a0 > 1e-12 else 0.0


def _euler_from_R(R: np.ndarray):
    """Return (pitch, yaw, roll) degrees from a rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:  # pragma: no cover - gimbal lock edge
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def head_pose_angles(image_pts: np.ndarray, camera_matrix: np.ndarray | None = None):
    """solvePnP head pose → (pitch, yaw, roll) degrees from 6 canonical image points."""
    image_pts = np.asarray(image_pts, dtype=np.float64)[:, :2]
    if camera_matrix is None:
        f = 640.0
        camera_matrix = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        CANONICAL_3D, image_pts, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    R, _ = cv2.Rodrigues(rvec)
    return _euler_from_R(R)


def gate_motion_magnitude(fm: float, fsm_val: float, pitch_deg: float, yaw_deg: float,
                          pitch_weight: float = 2.0) -> float:
    """Combine input-side features into a single normalised motion magnitude.

    Pitch (nodding) hurts rPPG ~2× more than yaw, so it is up-weighted.
    """
    pose_term = (pitch_weight * abs(pitch_deg) + abs(yaw_deg)) / 90.0
    return float(fm + fsm_val + 0.5 * pose_term)
