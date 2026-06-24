"""rPPG ROI extraction (D1, ADR-01).

Multi-patch mean over forehead + cheeks, excluding eyes/mouth/brows. Yaw beyond the
cutoff drops the occluded cheek patch (D1.AC2); facial hair drops both cheeks, leaving
forehead/glabella only (ADR-03 rev 1). Patch averaging absorbs landmark jitter (D1.AC3).
"""

from __future__ import annotations

import numpy as np

# Representative MediaPipe FaceMesh indices (R2I-rPPG): forehead/glabella, cheeks.
FOREHEAD_IDX = 151
LEFT_CHEEK_IDX = 50
RIGHT_CHEEK_IDX = 280


def sample_patch_mean(frame: np.ndarray, cx_norm: float, cy_norm: float, half: int = 10):
    """Mean RGB over a square patch centred at a normalised (x, y) landmark position."""
    h, w = frame.shape[:2]
    cx = int(np.clip(cx_norm * w, half, w - half - 1))
    cy = int(np.clip(cy_norm * h, half, h - half - 1))
    patch = frame[cy - half : cy + half + 1, cx - half : cx + half + 1, :3]
    return patch.reshape(-1, 3).mean(axis=0)


def roi_mean_rgb(
    frame: np.ndarray,
    landmarks: np.ndarray,
    yaw_deg: float = 0.0,
    yaw_cutoff: float = 25.0,
    facial_hair: bool = False,
    half: int = 10,
) -> np.ndarray:
    """Return the multi-patch mean RGB of the active ROI patches.

    Always includes the forehead. Cheeks are included unless dropped by yaw (the cheek on
    the side turned away) or by facial hair (both cheeks dropped).
    """
    lm = np.asarray(landmarks, dtype=float)
    centres = [lm[FOREHEAD_IDX]]
    if not facial_hair:
        if yaw_deg <= yaw_cutoff:  # right cheek visible
            centres.append(lm[RIGHT_CHEEK_IDX])
        if yaw_deg >= -yaw_cutoff:  # left cheek visible
            centres.append(lm[LEFT_CHEEK_IDX])
    means = [sample_patch_mean(frame, c[0], c[1], half) for c in centres]
    return np.mean(means, axis=0)


def active_patch_count(yaw_deg: float, yaw_cutoff: float = 25.0, facial_hair: bool = False) -> int:
    """How many ROI patches are active under the given conditions (for diagnostics/tests)."""
    if facial_hair:
        return 1
    n = 1  # forehead
    if yaw_deg <= yaw_cutoff:
        n += 1
    if yaw_deg >= -yaw_cutoff:
        n += 1
    return n
