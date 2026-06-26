"""rPPG ROI extraction (D1, ADR-01; algorithm-review §1).

Multi-patch mean over glabella + forehead + cheeks, excluding eyes/mouth/brows. The
**glabella** (between the eyebrows) is included and listed first: in the 28-region study
(npj Cardiovascular Health 2024) it ranked #1 on MAE/PCC/SNR under motion and cognitive
load, while the lower cheek degrades most under motion.

Two consumption modes:
- ``roi_mean_rgb``: the plain multi-patch mean (used by the HRV/respiration paths).
- ``roi_patch_stack`` + ``fuse_patches_by_snr`` (in :mod:`argus.dsp.sqi`): per-patch
  series fused by signal quality, so an occluded/motion-corrupted patch is down-weighted
  rather than hard-dropped (review §1: "per-patch SNR weighting + yaw fallback").

Yaw beyond the cutoff drops the occluded cheek patch (D1.AC2); facial hair drops both
cheeks, leaving glabella/forehead only (ADR-03 rev 1). Patch averaging absorbs landmark
jitter (D1.AC3).
"""

from __future__ import annotations

import numpy as np

# Representative MediaPipe FaceMesh indices (R2I-rPPG): glabella, forehead, cheeks.
GLABELLA_IDX = 9  # between the eyebrows — highest-SNR rPPG region (npj Cardiovascular Health 2024)
FOREHEAD_IDX = 151
LEFT_CHEEK_IDX = 50
RIGHT_CHEEK_IDX = 280

# Canonical patch order for the per-patch SNR-weighted stack (upper-face first).
STANDARD_PATCHES = ("glabella", "forehead", "left_cheek", "right_cheek")
_PATCH_IDX = {
    "glabella": GLABELLA_IDX,
    "forehead": FOREHEAD_IDX,
    "left_cheek": LEFT_CHEEK_IDX,
    "right_cheek": RIGHT_CHEEK_IDX,
}


def sample_patch_mean(frame: np.ndarray, cx_norm: float, cy_norm: float, half: int = 10):
    """Mean RGB over a square patch centred at a normalised (x, y) landmark position."""
    h, w = frame.shape[:2]
    cx = int(np.clip(cx_norm * w, half, w - half - 1))
    cy = int(np.clip(cy_norm * h, half, h - half - 1))
    patch = frame[cy - half : cy + half + 1, cx - half : cx + half + 1, :3]
    return patch.reshape(-1, 3).mean(axis=0)


def active_patches(
    yaw_deg: float = 0.0,
    yaw_cutoff: float = 25.0,
    facial_hair: bool = False,
) -> list[str]:
    """Labels of the ROI patches active under the given pose/appearance conditions.

    Glabella + forehead are always active (upper face, unaffected by facial hair or modest
    yaw). The cheek on the side turned away from the camera is dropped by yaw; facial hair
    drops both cheeks.
    """
    labels = ["glabella", "forehead"]
    if not facial_hair:
        if yaw_deg <= yaw_cutoff:  # right cheek visible
            labels.append("right_cheek")
        if yaw_deg >= -yaw_cutoff:  # left cheek visible
            labels.append("left_cheek")
    return labels


def roi_patch_means(
    frame: np.ndarray,
    landmarks: np.ndarray,
    yaw_deg: float = 0.0,
    yaw_cutoff: float = 25.0,
    facial_hair: bool = False,
    half: int = 10,
) -> dict[str, np.ndarray]:
    """Per-patch mean RGB for the *active* ROI patches → ``{label: rgb(3,)}``."""
    lm = np.asarray(landmarks, dtype=float)
    out: dict[str, np.ndarray] = {}
    for label in active_patches(yaw_deg, yaw_cutoff, facial_hair):
        c = lm[_PATCH_IDX[label]]
        out[label] = sample_patch_mean(frame, c[0], c[1], half)
    return out


def roi_patch_stack(
    frame: np.ndarray, landmarks: np.ndarray, half: int = 10
) -> np.ndarray:
    """Stacked mean RGB of ALL standard patches, in ``STANDARD_PATCHES`` order → (4, 3).

    Unlike :func:`roi_patch_means`, this never drops a patch — occluded/turned patches are
    sampled anyway and left for the SNR-weighted fusion to down-weight. This keeps a fixed
    feature width for the rolling buffer.
    """
    lm = np.asarray(landmarks, dtype=float)
    return np.stack(
        [sample_patch_mean(frame, lm[_PATCH_IDX[l]][0], lm[_PATCH_IDX[l]][1], half)
         for l in STANDARD_PATCHES]
    )


def roi_mean_rgb(
    frame: np.ndarray,
    landmarks: np.ndarray,
    yaw_deg: float = 0.0,
    yaw_cutoff: float = 25.0,
    facial_hair: bool = False,
    half: int = 10,
) -> np.ndarray:
    """Return the multi-patch mean RGB of the active ROI patches (glabella + forehead + cheeks).

    Always includes glabella + forehead. Cheeks are included unless dropped by yaw (the
    cheek on the side turned away) or by facial hair (both cheeks dropped).
    """
    means = list(
        roi_patch_means(frame, landmarks, yaw_deg, yaw_cutoff, facial_hair, half).values()
    )
    return np.mean(means, axis=0)


def active_patch_count(yaw_deg: float, yaw_cutoff: float = 25.0, facial_hair: bool = False) -> int:
    """How many ROI patches are active under the given conditions (for diagnostics/tests)."""
    return len(active_patches(yaw_deg, yaw_cutoff, facial_hair))
