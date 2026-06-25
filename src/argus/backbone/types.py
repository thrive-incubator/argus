"""Backbone result types (A2, A3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FACE_LANDMARKS = 478
N_IRIS = 10
N_BLENDSHAPES = 52
N_POSE_LANDMARKS = 33


@dataclass(frozen=True)
class FaceResult:
    """One MediaPipe Face Landmarker result (A2.AC1)."""

    landmarks: np.ndarray  # (478, 3)
    iris: np.ndarray  # (10, 3)
    blendshapes: np.ndarray  # (52,)
    head_pose: np.ndarray  # (4, 4) transformation matrix
    ts: float

    def __post_init__(self) -> None:
        assert self.landmarks.shape == (N_FACE_LANDMARKS, 3), "expected 478 landmarks"
        assert self.iris.shape == (N_IRIS, 3), "expected 10 iris points"
        assert self.blendshapes.shape == (N_BLENDSHAPES,), "expected 52 blendshapes"
        assert self.head_pose.shape == (4, 4), "expected 4x4 head pose"


@dataclass(frozen=True)
class PoseResult:
    """One MediaPipe Pose result with per-landmark visibility (A3.AC1/AC3).

    ``image_landmarks`` (optional) are normalised image-space coords for drawing overlays.
    """

    landmarks: np.ndarray  # (33, 3) world landmarks
    visibility: np.ndarray  # (33,) in [0, 1]
    ts: float
    image_landmarks: np.ndarray | None = None  # (33, 3) normalised image coords

    def __post_init__(self) -> None:
        assert self.landmarks.shape == (N_POSE_LANDMARKS, 3)
        assert self.visibility.shape == (N_POSE_LANDMARKS,)
