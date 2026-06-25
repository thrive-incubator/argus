"""Pose backbone (FR-3, A3). Separate Task (NOT Holistic), One-Euro filtered.

- ``MediaPipePoseBackbone``: real adapter (lazy ``mediapipe``); a *separate* PoseLandmarker
  Task — Holistic is explicitly not used (A3.AC1). Inference is the untested device line.
- ``SyntheticPoseBackbone``: deterministic pose with per-landmark visibility, One-Euro
  filtered (A3.AC2/AC3).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .oneeuro import LandmarkOneEuro
from .types import N_POSE_LANDMARKS, PoseResult

USES_HOLISTIC = False  # A3.AC1 — a separate Pose Task, never Holistic


class PoseBackbone(Protocol):
    def process(self, frame: np.ndarray, ts: float) -> PoseResult | None: ...


class MediaPipePoseBackbone:
    def __init__(self, model_path: str):
        import mediapipe as mp  # local import: device/model only

        base = mp.tasks.BaseOptions(model_asset_path=model_path)
        self._options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(self._options)
        self._filter = LandmarkOneEuro()

    def process(self, frame, ts):  # pragma: no cover - device/model inference
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 BGR -> MediaPipe RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(ts * 1000))
        if not result.pose_world_landmarks:
            return None
        lms = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
        vis = np.array([p.visibility for p in result.pose_world_landmarks[0]])
        img = None
        if result.pose_landmarks:  # normalised image-space coords for overlays
            img = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks[0]])
        return PoseResult(landmarks=self._filter(ts, lms), visibility=vis, ts=ts,
                          image_landmarks=img)


class SyntheticPoseBackbone:
    """Synthetic pose with an injectable chest-displacement signal for respiration tests."""

    def __init__(self, chest_signal=None, filtered: bool = True):
        self._filter = LandmarkOneEuro() if filtered else None
        self._chest = chest_signal  # callable ts -> vertical displacement, or None
        self._last_ts = float("-inf")

    def process(self, frame: np.ndarray, ts: float) -> PoseResult | None:
        self._last_ts = ts
        rng = np.random.default_rng(int(ts * 1000) % (2**32))
        lms = rng.normal(0, 0.001, (N_POSE_LANDMARKS, 3))
        if self._chest is not None:
            # shoulders (11, 12) carry the breathing displacement on the y axis
            disp = float(self._chest(ts))
            lms[11, 1] = disp
            lms[12, 1] = disp
        if self._filter is not None:
            lms = self._filter(ts, lms)
        vis = np.full(N_POSE_LANDMARKS, 0.99)
        vis[25:] = 0.2  # occluded lower body behind a desk (A3.AC3)
        return PoseResult(landmarks=lms, visibility=vis, ts=ts)
