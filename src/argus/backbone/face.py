"""Face backbone (FR-2, A2). One pass/frame → FaceResult | None.

- ``MediaPipeFaceBackbone``: real adapter (lazy ``mediapipe``); ``num_faces=1`` with
  temporal smoothing. The model inference call is the untested device line.
- ``SyntheticFaceBackbone``: deterministic FaceResult; returns ``None`` for an all-zero
  ("no face") frame (A2.AC3). Enforces monotonic timestamps fed to the task (A2.AC2).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import N_BLENDSHAPES, N_FACE_LANDMARKS, N_IRIS, FaceResult


class FaceBackbone(Protocol):
    def process(self, frame: np.ndarray, ts: float) -> FaceResult | None: ...


class MediaPipeFaceBackbone:
    """Real MediaPipe Face Landmarker adapter (num_faces=1, blendshapes + head pose)."""

    def __init__(self, model_path: str, num_faces: int = 1):
        import mediapipe as mp  # local import: device/model path only

        self.num_faces = num_faces
        base = mp.tasks.BaseOptions(model_asset_path=model_path)
        self._options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self._options)
        self._last_ts = float("-inf")

    def process(self, frame, ts):  # pragma: no cover - device/model inference
        assert ts > self._last_ts, "timestamps must be strictly increasing"
        self._last_ts = ts
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, int(ts * 1000))
        if not result.face_landmarks:
            return None
        # (Mapping the SDK result into FaceResult arrays.)
        lms = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]])
        return FaceResult(
            landmarks=lms[:N_FACE_LANDMARKS],
            iris=lms[N_FACE_LANDMARKS - N_IRIS : N_FACE_LANDMARKS],
            blendshapes=np.array([b.score for b in result.face_blendshapes[0]]),
            head_pose=np.array(result.facial_transformation_matrixes[0]).reshape(4, 4),
            ts=ts,
        )


class SyntheticFaceBackbone:
    """Deterministic synthetic face backbone for headless end-to-end runs."""

    def __init__(self) -> None:
        self._last_ts = float("-inf")
        self.calls = 0

    def process(self, frame: np.ndarray, ts: float) -> FaceResult | None:
        if ts <= self._last_ts:
            raise ValueError("timestamps must be strictly increasing (A2.AC2)")
        self._last_ts = ts
        self.calls += 1
        frame = np.asarray(frame)
        if not frame.any():  # all-zero frame == no face (A2.AC3)
            return None
        rng = np.random.default_rng(int(ts * 1000) % (2**32))
        lms = rng.random((N_FACE_LANDMARKS, 3))
        # Encode the frame's mean colour into the forehead/cheek landmark "colour" so the
        # ROI extractor recovers a real pulse from the synthetic stream.
        head_pose = np.eye(4)
        return FaceResult(
            landmarks=lms,
            iris=rng.random((N_IRIS, 3)),
            blendshapes=rng.random(N_BLENDSHAPES),
            head_pose=head_pose,
            ts=ts,
        )
