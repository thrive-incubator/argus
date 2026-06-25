"""OpenCV Haar-cascade face backbone for live rPPG without MediaPipe.

MediaPipe needs Python 3.11 + numpy<2; this adapter uses only OpenCV (already a dependency)
so the live rPPG path runs on a stock laptop today. It produces a ``FaceResult`` whose
forehead/cheek/eye landmark indices (the ones the ROI + gate use) are placed from the
detected face box. The Haar ``detectMultiScale`` call is the untested device line.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .face import DEFAULT_NUM_FACES  # noqa: F401  (kept for parity)
from .types import N_BLENDSHAPES, N_FACE_LANDMARKS, N_IRIS, FaceResult
from ..dsp.roi import FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX
from ..quality.gate_inputs import LEFT_EYE_OUTER, RIGHT_EYE_OUTER


def bbox_to_landmarks(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> np.ndarray:
    """Place the ROI/gate landmark indices from a face bounding box (normalised coords).

    Pure function (testable). Only the indices the pipeline reads are set; the rest are 0.
    """
    lm = np.zeros((N_FACE_LANDMARKS, 3), dtype=float)

    def put(idx, fx, fy):
        lm[idx, 0] = (x + fx * w) / frame_w
        lm[idx, 1] = (y + fy * h) / frame_h

    put(FOREHEAD_IDX, 0.50, 0.15)   # forehead / glabella
    put(LEFT_CHEEK_IDX, 0.30, 0.62)  # left cheek
    put(RIGHT_CHEEK_IDX, 0.70, 0.62)  # right cheek
    put(LEFT_EYE_OUTER, 0.25, 0.40)
    put(RIGHT_EYE_OUTER, 0.75, 0.40)
    return lm


class CascadeFaceBackbone:
    """Real cv2 Haar-cascade face backbone (FaceBackbone protocol)."""

    def __init__(self, cascade_path: str | None = None):
        path = cascade_path or os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():  # pragma: no cover - misconfig
            raise RuntimeError(f"could not load Haar cascade at {path}")
        self._last_ts = float("-inf")

    def process(self, frame: np.ndarray, ts: float) -> FaceResult | None:
        if ts <= self._last_ts:
            ts = self._last_ts + 1e-6
        self._last_ts = ts
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # device path (real frame)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # largest face
        fh, fw = frame.shape[:2]
        return FaceResult(
            landmarks=bbox_to_landmarks(x, y, w, h, fw, fh),
            iris=np.zeros((N_IRIS, 3)),
            blendshapes=np.zeros(N_BLENDSHAPES),
            head_pose=np.eye(4),
            ts=ts,
        )
