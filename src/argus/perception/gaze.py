"""Gaze estimation (FR-9, F2). Learned head → zones (no pixels).

- ``L2csGazeEstimator``: real ONNX adapter (lazy ``onnxruntime``); inference is the untested
  device/model line.
- ``FakeGazeEstimator``: returns configured (pitch, yaw) for headless tests.
- ``gaze_to_zone``: maps an angle to direction zones — never pixel coordinates (F2.AC2).
- ``GazeCalibration``: optional 5–9-point affine correction (F2.AC4).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class GazeEstimator(Protocol):
    def estimate(self, eye_crop) -> tuple[float, float]:
        """Return (pitch_deg, yaw_deg)."""
        ...


class L2csGazeEstimator:
    """Real L2CS-Net ONNX adapter (default model, ADR-09)."""

    def __init__(self, model_path: str, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort  # local import: model/runtime only

        self._sess = ort.InferenceSession(model_path, providers=list(providers))

    def estimate(self, eye_crop):  # pragma: no cover - model inference
        out = self._sess.run(None, {self._sess.get_inputs()[0].name: eye_crop})
        pitch, yaw = float(out[0][0]), float(out[1][0])
        return pitch, yaw


class FakeGazeEstimator:
    def __init__(self, pitch: float = 0.0, yaw: float = 0.0):
        self.pitch, self.yaw = pitch, yaw

    def estimate(self, eye_crop):
        return self.pitch, self.yaw


def gaze_to_zone(pitch_deg: float, yaw_deg: float, yaw_lr: float = 8.0,
                 screen_yaw: float = 20.0, screen_pitch: float = 20.0) -> dict:
    """Map a gaze angle to direction zones (F2.AC2 — no pixel point-of-regard)."""
    horizontal = "left" if yaw_deg < -yaw_lr else "right" if yaw_deg > yaw_lr else "center"
    on_screen = abs(yaw_deg) <= screen_yaw and abs(pitch_deg) <= screen_pitch
    return {
        "horizontal": horizontal,
        "screen": "on" if on_screen else "off",
        "attention": "present" if on_screen else "absent",
    }


class GazeCalibration:
    """Optional per-user affine correction from 5–9 calibration points (F2.AC4)."""

    def __init__(self) -> None:
        self._A: np.ndarray | None = None

    def fit(self, raw_pitch_yaw: np.ndarray, target_pitch_yaw: np.ndarray) -> None:
        raw = np.asarray(raw_pitch_yaw, dtype=float)
        tgt = np.asarray(target_pitch_yaw, dtype=float)
        X = np.hstack([raw, np.ones((len(raw), 1))])  # affine: [p, y, 1]
        self._A, *_ = np.linalg.lstsq(X, tgt, rcond=None)

    def apply(self, pitch: float, yaw: float) -> tuple[float, float]:
        if self._A is None:
            return pitch, yaw
        out = np.array([pitch, yaw, 1.0]) @ self._A
        return float(out[0]), float(out[1])


def confusion_matrix(true_labels, pred_labels, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        m[idx[t], idx[p]] += 1
    return m


def accuracy(true_labels, pred_labels) -> float:
    t = np.array(true_labels)
    p = np.array(pred_labels)
    return float((t == p).mean()) if t.size else 0.0


def accuracy_above_chance(acc: float, n_classes: int) -> float:
    return acc - 1.0 / n_classes
