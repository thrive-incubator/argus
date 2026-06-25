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


from ..contracts import Extractor, FrameContext, SignalRecord

_H_CODE = {"left": -1.0, "center": 0.0, "right": 1.0}

# MediaPipe 478-mesh iris + eye-corner indices.
_L_OUT, _L_IN, _L_IRIS, _L_TOP, _L_BOT = 263, 362, 473, 386, 374
_R_OUT, _R_IN, _R_IRIS, _R_TOP, _R_BOT = 33, 133, 468, 159, 145


def iris_gaze_angles(landmarks: np.ndarray, yaw_scale: float = 90.0,
                     pitch_scale: float = 60.0):
    """Geometric gaze (yaw, pitch) in degrees from MediaPipe iris vs eye-corner offsets.

    Coarse (zone-level) gaze — no model needed (ADR-09 iris front-end). Yaw>0 = looking to
    the subject's right; pitch>0 = looking down.
    """
    lm = np.asarray(landmarks, dtype=float)

    def h_off(out_i, in_i, iris_i):
        mid_x = (lm[out_i, 0] + lm[in_i, 0]) / 2.0
        width = abs(lm[out_i, 0] - lm[in_i, 0]) + 1e-9
        return (lm[iris_i, 0] - mid_x) / width

    def v_off(top_i, bot_i, iris_i):
        mid_y = (lm[top_i, 1] + lm[bot_i, 1]) / 2.0
        height = abs(lm[bot_i, 1] - lm[top_i, 1]) + 1e-9
        return (lm[iris_i, 1] - mid_y) / height

    yaw = (h_off(_L_OUT, _L_IN, _L_IRIS) + h_off(_R_OUT, _R_IN, _R_IRIS)) / 2.0 * yaw_scale
    pitch = (v_off(_L_TOP, _L_BOT, _L_IRIS) + v_off(_R_TOP, _R_BOT, _R_IRIS)) / 2.0 * pitch_scale
    return float(yaw), float(pitch)


def gaze_features(landmarks: np.ndarray) -> list[float]:
    """Compact feature vector for *calibrated screen-gaze* regression.

    Per-eye iris offset within the eye (normalised by eye width/height) captures eye
    rotation; nose-tip position captures head translation. The calibration regression maps
    these to screen coordinates for the user's specific camera/screen geometry.
    Returns [r_ix, r_iy, l_ix, l_iy, nose_x, nose_y].
    """
    lm = np.asarray(landmarks, dtype=float)

    def eye(inner, outer, top, bot, iris):
        cx = (lm[inner, 0] + lm[outer, 0]) / 2.0
        cy = (lm[top, 1] + lm[bot, 1]) / 2.0
        wx = abs(lm[outer, 0] - lm[inner, 0]) + 1e-6
        hy = abs(lm[bot, 1] - lm[top, 1]) + 1e-6
        return (lm[iris, 0] - cx) / wx, (lm[iris, 1] - cy) / hy

    r_ix, r_iy = eye(133, 33, 159, 145, 468)   # image-left eye
    l_ix, l_iy = eye(362, 263, 386, 374, 473)  # image-right eye
    return [r_ix, r_iy, l_ix, l_iy, float(lm[1, 0]), float(lm[1, 1])]


class GazeFeatureExtractor(Extractor):
    """Emits the raw gaze feature vector for browser-side screen-gaze calibration."""

    name = "gaze_raw"

    def __init__(self, fps: float = 30.0, hz: float = 20.0):
        self.period = 1.0 / hz
        self._last: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None or getattr(ctx.face, "landmarks", None) is None:
            return []
        if self._last is not None and (ctx.ts - self._last) < self.period:
            return []
        self._last = ctx.ts
        return [SignalRecord("gaze_raw", gaze_features(ctx.face.landmarks), 1.0, ctx.ts,
                             gate="unknown", meta={})]


class IrisGazeExtractor(Extractor):
    """Live gaze-zone extractor from MediaPipe iris geometry (no extra model)."""

    name = "gaze"

    def __init__(self, fps: float = 30.0, hz: float = 10.0):
        self.period = 1.0 / hz
        self._last: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None or getattr(ctx.face, "landmarks", None) is None:
            return []
        if self._last is not None and (ctx.ts - self._last) < self.period:
            return []
        self._last = ctx.ts
        yaw, pitch = iris_gaze_angles(ctx.face.landmarks)
        zone = gaze_to_zone(pitch, yaw)
        return [SignalRecord("gaze_zone", _H_CODE[zone["horizontal"]], 1.0, ctx.ts,
                             gate="unknown", meta={"zone": zone, "yaw": yaw, "pitch": pitch})]


class GazeExtractor(Extractor):
    """Live gaze-zone extractor (FR-9). Emits numeric zone codes; the human-readable zone
    dict rides in ``meta``."""

    name = "gaze"

    def __init__(self, estimator: GazeEstimator, fps: float = 30.0, hz: float = 10.0):
        self.estimator = estimator
        self.period = 1.0 / hz
        self._last: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:
            return []
        if self._last is not None and (ctx.ts - self._last) < self.period:
            return []
        self._last = ctx.ts
        pitch, yaw = self.estimator.estimate(ctx.frame)
        zone = gaze_to_zone(pitch, yaw)
        return [
            SignalRecord("gaze_zone", _H_CODE[zone["horizontal"]], 1.0, ctx.ts,
                         gate="unknown", meta={"zone": zone, "pitch": pitch, "yaw": yaw})
        ]


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
