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


def gaze_to_zone(pitch_deg: float, yaw_deg: float, yaw_lr: float = 8.0, pitch_ud: float = 7.0,
                 screen_yaw: float = 20.0, screen_pitch: float = 20.0) -> dict:
    """Map a gaze angle to direction zones (F2.AC2 — no pixel point-of-regard).

    Horizontal {left,center,right} from yaw, vertical {up,center,down} from pitch
    (pitch>0 = looking down).
    """
    horizontal = "left" if yaw_deg < -yaw_lr else "right" if yaw_deg > yaw_lr else "center"
    vertical = "up" if pitch_deg < -pitch_ud else "down" if pitch_deg > pitch_ud else "center"
    on_screen = abs(yaw_deg) <= screen_yaw and abs(pitch_deg) <= screen_pitch
    return {
        "horizontal": horizontal,
        "vertical": vertical,
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


def head_angles(head_pose) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees from a 4x4 head-pose matrix (scale-normalised)."""
    if head_pose is None:
        return 0.0, 0.0, 0.0
    R = np.asarray(head_pose, dtype=float)[:3, :3].copy()
    for c in range(3):  # strip scale
        n = np.linalg.norm(R[:, c])
        if n > 1e-9:
            R[:, c] /= n
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


def gaze_features(landmarks: np.ndarray, head_pose=None) -> list[float]:
    """Compact feature vector for *calibrated screen-gaze* regression.

    Combines BOTH parts of true gaze: eye-in-head rotation (averaged iris offset) AND head
    pose — head rotation (yaw/pitch) + head translation (nose position) — PLUS an
    inter-ocular-distance (iris-to-iris) scale feature that encodes distance-to-screen
    (review §7: the #2 failure mode is distance change silently breaking the mapping).

    Returns ``[eye_x, eye_y, head_yaw, head_pitch, nose_x, nose_y, iod]``.
    """
    lm = np.asarray(landmarks, dtype=float)

    def eye(inner, outer, top, bot, iris):
        cx = (lm[inner, 0] + lm[outer, 0]) / 2.0
        cy = (lm[top, 1] + lm[bot, 1]) / 2.0
        wx = abs(lm[outer, 0] - lm[inner, 0]) + 1e-6
        hy = abs(lm[bot, 1] - lm[top, 1]) + 1e-6
        return (lm[iris, 0] - cx) / wx, (lm[iris, 1] - cy) / hy

    r_ix, r_iy = eye(133, 33, 159, 145, 468)
    l_ix, l_iy = eye(362, 263, 386, 374, 473)
    eye_x, eye_y = (r_ix + l_ix) / 2.0, (r_iy + l_iy) / 2.0
    yaw, pitch, _ = head_angles(head_pose)
    iod = float(np.hypot(lm[473, 0] - lm[468, 0], lm[473, 1] - lm[468, 1]))  # distance/scale
    return [eye_x, eye_y, yaw / 90.0, pitch / 90.0, float(lm[1, 0]), float(lm[1, 1]), iod]


# Number of features emitted by gaze_features (kept in sync with the browser calibrator).
N_GAZE_FEATURES = 7


class PolynomialRidge:
    """Polynomial-ridge screen-gaze regressor — the Python parity of the browser calibrator.

    Review §7: ridge-on-polynomial-features is the empirically best regressor at a 9+ point
    budget, but a full degree-2 over 7 features is **underdetermined** from ~13–18 points, so
    ``degree=1`` is the most defensible default and ``alpha`` (ridge) is load-bearing.
    """

    def __init__(self, degree: int = 1, alpha: float = 1e-2):
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        self.degree = degree
        self.alpha = alpha
        self._W: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        cols = [np.ones((len(X), 1)), X]
        if self.degree == 2:
            cols.append(X ** 2)
            # pairwise cross terms
            n = X.shape[1]
            cross = [X[:, i] * X[:, j] for i in range(n) for j in range(i + 1, n)]
            if cross:
                cols.append(np.column_stack(cross))
        return np.hstack(cols)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PolynomialRidge":
        P = self._design(X)
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] != P.shape[0]:
            Y = Y.T
        reg = self.alpha * np.eye(P.shape[1])
        reg[0, 0] = 0.0  # don't penalise the bias term
        self._W = np.linalg.solve(P.T @ P + reg, P.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("fit() before predict()")
        out = self._design(X) @ self._W
        return out[0] if out.shape[0] == 1 else out

    def loo_cv_rmse(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Leave-one-point-out CV RMSE (review §7: report held-out, not training, error)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] != X.shape[0]:
            Y = Y.T
        n = len(X)
        if n < 3:
            return float("nan")
        errs = []
        idx = np.arange(n)
        for k in range(n):
            mask = idx != k
            model = PolynomialRidge(self.degree, self.alpha).fit(X[mask], Y[mask])
            pred = np.atleast_1d(model.predict(X[k:k + 1]))
            errs.append(float(np.sum((pred - Y[k]) ** 2)))
        return float(np.sqrt(np.mean(errs)))


def angle_to_screen_cm(angle_deg: float, viewing_distance_cm: float) -> float:
    """Convert a gaze angular error (deg) to an on-screen distance (cm) at a viewing distance.

    Review §7: report error in cm/degrees (benchmarkable), not "% of screen". At ~57 cm,
    1° ≈ 1 cm.
    """
    return float(viewing_distance_cm * np.tan(np.radians(angle_deg)))


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
        feats = gaze_features(ctx.face.landmarks, getattr(ctx.face, "head_pose", None))
        return [SignalRecord("gaze_raw", feats, 1.0, ctx.ts, gate="unknown", meta={})]


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
