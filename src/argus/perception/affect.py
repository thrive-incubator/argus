"""Live facial affect (FR-10, G1). Blendshapes + emotion + valence/arousal.

Outputs are always framed as *estimates with confidence*, never verdicts (G1.AC3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..contracts import Extractor, FrameContext, SignalRecord


class BlendshapeNormalizer:
    """Per-session neutral-subtract + z-score of the 52 blendshapes (G1.AC1)."""

    def __init__(self, baseline_frames: int = 60):
        self.baseline_frames = baseline_frames
        self._buf: list[np.ndarray] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def armed(self) -> bool:
        return self._mean is not None

    def update_and_normalize(self, blendshapes: np.ndarray):
        bs = np.asarray(blendshapes, dtype=float)
        if self._mean is None:
            self._buf.append(bs)
            if len(self._buf) >= self.baseline_frames:
                arr = np.array(self._buf)
                self._mean = arr.mean(axis=0)
                self._std = arr.std(axis=0) + 1e-6
            return None
        return (bs - self._mean) / self._std


@dataclass(frozen=True)
class AffectEstimate:
    emotion: str
    valence: float
    arousal: float
    confidence: float


class EmotionEstimator(Protocol):
    def estimate(self, face_crop) -> AffectEstimate: ...


def _va_from_scores(emotion, scores) -> AffectEstimate:
    """Parse a multi-task (emotion + V/A) score vector into an AffectEstimate.

    The Savchenko ``*_va_mtl`` models (HSEmotion / EmotiEffLib) lay out scores as
    ``[8 emotion probs/logits ..., valence, arousal]`` — the last two columns are the native
    V/A head. Accepts a 1-D vector or a 1×N batch.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if isinstance(emotion, (list, tuple, np.ndarray)):
        emotion = emotion[0]
    valence, arousal = float(scores[-2]), float(scores[-1])
    confidence = float(scores[:-2].max()) if scores.size > 2 else 1.0
    return AffectEstimate(str(emotion).lower(), valence, arousal, confidence)


class HSEmotionEstimator:
    """Real HSEmotion adapter via the ``hsemotion-onnx`` package (enet_b0_8_va_mtl).

    Uses the model's **native valence/arousal head** (the research-paper default, Savchenko's
    multi-task VA model) plus its 8-class emotion label.
    """

    backend = "hsemotion"

    def __init__(self, model_name: str = "enet_b0_8_va_mtl"):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer  # local import

        self.model_name = model_name
        self._rec = HSEmotionRecognizer(model_name=model_name)

    def estimate(self, face_bgr):  # pragma: no cover - model inference
        import cv2

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) if face_bgr.ndim == 3 else face_bgr
        emotion, scores = self._rec.predict_emotions(rgb, logits=False)
        return _va_from_scores(emotion, scores)


class EmotiEffLibAffectEstimator:
    """Affect via the maintained **EmotiEffLib** package with a newer multi-task V/A head.

    Review §8: HSEmotion's ``enet_b0_8_va_mtl`` V/A head is the one place with a real accuracy
    gap; EmotiEffLib exposes newer multi-task VA backbones (``mobilevit_va_mtl``, ``mbf_va_mtl``)
    with stronger valence/arousal CCC, same Apache-2.0 license, same ``[emotion…, V, A]`` score
    layout. The recognizer can be injected for testing.
    """

    backend = "emotiefflib"

    def __init__(self, va_model: str = "mobilevit_va_mtl", engine: str = "onnx",
                 device: str = "cpu", recognizer=None):
        self.model_name = va_model
        if recognizer is not None:
            self._rec = recognizer
            return
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # local import

        self._rec = EmotiEffLibRecognizer(engine=engine, model_name=va_model, device=device)

    def estimate(self, face_bgr):  # pragma: no cover - model inference
        import cv2

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) if face_bgr.ndim == 3 else face_bgr
        emotion, scores = self._rec.predict_emotions(rgb, logits=False)
        return _va_from_scores(emotion, scores)


def build_affect_estimator(prefer: str = "emotiefflib",
                           va_model: str = "mobilevit_va_mtl",
                           fallback_model: str = "enet_b0_8_va_mtl"):
    """Construct the best-available V/A affect estimator with graceful fallback (review §8).

    Tries the preferred backend (EmotiEffLib's newer V/A head by default); if its package or
    weights are unavailable, falls back to the HSEmotion ``enet_b0_8_va_mtl`` head. Always
    returns an object with ``.estimate(face_bgr)`` and a ``.backend`` attribute.
    """
    if prefer == "emotiefflib":
        try:
            return EmotiEffLibAffectEstimator(va_model=va_model)
        except Exception:
            pass
    return HSEmotionEstimator(model_name=fallback_model)


def _face_crop_bgr(ctx, pad: float = 0.15):
    """Crop the face region from the frame using landmark extent (whole frame if absent)."""
    frame = ctx.frame
    lm = getattr(ctx.face, "landmarks", None)
    if lm is None:
        return frame
    h, w = frame.shape[:2]
    xs, ys = lm[:, 0], lm[:, 1]
    x0 = int(max(0, (xs.min() - pad) * w)); x1 = int(min(w, (xs.max() + pad) * w))
    y0 = int(max(0, (ys.min() - pad) * h)); y1 = int(min(h, (ys.max() + pad) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame
    return frame[y0:y1, x0:x1]


# MediaPipe FaceMesh iris-centre indices (right eye, left eye).
_R_IRIS_C, _L_IRIS_C = 468, 473


def aligned_face_crop_bgr(ctx, size: int = 224, scale: float = 2.9, y_offset: float = 0.40):
    """Square, **eye-aligned** face crop sized ``size×size`` for the affect model.

    The AffectNet-trained V/A models do NO internal detection/alignment — they just
    ``cv2.resize(crop, (224,224))``. Feeding a raw (non-square) landmark bounding box stretches
    the face and pushes predictions toward "surprise" (review/diagnosis). This builds the crop
    the model expects: rotate so the eyes are level, scale by the inter-ocular distance, and
    centre on the face — a single affine warp. Falls back to the bbox crop if iris landmarks
    are unavailable.
    """
    frame = ctx.frame
    lm = getattr(ctx.face, "landmarks", None)
    if lm is None or len(lm) <= _L_IRIS_C:
        return _face_crop_bgr(ctx)
    import cv2

    h, w = frame.shape[:2]
    re = np.array([lm[_R_IRIS_C, 0] * w, lm[_R_IRIS_C, 1] * h], dtype=float)
    le = np.array([lm[_L_IRIS_C, 0] * w, lm[_L_IRIS_C, 1] * h], dtype=float)
    eye_c = (re + le) / 2.0
    d = float(np.linalg.norm(le - re))
    if d < 4.0:  # face too small / degenerate → fall back
        return _face_crop_bgr(ctx)
    angle = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))  # roll of the eye line
    side = scale * d  # square side in source pixels
    cx, cy = eye_c[0], eye_c[1] + y_offset * d  # centre a bit below the eyes
    # affine that rotates the eyes level, scales side→size, and centres (cx,cy) in the output.
    M = cv2.getRotationMatrix2D((cx, cy), angle, size / side)
    M[0, 2] += size / 2.0 - cx
    M[1, 2] += size / 2.0 - cy
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class FakeEmotionEstimator:
    def __init__(self, estimate: AffectEstimate | None = None):
        self._e = estimate or AffectEstimate("neutral", 0.0, 0.0, 0.8)

    def estimate(self, face_crop):
        return self._e


class AffectExtractor(Extractor):
    name = "affect"

    def __init__(self, estimator: EmotionEstimator, fps: float = 30.0, live_hz: float = 12.0,
                 align: bool = True):
        self.estimator = estimator
        self.period = 1.0 / live_hz
        self.align = align  # feed a square eye-aligned crop (the model expects it)
        self._last: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:
            return []
        if self._last is not None and (ctx.ts - self._last) < self.period:
            return []
        self._last = ctx.ts
        crop = aligned_face_crop_bgr(ctx) if self.align else _face_crop_bgr(ctx)
        est = self.estimator.estimate(crop)
        meta = {"label": "estimate", "is_verdict": False, "emotion": est.emotion}
        return [
            SignalRecord("affect_valence", est.valence, est.confidence, ctx.ts,
                         gate="unknown", meta=meta),
            SignalRecord("affect_arousal", est.arousal, est.confidence, ctx.ts,
                         gate="unknown", meta=meta),
        ]


# --- AU -> emotional state (research-grounded; complementary to the image-based model) ---
# Discrete-emotion prototypes: Du, Tao & Martinez (2014, PNAS) refinement of Ekman & Friesen
# EMFACS-7 (1983). Disgust uses the AU17 variant (AU16 is absent from py-feat); contempt is
# treated bilaterally (laterality is not recoverable from py-feat AUs).
EMOTION_PROTOTYPES = {
    "happiness": ("AU06", "AU12"),
    "sadness": ("AU01", "AU04", "AU15"),
    "surprise": ("AU01", "AU02", "AU05", "AU26"),
    "fear": ("AU01", "AU02", "AU04", "AU05", "AU07", "AU20", "AU26"),
    "anger": ("AU04", "AU05", "AU07", "AU23"),
    "disgust": ("AU09", "AU15", "AU17"),
    "contempt": ("AU12", "AU14"),
}
# Inhibitors: subtract these so a real smile (AU6) isn't read as contempt, and a fear pattern
# (AU4/7/20) isn't read as surprise.
EMOTION_INHIBITORS = {
    "contempt": ("AU06",),
    "surprise": ("AU04", "AU07", "AU20"),
}
# Valence/arousal weights: signs/which-AUs anchored in Zhang et al. (2024, Sci Rep 14:19563)
# + the corrugator/zygomaticus EMG consensus. Magnitudes are reasoned defaults (no single
# published per-AU coefficient table exists for this AU set — calibrate if quantitative VA needed).
_VALENCE_WEIGHTS = {
    "AU06": +0.50, "AU12": +1.00,
    "AU01": -0.30, "AU04": -0.60, "AU07": -0.30,
    "AU09": -0.50, "AU10": -0.40, "AU15": -0.50,
    "AU17": -0.20, "AU20": -0.40, "AU23": -0.40, "AU24": -0.30, "AU14": -0.20,
}
_AROUSAL_WEIGHTS = {
    "AU01": 0.40, "AU02": 0.50, "AU04": 0.40, "AU05": 0.80, "AU07": 0.40,
    "AU09": 0.30, "AU10": 0.30, "AU12": 0.40, "AU20": 0.50, "AU23": 0.40,
    "AU25": 0.50, "AU26": 0.60,
}


def au_emotion_probs(au: dict) -> dict:
    """Weighted prototype-match over the Du/Martinez AU sets → emotion 'probabilities'."""
    raw = {}
    for emo, proto in EMOTION_PROTOTYPES.items():
        s = float(np.mean([float(au.get(k, 0.0)) for k in proto]))
        for k in EMOTION_INHIBITORS.get(emo, ()):
            s -= float(au.get(k, 0.0))
        raw[emo] = max(s, 0.0)
    raw["neutral"] = max(1.0 - max(raw.values(), default=0.0), 0.0)
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def au_to_emotion(au: dict, threshold: float = 0.0) -> str:
    """Top emotion label from the prototype-match probabilities."""
    probs = au_emotion_probs(au)
    label = max(probs, key=probs.get)
    return label


def au_to_valence_arousal(au: dict, gain: float = 3.0) -> tuple[float, float]:
    """Valence/arousal on the circumplex from AU intensities (Zhang-2024-anchored weights).

    ``gain`` spreads the output across the axes since real faces rarely co-activate all AUs.
    """
    v = sum(float(au.get(k, 0.0)) * w for k, w in _VALENCE_WEIGHTS.items())
    a = sum(float(au.get(k, 0.0)) * w for k, w in _AROUSAL_WEIGHTS.items())
    v_max = sum(abs(w) for w in _VALENCE_WEIGHTS.values())
    a_max = sum(_AROUSAL_WEIGHTS.values())
    valence = float(np.clip(v / v_max * gain, -1.0, 1.0))
    arousal = float(np.clip(a / a_max * gain, 0.0, 1.0))
    return valence, arousal


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 1e-12 else 0.0


def face_validity_report(posed_happy_valence, posed_sad_valence) -> dict:
    """Model face-validity check (G1.AC4): posed-happy V should exceed posed-sad V.

    NOT a measure of felt affect — descriptive, no pass/fail (ADR-11).
    """
    happy = np.asarray(posed_happy_valence, dtype=float)
    sad = np.asarray(posed_sad_valence, dtype=float)
    return {
        "happy_mean": float(happy.mean()),
        "sad_mean": float(sad.mean()),
        "effect_size_d": cohens_d(happy, sad),
        "happy_gt_sad": bool(happy.mean() > sad.mean()),
        "note": "face-validity of the model on this face; posed != felt affect",
    }
