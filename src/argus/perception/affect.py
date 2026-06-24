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


class HSEmotionEstimator:
    """Real HSEmotion ONNX adapter (enet_b0_8_va_mtl)."""

    def __init__(self, model_path: str, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort  # local import

        self._sess = ort.InferenceSession(model_path, providers=list(providers))
        self._labels = ["anger", "contempt", "disgust", "fear", "happy", "neutral",
                        "sad", "surprise"]

    def estimate(self, face_crop):  # pragma: no cover - model inference
        out = self._sess.run(None, {self._sess.get_inputs()[0].name: face_crop})
        logits, va = out[0][0], out[1][0]
        i = int(np.argmax(logits))
        conf = float(np.exp(logits[i]) / np.exp(logits).sum())
        return AffectEstimate(self._labels[i], float(va[0]), float(va[1]), conf)


class FakeEmotionEstimator:
    def __init__(self, estimate: AffectEstimate | None = None):
        self._e = estimate or AffectEstimate("neutral", 0.0, 0.0, 0.8)

    def estimate(self, face_crop):
        return self._e


class AffectExtractor(Extractor):
    name = "affect"

    def __init__(self, estimator: EmotionEstimator, fps: float = 30.0, live_hz: float = 12.0):
        self.estimator = estimator
        self.period = 1.0 / live_hz
        self._last: float | None = None

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:
            return []
        if self._last is not None and (ctx.ts - self._last) < self.period:
            return []
        self._last = ctx.ts
        est = self.estimator.estimate(ctx.frame)
        meta = {"label": "estimate", "is_verdict": False, "emotion": est.emotion}
        return [
            SignalRecord("affect_valence", est.valence, est.confidence, ctx.ts,
                         gate="unknown", meta=meta),
            SignalRecord("affect_arousal", est.arousal, est.confidence, ctx.ts,
                         gate="unknown", meta=meta),
        ]


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
