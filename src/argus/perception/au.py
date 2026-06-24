"""Research Action Units (FR-11, G2). Decoupled, 0–5 FACS intensities.

- ``LibreFaceAuEstimator``: real ONNX adapter (lazy); ONNX preferred so it runs in the main
  venv (G2.AC3). Inference is the untested device/model line.
- ``FakeAuEstimator``: deterministic 0–5 intensities for tests.
- ``AuExtractor``: decoupled worker at 5–15 fps (G2.AC1); streams flagged ``research`` (G2.AC2).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..contracts import Extractor, FrameContext, SignalRecord

# A representative AU subset (FACS) with 0–5 intensity scale.
AU_KEYS = ("AU01", "AU02", "AU04", "AU06", "AU07", "AU12", "AU15", "AU25")


class AuEstimator(Protocol):
    def estimate(self, face_crop) -> dict[str, float]:
        """Return AU -> intensity in [0, 5]."""
        ...


class LibreFaceAuEstimator:
    def __init__(self, model_path: str, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort  # local import

        self._sess = ort.InferenceSession(model_path, providers=list(providers))

    def estimate(self, face_crop):  # pragma: no cover - model inference
        out = self._sess.run(None, {self._sess.get_inputs()[0].name: face_crop})[0][0]
        return {k: float(np.clip(v, 0.0, 5.0)) for k, v in zip(AU_KEYS, out)}


class FakeAuEstimator:
    def __init__(self, intensities: dict[str, float] | None = None):
        self._i = intensities or {k: 1.0 for k in AU_KEYS}

    def estimate(self, face_crop):
        return dict(self._i)


class AuExtractor(Extractor):
    name = "au"

    def __init__(self, estimator: AuEstimator, au_hz: float = 10.0):
        self.estimator = estimator
        self.period = 1.0 / au_hz
        self._last: float | None = None
        self.runs = 0  # observability: how many times the model actually ran

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        if ctx.face is None:
            return []
        if self._last is not None and (ctx.ts - self._last) < self.period:
            return []  # decoupled cadence — doesn't run every frame (G2.AC1)
        self._last = ctx.ts
        self.runs += 1
        aus = self.estimator.estimate(ctx.frame)
        out = []
        for k, v in aus.items():
            assert 0.0 <= v <= 5.0
            out.append(
                SignalRecord(f"au_{k}", float(v), 1.0, ctx.ts, gate="unknown",
                             meta={"research": True, "scale": "0-5 FACS", "method": "libreface"})
            )
        return out
