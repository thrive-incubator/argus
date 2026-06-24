"""Covariate layer (TECH §9 rev 2).

Lighting is a **relative brightness index** from mean luma — uncalibrated, explicitly
NOT lux (luma depends on exposure/gain/white-balance). Absolute lux is logged
separately from a free phone app per block (review item M4).
"""

from __future__ import annotations

import numpy as np

# BT.601 luma weights for an RGB frame.
_LUMA = np.array([0.299, 0.587, 0.114])


def brightness_index(frame: np.ndarray) -> float:
    """Mean luma normalised to 0..1 (assumes 8-bit RGB/BGR-ish channels)."""
    f = np.asarray(frame, dtype=float)
    if f.ndim == 3 and f.shape[-1] >= 3:
        luma = f[..., :3] @ _LUMA
    else:
        luma = f
    return float(np.clip(luma.mean() / 255.0, 0.0, 1.0))


def exposure_flags(frame: np.ndarray, under: float = 0.15, over: float = 0.85):
    """Return ``(underexposed, overexposed)`` from the brightness index."""
    bi = brightness_index(frame)
    return bi < under, bi > over


from dataclasses import dataclass
from enum import IntEnum

from ..contracts import Extractor, FrameContext, SignalRecord


class Fitzpatrick(IntEnum):
    I = 1
    II = 2
    III = 3
    IV = 4
    V = 5
    VI = 6


@dataclass(frozen=True)
class SessionCovariates:
    """Recorded once per session and stored with the XDF (FR-23, H3.AC2).

    Skin tone is **forward-compat for Phase 2 only** — no fairness info at n=1.
    """

    fitzpatrick: Fitzpatrick
    eyewear: bool
    facial_hair: bool
    monk: int | None = None  # Monk 10-point scale, optional


def cheek_reflectance_estimate(frame: np.ndarray) -> float:
    """Unvalidated automated skin-tone proxy (mean luma 0..1) — descriptive only, n=1."""
    return brightness_index(frame)


class CovariateExtractor(Extractor):
    """Publishes per-frame covariate streams (FR-15, H3.AC1)."""

    name = "covariate"

    def __init__(self, lux_provider=None):
        # lux is a *measured* value entered per block from a free phone app (not derived)
        self.lux_provider = lux_provider

    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        bi = brightness_index(ctx.frame)
        under, over = exposure_flags(ctx.frame)
        presence = 1.0 if ctx.face is not None else 0.0
        lux = float(self.lux_provider()) if self.lux_provider else float("nan")
        recs = [
            SignalRecord("lighting_index", bi, 1.0, ctx.ts, gate="unknown",
                         meta={"uncalibrated": True, "not_lux": True,
                               "measured_lux": lux, "under": under, "over": over}),
            SignalRecord("face_presence", presence, 1.0, ctx.ts, gate="unknown", meta={}),
        ]
        return recs
