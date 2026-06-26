"""Skin-tone fairness measurement for the rPPG path (algorithm-review §9, gap #1).

The review's #1 correctness gap: HR error degrades ~2–3× for Fitzpatrick V–VI, and we don't
currently *measure* it. This module (a) estimates a subject's skin tone from the ROI colour
via the Individual Typology Angle (ITA°) → Fitzpatrick category, and (b) computes
skin-tone-stratified error so reports can be broken out by group instead of pooled.

ITA° = arctan((L* − 50) / b*) · 180/π  (Chardon 1991). Category thresholds are the standard
dermatology bins; the Fitzpatrick mapping is the common 6-point approximation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkinTone:
    ita_deg: float
    fitzpatrick: int  # 1..6
    label: str


# ITA° category boundaries (high ITA = lighter), with Fitzpatrick approximations.
_ITA_BINS = [
    (55.0, 1, "very light"),
    (41.0, 2, "light"),
    (28.0, 3, "intermediate"),
    (10.0, 4, "tan"),
    (-30.0, 5, "brown"),
    (-1e9, 6, "dark"),
]


def _rgb_to_lab(rgb) -> tuple[float, float, float]:
    """sRGB (0..255) → CIE L*a*b* (L* 0..100, a*/b* centred at 0) via OpenCV."""
    import cv2
    import numpy as np

    px = np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    L = lab[0] * 100.0 / 255.0
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return float(L), float(a), float(b)


def individual_typology_angle(rgb) -> float:
    """ITA° from a mean skin RGB (higher = lighter)."""
    import numpy as np

    L, _, b = _rgb_to_lab(rgb)
    if abs(b) < 1e-6:
        b = 1e-6
    return float(np.degrees(np.arctan2(L - 50.0, b)))


def estimate_skin_tone(rgb) -> SkinTone:
    """Estimate skin tone (ITA° + Fitzpatrick category) from a mean ROI RGB."""
    ita = individual_typology_angle(rgb)
    for thr, fitz, label in _ITA_BINS:
        if ita > thr:
            return SkinTone(ita, fitz, label)
    return SkinTone(ita, 6, "dark")


def stratified_error(samples) -> dict:
    """Group per-sample errors by Fitzpatrick category and summarise (review §9).

    Args:
        samples: iterable of ``(fitzpatrick:int, error:float)``.

    Returns:
        ``{fitzpatrick: {"n", "mae", "errors"}}`` plus a ``"_disparity"`` key = ratio of the
        worst group's MAE to the best group's MAE (the fairness number to watch; >~2 is the
        documented rPPG skin-tone gap).
    """
    import numpy as np

    groups: dict[int, list[float]] = {}
    for fitz, err in samples:
        groups.setdefault(int(fitz), []).append(abs(float(err)))
    out: dict = {}
    maes = {}
    for fitz, errs in sorted(groups.items()):
        mae = float(np.mean(errs))
        out[fitz] = {"n": len(errs), "mae": mae, "errors": errs}
        maes[fitz] = mae
    if len(maes) >= 2:
        lo, hi = min(maes.values()), max(maes.values())
        out["_disparity"] = float(hi / lo) if lo > 1e-9 else float("inf")
    return out
