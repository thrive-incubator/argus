"""Agreement statistics for camera-vs-reference validation (ADR-18, TECH §12).

Bland-Altman (bias + 95% LoA), MAE/RMSE/MAPE, Lin's CCC, and the pre-registered
numeric thresholds (EC13 for HR; CTA-2065 MAPE<10%). Single-subject results are
feasibility, not conformance — that framing lives in the report, not these formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pair(measured, reference):
    m = np.asarray(measured, dtype=float)
    r = np.asarray(reference, dtype=float)
    if m.shape != r.shape:
        raise ValueError("measured and reference must have the same shape")
    if m.size == 0:
        raise ValueError("empty input")
    return m, r


def mae(measured, reference) -> float:
    m, r = _pair(measured, reference)
    return float(np.mean(np.abs(m - r)))


def rmse(measured, reference) -> float:
    m, r = _pair(measured, reference)
    return float(np.sqrt(np.mean((m - r) ** 2)))


def mape(measured, reference) -> float:
    """Mean absolute percentage error (%). Reference values must be non-zero."""
    m, r = _pair(measured, reference)
    if np.any(r == 0):
        raise ValueError("MAPE undefined when a reference value is 0")
    return float(np.mean(np.abs((m - r) / r)) * 100.0)


@dataclass(frozen=True)
class BlandAltman:
    bias: float
    sd: float
    loa_lower: float
    loa_upper: float


def bland_altman(measured, reference) -> BlandAltman:
    """Bias (mean difference) and 95% limits of agreement (bias ± 1.96·SD)."""
    m, r = _pair(measured, reference)
    diff = m - r
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return BlandAltman(bias, sd, bias - 1.96 * sd, bias + 1.96 * sd)


def pearson_r(measured, reference) -> float:
    """Pearson correlation (association, not agreement — reported alongside CCC)."""
    m, r = _pair(measured, reference)
    if m.size < 2 or m.std() < 1e-12 or r.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(m, r)[0, 1])


def lins_ccc(measured, reference) -> float:
    """Lin's Concordance Correlation Coefficient (accuracy + precision vs identity)."""
    m, r = _pair(measured, reference)
    mean_m, mean_r = m.mean(), r.mean()
    var_m = np.mean((m - mean_m) ** 2)
    var_r = np.mean((r - mean_r) ** 2)
    cov = np.mean((m - mean_m) * (r - mean_r))
    denom = var_m + var_r + (mean_m - mean_r) ** 2
    if denom <= 1e-20:
        return 1.0
    return float(2.0 * cov / denom)


def ec13_pass(measured, reference) -> bool:
    """ANSI/AAMI EC13 numeric threshold: MAE <= max(5 bpm, 10% of mean reference HR).

    NOTE: single-subject feasibility check, NOT EC13 conformance (which needs a
    powered multi-subject study) — review item sM5.
    """
    m, r = _pair(measured, reference)
    tol = max(5.0, 0.10 * float(np.mean(r)))
    return mae(m, r) <= tol


def cta2065_pass(measured, reference, limit_pct: float = 10.0) -> bool:
    """ANSI/CTA-2065 numeric threshold: MAPE < ``limit_pct`` (default 10%)."""
    return mape(measured, reference) < limit_pct
