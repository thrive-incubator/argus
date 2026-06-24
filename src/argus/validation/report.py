"""Validation report generator (FR-22, I2).

Per condition & signal: Bland-Altman, MAE, RMSE, MAPE, Pearson r, Lin's CCC, SNR. HR at a
60 s average and a 4–10 s quasi-instantaneous window. SDNN MAE≤12 + BA + CCC (length-
matched); ln-RMSSD BA in log-units (indicative). EC13 + CTA-2065 numeric thresholds. The
H10 is labelled a reference (its own seated bias is in the reported bias). Every report
carries the feasibility banner + Fitzpatrick caveat (I2.AC4). One command writes HTML (I2.AC5).
"""

from __future__ import annotations

import numpy as np

from .stats import (
    bland_altman,
    cta2065_pass,
    ec13_pass,
    lins_ccc,
    mae,
    mape,
    pearson_r,
    rmse,
)

FEASIBILITY_BANNER = (
    "Single-subject results are hypothesis-generating; they establish neither limits of "
    "agreement nor fairness."
)
SDNN_MAE_BAR_MS = 12.0


def fitzpatrick_caveat(fitzpatrick: int) -> str:
    if fitzpatrick >= 5:
        return ("Subject is Fitzpatrick V-VI: read rPPG bars as a near-worst-case single "
                "point (melanin lowers optical-pulse SNR).")
    if fitzpatrick <= 3:
        return ("Subject is Fitzpatrick I-III: bars are a near-best case and will not "
                "generalise.")
    return "Subject is Fitzpatrick IV: mid-range single point."


def agreement(measured, reference) -> dict:
    """Full agreement set for one signal/condition; H10 is the reference."""
    ba = bland_altman(measured, reference)
    return {
        "n": int(np.size(measured)),
        "bias": ba.bias,
        "loa_lower": ba.loa_lower,
        "loa_upper": ba.loa_upper,
        "mae": mae(measured, reference),
        "rmse": rmse(measured, reference),
        "mape": mape(measured, reference),
        "pearson_r": pearson_r(measured, reference),
        "lins_ccc": lins_ccc(measured, reference),
        "reference": "Polar H10 (reference, not gold truth; includes its own seated bias)",
    }


def hr_report(hr_measured, hr_ref, hr_inst_measured=None, hr_inst_ref=None, snr_db=None) -> dict:
    """HR agreement at the 60 s average plus a 4–10 s quasi-instantaneous window (I2.AC2)."""
    rep = {
        "avg_60s": agreement(hr_measured, hr_ref),
        "ec13_pass": ec13_pass(hr_measured, hr_ref),
        "cta2065_pass": cta2065_pass(hr_measured, hr_ref),
        "note": "EC13/CTA numeric thresholds are feasibility targets, not conformance.",
    }
    if snr_db is not None:  # I2.AC1 — report SNR alongside agreement
        rep["mean_snr_db"] = float(np.mean(np.asarray(snr_db, dtype=float)))
    if hr_inst_measured is not None:
        rep["quasi_instant_4_10s"] = agreement(hr_inst_measured, hr_inst_ref)
    return rep


def sdnn_report(sdnn_measured, sdnn_ref) -> dict:
    a = agreement(sdnn_measured, sdnn_ref)
    a["mae_bar_ms"] = SDNN_MAE_BAR_MS
    a["passes_bar"] = a["mae"] <= SDNN_MAE_BAR_MS
    a["units"] = "ms (length-matched windows)"
    return a


def ln_rmssd_report(rmssd_measured, rmssd_ref) -> dict:
    """ln-RMSSD Bland-Altman in LOG units (indicative; never a ms band) — I2.AC3."""
    lm = np.log(np.asarray(rmssd_measured, dtype=float))
    lr = np.log(np.asarray(rmssd_ref, dtype=float))
    ba = bland_altman(lm, lr)
    return {
        "units": "log-units (ln ms); indicative only",
        "bias_log": ba.bias,
        "loa_lower_log": ba.loa_lower,
        "loa_upper_log": ba.loa_upper,
        "ratio_bias": float(np.exp(ba.bias)),
        "ms_band_applied": False,
    }


def generate_report(data: dict, fitzpatrick: int = 4) -> dict:
    """Assemble the full report structure from per-condition arrays."""
    report = {
        "banner": FEASIBILITY_BANNER,
        "fitzpatrick_caveat": fitzpatrick_caveat(fitzpatrick),
        "conditions": {},
    }
    for cond, d in data.items():
        block = {}
        if "hr_measured" in d:
            block["hr"] = hr_report(
                d["hr_measured"], d["hr_ref"],
                d.get("hr_inst_measured"), d.get("hr_inst_ref"),
                snr_db=d.get("snr_db"),
            )
        if "sdnn_measured" in d:
            block["sdnn"] = sdnn_report(d["sdnn_measured"], d["sdnn_ref"])
        if "rmssd_measured" in d:
            block["ln_rmssd"] = ln_rmssd_report(d["rmssd_measured"], d["rmssd_ref"])
        report["conditions"][cond] = block
    return report


def write_report_html(report: dict, path: str) -> None:
    """One command → a standalone HTML report (I2.AC5)."""
    rows = [f"<h1>Argus validation report</h1>",
            f"<p class='banner'><strong>{report['banner']}</strong></p>",
            f"<p class='caveat'>{report['fitzpatrick_caveat']}</p>"]
    for cond, block in report["conditions"].items():
        rows.append(f"<h2>{cond}</h2><pre>{block}</pre>")
    html = "<html><head><meta charset='utf-8'><title>Argus report</title></head><body>" \
           + "".join(rows) + "</body></html>"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
