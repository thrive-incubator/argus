"""Honest accuracy-ceiling metadata per signal (algorithm-review §9, gap #5).

Every signal should surface its realistic accuracy bound, not just a number. These are the
review's literature-grounded ceilings for a single consumer RGB webcam on a seated adult —
displayed in the UI so an estimate is never mistaken for a calibrated measurement.
"""

from __future__ import annotations

# name → (best-case, typical/in-the-wild, one-line caveat)
SIGNAL_CEILINGS: dict[str, dict] = {
    "hr": {
        "best": "1–4 bpm MAE", "typical": "8–15 bpm MAE",
        "note": "POS on a still, well-lit face; motion/large gestures push to 17–30 bpm. "
                "Error degrades ~2–3× for darker skin — read with the fairness caveat.",
    },
    "hrv": {
        "best": "SDNN ~6 ms / RMSSD ~10 ms", "typical": "SDNN ~11 ms / RMSSD ~11 ms",
        "note": "RMSSD reliable only at group level; no pNN50/LF-HF from a webcam. "
                "Prefer 60 fps capture for RMSSD.",
    },
    "resp": {
        "best": "~0.5–1 brpm (chest-ROI optical flow)", "typical": "~1–2 brpm",
        "note": "Indicative — no respiratory belt. Non-breathing motion is the dominant error.",
    },
    "blink": {
        "best": "F1 ~0.95 (frontal, lit)", "typical": "degrades off-axis / with glasses glare",
        "note": "P80 PERCLOS needs the graded-openness signal; validate on DROZY/NTHU-DDD.",
    },
    "perclos": {
        "best": "P80 over a 60 s window", "typical": "blink-excluded sustained closure",
        "note": "Drowsiness proxy, not a clinical vigilance measure.",
    },
    "fidget": {
        "best": "relative trend", "typical": "relative trend",
        "note": "Descriptive restlessness (MEA energy + SPARC smoothness); not a clinical scale.",
    },
    "posture": {
        "best": "relative deviation vs baseline", "typical": "relative deviation vs baseline",
        "note": "Frontal proxy — cannot measure craniovertebral angle; not a clinical instrument.",
    },
    "gaze": {
        "best": "~1.5° (~1.5 cm at 60 cm)", "typical": "~2–4°",
        "note": "Good for coarse ~4–6 screen regions with calibration, not microsaccades.",
    },
    "affect": {
        "best": "~63% 8-class / CCC ~0.5", "typical": "~60–65% 8-class / CCC 0.45–0.55",
        "note": "Reads facial EXPRESSION, not felt emotion (Barrett 2019); a task/label ceiling.",
    },
}


def ceiling_for(name: str) -> dict | None:
    """Return the ceiling metadata for a signal name (or None if unknown)."""
    return SIGNAL_CEILINGS.get(name)
