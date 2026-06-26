"""Blink detection (Eye Aspect Ratio) and PERCLOS (ADR-08, TECH §6.4; algorithm-review §4).

EAR (Soukupová & Čech 2016) with a **per-session adaptive threshold** auto-calibrated
from the open-eye baseline; a blink is EAR below threshold for >= N consecutive frames.

PERCLOS (P80) = fraction of a window with the eye >= 80% closed. Review §4 notes that a
binary ``EAR < thr`` flag cannot compute a true P80 — it needs a **graded eye-openness**
signal. We therefore (a) fuse the MediaPipe ``eyeBlink`` blendshape with normalised EAR into
a 0..1 openness, and (b) compute P80 over a rolling window with short (<400 ms) blink events
excluded so genuine eyelid-closure (drowsiness) is separated from ordinary blinks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR for one eye given 6 landmark points p1..p6 (shape (6, 2)).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).
    Points ordered: p1,p4 = horizontal corners; p2,p3,p5,p6 = vertical lids.
    """
    p = np.asarray(pts, dtype=float)
    if p.shape != (6, 2):
        raise ValueError("expected 6 (x, y) eye landmarks")
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    if c <= 1e-9:
        return 0.0
    return float((a + b) / (2.0 * c))


@dataclass
class AdaptiveBlinkDetector:
    """Streaming blink detector with a per-session adaptive threshold.

    Calibrates an open-eye baseline over the first ``baseline_frames`` *valid* frames,
    then flags a blink when EAR < ``ratio * baseline`` for >= ``min_frames`` in a row.
    Stays un-armed (returns no blinks) until enough baseline frames are seen — review
    item B-blink-baseline.
    """

    baseline_frames: int = 300  # ~10 s at 30 fps
    ratio: float = 0.6
    min_frames: int = 2
    _samples: list[float] = field(default_factory=list)
    _baseline: float | None = None
    _below: int = 0
    blink_count: int = 0

    @property
    def armed(self) -> bool:
        return self._baseline is not None

    @property
    def threshold(self) -> float | None:
        return None if self._baseline is None else self.ratio * self._baseline

    def update(self, ear: float) -> bool:
        """Feed one EAR sample. Returns True on the frame a blink completes."""
        if self._baseline is None:
            self._samples.append(ear)
            if len(self._samples) >= self.baseline_frames:
                self._baseline = float(np.median(self._samples))
            return False

        thr = self.ratio * self._baseline
        completed = False
        if ear < thr:
            self._below += 1
        else:
            if self._below >= self.min_frames:
                self.blink_count += 1
                completed = True
            self._below = 0
        return completed


def blink_f1(detected_ts, annotated_ts, tol_s: float = 0.1) -> dict:
    """F1 of detected blink events vs frame-level annotation with a ±tol match (F1.AC3).

    Each annotation may match at most one detection within ``tol_s``.
    """
    det = sorted(float(t) for t in detected_ts)
    ann = sorted(float(t) for t in annotated_ts)
    used = [False] * len(det)
    tp = 0
    for a in ann:
        best = -1
        best_dt = tol_s + 1e-9
        for i, d in enumerate(det):
            if used[i]:
                continue
            dt = abs(d - a)
            if dt <= tol_s and dt < best_dt:
                best, best_dt = i, dt
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(det) - tp
    fn = len(ann) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def perclos(ear_series, baseline: float, closed_frac: float = 0.8) -> float:
    """PERCLOS-P80: fraction of frames with the eye at least 80% closed.

    "80% closed" means EAR <= (1 - closed_frac) * open-eye baseline.
    """
    ear = np.asarray(ear_series, dtype=float)
    if len(ear) == 0:
        return 0.0
    thr = (1.0 - closed_frac) * baseline
    return float(np.mean(ear <= thr))


def eye_openness(ear: float, baseline: float, blink_score: float | None = None) -> float:
    """Graded 0..1 eye-openness (review §4): 1 = fully open, 0 = fully closed.

    Normalises EAR by the open-eye baseline, and — when a MediaPipe ``eyeBlink`` blendshape
    score (0 = open, 1 = closed) is available — fuses it in (mean of the two cues). The
    blendshape de-risks the EAR failure mode of conflating squint/downward-gaze with closure.
    """
    if baseline <= 1e-9:
        ear_open = 0.0
    else:
        ear_open = float(np.clip(ear / baseline, 0.0, 1.0))
    if blink_score is None:
        return ear_open
    bs_open = float(np.clip(1.0 - blink_score, 0.0, 1.0))
    return float((ear_open + bs_open) / 2.0)


@dataclass
class PerclosP80:
    """Streaming P80 PERCLOS over a rolling time window with blink-event exclusion.

    Feed graded openness (0..1) per frame; the metric is the fraction of *valid* (non-blink)
    frames with openness <= ``1 - closed_frac`` (i.e. eyes ≥80% closed). Closure runs shorter
    than ``blink_max_s`` (default 400 ms) are treated as ordinary blinks and excluded, so the
    P80 reflects sustained eyelid droop (drowsiness), not blink rate.
    """

    fps: float = 30.0
    window_s: float = 60.0
    closed_frac: float = 0.8
    blink_max_s: float = 0.4
    _ts: deque = field(default_factory=deque)
    _open: deque = field(default_factory=deque)

    @property
    def _closed_thr(self) -> float:
        return 1.0 - self.closed_frac

    def update(self, ts: float, openness: float) -> None:
        self._ts.append(float(ts))
        self._open.append(float(openness))
        cutoff = ts - self.window_s
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
            self._open.popleft()

    def value(self) -> float:
        n = len(self._open)
        if n == 0:
            return 0.0
        closed = np.asarray(self._open, dtype=float) <= self._closed_thr
        valid = closed.copy()
        # exclude short closure runs (blinks) from the "closed" count
        max_frames = max(int(self.blink_max_s * self.fps), 1)
        i = 0
        while i < n:
            if closed[i]:
                j = i
                while j < n and closed[j]:
                    j += 1
                if (j - i) <= max_frames:  # a blink, not sustained droop
                    valid[i:j] = False
                i = j
            else:
                i += 1
        return float(valid.sum() / n)
