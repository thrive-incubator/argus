"""Posture monitor (relative to a captured 'good posture' baseline).

A single frontal webcam can't measure absolute sagittal slouch (forward-head is a
craniovertebral angle that needs a side view; algorithm-review §6), but against a per-user
reference it can flag the practical issues. Review §6 hardens the original single-frame
heuristic into the "robust best-in-class frontal" version:

- **Median-window baseline** (not a single click) — one blink/lean at capture won't poison it.
- **Temporal persistence / hysteresis** (:class:`PostureDebouncer`) — only flag after the
  state is sustained, to stop per-frame flicker.
- **Head roll** (inter-eye line angle) — an honestly-frontal signal added to shoulder tilt.
- **Forward-head requires face-closer AND neck-shorter** — separates true forward-head from
  merely looking down (face closer alone is "leaning in"; neck shorter alone is "head dropped").
- **Confidence gating** — withhold a verdict when the landmarks aren't reliably in frame.

All geometric features are normalised by shoulder width (scale/distance-invariant); deviations
are measured as ratios to the baseline, so the camera distance/aspect cancels out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 1, 11, 12
RIGHT_EYE_OUTER, LEFT_EYE_OUTER = 33, 263  # MediaPipe FaceMesh outer eye corners


@dataclass(frozen=True)
class PostureFeatures:
    shoulder_width: float   # image-normalised, ~ inverse distance to camera
    neck_ratio: float       # (shoulder_mid_y - nose_y) / shoulder_width  (head-up-ness)
    lateral: float          # (nose_x - shoulder_mid_x) / shoulder_width  (sideways shift)
    tilt_deg: float         # shoulder tilt from horizontal
    roll_deg: float = 0.0   # head roll from the inter-eye line (honestly-frontal; review §6)


def _median_feats(samples: list[PostureFeatures]) -> PostureFeatures:
    arr = np.array([[s.shoulder_width, s.neck_ratio, s.lateral, s.tilt_deg, s.roll_deg]
                    for s in samples], dtype=float)
    med = np.median(arr, axis=0)
    return PostureFeatures(*[float(x) for x in med])


def posture_features(face_landmarks, pose_image_landmarks, aspect: float = 16 / 9):
    """Compute posture geometry from face + image-space pose landmarks (or None)."""
    if face_landmarks is None or pose_image_landmarks is None:
        return None
    fl = np.asarray(face_landmarks, dtype=float)
    pl = np.asarray(pose_image_landmarks, dtype=float)
    if pl.shape[0] <= RIGHT_SHOULDER:
        return None
    nose = fl[NOSE, :2]
    ls, rs = pl[LEFT_SHOULDER, :2], pl[RIGHT_SHOULDER, :2]
    sw = float(abs(rs[0] - ls[0])) + 1e-6
    mid = (ls + rs) / 2.0
    neck = float((mid[1] - nose[1]) / sw)
    lateral = float((nose[0] - mid[0]) / sw)
    dy = (rs[1] - ls[1]) * aspect
    tilt = float(np.degrees(np.arctan2(dy, abs(rs[0] - ls[0]) + 1e-6)))
    # head roll from the inter-eye line, when those landmarks are present.
    roll = 0.0
    if fl.shape[0] > LEFT_EYE_OUTER:
        re, le = fl[RIGHT_EYE_OUTER, :2], fl[LEFT_EYE_OUTER, :2]
        if np.any(re) or np.any(le):
            edy = (le[1] - re[1]) * aspect
            roll = float(np.degrees(np.arctan2(edy, abs(le[0] - re[0]) + 1e-6)))
    return PostureFeatures(sw, neck, lateral, tilt, roll)


class PostureDebouncer:
    """Temporal hysteresis for the posture status (review §6).

    A new status is only *reported* after it has persisted for ``hold_s`` — so a momentary
    lean or a single noisy frame doesn't flip the badge. Defaults to "good" until a worse
    state is sustained, and only clears back after a better state is sustained.
    """

    _RANK = {"good": 0, "fair": 1, "poor": 2}

    def __init__(self, hold_s: float = 3.0):
        self.hold_s = hold_s
        self.reported = "good"
        self._candidate: str | None = None
        self._since: float | None = None

    def update(self, status: str, ts: float) -> str:
        if status not in self._RANK:  # e.g. "low confidence" / "no baseline" — pass through
            return status
        if status == self.reported:
            self._candidate = None
            self._since = None
            return self.reported
        if status != self._candidate:
            self._candidate = status
            self._since = ts
        elif self._since is not None and (ts - self._since) >= self.hold_s:
            self.reported = status
            self._candidate = None
            self._since = None
        return self.reported


class PostureMonitor:
    def __init__(self, forward: float = 1.12, drop: float = 0.85,
                 tilt_thr: float = 7.0, lat_thr: float = 0.18, roll_thr: float = 7.0,
                 min_visibility: float = 0.3):
        self.baseline: PostureFeatures | None = None
        self.forward, self.drop = forward, drop
        self.tilt_thr, self.lat_thr, self.roll_thr = tilt_thr, lat_thr, roll_thr
        self.min_visibility = min_visibility
        self._capture: list[PostureFeatures] | None = None
        self._capture_target = 0

    # --- baseline capture -------------------------------------------------
    def set_baseline(self, feats: PostureFeatures | None) -> bool:
        """Set the baseline immediately from a single sample (back-compat)."""
        if feats is None:
            return False
        self.baseline = feats
        return True

    def begin_baseline(self, n_frames: int = 20) -> None:
        """Begin a median-window baseline capture over the next ``n_frames`` valid samples."""
        self._capture = []
        self._capture_target = max(int(n_frames), 1)

    @property
    def capturing(self) -> bool:
        return self._capture is not None

    def feed_baseline(self, feats: PostureFeatures | None) -> bool:
        """Add a sample to an in-progress capture; returns True when the baseline is set."""
        if self._capture is None:
            return False
        if feats is not None:
            self._capture.append(feats)
        if len(self._capture) >= self._capture_target:
            self.baseline = _median_feats(self._capture)
            self._capture = None
            return True
        return False

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    def save_baseline(self, path) -> bool:
        """Persist the baseline to a JSON file for reuse across sessions."""
        import json

        if self.baseline is None:
            return False
        with open(path, "w") as f:
            json.dump(asdict(self.baseline), f)
        return True

    def load_baseline(self, path) -> bool:
        """Load a previously saved baseline if the file exists."""
        import json
        import os

        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            data.setdefault("roll_deg", 0.0)  # tolerate baselines saved before roll existed
            self.baseline = PostureFeatures(**data)
            return True
        except Exception:
            return False

    # --- assessment -------------------------------------------------------
    def assess(self, feats: PostureFeatures | None, visibility: float | None = None) -> dict:
        if visibility is not None and visibility < self.min_visibility:
            return {"status": "low confidence", "issues": [], "deviation": 0.0}
        if self.baseline is None or feats is None:
            return {"status": "no baseline", "issues": [], "deviation": 0.0}
        b = self.baseline
        sw_r = feats.shoulder_width / (b.shoulder_width + 1e-9)
        neck_r = feats.neck_ratio / (b.neck_ratio + 1e-9)
        lat_d = feats.lateral - b.lateral
        tilt = feats.tilt_deg
        roll_d = feats.roll_deg - b.roll_deg

        closer = sw_r > self.forward
        shorter = neck_r < self.drop
        issues = []
        # forward-head requires BOTH face-closer AND neck-shorter (review §6) — otherwise
        # report the more specific single-cause issue.
        if closer and shorter:
            issues.append("forward-head")
        elif shorter:
            issues.append("head dropped / slouch")
        elif closer:
            issues.append("leaning toward screen")
        if abs(tilt) > self.tilt_thr:
            issues.append("shoulders tilted")
        if abs(lat_d) > self.lat_thr:
            issues.append("leaning " + ("right" if lat_d > 0 else "left"))
        if abs(roll_d) > self.roll_thr:
            issues.append("head tilted")

        dev = max(
            abs(sw_r - 1.0) / (self.forward - 1.0),
            abs(1.0 - neck_r) / (1.0 - self.drop),
            abs(tilt) / (self.tilt_thr + 3.0),
            abs(lat_d) / self.lat_thr,
            abs(roll_d) / (self.roll_thr + 3.0),
        )
        status = "good" if dev < 1.0 else "fair" if dev < 2.0 else "poor"
        return {
            "status": status,
            "issues": issues,
            "deviation": round(float(dev), 2),
            "sw_ratio": round(float(sw_r), 2),
            "neck_ratio": round(float(neck_r), 2),
            "tilt": round(float(tilt), 1),
            "roll": round(float(roll_d), 1),
        }
