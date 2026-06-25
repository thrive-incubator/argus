"""Posture monitor (relative to a captured 'good posture' baseline).

A single frontal webcam can't measure absolute slouch, but against a per-user reference it
can flag the practical issues:
- slouch / forward-head: the head drops toward the shoulders (neck ratio shrinks) and/or the
  face gets closer to the camera (shoulder width grows),
- shoulders tilted sideways,
- leaning left/right.

All geometric features are normalised by shoulder width (scale/distance-invariant); deviations
are measured as ratios to the baseline, so the camera distance/aspect cancels out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 1, 11, 12


@dataclass(frozen=True)
class PostureFeatures:
    shoulder_width: float   # image-normalised, ~ inverse distance to camera
    neck_ratio: float       # (shoulder_mid_y - nose_y) / shoulder_width  (head-up-ness)
    lateral: float          # (nose_x - shoulder_mid_x) / shoulder_width  (sideways shift)
    tilt_deg: float         # shoulder tilt from horizontal


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
    return PostureFeatures(sw, neck, lateral, tilt)


class PostureMonitor:
    def __init__(self, forward: float = 1.12, drop: float = 0.85,
                 tilt_thr: float = 7.0, lat_thr: float = 0.18):
        self.baseline: PostureFeatures | None = None
        self.forward, self.drop = forward, drop
        self.tilt_thr, self.lat_thr = tilt_thr, lat_thr

    def set_baseline(self, feats: PostureFeatures | None) -> bool:
        if feats is None:
            return False
        self.baseline = feats
        return True

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    def assess(self, feats: PostureFeatures | None) -> dict:
        if self.baseline is None or feats is None:
            return {"status": "no baseline", "issues": [], "deviation": 0.0}
        b = self.baseline
        sw_r = feats.shoulder_width / (b.shoulder_width + 1e-9)
        neck_r = feats.neck_ratio / (b.neck_ratio + 1e-9)
        lat_d = feats.lateral - b.lateral
        tilt = feats.tilt_deg

        issues = []
        if sw_r > self.forward or neck_r < self.drop:
            issues.append("slouch / forward-head")
        if abs(tilt) > self.tilt_thr:
            issues.append("shoulders tilted")
        if abs(lat_d) > self.lat_thr:
            issues.append("leaning " + ("right" if lat_d > 0 else "left"))

        dev = max(
            abs(sw_r - 1.0) / (self.forward - 1.0),
            abs(1.0 - neck_r) / (1.0 - self.drop),
            abs(tilt) / (self.tilt_thr + 3.0),
            abs(lat_d) / self.lat_thr,
        )
        status = "good" if dev < 1.0 else "fair" if dev < 2.0 else "poor"
        return {
            "status": status,
            "issues": issues,
            "deviation": round(float(dev), 2),
            "sw_ratio": round(float(sw_r), 2),
            "neck_ratio": round(float(neck_r), 2),
            "tilt": round(float(tilt), 1),
        }
