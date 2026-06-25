"""Debug overlay drawing for the live camera feed.

Draws what the pipeline is tracking on top of the frame: the face mesh points, the rPPG ROI
patches (forehead + cheeks), iris points, the eye landmarks used for blink (EAR), a gaze
arrow, and a compact HUD of the current signal values. Returns an annotated BGR frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..dsp.roi import FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX
from ..extractors.blink_extractor import RIGHT_EYE_IDX

_ACCENT = (255, 182, 63)   # BGR ~ #3fb6ff
_PULSE = (115, 93, 255)    # BGR ~ #ff5d73
_GREEN = (155, 209, 90)    # BGR ~ #5ad19b
_IRIS = (255, 120, 255)
_MESH = (90, 100, 110)


def _px(lm, idx, w, h):
    return int(lm[idx, 0] * w), int(lm[idx, 1] * h)


def draw_debug(frame: np.ndarray, ctx, hud: dict | None = None, roi_half: int = 14) -> np.ndarray:
    """Annotate a copy of ``frame`` with the pipeline's tracking + a HUD."""
    img = frame.copy()
    h, w = img.shape[:2]
    face = getattr(ctx, "face", None)
    lm = getattr(face, "landmarks", None) if face is not None else None

    if lm is not None:
        # faint face mesh
        step = max(1, len(lm) // 478)
        for x, y, _z in lm[::step]:
            cv2.circle(img, (int(x * w), int(y * h)), 1, _MESH, -1, cv2.LINE_AA)
        # rPPG ROI patches
        for idx in (FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX):
            cx, cy = _px(lm, idx, w, h)
            cv2.rectangle(img, (cx - roi_half, cy - roi_half),
                          (cx + roi_half, cy + roi_half), _ACCENT, 2, cv2.LINE_AA)
        cv2.putText(img, "rPPG ROI", (_px(lm, FOREHEAD_IDX, w, h)[0] - 30,
                    _px(lm, FOREHEAD_IDX, w, h)[1] - roi_half - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _ACCENT, 1, cv2.LINE_AA)
        # eye landmarks used for blink (EAR)
        for idx in RIGHT_EYE_IDX:
            cv2.circle(img, _px(lm, idx, w, h), 2, _GREEN, -1, cv2.LINE_AA)
        # iris + gaze arrow
        if lm.shape[0] >= 478:
            for idx in (468, 473):
                cv2.circle(img, _px(lm, idx, w, h), 3, _IRIS, -1, cv2.LINE_AA)
            from ..perception.gaze import iris_gaze_angles
            yaw, pitch = iris_gaze_angles(lm)
            ex = int((lm[468, 0] + lm[473, 0]) / 2 * w)
            ey = int((lm[468, 1] + lm[473, 1]) / 2 * h)
            cv2.arrowedLine(img, (ex, ey), (int(ex + yaw * 4), int(ey + pitch * 4)),
                            (180, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

    _draw_hud(img, hud or {})
    return img


def _draw_hud(img, hud: dict) -> None:
    h, w = img.shape[:2]
    lines = []
    if "hr" in hud:
        lines.append((f"HR {hud['hr']:.0f} bpm", _PULSE))
    if "resp" in hud:
        lines.append((f"RESP {hud['resp']:.1f} brpm", _GREEN))
    if "emotion" in hud:
        lines.append((f"AFFECT {hud['emotion']}", _ACCENT))
    if "gaze" in hud:
        lines.append((f"GAZE {hud['gaze']}", (200, 200, 200)))
    y = 24
    for text, color in lines:
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 24


def encode_jpeg_b64(img: np.ndarray, max_width: int = 540, quality: int = 60) -> str:
    """Resize for bandwidth, JPEG-encode, return a base64 string (no data: prefix)."""
    import base64

    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")
