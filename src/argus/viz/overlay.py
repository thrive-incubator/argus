"""Debug overlay drawing for the live camera feed.

Draws what the pipeline is tracking on top of the frame:
- rPPG ROI patches (forehead + cheeks), coloured by heart-rate signal quality
- the chest/shoulder points used for respiration
- inline pulse + breathing sparklines (the live rPPG and chest waveforms)
- face mesh, iris, eye (blink) landmarks, a gaze arrow
- a compact HUD of current values
Returns an annotated BGR frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..dsp.roi import FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX
from ..extractors.blink_extractor import RIGHT_EYE_IDX

_PULSE = (115, 93, 255)    # BGR ~ #ff5d73
_GREEN = (155, 209, 90)    # BGR ~ #5ad19b
_IRIS = (255, 120, 255)
_MESH = (90, 100, 110)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_L_SHOULDER, _R_SHOULDER = 11, 12


def _px(lm, idx, w, h):
    return int(lm[idx, 0] * w), int(lm[idx, 1] * h)


def _quality_color(sqi):
    if sqi is None:
        return (255, 182, 63)            # accent (unknown)
    if sqi >= 0.5:
        return (113, 204, 46)            # green
    if sqi >= 0.2:
        return (61, 201, 244)            # amber
    return (115, 93, 255)                # red


def _spark(img, vals, x, y, w, h, color, label):
    a = np.asarray(vals, dtype=float)
    if a.size < 2:
        return
    m = float(np.max(np.abs(a))) + 1e-6
    n = a.size
    pts = [(int(x + w * i / (n - 1)), int(y + h / 2 - (a[i] / m) * (h * 0.45))) for i in range(n)]
    cv2.rectangle(img, (x - 4, y - 14), (x + w + 4, y + h + 4), (20, 26, 34), -1)
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (x, y - 3), _FONT, 0.38, color, 1, cv2.LINE_AA)


def draw_debug(frame: np.ndarray, ctx, hud: dict | None = None,
               waves: dict | None = None, roi_half: int = 14) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    hud = hud or {}
    waves = waves or {}
    face = getattr(ctx, "face", None)
    lm = getattr(face, "landmarks", None) if face is not None else None

    if lm is not None:
        step = max(1, len(lm) // 478)
        for x, y, _z in lm[::step]:
            cv2.circle(img, (int(x * w), int(y * h)), 1, _MESH, -1, cv2.LINE_AA)
        roi_color = _quality_color(hud.get("hr_sqi"))
        for idx in (FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX):
            cx, cy = _px(lm, idx, w, h)
            cv2.rectangle(img, (cx - roi_half, cy - roi_half),
                          (cx + roi_half, cy + roi_half), roi_color, 2, cv2.LINE_AA)
        fx, fy = _px(lm, FOREHEAD_IDX, w, h)
        cv2.putText(img, "rPPG ROI (HR)", (fx - 36, fy - roi_half - 6),
                    _FONT, 0.4, roi_color, 1, cv2.LINE_AA)
        for idx in RIGHT_EYE_IDX:
            cv2.circle(img, _px(lm, idx, w, h), 2, _GREEN, -1, cv2.LINE_AA)
        if lm.shape[0] >= 478:
            for idx in (468, 473):
                cv2.circle(img, _px(lm, idx, w, h), 3, _IRIS, -1, cv2.LINE_AA)
            from ..perception.gaze import iris_gaze_angles
            yaw, pitch = iris_gaze_angles(lm)
            ex = int((lm[468, 0] + lm[473, 0]) / 2 * w)
            ey = int((lm[468, 1] + lm[473, 1]) / 2 * h)
            cv2.arrowedLine(img, (ex, ey), (int(ex + yaw * 4), int(ey + pitch * 4)),
                            (180, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

    # respiration: the chest/shoulder points actually tracked
    pose = getattr(ctx, "pose", None)
    pim = getattr(pose, "image_landmarks", None) if pose is not None else None
    if pim is not None and pim.shape[0] > _R_SHOULDER:
        ls, rs = _px(pim, _L_SHOULDER, w, h), _px(pim, _R_SHOULDER, w, h)
        cv2.line(img, ls, rs, _GREEN, 2, cv2.LINE_AA)
        for p in (ls, rs):
            cv2.circle(img, p, 5, _GREEN, -1, cv2.LINE_AA)
        midx = (ls[0] + rs[0]) // 2
        cv2.putText(img, "chest (respiration)", (midx - 60, ls[1] + 22),
                    _FONT, 0.4, _GREEN, 1, cv2.LINE_AA)

    # inline waveforms
    sw, sh = 150, 34
    _spark(img, waves.get("pulse", []), 14, h - 2 * sh - 26, sw, sh, _PULSE, "pulse (rPPG)")
    _spark(img, waves.get("breath", []), 14, h - sh - 12, sw, sh, _GREEN, "breathing")

    _draw_hud(img, hud)
    return img


def _draw_hud(img, hud: dict) -> None:
    lines = []
    if "hr" in hud:
        q = hud.get("hr_sqi")
        lines.append((f"HR {hud['hr']:.0f} bpm" + (f"  q{q:.2f}" if q is not None else ""), _PULSE))
    if "resp" in hud:
        lines.append((f"RESP {hud['resp']:.1f} brpm", _GREEN))
    if "emotion" in hud:
        lines.append((f"AFFECT {hud['emotion']}", (255, 182, 63)))
    if "gaze" in hud:
        lines.append((f"GAZE {hud['gaze']}", (200, 200, 200)))
    y = 24
    for text, color in lines:
        cv2.putText(img, text, (12, y), _FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), _FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 24


def annotations(ctx, metrics: dict | None = None, mesh_step: int = 6) -> dict:
    """Vector annotations (normalised coords) for the browser to draw over the raw frame.

    Higher quality than baking pixels into the JPEG, and lets each layer be toggled in JS.
    """
    out: dict = {"metrics": metrics or {}}
    face = getattr(ctx, "face", None)
    lm = getattr(face, "landmarks", None) if face is not None else None
    if lm is not None:
        out["mesh"] = [[float(lm[i, 0]), float(lm[i, 1])] for i in range(0, len(lm), mesh_step)]
        out["roi"] = [[float(lm[i, 0]), float(lm[i, 1])]
                      for i in (FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX)]
        out["eyes"] = [[float(lm[i, 0]), float(lm[i, 1])] for i in RIGHT_EYE_IDX]
        if lm.shape[0] >= 478:
            out["iris"] = [[float(lm[i, 0]), float(lm[i, 1])] for i in (468, 473)]
            from ..perception.gaze import iris_gaze_angles
            yaw, pitch = iris_gaze_angles(lm)
            ex = (lm[468, 0] + lm[473, 0]) / 2.0
            ey = (lm[468, 1] + lm[473, 1]) / 2.0
            out["gaze"] = [float(ex), float(ey), float(yaw), float(pitch)]
    pose = getattr(ctx, "pose", None)
    pim = getattr(pose, "image_landmarks", None) if pose is not None else None
    if pim is not None and pim.shape[0] > _R_SHOULDER:
        out["shoulders"] = [[float(pim[_L_SHOULDER, 0]), float(pim[_L_SHOULDER, 1])],
                            [float(pim[_R_SHOULDER, 0]), float(pim[_R_SHOULDER, 1])]]
    return out


def encode_jpeg_b64(img: np.ndarray, max_width: int = 540, quality: int = 60) -> str:
    import base64

    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else ""
