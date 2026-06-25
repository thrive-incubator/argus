"""FR-9/F2 gaze, FR-10/G1 affect, FR-11/G2 AUs."""

import numpy as np
import pytest

from argus.contracts import FrameContext
from argus.perception.affect import (
    AffectEstimate,
    AffectExtractor,
    BlendshapeNormalizer,
    FakeEmotionEstimator,
    face_validity_report,
)
from argus.perception.au import AU_KEYS, AuExtractor, FakeAuEstimator
from argus.perception.gaze import (
    GazeCalibration,
    accuracy,
    accuracy_above_chance,
    confusion_matrix,
    gaze_to_zone,
)
from argus.backbone.types import N_BLENDSHAPES


def _ctx(ts, face=True):
    class F:
        pass

    return FrameContext(frame=np.full((8, 8, 3), 50, np.uint8), ts=ts, frame_id=0,
                        face=(F() if face else None))


# F2.AC2 — zones, never pixels.
def test_gaze_to_zone():
    z = gaze_to_zone(0.0, -15.0)
    assert z["horizontal"] == "left" and z["screen"] == "on" and z["attention"] == "present"
    assert gaze_to_zone(0.0, 30.0)["screen"] == "off"
    assert "x" not in z and "y" not in z  # no pixel coordinates


# F2.AC3 — confusion matrix + above-chance over a scripted look-target protocol.
def test_gaze_confusion_above_chance():
    # targets at strong eccentricities -> zone classification well above 33% chance
    targets = {"left": -25.0, "center": 0.0, "right": 25.0}
    true_l, pred_l = [], []
    rng = np.random.default_rng(0)
    for label, yaw in targets.items():
        for _ in range(20):
            noisy = yaw + rng.normal(0, 5.0)
            true_l.append(label)
            pred_l.append(gaze_to_zone(0.0, noisy)["horizontal"])
    cm = confusion_matrix(true_l, pred_l, ["left", "center", "right"])
    assert cm.shape == (3, 3) and cm.sum() == 60
    acc = accuracy(true_l, pred_l)
    assert accuracy_above_chance(acc, 3) > 0.3  # well above chance


# F2.AC4 — calibration improves zone/angle error.
def test_gaze_calibration_reduces_error():
    rng = np.random.default_rng(1)
    targets = rng.uniform(-20, 20, (9, 2))
    raw = targets * 1.3 + np.array([5.0, -4.0]) + rng.normal(0, 0.5, targets.shape)
    cal = GazeCalibration()
    err_before = np.linalg.norm(raw - targets, axis=1).mean()
    cal.fit(raw, targets)
    corrected = np.array([cal.apply(p, y) for p, y in raw])
    err_after = np.linalg.norm(corrected - targets, axis=1).mean()
    assert err_after < err_before * 0.5


# G1.AC1 — blendshape neutral-subtract + z-score.
def test_blendshape_normalizer():
    norm = BlendshapeNormalizer(baseline_frames=20)
    rng = np.random.default_rng(0)
    base = np.full(N_BLENDSHAPES, 0.2)
    for _ in range(20):
        assert norm.update_and_normalize(base + rng.normal(0, 0.01, N_BLENDSHAPES)) is None
    assert norm.armed
    out = norm.update_and_normalize(base)  # ~ baseline -> near zero
    assert np.abs(out).mean() < 3.0
    big = norm.update_and_normalize(base + 1.0)  # large deviation -> large z
    assert np.abs(big).mean() > np.abs(out).mean()


# G1.AC2/AC3 — affect extractor emits V/A as estimates at the live cadence.
def test_affect_extractor_estimates_and_cadence():
    est = FakeEmotionEstimator(AffectEstimate("happy", 0.7, 0.3, 0.9))
    ext = AffectExtractor(est, fps=30.0, live_hz=10.0)
    emitted = []
    for i in range(60):  # 2 s at 30 fps
        emitted.extend(ext.consume(_ctx((i + 1) / 30.0)))
    val = [r for r in emitted if r.name == "affect_valence"]
    assert val and val[0].meta["label"] == "estimate" and val[0].meta["is_verdict"] is False
    assert val[0].sqi == pytest.approx(0.9)  # confidence as sqi
    # ~10 Hz cadence over 2 s -> far fewer than 60 emissions
    assert 15 <= len(val) <= 25


# G1.AC4 — face-validity report.
def test_face_validity_report():
    rep = face_validity_report([0.6, 0.7, 0.65], [-0.4, -0.5, -0.3])
    assert rep["happy_gt_sad"] is True
    assert rep["effect_size_d"] > 0.8
    assert "posed != felt" in rep["note"]


# FR-11 / G2 — AU extractor: 0–5 intensities, research-flagged, decoupled cadence.
def test_au_extractor_intensities_and_cadence():
    ext = AuExtractor(FakeAuEstimator({k: 2.5 for k in AU_KEYS}), au_hz=10.0)
    out = []
    for i in range(60):  # 2 s at 30 fps
        out.extend(ext.consume(_ctx((i + 1) / 30.0)))
    assert all(r.meta["research"] and 0.0 <= r.value <= 5.0 for r in out)
    assert ext.runs <= 22  # decoupled ~10 Hz, not 60
    assert any(r.name == "au_AU12" for r in out)


# FR-9 — gaze extractor emits numeric zone codes with the zone dict in meta.
def test_gaze_extractor_emits_zone():
    from argus.perception.gaze import FakeGazeEstimator, GazeExtractor

    ext = GazeExtractor(FakeGazeEstimator(pitch=0.0, yaw=-25.0), fps=30.0, hz=10.0)
    out = []
    for i in range(30):
        out.extend(ext.consume(_ctx((i + 1) / 30.0)))
    assert out and out[0].name == "gaze_zone"
    assert out[0].value == -1.0  # left
    assert out[0].meta["zone"]["horizontal"] == "left"


# Iris geometric gaze (no model) — centered vs shifted.
def test_iris_gaze_angles():
    from argus.perception.gaze import iris_gaze_angles
    lm = np.zeros((478, 3))
    lm[33]=[0.2,0.5,0]; lm[133]=[0.4,0.5,0]; lm[159]=[0.45,0.45,0]; lm[145]=[0.45,0.55,0]
    lm[263]=[0.8,0.5,0]; lm[362]=[0.6,0.5,0]; lm[386]=[0.65,0.45,0]; lm[374]=[0.65,0.55,0]
    lm[468]=[0.3,0.5,0]; lm[473]=[0.7,0.5,0]            # iris centered
    yaw, pitch = iris_gaze_angles(lm)
    assert abs(yaw) < 2 and abs(pitch) < 5
    lm[468]=[0.35,0.5,0]; lm[473]=[0.75,0.5,0]          # both irises shifted +x
    yaw2, _ = iris_gaze_angles(lm)
    assert yaw2 > yaw + 10  # detectably looking to one side


# Screen-gaze feature vector (for calibration).
def test_gaze_features():
    from argus.perception.gaze import gaze_features
    lm = np.zeros((478, 3))
    lm[133]=[.4,.5,0]; lm[33]=[.2,.5,0]; lm[159]=[.3,.45,0]; lm[145]=[.3,.55,0]; lm[468]=[.3,.5,0]
    lm[362]=[.6,.5,0]; lm[263]=[.8,.5,0]; lm[386]=[.7,.45,0]; lm[374]=[.7,.55,0]; lm[473]=[.7,.5,0]
    lm[1]=[.5,.6,0]
    f = gaze_features(lm)
    assert len(f) == 6
    assert abs(f[0]) < 0.1                         # centered iris -> ~0 eye offset
    assert abs(f[2]) < 0.1 and abs(f[3]) < 0.1    # no head_pose -> 0 head yaw/pitch
    assert f[4] == 0.5 and f[5] == 0.6           # nose position carried through
    lm[468] = [.35, .5, 0]                         # shift right-eye iris
    assert gaze_features(lm)[0] > f[0]
    th = np.radians(30)                            # head rotation enters the features
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    M = np.eye(4); M[:3, :3] = Ry
    assert abs(gaze_features(lm, M)[2]) > 0.2      # head yaw now non-zero


# Affect derived from FACS Action Units (EMFACS) — the robust, interpretable path.
def test_au_to_valence_arousal_and_emotion():
    from argus.perception.affect import au_to_emotion, au_to_valence_arousal
    smile = {"AU06": 0.9, "AU12": 1.0}
    v, a = au_to_valence_arousal(smile)
    assert v > 0.7 and au_to_emotion(smile) == "happiness"
    frown = {"AU04": 0.8, "AU15": 0.8, "AU01": 0.6}
    v2, _ = au_to_valence_arousal(frown)
    assert v2 < -0.2 and au_to_emotion(frown) == "sadness"
    wide = {"AU05": 0.8, "AU02": 0.7, "AU26": 0.7, "AU25": 0.8}
    _, a3 = au_to_valence_arousal(wide)
    assert a3 > 0.4
    assert au_to_emotion({}) == "neutral"   # no AUs -> neutral
