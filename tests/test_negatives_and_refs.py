"""Withdrawn/negative ACs (F3, G3, J2, NFR-6) + reference cross-checks + A2.AC2/A4.AC1."""

import importlib.util
import inspect
import time
from pathlib import Path

import numpy as np

from argus.backbone.face import DEFAULT_NUM_FACES, MediaPipeFaceBackbone, SyntheticFaceBackbone
from argus.contracts import Extractor, FrameContext
from argus.capture.frame_source import SyntheticCamera
from argus.core.pipeline import Pipeline
from argus.dsp.rppg import pos
from argus.extractors import RppgExtractor

SRC = Path(__file__).resolve().parents[1] / "src" / "argus"


# F3.AC1 — pupillometry is explicitly NOT implemented.
def test_no_pupillometry_feature():
    assert importlib.util.find_spec("argus.dsp.pupil") is None
    assert importlib.util.find_spec("argus.perception.pupil") is None
    # no extractor registered for pupil
    assert not any("pupil" in name for name in Extractor.REGISTRY)
    # no source file mentions a pupil-size feature
    hits = [p.name for p in SRC.rglob("*.py") if "pupil" in p.read_text().lower()
            and "pupillometry" not in p.read_text().lower()]
    assert hits == []


# G3 / J2 / NFR-6 — licensing firewall / quarantine WITHDRAWN: no such machinery.
def test_no_licensing_firewall():
    assert importlib.util.find_spec("argus.research") is None
    assert not (SRC / "research").exists()
    # no import-lint / quarantine module anywhere
    for p in SRC.rglob("*.py"):
        text = p.read_text()
        assert "import_lint" not in text
        assert "QUARANTINE" not in text


# A2.AC2 — num_faces=1 is the documented default (temporal smoothing).
def test_face_backbone_num_faces_default_is_one():
    assert DEFAULT_NUM_FACES == 1
    sig = inspect.signature(MediaPipeFaceBackbone.__init__)
    assert sig.parameters["num_faces"].default == 1


# D2.AC2 — POS matches an INDEPENDENT reference implementation on a fixed clip.
def test_pos_matches_independent_reference():
    rng = np.random.default_rng(3)
    n = 300
    t = np.arange(n) / 30.0
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.array([0.8, 0.5, 0.4])[None, :] + np.array([0.01, 0.03, 0.005])[None, :] * pulse[:, None]
    rgb += 0.001 * rng.standard_normal((n, 3))

    def ref_pos(x):  # independent textbook POS (Wang 2017)
        C = x / x.mean(axis=0)
        P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
        S = P @ C.T
        alpha = S[0].std() / S[1].std()
        h = S[0] + alpha * S[1]
        return h - h.mean()

    np.testing.assert_allclose(pos(rgb, 30.0), ref_pos(rgb), rtol=1e-9, atol=1e-9)


# A4.AC1 (headless proxy) — the synthetic pipeline keeps up well above real-time.
def test_pipeline_throughput_proxy():
    pipe = Pipeline(extractors=[RppgExtractor(fps=30.0)],
                    face_backbone=SyntheticFaceBackbone(), emit_clock=lambda: 0.0)
    n = 300  # 10 s of frames
    t0 = time.perf_counter()
    pipe.run_source(SyntheticCamera(width=64, height=64, n_frames=n), max_frames=n)
    elapsed = time.perf_counter() - t0
    assert pipe.metrics.frames == n
    assert elapsed < 10.0  # processes 10 s of frames far faster than real time
