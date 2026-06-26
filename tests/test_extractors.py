"""FR-5/7/8/12 — extractor plugins driven by synthetic backbones end-to-end."""

import numpy as np
import pytest

from argus.backbone.face import SyntheticFaceBackbone
from argus.backbone.pose import SyntheticPoseBackbone
from argus.contracts import FrameContext
from argus.capture.frame_source import SyntheticCamera
from argus.extractors import (
    BlinkExtractor,
    HrvExtractor,
    MotionExtractor,
    RespirationExtractor,
    RppgExtractor,
)


def drive(extractor, n_frames, fps=30.0, hr_bpm=72.0, face=True, pose=None):
    """Feed an extractor a synthetic frame stream; return all emitted records."""
    cam = SyntheticCamera(width=64, height=64, fps=fps, hr_bpm=hr_bpm, n_frames=n_frames)
    fb = SyntheticFaceBackbone() if face else None
    out = []
    i = 0
    while True:
        frame, ok = cam.read()
        if not ok:
            break
        ts = i / fps + 0.001 * (i == 0)  # strictly > 0, increasing
        ts = (i + 1) / fps
        f = fb.process(frame, ts) if fb else None
        p = pose.process(frame, ts) if pose else None
        ctx = FrameContext(frame=frame, ts=ts, frame_id=i, face=f, pose=p)
        out.extend(extractor.consume(ctx))
        i += 1
    return out


# FR-5 / D2 — HR via POS recovers ~72 bpm, in band, ~1 Hz update, gate "unknown".
def test_rppg_extractor_recovers_hr():
    ext = RppgExtractor(fps=30.0, window_s=10.0, min_window_s=8.0, update_period_s=1.0)
    recs = drive(ext, n_frames=int(30 * 12))
    hrs = [r for r in recs if r.name == "hr"]
    assert len(hrs) >= 2  # ~1 Hz updates over the back ~4 s
    last = hrs[-1]
    assert 42.0 <= last.value <= 240.0
    assert abs(last.value - 72.0) <= 6.0
    assert last.gate == "unknown"
    assert 0.0 <= last.sqi <= 1.0


def test_rppg_extractor_no_face_no_emit():
    ext = RppgExtractor()
    recs = drive(ext, n_frames=300, face=False)
    assert recs == []


# Regression: HR must still emit when live throughput is well below the nominal fps
# (time-span gate, not frame-count) — e.g. heavy per-frame work drops the loop to ~10 fps.
def test_rppg_extractor_emits_at_low_throughput():
    ext = RppgExtractor(fps=30.0, window_s=10.0, min_window_s=8.0, update_period_s=1.0)
    fb = SyntheticFaceBackbone()
    cam = SyntheticCamera(width=64, height=64, fps=30.0, hr_bpm=72.0, n_frames=130)
    out = []
    for i in range(130):
        frame, ok = cam.read()
        if not ok:
            break
        ts = (i + 1) / 10.0  # only ~10 fps of wall-clock throughput
        f = fb.process(frame, ts)
        out.extend(ext.consume(FrameContext(frame=frame, ts=ts, frame_id=i, face=f)))
    hrs = [r for r in out if r.name == "hr"]
    assert len(hrs) >= 1                         # emits despite low fps
    assert 42.0 <= hrs[-1].value <= 240.0


# FR-6 / D3 — HRV extractor emits SDNN/RMSSD once the window fills; insufficient otherwise.
def test_hrv_extractor_emits_after_window():
    ext = HrvExtractor(fps=30.0, window_s=20.0, min_window_s=15.0, update_period_s=5.0)
    recs = drive(ext, n_frames=int(30 * 22))
    hrv = [r for r in recs if r.name == "hrv"]
    assert len(hrv) >= 1
    r = hrv[-1]
    assert np.isfinite(r.value)  # SDNN ms
    assert r.meta["rest_only"] is True
    assert "rmssd_ms" in r.meta and r.meta["n_beats"] >= 5


def test_hrv_extractor_holds_until_window_full():
    ext = HrvExtractor(fps=30.0, window_s=20.0, min_window_s=15.0)
    recs = drive(ext, n_frames=int(30 * 5))  # only 5 s
    assert [r for r in recs if r.name == "hrv"] == []


# FR-7 / E1 — respiration recovers a known breathing rate from chest motion.
def test_respiration_extractor_recovers_rate():
    brpm = 15.0
    chest = lambda ts: np.sin(2 * np.pi * (brpm / 60.0) * ts)
    pose = SyntheticPoseBackbone(chest_signal=chest)
    ext = RespirationExtractor(fps=30.0, window_s=20.0, min_window_s=15.0, update_period_s=2.0)
    recs = drive(ext, n_frames=int(30 * 22), face=False, pose=pose)
    resp = [r for r in recs if r.name == "resp"]
    assert len(resp) >= 1
    assert abs(resp[-1].value - brpm) <= 2.0
    assert resp[-1].meta["indicative"] is True


# FR-8 / F1 — blink extractor counts blinks via injected EAR; withholds until armed.
def test_blink_extractor_counts_and_withholds():
    # scripted EAR: baseline open (0.30), then 3 blinks
    series = [0.30] * 30 + ([0.10, 0.10, 0.10, 0.30, 0.30] * 3)
    it = iter(series)
    ext = BlinkExtractor(fps=30.0, baseline_frames=30, ratio=0.6, min_frames=2,
                         ear_fn=lambda ctx: next(it, 0.30))
    out = []
    for i in range(len(series)):
        ctx = FrameContext(frame=np.zeros((4, 4, 3), np.uint8), ts=(i + 1) / 30.0,
                           frame_id=i, face=None)
        out.extend(ext.consume(ctx))
    events = [r for r in out if r.name == "blink_event"]
    rates = [r for r in out if r.name == "blink_rate"]
    assert len(events) == 3
    assert rates and rates[-1].value > 0.0
    assert ext.armed


def test_blink_fps_warning_flag():
    ext = BlinkExtractor(fps=10.0, baseline_frames=2, ear_fn=lambda ctx: 0.3)
    assert ext.fps_warning is True


# FR-12 — fidget index is higher for a jittery subject than a still one.
def test_motion_fidget_distinguishes_jitter():
    def run(jitter):
        pose = SyntheticPoseBackbone(chest_signal=None, filtered=False)
        ext = MotionExtractor(fps=30.0, window_s=5.0, update_period_s=1.0)
        cam = SyntheticCamera(width=16, height=16, n_frames=180)
        rng = np.random.default_rng(0)
        last = None
        i = 0
        while True:
            frame, ok = cam.read()
            if not ok:
                break
            ts = (i + 1) / 30.0
            p = pose.process(frame, ts)
            # inject extra jitter into shoulder landmarks
            p.landmarks[[11, 12]] += rng.normal(0, jitter, (2, 3))
            ctx = FrameContext(frame=frame, ts=ts, frame_id=i, face=None, pose=p)
            recs = ext.consume(ctx)
            for r in recs:
                if r.name == "fidget":
                    last = r.value
            i += 1
        return last

    still = run(0.0)
    jittery = run(0.05)
    assert jittery > still
