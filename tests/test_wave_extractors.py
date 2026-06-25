"""Per-frame waveform extractors for the web viz."""

import numpy as np

from argus.backbone.face import SyntheticFaceBackbone
from argus.backbone.pose import SyntheticPoseBackbone
from argus.contracts import FrameContext
from argus.capture.frame_source import SyntheticCamera
from argus.extractors.wave import BreathWaveExtractor, PulseWaveExtractor


def test_pulse_wave_emits_per_frame_and_varies():
    ext = PulseWaveExtractor()
    fb = SyntheticFaceBackbone()
    cam = SyntheticCamera(width=32, height=32, hr_bpm=72.0, n_frames=120)
    vals = []
    i = 0
    while True:
        frame, ok = cam.read()
        if not ok:
            break
        ts = (i + 1) / 30.0
        out = ext.consume(FrameContext(frame=frame, ts=ts, frame_id=i, face=fb.process(frame, ts)))
        assert len(out) == 1 and out[0].name == "pulse_wave"
        vals.append(out[0].value)
        i += 1
    assert np.std(vals) > 0  # it actually oscillates (a waveform, not a constant)


def test_pulse_wave_no_face_no_emit():
    ext = PulseWaveExtractor()
    out = ext.consume(FrameContext(frame=np.zeros((8, 8, 3), np.uint8), ts=1.0, frame_id=0, face=None))
    assert out == []


def test_breath_wave_emits_and_varies():
    ext = BreathWaveExtractor()
    pose = SyntheticPoseBackbone(chest_signal=lambda ts: np.sin(2 * np.pi * 0.25 * ts))
    vals = []
    for i in range(150):
        ts = (i + 1) / 30.0
        out = ext.consume(FrameContext(frame=np.zeros((8, 8, 3), np.uint8), ts=ts,
                                       frame_id=i, pose=pose.process(np.zeros((8, 8, 3), np.uint8), ts)))
        assert len(out) == 1 and out[0].name == "breath_wave"
        vals.append(out[0].value)
    assert np.std(vals) > 0
