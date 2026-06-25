"""FR-1, A1.AC1-4, FR-4/A4.AC2 (buffers), NFR-8 counters."""

import numpy as np
import pytest

from argus.capture.buffers import LatestFrameSlot, TimeSeriesRing
from argus.capture.calibration import detect_flash_index, exposure_offset
from argus.capture.capture_thread import CaptureThread
from argus.capture.clock import FakeClock
from argus.capture.frame_source import BlankFrameSource, SyntheticCamera


# A1.AC1 — synthetic source honours configured size; OpenCVCamera sets buffersize=1.
def test_synthetic_camera_size_and_fps():
    cam = SyntheticCamera(width=1280, height=720, fps=30.0, n_frames=3)
    frame, ok = cam.read()
    assert ok and frame.shape == (720, 1280, 3) and frame.dtype == np.uint8


def test_opencv_camera_uses_buffersize_one():
    # Drive the real adapter against a fake cv2 to prove BUFFERSIZE=1 is set (the only
    # untested line is the actual VideoCapture device open).
    import argus.capture.frame_source as fs

    calls = {}

    class FakeCap:
        def set(self, prop, val):
            calls[prop] = val

        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            calls["released"] = True

    class FakeCv2:
        CAP_PROP_BUFFERSIZE = 38
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5

        def VideoCapture(self, idx):
            return FakeCap()

    import sys

    sys.modules_backup = sys.modules.get("cv2")
    sys.modules["cv2"] = FakeCv2()
    try:
        cam = fs.OpenCVCamera(index=0, width=1280, height=720, fps=30)
        assert calls[FakeCv2.CAP_PROP_BUFFERSIZE] == 1
        assert calls[FakeCv2.CAP_PROP_FRAME_WIDTH] == 1280
    finally:
        if sys.modules_backup is not None:
            sys.modules["cv2"] = sys.modules_backup
        else:
            del sys.modules["cv2"]


# mirror=True flips the frame horizontally (selfie view); off by default.
def test_opencv_camera_mirror_flips_horizontally():
    import sys

    import argus.capture.frame_source as fs

    asymmetric = np.zeros((2, 3, 3), np.uint8)
    asymmetric[:, 0, :] = 255  # bright on the LEFT column

    class FakeCap:
        def set(self, *a):
            pass

        def read(self):
            return True, asymmetric.copy()

    class FakeCv2:
        CAP_PROP_BUFFERSIZE = CAP_PROP_FRAME_WIDTH = CAP_PROP_FRAME_HEIGHT = CAP_PROP_FPS = 0

        def VideoCapture(self, idx):
            return FakeCap()

        def flip(self, img, code):
            import numpy as _np
            return _np.flip(img, axis=1) if code == 1 else img

    backup = sys.modules.get("cv2")
    sys.modules["cv2"] = FakeCv2()
    try:
        plain, _ = fs.OpenCVCamera(mirror=False).read()
        flipped, _ = fs.OpenCVCamera(mirror=True).read()
        assert plain[0, 0, 0] == 255 and plain[0, -1, 0] == 0    # bright stays on left
        assert flipped[0, 0, 0] == 0 and flipped[0, -1, 0] == 255  # bright moved to right
    finally:
        if backup is not None:
            sys.modules["cv2"] = backup
        else:
            del sys.modules["cv2"]


# A1.AC2 — frame ts from injected clock at grab; strictly monotonic.
def test_capture_timestamps_monotonic():
    slot = LatestFrameSlot()
    clk = FakeClock(start=100.0, step=1 / 30)
    cap = CaptureThread(SyntheticCamera(width=8, height=8, n_frames=20), slot, clock=clk)
    seen = []
    # tap the slot each grab
    import argus.capture.capture_thread as ct

    n = cap.run_sync()
    assert n == 20 and cap.frames == 20
    # monotonicity is enforced internally; verify via a fresh run reading the slot
    slot2 = LatestFrameSlot()
    cap2 = CaptureThread(SyntheticCamera(width=8, height=8, n_frames=5), slot2, clock=FakeClock())
    tss = []
    for _ in range(5):
        cap2._next()
        item = slot2.get()
        if item:
            tss.append(item[1])
    assert all(b > a for a, b in zip(tss, tss[1:]))


def test_capture_enforces_monotonic_with_constant_clock():
    slot = LatestFrameSlot()
    cap = CaptureThread(SyntheticCamera(width=8, height=8, n_frames=4), slot, clock=lambda: 5.0)
    tss = []
    for _ in range(4):
        cap._next()
        tss.append(slot.get()[1])
    assert all(b > a for a, b in zip(tss, tss[1:]))  # bumped despite constant clock


# A1.AC3 / A4.AC2 — slot drops oldest under a slow consumer; ring loses zero.
def test_latest_frame_slot_drops_oldest():
    slot = LatestFrameSlot()
    for i in range(5):
        slot.set(("f", i))
    assert slot.dropped == 4  # only the last survives
    assert slot.get() == ("f", 4)
    assert slot.get() is None


def test_timeseries_ring_is_lossless():
    ring = TimeSeriesRing(capacity=1000)
    for i in range(500):
        ring.append(i / 30.0, float(i))
    assert ring.appended == 500  # zero lost
    t, v = ring.window(seconds=1.0)
    assert t.size == v.size and t.size <= 31


def test_slot_drops_while_ring_lossless_under_load():
    slot = LatestFrameSlot()
    ring = TimeSeriesRing()
    produced = 200
    for i in range(produced):
        slot.set((i,))  # per-frame path: consumer is "slow" (never reads) -> drops
        ring.append(i / 30.0, float(i))  # lossless path
    assert slot.dropped == produced - 1
    assert ring.appended == produced  # zero samples lost on the lossless ring


# A1.AC4 — LED-flash exposure calibration.
def test_exposure_offset_from_flash():
    brightness = np.array([10, 10, 11, 10, 200, 205, 203])  # flash at index 4
    capture_ts = np.array([0.0, 0.033, 0.066, 0.099, 0.132, 0.165, 0.198])
    assert detect_flash_index(brightness) == 4
    off = exposure_offset(brightness, capture_ts, flash_emit_ts=0.100)
    assert off == pytest.approx(0.032, abs=1e-6)  # host latency 0.132 - 0.100


def test_capture_subtracts_exposure_offset():
    slot = LatestFrameSlot()
    cap = CaptureThread(
        SyntheticCamera(width=4, height=4, n_frames=1),
        slot,
        clock=lambda: 10.0,
        exposure_offset_s=0.05,
    )
    cap._next()
    _, ts, _ = slot.get()
    assert ts == pytest.approx(9.95)


# A2.AC3 prep — blank source yields all-zero frames (no-face simulation downstream).
def test_blank_source():
    src = BlankFrameSource(n_frames=2)
    f, ok = src.read()
    assert ok and not f.any()
