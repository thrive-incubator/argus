"""Frame sources (FR-1). A ``FrameSource`` yields ``(frame, ok)``.

- ``OpenCVCamera``: real webcam adapter (cv2). The single untested device-driver line
  is ``cv2.VideoCapture(index)`` opening physical hardware (listed in SUMMARY.md).
- ``SyntheticCamera``: deterministic synthetic frames carrying an embedded rPPG pulse,
  so the *whole* pipeline can run end-to-end headlessly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    def read(self) -> tuple[np.ndarray | None, bool]:
        """Return ``(frame, ok)``; ``ok`` False signals end/failure."""
        ...

    def release(self) -> None: ...


class OpenCVCamera:
    """Real webcam via OpenCV with CAP_PROP_BUFFERSIZE=1 (A1.AC1).

    ``mirror=True`` horizontally flips each frame to a natural selfie view, so the camera
    preview matches a mirror and gaze left/right is egocentric (your left = "left").
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 mirror: bool = False):
        import cv2  # local import: only needed for the real device path

        self._cv2 = cv2
        self.mirror = mirror
        self._cap = cv2.VideoCapture(index)  # <-- untested device-driver line
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> tuple[np.ndarray | None, bool]:
        ok, frame = self._cap.read()
        if ok and self.mirror:
            frame = self._cv2.flip(frame, 1)  # horizontal flip -> selfie view
        return (frame if ok else None), bool(ok)

    def release(self) -> None:  # pragma: no cover - device
        self._cap.release()


class SyntheticCamera:
    """Deterministic frames with an embedded blood-volume pulse at ``hr_bpm``.

    Each frame is HxWx3; the per-frame mean carries a sinusoidal pulse so rPPG/POS can
    recover ``hr_bpm`` from a stream of these frames.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        hr_bpm: float = 72.0,
        n_frames: int | None = None,
        noise: float = 0.5,
        seed: int = 0,
    ):
        self.width, self.height, self.fps = width, height, fps
        self.hr_bpm, self.n_frames = hr_bpm, n_frames
        self._rng = np.random.default_rng(seed)
        self._i = 0
        self._base = np.array([180.0, 120.0, 100.0])  # BGR-ish DC
        self._amp = np.array([3.0, 9.0, 1.5])  # green carries strongest pulse
        self._noise = noise

    def read(self) -> tuple[np.ndarray | None, bool]:
        if self.n_frames is not None and self._i >= self.n_frames:
            return None, False
        t = self._i / self.fps
        pulse = np.sin(2 * np.pi * (self.hr_bpm / 60.0) * t)
        color = self._base + self._amp * pulse
        frame = np.empty((self.height, self.width, 3), dtype=np.float64)
        frame[:] = color
        frame += self._noise * self._rng.standard_normal(frame.shape)
        self._i += 1
        return np.clip(frame, 0, 255).astype(np.uint8), True

    def release(self) -> None:
        pass


class BlankFrameSource:
    """Yields all-zero frames (used to simulate 'no face' / black-frame failure)."""

    def __init__(self, width: int = 64, height: int = 64, n_frames: int = 10):
        self.width, self.height, self.n = width, height, n_frames
        self._i = 0

    def read(self) -> tuple[np.ndarray | None, bool]:
        if self._i >= self.n:
            return None, False
        self._i += 1
        return np.zeros((self.height, self.width, 3), dtype=np.uint8), True

    def release(self) -> None:
        pass
