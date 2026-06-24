"""Capture thread (FR-1, A1, NFR-8).

Grabs frames from a ``FrameSource`` on a dedicated thread, stamps each with the injected
clock at grab time (minus a calibrated exposure offset), enforces monotonic timestamps,
and writes to a ``LatestFrameSlot``. Exposes counters for observability (NFR-8).
"""

from __future__ import annotations

import threading

from .buffers import LatestFrameSlot
from .clock import Clock, local_clock
from .frame_source import FrameSource


class CaptureThread:
    def __init__(
        self,
        source: FrameSource,
        slot: LatestFrameSlot,
        clock: Clock = local_clock,
        exposure_offset_s: float = 0.0,
    ) -> None:
        self.source = source
        self.slot = slot
        self.clock = clock
        self.exposure_offset_s = exposure_offset_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames = 0
        self.frame_id = 0
        self._last_ts = float("-inf")

    def _next(self) -> bool:
        """Grab one frame; return False when the source is exhausted."""
        frame, ok = self.source.read()
        if not ok:
            return False
        ts = self.clock() - self.exposure_offset_s
        if ts <= self._last_ts:  # enforce strict monotonicity (A1.AC2)
            ts = self._last_ts + 1e-9
        self._last_ts = ts
        self.slot.set((frame, ts, self.frame_id))
        self.frames += 1
        self.frame_id += 1
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._next():
                break

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def run_sync(self, max_frames: int | None = None) -> int:
        """Synchronous drain (deterministic, for tests). Returns frames captured."""
        count = 0
        while max_frames is None or count < max_frames:
            if not self._next():
                break
            count += 1
        return count
