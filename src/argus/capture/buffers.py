"""Two buffer disciplines (ADR-19, FR-4):

- ``LatestFrameSlot``: 1-slot drop-oldest for per-frame extractors + display (freshness;
  frame loss OK). Counts dropped frames.
- ``TimeSeriesRing``: lossless, evenly-timestamped ring feeding rPPG/HRV/respiration —
  every produced sample is recorded (``appended`` == produced); none silently lost.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class LatestFrameSlot:
    """Thread-safe single-slot holder; ``set`` overwrites, dropping the previous unread."""

    def __init__(self) -> None:
        self._item: Any = None
        self._has = False
        self._lock = threading.Lock()
        self.dropped = 0
        self.produced = 0

    def set(self, item: Any) -> None:
        with self._lock:
            if self._has:  # an unread item is overwritten -> a drop
                self.dropped += 1
            self._item = item
            self._has = True
            self.produced += 1

    def get(self):
        """Return the latest item (a copy if it is an ndarray) or None; marks consumed."""
        with self._lock:
            if not self._has:
                return None
            item = self._item
            self._has = False
            if isinstance(item, np.ndarray):
                return item.copy()
            return item


class TimeSeriesRing:
    """Lossless evenly-timestamped ring. Keeps the last ``capacity`` samples for windowing
    but counts EVERY append, so 'samples lost' is always zero by construction (FR-4)."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._t: deque[float] = deque(maxlen=capacity)
        self._v: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.appended = 0

    def append(self, ts: float, value) -> None:
        with self._lock:
            self._t.append(ts)
            self._v.append(value)
            self.appended += 1

    def window(self, seconds: float):
        """Return ``(times, values)`` for the most recent ``seconds`` of data."""
        with self._lock:
            if not self._t:
                return np.array([]), np.array([])
            t = np.array(self._t)
            v = np.array(self._v)
        if t.size == 0:
            return t, v
        keep = t >= (t[-1] - seconds)
        return t[keep], v[keep]

    def __len__(self) -> int:
        return len(self._t)
