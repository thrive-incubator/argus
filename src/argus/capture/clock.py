"""Monotonic timebase (ADR-16). Uses ``pylsl.local_clock`` when available, else
``time.monotonic`` — both are steady/monotonic clocks suitable as the single timebase.
"""

from __future__ import annotations

import time
from typing import Callable

try:  # pragma: no cover - exercised only when pylsl is installed
    from pylsl import local_clock as _lsl_local_clock
except Exception:  # pragma: no cover
    _lsl_local_clock = None


def local_clock() -> float:
    """Return monotonic seconds from the project timebase."""
    if _lsl_local_clock is not None:  # pragma: no cover - device line
        return float(_lsl_local_clock())
    return time.monotonic()


# A ``Clock`` is any zero-arg callable returning monotonic seconds (injectable for tests).
Clock = Callable[[], float]


class FakeClock:
    """Deterministic injectable clock that advances by a fixed step per call."""

    def __init__(self, start: float = 0.0, step: float = 1.0 / 30.0) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        t = self._t
        self._t += self._step
        return t
