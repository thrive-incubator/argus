"""Multiprocessing escape hatch passing frame *handles* via shared memory (J1.AC2).

Heavy CPU-bound extractors (e.g. a PyTorch AU model) run in a subprocess fed a shared-memory
handle to the frame — never a pickled array copy. This module implements the real shared-memory
transfer (testable headlessly).
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


@dataclass(frozen=True)
class FrameHandle:
    """A picklable reference to a frame living in shared memory (NOT the pixels)."""

    shm_name: str
    shape: tuple
    dtype: str


def put_frame(frame: np.ndarray) -> tuple[FrameHandle, shared_memory.SharedMemory]:
    """Copy a frame into shared memory; return its handle and the SHM object (keep it alive)."""
    shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
    buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    buf[:] = frame[:]
    return FrameHandle(shm.name, tuple(frame.shape), str(frame.dtype)), shm


def get_frame(handle: FrameHandle) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    """Attach to the shared frame by handle; returns a view + the SHM (close when done)."""
    shm = shared_memory.SharedMemory(name=handle.shm_name)
    arr = np.ndarray(handle.shape, dtype=np.dtype(handle.dtype), buffer=shm.buf)
    return arr, shm
