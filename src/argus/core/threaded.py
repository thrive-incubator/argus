"""Threaded pipeline runner (J1.AC1).

Architecture: a dedicated **capture thread** fills a latest-frame slot; a **worker thread**
runs the per-frame backbone + extractors (the CV hot path — synchronous, no asyncio); emitted
records are handed to an **asyncio I/O edge** running in its own thread that drains them to the
bus. asyncio is used ONLY at the I/O edge, never on the CV hot path.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

from ..capture.buffers import LatestFrameSlot
from ..capture.capture_thread import CaptureThread
from .pipeline import Pipeline

_SENTINEL = object()


class ThreadedPipeline:
    def __init__(self, pipeline: Pipeline, source, fps: float = 30.0):
        self.pipeline = pipeline
        self.source = source
        self.fps = fps
        self.slot = LatestFrameSlot()
        self._emit_q: queue.Queue = queue.Queue()
        self._capture_done = threading.Event()
        self.processed = 0
        self.emitted = 0

    def _worker(self) -> None:
        """CV hot path — synchronous, on its own thread (no asyncio here)."""
        while True:
            item = self.slot.get()
            if item is None:
                if self._capture_done.is_set():
                    break  # capture finished and the slot is drained
                time.sleep(0.001)  # idle wait — no busy-spin
                continue
            frame, ts, fid = item
            for r in self.pipeline.process_frame(frame, ts, fid):
                self._emit_q.put(r)
            self.processed += 1
        self._emit_q.put(_SENTINEL)  # stop the I/O edge

    async def _io_edge(self, bus) -> None:
        """asyncio I/O edge: drains the emit queue to the bus."""
        loop = asyncio.get_event_loop()
        while True:
            rec = await loop.run_in_executor(None, self._emit_q.get)
            if rec is _SENTINEL:
                break
            if bus is not None:
                bus.publish(rec)
            self.emitted += 1

    def run(self, bus=None, timeout_s: float = 10.0) -> None:
        cap = CaptureThread(self.source, self.slot)

        def _capture_loop():
            cap.run_sync()
            self._capture_done.set()

        cap_thread = threading.Thread(target=_capture_loop, daemon=True)
        worker = threading.Thread(target=self._worker, daemon=True)
        io = threading.Thread(target=lambda: asyncio.run(self._io_edge(bus)), daemon=True)

        io.start()
        worker.start()
        cap_thread.start()

        cap_thread.join(timeout=timeout_s)
        worker.join(timeout=timeout_s)
        io.join(timeout=timeout_s)
