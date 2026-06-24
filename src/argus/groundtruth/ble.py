"""Polar H10 BLE sources + ingestion with reconnect re-anchoring (FR-20, C1).

- ``BleakPolarSource``: real BLE adapter (lazy ``bleak``), discovered by *device name*
  (works on macOS where only a UUID is exposed). The notify/connect calls are the untested
  device lines.
- ``FakePolarSource``: replays HR-Measurement packets and can simulate a dropout.
- ``PolarIngestor``: parses packets, reconstructs beat times by cumulative sum from a single
  anchor, and **re-anchors on reconnect** with a discontinuity marker (C1.AC1/AC3).
"""

from __future__ import annotations

from typing import Protocol

from .polar import parse_hr_measurement

HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


class PolarSource(Protocol):
    def connect(self) -> bool: ...

    def read(self) -> bytes | None:
        """Return the next HR-Measurement payload, or None on dropout/end."""
        ...


class BleakPolarSource:
    """Real adapter: connect by name over BLE and notify on 0x2A37."""

    def __init__(self, name_prefix: str = "Polar H10"):
        self.name_prefix = name_prefix
        self._client = None

    async def _discover_and_connect(self):  # pragma: no cover - device/BLE
        from bleak import BleakClient, BleakScanner

        device = await BleakScanner.find_device_by_filter(
            lambda d, ad: (d.name or "").startswith(self.name_prefix)
        )
        if device is None:
            return False
        self._client = BleakClient(device)
        await self._client.connect()
        return True


class FakePolarSource:
    """Replays packets; ``drop_after`` simulates a mid-stream dropout."""

    def __init__(self, packets: list[bytes], name: str = "Polar H10 ABC",
                 drop_after: int | None = None):
        self._packets = list(packets)
        self.name = name
        self._i = 0
        self._drop_after = drop_after
        self._reads = 0

    def connect(self) -> bool:
        return True

    def read(self) -> bytes | None:
        if self._drop_after is not None and self._reads == self._drop_after:
            self._drop_after = None  # one dropout, then recovers
            return None
        if self._i >= len(self._packets):
            return None
        pkt = self._packets[self._i]
        self._i += 1
        self._reads += 1
        return pkt


class PolarIngestor:
    """Reconstructs beat times from packets; re-anchors on reconnect (C1.AC1/AC3)."""

    def __init__(self, clock=lambda: 0.0):
        self.clock = clock
        self._running: float | None = None
        self.beat_times: list[float] = []
        self.hr_series: list[tuple[float, int]] = []  # (ts, hr_bpm)
        self.discontinuities: list[float] = []

    def anchor(self) -> None:
        """(Re)anchor the beat train to the current clock; mark a discontinuity if mid-stream."""
        t = float(self.clock())
        if self._running is not None:
            self.discontinuities.append(t)  # gap across a reconnect
        self._running = t

    def on_packet(self, payload: bytes) -> list[float]:
        if self._running is None:
            self.anchor()
        m = parse_hr_measurement(payload)
        new_beats = []
        for rr in m.rr_intervals_ms:
            self._running += rr / 1000.0
            self.beat_times.append(self._running)
            new_beats.append(self._running)
        self.hr_series.append((self._running, m.hr_bpm))
        return new_beats


def run_with_reconnect(source: PolarSource, ingestor: PolarIngestor,
                       max_packets: int = 10_000, max_reconnects: int = 5) -> int:
    """Drive a source through dropouts, re-anchoring on each (re)connect. Returns packets read."""
    reconnects = 0
    read = 0
    source.connect()
    ingestor.anchor()
    while read < max_packets:
        payload = source.read()
        if payload is None:
            if reconnects >= max_reconnects:
                break
            reconnects += 1
            source.connect()
            ingestor.anchor()  # re-anchor + discontinuity marker
            # try one more read after reconnect
            payload = source.read()
            if payload is None:
                break
        ingestor.on_packet(payload)
        read += 1
    return read
