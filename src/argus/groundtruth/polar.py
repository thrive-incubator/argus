"""Polar H10 Heart-Rate-Measurement (0x2A37) parsing + beat-time reconstruction.

This is the load-bearing correctness item from the review (B4): the prior shorthand
("flags bit0/bit4") silently mis-parsed RR for some flag combinations, which would
invalidate all HRV. This implements the **full flag-driven offset algorithm** (ADR-17):

    offset = 1                       # skip flags byte
    bit0 -> HR is uint16 (else uint8)
    bit3 -> Energy Expended present (skip 2 bytes)
    bit4 -> RR-Interval(s) present, uint16 LE pairs to end of packet, units 1/1024 s
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Flag bit masks (Bluetooth Heart Rate Measurement characteristic).
_HR_FORMAT_UINT16 = 0x01  # bit 0
_ENERGY_EXPENDED = 0x08  # bit 3
_RR_PRESENT = 0x10  # bit 4

_RR_UNIT_SECONDS = 1.0 / 1024.0  # RR-interval LSB is 1/1024 s


@dataclass(frozen=True)
class HrMeasurement:
    hr_bpm: int
    rr_intervals_ms: list[float] = field(default_factory=list)


def parse_hr_measurement(buf: bytes) -> HrMeasurement:
    """Parse one HR-Measurement notification payload.

    Args:
        buf: raw characteristic bytes (flags byte first).

    Returns:
        ``HrMeasurement`` with the heart rate and *all* RR intervals in ms.

    Raises:
        ValueError: if the buffer is empty or truncated mid-field.
    """
    if not buf:
        raise ValueError("empty HR-measurement buffer")

    flags = buf[0]
    offset = 1

    # Heart rate: uint16 if bit0 set, else uint8.
    if flags & _HR_FORMAT_UINT16:
        if offset + 2 > len(buf):
            raise ValueError("truncated 16-bit HR field")
        hr = int.from_bytes(buf[offset : offset + 2], "little")
        offset += 2
    else:
        if offset + 1 > len(buf):
            raise ValueError("truncated 8-bit HR field")
        hr = buf[offset]
        offset += 1

    # Energy Expended (bit3): 2 bytes we skip.
    if flags & _ENERGY_EXPENDED:
        if offset + 2 > len(buf):
            raise ValueError("truncated energy-expended field")
        offset += 2

    # RR intervals (bit4): zero or more uint16 LE, units 1/1024 s. Iterate to end.
    rr_ms: list[float] = []
    if flags & _RR_PRESENT:
        while offset + 2 <= len(buf):
            rr_raw = int.from_bytes(buf[offset : offset + 2], "little")
            rr_ms.append(rr_raw * _RR_UNIT_SECONDS * 1000.0)
            offset += 2

    return HrMeasurement(hr_bpm=hr, rr_intervals_ms=rr_ms)


def reconstruct_beat_times(rr_intervals_ms, anchor_s: float = 0.0) -> list[float]:
    """Reconstruct absolute beat times by cumulative-summing RR from one anchor.

    Beat times are placed at the END of each RR interval, measured forward from a
    single anchored clock value — NOT from BLE packet-arrival time (ADR-16 #3).

    Args:
        rr_intervals_ms: ordered RR intervals in milliseconds.
        anchor_s: the single LSL ``local_clock`` anchor (seconds) the train hangs off.

    Returns:
        List of beat timestamps (seconds), one per RR interval, strictly increasing.
    """
    times: list[float] = []
    t = anchor_s
    for rr in rr_intervals_ms:
        t += rr / 1000.0
        times.append(t)
    return times
