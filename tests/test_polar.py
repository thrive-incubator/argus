"""R5–R6: Polar H10 HR-Measurement parsing + beat-time reconstruction.

Test vectors cover all four relevant flag combinations (FEAT C1.AC2): the latent bug
(B4) is that bit3 energy-expended and bit0 16-bit HR shift the RR offset.
"""

import pytest

from argus.groundtruth.polar import parse_hr_measurement, reconstruct_beat_times


def _rr_bytes(*raw_1024):
    out = bytearray()
    for v in raw_1024:
        out += int(v).to_bytes(2, "little")
    return bytes(out)


# R5 — flag combo 1: 8-bit HR, no energy, RR present.
def test_parse_8bit_hr_with_rr():
    buf = bytes([0x10, 60]) + _rr_bytes(1024, 512)  # 1024/1024 s = 1000 ms, 512 = 500 ms
    m = parse_hr_measurement(buf)
    assert m.hr_bpm == 60
    assert m.rr_intervals_ms == pytest.approx([1000.0, 500.0])


# R5 — flag combo 2: 16-bit HR (bit0), RR present. RR must start AFTER 2 HR bytes.
def test_parse_16bit_hr_shifts_rr_offset():
    buf = bytes([0x11]) + (300).to_bytes(2, "little") + _rr_bytes(1024)
    m = parse_hr_measurement(buf)
    assert m.hr_bpm == 300  # 16-bit value
    assert m.rr_intervals_ms == pytest.approx([1000.0])


# R5 — flag combo 3: 8-bit HR + Energy Expended (bit3) + RR. Energy must be skipped.
def test_parse_energy_expended_shifts_rr_offset():
    buf = bytes([0x18, 75]) + (1234).to_bytes(2, "little") + _rr_bytes(2048)
    m = parse_hr_measurement(buf)
    assert m.hr_bpm == 75
    # If energy were NOT skipped, the parser would read 0x04D2 as the first RR.
    assert m.rr_intervals_ms == pytest.approx([2000.0])  # 2048/1024 s = 2 s


# R5 — flag combo 4: no RR flag -> empty RR list, HR still parsed.
def test_parse_no_rr_present():
    buf = bytes([0x00, 80])
    m = parse_hr_measurement(buf)
    assert m.hr_bpm == 80
    assert m.rr_intervals_ms == []


def test_parse_multiple_rr_per_packet():
    buf = bytes([0x10, 70]) + _rr_bytes(800, 850, 820)
    m = parse_hr_measurement(buf)
    assert len(m.rr_intervals_ms) == 3


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_hr_measurement(b"")


# R6 — beat times = cumulative sum from a single anchor (not packet arrival).
def test_reconstruct_beat_times_cumsum():
    times = reconstruct_beat_times([1000.0, 500.0, 750.0], anchor_s=10.0)
    assert times == pytest.approx([11.0, 11.5, 12.25])
    assert all(b > a for a, b in zip(times, times[1:]))  # strictly increasing


def test_reconstruct_empty():
    assert reconstruct_beat_times([], anchor_s=5.0) == []
