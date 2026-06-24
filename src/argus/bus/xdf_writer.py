"""Minimal XDF 1.0 writer (FR-17, B2) that ``pyxdf.load_xdf`` can read.

Implements the chunked XDF container: FileHeader, StreamHeader, Samples, ClockOffset,
StreamFooter. Channel format is float32. This lets us record all Argus streams + the
Polar stream into one synchronized file with clock-offset metadata (B2.AC2/AC3).

Spec: https://github.com/sccn/xdf/wiki/Specifications
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"XDF:"
_TAG_FILEHEADER = 1
_TAG_STREAMHEADER = 2
_TAG_SAMPLES = 3
_TAG_CLOCKOFFSET = 4
_TAG_STREAMFOOTER = 6


def _varlen(n: int) -> bytes:
    if n < 256:
        return b"\x01" + struct.pack("<B", n)
    if n < 2**32:
        return b"\x04" + struct.pack("<I", n)
    return b"\x08" + struct.pack("<Q", n)


def _chunk(tag: int, content: bytes) -> bytes:
    body = struct.pack("<H", tag) + content
    return _varlen(len(body)) + body


@dataclass
class XdfStream:
    stream_id: int
    name: str
    timestamps: list[float]
    samples: list[list[float]]  # N x C, float32
    channel_count: int
    nominal_srate: float = 0.0
    stream_type: str = "argus"
    unit: str = ""
    clock_offsets: list[tuple[float, float]] = field(default_factory=list)


def _stream_header_xml(s: XdfStream) -> bytes:
    xml = (
        f"<?xml version=\"1.0\"?><info><name>{s.name}</name>"
        f"<type>{s.stream_type}</type><channel_count>{s.channel_count}</channel_count>"
        f"<nominal_srate>{s.nominal_srate}</nominal_srate>"
        f"<channel_format>float32</channel_format>"
        f"<desc><unit>{s.unit}</unit></desc></info>"
    )
    return xml.encode("utf-8")


def _stream_footer_xml(s: XdfStream) -> bytes:
    first = s.timestamps[0] if s.timestamps else 0.0
    last = s.timestamps[-1] if s.timestamps else 0.0
    xml = (
        f"<?xml version=\"1.0\"?><info><first_timestamp>{first}</first_timestamp>"
        f"<last_timestamp>{last}</last_timestamp>"
        f"<sample_count>{len(s.samples)}</sample_count></info>"
    )
    return xml.encode("utf-8")


def write_xdf(path: str, streams: list[XdfStream]) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(_chunk(_TAG_FILEHEADER, b"<?xml version=\"1.0\"?><info><version>1.0</version></info>"))
        for s in streams:
            # StreamHeader
            f.write(_chunk(_TAG_STREAMHEADER, struct.pack("<I", s.stream_id) + _stream_header_xml(s)))
            # Samples chunk
            content = struct.pack("<I", s.stream_id) + _varlen(len(s.samples))
            for ts, row in zip(s.timestamps, s.samples):
                content += b"\x08" + struct.pack("<d", float(ts))
                content += struct.pack("<%df" % s.channel_count, *[float(v) for v in row])
            f.write(_chunk(_TAG_SAMPLES, content))
            # ClockOffset chunks (clock-offset metadata, B2.AC2)
            for collection_time, offset in s.clock_offsets:
                f.write(
                    _chunk(
                        _TAG_CLOCKOFFSET,
                        struct.pack("<I", s.stream_id) + struct.pack("<dd", collection_time, offset),
                    )
                )
            # StreamFooter
            f.write(_chunk(_TAG_STREAMFOOTER, struct.pack("<I", s.stream_id) + _stream_footer_xml(s)))
