"""Synchronized recording of all streams + Polar into one XDF (FR-17, B2).

A ``Recorder`` accumulates per-stream samples (with capture timestamps and clock-offset
metadata) and writes them to a single XDF via the writer. ``argus record`` (J3.AC2) drives
it; ``pyxdf.load_xdf`` reads it back (B2.AC2/AC3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import SignalRecord
from .format import channel_layout
from .outlet import StreamSpec
from .xdf_writer import XdfStream, write_xdf


@dataclass
class _StreamBuf:
    spec: StreamSpec
    stream_id: int
    timestamps: list[float] = field(default_factory=list)
    samples: list[list[float]] = field(default_factory=list)
    clock_offsets: list[tuple[float, float]] = field(default_factory=list)


class Recorder:
    def __init__(self) -> None:
        self._streams: dict[str, _StreamBuf] = {}
        self._next_id = 1

    def declare(self, spec: StreamSpec) -> None:
        if spec.name not in self._streams:
            self._streams[spec.name] = _StreamBuf(spec, self._next_id)
            self._next_id += 1

    def record(self, record: SignalRecord) -> None:
        if record.name not in self._streams:
            n = len(channel_layout(record))
            self.declare(StreamSpec(record.name, channel_count=n))
        buf = self._streams[record.name]
        buf.timestamps.append(record.ts)
        buf.samples.append(channel_layout(record))

    def add_clock_offset(self, stream_name: str, collection_time: float, offset: float) -> None:
        self._streams[stream_name].clock_offsets.append((collection_time, offset))

    def sample_count(self, stream_name: str) -> int:
        return len(self._streams[stream_name].samples)

    def stream_names(self) -> list[str]:
        return list(self._streams.keys())

    def write(self, path: str) -> None:
        streams = [
            XdfStream(
                stream_id=b.stream_id,
                name=b.spec.name,
                timestamps=b.timestamps,
                samples=b.samples,
                channel_count=b.spec.channel_count,
                nominal_srate=b.spec.nominal_srate,
                stream_type=b.spec.stream_type,
                unit=b.spec.unit,
                clock_offsets=b.clock_offsets,
            )
            for b in self._streams.values()
        ]
        write_xdf(path, streams)
