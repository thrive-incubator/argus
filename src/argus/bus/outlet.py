"""Stream outlets and an in-memory bus (FR-16, B1).

Each signal/covariate gets its own outlet with a declared rate (regular ``nominal_srate``
or ``IRREGULAR_RATE`` for events) and metadata (unit/method/window). ``LslOutlet`` is the
real adapter (lazy ``pylsl``); ``InMemoryBus`` is the headless substitute a second
consumer can read from (B1.AC3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import SignalRecord
from .format import channel_layout

IRREGULAR_RATE = 0.0


@dataclass
class StreamSpec:
    name: str
    channel_count: int
    nominal_srate: float = IRREGULAR_RATE
    unit: str = ""
    method: str = ""
    window_s: float | None = None
    stream_type: str = "argus"

    @property
    def is_regular(self) -> bool:
        return self.nominal_srate > 0.0


class BusOutlet(Protocol):
    spec: StreamSpec

    def push(self, record: SignalRecord) -> None: ...


class InMemoryOutlet:
    """Records pushed samples ``(ts, channels)``; a consumer can read them back."""

    def __init__(self, spec: StreamSpec):
        self.spec = spec
        self.samples: list[tuple[float, list[float]]] = []

    def push(self, record: SignalRecord) -> None:
        self.samples.append((record.ts, channel_layout(record)))


class LslOutlet:
    """Real LSL outlet (lazy ``pylsl``). The ``push_sample`` call is the device line."""

    def __init__(self, spec: StreamSpec):
        import pylsl  # local import: native liblsl required

        self.spec = spec
        info = pylsl.StreamInfo(
            name=spec.name, type=spec.stream_type, channel_count=spec.channel_count,
            nominal_srate=spec.nominal_srate, channel_format="float32",
            source_id=f"argus-{spec.name}",
        )
        desc = info.desc()
        desc.append_child_value("unit", spec.unit)
        desc.append_child_value("method", spec.method)
        desc.append_child_value("window_s", str(spec.window_s))
        self._outlet = pylsl.StreamOutlet(info)

    def push(self, record: SignalRecord) -> None:  # pragma: no cover - device
        self._outlet.push_sample(channel_layout(record), record.ts)


@dataclass
class InMemoryBus:
    """Creates per-stream outlets on demand and lets consumers read every stream (B1.AC3)."""

    outlets: dict[str, InMemoryOutlet] = field(default_factory=dict)
    specs: dict[str, StreamSpec] = field(default_factory=dict)

    def declare(self, spec: StreamSpec) -> InMemoryOutlet:
        outlet = InMemoryOutlet(spec)
        self.outlets[spec.name] = outlet
        self.specs[spec.name] = spec
        return outlet

    def publish(self, record: SignalRecord) -> None:
        if record.name not in self.outlets:
            # auto-declare an irregular stream for an unseen signal
            n = len(channel_layout(record))
            self.declare(StreamSpec(record.name, channel_count=n))
        self.outlets[record.name].push(record)

    def resolve(self, name: str) -> InMemoryOutlet:
        """A second consumer resolves a stream by name and reads its samples."""
        return self.outlets[name]

    def stream_names(self) -> list[str]:
        return list(self.outlets.keys())
