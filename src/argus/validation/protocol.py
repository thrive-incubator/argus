"""Validation protocol runner (FR-21, I1).

Scripted blocks with on-screen prompts (injected ``Prompter``) and markers injected into
the recording. HRV is taken from rest blocks only (excluding the 6-brpm block, whose 0.1 Hz
resonance inflates HRV); respiration from the paced blocks (I1.AC2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import SignalRecord


@dataclass(frozen=True)
class ProtocolBlock:
    name: str
    kind: str  # rest | paced_breathing | light_motion | lighting | gaze | eye_closure
    duration_s: float
    params: dict = field(default_factory=dict)


def default_protocol(repeats: int = 2) -> list[ProtocolBlock]:
    """The standard validation protocol (I1.AC1)."""
    blocks: list[ProtocolBlock] = []
    for _ in range(repeats):
        blocks += [
            ProtocolBlock("rest", "rest", 120.0),
            ProtocolBlock("paced_6", "paced_breathing", 60.0, {"brpm": 6}),
            ProtocolBlock("paced_10", "paced_breathing", 60.0, {"brpm": 10}),
            ProtocolBlock("paced_15", "paced_breathing", 60.0, {"brpm": 15}),
            ProtocolBlock("light_motion", "light_motion", 60.0),
            ProtocolBlock("lighting_A", "lighting", 60.0, {"lux": 150}),
            ProtocolBlock("lighting_B", "lighting", 60.0, {"lux": 500}),
            ProtocolBlock("gaze_targets", "gaze", 30.0),
            ProtocolBlock("eye_closure", "eye_closure", 30.0, {"closure_fractions": [0, 25, 50, 75]}),
        ]
    return blocks


class Prompter(Protocol):
    def show(self, text: str) -> None: ...


class FakePrompter:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def show(self, text: str) -> None:
        self.prompts.append(text)


@dataclass
class Marker:
    name: str
    kind: str
    t_start: float
    t_end: float
    params: dict


class ProtocolRunner:
    """Runs blocks, prompts the subject, and injects markers into the recorder."""

    def __init__(self, recorder=None, prompter: Prompter | None = None, clock=None):
        self.recorder = recorder
        self.prompter = prompter or FakePrompter()
        self._t = 0.0
        self.clock = clock  # optional external clock; else virtual time advances by block
        self.markers: list[Marker] = []

    def _now(self) -> float:
        return float(self.clock()) if self.clock else self._t

    def run(self, blocks: list[ProtocolBlock]) -> list[Marker]:
        for i, b in enumerate(blocks):
            t0 = self._now()
            lux = b.params.get("lux")
            prompt = f"[{b.name}] {b.kind}" + (f" (lux={lux})" if lux else "")
            self.prompter.show(prompt)
            if self.recorder is not None:
                # marker onset event (numeric block index) injected into the XDF stream
                self.recorder.record(
                    SignalRecord("marker", float(i), 1.0, t0, gate="unknown",
                                 meta={"block": b.name, "kind": b.kind, "params": b.params})
                )
            self._t = t0 + b.duration_s
            self.markers.append(Marker(b.name, b.kind, t0, self._t, dict(b.params)))
        return self.markers


def hrv_eligible_blocks(markers: list[Marker]) -> list[Marker]:
    """Rest blocks only, excluding the 6-brpm paced block (I1.AC2)."""
    return [m for m in markers if m.kind == "rest"]


def respiration_blocks(markers: list[Marker]) -> list[Marker]:
    """Paced blocks supply the respiration plausibility check (I1.AC2)."""
    return [m for m in markers if m.kind == "paced_breathing"]


def measured_lux(markers: list[Marker]) -> dict[str, float]:
    return {m.name: m.params["lux"] for m in markers if m.kind == "lighting"}
