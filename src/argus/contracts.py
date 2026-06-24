"""Core data contracts shared across the pipeline (TECH_DESIGN §3, ADR-19).

- ``SignalRecord`` — the uniform output of every extractor.
- ``FrameContext`` — produced once per frame by the backbone; ``frame`` is read-only.
- ``Extractor`` — ABC; ``consume(ctx) -> list[SignalRecord]``; self-registering.
- gate enum + numeric ``gate_code`` mapping (good=0, usable=1, reject=2, unknown=3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

# Gate states (TECH §3.2). "unknown" is the Phase-0 stub gate (review item M3).
GATE_STATES: tuple[str, ...] = ("good", "usable", "reject", "unknown")

# Numeric channel encoding for LSL (TECH §3.2/§10, review item m1).
GATE_CODE: dict[str, int] = {"good": 0, "usable": 1, "reject": 2, "unknown": 3}


def gate_code(gate: str) -> int:
    """Map a gate string to its numeric channel code."""
    try:
        return GATE_CODE[gate]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown gate state: {gate!r}") from exc


@dataclass(frozen=True)
class SignalRecord:
    """The uniform output of every extractor (TECH §3.2).

    ``timestamp`` is the capture time the value pertains to (window-centre for DSP).
    """

    name: str
    value: Any
    sqi: float
    ts: float
    gate: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gate not in GATE_STATES:
            raise ValueError(
                f"gate must be one of {GATE_STATES}, got {self.gate!r}"
            )

    @property
    def gate_code(self) -> int:
        return gate_code(self.gate)


@dataclass(frozen=True)
class FrameContext:
    """Produced once per frame by the backbone (TECH §3.1).

    ``frame`` is shared across all fan-out consumers and is **read-only**:
    ``frame.flags.writeable`` is forced to ``False`` on construction so an extractor
    cannot mutate it in place and race the others (review item M2).
    """

    frame: np.ndarray
    ts: float
    frame_id: int
    face: Any = None
    pose: Any = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.frame, np.ndarray):
            # Make the shared buffer read-only. frozen=True only stops field
            # rebinding; it does not freeze the underlying ndarray.
            self.frame.flags.writeable = False


class Extractor(ABC):
    """Base class for all signal extractors (TECH §3.3, ADR-19).

    Subclasses set a class-level ``name`` and implement ``consume`` returning a
    ``list[SignalRecord]`` (``[]`` when no new window is ready — review item M1).
    Subclasses are auto-registered by ``name`` in :data:`REGISTRY`.
    """

    name: ClassVar[str] = ""
    REGISTRY: ClassVar[dict[str, type["Extractor"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            Extractor.REGISTRY[cls.name] = cls

    @abstractmethod
    def consume(self, ctx: FrameContext) -> list[SignalRecord]:
        """Process one frame; return zero or more records (never None)."""
        raise NotImplementedError
