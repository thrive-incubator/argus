"""Pure bus-formatting helpers (TECH §3.2/§10).

- ``channel_layout`` builds the numeric LSL sample ``[value..., sqi, gate_code]``.
- ``osc_address`` maps a signal name to its TouchDesigner OSC address (rev 2 default).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import SignalRecord, gate_code

OSC_NAMESPACE = "/argus"


def channel_layout(record: SignalRecord) -> list[float]:
    """Flatten a record into the numeric channel vector ``[value..., sqi, gate_code]``.

    Scalar values become one channel; sequence values are expanded in order.
    """
    value = record.value
    if isinstance(value, (int, float)):
        values = [float(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(v) for v in value]
    else:
        raise TypeError(f"non-numeric value not supported on the numeric bus: {value!r}")
    return [*values, float(record.sqi), float(gate_code(record.gate))]


def osc_address(signal_name: str, namespace: str = OSC_NAMESPACE) -> str:
    """Map a signal name to its OSC address, e.g. ``hr`` -> ``/argus/hr``."""
    if not signal_name:
        raise ValueError("signal_name must be non-empty")
    return f"{namespace}/{signal_name.strip('/')}"
