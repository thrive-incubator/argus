"""WebSocket bridge for a browser (p5.js) art consumer (FR-18, B4).

Records are serialised to JSON. The real transport is a websocket connection; tests inject
a fake. No back-pressure if the consumer disconnects (B4.AC3).
"""

from __future__ import annotations

import json
import math
from typing import Protocol

from ..contracts import SignalRecord, gate_code


class WsTransport(Protocol):
    def send(self, text: str) -> None: ...


def _json_clean(v):
    """Recursively convert non-finite floats (NaN/Inf) to None so the output is valid JSON.

    ``json.dumps`` defaults to ``allow_nan=True`` and emits the bare token ``NaN``, which the
    browser's ``JSON.parse`` rejects — dropping the WHOLE message (e.g. respiration's cross-check
    fields are NaN until they're computed, which silently blanked the card). Sanitise instead.
    """
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_clean(x) for x in v]
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


class WebSocketBridge:
    def __init__(self, transport: WsTransport):
        self.transport = transport
        self.sent = 0
        self.dropped = 0

    @staticmethod
    def to_json(record: SignalRecord) -> str:
        return json.dumps(
            _json_clean({
                "name": record.name,
                "value": record.value,
                "sqi": record.sqi,
                "ts": record.ts,
                "gate": record.gate,
                "gate_code": gate_code(record.gate),
                "meta": dict(record.meta),
            }),
            allow_nan=False,
        )

    def publish(self, record: SignalRecord) -> None:
        try:
            self.transport.send(self.to_json(record))
            self.sent += 1
        except Exception:
            self.dropped += 1
