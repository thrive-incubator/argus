"""WebSocket bridge for a browser (p5.js) art consumer (FR-18, B4).

Records are serialised to JSON. The real transport is a websocket connection; tests inject
a fake. No back-pressure if the consumer disconnects (B4.AC3).
"""

from __future__ import annotations

import json
from typing import Protocol

from ..contracts import SignalRecord, gate_code


class WsTransport(Protocol):
    def send(self, text: str) -> None: ...


class WebSocketBridge:
    def __init__(self, transport: WsTransport):
        self.transport = transport
        self.sent = 0
        self.dropped = 0

    @staticmethod
    def to_json(record: SignalRecord) -> str:
        def _safe(v):
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                return str(v)

        return json.dumps(
            {
                "name": record.name,
                "value": record.value,
                "sqi": record.sqi,
                "ts": record.ts,
                "gate": record.gate,
                "gate_code": gate_code(record.gate),
                "meta": {k: _safe(v) for k, v in record.meta.items()},
            }
        )

    def publish(self, record: SignalRecord) -> None:
        try:
            self.transport.send(self.to_json(record))
            self.sent += 1
        except Exception:
            self.dropped += 1
