"""OSC bridge for the TouchDesigner art leg (FR-18, B4).

OSC 1.0 message encoding/decoding is implemented here (no third-party dependency) so the
bridge is fully testable headlessly. The real transport is a UDP socket; tests inject a
fake transport. The bridge never back-pressures the pipeline (B4.AC3).
"""

from __future__ import annotations

import struct
from typing import Protocol

from ..contracts import SignalRecord
from .format import channel_layout, osc_address


def _pad(b: bytes) -> bytes:
    return b + b"\x00" * ((4 - len(b) % 4) % 4)


def osc_encode_message(address: str, args: list) -> bytes:
    out = _pad(address.encode() + b"\x00")
    tags = ","
    for a in args:
        tags += "i" if isinstance(a, bool) or isinstance(a, int) else "f"
    out += _pad(tags.encode() + b"\x00")
    for a in args:
        if isinstance(a, bool) or isinstance(a, int):
            out += struct.pack(">i", int(a))
        else:
            out += struct.pack(">f", float(a))
    return out


def osc_decode_message(data: bytes):
    """Decode a simple OSC message → ``(address, [args])`` (floats/ints only)."""
    end = data.index(b"\x00")
    address = data[:end].decode()
    i = (len(data[:end]) // 4 + 1) * 4
    tend = data.index(b"\x00", i)
    tags = data[i:tend].decode().lstrip(",")
    i = ((tend) // 4 + 1) * 4
    args = []
    for t in tags:
        if t == "i":
            args.append(struct.unpack(">i", data[i : i + 4])[0])
        else:
            args.append(struct.unpack(">f", data[i : i + 4])[0])
        i += 4
    return address, args


class Transport(Protocol):
    def send(self, data: bytes) -> None: ...


class UdpTransport:
    """Real UDP transport (the socket sendto is the device/network line)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7000):
        import socket

        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data: bytes) -> None:  # pragma: no cover - network
        self._sock.sendto(data, self._addr)


class OscBridge:
    """Re-emit selected records as OSC messages with a forward-sync look-ahead."""

    def __init__(self, transport: Transport, namespace: str = "/argus",
                 forward_sync_ms: float = 50.0):
        self.transport = transport
        self.namespace = namespace
        self.forward_sync_ms = forward_sync_ms
        self.sent = 0
        self.dropped = 0

    def publish(self, record: SignalRecord) -> None:
        addr = osc_address(record.name, self.namespace)
        # forward-sync: include the look-ahead-adjusted timestamp as a trailing arg
        args = list(channel_layout(record)) + [record.ts + self.forward_sync_ms / 1000.0]
        try:
            self.transport.send(osc_encode_message(addr, args))
            self.sent += 1
        except Exception:  # B4.AC3 — consumer gone: drop, never block the pipeline
            self.dropped += 1
