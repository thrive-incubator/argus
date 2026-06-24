"""FR-16/B1 outlets, FR-18/B4 OSC+WS bridges, FR-17/B2 XDF recording."""

import numpy as np
import pyxdf
import pytest

from argus.bus.osc import OscBridge, osc_decode_message, osc_encode_message
from argus.bus.outlet import IRREGULAR_RATE, InMemoryBus, StreamSpec
from argus.bus.recorder import Recorder
from argus.bus.ws import WebSocketBridge
from argus.contracts import SignalRecord


# B1.AC1/AC2/AC3 — outlets, metadata, second consumer reads.
def test_inmemory_bus_outlets_and_second_consumer():
    bus = InMemoryBus()
    bus.declare(StreamSpec("hr", channel_count=3, nominal_srate=1.0, unit="bpm", method="pos"))
    bus.declare(StreamSpec("blink_event", channel_count=3, nominal_srate=IRREGULAR_RATE))
    bus.publish(SignalRecord("hr", 72.0, 0.9, 1.0, gate="good"))
    bus.publish(SignalRecord("hr", 73.0, 0.8, 2.0, gate="usable"))
    assert bus.specs["hr"].is_regular and not bus.specs["blink_event"].is_regular
    # a second consumer resolves and reads
    samples = bus.resolve("hr").samples
    assert samples[0][1] == [72.0, 0.9, 0.0]  # value, sqi, gate_code(good)=0
    assert samples[1][1] == [73.0, 0.8, 1.0]  # gate_code(usable)=1
    assert set(bus.stream_names()) == {"hr", "blink_event"}


def test_bus_autodeclares_unknown_signal():
    bus = InMemoryBus()
    bus.publish(SignalRecord("resp", 15.0, 0.7, 1.0))
    assert "resp" in bus.stream_names()


# B4.AC1/AC2 — OSC encode/decode round-trip + bridge sends with forward-sync arg.
def test_osc_encode_decode_roundtrip():
    data = osc_encode_message("/argus/hr", [72.0, 0.9, 0])
    addr, args = osc_decode_message(data)
    assert addr == "/argus/hr"
    assert args[0] == pytest.approx(72.0) and args[2] == 0


def test_osc_bridge_sends_value_and_lookahead():
    received = []

    class FakeT:
        def send(self, data):
            received.append(osc_decode_message(data))

    bridge = OscBridge(FakeT(), forward_sync_ms=50.0)
    bridge.publish(SignalRecord("hr", 72.0, 0.9, 1.0, gate="good"))
    assert bridge.sent == 1
    addr, args = received[0]
    assert addr == "/argus/hr"
    assert args[0] == pytest.approx(72.0)
    assert args[-1] == pytest.approx(1.05)  # ts + 50 ms look-ahead


# B4.AC3 — bridge does not back-pressure when the consumer disconnects.
def test_osc_bridge_no_backpressure_on_disconnect():
    class DeadT:
        def send(self, data):
            raise ConnectionError("consumer gone")

    bridge = OscBridge(DeadT())
    bridge.publish(SignalRecord("hr", 72.0, 0.9, 1.0))  # must not raise
    assert bridge.dropped == 1 and bridge.sent == 0


def test_ws_bridge_json():
    sent = []
    bridge = WebSocketBridge(type("T", (), {"send": lambda self, t: sent.append(t)})())
    bridge.publish(SignalRecord("affect_valence", 0.5, 0.9, 2.0, gate="good"))
    import json

    obj = json.loads(sent[0])
    assert obj["name"] == "affect_valence" and obj["gate_code"] == 0 and obj["value"] == 0.5


# FR-17 / B2 — record streams + Polar to one XDF; pyxdf loads it; sample counts match.
def test_xdf_recording_roundtrip(tmp_path):
    rec = Recorder()
    rec.declare(StreamSpec("hr", channel_count=3, nominal_srate=1.0, unit="bpm"))
    rec.declare(StreamSpec("polar_hr", channel_count=3, nominal_srate=1.0, unit="bpm"))
    for i in range(5):
        rec.record(SignalRecord("hr", 70.0 + i, 0.9, float(i), gate="good"))
    for i in range(7):
        rec.record(SignalRecord("polar_hr", 71.0 + i, 1.0, float(i) * 0.9, gate="good"))
    rec.add_clock_offset("hr", collection_time=0.5, offset=0.001)  # B2.AC2 metadata

    path = str(tmp_path / "session.xdf")
    rec.write(path)

    streams, header = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=True)
    by_name = {s["info"]["name"][0]: s for s in streams}
    assert set(by_name) == {"hr", "polar_hr"}
    # B2.AC3 — per-signal sample counts reproduced
    assert by_name["hr"]["time_series"].shape[0] == 5
    assert by_name["polar_hr"]["time_series"].shape[0] == 7
    # channel values survive the round-trip
    assert by_name["hr"]["time_series"][0][0] == pytest.approx(70.0)
    # B2.AC2 — clock-offset metadata present for the hr stream
    assert len(by_name["hr"]["clock_times"]) >= 1
