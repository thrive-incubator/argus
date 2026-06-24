"""R1–R4: data contracts (SignalRecord, FrameContext, gate codes, Extractor)."""

import numpy as np
import pytest

from argus.contracts import (
    GATE_CODE,
    Extractor,
    FrameContext,
    SignalRecord,
    gate_code,
)


# R1
def test_signalrecord_fields_and_default_gate():
    r = SignalRecord(name="hr", value=72.0, sqi=0.9, ts=1.0)
    assert r.name == "hr" and r.value == 72.0 and r.sqi == 0.9 and r.ts == 1.0
    assert r.gate == "unknown"  # Phase-0 stub default
    assert r.meta == {}


def test_signalrecord_rejects_bad_gate():
    with pytest.raises(ValueError):
        SignalRecord(name="hr", value=1.0, sqi=1.0, ts=0.0, gate="great")


# R2
def test_gate_code_mapping_exact():
    assert GATE_CODE == {"good": 0, "usable": 1, "reject": 2, "unknown": 3}
    assert gate_code("good") == 0
    assert gate_code("reject") == 2
    assert SignalRecord("x", 1.0, 1.0, 0.0, gate="usable").gate_code == 1


# R3
def test_framecontext_frame_is_read_only():
    arr = np.ones((4, 4, 3), dtype=np.uint8)
    ctx = FrameContext(frame=arr, ts=0.0, frame_id=0)
    assert ctx.frame.flags.writeable is False
    with pytest.raises(ValueError):
        ctx.frame[0, 0, 0] = 5  # mutating a read-only ndarray raises


def test_framecontext_is_frozen():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    ctx = FrameContext(frame=arr, ts=0.0, frame_id=1)
    with pytest.raises(Exception):
        ctx.ts = 9.0  # frozen dataclass


# R4
def test_extractor_registry_and_list_return():
    class Dummy(Extractor):
        name = "dummy_test"

        def consume(self, ctx):
            return [SignalRecord("dummy_test", 1.0, 1.0, ctx.ts, gate="good")]

    assert Extractor.REGISTRY.get("dummy_test") is Dummy
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    out = Dummy().consume(FrameContext(arr, 3.0, 0))
    assert isinstance(out, list) and len(out) == 1
    assert out[0].ts == 3.0


def test_extractor_is_abstract():
    with pytest.raises(TypeError):
        Extractor()  # cannot instantiate the ABC
