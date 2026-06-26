"""WebSocket record serialisation must always produce valid JSON (NaN/Inf → null).

Regression: respiration's cross-check meta fields (flow_rr/rppg_rr/agreement) are NaN until
computed; the old serialiser emitted the bare ``NaN`` token, which the browser's JSON.parse
rejects — silently dropping the whole message and blanking the Respiration card.
"""

import json

from argus.bus.ws import WebSocketBridge
from argus.contracts import SignalRecord


def test_nan_and_inf_in_meta_serialize_to_null():
    r = SignalRecord("resp", 14.9, 1.0, 1.0, gate="unknown",
                     meta={"flow_rr": float("nan"), "rppg_rr": float("inf"),
                           "agreement": float("-inf"), "motion_rr": 15.2, "primary": "chest_motion"})
    s = WebSocketBridge.to_json(r)
    assert "NaN" not in s and "Infinity" not in s
    d = json.loads(s)  # must not raise (this is what the browser does)
    assert d["meta"]["flow_rr"] is None
    assert d["meta"]["rppg_rr"] is None and d["meta"]["agreement"] is None
    assert d["meta"]["motion_rr"] == 15.2 and d["meta"]["primary"] == "chest_motion"
    assert d["name"] == "resp" and d["value"] == 14.9


def test_nan_value_serializes_to_null():
    r = SignalRecord("hr", float("nan"), float("nan"), 1.0, gate="unknown", meta={})
    d = json.loads(WebSocketBridge.to_json(r))
    assert d["value"] is None and d["sqi"] is None


def test_nested_meta_is_sanitized():
    r = SignalRecord("x", 1.0, 1.0, 1.0, gate="unknown",
                     meta={"zone": {"a": float("nan"), "b": 2.0}, "list": [1.0, float("inf")]})
    d = json.loads(WebSocketBridge.to_json(r))
    assert d["meta"]["zone"]["a"] is None and d["meta"]["zone"]["b"] == 2.0
    assert d["meta"]["list"] == [1.0, None]
