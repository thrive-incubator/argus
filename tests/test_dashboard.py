"""FR-19, B3.AC1-3, NFR-9 — dashboard render model."""

from argus.contracts import SignalRecord
from argus.dashboard.render import Dashboard, traffic_light


def test_traffic_light_mapping():
    assert traffic_light("good") == "🟢"
    assert traffic_light("usable") == "🟡"
    assert traffic_light("reject") == "🔴"
    assert traffic_light("unknown") == "⚪"


# B3.AC1 — Phase 0 shows HR with SQI + gate.
def test_phase0_shows_hr_only():
    d = Dashboard(phase=0)
    d.update(SignalRecord("hr", 72.0, 0.9, 1.0, gate="good"))
    d.update(SignalRecord("affect_valence", 0.5, 0.8, 1.0, gate="good"))  # ignored in P0
    view = d.render(now=1.0)
    assert set(view) == {"hr"}
    assert view["hr"].light == "🟢" and view["hr"].sqi == 0.9 and view["hr"].status == "ok"


# B3.AC2 — Phase 1 renders every signal with value+SQI+traffic-light.
def test_phase1_renders_all_signals():
    d = Dashboard(phase=1)
    d.update(SignalRecord("hr", 72.0, 0.9, 1.0, gate="good"))
    d.update(SignalRecord("resp", 15.0, 0.7, 1.0, gate="usable"))
    view = d.render(now=1.0)
    assert set(view) == {"hr", "resp"}
    assert view["resp"].light == "🟡"
    text = d.to_text(now=1.0)
    assert "hr" in text and "resp" in text


# B3.AC3 / NFR-9 — REJECT shows "re-acquiring", never a silently frozen value.
def test_reject_shows_reacquiring():
    d = Dashboard(phase=1)
    d.update(SignalRecord("hr", 72.0, 0.2, 5.0, gate="reject"))
    view = d.render(now=5.0)
    assert view["hr"].status == "re-acquiring"
    assert "[re-acquiring]" in d.to_text(now=5.0)


def test_stale_signal_marked():
    d = Dashboard(phase=1, stale_after_s=2.0)
    d.update(SignalRecord("hr", 72.0, 0.9, 1.0, gate="good"))
    assert d.render(now=10.0)["hr"].status == "stale"  # not updated -> not silently frozen


def test_low_sqi_marked_degraded():
    d = Dashboard(phase=1, low_sqi=0.5)
    d.update(SignalRecord("hr", 72.0, 0.2, 1.0, gate="good"))
    assert d.render(now=1.0)["hr"].status == "degraded"
