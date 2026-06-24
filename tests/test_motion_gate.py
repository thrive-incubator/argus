"""R17–R18: motion quality gate (3-tier + hysteresis) and normalization."""

import pytest

from argus.quality.motion_gate import GOOD, REJECT, USABLE, MotionGate, normalize_motion


# R18
def test_normalize_motion_by_inter_ocular_distance():
    assert normalize_motion(30.0, 100.0) == pytest.approx(0.30)


def test_normalize_motion_rejects_zero_iod():
    with pytest.raises(ValueError):
        normalize_motion(1.0, 0.0)


# R17 — steady-state states after dwell.
def _run(gate, motion, snr_db, frames):
    state = None
    for _ in range(frames):
        state = gate.update(motion, snr_db)
    return state


def test_gate_reaches_good_with_low_motion_high_snr():
    g = MotionGate(dwell_frames=5)
    assert _run(g, motion=0.05, snr_db=6.0, frames=10) == GOOD


def test_gate_reaches_reject_with_high_motion():
    g = MotionGate(dwell_frames=5)
    assert _run(g, motion=0.9, snr_db=6.0, frames=10) == REJECT


def test_gate_usable_in_between():
    g = MotionGate(dwell_frames=5)
    # mild motion, ok-ish snr -> usable
    assert _run(g, motion=0.25, snr_db=2.0, frames=10) == USABLE


# R17 — hysteresis: a single borderline excursion shorter than dwell must not flip state.
def test_hysteresis_suppresses_flicker():
    g = MotionGate(dwell_frames=10)
    _run(g, motion=0.9, snr_db=6.0, frames=15)  # settle into REJECT
    assert g.state == REJECT
    # brief GOOD-looking inputs (fewer than dwell) should NOT flip to GOOD
    flipped = False
    for _ in range(5):  # 5 < dwell(10)
        if g.update(0.05, 8.0) == GOOD:
            flipped = True
    assert not flipped
    assert g.state == REJECT
    # sustained good input beyond dwell DOES flip it
    assert _run(g, motion=0.05, snr_db=8.0, frames=12) == GOOD
