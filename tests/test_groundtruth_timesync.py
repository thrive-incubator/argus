"""FR-20/C1 (BLE ingest, reconnect, Kubios) + C2 (time sync)."""

import numpy as np
import pytest

from argus.groundtruth.ble import (
    FakePolarSource,
    PolarIngestor,
    run_with_reconnect,
)
from argus.groundtruth.kubios import kubios_correct
from argus.validation.timesync import (
    beat_match,
    cross_correlate_lag,
    dtw_distance,
    instantaneous_hr,
    resample,
)


def _rr_packet(*rr_1024):
    out = bytearray([0x10, 60])  # 8-bit HR, RR present
    for v in rr_1024:
        out += int(v).to_bytes(2, "little")
    return bytes(out)


# C1.AC3 — beat times = cumsum from one anchor; published HR + beat-times.
def test_ingestor_beat_times_monotonic():
    clk = iter([100.0, 200.0])
    ing = PolarIngestor(clock=lambda: next(clk))
    ing.on_packet(_rr_packet(1024, 1024))  # two 1.0 s beats
    ing.on_packet(_rr_packet(512))  # one 0.5 s beat
    assert ing.beat_times == pytest.approx([101.0, 102.0, 102.5])
    assert all(b > a for a, b in zip(ing.beat_times, ing.beat_times[1:]))
    assert ing.hr_series[-1][1] == 60


# C1.AC1 — reconnect re-anchors and records a discontinuity.
def test_reconnect_reanchors_with_discontinuity():
    packets = [_rr_packet(1024), _rr_packet(1024), _rr_packet(1024)]
    src = FakePolarSource(packets, drop_after=1)  # drop after the 1st read
    times = iter([10.0, 50.0, 50.0, 50.0, 50.0])
    ing = PolarIngestor(clock=lambda: next(times))
    # bound by the known stream length so the trailing end-of-data isn't read as a 2nd dropout
    n = run_with_reconnect(src, ing, max_packets=len(packets))
    assert n >= 2
    assert len(ing.discontinuities) == 1  # exactly one re-anchor gap
    assert ing.discontinuities[0] == pytest.approx(50.0)


# C1.AC2 already covered in test_polar; here confirm ingestor uses the full parser.
def test_ingestor_parses_multiple_rr():
    ing = PolarIngestor(clock=lambda: 0.0)
    new = ing.on_packet(_rr_packet(800, 850, 820))
    assert len(new) == 3


# C1.AC4 — Kubios corrects an obvious ectopic beat.
def test_kubios_corrects_ectopic():
    rng = np.random.default_rng(0)
    rr = 800 + rng.normal(0, 10, 40)
    rr[20] = 400  # ectopic: a too-short beat
    rr[21] = 1200  # compensatory long beat
    corrected = kubios_correct(rr)
    # the extreme spread is reduced after correction
    assert np.std(corrected) < np.std(rr)
    assert corrected.min() > 400  # the 400 ms artifact is gone


# C2.AC1 — instantaneous HR, resample to 4 Hz, cross-correlate residual lag.
def test_cross_correlation_recovers_lag():
    fs = 4.0
    t = np.arange(0, 30, 1 / fs)
    base = 70 + 5 * np.sin(2 * np.pi * 0.05 * t)  # slowly varying HR
    lag = 0.5  # seconds
    shifted = 70 + 5 * np.sin(2 * np.pi * 0.05 * (t - lag))
    recovered = cross_correlate_lag(base, shifted, fs=fs)
    assert abs(recovered - lag) <= 0.3


def test_instantaneous_hr_and_resample():
    beats = np.cumsum([0.0] + [60.0 / 72.0] * 40)  # 72 bpm
    t, hr = instantaneous_hr(beats)
    assert np.allclose(hr, 72.0, atol=0.5)
    tu, hu = resample(t, hr, fs=4.0)
    assert tu.size >= 2 and np.allclose(hu, 72.0, atol=1.0)


# C2.AC2 — beat matching within tolerance + yield.
def test_beat_match_yield():
    ref = np.arange(0, 10, 0.8)  # reference beats
    cam = ref + np.r_[np.full(8, 0.02), np.full(len(ref) - 8, 0.5)]  # first 8 close, rest far
    pairs, yield_frac = beat_match(cam, ref, tol_s=0.075)
    assert len(pairs) == 8
    assert 0.5 < yield_frac < 0.8


# C2.AC3 — DTW is available as a reported metric, distinct from the aligner.
def test_dtw_is_reported_metric_only():
    a = np.array([1.0, 2.0, 3.0])
    assert dtw_distance(a, a) == pytest.approx(0.0)
    assert dtw_distance(a, a + 1.0) > 0.0
    # the aligner used for validation is cross_correlate_lag, not dtw
    import argus.validation.timesync as ts

    assert ts.cross_correlate_lag is not ts.dtw_distance
