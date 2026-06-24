"""FR-21/I1 protocol runner + FR-22/I2 report generator + Pearson r."""

import numpy as np
import pytest

from argus.bus.recorder import Recorder
from argus.validation.protocol import (
    FakePrompter,
    ProtocolRunner,
    default_protocol,
    hrv_eligible_blocks,
    measured_lux,
    respiration_blocks,
)
from argus.validation.report import (
    FEASIBILITY_BANNER,
    fitzpatrick_caveat,
    generate_report,
    write_report_html,
)
from argus.validation.stats import pearson_r


# Pearson r — association.
def test_pearson_r():
    x = np.array([1.0, 2, 3, 4])
    assert pearson_r(x, x) == pytest.approx(1.0)
    assert pearson_r(x, -x) == pytest.approx(-1.0)


# I1.AC1 — scripted blocks + markers injected into the recorder.
def test_protocol_runner_injects_markers():
    rec = Recorder()
    runner = ProtocolRunner(recorder=rec, prompter=FakePrompter())
    blocks = default_protocol(repeats=2)
    markers = runner.run(blocks)
    assert len(markers) == len(blocks)
    assert rec.sample_count("marker") == len(blocks)  # one marker per block in the XDF
    assert len(runner.prompter.prompts) == len(blocks)
    # ≥2 repeats of rest
    assert sum(1 for m in markers if m.kind == "rest") == 2


# I1.AC2 — HRV from rest only (excl 6-brpm); respiration from paced; measured lux logged.
def test_block_classification_and_lux():
    runner = ProtocolRunner(prompter=FakePrompter())
    markers = runner.run(default_protocol(repeats=1))
    assert all(m.kind == "rest" for m in hrv_eligible_blocks(markers))
    assert all("6" not in m.name for m in hrv_eligible_blocks(markers))  # 6-brpm excluded
    paced = respiration_blocks(markers)
    assert {m.params["brpm"] for m in paced} == {6, 10, 15}
    assert measured_lux(markers) == {"lighting_A": 150, "lighting_B": 500}


def test_marker_timestamps_advance():
    runner = ProtocolRunner(prompter=FakePrompter())
    markers = runner.run(default_protocol(repeats=1))
    starts = [m.t_start for m in markers]
    assert all(b > a for a, b in zip(starts, starts[1:]))


# I2.AC1-4 — report content.
def test_report_full_agreement_set():
    rng = np.random.default_rng(0)
    ref = 70 + rng.normal(0, 3, 30)
    rep = generate_report(
        {
            "rest": {
                "hr_measured": ref + rng.normal(0, 1.5, 30),
                "hr_ref": ref,
                "hr_inst_measured": ref + rng.normal(0, 3, 30),
                "hr_inst_ref": ref,
                "sdnn_measured": np.array([45.0, 50, 48]),
                "sdnn_ref": np.array([46.0, 49, 47]),
                "rmssd_measured": np.array([30.0, 35, 33]),
                "rmssd_ref": np.array([31.0, 34, 32]),
            }
        },
        fitzpatrick=5,
    )
    assert rep["banner"] == FEASIBILITY_BANNER
    assert "worst-case" in rep["fitzpatrick_caveat"]
    hr = rep["conditions"]["rest"]["hr"]
    for k in ("bias", "loa_lower", "mae", "rmse", "mape", "pearson_r", "lins_ccc"):
        assert k in hr["avg_60s"]
    assert "Polar H10 (reference" in hr["avg_60s"]["reference"]
    assert hr["ec13_pass"] is True
    assert "quasi_instant_4_10s" in hr  # I2.AC2 second window
    sdnn = rep["conditions"]["rest"]["sdnn"]
    assert sdnn["mae_bar_ms"] == 12.0 and sdnn["passes_bar"] is True
    ln = rep["conditions"]["rest"]["ln_rmssd"]
    assert ln["ms_band_applied"] is False and "log-units" in ln["units"]


def test_fitzpatrick_caveat_branches():
    assert "best case" in fitzpatrick_caveat(2)
    assert "worst-case" in fitzpatrick_caveat(6)
    assert "mid-range" in fitzpatrick_caveat(4)


# I2.AC5 — one command writes the HTML report.
def test_write_report_html(tmp_path):
    rep = generate_report({"rest": {"hr_measured": [70.0, 71], "hr_ref": [70.0, 70]}})
    out = tmp_path / "report.html"
    write_report_html(rep, str(out))
    html = out.read_text()
    assert "<html" in html and FEASIBILITY_BANNER in html and "rest" in html
