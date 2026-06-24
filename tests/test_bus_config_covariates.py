"""R23–R26: bus formatting, config defaults, covariates."""

import numpy as np
import pytest

from argus.bus.format import channel_layout, osc_address
from argus.config import Config
from argus.contracts import SignalRecord
from argus.quality.covariates import brightness_index, exposure_flags


# R23 — LSL channel layout [value..., sqi, gate_code]
def test_channel_layout_scalar():
    rec = SignalRecord("hr", 72.0, 0.8, 1.0, gate="good")
    assert channel_layout(rec) == [72.0, 0.8, 0.0]  # gate_code good=0


def test_channel_layout_vector_value():
    rec = SignalRecord("valence_arousal", [0.5, -0.2], 0.6, 1.0, gate="usable")
    assert channel_layout(rec) == [0.5, -0.2, 0.6, 1.0]  # usable=1


def test_channel_layout_rejects_non_numeric():
    rec = SignalRecord("gaze_zone", "left", 0.5, 1.0, gate="good")
    with pytest.raises(TypeError):
        channel_layout(rec)


# R24 — OSC address mapping for the TouchDesigner art leg
def test_osc_address_mapping():
    assert osc_address("hr") == "/argus/hr"
    assert osc_address("valence") == "/argus/valence"
    assert osc_address("/hr") == "/argus/hr"


def test_osc_address_empty_raises():
    with pytest.raises(ValueError):
        osc_address("")


# R25 — config defaults match the documented tunables
def test_config_documented_defaults():
    c = Config()
    assert c.camera.fps == 30 and c.camera.buffer_size == 1
    assert 8.0 <= c.rppg.window_s <= 15.0
    assert c.hrv.upsample_hz >= 256.0
    assert c.hrv.produce_lf_hf is False  # LF/HF never produced
    assert c.hrv.good_fraction_min == pytest.approx(0.80)
    assert c.respiration.band_hz == (0.08, 0.5)
    assert c.respiration.committed is False  # Indicative (rev 2)
    assert c.motion_gate.dwell_s >= 1.0
    assert c.gaze.model == "l2cs"
    assert c.bus.canvas == "touchdesigner"
    assert c.affect.au_model == "libreface"


def test_config_is_frozen():
    c = Config()
    with pytest.raises(Exception):
        c.camera.fps = 60


# R26 — relative brightness index + exposure flags
def test_brightness_index_dark_and_bright():
    dark = np.zeros((8, 8, 3), dtype=np.uint8)
    bright = np.full((8, 8, 3), 255, dtype=np.uint8)
    assert brightness_index(dark) == pytest.approx(0.0)
    assert brightness_index(bright) == pytest.approx(1.0)
    assert exposure_flags(dark) == (True, False)
    assert exposure_flags(bright) == (False, True)


def test_brightness_mid_not_flagged():
    mid = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert exposure_flags(mid) == (False, False)
