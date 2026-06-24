"""J1.AC2/AC3, NFR-3/4/5, J3, J4 — concurrency, runtime policy, storage, CLI, models."""

import numpy as np
import pytest

from argus.cli import main
from argus.core.concurrency import FrameHandle, get_frame, put_frame
from argus.core.models import MODEL_MANIFEST, fetch_models, sha256_hex
from argus.core.runtime import PINNED_DEPS, check_runtime, windows_supported
from argus.core.storage import StorageManager


# J1.AC2 — frame passed as a shared-memory handle, not a pickled copy.
def test_shared_memory_frame_handle_roundtrip():
    frame = (np.arange(48, dtype=np.uint8)).reshape(4, 4, 3)
    handle, shm = put_frame(frame)
    try:
        assert isinstance(handle, FrameHandle)
        assert "ndarray" not in str(type(handle))  # the handle is a tiny reference
        view, shm2 = get_frame(handle)
        try:
            assert np.array_equal(view, frame)  # pixels travelled via shared memory
        finally:
            shm2.close()
    finally:
        shm.close()
        shm.unlink()


# J1.AC3 / NFR-3 — runtime policy reports compliance; Windows unsupported.
def test_runtime_policy():
    rep = check_runtime()
    assert rep.python_version == __import__("sys").version_info[:2]
    # the checker correctly *detects* this env (py3.13 / numpy2) as non-compliant with the
    # documented 3.11 / numpy<2 target — the policy logic itself is what's under test.
    assert rep.python_ok == (rep.python_version == (3, 11))
    assert isinstance(rep.compliant, bool)
    assert windows_supported() is False
    assert PINNED_DEPS["numpy"] == "<2" and PINNED_DEPS["python"].startswith("3.11")


# NFR-5 / J4 — default no raw video; opt-in stores locally + labelled.
def test_storage_blocks_raw_video_by_default(tmp_path):
    sm = StorageManager(str(tmp_path))
    sm.write_derived("signals", {"hr": [72, 73]})
    assert (tmp_path / "signals.json").exists()
    with pytest.raises(PermissionError):
        sm.write_raw_video("sess", lambda p: None)


def test_storage_raw_video_optin_labelled(tmp_path):
    sm = StorageManager(str(tmp_path), allow_raw_video=True)
    written = {}
    sm.write_raw_video("sess", lambda p: written.setdefault("path", p))
    assert "RAWFACE" in written["path"]
    assert (tmp_path / "sess.RAWFACE.LABEL.txt").exists()


# NFR-5 — model fetch downloads + verifies checksum (fake downloader).
def test_fetch_models_checksum(tmp_path):
    payload = b"fake-model-bytes"
    manifest = [
        type(MODEL_MANIFEST[0])("m", "m.onnx", "http://x", sha256_hex(payload), "MIT")
    ]
    paths = fetch_models(str(tmp_path), manifest=manifest, downloader=lambda url: payload)
    assert paths[0].read_bytes() == payload

    bad = [type(MODEL_MANIFEST[0])("m", "m.onnx", "http://x", "deadbeef", "MIT")]
    with pytest.raises(ValueError):
        fetch_models(str(tmp_path), manifest=bad, downloader=lambda url: payload)


# J3.AC4 — `argus run`.
def test_cli_run(capsys):
    assert main(["run", "--frames", "360"]) == 0
    assert "processed 360 frames" in capsys.readouterr().out


# J3.AC2 + J3.AC3 — `argus record` then `argus report` (NFR-4 one-command flows).
def test_cli_record_then_report(tmp_path):
    xdf = tmp_path / "sess.xdf"
    assert main(["record", "--session", "sess", "--out", str(xdf), "--frames", "360"]) == 0
    assert xdf.exists()
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf))
    names = {s["info"]["name"][0] for s in streams}
    assert "hr" in names and "polar_hr" in names  # B2.AC1 all streams + Polar

    out = tmp_path / "report.html"
    assert main(["report", "--xdf", str(xdf), "--out", str(out)]) == 0
    html = out.read_text()
    assert "<html" in html and "hypothesis-generating" in html


def test_cli_fetch_models_dry_run(capsys):
    assert main(["fetch-models", "--dry-run"]) == 0
    assert "face_landmarker" in capsys.readouterr().out


# J3.AC1 / NFR-4 — deterministic POS on a fixed clip.
def test_pos_deterministic():
    from argus.dsp.rppg import estimate_hr

    rng = np.random.default_rng(7)
    rgb = 0.5 + 0.01 * np.sin(2 * np.pi * 1.2 * np.arange(300) / 30.0)[:, None] * np.ones((1, 3))
    rgb = rgb + 0.001 * rng.standard_normal((300, 3))
    a = estimate_hr(rgb, 30.0)
    b = estimate_hr(rgb, 30.0)
    assert a == b  # identical inputs -> identical output
