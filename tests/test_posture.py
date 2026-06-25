"""Posture monitor: baseline + slouch/tilt/lean detection."""

import numpy as np

from argus.perception.posture import PostureMonitor, posture_features


def _face(nose_x=0.5, nose_y=0.4):
    fl = np.zeros((478, 3))
    fl[1] = [nose_x, nose_y, 0]
    return fl


def _pose(lx=0.35, ly=0.70, rx=0.65, ry=0.70):
    pl = np.zeros((33, 3))
    pl[11] = [lx, ly, 0]
    pl[12] = [rx, ry, 0]
    return pl


def test_features_basic():
    f = posture_features(_face(), _pose())
    assert abs(f.shoulder_width - 0.3) < 1e-4
    assert abs(f.neck_ratio - 1.0) < 1e-3   # (0.7-0.4)/0.3
    assert abs(f.lateral) < 1e-6
    assert abs(f.tilt_deg) < 1.0


def test_features_none_without_pose():
    assert posture_features(_face(), None) is None


def test_good_posture_matches_baseline():
    mon = PostureMonitor()
    assert mon.assess(posture_features(_face(), _pose()))["status"] == "no baseline"
    mon.set_baseline(posture_features(_face(), _pose()))
    a = mon.assess(posture_features(_face(), _pose()))
    assert a["status"] == "good" and a["issues"] == []


def test_slouch_detected_when_head_drops():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(nose_y=0.40), _pose()))
    a = mon.assess(posture_features(_face(nose_y=0.62), _pose()))  # head dropped toward shoulders
    assert "slouch / forward-head" in a["issues"]
    assert a["status"] in ("fair", "poor")


def test_forward_lean_detected_when_face_closer():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(), _pose(lx=0.35, rx=0.65)))   # sw=0.30
    a = mon.assess(posture_features(_face(), _pose(lx=0.30, rx=0.70)))     # sw=0.40 (closer)
    assert "slouch / forward-head" in a["issues"]


def test_shoulder_tilt_detected():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(), _pose()))
    a = mon.assess(posture_features(_face(), _pose(ly=0.66, ry=0.74)))  # one shoulder lower
    assert "shoulders tilted" in a["issues"]


def test_side_lean_detected():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(nose_x=0.50), _pose()))
    a = mon.assess(posture_features(_face(nose_x=0.62), _pose()))  # nose shifted right
    assert any("leaning" in i for i in a["issues"])


def test_baseline_persistence(tmp_path):
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(), _pose()))
    p = tmp_path / "b.json"
    assert mon.save_baseline(str(p)) is True
    mon2 = PostureMonitor()
    assert mon2.load_baseline(str(p)) is True and mon2.has_baseline
    assert mon2.assess(posture_features(_face(), _pose()))["status"] == "good"
    assert PostureMonitor().load_baseline(str(tmp_path / "missing.json")) is False
