"""Posture monitor: baseline + slouch/tilt/lean detection (review §6 hardening)."""

import numpy as np

from argus.perception.posture import (
    PostureDebouncer,
    PostureMonitor,
    posture_features,
)


def _face(nose_x=0.5, nose_y=0.4, roll=0.0):
    fl = np.zeros((478, 3))
    fl[1] = [nose_x, nose_y, 0]
    # eye outer corners for head-roll; rotate the inter-eye line by `roll` degrees
    cx, cy, half = 0.5, 0.35, 0.06
    a = np.radians(roll)
    fl[33] = [cx - half * np.cos(a), cy + half * np.sin(a), 0]   # right eye outer
    fl[263] = [cx + half * np.cos(a), cy - half * np.sin(a), 0]  # left eye outer
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
    assert abs(f.roll_deg) < 1.0             # level eyes -> ~0 roll


def test_features_none_without_pose():
    assert posture_features(_face(), None) is None


def test_good_posture_matches_baseline():
    mon = PostureMonitor()
    assert mon.assess(posture_features(_face(), _pose()))["status"] == "no baseline"
    mon.set_baseline(posture_features(_face(), _pose()))
    a = mon.assess(posture_features(_face(), _pose()))
    assert a["status"] == "good" and a["issues"] == []


def test_head_drop_reports_slouch():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(nose_y=0.40), _pose()))
    a = mon.assess(posture_features(_face(nose_y=0.62), _pose()))  # head dropped, distance same
    assert "head dropped / slouch" in a["issues"]
    assert a["status"] in ("fair", "poor")


def test_leaning_in_reports_leaning_toward_screen():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(nose_y=0.40), _pose(lx=0.35, rx=0.65)))  # sw=0.30
    # face closer (sw=0.40) but nose raised so neck_ratio stays ~constant -> NOT forward-head
    a = mon.assess(posture_features(_face(nose_y=0.30), _pose(lx=0.30, rx=0.70)))
    assert "leaning toward screen" in a["issues"]
    assert "forward-head" not in a["issues"]


def test_forward_head_requires_closer_and_shorter():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(nose_y=0.40), _pose(lx=0.35, rx=0.65)))
    # both face closer (sw up) AND neck shorter (nose dropped) -> true forward-head
    a = mon.assess(posture_features(_face(nose_y=0.60), _pose(lx=0.30, rx=0.70)))
    assert "forward-head" in a["issues"]
    assert "head dropped / slouch" not in a["issues"]  # not the single-cause label


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


def test_head_roll_detected():
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(roll=0.0), _pose()))
    a = mon.assess(posture_features(_face(roll=20.0), _pose()))  # head tilted ~20°
    assert "head tilted" in a["issues"]


def test_confidence_gate_withholds_on_low_visibility():
    mon = PostureMonitor(min_visibility=0.3)
    mon.set_baseline(posture_features(_face(), _pose()))
    a = mon.assess(posture_features(_face(nose_y=0.62), _pose()), visibility=0.1)
    assert a["status"] == "low confidence" and a["issues"] == []


def test_median_window_baseline_ignores_outlier():
    mon = PostureMonitor()
    mon.begin_baseline(n_frames=11)
    done = False
    for _ in range(10):
        done = mon.feed_baseline(posture_features(_face(nose_y=0.40), _pose()))
    # one corrupted outlier frame at capture time
    done = mon.feed_baseline(posture_features(_face(nose_y=0.95), _pose()))
    assert done and mon.has_baseline
    # median ignored the outlier -> a clean upright pose still reads "good"
    assert mon.assess(posture_features(_face(nose_y=0.40), _pose()))["status"] == "good"


def test_debouncer_requires_sustained_state():
    deb = PostureDebouncer(hold_s=3.0)
    assert deb.update("good", 0.0) == "good"
    # a momentary "poor" should not flip the badge
    assert deb.update("poor", 1.0) == "good"
    assert deb.update("good", 1.2) == "good"
    # sustained "poor" for >= hold_s eventually reports
    assert deb.update("poor", 2.0) == "good"
    assert deb.update("poor", 5.5) == "poor"


def test_baseline_persistence(tmp_path):
    mon = PostureMonitor()
    mon.set_baseline(posture_features(_face(), _pose()))
    p = tmp_path / "b.json"
    assert mon.save_baseline(str(p)) is True
    mon2 = PostureMonitor()
    assert mon2.load_baseline(str(p)) is True and mon2.has_baseline
    assert mon2.assess(posture_features(_face(), _pose()))["status"] == "good"
    assert PostureMonitor().load_baseline(str(tmp_path / "missing.json")) is False


def test_load_tolerates_baseline_without_roll(tmp_path):
    import json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"shoulder_width": 0.3, "neck_ratio": 1.0,
                             "lateral": 0.0, "tilt_deg": 0.0}))  # pre-roll schema
    mon = PostureMonitor()
    assert mon.load_baseline(str(p)) is True
    assert mon.baseline.roll_deg == 0.0
