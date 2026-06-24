"""Typed configuration with the documented tunable defaults (TECH §13).

Plain dataclasses (no third-party dependency) so the core stays light. Every value
here is a ``[tunable]`` from the design docs; defaults match the doc text.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    fps: int = 30
    width: int = 1280
    height: int = 720
    buffer_size: int = 1  # CAP_PROP_BUFFERSIZE=1 (TECH §11 / ADR-16)


@dataclass(frozen=True)
class RppgConfig:
    window_s: float = 10.0  # 8–15 s rolling window (ADR-03)
    update_hz: float = 1.0
    method: str = "pos"  # POS live path; CHROM selectable
    band_hz: tuple[float, float] = (0.7, 4.0)
    hr_min_bpm: float = 42.0
    hr_max_bpm: float = 240.0
    yaw_cutoff_deg: float = 25.0


@dataclass(frozen=True)
class HrvConfig:
    upsample_hz: float = 256.0  # >= 256 Hz (ADR-05)
    bsqi_threshold: float = 0.86  # PROVISIONAL — calibrate vs H10 (review I5)
    window_s: float = 90.0  # 60–120 s
    good_fraction_min: float = 0.80  # review M9
    produce_lf_hf: bool = False  # NEVER produce LF/HF (ADR-05)


@dataclass(frozen=True)
class RespirationConfig:
    window_s: float = 20.0  # 15–30 s
    band_hz: tuple[float, float] = (0.08, 0.5)  # widened low edge (review sM1)
    committed: bool = False  # Indicative (no belt, rev 2)


@dataclass(frozen=True)
class BlinkConfig:
    baseline_s: float = 10.0
    threshold_ratio: float = 0.6
    min_frames: int = 2
    perclos_window_s: float = 75.0  # 60–90 s
    min_fps: float = 25.0  # blink metrics invalid below this


@dataclass(frozen=True)
class GazeConfig:
    model: str = "l2cs"  # default L2CS-Net ResNet-18 (ADR-09 rev 1)
    calibration_points: int = 0  # optional 5–9 point calibration


@dataclass(frozen=True)
class MotionGateConfig:
    motion_good: float = 0.15
    motion_reject: float = 0.40
    snr_good_db: float = 3.0
    snr_reject_db: float = 0.0
    dwell_s: float = 1.0  # hysteresis dwell >= 1 s (ADR-14)
    pitch_weight: float = 2.0  # nodding hurts ~2x more than yaw


@dataclass(frozen=True)
class BusConfig:
    osc_namespace: str = "/argus"
    osc_target: tuple[str, int] = ("127.0.0.1", 7000)
    ws_target: tuple[str, int] = ("127.0.0.1", 8765)
    forward_sync_ms: float = 50.0
    canvas: str = "touchdesigner"  # rev 2 default (OSC art leg)


@dataclass(frozen=True)
class AffectConfig:
    live_hz: float = 12.0  # 10–15 Hz tunable (review m6)
    au_model: str = "libreface"  # LibreFace / OpenFace 3.0 (ADR-12 rev 2)
    au_hz: float = 10.0  # 5–15 fps decoupled


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    rppg: RppgConfig = field(default_factory=RppgConfig)
    hrv: HrvConfig = field(default_factory=HrvConfig)
    respiration: RespirationConfig = field(default_factory=RespirationConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    motion_gate: MotionGateConfig = field(default_factory=MotionGateConfig)
    bus: BusConfig = field(default_factory=BusConfig)
    affect: AffectConfig = field(default_factory=AffectConfig)
