"""Motion quality gate — 3-tier traffic light with hysteresis (ADR-14, TECH §8).

GOOD  (low motion AND snr >= snr_good) -> HR + HRV + RR
USABLE(mild motion OR  snr in [snr_reject, snr_good)) -> HR only, suppress HRV/RR
REJECT(high motion OR  snr < snr_reject) -> drop window, hold last good value

Hysteresis: a tentative new state must persist for ``dwell_frames`` before it is
adopted, so borderline input does not flicker the gate (review item H2.AC3).
Motion is normalised by inter-ocular distance for scale invariance (H2.AC2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

GOOD, USABLE, REJECT = "good", "usable", "reject"


def normalize_motion(displacement: float, inter_ocular_distance: float) -> float:
    """Scale a raw landmark displacement by inter-ocular distance (H2.AC2)."""
    if inter_ocular_distance <= 1e-9:
        raise ValueError("inter-ocular distance must be > 0")
    return float(displacement) / float(inter_ocular_distance)


def _instant_state(
    motion: float,
    snr_db: float,
    motion_good: float,
    motion_reject: float,
    snr_good: float,
    snr_reject: float,
) -> str:
    if motion > motion_reject or snr_db < snr_reject:
        return REJECT
    if motion <= motion_good and snr_db >= snr_good:
        return GOOD
    return USABLE


@dataclass
class MotionGate:
    """Stateful 3-tier gate with dwell-based hysteresis."""

    motion_good: float = 0.15  # normalised displacement
    motion_reject: float = 0.40
    snr_good: float = 3.0  # dB
    snr_reject: float = 0.0  # dB
    dwell_frames: int = 30  # ~1 s at 30 fps (ADR-14 dwell >= 1 s)
    state: str = USABLE
    _pending: str = field(default=USABLE)
    _pending_count: int = 0

    def update(self, motion: float, snr_db: float) -> str:
        """Advance one frame; return the (possibly unchanged) committed state."""
        target = _instant_state(
            motion,
            snr_db,
            self.motion_good,
            self.motion_reject,
            self.snr_good,
            self.snr_reject,
        )
        if target == self.state:
            self._pending = self.state
            self._pending_count = 0
            return self.state

        if target == self._pending:
            self._pending_count += 1
        else:
            self._pending = target
            self._pending_count = 1

        if self._pending_count >= self.dwell_frames:
            self.state = target
            self._pending_count = 0
        return self.state
