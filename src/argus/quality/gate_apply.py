"""Apply the motion gate to emitted records (H2.AC4/AC5).

The current gate state rides on every record. Under USABLE/REJECT, HRV and respiration are
suppressed; under REJECT, HR holds its last-good value (and is flagged), so the dashboard
shows "re-acquiring" rather than a silently frozen number.
"""

from __future__ import annotations

from dataclasses import replace

from ..contracts import SignalRecord
from .motion_gate import GOOD, REJECT, USABLE, MotionGate

SUPPRESS_UNDER = {USABLE, REJECT}


class GateController:
    def __init__(self, gate: MotionGate | None = None):
        self.gate = gate or MotionGate()
        self.state = self.gate.state
        self._last_good_hr: SignalRecord | None = None

    def step(self, motion: float, snr_db: float) -> str:
        self.state = self.gate.update(motion, snr_db)
        return self.state

    def apply(self, records: list[SignalRecord]) -> list[SignalRecord]:
        out: list[SignalRecord] = []
        for r in records:
            tagged = replace(r, gate=self.state)
            if r.name == "hr":
                if self.state == REJECT:
                    if self._last_good_hr is not None:  # hold last-good (H2.AC4)
                        out.append(
                            replace(self._last_good_hr, ts=r.ts, gate=REJECT,
                                    meta={**self._last_good_hr.meta, "held": True})
                        )
                    continue
                self._last_good_hr = tagged
                out.append(tagged)
            elif r.name in ("hrv", "resp"):
                if self.state in SUPPRESS_UNDER:  # suppress under motion
                    continue
                out.append(tagged)
            else:
                out.append(tagged)
        return out
