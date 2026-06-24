"""Dashboard render model (FR-19, B3, NFR-9).

Each signal renders value + SQI + traffic-light. A REJECT/degraded signal is visibly
marked and shows "re-acquiring" — never a silently frozen value (B3.AC3 / NFR-9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import SignalRecord

TRAFFIC_LIGHT = {"good": "🟢", "usable": "🟡", "reject": "🔴", "unknown": "⚪"}

PHASE0_SIGNALS = ("hr",)


def traffic_light(gate: str) -> str:
    return TRAFFIC_LIGHT.get(gate, "⚪")


@dataclass
class SignalView:
    name: str
    value: object
    sqi: float
    gate: str
    ts: float
    status: str  # ok | re-acquiring | degraded | stale

    @property
    def light(self) -> str:
        return traffic_light(self.gate)


@dataclass
class Dashboard:
    """Holds the latest record per signal and renders a snapshot."""

    phase: int = 1  # 0 = HR only, 1 = full bundle
    low_sqi: float = 0.3
    stale_after_s: float = 3.0
    _latest: dict[str, SignalRecord] = field(default_factory=dict)

    def update(self, record: SignalRecord) -> None:
        if self.phase == 0 and record.name not in PHASE0_SIGNALS:
            return
        self._latest[record.name] = record

    def _status(self, rec: SignalRecord, now: float) -> str:
        if rec.gate == "reject":
            return "re-acquiring"  # never a silently frozen value (NFR-9)
        if now - rec.ts > self.stale_after_s:
            return "stale"
        if rec.sqi < self.low_sqi:
            return "degraded"
        return "ok"

    def render(self, now: float | None = None) -> dict[str, SignalView]:
        out: dict[str, SignalView] = {}
        for name, rec in self._latest.items():
            t = now if now is not None else rec.ts
            out[name] = SignalView(
                name=name, value=rec.value, sqi=rec.sqi, gate=rec.gate, ts=rec.ts,
                status=self._status(rec, t),
            )
        return out

    def to_text(self, now: float | None = None) -> str:
        lines = []
        for v in self.render(now).values():
            val = f"{v.value:.1f}" if isinstance(v.value, (int, float)) else str(v.value)
            tag = "" if v.status == "ok" else f"  [{v.status}]"
            lines.append(f"{v.light} {v.name:<16} {val:>8}  sqi={v.sqi:.2f}{tag}")
        return "\n".join(lines)
