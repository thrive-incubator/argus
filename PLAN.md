# PLAN — Testable Requirements Checklist

Derived from `./docs` (ADR / PRD / TECH_DESIGN / FEATURES, rev 2). Each item is a concrete,
testable requirement for the **deterministic core** (see NOTES.md for scope). Checked only when
its test(s) pass and the full suite is green.

## Data contracts
- [x] R1  `SignalRecord{name,value,sqi,ts,gate,meta}` dataclass; `gate` ∈ {good,usable,reject,unknown} (TECH §3.2)
- [x] R2  `gate_code` mapping good=0,usable=1,reject=2,unknown=3 (TECH §3.2/§10)
- [x] R3  `FrameContext` is frozen and `frame` is read-only (writeable=False enforced) (TECH §3.1)
- [x] R4  `Extractor` ABC: `consume(ctx)->list[SignalRecord]`; self-registering registry (TECH §3.3, ADR-19)

## Ground truth — Polar H10
- [x] R5  `parse_hr_measurement` full flag-driven offset: bit0 16-bit HR, bit3 energy-expended skip, bit4 RR; RR in 1/1024 s→ms; iterate all RR (ADR-17, FEAT C1.AC2)
- [x] R6  Beat-time reconstruction = cumulative sum of RR from a single anchor (ADR-16/17)

## Cardiac — rPPG / POS
- [x] R7  `pos(rgb_window)` recovers an injected HR from a synthetic RGB pulse trace within tolerance (ADR-03, TECH §6.1)
- [x] R8  HR extracted as PSD peak constrained to [42,240] bpm; band-pass [0.7,4.0] Hz (TECH §6.1)
- [x] R9  De Haan spectral SNR: in-band/(out-of-band only) over [0.7,4] Hz; signal bins excluded from denominator (ADR-06, TECH §7)

## HRV
- [x] R10 Cubic-spline upsample of BVP to ≥256 Hz before peak detection (ADR-05, TECH §6.2)
- [x] R11 SDNN & RMSSD from known IBI series match reference formulas within tolerance (TECH §6.2)
- [x] R12 LF/HF / frequency-domain HRV is NOT produced (no such API) (ADR-05, FEAT D3.AC5)
- [x] R13 HRV emit policy: compute only when ≥80% of window beats are GOOD+accepted (TECH §6.2, FEAT D3.AC3)

## Respiration (Indicative)
- [x] R14 Respiration band-pass [0.08,0.5] Hz + FFT peak recovers a known breathing rate from synthetic chest displacement; no accuracy bar (ADR-07 rev2, TECH §6.3)

## Ocular — blink / PERCLOS
- [x] R15 EAR from eye landmarks; adaptive per-session threshold from open-eye baseline; blink = EAR<thr for ≥N frames (ADR-08, TECH §6.4)
- [x] R16 PERCLOS (P80) = fraction of window with eyes ≥80% closed; increases with induced closure (FEAT F1.AC4)

## Motion quality gate
- [x] R17 3-tier gate GOOD/USABLE/REJECT from motion + SNR thresholds; **hysteresis** prevents flicker on borderline input (ADR-14, FEAT H2.AC3)
- [x] R18 Motion metrics normalized by inter-ocular distance (scale invariance) (ADR-14, FEAT H2.AC2)

## Validation statistics
- [x] R19 Bland-Altman: bias + 95% limits of agreement (ADR-18, FEAT I2.AC1)
- [x] R20 MAE, RMSE, MAPE computed correctly (ADR-18)
- [x] R21 Lin's Concordance Correlation Coefficient (CCC) (ADR-18)
- [x] R22 EC13 numeric-threshold check (≤ max(5 bpm,10%)) and CTA-2065 (MAPE<10%) pass/fail (ADR-18, FEAT D2.AC3)

## Bus formatting (pure helpers)
- [x] R23 LSL channel layout `[value..., sqi, gate_code]` builder (TECH §3.2/§10)
- [x] R24 OSC address mapping `/argus/<signal>` for the TouchDesigner art leg (TECH §10 rev2)

## Config & covariates
- [x] R25 Typed config with documented tunable defaults (window_s, bands, thresholds, bSQI, dwell…) (TECH §13)
- [x] R26 Relative brightness index from mean luma + over/under-exposure flags (uncalibrated, not lux) (TECH §9 rev2)
