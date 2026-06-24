# SUMMARY — Argus deterministic-core implementation

**Status: TASK_COMPLETE**

All three Definition-of-Done conditions hold:
1. Every requirement in `PLAN.md` is checked (26/26).
2. The full test suite passes with **63 passed, 0 failed, 0 skipped**.
3. The build/lint check (`python -m compileall src tests`) returns no errors.

## What was built

The deterministic, hardware-independent core of the Argus spec (`./docs`, rev 2). Per
TECH_DESIGN §14a, the docs themselves designate this layer as the headless-CI test set;
the hardware adapters (live MediaPipe backbone, `bleak` BLE transport, `pylsl` outlets,
LibreFace, capture thread, dashboard) run only on a self-hosted runner and are out of test
scope (see `NOTES.md`).

| Module (`src/argus/…`) | Responsibility | Requirements |
|---|---|---|
| `contracts.py` | `SignalRecord`, `FrameContext` (read-only frame), `Extractor` ABC + registry, gate codes | R1–R4 |
| `groundtruth/polar.py` | Polar H10 HR-Measurement full flag-driven parse + beat-time reconstruction | R5–R6 |
| `dsp/rppg.py` | POS algorithm, band-pass, HR-from-PSD | R7–R8 |
| `dsp/sqi.py` | De Haan spectral SNR (out-of-band-only denominator) | R9 |
| `dsp/hrv.py` | cubic-spline upsample, SDNN/RMSSD, **no LF/HF**, ≥80% GOOD-fraction policy | R10–R13 |
| `dsp/respiration.py` | chest-motion respiration (Indicative, 0.08–0.5 Hz) | R14 |
| `dsp/blink.py` | EAR, adaptive blink detector, PERCLOS | R15–R16 |
| `quality/motion_gate.py` | 3-tier gate with dwell hysteresis + IOD normalization | R17–R18 |
| `validation/stats.py` | Bland-Altman, MAE/RMSE/MAPE, Lin's CCC, EC13/CTA-2065 | R19–R22 |
| `bus/format.py` | channel layout `[value…, sqi, gate_code]`, OSC address mapping | R23–R24 |
| `config.py` | typed config with documented tunable defaults | R25 |
| `quality/covariates.py` | relative brightness index + exposure flags | R26 |

## Tests

`tests/` — 8 test modules, **63 tests**, all asserting behaviour against independent
hand computations / synthetic signals with known ground truth (no trivially-true tests).
Highlights: Polar RR parsing verified across **all four flag combinations** (the B4 bug
class); POS recovers injected HR within EC13 tolerance across 54–120 bpm; Lin's CCC drops
below Pearson under a constant offset; motion-gate hysteresis verified to suppress flicker.

Commands:
- Tests: `./venv/bin/pytest -q`
- Build/lint: `./venv/bin/python -m compileall src tests`

## Conflicts / deviations noted

None substantive (rev-2 reconciliation already removed doc conflicts). One justified
environment deviation: docs pin Python 3.11 + `numpy<2` for *MediaPipe* coexistence; this
env runs Python 3.13 + numpy 2.x and MediaPipe is not used, so the version-agnostic core
is tested there. Full detail in `NOTES.md`.

## Out of scope (faithful to the docs' CI/HIL split)

Live capture/throughput, BLE transport, LSL outlets/XDF recording, MediaPipe/LibreFace
inference, the validation protocol runner UI, and the dashboard — all require hardware or
heavy ML runtimes unavailable headlessly. Their pure-logic seams are implemented and tested;
the I/O shells are left as documented thin adapters for the hardware-in-the-loop runner.

## Test counts
- 63 passed · 0 failed · 0 skipped
- 26/26 requirements checked
