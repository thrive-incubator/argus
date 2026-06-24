# Implementation Notes

Decisions and conflict resolutions made while implementing the `./docs` spec.

## Scope of this implementation

The docs specify a real-time, hardware-dependent pipeline (live webcam via MediaPipe,
Polar H10 over BLE, ONNX models, Lab Streaming Layer). **None of that hardware/those heavy
ML runtimes exist in this environment**, and TECH_DESIGN §14a *itself* designates a
**headless-CI test set that is deterministic and fixture-free**:

> POS on a synthetic RGB trace; HRV on a synthetic BVP with known IBIs; pure unit tests;
> config validation. … Hardware-in-the-loop (camera, BLE, throughput) runs on a self-hosted
> runner.

This implementation builds and tests exactly that **deterministic core** — the pure-logic and
DSP layers that the hardware adapters depend on:

- Data contracts: `SignalRecord`, `FrameContext`, `Extractor` ABC + registry, gate enum + codes.
- Polar H10 HR-Measurement **byte parsing** (the full flag-driven offset algorithm) + beat-time
  reconstruction — pure function, the load-bearing correctness item (review B4).
- rPPG **POS** algorithm + windowing + HR-from-PSD, tested on synthetic RGB traces.
- **De Haan spectral SNR** (out-of-band-only denominator).
- **HRV** (SDNN, RMSSD via NeuroKit-equivalent formulas) on synthetic BVP/IBIs; cubic-spline
  upsampling; **no LF/HF**; GOOD-fraction ≥80% emit policy.
- **Respiration** band-pass (0.08–0.5 Hz) + FFT peak on synthetic chest displacement (Indicative).
- **Blink/EAR** + adaptive threshold + PERCLOS on synthetic EAR series.
- **Motion gate** 3-tier state machine with hysteresis + inter-ocular normalization.
- **Validation statistics**: Bland-Altman, MAE/RMSE/MAPE, Lin's CCC, EC13 + CTA-2065 checks.
- **Bus formatting** helpers (gate-code channel layout, OSC address mapping) — pure, no pylsl/osc
  import required.
- **Config** (typed, dataclass-based) with the documented tunable defaults.
- **Covariates**: relative brightness index + over/under-exposure flags.

Hardware **adapters** (MediaPipe backbone, `bleak` BLE transport, `pylsl` outlets, LibreFace,
the live dashboard, the capture thread) are **out of scope for the test suite** because they
cannot run headlessly; where useful, their pure-logic seams are implemented and tested, and the
I/O shell is left as thin, documented stubs. This is faithful to the docs' own CI/HIL split,
not a shortcut around it.

## Conflicts / "most recently modified" resolutions

- All six docs were last modified in the **rev 2** pass; the rev-2 amendment blocks
  (README top, ADR top) are authoritative. Applied accordingly:
  - **Respiration is Indicative** (no belt) → tests assert it *recovers a known rate*, not an
    accuracy bar (ADR-07 rev 2).
  - **Licensing firewall / `argus/research/` quarantine removed** (rev 2) → no import-lint
    requirement implemented; not a test.
  - **Canvas = TouchDesigner/OSC** → bus OSC formatting is the tested art-leg helper.
- No remaining substantive contradictions were found (the rev-1 verification pass + rev-2
  reconciliation already removed them).

## Environment deviation (justified)

- Docs pin **Python 3.11 + `numpy<2`** for *MediaPipe* coexistence (TECH §2a). This env has
  **Python 3.13 + numpy 2.x**, and **MediaPipe is not installed/used** here. The deterministic
  core is version-agnostic, so the test suite runs on 3.13/numpy-2. The 3.11/numpy<2 pin remains
  the requirement for the *full hardware build* and is unchanged in the docs.

## Test/build commands

- Test: `./venv/bin/pytest -q`
- Lint/build check: `./venv/bin/python -m compileall src tests` (no third-party linter assumed).
