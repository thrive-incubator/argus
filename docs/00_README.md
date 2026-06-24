# Argus — Documentation Index

*Camera-only physiological & behavioral sensing layer. Phase 0 & Phase 1.*

**For:** the engineering team implementing the sensing layer. **Status:** rev 2, approved for
implementation. **Date:** 2026-06-24.

> ## Scope amendment (rev 2) — authoritative; supersedes earlier text where they conflict
> 1. **No new hardware purchases.** Respiration is therefore **Indicative** (motion-only — no
>    respiratory belt). Lighting lux is logged via a **free phone app** (no purchased meter).
>    The optional pulse-ox is used **only if already owned**. *(Ground-truth Polar H10 — see the
>    open question at the bottom: confirm whether one is already on hand.)*
> 2. **Productification & licensing are NOT a constraint at this stage.** The entire
>    licensing-firewall / `argus/research/` quarantine / import-lint apparatus is **removed**.
>    Use the **best available model regardless of license** — research Action Units now use
>    **LibreFace (or OpenFace 3.0)** directly. Commercial diligence items are dropped for now.
> 3. **Accuracy/fairness clearance is explicitly out of scope** at this stage (single-subject
>    feasibility framing stays).
> 4. **Art-canvas medium = TouchDesigner over OSC** (default). Swap to p5.js/WebSocket only if
>    the piece needs to be a shareable web link — a one-line bridge change.

## What Argus is

A real-time pipeline that extracts latent signals (heart rate, HRV, respiration, facial affect,
blink/gaze, motion) from a single ordinary webcam and emits them on a synchronized, quality-
tagged bus. The same sensing core feeds (a) a generative-art / biofeedback canvas and (b) Thrive
research. **Phase 0/1 builds and validates the sensing layer on a single consenting adult (the
developer)**; children, dyads, and field deployment are later phases and out of scope here.

## Read in this order

| Doc | What it answers | Read if you are… |
|---|---|---|
| **[01_ARCHITECTURE_DECISIONS.md](01_ARCHITECTURE_DECISIONS.md)** | *Why* each technology was chosen (20 ADRs, SOTA-reviewed, with what changed vs the brief) | deciding or questioning a technical choice |
| **[02_PRD.md](02_PRD.md)** | *What* we're building and the goals/scope/signals/requirements | setting scope or priorities |
| **[03_TECH_DESIGN.md](03_TECH_DESIGN.md)** | *How* to build it — modules, data contracts, algorithms, threading, bus, validation, tests, build order | writing code |
| **[04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md)** | The backlog: features + testable acceptance criteria + phase gates + traceability | implementing or QA-ing a feature |
| **[05_REVIEW_LOG.md](05_REVIEW_LOG.md)** | Every issue the adversarial review found and how it was resolved (rev 1) | curious why a doc says what it says |

The original [../PROJECT_BRIEF.md](../PROJECT_BRIEF.md) is the founding context (all phases).

## The two phase gates (the only milestones that matter here)

- **Phase 0 — "the spine":** webcam → MediaPipe → POS heart rate → bus → dashboard, validated
  against a Polar H10. Exit: HR meets the EC13 numeric threshold at rest (≥300 lux),
  capture→emit ≤ 2 s, synchronized XDF + first report.
- **Phase 1 — "engineer-ready sensing layer":** full signal bundle + motion gate + quality/
  covariate layer + dual bus (LSL→XDF + OSC/WebSocket) + validation harness. Exit: every
  committed signal passes its feasibility bar or is documented as flagged; licensing gate green.

## Start here (engineers)

**Spike-0 first** (TECH §2a) — resolve the dependency matrix (MediaPipe + ONNXRuntime, Py-Feat
isolated) and the macOS Camera+Bluetooth permissions. Nothing else proceeds until Spike-0 is
green. Then follow the build order in TECH §15.

## Non-negotiable principles

1. **Signals are estimates; states are hypotheses.** Every output carries an SQI/confidence;
   nothing is shown as a verdict. (Especially important before this ever touches a child.)
2. **Quality-gate everything**; emit low-quality with a flag, never a silently frozen value.
3. **Validate against ground truth** before believing any number; single-subject results are
   **feasibility, not accuracy/fairness clearance**.
4. **Privacy by construction** — derived signals + XDF by default, not raw face video.
5. *(rev 2: the "commercial-license-clean product path" principle is **dropped** — use the best
   model regardless of license; no quarantine.)*

## Hardware you need (rev 2 — buy nothing)

Modern laptop (M-series Mac or recent x86), 720p+ webcam, and a **Polar H10** chest strap **if
already owned** (cardiac ground truth — see open question). Lighting lux via a **free phone app**.
No belt, no lux meter, no required purchases. Optional Contec CMS50E pulse-ox only if already on
hand.
