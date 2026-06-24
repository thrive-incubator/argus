# Argus — Product Requirements Document (PRD)

*Camera-only physiological & behavioral sensing layer — Phase 0 & Phase 1.*

**Status:** Approved for implementation · **Date:** 2026-06-24 · **Owner:** Jean-Baptiste Passot
**Companion docs:** [01_ARCHITECTURE_DECISIONS.md](01_ARCHITECTURE_DECISIONS.md) · [03_TECH_DESIGN.md](03_TECH_DESIGN.md) · [04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md)

---

## 1. Purpose & background

Argus is a real-time pipeline that extracts latent physiological and behavioral signals
from a single ordinary RGB camera and emits them as a synchronized, quality-tagged stream.
The same sensing core serves two consumers: (a) a generative-art / biofeedback canvas, and
(b) Thrive research on the emotional and physiological states of children and families.

This PRD covers **only Phase 0 and Phase 1**: building and *validating* the sensing layer on
a **single consenting adult (the developer)**. Children, dyads, and naturalistic field
deployment are deliberately out of scope here (Phases 2–4) but the architecture must not
foreclose them.

**Why this sequencing:** validating signal extraction on a single adult who can wear
ground-truth sensors is the fastest, lowest-risk way to earn trust in the numbers before
any of them inform research conclusions or drive an art piece. If the canvas feels alive and
the numbers agree with a chest strap, the stream is good enough to do science with.

---

## 2. Goals & non-goals

### 2.1 Goals
- **G1.** Extract a defined bundle of signals from a webcam in real time on commodity
  hardware (a single modern laptop, CPU-first).
- **G2.** Emit signals on a low-latency, synchronized **bus** that both a research logger and
  a biofeedback/art consumer can subscribe to, decoupled from the pipeline internals.
- **G3.** **Validate** each committed signal against contact-sensor ground truth (Polar H10;
  optional pulse-ox) using a rigorous, pre-registered methodology.
- **G4.** Attach a **quality/confidence value (SQI)** to every emitted signal and gate
  unreliable output, so no consumer is misled by bad signal.
- **G5.** *(rev 2: WITHDRAWN — productization is out of scope at this stage, so license is not a
  selection criterion; use the best model regardless of license.)*

### 2.2 Non-goals (Phase 0/1)
- **NG1.** No children, no dyads, no multi-subject. Single adult only.
- **NG2.** No clinical/diagnostic claims. Argus is a research & wellness instrument, **not a
  medical device**. Outputs are estimates; states are hypotheses with confidence.
- **NG3.** No pupillometry, no SpO₂, no blood pressure, no LF/HF HRV (see ADR-05, ADR-10).
- **NG4.** No deep-learning rPPG in the live path (offline reference only).
- **NG5.** The art canvas itself is not built here — Phase 0/1 delivers the bus + a
  validation dashboard. The canvas is a downstream consumer.
- **NG6.** No cloud. All processing is local/on-device in Phase 0/1.

---

## 3. Users & use cases (Phase 0/1)

| User | Role | Primary use case |
|---|---|---|
| **Developer/researcher (JB)** | Builder + sole subject | Run the rig on themselves, watch live signals, run validation sessions against the Polar H10, export data. |
| **Engineering team** | Implementers | Build to this spec; extend extractors via a stable plugin interface. |
| **Future research consumer** | Downstream (designed-for, not active) | Subscribe to the LSL stream / read XDF recordings for analysis. |
| **Future art/biofeedback consumer** | Downstream (designed-for, not active) | Subscribe to the OSC/WebSocket bridge to drive a canvas. |

**Primary Phase 0/1 user journey:**
1. Developer attaches Polar H10, launches Argus, faces the webcam.
2. Within a few seconds, a live dashboard shows heart rate (Phase 0) and the full signal set
   (Phase 1), each with a quality indicator.
3. Developer runs a structured validation protocol (rest / paced-breathing / light-motion /
   two lighting levels), recorded to a synchronized XDF file alongside the H10 reference.
4. Developer runs the validation report generator → agreement statistics vs ground truth.
5. Data and report are exported for the record.

---

## 4. Scope by phase

### 4.1 Phase 0 — *Prove the spine* (single signal, end-to-end)
The minimal end-to-end vertical slice: **camera → MediaPipe backbone → live HR via POS →
bus → dashboard**, validated against the Polar H10.

**In scope:**
- Webcam capture with correct per-frame timestamping (ADR-16).
- MediaPipe Face Landmarker backbone (ADR-01).
- rPPG ROI extraction + POS heart-rate estimator on a rolling window (ADR-03).
- Spectral-SNR SQI on the HR estimate (ADR-06, partial).
- LSL outlet for HR + a minimal live dashboard showing HR and its quality (ADR-15).
- Polar H10 ingestion + LSL outlet + synchronized XDF recording (ADR-16, ADR-17).
- A first validation report: live HR vs H10 (Bland-Altman, MAE, MAPE) (ADR-18).

**Phase 0 exit criteria:** live HR on the developer agrees with the Polar H10 within the
**EC13 *numeric threshold* (≤ ±5 bpm or ±10%)** — a single-subject feasibility target, not
EC13 conformance — under **good lighting (≥ 300 lux at the face)** at rest, with **capture→emit
latency ≤ 2 s** (see NFR-1 for the precise definition), and a recorded, synchronized session
that the report generator can process.

### 4.2 Phase 1 — *Full signal set + quality gating + validation harness*
Expand to the full Phase 0/1 signal bundle, add the quality/covariate layer, the research
logger, the motion gate, and the complete validation harness.

**In scope (added on top of Phase 0):**
- **HRV** (SDNN committed, RMSSD indicative) with waveform upsampling + bSQI beat gating
  (ADR-05, ADR-06).
- **Respiration** — primary chest/shoulder-motion estimator + rPPG-RR cross-check (ADR-07).
- **Blink rate / blink duration / PERCLOS** with adaptive threshold (ADR-08).
- **Gaze** — zone-level direction with learned head + MediaPipe front-end (ADR-09).
- **Facial affect (live)** — blendshapes + HSEmotion emotion + valence/arousal (ADR-11).
- **Facial Action Units (research)** — Py-Feat, decoupled/offline (ADR-12); optional
  quarantined OpenFace 3.0/LibreFace reference (ADR-13).
- **Head/posture motion & fidget** from BlazePose (ADR-02).
- **Motion quality gate** — 3-tier hysteretic traffic light, applied to all signals (ADR-14).
- **Covariate layer** — skin-tone estimate (Monk/Fitzpatrick), lighting (lux proxy), global
  motion, face-presence — published alongside signals (ADR-18).
- **Dual bus** fully realized: LSL→XDF research logging + LSL→OSC/WebSocket art bridge
  (ADR-15).
- **Validation harness** — structured protocol runner + full agreement-statistics report
  (BA, MAE, RMSE, MAPE, Pearson r, Lin's CCC; **SDNN MAE ≤ 12 ms vs ±15 ms band; ln-RMSSD in
  log-units, indicative**) (ADR-18).
- **Richer dashboard** — all signals, each with SQI and traffic-light state.

**Phase 1 exit criteria:** every signal in the bundle has a status of **validated** (meets
its pre-registered acceptance bar against ground truth on the developer) **or explicitly
flagged** (with documented performance envelope), each emits an SQI, the motion gate
demonstrably suppresses output under induced motion, and a single command produces the
synchronized recording + validation report.

---

## 5. Signal bundle & commitments (Phase 0/1)

Each signal's **commitment level** sets whether it has a hard acceptance bar or is logged as
exploratory. "GT" = ground-truth reference for validation.

| Signal | Phase | Commitment | GT reference | Acceptance bar (single-subject feasibility) |
|---|---|---|---|---|
| Heart rate (HR) | 0 | **Committed** | Polar H10 | EC13 *numeric threshold*: MAE ≤ max(5 bpm, 10%); MAPE < 10% at rest, ≥300 lux |
| HRV — SDNN | 1 | **Committed** | Polar H10 R-R | **SDNN MAE ≤ 12 ms** at rest; also report BA bias+LoA & Lin's CCC; length-matched windows |
| HRV — RMSSD | 1 | Indicative | Polar H10 R-R | ln-RMSSD BA in log-units/ratio + CI; no pass/fail, no ms band |
| HRV — LF/HF | — | **Excluded** | — | Not produced (ADR-05) |
| Respiration rate | 1 | **Indicative (rev 2 — no belt)** | Metronome = coarse plausibility only (no valid RR GT) | Estimate + plausibility reported; **no accuracy/MAE claim** |
| Blink rate / duration | 1 | **Committed** | Frame-level manual annotation (±N-frame match; κ on subset) | F1 ≥ 0.90 at 30 fps, ≥300 lux, **no glasses** |
| PERCLOS | 1 | Indicative | Commanded closure-fraction blocks | Tracks commanded closure fraction; no drowsiness GT in scope |
| Gaze (zones) | 1 | Indicative | Scripted look-target protocol (fixed geometry) | **3×3 confusion matrix + accuracy above 33% chance** (not a bare %) |
| Affect — valence/arousal | 1 | Exploratory | Posed-expression protocol (stimulus, not GT) | **Model face-validity check** only: V/A covaries with posed config; ≠ felt affect |
| Affect — emotion class | 1 | Exploratory | Posed-expression protocol | Descriptive confusion matrix on posed set; no pass/fail |
| Facial Action Units | 1 | Exploratory (research log) | **LibreFace / OpenFace 3.0** (0–5 AU intensity; license irrelevant, rev 2) | Descriptive; no gate |
| Head/posture motion, fidget | 1 | Exploratory | n/a (descriptive) | Stable, lightly-filtered, plausible |
| Skin-tone covariate | 1 | Covariate | Self-reported Monk/Fitzpatrick (+ unvalidated cheek estimate) | Recorded **for Phase-2 forward-compat only** (no fairness info at n=1) |
| **Eyewear covariate** | 1 | Covariate | Self-reported | Recorded per session (gates blink/gaze) |
| **Facial-hair covariate** | 1 | Covariate | Self-reported | Recorded per session (gates cheek rPPG ROI) |
| Lighting / motion / presence covariates | 1 | Covariate | n/a | Published alongside every signal; lighting is a **relative brightness index (uncalibrated)** + a measured lux value per block |

**Guiding principle (normative, from the brief):** *Signals are estimates; states are
hypotheses.* No signal is ever displayed as a verdict. Every output carries an SQI/confidence
and, for fused states (arousal/valence), an explicit "estimate, confidence X" framing.

---

## 6. Functional requirements

> IDs (FR-n) are referenced by acceptance criteria in [04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md).

### Capture & backbone
- **FR-1.** Capture from a webcam at ≥ 30 fps, 720p+, with `CAP_PROP_BUFFERSIZE=1`, stamping
  each frame with `pylsl.local_clock()` at grab time.
- **FR-2.** Run a single MediaPipe Face Landmarker pass per frame producing a `FrameContext`
  (landmarks, iris, blendshapes, head-pose, capture timestamp).
- **FR-3.** Run a single MediaPipe Pose pass per frame (separate Task), One-Euro-filtered.
- **FR-4.** The system must sustain real-time throughput; under load it drops *frames* for
  per-frame extractors (latest-frame wins) but never drops samples from the lossless
  time-series ring buffer feeding rPPG/HRV/respiration.

### Extractors (each emits `{signal_name, value, sqi, timestamp}`)
- **FR-5.** HR via POS over a rolling 8–15 s window, ~1 Hz update (Phase 0).
- **FR-6.** HRV (SDNN, RMSSD) via waveform upsampling ≥256 Hz → systolic-peak detection →
  IBI → NeuroKit2, gated by bSQI; rest-only validity flag (Phase 1).
- **FR-7.** Respiration via chest/shoulder displacement band-passed **0.08–0.5 Hz** over a
  15–30 s window, referenced to a **respiratory belt (RIP)**; rPPG-RR cross-check (Phase 1).
- **FR-8.** Blink (rate, duration), PERCLOS via adaptive-threshold EAR (Phase 1).
- **FR-9.** Gaze zone (L/C/R, screen on/off, attention present/absent) via learned head +
  iris front-end; optional 5–9-point calibration (Phase 1).
- **FR-10.** Live affect: blendshapes + HSEmotion emotion + valence/arousal (Phase 1).
- **FR-11.** Research AUs: Py-Feat, decoupled at 5–15 fps or offline (Phase 1).
- **FR-12.** Head/posture motion + fidget index from BlazePose (Phase 1).

### Quality, covariates, fusion
- **FR-13.** Compute per-window SQI (De Haan SNR + skewness; Orphanidou bSQI for HRV beats).
- **FR-14.** Motion quality gate: 3-tier hysteretic traffic light (GOOD/USABLE/REJECT) from
  landmark motion + head pose + pulse SNR; gate downstream signal emission accordingly.
- **FR-15.** Publish covariates (skin-tone, lighting proxy, global motion, face presence)
  as their own streams.

### Bus, logging, dashboard
- **FR-16.** Publish every signal/covariate on an LSL `StreamOutlet` (regular-rate where
  applicable; `IRREGULAR_RATE` for events).
- **FR-17.** Record all LSL streams + the Polar H10 stream to a single synchronized **XDF**
  file (LabRecorder-compatible), with clock-offset metadata.
- **FR-18.** Bridge selected signals LSL→OSC (and LSL→WebSocket) for the biofeedback/art
  consumer, with a small forward-sync look-ahead for jitter-free visuals.
- **FR-19.** Live dashboard: render each signal with value, SQI, and traffic-light state;
  Phase 0 = HR only; Phase 1 = full bundle.

### Ground truth & validation
- **FR-20.** Ingest Polar H10 over BLE (`0x2A37`), parse RR (1/1024 s units), reconstruct
  beat times by cumulative-sum from one anchor; Kubios artifact correction.
- **FR-21.** Validation protocol runner: scripted blocks (rest, paced-breathing, light-motion,
  ≥2 lighting levels), with timestamps/markers in the XDF.
- **FR-22.** Validation report generator: per-condition Bland-Altman (bias + 95% LoA), MAE,
  RMSE, MAPE, Pearson r, Lin's CCC, SNR; **SDNN (MAE ≤ 12 ms, ±15 ms band, length-matched);
  ln-RMSSD in log-units (indicative)**; EC13 + CTA-2065 numeric-threshold pass/fail; H10 as
  reference; feasibility framing.
- **FR-23.** Record subject Monk/Fitzpatrick skin tone per session.

---

## 7. Non-functional requirements

- **NFR-1 — Latency.** *Defined (rev 1):* the wall-clock delay between a frame's **capture
  timestamp** and the bus emission of the first `SignalRecord` whose window includes that
  frame. It **explicitly excludes** the estimator's intrinsic windowing lag — a 10 s rPPG
  window carries ~½-window (~5 s) of inherent physiological smoothing, which is separate and
  *cannot* be made ≤ 2 s. Targets: capture→emit ≤ 2 s for the live HR channel in Phase 0; the
  per-frame display path (landmarks/affect/gaze/motion) ≤ 150 ms. *(The biofeedback "feels
  alive" target — which also depends on intrinsic window lag — is an open tuning question.)*
- **NFR-2 — Throughput.** Sustain ≥ 30 fps backbone on a single modern laptop CPU (M-series
  or recent x86); GPU optional, not required.
- **NFR-3 — Platforms.** macOS (primary dev) and Linux. **Windows is unsupported in Phase
  0/1** (rev 1 — `bleak` WinRT BLE and camera backends differ enough that "best-effort" was
  meaningless; not tested, not a target).
- **NFR-4 — Reproducibility.** Pinned dependencies; deterministic POS path; a single command
  to run a validation session and a single command to produce its report.
- **NFR-5 — Privacy by construction.** Default to storing *derived signals + XDF*, not raw
  face video. Raw-video capture is an explicit, off-by-default opt-in for the developer's own
  validation clips (e.g., blink-annotation ground truth), stored locally only and **never
  committed to the repo** (fixtures with faces are local/LFS-restricted, never in CI history).
  *Note:* "no cloud" (NFR-6) refers to runtime processing; first-run **model-weight downloads**
  are permitted and then cached/pinned for offline operation (TECH §17a, `argus fetch-models`).
- **NFR-6 — Licensing.** *(rev 2: WITHDRAWN — productification is out of scope. No license
  constraint, no quarantine, no import-lint. Use the best model regardless of license. "No cloud"
  in NFR-5 still holds for runtime data; first-run model downloads are still permitted.)*
- **NFR-7 — Extensibility.** New extractors implement a single `Extractor.consume(FrameContext)
  -> list[SignalRecord]` interface (rev 1: returns a list — `[]` when no window is ready) and
  self-register; adding one must not require changes to the backbone or bus.
- **NFR-8 — Observability.** Per-extractor timing/health metrics; dropped-frame counters;
  bus-emit rates exposed for debugging.
- **NFR-9 — Honesty/UX.** Every displayed signal shows its quality; degraded/rejected signals
  are visibly marked, never silently frozen without a "re-acquiring" indicator.

---

## 8. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Motion artifacts corrupt rPPG/HRV | High | High | First-class motion gate (FR-14); HRV rest-only; hold-last-good + re-acquire UX. |
| HRV over-claim (esp. RMSSD/LF-HF) | High | High (credibility) | Scope locked (ADR-05): SDNN committed, RMSSD indicative, LF/HF excluded. |
| Latency too high for biofeedback | Med | Med | Windowed DSP + drop-oldest; measure latency as a first-class metric (NFR-1); decide deep-rPPG live path later. |
| Licensing blocks future product | Med | High | Permissive-only product path (NFR-6); OpenFace/LibreFace quarantined (ADR-13); diligence items tracked. |
| GIL/CPU contention misses 30 fps | Med | Med | Threading + one backbone + decoupled extractors + MP escape hatch (ADR-19). |
| Time-sync error invalidates validation | Med | High | LSL timebase + cross-correlation residual alignment + XDF clock metadata (ADR-16). |
| Single-subject results over-generalized | High | Med | Mandatory feasibility framing in every report (ADR-18); skin-tone recorded for later. |
| Gaze/affect interpreted as precise | Med | Med | Zone-level gaze, "estimate+confidence" affect, SQI on everything (FR-13/14). |

---

## 9. Dependencies & assumptions

- **Hardware (rev 2 — buy nothing):** one modern laptop (M-series Mac or recent x86 + GPU
  optional), a 720p+ webcam, and a **Polar H10 if already owned** (cardiac ground truth — confirm
  availability). Lighting lux via a **free phone app**. No belt (→ respiration Indicative), no lux
  meter. Optional **Contec CMS50E** pulse-ox only if already on hand (PPG-waveform reference only).
- **Software:** Python **3.11** (not 3.13t free-threaded — ADR-19); MediaPipe, OpenCV,
  NumPy (**pinned `<2` to match MediaPipe**), SciPy, NeuroKit2, pyVHR (reference), HeartPy,
  HSEmotion/EmotiEffLib (ONNXRuntime), **LibreFace/OpenFace 3.0 (AUs, rev 2)**, pylsl, LabRecorder,
  pyxdf, python-osc, bleak, polar-python. rPPG-Toolbox for offline reference.
- **Dependency-coexistence risk (Spike-0):** MediaPipe (pins protobuf/`numpy<2`), ONNXRuntime,
  and the AU model (LibreFace/OpenFace 3.0, PyTorch) have a history of incompatible pins.
  Resolution: target `numpy<2`, CPU-only torch, prefer the AU model's **ONNX** export, else run it
  in an isolated subprocess/venv (rev 2 — runtime hygiene, no longer a licensing concern). Pinned
  matrix frozen by Spike-0 before feature work (TECH §2a).
- **Assumptions:** cooperative single adult, seated ~60 cm from camera; controlled→staged-
  naturalistic lighting; developer can self-administer the validation protocol.

---

## 10. Success criteria (Phase 0/1 done)

The phase is "done" — i.e., deliverable to the engineering team's satisfaction and ready for
Phase 2 planning — when **all** hold:
1. Phase 0 and Phase 1 exit criteria (§4) are met on the developer.
2. Every committed signal (§5) passes its pre-registered bar or is documented as flagged with
   a performance envelope.
3. A single command runs a validation session (synchronized XDF) and a single command emits a
   full validation report with the agreement-statistics set.
4. The bus demonstrably feeds both a logger (XDF) and a live OSC/WebSocket consumer.
5. *(rev 2: licensing/quarantine criterion withdrawn — productification out of scope.)*
6. Acceptance criteria in [04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md) all pass.
