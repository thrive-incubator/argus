# Argus — Project Brief

*A camera-only sensing layer for latent physiological and behavioral signals.*

**Owner:** Jean-Baptiste Passot
**Context:** Research at Thrive (mental health for children and families)
**Status:** Draft v0.1 — 2026-06-24
**Working name:** Argus (the hundred-eyed watcher — one camera, many signals)

---

## 1. One-paragraph summary

Argus is a real-time pipeline that takes a single ordinary camera feed (laptop
webcam or phone) and extracts a bundle of *latent* signals from the person in
frame — cardiac (heart rate, heart-rate variability), respiration, facial affect,
and ocular cues (blink, gaze, pupil) — plus higher-order state estimates derived
from them (arousal, valence, autonomic/stress index, attention). It emits these as
a low-latency stream that downstream consumers can subscribe to. The first consumer
is a **biofeedback / generative-art canvas** that changes character in response to
the signals; the longer-term consumer is **Thrive research** on the emotional and
physiological states of children and families. The sensing core is the same for
both; only the consumers differ.

---

## 2. Why this is interesting (the "why," not just the "what")

- **One sensor, many signals.** A camera is the most ubiquitous, least intrusive
  sensor a family already owns. No chest strap, no wristband, nothing to put on a
  child. If we can pull a meaningful fraction of what a clinic gets from contact
  sensors, the deployment surface is enormous.
- **Real-time changes the design.** Because the first goal is a *biofeedback tool*,
  not an offline analysis, the architecture is organized around a streaming **signal
  bus** with bounded latency. Accuracy still matters, but a slightly noisier signal
  delivered live is more useful here than a perfect signal delivered after the fact.
- **The art project and the research share a spine.** The responsive canvas is not a
  detour — it's the best possible *forcing function* for a clean, real-time,
  well-typed signal stream. If the canvas feels alive and responsive, the stream is
  good enough to do science with.

---

## 3. Guiding principles

1. **Signals are estimates; states are hypotheses.** Every output carries a
   confidence/quality value. We never display "anxiety" — we display "elevated
   arousal estimate, confidence 0.6." This is scientific honesty and, once children
   are involved, an ethical requirement.
2. **Quality-gate everything.** A signal computed from a badly lit, motion-blurred
   frame is worse than no signal. Each extractor publishes a Signal Quality Index
   (SQI); consumers decide how to handle low-quality spans.
3. **Validate against ground truth before believing anything.** Every signal is
   checked against a contact sensor on a known subject (you) before it's trusted.
4. **Privacy by construction.** Process on-device/edge where possible. Default to
   storing *derived signals*, not raw video of faces — especially non-negotiable
   for the future children's phase.
5. **Decouple the sensing core from its consumers.** The art canvas and the research
   logger are both just subscribers to the bus. Neither can reach into the pipeline.

---

## 4. Signal catalog

Grouped by mechanism, with a rough maturity/difficulty read. "You-first" means it's
realistic to stand up and validate on a single seated adult early.

### A. Optical / blood-volume (rPPG)
| Signal | Method | Difficulty | Notes |
|---|---|---|---|
| **Heart rate (HR)** | rPPG on forehead/cheek ROI (POS or CHROM classical; deep models for offline) | Low–Med | The anchor signal. Validate first. |
| **Heart-rate variability (HRV)** | Beat-to-beat intervals from a *clean* rPPG waveform | Med–High | Needs good waveform + robust peak detection. High research value (autonomic proxy). |
| **Respiration rate** | Respiratory sinus arrhythmia from rPPG, *and/or* chest/shoulder/head motion | Med | Two independent paths — fuse them. Pairs naturally with breathing interventions. |
| **SpO₂ (blood oxygen)** | Ratio of pulsatility across color channels | High | Optional / later. Hard to validate safely; skip for v1. |
| **Blood pressure** | rPPG morphology / pulse transit proxies | High (research-stage) | Explicitly out of scope for now. |
| **Flushing / blushing** | Slow color shifts in cheeks | Med | A genuine affect/arousal cue; cheap once rPPG ROI exists. |

### B. Ocular / behavioral
| Signal | Method | Difficulty | Notes |
|---|---|---|---|
| **Blink rate & duration** | Eye Aspect Ratio (EAR) threshold over frames | Low | Fatigue/arousal/cognitive-load marker. |
| **PERCLOS** | % time eyes closed | Low | Classic drowsiness/regulation signal. |
| **Gaze direction / fixations / saccades** | Iris landmarks (MediaPipe iris) | Med | Attention, gaze aversion (socially meaningful for kids work). |
| **Pupil size (pupillometry)** | Iris diameter in pixels, normalized | High (resolution-hungry) | Cognitive-load proxy; treat as best-effort. |

### C. Facial affect / structural
| Signal | Method | Difficulty | Notes |
|---|---|---|---|
| **Facial action units (FACS)** | OpenFace 2.0 AU intensities | Med | The rigorous, interpretable representation. |
| **Blendshapes (52)** | MediaPipe Face Landmarker | Low | Fast, real-time; good for the live canvas. |
| **Valence / arousal estimate** | Mapped from AUs/blendshapes | Med | The affect summary the canvas and research both want. |
| **Expressiveness / micro-expressions** | AU dynamics over time | High | Promising but preliminary; flag confidence low. |

### D. Movement / motor (the "anything else" you asked for)
| Signal | Method | Difficulty | Notes |
|---|---|---|---|
| **Head motion / nodding** | Landmark/pose tracking | Low | Doubles as a respiration and a fidget cue. |
| **Posture & postural shifts** | Pose estimation (BlazePose) | Low–Med | Slumping, leaning, withdrawal. |
| **Fidgeting / motor restlessness** | Aggregate optical flow / pose jitter | Med | Relevant to attention/regulation; also the #1 *noise* source — measuring it helps gate other signals. |
| **Self-soothing / hand-to-face** | Pose + proximity heuristics | Med | Behaviorally meaningful in kids/family contexts. |

### E. Fused / higher-order states (derived, lowest confidence)
- **Autonomic / stress index** — driven mainly by HRV (e.g., LF/HF balance) + respiration.
- **Arousal** — fused from HR, respiration, pupil, motion.
- **Valence** — fused from facial affect.
- **Attention / engagement** — fused from gaze stability, blink, head pose.
- **(Future) Physiological synchrony** — correlation of two people's signals in a
  parent–child dyad. A distinctive Thrive angle; needs the dyadic phase.

### F. Adjacent (explicitly noted, not committed)
- **Voice / prosody** (pitch, rate, jitter) if a mic is permitted — not camera-only,
  so out of scope unless we consciously widen the definition. Worth a flag because
  it's cheap and high-value for affect.

---

## 5. Architecture

```
 ┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────┐
 │  CAPTURE    │──▶│  FACE / POSE      │──▶│  SIGNAL EXTRACTORS   │──▶│  FUSION +    │
 │ webcam/phone│   │  BACKBONE         │   │  (rPPG, EAR, gaze,   │   │  STATE       │
 │ 30fps RGB   │   │  MediaPipe mesh,  │   │   AUs, motion ...)   │   │  ESTIMATION  │
 │ + timestamp │   │  iris, pose       │   │  each emits value+SQI│   │  + confidence│
 └─────────────┘   └──────────────────┘   └─────────────────────┘   └──────┬───────┘
        │                                                                    │
        │                   ┌──────────────────────────────────────────────┘
        ▼                   ▼
 ┌──────────────┐   ┌──────────────────────────────────────────────────────────────┐
 │ QUALITY /    │   │                       SIGNAL BUS                              │
 │ COVARIATE    │──▶│      (WebSocket / OSC / ZeroMQ — fixed-rate, low-latency)     │
 │ skin tone,   │   └───────────────┬───────────────────────┬──────────────────────┘
 │ lighting,    │                   ▼                       ▼
 │ motion, SQI  │          ┌─────────────────┐     ┌──────────────────┐
 └──────────────┘          │  ART CANVAS     │     │  RESEARCH LOGGER  │
                           │  (live consumer)│     │  + LIVE DASHBOARD │
                           └─────────────────┘     │  + VALIDATION     │
                                                    └──────────────────┘
```

**Why this shape:**
- A **shared face/pose backbone** runs once per frame and hands ROIs/landmarks to
  every extractor — no redundant detection.
- Every extractor outputs **`{value, sqi, timestamp}`**, never a bare number.
- The **quality/covariate layer** (skin-tone estimate, lighting, global motion) runs
  alongside and is published *too*, so consumers and the validation harness can
  condition on conditions. This is how we'll catch skin-tone and lighting bias
  honestly rather than hiding it.
- The **bus** is the contract. The canvas and the logger never import pipeline code.

---

## 6. Recommended stack

- **Language:** Python (real-time prototype), OpenCV for capture/IO.
- **Backbone:** MediaPipe Face Landmarker (mesh + blendshapes + iris) and Pose.
- **rPPG (real-time):** classical **POS** / **CHROM** — they run live and are robust.
- **rPPG (offline validation / upgrade path):** **rPPG-Toolbox** (PhysNet, TS-CAN,
  EfficientPhys) for accuracy benchmarking against the classical real-time path.
  *(Open: confirm whether a DL model can hit our latency budget on your hardware, or
  stays offline-only as a quality reference.)*
- **Facial AUs:** OpenFace 2.0 for rigorous FACS; MediaPipe blendshapes for the
  low-latency live path.
- **Ground truth:** **Polar H10** chest strap (HR + R-R intervals for HRV truth), a
  fingertip **pulse oximeter**, and optionally a **respiration belt**.
- **Bus / canvas:** WebSocket or OSC (OSC is friendly to creative-coding canvases —
  TouchDesigner, openFrameworks, p5.js, Max).

---

## 7. Phasing

### Phase 0 — Single-subject dev rig (you)  → *prove the spine*
- Capture + MediaPipe backbone + **HR via POS**, streamed live over the bus, with a
  bare dashboard.
- Wear the Polar H10; show real-time HR agreement (Bland–Altman, MAE).
- **Exit criterion:** live HR on yourself within a few bpm of the chest strap under
  good lighting, end-to-end latency under ~1–2 s.

### Phase 1 — Full signal set + quality gating (you, controlled)
- Add HRV, respiration (both paths), blink/EAR/PERCLOS, gaze, blendshape affect,
  head/posture motion.
- Stand up the quality/covariate layer and per-signal SQI.
- Build the research logger (timestamped raw + derived) and a real richer dashboard.
- **Exit criterion:** every signal has a validated-or-flagged status and an SQI.

### Phase 2 — Break it toward naturalistic (you + consenting adults)
- Deliberately vary lighting, distance, movement; recruit a few **skin-tone-diverse
  consenting adults** to test bias early (do *not* wait for the kids phase to
  discover this).
- Add fusion/state estimates with explicit confidence.
- **Exit criterion:** documented performance envelope — where each signal holds up
  and where it degrades.

### Phase 3 — Dyads + the art canvas (parallel track)
- Two-person capture; first pass at physiological synchrony.
- The art canvas consumes the live bus as a real exhibit/biofeedback piece.

### Phase 4 — Thrive children & families  → *the ethics-serious phase*
- **Gated on IRB approval.** Parental consent + child assent, COPPA-aware data
  handling, edge processing, derived-signal-only storage, pre-registered bias
  validation across skin tones, and clinical/wellness disclaimers (this is **not** a
  medical device).
- Nothing in Phases 0–3 should architecturally block these requirements — design for
  them now (privacy-by-construction principle).

---

## 8. Risks & honest limitations

- **Motion artifacts** are the dominant error source — and children move constantly.
  Mitigation: measure motion explicitly and use it to gate, not just suffer it.
- **Lighting** sensitivity, especially naturalistic. Mitigation: covariate layer +
  staged testing.
- **Skin-tone bias** is the most serious equity risk: lower reflectance at higher
  melanin degrades rPPG. Mitigation: skin-tone-diverse testing from Phase 2, report
  performance *per group*, never as a single headline number.
- **Real-time vs. accuracy tradeoff:** the live path (POS/CHROM, blendshapes) is
  faster but noisier than the offline DL/AU path. We run both and treat the offline
  path as the quality reference.
- **Over-interpretation** is the research-integrity risk: a camera does not measure
  emotion. Mitigation: the "states are hypotheses" principle, baked into the data
  model as confidence values.
- **Privacy:** faces of families (and children) are maximally sensitive. Mitigation:
  derived-signal-only storage and edge processing as defaults.

---

## 9. Open questions to resolve next

1. **Latency budget:** what end-to-end delay still feels "alive" for the canvas? (Sets
   whether DL rPPG is in or out of the live path.)
2. **Hardware target:** laptop webcam only, or phone too? (Phone unlocks naturalistic
   home use but changes the engineering.)
3. **Bus protocol:** OSC (creative-tooling friendly) vs WebSocket (web-canvas
   friendly) — driven by what you want to build the canvas *in*.
4. **Voice/prosody:** keep strictly camera-only, or allow an optional mic channel for
   the affect signals? (Cheap, high-value, but breaks the "camera-only" purity.)
5. **The art canvas medium:** TouchDesigner / openFrameworks / p5.js / Max? Decides
   the consumer-side stack.

---

## 10. What I'd build first (recommendation)

Phase 0, this week, on you: **webcam → MediaPipe → POS heart rate → WebSocket → a
one-number live dashboard, validated against a Polar H10.** It's small, it's
end-to-end, and it proves the single hardest claim (a real physiological signal,
live, from a camera, that agrees with ground truth). Everything else hangs off that
spine.
