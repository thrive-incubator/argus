# Argus — Feature List & Acceptance Criteria

*Engineer-ready backlog for Phase 0 & Phase 1, with testable acceptance criteria.*

**Status:** Approved for implementation · **Date:** 2026-06-24
**Companion docs:** [01_ARCHITECTURE_DECISIONS.md](01_ARCHITECTURE_DECISIONS.md) · [02_PRD.md](02_PRD.md) · [03_TECH_DESIGN.md](03_TECH_DESIGN.md)

**How to read this:** features are grouped into epics. Each feature has an ID, the FR/ADR it
satisfies, a phase tag, and **acceptance criteria** in Given/When/Then form. A feature is
"done" only when every AC passes *and* its mapped test (TECH_DESIGN §14) is green. Acceptance
bars for signals are **single-subject feasibility** bars (PRD §5), not population accuracy.

**Definition of Done (applies to every feature):** code + unit/integration tests + docstring
+ config knobs exposed (no magic numbers) + no regression to Phase 0 exit criteria. *(rev 2:
the licensing CI gate is removed.)*

---

## Epic A — Capture & Backbone  *(Phase 0)*

### A1. Timestamped webcam capture — `FR-1`, `ADR-16`
- **AC1.** Given a webcam, when capture runs, then frames are delivered at ≥ 30 fps
  (effective, measured) at ≥ 720p with `CAP_PROP_BUFFERSIZE=1`.
- **AC2.** Each frame carries a `ts` from `pylsl.local_clock()` stamped at grab; timestamps
  are strictly monotonic.
- **AC3.** Given a downstream consumer that stalls, when new frames arrive, then the
  latest-frame slot drops older frames (no unbounded queue growth, no blocking of capture).
- **AC4.** The exposure→host latency calibration (LED-flash) produces a constant offset that
  is recorded in config and subtracted from frame `ts`.

### A2. Face backbone (MediaPipe FaceLandmarker) — `FR-2`, `ADR-01`
- **AC1.** Exactly one FaceLandmarker pass runs per frame, producing 478 landmarks, 10 iris
  points, 52 blendshapes, and a 4×4 head-pose matrix in a `FrameContext`.
- **AC2.** `num_faces=1`; temporal smoothing enabled; monotonic timestamps fed to the Task.
- **AC3.** When no face is present, `FrameContext.face is None` and downstream extractors
  emit nothing (no crash, no stale value without a flag).

### A3. Pose backbone (BlazePose, separate Task + One-Euro) — `FR-3`, `ADR-02`
- **AC1.** A separate PoseLandmarker Task runs per frame (Holistic is **not** used).
- **AC2.** Landmarks are One-Euro filtered before any motion feature; raw vs filtered jitter
  is measurably reduced on a static-subject clip.
- **AC3.** Per-landmark visibility/presence is exposed so occluded lower-body joints are
  handled gracefully.

### A4. Real-time throughput & frame-drop policy — `FR-4`, `NFR-2`, `ADR-19`
- **AC1.** On the target laptop CPU, the backbone sustains ≥ 30 fps for ≥ 10 min without
  unbounded memory growth.
- **AC2.** Under induced CPU load, per-frame extractors drop frames (latest wins) **but** the
  lossless RGB-trace ring buffer feeding rPPG/HRV/respiration loses **zero** samples.
- **AC3.** Dropped-frame counters and per-extractor timing are exposed (NFR-8).

---

## Epic B — Bus, Logging & Dashboard  *(Phase 0 → Phase 1)*

### B1. LSL outlets — `FR-16`, `ADR-15`  *(P0: HR; P1: all)*
- **AC1.** Each signal/covariate is published on its own LSL `StreamOutlet`; regular-rate
  streams declare `nominal_srate`, event streams use `IRREGULAR_RATE`.
- **AC2.** Stream XML carries units, method, and window metadata; channel layout includes
  `sqi` and `gate_code`.
- **AC3.** A second process can `resolve_stream()` and read every Argus stream.

### B2. Synchronized XDF recording — `FR-17`, `ADR-16`  *(P0)*
- **AC1.** A single command starts/stops a labeled recording capturing **all** Argus streams
  + the Polar H10 stream into one XDF.
- **AC2.** The XDF loads via `pyxdf.load_xdf(proc_clocksync=ON, proc_dejitter=ON)` and
  contains clock-offset metadata for each stream.
- **AC3.** Replaying the XDF reproduces per-signal sample counts within rounding of live rates.

### B3. Live dashboard — `FR-19`, `NFR-9`  *(P0: HR; P1: full)*
- **AC1.** Phase 0: dashboard shows live HR with its SQI and gate state, updating ≥ 1 Hz.
- **AC2.** Phase 1: every signal renders value + SQI + traffic-light (🟢/🟡/🔴).
- **AC3.** A REJECT/degraded signal is visibly marked and shows "re-acquiring" — never a
  silently frozen value.

### B4. Art/biofeedback bridge (LSL→OSC / LSL→WebSocket) — `FR-18`, `ADR-15`  *(P1)*
- **AC1.** Selected streams are re-emitted as OSC addresses (e.g. `/argus/hr`,
  `/argus/valence`) and/or WebSocket JSON.
- **AC2.** A trivial test consumer (e.g. a printing OSC server) receives values at the
  expected rate with the configured forward-sync look-ahead.
- **AC3.** The bridge adds no back-pressure to the pipeline if the consumer disconnects.

---

## Epic C — Ground Truth & Time Sync  *(Phase 0)*

### C1. Polar H10 ingestion — `FR-20`, `ADR-17`
- **AC1.** Connects over BLE by **device name** (works on macOS where only a UUID is exposed);
  reconnects automatically after a dropout.
- **AC2.** Parses `0x2A37` with the **full flag-driven offset algorithm** (bit0 16-bit HR,
  bit3 energy-expended, bit4 RR — ADR-17); extracts **all** RR fields per packet in 1/1024 s
  units; converts to ms correctly, **verified against test vectors for all 4 relevant flag
  combinations** (not just one capture).
- **AC3.** Beat times are reconstructed by cumulative sum from a single anchored
  `local_clock()` (not packet-arrival), and published to LSL as HR + beat-times.
- **AC4.** RR artifacts are corrected with NeuroKit2 Kubios before any HRV statistic.

### C2. Offline cross-correlation alignment — `FR-22`, `ADR-16`
- **AC1.** Given a recorded session, instantaneous-HR series from camera and H10 are resampled
  to 4 Hz and cross-correlated; the recovered residual lag is reported.
- **AC2.** Camera beats are matched to H10 beats within ±50–100 ms for IBI-level HRV
  comparison; unmatched beats are counted (yield reported).
- **AC3.** DTW, if used, appears only as a reported agreement metric — never as the aligner
  feeding the validation stats.

---

## Epic D — Cardiac (rPPG)  *(Phase 0: HR · Phase 1: HRV)*

### D1. rPPG ROI extraction — `ADR-01`  *(P0)*
- **AC1.** ROI is a multi-patch mean (forehead + cheeks) excluding eyes/mouth/brows.
- **AC2.** Yaw beyond the configured cutoff drops the occluded cheek patch; ROI remains valid.
- **AC3.** Landmark jitter does not produce ROI-mean spikes beyond a tolerance on a
  static-subject clip (patch averaging works).

### D2. HR via POS — `FR-5`, `ADR-03`  *(P0)*
- **AC1.** POS runs on a rolling 8–15 s window, updating ~1 Hz, output in [42–240 bpm].
- **AC2.** POS output matches the pyVHR reference on a fixed golden clip within tolerance.
- **AC3.** **Validation bar (Phase 0 exit):** at rest, **≥300 lux**, HR vs Polar H10 meets the
  **EC13 *numeric threshold*** (MAE ≤ max(5 bpm, 10%)) and **MAPE < 10%** (feasibility, not
  conformance), with **capture→emit latency ≤ 2 s** (per NFR-1; excludes intrinsic ~½-window
  smoothing).
- **AC4.** Each HR record carries De Haan SNR as `sqi` and a gate state; **in Phase 0 the gate
  is the always-`"unknown"` stub** (the real gate arrives in build-step 3).

### D3. HRV (SDNN committed, RMSSD indicative) — `FR-6`, `ADR-05`, `ADR-06`  *(P1)*
- **AC1.** BVP is cubic-spline upsampled to ≥ 256 Hz before peak detection.
- **AC2.** Per-beat Orphanidou bSQI gates beats at a **threshold calibrated against
  H10-confirmed beats** (ROC) — not assumed at 0.86; low-SQI beats are dropped before IBI.
- **AC3.** SDNN and RMSSD are computed via NeuroKit2 over a **fixed-length, length-matched**
  window when **≥ 80% of the window's beats are GOOD-gated and bSQI-accepted** (not 100%
  continuous GOOD), flagged `rest_only`. Zero qualifying windows → report "insufficient clean
  data," not a silent pass.
- **AC4.** **Validation bar (rev 1):** at rest, SDNN vs H10 meets **MAE ≤ 12 ms** with BA
  bias+95% LoA and **Lin's CCC** reported (bias-only bar dropped); **ln-RMSSD** is reported as a
  BA in **log-units/ratio + CI** (NOT ms, no ±15 ms band) as *indicative*. The ±15 ms band, if
  shown, applies to SDNN only. Reported HRV bias includes the H10's own seated bias.
- **AC5.** LF/HF and frequency-domain HRV are **not produced** anywhere in the system.

---

## Epic E — Respiration  *(Phase 1)*

### E1. Respiration from chest/shoulder motion — `FR-7`, `ADR-07`
- **AC1.** RR is computed from band-passed (**0.08**–0.5 Hz, rev 1) chest/shoulder displacement
  over a 15–30 s window via FFT/peak-count.
- **AC2.** rPPG-derived RR is computed as a secondary cross-check and its agreement reported;
  the motion estimate is the primary output.
- **AC3.** **Respiration is *Indicative* (rev 2 — no belt):** no accuracy/MAE bar. Against the
  metronome (6/10/15 brpm) the estimate is reported as a **plausibility check** only (does it
  track the commanded direction of change). No pass/fail.
- **AC4.** RR carries an SQI from band-power ratio + pose visibility and is motion-gated.

---

## Epic F — Ocular  *(Phase 1)*

### F1. Blink & PERCLOS (adaptive EAR) — `FR-8`, `ADR-08`
- **AC1.** Open-eye baseline auto-calibrates per session; threshold is personalized.
- **AC2.** A blink = EAR below threshold for ≥ 2–3 frames (~250 ms at 30 fps); blink events
  (with duration) and a windowed blink-rate are emitted.
- **AC3.** **Validation bar:** blink-detection F1 ≥ 0.90 vs **frame-level manual annotation
  with a defined event-matching tolerance (±N frames)** at 30 fps, **≥300 lux, no glasses**
  (eyewear is a recorded covariate); κ on a re-annotated subset where feasible.
- **AC4.** PERCLOS (P80) over 60–90 s **tracks a commanded closure-fraction manipulation**
  (scripted 0/25/50/75% eye-closure blocks) within tolerance; **no drowsiness ground truth is
  claimed** (Indicative).
- **AC5.** The system asserts/warns if effective fps < 25 (blink metrics invalid). **Blink
  metrics are withheld until the open-eye baseline is armed** with enough valid frames.

### F2. Gaze zones — `FR-9`, `ADR-09`
- **AC1.** Gaze vector comes from a learned head (L2CS-Net/MobileGaze, ONNX/CPU); MediaPipe
  iris provides the front-end only.
- **AC2.** Output is **zones** {left/center/right, screen on/off, attention present/absent}
  with confidence — **no pixel coordinates are emitted** for uncalibrated users.
- **AC3.** **Validation deliverable (rev 1):** against a scripted look-target protocol **with
  fixed target eccentricities and zone half-widths**, report a **3×3 confusion matrix and
  accuracy above the 33% three-way chance** (not a bare ≥80% number). Indicative, not gated.
- **AC4.** Optional 5–9-point calibration writes a per-user correction and is shown to improve
  zone accuracy (Phase-1 upgrade, may be deferred without failing the epic).

### F3. Pupillometry — **explicitly not implemented** — `ADR-10`
- **AC1.** No pupillometry/pupil-size feature exists in Phase 0/1; the decision and rationale
  are documented; an arousal proxy (blink rate / gaze dispersion / HRV) is used instead.

---

## Epic G — Facial Affect  *(Phase 1)*

### G1. Live affect (blendshapes + HSEmotion) — `FR-10`, `ADR-11`
- **AC1.** Blendshapes are neutral-subtracted + z-scored per session before use.
- **AC2.** HSEmotion (`enet_b0_8_va_mtl`) emits 8-class emotion + continuous valence/arousal
  at **`[tunable: 10–15 Hz]`** (rev 1, not a hard 30 fps), each with confidence as `sqi`.
- **AC3.** Outputs are labeled "estimate, confidence X" in the bus metadata and dashboard —
  never presented as a verdict.
- **AC4.** **Descriptive deliverable, not pass/fail (rev 1):** a report showing the model's V/A
  output covaries with **deliberately posed** configurations (e.g. posed-happy V > posed-sad V),
  with effect sizes and an emotion confusion matrix. This is a **model face-validity check on
  this face — NOT validation of the subject's felt affect** (posed ≠ felt). Flagged exploratory.

### G2. Research AUs (LibreFace / OpenFace 3.0, decoupled) — `FR-11`, `ADR-12` *(rev 2)*
- **AC1.** **LibreFace** (or OpenFace 3.0) runs on a decoupled worker at 5–15 fps (or offline on
  logged frames) and does not reduce the live path below 30 fps.
- **AC2.** AU streams emit **0–5 AU intensities**, flagged `research`, with method/units metadata;
  descriptive, no pass/fail.
- **AC3.** License is **not** a constraint (rev 2). Prefer the **ONNX** build in the main venv;
  if using the PyTorch build, isolate it via the §4 escape hatch (runtime hygiene, not licensing).

### G3. ~~Quarantined AU gold reference~~ — **WITHDRAWN (rev 2)**
- The quarantine/firewall is removed (productification out of scope). LibreFace/OpenFace 3.0 are
  the primary AU extractor (G2), used directly. No `argus/research/`, no flag, no import-lint.

---

## Epic H — Motion, Quality & Covariates  *(Phase 1; gate scaffolding in Phase 0)*

### H1. SQI computation — `FR-13`, `ADR-06`
- **AC1.** De Haan spectral SNR is computed per rPPG window and normalized to 0..1 as `sqi`.
- **AC2.** Skewness/perfusion supplements and Orphanidou bSQI (for HRV beats) are implemented.
- **AC3.** Low-SQI records are emitted **with a flag** (HRV beats excepted — dropped).

### H2. Motion quality gate (3-tier traffic light) — `FR-14`, `ADR-14`
- **AC1.** Gate inputs (FM_X/FM_Y, FSM, solvePnP head pose with pitch×2, pulse SNR/skewness)
  are computed from existing FaceMesh data (no second tracker).
- **AC2.** Motion metrics are normalized by inter-ocular distance.
- **AC3.** GOOD/USABLE/REJECT transitions use hysteresis (dwell ≥ 1 s) — **no state flicker**
  on a borderline-motion clip.
- **AC4.** **Behavioral bar:** under induced head motion, the gate reaches REJECT, HRV/RR are
  suppressed, HR holds last-good, and the dashboard shows "re-acquiring"; on return to
  stillness, GOOD resumes. **HRV uses the ≥80%-GOOD-window policy** (D3.AC3), not 100% GOOD.
- **AC5.** Gate state rides on every `SignalRecord.gate` and is published as its own stream;
  **Phase 0 emits the `"unknown"` stub** until the gate exists (build-step 3).

### H3. Covariate layer — `FR-15`, `FR-23`, `ADR-18`
- **AC1.** Skin-tone (self-reported Monk/Fitzpatrick + **unvalidated** cheek-reflectance
  estimate), **eyewear, facial hair**, lighting (**relative brightness index — uncalibrated, NOT
  lux — plus a measured lux value per block**, with over/under-exposure flags), global motion,
  and face presence are each published as streams.
- **AC2.** Skin tone, eyewear, and facial hair are recorded once per session and stored with the
  XDF. **Skin tone is forward-compat for Phase 2 only — it provides no fairness info at n=1.**

---

## Epic I — Validation Harness  *(Phase 1)*

### I1. Protocol runner — `FR-21`, `ADR-18`
- **AC1.** Scripted blocks (rest ×2, paced-breathing **belt-referenced**, light-motion, lighting
  A≈150 / B≈500 lux, optional gaze-target & eye-closure, ≥2 repeats) present on-screen prompts
  and inject markers into the XDF.
- **AC2.** **Measured lux** is entered per lighting block; **HRV stats from `rest` blocks only,
  and the 6-brpm block is excluded from HRV** (0.1 Hz resonance inflates it); **respiration
  stats from the paced blocks vs the belt**.

### I2. Validation report generator — `FR-22`, `ADR-18`
- **AC1.** Per condition & signal, the report includes Bland-Altman (bias + 95% LoA), MAE,
  RMSE, MAPE, Pearson r, **Lin's CCC**, and SNR. **The H10 is labeled a *reference* (its own
  seated bias is included in reported bias, not attributed to the camera).**
- **AC2.** HR is reported at both a 60 s average and a 4–10 s quasi-instantaneous window.
- **AC3.** HRV (rev 1): **SDNN — MAE ≤ 12 ms bar + BA bias+LoA + Lin's CCC over length-matched
  windows** (the ±15 ms band, if shown, is SDNN-only); **ln-RMSSD — BA in log-units/ratio, NOT
  ms, indicative**. HR pass/fail vs the **EC13 + CTA-2065 *numeric thresholds*** (feasibility,
  not conformance).
- **AC4.** Every report renders the **feasibility banner** stating single-subject results are
  hypothesis-generating and establish neither limits of agreement nor fairness, **with the
  Fitzpatrick best/worst-case caveat**.
- **AC5.** One command produces the full HTML/PDF report from an XDF.

---

## Epic J — Cross-cutting: Concurrency, Licensing, Reproducibility

### J1. Concurrency model — `ADR-19`, `NFR-2`
- **AC1.** Architecture is threading-based with one capture thread, one backbone thread, N
  extractor threads, and an asyncio I/O edge; no asyncio on the CV hot path.
- **AC2.** A documented multiprocessing+`shared_memory` escape hatch exists and is used for
  **Py-Feat (isolated venv, pulls PyTorch)** and any extractor that cannot meet cadence; frames
  are passed as handles, never pickled.
- **AC3.** The system runs on **CPython 3.11** (not 3.13t); `numpy<2`; CPU-only torch (§2a).

### J2. ~~Licensing firewall~~ — **WITHDRAWN (rev 2)**
- Productification is out of scope; there is no license constraint, no quarantine, and no
  import-lint. Use the best model regardless of license. *(Privacy "no cloud" at runtime and the
  raw-video storage rules in J4 still hold — those are not licensing.)*

### J3. Reproducibility & one-command flows — `NFR-4`
- **AC1.** Dependencies are pinned; the POS path is deterministic on a fixed clip.
- **AC2.** `argus record --session NAME` runs a full validation session to a synchronized XDF.
- **AC3.** `argus report --xdf FILE` produces the validation report.
- **AC4.** `argus run` launches the live pipeline + dashboard + bus.

### J4. Privacy by construction — `NFR-5`
- **AC1.** Default storage is derived signals + XDF; **no raw face video** is written unless an
  explicit off-by-default flag is set.
- **AC2.** When raw-video opt-in is enabled (for blink-annotation GT), it is stored locally
  only and labeled.

---

## Phase gates (the two checkpoints)

**Phase 0 gate — "the spine":** A1–A4, B1–B3, C1, D1–D2 complete; **D2.AC3 met** (HR within
EC13, latency ≤ 2 s, synchronized XDF + first report). Nothing in Phase 1 starts until this
holds.

**Phase 1 gate — "engineer-ready sensing layer":** all epics complete; every **committed** signal
(**D2 HR, D3 SDNN, F1 blink** — respiration E1 is now Indicative, rev 2) passes its validation bar
or is documented as flagged with an envelope; H2 motion-gate behavioral bar met; I2 produces the
full report. This is the deliverable handed to the team for Phase 2 planning (PRD §10).

---

## Traceability matrix (feature → FR → ADR → test)

| Feature | FR | ADR | Test (TECH_DESIGN §14) |
|---|---|---|---|
| A1 | FR-1 | ADR-16 | Capture |
| A2 | FR-2 | ADR-01 | Backbone |
| A3 | FR-3 | ADR-02 | Backbone |
| A4 | FR-4 | ADR-19 | Backbone/throughput |
| B1 | FR-16 | ADR-15 | Bus |
| B2 | FR-17 | ADR-16 | Sync/Bus |
| B3 | FR-19 | — | Integration |
| B4 | FR-18 | ADR-15 | Bus |
| C1 | FR-20 | ADR-17 | Sync |
| C2 | FR-22 | ADR-16 | Sync |
| D1 | — | ADR-01 | rPPG |
| D2 | FR-5 | ADR-03 | rPPG/POS |
| D3 | FR-6 | ADR-05/06 | HRV |
| E1 | FR-7 | ADR-07 | Resp |
| F1 | FR-8 | ADR-08 | Blink |
| F2 | FR-9 | ADR-09 | Gaze |
| F3 | — | ADR-10 | (negative) |
| G1 | FR-10 | ADR-11 | Integration |
| G2 | FR-11 | ADR-12 | Integration |
| G3 | — | ADR-13 | Licensing |
| H1 | FR-13 | ADR-06 | SQI |
| H2 | FR-14 | ADR-14 | Motion gate |
| H3 | FR-15/23 | ADR-18 | Integration |
| I1 | FR-21 | ADR-18 | Validation session |
| I2 | FR-22 | ADR-18 | End-to-end/report |
| J1 | — | ADR-19 | Backbone/throughput |
| J2 | — | ADR-12/13 | Licensing |
| J3 | — | — | End-to-end |
| J4 | — | — | Integration |
