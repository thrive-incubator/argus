# Argus — Architecture Decision Record (ADR)

*State-of-the-art review and locked technical decisions for Phase 0 & Phase 1.*

**Status:** Approved for implementation · **Date:** 2026-06-24 · **Scope:** Phase 0 & 1 only
**Companion docs:** [02_PRD.md](02_PRD.md) · [03_TECH_DESIGN.md](03_TECH_DESIGN.md) · [04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md)

This document records *every* architecture decision, the 2023–2026 SOTA evidence behind
it, and what changed versus the original project brief. Each decision is an ADR with a
status of **KEEP** (brief was right), **CHANGE** (brief was wrong/outdated), or **ADD**
(missing from the brief). Citations are inline; a consolidated source list is at the end.

> **Reading guide for engineers:** decisions are normative. Where a decision says
> "live path," it must run inside the per-frame real-time loop; "offline/research path"
> may run post-hoc or on a decoupled worker. Every signal emits `{value, sqi, timestamp}`.

> **Rev 2 amendment (authoritative — supersedes the rows/notes below where they conflict):**
> (a) **No hardware purchases** → **ADR-07 respiration is Indicative** (motion-only, no belt).
> (b) **Productification/licensing dropped** → **ADR-12** research AUs use **LibreFace (or
> OpenFace 3.0)** directly; **ADR-13 quarantine and the licensing firewall are removed**; the
> "commercial-clean" constraint and the licensing diligence items no longer apply.
> (c) **Canvas medium = TouchDesigner over OSC** (open item 5 resolved); p5.js/WebSocket only if
> a web-shareable piece is wanted.

---

## Decision summary table

| # | Area | Decision | Status vs brief |
|---|---|---|---|
| ADR-01 | Real-time face/landmark backbone | MediaPipe Face Landmarker v2 (Tasks API) | **KEEP** |
| ADR-02 | Real-time pose/motion backbone | MediaPipe Pose Landmarker (BlazePose), separate Task + One-Euro filter | **KEEP** |
| ADR-03 | Live rPPG (cardiac) | Classical **POS** (CHROM fallback), windowed CPU | **KEEP** |
| ADR-04 | Offline rPPG reference | rPPG-Toolbox with **RhythmFormer + RhythmMamba** (PhysNet/TS-CAN as baselines) | **CHANGE** (model set modernized) |
| ADR-05 | HRV scope | HR + **SDNN** committed; RMSSD indicative; **LF/HF not promised** | **CHANGE** (scope tightened) |
| ADR-06 | Signal Quality Index | De Haan spectral SNR (gate) + Orphanidou bSQI (HRV beat acceptance) | **ADD** |
| ADR-07 | Respiration | Chest/shoulder motion (MediaPipe Pose); **Indicative — no belt (rev 2)** | **CHANGE** |
| ADR-08 | Blink / PERCLOS | EAR on FaceMesh + **per-session adaptive threshold** + PERCLOS window | **KEEP+** |
| ADR-09 | Gaze | MediaPipe iris front-end + **learned head (L2CS-Net/MobileGaze)**; output **zones not pixels** | **CHANGE** (add learned head) |
| ADR-10 | Pupillometry | **CUT** from Phase 0/1 (category error + RGB-infeasible) | **CHANGE** (removed) |
| ADR-11 | Facial affect — live | MediaPipe 52 blendshapes + **HSEmotion** (emotion + valence/arousal) | **CHANGE** (add HSEmotion) |
| ADR-12 | Facial affect — research AUs | **LibreFace** (or OpenFace 3.0) — best intensity; license irrelevant (rev 2) | **CHANGE** |
| ADR-13 | ~~AU quarantine~~ | **WITHDRAWN (rev 2)** — no licensing firewall | — |
| ADR-14 | Motion quality gate | First-class 3-tier hysteretic traffic light (landmark motion + SNR) | **ADD** |
| ADR-15 | Streaming bus | **Dual: LSL→XDF (research) + LSL→OSC/WebSocket bridge (art/biofeedback)** | **CHANGE** (upgrade from "WebSocket or OSC") |
| ADR-16 | Time synchronization | LSL monotonic timebase; frame stamp at grab; Polar cumsum-RR; cross-corr residual align | **ADD** |
| ADR-17 | Ground-truth integration | Polar H10 via `bleak` + `polar-python`; CMS50E pulse-ox optional | **KEEP+** |
| ADR-18 | Validation methodology | BA + MAE/RMSE/MAPE + **Lin's CCC** + pre-registered EC13/CTA-2065; feasibility framing | **CHANGE** (upgrade) |
| ADR-19 | Concurrency model | Threading default; one backbone → decoupled extractor plugins; windowed DSP; MP escape hatch | **ADD** |
| ADR-20 | Signal-processing libs | NeuroKit2 (HRV/resp/quality) + pyVHR (camera→BVP ref) + HeartPy (cross-check) | **ADD** |

---

## ADR-01 — Real-time face/landmark backbone: **MediaPipe Face Landmarker v2** (KEEP)

**Decision.** Use MediaPipe Face Landmarker v2 (`face_landmarker_v2`) in VIDEO/LIVE_STREAM
mode as the single per-frame face backbone. One inference pass yields: 478 mesh landmarks
(incl. 10 iris points), 52 ARKit blendshapes, and a 4×4 facial-transform (head-pose)
matrix. It feeds **every** downstream face extractor — rPPG ROI, blink/EAR, gaze
front-end, blendshape affect, and the landmark-motion gate.

**Why (SOTA evidence).** No 2023–2026 alternative provides mesh + iris + blendshapes +
head-pose in one real-time Python pass under a permissive license. Runs ~30 fps on laptop
CPU. License **Apache-2.0** (clean for a future product). Alternatives evaluated and
rejected: Microsoft "Dense Landmarks" (ECCV'22, no code/weights released);
DECA/EMOCA/SMIRK (not real-time, FLAME non-commercial); 3DDFA_V2 (MIT, 1.35 ms CPU but no
iris/blendshapes, unmaintained since 2020 — kept only as a self-hosted dense-mesh
fallback). There is **no Google "v3"**; v2 is current (v0.10.x, maintained into 2026).

**Caveats to engineer around.** Blendshapes are coarse; z-depth is least reliable;
temporal smoothing only applies when `num_faces=1`; frame jitter must be filtered before
defining a stable rPPG ROI (absorb via patch averaging — see TECH_DESIGN §rPPG).

---

## ADR-02 — Real-time pose backbone: **MediaPipe Pose (BlazePose)** (KEEP)

**Decision.** Use MediaPipe Pose Landmarker (BlazePose) as a **separate Task** (NOT
MediaPipe Holistic, which is deprecated/"coming soon" in the Python Tasks API). Apply a
**One-Euro filter** to landmarks before deriving any motion feature. Drives respiration
(chest/shoulder), posture, and gross motion.

**Why.** Only real-time option giving 3D world landmarks (x,y,z) out of the box, plus
per-landmark visibility/presence scores that gracefully handle the desk-occluded lower
body. Apache-2.0. ~27–30 fps CPU. Alternatives: **YOLO11-pose is AGPL-3.0** (network
copyleft triggers source disclosure even for SaaS → disqualified for a closed product);
RTMPose-m (Apache-2.0 code, ~90 fps CPU, but 2D-only, needs a separate detector, and some
wholebody weights carry non-commercial training data); MoveNet (2D, frozen since 2021);
ViTPose/Sapiens (not real-time). Revisit RTMPose via `rtmlib` only if a later phase needs
higher 2D precision.

---

## ADR-03 — Live rPPG: **classical POS** (KEEP)

**Decision.** The live cardiac path is **POS** (Plane-Orthogonal-to-Skin, Wang et al.
2017), with **CHROM** as a runtime-selectable fallback, computed over a rolling 8–15 s
window with HR updating ~1 Hz. Pure NumPy/CPU, deterministic, <1 ms/window, no GPU, no
training.

**Why.** For a clean single seated adult (Phase 0/1), POS reaches MAE ~1–4 bpm on
UBFC-rPPG/PURE — close enough to deep models that the accuracy gap is small on easy data,
at a fraction of the cost and complexity. POS is generally the best all-round classical
method; OMIT (2023) is competitive but not a categorical leap. **Known failure modes
(by design, our Phase-2 trajectory):** motion, illumination change, and skin tone —
chrominance methods degrade from MAE ~5.2 bpm (Fitzpatrick I–III) to ~14.1 bpm (V–VI).
We accept this for Phase 0/1 and gate on quality (ADR-06, ADR-14).

**Facial-hair caveat (rev 1).** The ROI uses cheek patches (landmarks 50/280). Beard/
stubble blocks visible skin perfusion and destroys the cheek rPPG signal. **Facial hair is
a recorded covariate** (TECH §9); when present, cheek patches are dropped (like the yaw
rule) and the estimator falls back to forehead/glabella, which lowers SNR and HRV yield.
State this in any report for a bearded subject.

---

## ADR-04 — Offline rPPG reference: **rPPG-Toolbox, modernized model set** (CHANGE)

**Decision.** For offline/research-grade comparison, use **rPPG-Toolbox** (the community
benchmark standard) but update the model set the brief named (PhysNet/TS-CAN/EfficientPhys
are now dated). Primary references: **RhythmFormer** (Pattern Recognition 2025) and
**RhythmMamba** (AAAI 2025); keep PhysNet/TS-CAN/EfficientPhys as legacy baselines.
Optionally add a self-supervised model (**Contrast-Phys+** TPAMI'24 or **Periodic-MAE**
2025) to enable fine-tuning to our own subject without ECG labels.

**Why.** The Mamba/state-space wave (2024–25) is the real shift: RhythmMamba matches
transformer accuracy at ~1.07M params / ~20 Kfps. On the discriminating benchmark (MMPD),
RhythmFormer/RhythmMamba reach ~3 bpm intra-dataset where older models sit at 10–17.
**Important honesty:** all models collapse cross-dataset (UBFC→MMPD MAE ~10–17) —
generalization to new lighting/skin/motion is unsolved; expect a large train→deploy gap.
**rPPG-Toolbox is offline by design** (do not force it into the live loop). For an optional
real-time *deep* path later, `open-rppg` (MIT) ships RhythmMamba/PhysMamba with a threaded
webcam pipeline and native SQI+HRV; or `ME-rPPG` (2025) runs at 9.46 ms/frame on a laptop
CPU. These are **out of Phase 0/1 scope** but recorded as the upgrade path.

---

## ADR-05 — HRV scope: **HR + SDNN committed; RMSSD indicative; no LF/HF** (CHANGE)

**Decision.** We commit (with acceptance criteria) to **HR** and **SDNN**. **RMSSD** is
logged and reported as *indicative only*. **LF/HF and frequency-domain HRV are explicitly
out of scope** — they will not be presented as validated outputs.

**Why (be honest — this is the most over-claimed area).** Two independent high-quality
results bound reality:
- 2026 naturalistic webcam validation (n=77, POS+demodulation): HR MAE **1.67 bpm**;
  SDNN MAE **11.45 ms**; RMSSD MAE **11.02 ms**; instantaneous-HR trajectory r=0.56.
  Authors' conclusion: rPPG reliably captures *average HR*; individual-level HRV is
  *limited*.
- WaveHRV (Bioengineering 2023), a dedicated method on clean UBFC data — the SOTA ceiling:
  SDNN MAE **6.15 ms**, RMSSD MAE **10.46 ms**.

RMSSD's ~10 ms error floor is large relative to typical RMSSD (20–50 ms). The mechanism:
ECG times the electrical R-peak; rPPG times an optical systolic upstroke carrying
pulse-arrival-time jitter (~tens of ms, varies beat-to-beat) — this sets an IBI noise
floor HRV cannot beat. LF/HF needs precise IBI over minutes and is untrustworthy from a
webcam. **Even the Polar H10 ground truth has seated-position RMSSD bias (−2 to −8 ms)** —
build that into the error budget (ADR-18: the H10 is a *reference*, not gold truth).
Required technique: cubic-spline upsample the pulse waveform to ≥256 Hz before peak
detection (so IBI isn't quantized to the 33 ms frame period), and gate hard on SQI (ADR-06).

**Acceptance-bar correction (rev 1).** The committed SDNN bar is stated as **SDNN MAE ≤ 12
ms** (consistent with the cited 11.45 ms naturalistic / 6.15 ms clean evidence) — **not**
`|bias| ≤ 15 ms`. Bias alone is the wrong statistic (a wildly noisy but symmetric estimator
passes it for the wrong reason). Always report Bland-Altman bias **and** 95% LoA **and**
Lin's CCC alongside the MAE. SDNN and reference must be computed over **identical,
time-aligned windows** (SDNN grows with window length).

---

## ADR-06 — Signal Quality Index: **De Haan SNR + Orphanidou bSQI** (ADD)

**Decision.** Every rPPG window emits an SQI. Two complementary measures:
- **Spectral SNR** (De Haan & Jeanne 2013): in-band power (±6 bpm around the HR fundamental +
  first harmonic) divided by **out-of-band power only** (signal bins removed from the
  denominator), evaluated over the band-pass support [0.7–4 Hz]. **Primary hard gate** (used
  by rPPG-Toolbox, pyVHR). *(rev 1: denominator excludes signal bins — otherwise it is not
  the De Haan ratio.)*
- **Orphanidou bSQI** (2015): per-beat template correlation. **Required for HRV beat
  acceptance** — drop low-SQI beats rather than emit a bad IBI. **The 0.86 threshold is
  provisional (rev 1):** it was derived for contact ECG/PPG morphology; rPPG upstrokes are
  smoother/lower-SNR, so the threshold must be **empirically calibrated on this subject's
  rPPG beats against H10-confirmed beats** (ROC), not assumed.

Supplement cheaply with skewness/perfusion-index. The motion gate (ADR-14) feeds the same
decision. Low-SQI records are emitted *with a flag*, not silently dropped (consumers
decide). The npj Biosensing 2024 "optimal SQI" study (Samsung) is the reference for the
supplemental set.

---

## ADR-07 — Respiration: **primary = chest/shoulder motion** (CHANGE)

**Decision.** Respiration's primary estimator is **chest/shoulder vertical displacement**
from MediaPipe Pose landmarks → band-pass **0.08–0.5 Hz** → FFT/peak-count over a 15–30 s
sliding window. **rPPG-derived respiration (RSA/amplitude modulation) is a secondary
cross-check only.** Head-motion respiration is a weak fallback. EVM/motion-magnification is
*not* a core estimator (it's a fragile visualization step).

**Ground-truth status (rev 2): respiration is *Indicative*, not Committed.** No respiratory belt
is being purchased, and a paced-breathing **metronome is a commanded target, not a measurement**
(a subject drifts/skips/doubles, so scoring against it measures *compliance*, not accuracy). With
no belt there is no valid RR ground truth, so respiration ships as an **Indicative** signal: the
metronome (6/10/15 brpm) is a coarse plausibility check only — report the estimate and its
plausibility, make **no MAE accuracy claim**. *(If a belt is ever obtained, it becomes the
reference and respiration can be promoted to Committed at MAE ≤ 2 brpm.)* Note the 6-brpm target
sits at the 0.08–0.1 Hz band edge — the hardest case (hence the widened low edge).

**Why.** Direct motion tracking beats rPPG-derived RR and EVM for a seated subject: chest/
shoulder methods report MAE **0.69–1.62 brpm** (15 s window) vs rPPG-RR's 2.8–5.6 brpm.
Trivial real-time Python. Target acceptance: **MAE ≤ 2 brpm vs the belt**, across paced
blocks (not "at rest," where the true free-breathing rate is unmeasured).

---

## ADR-08 — Blink / PERCLOS: **EAR + adaptive threshold** (KEEP+)

**Decision.** Blink detection and PERCLOS via **Eye Aspect Ratio on MediaPipe FaceMesh**,
with a **per-session adaptive (personalized) threshold** auto-calibrated from the open-eye
baseline, plus a PERCLOS (P80) window of 60–90 s. **30 fps is mandatory** (blink metrics
collapse from ~95–98% at 30 fps to ~51% at 10 fps).

**Why.** Learned detectors (BlinkLinMulT F1 0.99; mEBAL2 ~99%) beat fixed-threshold EAR but
need a GPU; for real-time CPU, EAR's cost/accuracy is still best, and the dominant cost is
the landmarker (shared, ADR-01), not EAR. The cheap, high-leverage 2023–26 fix is adaptive
thresholding, not abandoning EAR. EAR's documented failure modes (head pose, glasses
glare, fixed-threshold sensitivity) are mitigated by adaptive thresholds + the motion gate.
**Optional Phase-1 robustness upgrade:** a small MobileNetV2 eye-state CNN on the eye crop
(+few ms) for glasses/pose robustness — deferrable.

---

## ADR-09 — Gaze: **MediaPipe iris + learned head, output zones** (CHANGE)

**Decision.** Keep MediaPipe Iris for the real-time front-end (face/head-pose/eye-crops/
presence), but **do not rely on its geometric gaze for precision**. Add a compact
appearance model — **L2CS-Net (ResNet-18)** or **MobileGaze (MobileOne-S0)**, ONNX/CPU —
for the gaze vector. **Output direction zones** (left/center/right; screen on/off;
attention present/absent), **not** pixel point-of-regard or fine quadrants. A 5–9-point
per-user calibration is an **optional Phase-1 upgrade** (gets ~1.5–4.5°).

**Why (honest accuracy ceiling).** Benchmark headlines (3–4°) are known-user/known-camera.
A new uncalibrated user at 60 cm realistically gets **~6–10°** (~7–11 cm on-screen, >50% of
a laptop's vertical extent). MediaPipe's geometric gaze is ~8–15° in the wild — 2–4× worse
than L2CS-Net (3.92° MPIIGaze within-dataset). MobileGaze runs ~30 fps on CPU. Scoping the
output to zones matches what the hardware can actually support for an uncalibrated user.
(Note: EyeTrackVR's 4.8° uses near-eye IR in a headset — not comparable to a webcam.)

**Geometry & model-swap notes (rev 1).** "Zone accuracy" is meaningful only relative to zone
geometry: the validation protocol fixes target eccentricities and zone half-widths at the
viewing distance, and reports a **3×3 confusion matrix + accuracy above the 33% three-way
chance**, never a bare percentage (center↔side confusions dominate). L2CS-Net and MobileGaze
are **not drop-in** swaps (different input preprocessing, pitch/yaw sign, degrees vs radians,
gaze origin) — the zone-mapping layer is model-specific. **Default = L2CS-Net ResNet-18;**
MobileGaze is a stretch alternative behind config.

---

## ADR-10 — Pupillometry: **CUT from Phase 0/1** (CHANGE)

**Decision.** Remove pupillometry entirely from Phase 0/1, and specifically delete the
brief's "pupil-size proxy via MediaPipe iris." If an arousal proxy is needed, substitute
blink rate, gaze dispersion, and rPPG-derived HR/HRV.

**Why (two independent disqualifiers).**
1. **Category error.** MediaPipe Iris tracks the *iris*, whose physical diameter is
   biologically ~constant (~11.7 ± 0.5 mm) per person — that constancy is literally how
   MediaPipe estimates camera distance. The *pupil* (inner aperture) dilates with
   cognitive load; MediaPipe does not segment it. Tracking iris "size" yields a near-
   constant number plus jitter, not arousal.
2. **RGB infeasibility.** The task-evoked pupillary response is 0.1–0.5 mm. Best published
   RGB-webcam pupil error is MAE ≈ 0.13 mm (EyeDentify 2024) — the noise floor ≈ the entire
   signal. Visible light gives almost no pupil/iris contrast on brown/dark irises; the eye
   crop is only ~32–64 px wide at 60 cm. Every accurate system used IR hardware.

Revisit only with a ~$30–60 IR-LED + IR-pass-filter camera in a later phase (out of scope).

---

## ADR-11 — Facial affect (live): **blendshapes + HSEmotion** (CHANGE)

**Decision.** The live affect channel = MediaPipe 52 blendshapes (coarse proxy) **+
HSEmotion / EmotiEffLib** (`enet_b0_8_va_mtl`) for real-time 8-class emotion **and
continuous valence/arousal** from one model. Both run comfortably at 30 fps.

**Why.** AffectNet-8 SOTA is only ~64% overall — top models are within 1–2 points, so
**license and speed matter more than chasing accuracy**. HSEmotion is **Apache-2.0**
(commercial-OK), real-time (EfficientNet ONNX), multi-task (emotion + V/A), ~63%
AffectNet-8, ABAW-battle-tested. **EmoNet is disqualified** (CC BY-NC-ND); POSTER++/DAN are
MIT *code* but AffectNet/RAF-DB *weights* are commercially gray. **Blendshapes are NOT a
research feature** (uncalibrated 0–1 rig weights, ~0.20 MAD identity bias, lossy
many-to-one FACS mapping) — they're a low-latency live cue only, and must be neutral-
subtracted + z-scored per subject. **Diligence:** confirm HSEmotion's AffectNet-derived
weight provenance with counsel before any commercial/clinical deployment (or fine-tune on
licensed data). Live affect cadence is a **tunable (default 10–15 Hz)** — asserting a true
30 Hz HSEmotion pass competing with the backbone for CPU is optimistic; the live path stays
smooth by downsampling affect.

**Affect "validation" is a face-validity check, not affect measurement (rev 1).** Posed
expressions are a *stimulus*, not ground truth: a subject posing "happy" on cue performs a
facial configuration whose *felt* valence may be neutral or negative. Checking that V/A
covaries with posed expressions tests the **model's expression→V/A mapping on this face**,
NOT whether Argus measures the subject's affective state, and NOT spontaneous emotion. Affect
is **Exploratory**; the word "validation" is replaced by "face-validity check" for all affect
outputs. True affect validity would require an elicitation paradigm (film clips/IAPS) +
self-report — out of scope.

---

## ADR-12 — Facial affect (research AUs): **LibreFace (or OpenFace 3.0)** (CHANGE) — *rev 2*

**Decision (rev 2 — license no longer a constraint).** Research-grade Action Unit logging uses
the **best available model regardless of license: LibreFace** (best turnkey 0–5 AU intensity,
PCC 0.63, ~40 fps; ONNX/C# deployable) — or **OpenFace 3.0** (unified multitask, AU F1 ~59,
near-SOTA landmarks). Run it **decoupled at 5–15 fps or offline** (AU dynamics rarely need
30 Hz; downsampling the AU model is the single biggest lever for keeping the live path smooth).
Py-Feat (MIT) and ME-GraphAU remain fine fallbacks but are **no longer required** — dropping the
MIT-only constraint also removes the Py-Feat PyTorch-isolation headache (B2).

**Why.** With productification out of scope, the earlier blocker — that OpenFace 2.0/3.0 and
LibreFace are non-commercial — **no longer applies**. We simply pick the most accurate
research-grade intensity estimator: **LibreFace** (calibrated 0–5 FACS intensities, the gold
standard for affect research logging). This supersedes ADR-13.

---

## ADR-13 — ~~AU gold-reference (quarantined)~~ **WITHDRAWN (rev 2)**

**Withdrawn.** This ADR existed only to quarantine non-commercial AU models behind a
licensing firewall for a future product. With productification out of scope (rev 2), there is
**no quarantine, no `argus/research/` firewall, and no CI import-lint** — LibreFace/OpenFace 3.0
are used directly as the primary AU extractor (ADR-12). Delete all licensing-firewall machinery
referenced elsewhere (TECH §2 firewall, §14a import-lint; FEATURES G3, J2).

---

## ADR-14 — Motion quality gate: **first-class 3-tier traffic light** (ADD)

**Decision.** Add an explicit motion/quality gate as a first-class pipeline component,
reusing the *same* MediaPipe Face Mesh landmarks (no second tracker):
- **Input-side:** mean landmark displacement (FM_X/FM_Y), ROI-area change (FSM, ≈ z-motion),
  head pose via `solvePnP`. Pitch (nodding) hurts rPPG ~2× more than yaw — weight it.
- **Output-side:** De Haan SNR + skewness on the pulse; reject HR outside [42, 240] bpm or
  with implausible jumps.
- **Traffic light with hysteresis:** 🟢 GOOD (low motion AND SNR ≥ ~3 dB) → report HR+HRV;
  🟡 USABLE (mild motion OR SNR 0–3 dB) → HR only, widened CI, suppress HRV/RR; 🔴 REJECT
  (high motion OR SNR < 0 dB) → drop window, hold last good value, surface "re-acquiring."

**Why.** Motion is the #1 error source for every physiological signal here. The "Motion-
Based Confidence Score" approach (J. Med Syst 2026) feeds these features to a classifier at
AUC > 0.93; its labeling rule (good = HR error < 2 bpm; reject = > 6 bpm) is our template.
Normalize thresholds by inter-ocular distance; calibrate to our camera. Hysteresis prevents
state flicker.

---

## ADR-15 — Streaming bus: **dual LSL + OSC/WebSocket bridge** (CHANGE)

**Decision.** Two buses, bridged — not the brief's single "WebSocket or OSC":
- **Research path: Lab Streaming Layer (`pylsl`)** → synchronized multi-stream recording to
  a single **XDF** file via **LabRecorder**. One `StreamOutlet` per logical signal (or one
  multi-channel outlet); regular-rate signals (HR, respiration, motion) declare a
  `nominal_srate`; event-like signals (affect labels, fixations) use `IRREGULAR_RATE`.
- **Art/biofeedback path: a thin `pylsl` inlet → `python-osc` sender** feeding the canvas
  (swap OSC for **WebSocket** if the canvas is browser-based; keep LSL/XDF unchanged).

**Why (highest-leverage upgrade).** LSL is the de-facto standard in psychophysiology and
gives three things OSC/WebSocket cannot: NTP-like cross-device clock-offset estimation
(sub-ms on LAN), synchronized multi-stream recording with per-sample timestamps **and**
measured clock offsets stored in XDF, and zero-config stream discovery. This is what makes
the validation defensible. OSC remains the lingua franca of TouchDesigner/Max/openFrameworks
for the art side. The bridged pattern (LSL for research, OSC for creative) is exactly how
the NeuroPype/Petal/OpenBCI ecosystems are built. ZeroMQ/gRPC/MQTT are eliminated (no
art-tool ecosystem, no synced recording).

---

## ADR-16 — Time synchronization: **LSL timebase + cross-correlation residual align** (ADD)

**Decision.**
1. LSL `local_clock()` (steady_clock seconds) is the single monotonic timebase.
2. **Stamp each camera frame at grab time** — set `CAP_PROP_BUFFERSIZE=1`, grab
   continuously on a dedicated thread, stamp immediately after `read()`. Do **not** trust
   OpenCV `CAP_PROP_POS_MSEC` on a live stream. Calibrate the constant exposure→host
   latency once (LED-flash test) and subtract it.
3. **Polar H10 beat times = cumulative sum of R-R intervals from a single anchored LSL
   timestamp**, never from BLE packet-arrival time (beats arrive batched; packet arrival
   carries tens of ms of jitter).
4. Load XDF with `pyxdf.load_xdf(..., proc_clocksync=ON, proc_dejitter=ON)`.
5. **Fine residual alignment:** derive instantaneous-HR series from camera and H10, resample
   both to 4 Hz, **cross-correlate** to recover residual lag; then beat-match within
   ±50–100 ms for IBI-level HRV comparison. Use DTW only as a *reported* metric, never as
   the aligner you then validate against (circular).

**Two alignment refinements (rev 1).**
- **Run the cross-correlation on a motion/talking block, not a rest block.** At rest, HR is
  nearly flat and has no features to lock onto, so residual-lag recovery degenerates. Estimate
  the lag where HR varies, then apply it to the rest blocks.
- **For HRV, apply a single fixed lag (median pulse-transit-time), not a per-segment
  re-optimized lag.** The optical pulse genuinely lags the ECG R-peak by a real, beat-varying
  PTT (~tens–200+ ms) comparable to the ±50–100 ms match window. Re-optimizing the lag per
  segment would *absorb* part of the very beat-to-beat IBI jitter HRV is trying to measure.
  Cross-correlation alignment is for **HR-rate** agreement; HRV timing uses the fixed median
  lag and reports matched-beat yield.

**Why.** HRV validity hinges on beat-time alignment. The H10's R-R *durations* are <1 ms
accurate but the *absolute placement* of the beat train carries BLE/host uncertainty;
cross-correlation on a common signal recovers it. Apply linear `t' = α·t + β` resampling if
drift × session length approaches the beat-match tolerance.

---

## ADR-17 — Ground-truth integration: **Polar H10 via `bleak` + `polar-python`** (KEEP+)

**Decision.** Read the Polar H10 in Python via `bleak` (cross-platform async BLE) →
notifications on Heart Rate Measurement char `0x2A37`. **Parse with the full flag-driven
offset algorithm (rev 1 — the prior "flags bit0/bit4" shorthand was incomplete and would
silently mis-parse RR, invalidating all HRV):**

```
offset = 1                                  # skip flags byte
if flags & 0x01:  hr = u16le(buf, offset); offset += 2   # bit0=1 → HR is uint16
else:             hr = buf[offset];        offset += 1   # bit0=0 → HR is uint8
if flags & 0x08:  offset += 2               # bit3=1 → Energy Expended present (skip 2 bytes)
if flags & 0x10:                            # bit4=1 → RR-Interval(s) present
    while offset + 1 < len(buf):            # iterate ALL RR pairs to end of packet
        rr_raw = u16le(buf, offset); offset += 2
        rr_ms  = rr_raw / 1024.0 * 1000.0   # units are 1/1024 s
```

Use **`polar-python`** (typed, context-manager API) or **BleakHeart**; optionally reuse
**`PolarBand2lsl`** to publish straight to LSL. Correct RR artifacts with **NeuroKit2 Kubios**
(`signal_fixpeaks`) before any HRV stat. **Reference, not gold truth (rev 1):** the H10 is an
accepted research criterion device but carries its own seated-posture RMSSD bias (−2 to −8 ms);
reported HRV "error" therefore includes the reference device's bias and must not be fully
attributed to the camera (ADR-18). Optional richer reference: **Contec CMS50E** pulse-ox over
USB serial — used in Phase 0/1 as a **PPG-waveform reference only**; its SpO₂ field is ignored
(SpO₂ is out of scope, NG3).

**Why.** The H10 does its own R-peak detection → clean IBIs without writing a QRS detector,
and is an accepted research criterion (>99% HR vs ECG; R-R closely matches ECG, n=25).
**Gotchas:** on macOS, CoreBluetooth exposes a random UUID, not a MAC — discover by name;
add reconnect retries; PMD/ECG won't stream without skin contact.

---

## ADR-18 — Validation methodology: **full agreement set + pre-registered bars** (CHANGE)

**Decision.** Bland-Altman alone is insufficient. For every committed signal report:
**Bland-Altman bias + 95% LoA, MAE, RMSE, MAPE, Pearson r, and Lin's Concordance
Correlation Coefficient (CCC), plus SNR.** Pre-registered numeric bars:
- **HR:** meets the **EC13 numeric threshold** (error ≤ ±10% or ±5 bpm, whichever greater) and
  ANSI/CTA-2065 **MAPE < 10%**. *(rev 1: meeting the EC13/CTA-2065 numeric threshold on a
  single subject is a feasibility target — NOT EC13/CTA-2065 conformance, which requires a
  powered multi-subject study.)*
- **HRV — SDNN (committed):** **SDNN MAE ≤ 12 ms** (rev 1 — replaces the prior `|bias| ≤ 15 ms`,
  which is the wrong statistic: a noisy-but-symmetric estimator passes bias for the wrong
  reason). Also report BA bias + 95% LoA and Lin's CCC. **The ±15 ms band, if used, applies to
  the SDNN BA only.** SDNN and reference must be computed over **identical, time-aligned
  windows** (SDNN grows with window length).
- **HRV — RMSSD (indicative):** report ln-RMSSD BA in **log units / as a ratio with CI** — NOT
  in ms, and **not** against a ms band (rev 1: a ms band cannot apply to a log-scale BA). No
  pass/fail.

**The H10 is a reference, not gold truth (rev 1):** reported HRV bias includes the H10's own
seated-posture RMSSD bias; do not attribute it all to the camera. **Record covariates now:**
Monk/Fitzpatrick skin tone, **eyewear (glasses), and facial hair** — eyewear gates blink/gaze,
facial hair gates the cheek rPPG ROI; all three are otherwise invisible confounds. Skin tone
is recorded **for forward-compatibility with Phase 2 only** — at n=1 there is nothing to
stratify, and the automated cheek-reflectance estimate is **unvalidated/descriptive**. **If
the sole subject is Fitzpatrick V–VI, read the HR/HRV bars as a near-worst-case single point;
if I–III, a near-best case that will not generalize.** **Frame all single-subject results as
feasibility/repeatability, not accuracy clearance** — one subject cannot establish limits of
agreement or fairness.

**Why.** Pearson r measures association, not agreement; Bland-Altman is necessary but should
be paired with Lin's CCC (penalizes deviation from identity). MAE/RMSE/MAPE/Pearson/SNR is
the set the rPPG field (rPPG-Toolbox) expects. *Verify before quoting in any regulatory
doc:* ISO 80601-2-61 (pulse-ox A_RMS ≤ 4%, 2025 tightening toward 3%) and FDA 3.5% figures
come from secondary summaries.

---

## ADR-19 — Concurrency model: **threading + one backbone + windowed DSP** (ADD)

**Decision.**
- **Threading is the default** (OpenCV, NumPy, MediaPipe C++, ONNX all release the GIL).
  One dedicated **capture thread** + one **backbone thread** + N **extractor threads**.
- **One MediaPipe backbone pass per frame** → emits a `FrameContext{frame, ts, landmarks,
  roi, blendshapes, head_pose}`; each extractor is a plugin `consume(FrameContext) ->
  list[SignalRecord]` (rev 1 — **list, not Optional**; respiration legitimately returns both a
  primary and a cross-check record; `[]` means "no window ready"). Each owns its own
  buffer/cadence, never blocks the backbone, emits standardized `{signal_name, value, sqi,
  timestamp}` (timestamp = **capture** time). **`FrameContext.frame` is read-only** — it is
  shared across all fan-out consumers; no extractor may write to it (`frame.flags.writeable =
  False` is set on construction; extractors needing pixels copy first).
- **Two buffer disciplines:** latest-frame **drop-oldest** (1-slot, `.copy()` on hand-off)
  for per-frame extractors + display; **lossless evenly-timestamped ring buffer** for
  time-series extractors (rPPG/HRV/respiration).
- **Windowed DSP, not per-frame DSP** — accumulate a sliding window, run batch DSP (pyVHR-
  style) on it.
- **Multiprocessing + `shared_memory`** only as a per-extractor escape hatch for a genuinely
  Python-CPU-bound extractor (ship a frame *handle*, not a pickled array). **asyncio only
  at the I/O edge** (LSL/OSC emit), never on the CV hot path.
- **Skip Python 3.13 free-threading for now** — ~40% single-thread penalty and no OpenCV
  free-threaded wheels as of mid-2026; design the MP escape hatch so we don't depend on it.

**Why.** This is what makes 30 fps multi-signal physiology tractable in Python without
fighting the GIL. pyVHR is the reference architecture for the windowed rPPG path.

---

## ADR-20 — Signal-processing libraries: **NeuroKit2 + pyVHR + HeartPy** (ADD)

**Decision.** **NeuroKit2** is the primary analysis engine (HRV time/nonlinear, respiration
RSP+RRV, signal quality `ppg_quality`, Kubios artifact correction) — it accepts either a raw
waveform or precomputed peaks/intervals, so it serves both the camera BVP and the H10 RR.
**pyVHR** is the reference for the camera→BVP extraction step (classical POS/CHROM on
MediaPipe ROIs). **HeartPy** is a streaming/robustness cross-check (second peak-detector
opinion). All three are batch/window-oriented — call on the sliding window (fits ADR-19).

**Why.** NeuroKit2 is the only lib covering HRV + respiration + quality under one uniform,
maintained, citable API. rPPG→HRV is much noisier than contact PPG — gate aggressively
(`ppg_quality` + Kubios) before computing HRV, and confine HRV claims to rest.

---

## What we are explicitly NOT doing in Phase 0/1 (scope guards)

- ❌ Pupillometry / cognitive-load from pupil (ADR-10).
- ❌ LF/HF or frequency-domain HRV as a validated output (ADR-05).
- ❌ Deep-learning rPPG in the **live** path (offline reference only; real-time deep path is
  a recorded upgrade, not Phase 0/1).
- ❌ Pixel-accurate gaze / fine quadrant gaze for uncalibrated users (ADR-09).
- ❌ *(rev 2: licensing is no longer a scope guard — non-commercial models like LibreFace/
  OpenFace 3.0 are used freely; no quarantine.)*
- ❌ Respiration **accuracy** claims — Indicative only, no belt (rev 2, ADR-07).
- ❌ The art canvas itself (it is a downstream bus consumer; Phase 0/1 delivers the bus + a
  validation dashboard, **canvas = TouchDesigner/OSC**).
- ❌ Multi-subject, dyads, children (Phases 2–4).
- ❌ SpO₂ / blood pressure.

---

## Open items (rev 2 — licensing/regulatory diligence dropped)

1. ~~HSEmotion / Py-Feat license diligence~~ — **dropped (rev 2, productification out of scope).**
2. ~~ISO/FDA A_RMS figure verification~~ — **dropped** (no regulatory citation at this stage).
3. Measure actual **Polar H10 BLE end-to-end latency** distribution on our hardware *(only if a
   Polar H10 is on hand — see below)*.
4. ~~Decide art-canvas medium~~ — **resolved: TouchDesigner over OSC** (p5.js/WebSocket if a
   web-shareable piece is later wanted).
5. **Confirm cardiac ground-truth availability** — does the team already own a Polar H10 (or any
   ECG/contact-PPG reference)? If not, the cardiac validation method needs revisiting (no purchase).

---

## Consolidated sources (selected, by area)

**rPPG / cardiac:** Wang et al. POS (IEEE TBME 2017); De Haan & Jeanne SNR (TBME 2013);
RhythmFormer (arXiv 2402.12788, PR 2025); RhythmMamba (arXiv 2404.06483, AAAI 2025);
ME-rPPG (arXiv 2504.01774, 2025); Contrast-Phys+ (TPAMI 2024); rPPG-Toolbox (arXiv
2210.00716, NeurIPS 2023); pyVHR (PeerJ CS 2022); open-rppg (KegangWangCCNU/open-rppg, MIT);
webcam HRV validation (Behavior Research Methods 2026, PMC13106271); WaveHRV (Bioengineering
2023, MDPI 10(7):851); Orphanidou bSQI (IEEE J-BHI 2015); npj Digital Medicine 2025 skin-tone
bias (s41746-025-01973-9); MMPD (arXiv 2302.03840, EMBC 2023).
**Face backbone / affect:** MediaPipe Face Landmarker (Apache-2.0); Blendshapes GHUM (arXiv
2309.05782); OpenFace 3.0 (arXiv 2506.02891, FG 2025, CMU non-commercial — verified);
LibreFace (arXiv 2308.10713, WACV 2024, USC research license — verified); Py-Feat (arXiv
2104.03509, MIT); ME-GraphAU (IJCAI 2022, MIT); HSEmotion/EmotiEffLib (Savchenko, Apache-2.0);
R2I-rPPG ROI (arXiv 2410.15851, 2024); npj Digital Medicine rPPG-ROI review 2025 (PMC12297079).
**Ocular / motion:** EAR (Soukupová & Čech 2016); BlinkLinMulT (J. Imaging 2023); L2CS-Net
(arXiv 2203.03339); ETH-XGaze (arXiv 2007.15837); UniGaze (arXiv 2502.02307); MobileGaze
(yakhyo/gaze-estimation); EyeDentify pupil (arXiv 2408.10397); respiration-from-motion
("Seconds Matter" 2026, PMC12987125; MediaPipe-Pose ESWA 2023); BlazePose (Apache-2.0);
RTMPose (arXiv 2303.07399); motion-confidence gate (J. Med Syst 2026).
**Streaming / validation / architecture:** Lab Streaming Layer (Kothe et al., Imaging
Neuroscience 2024); pylsl / LabRecorder / pyxdf; python-osc; `bleak`; `polar-python`;
BleakHeart; PolarBand2lsl; Polar H10 validation (Gilgen-Ammann 2019; Schaffarczyk 2022,
Sensors 22:6536); NeuroKit2 (Makowski et al. 2021); HeartPy (Med Eng & Physics 2019);
Lin's CCC; ANSI/AAMI EC13; ANSI/CTA-2065; ISO 80601-2-61; FDA pulse-ox draft (Jan 2025).
