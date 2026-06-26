# Argus — Algorithm Review (are we using the best methods?)

*A 2023–2026 literature review of every signal-estimation choice, with a verdict on whether
Argus is using the best available approach or missed something better.*

**Date:** 2026-06-25 · **Method:** parallel literature-review agents per signal cluster, each
given Argus's actual implementation and asked to assess it against recent SOTA, with citations.

> **How to read the verdicts:** **KEEP** = our choice is at or near SOTA for the constraint
> (single RGB webcam, real-time, runnable on a laptop); **HARDEN** = keep the method, add
> specific upgrades; **SWITCH** = a named alternative is meaningfully better and practical;
> **MEASURE** = a correctness gap we aren't currently quantifying.

---

## TL;DR scorecard

| Signal | Our current method | Verdict | Highest-value change |
|---|---|---|---|
| **Heart rate (rPPG)** | Classical POS, multi-patch ROI, PSD peak | **KEEP** (+ pilot ME-rPPG) | Per-patch SNR weighting + add **glabella**; measure **skin-tone bias** |
| **HRV** | POS BVP → upsample → peaks → NeuroKit2, SDNN/RMSSD | **HARDEN** | **Parabolic peak interpolation** + better SQI; prefer 60 fps |
| **Respiration** | Pose shoulder displacement + FFT (~1.6 brpm) | **HARDEN / SWITCH** | Move to **chest-ROI optical flow (Farnebäck)** → ~0.5–1 brpm |
| **Blink / PERCLOS** | Adaptive-threshold EAR | **KEEP counting; HARDEN PERCLOS** | Add **graded eye-openness** (blendshape/RT-BENE) for true P80 |
| **Fidget / restlessness** | Shoulder-landmark jitter energy | **AUGMENT** | Add **MEA frame-differencing** + a smoothness axis (SPARC/jerk) |
| **Posture / slouch** | Baseline-relative geometry (neck ratio, shoulder width, tilt) | **KEEP (SOTA-for-frontal) + HARDEN** | Median-window baseline, temporal hysteresis, add head-roll |
| **Gaze (zones + screen)** | Iris geometry + calibrated polynomial-ridge w/ head pose | **KEEP (fixes in plumbing)** | Report cm/deg error, add distance feature, one-euro filter |
| **Affect (main)** | HSEmotion `enet_b0_8_va_mtl` native V/A | **KEEP + small upgrade** | Swap V/A head to **MT-DDAMFN** (valence CCC 0.59→0.73) |
| **Affect (complementary)** | AU→emotion (Du/Martinez + Zhang'24) | **KEEP** | Already research-grounded |

**The single biggest gap across the whole system is not any one algorithm — it is
skin-tone fairness measurement on the rPPG path** (HR error degrades ~2–3× for Fitzpatrick
V–VI). That dwarfs the POS-vs-deep question in real-world impact. See §9.

---

## 1. Heart rate (rPPG) — **KEEP POS, pilot ME-rPPG, harden the ROI**

**Our choice:** classical POS (Wang 2017) on forehead+cheek patches, 10 s window, PSD peak.

**Verdict: KEEP for the live default.** The 2023–2026 benchmark literature is consistent:
POS still **beats or ties most deep rPPG models cross-dataset on the hard motion/lighting set
(MMPD)**, and it is training-free, zero-dependency, and trivially real-time. Cross-dataset HR
MAE on MMPD (rPPG-Toolbox, NeurIPS 2023): **POS 12.36**, CHROM 13.66, TS-CAN 13.9–19.0,
EfficientPhys 13.8–20.4, PhysFormer 12.1–22.7. POS beats the deep models in most cross-dataset
configs; the deep advantage is real **only with a large domain-matched training corpus**.

**The one practical upgrade worth piloting:** **ME-rPPG** (arXiv 2504.01774, 2025) via
**`pip install open-rppg`** (MIT, ONNX) — the only deep model that is simultaneously SOTA
(MMPD MAE **5.38** when RLAP-trained), CPU-real-time (**~9.46 ms/frame**, self-reported), and
installable. Adopt as an **optional enhanced mode gated behind our own validation**, not a
forced switch. Expected gain: ~5–7 bpm on hard-motion clips *if the training domain matches
our users*; ~zero on clean seated/well-lit faces where POS already hits 1–4 bpm.

**ROI — evolve, don't replace (highest-ROI HR change):**
- **Add the glabella** (between eyebrows). The most rigorous 28-region study (npj Cardiovascular
  Health 2024) ranked glabella **#1** on MAE/PCC/SNR under motion and cognitive load; de-emphasize
  the lower cheek (ranks 8th–9th under motion).
- **Add per-patch SNR/PSD quality weighting + yaw-based occlusion fallback** — reject patches with
  motion/occlusion/specular highlights each frame, fuse the survivors (R2I-rPPG 2024). Largely
  orthogonal to which patches we pick; biggest robustness win.
- Learned skin segmentation (SkinMap, BiSeNet) halves HR error under rotation/talking but adds
  compute — adopt only if the live camera sees real motion.

**Skip:** RhythmMamba/PhysMamba (no CPU proof, no pip); rPPG-Toolbox as a *live* engine (offline
by design).

**Honest ceiling:** clean seated/well-lit ~1–4 bpm (POS already there); webcam-in-the-wild
~8–15 bpm; walking ~17–30 bpm regardless of method. *Switching algorithms buys robustness at
the margins; quality-gating, ROI hygiene, fairness, and 60 fps buy more.*

---

## 2. HRV — **HARDEN: parabolic interpolation + better SQI**

**Our choice:** POS BVP → cubic-spline upsample ≥256 Hz → systolic peaks → IBI → SDNN/RMSSD
(NeuroKit2), Orphanidou bSQI gate, no LF/HF.

**Verdict: broadly best-practice; two concrete upgrades move it to SOTA-classical:**
1. **Add parabolic/quadratic peak interpolation** on the 3 points around each peak. Spline
   upsampling raises the grid but does **not** recover sub-sample peak timing — which is exactly
   what limits RMSSD. Keep the spline, add parabolic interpolation, and run NeuroKit2
   `signal_fixpeaks` (Lipponen–Tarvainen) for IBI correction.
2. **Upgrade the SQI.** Orphanidou bSQI is a dated baseline. Add per-window **spectral-SNR / NSQI**
   gating (rPPG-specific NSQI threshold ~0.293, npj Biosensing 2024) and a **skewness SQI**
   (repeatedly the best single PPG quality index). Gate per-window, not per-recording.

**Learned/joint rPPG-HRV models: not worth it yet** — deep models optimize HR (bpm MAE); there is
no robust evidence they beat classical peak-detection on RMSSD/SDNN from face video.

**Honest ceiling (30 fps webcam):** best case (still, good light) SDNN MAE ~6 ms, RMSSD ~10 ms;
realistic naturalistic (77-subject 2026 study) SDNN ~11 ms, RMSSD ~11 ms. RMSSD is reliable
**only at group level**; pNN50/LF-HF are unrecoverable (our exclusion is correct). **Prefer 60 fps**
if hardware allows (halves 33→16.7 ms timing quantization, materially helps RMSSD).

---

## 3. Respiration — **SWITCH the method: chest-ROI optical flow**

**Our choice:** MediaPipe Pose shoulder-landmark vertical displacement + FFT (~1.6 brpm in the one
direct study of this exact method).

**Verdict: HARDEN→SWITCH.** Landmark-displacement + FFT is the *noisiest* of the camera-RR families
(MediaPipe-Pose RR: **MAE 1.62, PCC 0.45**). The strongest classical family is **optical flow on a
chest ROI (Farnebäck)**: Protopopov 2025 reports **MAE 0.57 brpm** on a 720p webcam ~60 cm from a
*seated* adult with motion/talking allowed — exactly our scenario; Romano 2021 reports 95% of
optical-flow errors under ±1 brpm seated (vs 76–81% for RGB-intensity methods). Farnebäck is
repeatedly the most consistent flow variant.

**Recommendation:** add a **chest-ROI Farnebäck optical-flow** respiration estimator (band-pass
0.08–0.5 Hz, peak/FFT) as the primary, keeping the shoulder-landmark estimate and rPPG-derived RR
as cross-checks. Expected: ~1.6 → ~0.5–1 brpm MAE for a seated subject. Dominant error source
across all studies is non-respiratory motion — keep motion gating.

**Honest target:** ~1–2 brpm out-of-the-box (landmark), ~0.5–1 brpm with chest-ROI optical flow.

---

## 4. Blink / PERCLOS — **KEEP EAR for counting; HARDEN the PERCLOS path**

**Our choice:** adaptive-threshold EAR; blink = EAR < thr for ≥2–3 frames; PERCLOS = % eye-closed
over a window.

**Verdict on blink *counting*: KEEP.** In our benign case (seated, frontal, decent light), EAR is
only ~3–6 F1 points behind SOTA — not worth a model swap. Reference numbers: BlinkLinMulT (Fodor
2023, *J. Imaging*) F1 1.000 talkingFace / 0.991 Eyeblink8 vs EAR ~0.943 / 0.955; mEBAL2 CNN ~97%
(Daza 2024). Personalizing the threshold (which we do) is worth only ~2–3% over a fixed one — we've
already captured that win. A full sequence transformer (BlinkLinMulT / DE-ViViT) wins on in-the-wild
multi-person untrimmed video, **not** our seated frontal case — skip it.

**Verdict on PERCLOS: HARDEN — this is the real gap.** True PERCLOS is the **P80** standard (% time
eyes are **>80% closed**), which needs a **graded eye-aperture** signal. A binary `EAR < thr` flag
**cannot compute genuine P80** — it conflates squint/downward-gaze with closure. Fixes, in priority:
- **Add a continuous eye-openness signal.** Cheapest high-value add: fuse the MediaPipe
  FaceLandmarker `eyeBlink` **blendshape** scores (nearly free — we already run the model) with EAR.
  Alternative: an RT-BENE-style eyelid-distance regressor (F1 ~0.976).
- **Proper blink exclusion** (<400 ms events) and a **documented rolling window** (start 60 s) for
  the PERCLOS calc.
- **Know the operating envelope:** even SOTA collapses off-axis (>25° pitch) and under glare. If the
  deployment sees looking-down or glasses-glare, consider a 3D-landmark eyelid metric or small
  eye-state CNN — otherwise not needed.

**Validate:** blink on mEBAL2 / Eyeblink8; drowsiness/P80 on DROZY / NTHU-DDD. No one has published
EAR-vs-learned numbers on *our exact* MediaPipe pipeline — measure our current adaptive-EAR F1 and
P80 error before changing anything.

---

## 5. Fidget / restlessness — **AUGMENT (don't just keep)** *(lower-confidence section)*

**Our choice:** variance-of-velocity (jitter energy) of shoulder image-landmarks over ~5 s.

**Verdict: AUGMENT.** Variance-of-velocity is a *reasonable but under-validated* energy proxy. The
**field-standard** metric in mental-health / psychotherapy video is **Motion Energy Analysis (MEA)** —
frame-differencing pixel change within body ROIs (Ramseyer & Tschacher; tool MEA5/psync, R package
**rMEA**), not pose-keypoint variance. MEA has direct clinical lineage: depression severity tracks
reduced gross body movement via MEA (HAM-D/BDI-II, n=41); ADHD/hyperactivity has been quantified from
compressed webcam video and skeleton features; psychomotor agitation/retardation is classically
actigraphy — MEA is its "video actigraphy" equivalent.

**Three weaknesses of our current metric:** (i) sensitive to MediaPipe landmark jitter (false motion
when the subject is still); (ii) not validated against any clinical scale the way MEA is;
(iii) captures *amount* of motion, not its *character* (smooth large motion vs small jitter score
alike). And the synchrony literature warns that different motion metrics can correlate **negatively**
(e.g. r ≈ −0.58) and yield opposite conclusions — a single scalar is risky.

**Recommended augmentation (low cost):**
- **Primary restlessness scalar → MEA-style frame-differencing** over a torso/hand ROI (robust to
  landmark jitter; directly comparable to a large clinical literature).
- **Second axis → movement-smoothness:** SPARC (spectral arc length) or log-dimensionless jerk on
  the keypoint velocity profile — captures "jittery vs smooth," closer to "fidgety" than total energy.
- Keep posture-lean as-is (orthogonal).

**Caveat (high uncertainty):** exact effect sizes for the depression-MEA / ADHD-webcam studies
weren't fully verified (the research stream was rate-limited). The *direction* and MEA's standing as
the validated default are well-supported; treat specific numbers as needing a confirmation pass.

---

## 6. Posture / slouch — **KEEP (it IS SOTA-for-frontal), HARDEN robustness**

**Our choice:** capture a "good posture" baseline; flag deviations in shoulder-width (distance
proxy), neck ratio, lateral offset, shoulder tilt.

**Verdict: KEEP — this is the de-facto state of the art for a single frontal webcam, and we are on
it, not behind it.** The literature is clear on two things:
1. **A frontal camera fundamentally cannot directly measure sagittal slouch / forward-head.**
   Forward-head is the craniovertebral angle (CVA), a *sagittal* angle needing a **side view**
   (tragus + C7); monocular projection is depth-ambiguous along the optical axis — exactly the
   direction the head moves when it goes forward. The best single-frontal-image method in the
   literature (JMIR 2024) **abandons CVA measurement and classifies** FHP at only ~78% accuracy.
2. **The leading real webcam tools converge on exactly our strategy** — SitSense, slouch-detector,
   Nekoze all use **personal baseline + face-size/nose-to-shoulder proxies + relative deviation**.
   SitSense openly admits depth "is estimated rather than directly measured."

**HARDEN (move from "reasonable heuristic" to "robust best-in-class frontal"):**
- **Median-window baseline** (e.g., 15–30 frames) instead of a single click — one blink/lean at
  capture poisons the reference. *(highest-value, ~10 lines)*
- **Temporal persistence / hysteresis** — only flag "poor" after N seconds sustained (LearnOpenCV
  uses 180 s; we can use ~3–5 s) to stop flicker.
- **Add head-tilt (roll)** — inter-eye/ear-line angle; honestly frontal, near-zero cost.
- **Require face-closer AND neck-shorter** to call "forward-head" (vs just "looking down").
- **Confidence-gate** on landmark visibility; withhold a verdict when the body isn't in frame.

**Honest caveat to keep in the docs:** baseline-relative / frontal-proxy posture has **no
peer-reviewed validation against goniometer/CVA/mocap** — it's a well-adopted heuristic, fine for
relative trend monitoring, **not** a clinical instrument. Never report CVA degrees. The only path
to validated measurement is an **optional side-view capture** — the one upgrade our frontal-only
design structurally can't do otherwise.

---

## 7. Gaze — **KEEP the architecture; the gains are in the plumbing, not the model**

**Our choice:** coarse zones from MediaPipe-iris geometry (L2CS-Net ONNX as an option); calibrated
screen point-of-regard via engineered 6-feature vector → degree-2 polynomial **ridge** regression,
with a **2-step calibration** (eyes-still, then fixate-while-moving-head), EMA-smoothed cursor.

**Verdict: KEEP — this is a sound, published design, not a naive one.** Two independent confirmations:
1. **Ridge-on-polynomial-features is the empirically best regressor at our point budget.** At ≥9
   calibration points it significantly outperforms other methods; GPR only wins with *many* points
   and is worse below 9 (Sesma-Sanchez ETRA 2016; arXiv 2009.01270). Our λ=1e-2 ridge on 28 poly
   terms with ~13–18 points is squarely in the recommended regime. Error saturates ~9 points — our
   13 dots are already past the knee.
2. **Our step-2 ("fixate while moving the head") is the exact technique a 2025 paper independently
   proposes** (Zhao et al., BMVC 2025 / arXiv 2508.10268). Hard numbers: dynamic multi-pose
   calibration hit **1.10 cm vs 1.52 cm static (28% better)**, and a single point under 3–4 head
   poses beat 9–13 points under one pose. Critically, static single-pose calibration applied
   cross-pose **blows up to 7.93 cm (worse than no calibration)** — exactly the failure our step-2
   prevents. This is the strongest single reason to **keep** our design.

**Would going deep (FAZE / WebEyeTrack) help?** Only modestly — **~0.5–1.5°**, mostly head-pose
robustness which we already approximate by sampling poses in calibration. FAZE lands ~3.18°
(GazeCapture) / 3.53° (MPIIGaze) at k≤9 — same ballpark as a well-tuned webcam regression — at the
cost of a full ONNX/training/runtime burden. **Not worth a wholesale switch.**

**The accuracy we're leaving on the table is in the plumbing (days of work, likely moves us from
~3–4° toward the ~1.5–2° ceiling):**
- **a. Report error in cm/degrees, not "% of screen."** Currently un-benchmarkable against the
  literature and hides whether we're at 1.5° or 4°. Capture screen size + viewing distance. *(fix first)*
- **b. Add a distance/scale feature** (inter-pupillary distance in pixels, or MediaPipe depth) as a
  7th feature — distance-to-screen change is the #2 failure mode and silently breaks the mapping.
- **c. Replace fixed EMA (α=0.35) with a one-euro filter** — standard for gaze cursors, adapts cutoff
  to velocity (less lag on saccades, less jitter on fixations), smallest error margin in comparisons.
- **d. Add drift detection / online recalibration** — embed validation dots; optionally click-anchored
  correction (gaze≈cursor at click time, WebGazer's trick); recalibrate every ~3–6 min.
- **e. Reduce the polynomial degree and report held-out CV error.** A full degree-2 over 6 features
  is **28 terms per axis** fit from ~13–18 points — **underdetermined by 10–15 DOF**, so our λ=1e-2
  ridge isn't optional, it's load-bearing, and extrapolation outside the calibration hull is still
  ~8–10× worse (~0.6° inside vs ~5.9° outside). Most defensible at this budget: **degree-1 + ridge
  (7 terms, overdetermined)**, keeping degree-2 only with cross-validated λ and corner-reaching dots.
  Report leave-one-point-out CV error, not training error — we currently can't see the overfitting.
- **f. Coarse zones:** keep iris geometry; **drop L2CS-Net as the learned fallback** (heavy ResNet-50,
  repo stale since 2024-02). If a learned fallback is wanted for off-axis robustness, use a
  lightweight ONNX net (GazeTR-class ~11M, or MobileOne-S0 ~4.8 MB), not L2CS. Note: MediaPipe Iris
  doesn't emit a gaze *vector* — our `iris_gaze_angles()` is a hand-rolled offset heuristic, so there
  is no published degree figure to compare it against; don't claim one.

**Honest ceiling:** webcam + calibration → ~1.4–1.5° best case (~1.5 cm at 60 cm, cooperative + virtual
chinrest), ~2–4° typical. IR trackers reach ~0.3° and ~20–100× better precision — webcam is good for
coarse AOIs (~4–6 regions), not microsaccades/fixation-duration.

---

## 8. Affect — **KEEP HSEmotion; close the one real gap (V/A head)**

**Our choice (main panel):** HSEmotion `enet_b0_8_va_mtl` native valence/arousal + 8-class emotion.
**Complementary (under AU bars):** AU→emotion via Du/Martinez (2014 PNAS) prototypes + Zhang'24 V/A.

**Verdict: KEEP HSEmotion — do not switch libraries.** It is within ~1.8 pts of the best
*installable, commercially-licensed* discrete FER (POSTER++ 63.8% vs our 61.9% on AffectNet-8 —
within label noise), and every genuinely-more-accurate option (ABAW winners, temporal video,
multimodal) is a non-installable research ensemble. The honest live-webcam ceiling is **~60–65%
8-class / CCC ~0.45–0.55** — a property of the *task and its labels* (AffectNet's two human
annotators agreed only 60.7% of the time), not our model.

**The one meaningful, in-library upgrade:** swap the V/A inference from `enet_b0_8_va_mtl` to
**MT-DDAMFN** (same `hsemotion`/EmotiEffLib package, Apache-2.0): AffectNet V/A CCC
**0.594/0.549 → 0.729/0.643** (~+0.13 valence). Optionally `enet_b2_8` for the discrete bump
(61.9% → 63.0%), at ~3× per-frame cost (still real-time-ish). Migrate to the maintained
**EmotiEffLib** package (the standalone `hsemotion-onnx` wrapper is stale).

**Optional consolidation:** **py-feat v2's** single ConvNeXt-V2 model now emits 20 AUs + 7-class
emotion + **native V/A** in one pass — since we already run py-feat for AUs, a side-by-side could
collapse two dependencies into one (cost: 8→7 emotion classes, heavier inference).

**Keep the complementary AU-affect** — it's already on the research-backed Du/Martinez + Zhang'24
method, and it's a genuinely *independent* second opinion (py-feat's emotion head is image-based,
not AU-derived).

**Framing (important for a wellness/research demo):** basic-emotion theory is contested (Barrett
2019: facial configs are "not diagnostic of emotional state"); these read *expressions, not felt
emotion*; demographic bias is documented; the EU AI Act bans emotion inference in workplace/
education (Feb 2025). Keep the "estimate, not measurement" framing and consent.

---

## 9. Cross-cutting gaps we MISSED (prioritized)

1. **Skin-tone fairness on rPPG — the #1 correctness risk.** HR error degrades ~2–3× for
   Fitzpatrick V–VI (CHROM/POS 5.2 → 14.1 bpm; meta-analysis 4.23 → 13.58). We don't currently
   *measure* this. Action: record Monk/Fitzpatrick, **report HR/HRV error stratified by skin tone**,
   consider a deep core (less biased) + **PhysFlow** skin-tone augmentation. HRV-across-skin-tone is
   essentially unstudied — validate, don't assume HR fairness transfers.
2. **Per-window SQI everywhere** (rPPG SNR/skewness, respiration band-power) — gate per-window, not
   per-recording.
3. **Temporal smoothing of noisy traces** (affect V/A — done; consider for gaze and HRV display).
4. **60 fps capture** for HRV (materially better RMSSD) — a hardware/config change, not an algorithm.
5. **Honest-ceiling framing in the UI** — every signal should show its realistic accuracy bound,
   not just a number.
6. **No ground-truth validation set of our own.** Every verdict here is against *literature* numbers,
   not *ours*. The single most valuable engineering investment is a small in-house validation set:
   HR/HRV vs a contact PPG/ECG, RR vs a belt, blink/P80 vs DROZY-style labels, fidget vs MEA. We're
   optimizing against papers, not against measured Argus error.
7. **A shared per-frame motion / signal-quality index.** Non-task body motion is the #1 error source
   for rPPG, respiration, blink-P80, *and* fidget simultaneously. One shared quality signal could
   drive rPPG-vs-motion fusion weight, PERCLOS validity, and separate true fidget from tracking
   noise — more leverage than any single algorithm swap.

---

## 10. What this review did NOT change

The earlier architectural decisions (ADR) hold up well: MediaPipe backbone, POS live + deep offline,
HRV scoping (SDNN/RMSSD only, no LF/HF), pupillometry cut, gaze-as-zones, the licensing/affect
choices. The reviews **confirm** most live choices are SOTA-for-constraint; the deltas are
refinements (ROI hygiene, optical-flow respiration, parabolic HRV peaks, MT-DDAMFN V/A) and one
genuine gap (skin-tone fairness measurement), not wholesale replacements.

*Sources are inline per section; full citation lists are in the per-cluster research outputs.*

---

## 11. Implementation status (all items shipped — 2026-06-25)

Every recommendation above is implemented and unit-tested (suite: 220 passing). Map:

| Review item | Where |
|---|---|
| §1 Glabella patch + per-patch SNR fusion + occlusion fallback | `dsp/roi.py` (`roi_patch_stack`, `active_patches`), `dsp/sqi.py` (`fuse_patches_by_snr`), `extractors/rppg_extractor.py` (`per_patch=True`) |
| §2 Parabolic peak interpolation + Lipponen–Tarvainen IBI correction | `dsp/hrv.py` (`detect_peaks(interpolate=True)`, `correct_ibis`) |
| §2 NSQI + per-window SQI gate (SNR + NSQI + skewness) | `dsp/sqi.py` (`nsqi`, `window_sqi_gate`), wired in `extractors/hrv_extractor.py` |
| §3 Chest-ROI Farnebäck optical-flow respiration (SQI-selected primary) | `dsp/respiration.py` (`ChestFlowRespiration`, `farneback_vertical_motion`), `extractors/respiration_extractor.py` |
| §4 Graded eye-openness (EAR + eyeBlink blendshape) + P80 with blink exclusion | `dsp/blink.py` (`eye_openness`, `PerclosP80`), `extractors/blink_extractor.py` (emits `perclos`) |
| §5 MEA frame-differencing + SPARC/LDLJ smoothness | `dsp/motion.py`, `extractors/motion_extractor.py` (emits `motion_energy`, `fidget_smoothness`) |
| §6 Median baseline, temporal hysteresis, head-roll, forward-head AND-gate, confidence gate | `perception/posture.py` (`begin/feed_baseline`, `PostureDebouncer`, `roll_deg`), `scripts/web_server.py` |
| §7 Inter-ocular distance feature, degree-1 ridge, one-euro cursor, cm/° error, drift recal | `perception/gaze.py` (`gaze_features` 7-dim, `PolynomialRidge`, `angle_to_screen_cm`), `web/gaze.html`, `web/index.html` |
| §8 Newer multi-task V/A head (EmotiEffLib `mobilevit_va_mtl`) with graceful fallback | `perception/affect.py` (`EmotiEffLibAffectEstimator`, `build_affect_estimator`), `scripts/web_server.py`, `requirements.txt` |
| §9 Skin-tone fairness (ITA°→Fitzpatrick + stratified error) | `quality/fairness.py`, surfaced live in `web_server` + `web/index.html` |
| §9 Shared per-frame motion-quality index | `quality/motion_index.py`, surfaced as `motion_quality` |
| §9 Honest accuracy ceilings per signal | `quality/ceilings.py`, surfaced as hover tooltips in `web/index.html` |

Model-dependent paths (EmotiEffLib V/A, py-feat AUs) degrade gracefully when weights/packages are
absent and their inference lines are `# pragma: no cover`; the selection/fallback logic and all DSP
is covered by deterministic tests against synthetic/fake inputs. Real EmotiEffLib `mobilevit_va_mtl`
inference was validated end-to-end (weights auto-download on first use).
