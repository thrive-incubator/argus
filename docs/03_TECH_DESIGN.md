# Argus — Technical Design Document

*Real-time camera-only sensing pipeline — Phase 0 & Phase 1 implementation design.*

**Status:** Approved for implementation · **Date:** 2026-06-24
**Audience:** Engineering team (implementers)
**Companion docs:** [01_ARCHITECTURE_DECISIONS.md](01_ARCHITECTURE_DECISIONS.md) · [02_PRD.md](02_PRD.md) · [04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md)

This document is the build spec. It assumes the decisions in the ADR are locked and tells you
*how* to assemble them: module layout, data contracts, threading model, per-extractor
algorithms, the bus, the validation harness, and the test plan. Where a number is a tunable,
it is marked `[tunable]` with a sensible default.

---

## 1. System overview

```
                                  ARGUS PIPELINE (single process)
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│  [Capture thread]                                                                       │
│   webcam @30fps, buffersize=1, stamp ts=local_clock() at grab                           │
│        │ writes                                                                          │
│        ▼                                                                                 │
│   LatestFrameSlot (1-slot, drop-oldest, .copy() on read)                                 │
│        │                                                                                 │
│        ▼                                                                                 │
│  [Backbone thread]  MediaPipe FaceLandmarker + PoseLandmarker (separate Tasks)          │
│   builds FrameContext{frame, ts, face_landmarks, iris, blendshapes, head_pose,          │
│                       pose_landmarks(One-Euro)}                                          │
│        │ fan-out (each extractor pulls from its own queue / ring)                        │
│        ├───────────────┬───────────────┬──────────────┬──────────────┬──────────────┐   │
│        ▼               ▼               ▼              ▼              ▼              ▼   │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │
│   │ rPPG    │    │ Resp     │    │ Blink/   │   │ Gaze    │   │ Affect  │   │ Motion  │ │
│   │ (POS)   │    │ (chest)  │    │ PERCLOS  │   │ (zones) │   │ (live)  │   │ +Pose   │ │
│   │ ring-buf│    │ ring-buf │    │ EAR      │   │ L2CS    │   │ HSEmo   │   │ fidget  │ │
│   │ →BVP→HR │    │ →RR      │   │ adaptive │   │ +iris   │   │ +blend  │   │ +gate   │ │
│   │ →HRV    │    │          │   │          │   │         │   │ feat    │   │ inputs  │ │
│   └────┬────┘    └────┬─────┘   └────┬─────┘   └───┬─────┘   └───┬─────┘   └────┬────┘ │
│        │  each emits SignalRecord{name,value,sqi,ts,meta}        │              │       │
│        └───────────────┴───────────────┴────────────┴───────────┴──────────────┘       │
│                                     │                                                    │
│                                     ▼                                                    │
│                        [Quality/Fusion stage]                                            │
│              motion gate (traffic light) + covariates + SQI weighting                    │
│                                     │                                                    │
│                                     ▼                                                    │
│                        [I/O edge — asyncio]                                              │
│                LSL StreamOutlets (one per signal)  ──────────► LabRecorder → XDF         │
│                          │                                                               │
│                          └── LSL→OSC / LSL→WebSocket bridge ──► art / biofeedback canvas │
│                                                                                          │
│  [Async BLE thread]  Polar H10 (bleak) → RR(1/1024s) → beat-times(cumsum) → LSL outlet   │
│  [Dashboard]  subscribes to LSL inlets, renders value+SQI+traffic-light                  │
└──────────────────────────────────────────────────────────────────────────────────────-┘

   Decoupled / offline workers (not in the 30fps loop):
   • Py-Feat AU extractor (5–15 fps worker or post-hoc on logged frames)
   • [QUARANTINED] OpenFace3/LibreFace reference (research flag, never imported by product path)
   • rPPG-Toolbox offline reference (RhythmFormer/RhythmMamba) for batch re-analysis
   • Validation report generator (reads XDF)
```

**Design tenets:** one camera owner; one backbone pass per frame; extractors are decoupled
plugins emitting a uniform record; physiology is computed on **sliding windows** not per
frame; the bus is the only contract between sensing and consumers; everything carries a
timestamp from *capture* and an SQI.

---

## 2. Repository & module layout

```
argus/
├── argus/
│   ├── capture/
│   │   ├── camera.py            # CaptureThread, LatestFrameSlot, ts stamping
│   │   └── calibration.py       # LED-flash exposure→host latency calibration
│   ├── backbone/
│   │   ├── face.py              # MediaPipe FaceLandmarker wrapper
│   │   ├── pose.py              # MediaPipe PoseLandmarker wrapper + One-Euro
│   │   └── context.py           # FrameContext dataclass
│   ├── extractors/
│   │   ├── base.py              # Extractor ABC, registry, SignalRecord
│   │   ├── rppg.py              # POS/CHROM → BVP → HR
│   │   ├── hrv.py               # upsample → peaks → IBI → SDNN/RMSSD (NeuroKit2)
│   │   ├── respiration.py       # chest/shoulder displacement → RR
│   │   ├── blink.py             # adaptive EAR, PERCLOS
│   │   ├── gaze.py              # iris front-end + L2CS/MobileGaze → zones
│   │   ├── affect_live.py       # blendshapes + HSEmotion (emotion + V/A)
│   │   ├── affect_au.py         # LibreFace/OpenFace 3.0 → 0–5 AU intensities (decoupled, rev 2)
│   │   └── motion.py            # head/posture/fidget from pose
│   ├── quality/
│   │   ├── sqi.py               # De Haan SNR, skewness, Orphanidou bSQI
│   │   ├── motion_gate.py       # FM/FSM + solvePnP + SNR → traffic light (hysteresis)
│   │   └── covariates.py        # skin-tone, lighting proxy, presence
│   ├── bus/
│   │   ├── lsl_out.py           # StreamOutlet management
│   │   ├── bridge_osc.py        # LSL→OSC
│   │   └── bridge_ws.py         # LSL→WebSocket
│   ├── groundtruth/
│   │   ├── polar_h10.py         # bleak + RR parsing + beat-time reconstruction → LSL
│   │   └── cms50e.py            # optional pulse-ox serial reader
│   ├── dashboard/
│   │   └── app.py               # live dashboard (LSL inlets)
│   ├── validation/
│   │   ├── protocol.py          # scripted block runner + XDF markers
│   │   └── report.py            # agreement stats, BA/CCC/MAPE, HRV band, EC13/CTA-2065
│   ├── offline/
│   │   └── rppg_toolbox_ref.py  # batch RhythmFormer/RhythmMamba re-analysis
│   ├── config.py                # typed config (pydantic), all [tunable]s
│   └── app.py                   # orchestrator: wires threads, backbone, extractors, bus
├── tests/
├── docs/
├── pyproject.toml               # pinned deps (rev 2: no research extra / firewall)
└── README.md
```

**Licensing firewall: REMOVED (rev 2).** Productification is out of scope, so there is no
quarantine, no `argus/research/` firewall, and no import-lint. Fold the AU extractor
(**LibreFace / OpenFace 3.0**, ADR-12) into `extractors/affect_au.py` as an ordinary decoupled
extractor. The `argus/research/` package and the `[research]` extra are deleted. *(No
`resp_belt.py` either — respiration is Indicative, motion-only, no belt, rev 2.)*

---

## 2a. Platform prerequisites & dependency resolution (rev 1)

**macOS permissions (TCC) — Spike-0, blocks first run (B5).** On macOS both **Camera** and
**Bluetooth** access are TCC-gated. A Python process launched from a terminal inherits the
**launching app's** grants, so the terminal (e.g. Terminal.app / iTerm / the IDE) must be
granted Camera **and** Bluetooth under System Settings → Privacy & Security. Symptoms of a
missing grant are silent: a **black/all-zero camera frame** or a **BLE scan that returns no
devices**, with no exception. **Mitigation:** a startup self-check (`capture/calibration.py`,
`groundtruth/polar_h10.py`) that fails loudly with remediation text if the first N frames are
all-black or the BLE adapter/scan is empty. Document the grant steps in the README.

**Dependency coexistence — Spike-0, blocks `pyproject.toml` (B2).** MediaPipe, ONNXRuntime, and
the AU model (LibreFace/OpenFace 3.0, PyTorch) have historically conflicting `numpy`/`protobuf`
pins. Resolved policy (rev 2 — licensing is no longer a factor, but the *runtime* isolation still
helps):
- **Python 3.11**, **`numpy<2`** (MediaPipe constraint), **CPU-only torch**.
- **Main venv** = MediaPipe + OpenCV + ONNXRuntime (HSEmotion, gaze) + NeuroKit2 + pyVHR +
  pylsl/bus + bleak.
- **The AU extractor (LibreFace/OpenFace 3.0) runs decoupled** — prefer its **ONNX export**
  (LibreFace ships one) in the main venv, or, if using the PyTorch build, an **isolated
  subprocess/venv** via the §4 escape hatch / post-hoc on logged frames. Keeps the live path light.
- The exact pinned matrix is produced by Spike-0 and frozen in `pyproject.toml` before feature
  work begins.

---

## 3. Core data contracts

### 3.1 FrameContext (produced once per frame by the backbone)
```python
@dataclass(frozen=True)
class FrameContext:
    frame: np.ndarray            # BGR HxWx3; a .copy() owned by this context
    ts: float                    # capture time, pylsl.local_clock() seconds
    frame_id: int
    face: FaceResult | None      # 478 landmarks, 10 iris, 52 blendshapes, 4x4 head_pose
    pose: PoseResult | None      # 33 world+image landmarks (One-Euro filtered), visibility
    img_meta: ImgMeta            # width, height, mean luma (relative brightness index)
```
**`frame` is read-only (rev 1).** `frozen=True` only prevents field rebinding — it does NOT
make the NumPy buffer immutable, and one `FrameContext` is fanned out to N extractors sharing
the same buffer. On construction, set `frame.flags.writeable = False`. **No extractor may write
to `ctx.frame`;** any extractor needing to modify pixels must `.copy()` first. This prevents a
cross-extractor data race.

### 3.2 SignalRecord (the uniform output of every extractor)
```python
@dataclass(frozen=True)
class SignalRecord:
    name: str                    # e.g. "hr", "hrv_sdnn", "resp_rate", "blink_rate",
                                 #      "gaze_zone", "affect_valence", "au12", "fidget_index"
    value: float | str | dict    # scalar, label, or small struct (e.g. {x,y} or class probs)
    sqi: float                   # 0..1 normalized quality/confidence
    ts: float                    # capture time the value pertains to (window-center for DSP)
    gate: Literal["good","usable","reject","unknown"]  # "unknown" = Phase-0 stub gate (rev 1)
    meta: dict                   # window_s, method, units, flags (e.g. {"rest_only": True})
```

**Bus mapping:** each `name` is one LSL stream. Regular-rate signals (hr ~1 Hz, resp ~0.2 Hz,
affect ~**10–15 Hz tunable**) declare `nominal_srate`; event signals (blink onsets, gaze-zone
changes) use `IRREGULAR_RATE`. Channel layout per stream: `[value(s)..., sqi, gate_code]`,
where **`gate_code`: good=0, usable=1, reject=2, unknown=3** (rev 1 — LSL channels are
numeric). `meta` is written into the stream's XML description (units, method, window).

### 3.3 Extractor interface
```python
class Extractor(ABC):
    name: str
    update_hz: float             # cadence; backbone calls consume() but extractor may buffer
    @abstractmethod
    def consume(self, ctx: FrameContext) -> list[SignalRecord]: ...
    # owns its own ring/window; must NOT block; returns [] when no new window is ready
```
Extractors self-register via a decorator; `app.py` discovers them. Adding an extractor never
touches backbone or bus (NFR-7).

---

## 4. Threading & buffering model (ADR-19)

- **Capture thread:** tight loop `grab()→read()→stamp→LatestFrameSlot.set(copy)`. Never
  blocks on consumers.
- **Backbone thread:** reads latest frame, runs FaceLandmarker + PoseLandmarker, builds
  `FrameContext`, pushes to (a) each per-frame extractor's **drop-oldest 1-slot queue** and
  (b) the **lossless ring buffer** that rPPG/HRV/respiration sample RGB traces from.
- **Extractor threads:** one per heavy extractor, or a small thread pool for light ones.
  Each maintains its own window/ring and emits on its `update_hz`.
- **I/O edge:** an asyncio loop owns LSL outlets + OSC/WS bridges; extractor threads hand
  records to it via a thread-safe queue.
- **BLE thread:** dedicated asyncio loop for `bleak` (Polar).
- **Two buffers, explicitly:**
  - `LatestFrameSlot` → freshness for affect/gaze/blink/motion/display (frame loss OK).
  - `RGBTraceRing` (lossless, evenly timestamped) → rPPG/HRV/respiration (sample loss NOT OK).
- **Escape hatch:** if Py-Feat (or any pure-Python-CPU extractor) can't keep cadence, move it
  to a `multiprocessing` worker fed frame *handles* via `shared_memory` (never pickle frames).
- **No asyncio on the CV hot path.** No reliance on free-threaded CPython (no OpenCV FT
  wheels mid-2026).

**Backpressure:** per-frame extractors that fall behind simply process the newest frame
(drop-oldest). Time-series extractors must keep up by construction (their per-frame work is
just appending an RGB mean to a ring; heavy DSP runs once per window).

---

## 5. Backbone (ADR-01, ADR-02)

- **FaceLandmarker:** `RunningMode.VIDEO` (or `LIVE_STREAM` with the result callback),
  `num_faces=1` (enables temporal smoothing), `output_face_blendshapes=True`,
  `output_facial_transformation_matrixes=True`. Feed monotonically increasing timestamps.
- **PoseLandmarker:** separate Task instance (do **not** use Holistic). Apply **One-Euro
  filter** `[tunable: min_cutoff=1.0, beta=0.007]` to each landmark before motion features.
- **rPPG ROI:** build from FaceMesh — forehead patch (index 151 region / glabella) + left
  cheek (50) + right cheek (280), excluding eyes/mouth/brows; multi-patch mean. Yaw-adaptive:
  drop a cheek patch when head yaw exceeds `[tunable: 25°]`. **Facial-hair fallback (rev 1):**
  if the recorded facial-hair covariate is set, **drop both cheek patches and use forehead/
  glabella only** (beard blocks cheek perfusion) — expect lower SNR and HRV yield. Absorb
  landmark jitter by patch averaging. (Reference indices per R2I-rPPG; pyVHR is the reference.)

---

## 6. Extractor algorithms

### 6.1 rPPG → HR (POS) — `extractors/rppg.py` (ADR-03)
1. Each frame: append spatial-mean RGB of the ROI to `RGBTraceRing` at the frame ts.
2. Every `[tunable: 1.0 s]`, take the last `[tunable: 10 s]` window (≥ 8 s), resample to even
   spacing, detrend, band-pass `[0.7–4.0 Hz]`.
3. Apply **POS** (Wang 2017) to the windowed RGB → BVP. (CHROM selectable via config.)
4. HR = PSD peak in `[42–240 bpm]`; emit `SignalRecord("hr", bpm, sqi=SNR_norm, ...)`.
5. SQI = normalized De Haan spectral SNR (§7). Gate state from motion gate (§8); **in Phase 0,
   before the gate exists, emit `gate="unknown"`** (the stub gate — M3/§3.2).
   *Library:* implement POS in NumPy (deterministic) or call pyVHR's POS; cross-check vs
   pyVHR in tests.
6. **Latency note (rev 1):** the HR value is timestamped at the **window center**; NFR-1
   latency (capture→emit ≤ 2 s) is the pipeline delay from a frame to the first emit whose
   window includes it — it does **not** include the ~½-window (~5 s) intrinsic smoothing lag
   for a *change* in true HR to appear. Do not conflate the two.

### 6.2 HRV (SDNN, RMSSD) — `extractors/hrv.py` (ADR-05, ADR-06)
1. From the BVP window, **cubic-spline upsample to ≥ 256 Hz** before peak detection (avoids
   33 ms IBI quantization).
2. Detect systolic peaks (derivative/template); compute IBIs.
3. **Per-beat Orphanidou bSQI** (template corr ≥ `[tunable: 0.86 — PROVISIONAL]`) — drop
   low-SQI beats. **The 0.86 threshold must be calibrated (rev 1):** it was derived for contact
   ECG/PPG; run a ROC on this subject's rPPG beats vs H10-confirmed beats and set the threshold
   from that, don't assume 0.86.
4. Correct artifacts with **NeuroKit2 `signal_fixpeaks(method="kubios")`**.
5. Compute SDNN (committed) + RMSSD (indicative) via NeuroKit2 over a `[tunable: 60–120 s]`
   window; ln-transform RMSSD for reporting. **The window length is fixed per session and the
   reference SDNN is computed over the identical time-aligned window** (SDNN grows with length).
6. **Emit policy (rev 1 — avoids the GOOD-deadlock M9):** compute HRV over the window when
   **≥ `[tunable: 80%]` of the window's beats are GOOD-gated AND bSQI-accepted** — do NOT
   require 100% continuous GOOD (a swallow/postural shift would otherwise starve HRV forever).
   Flag `meta={"rest_only": True, "good_fraction": f}`. If a rest block yields zero qualifying
   windows, the report states **"insufficient clean data"** (not a silent pass/fail).
   **Never compute LF/HF.**

### 6.3 Respiration — `extractors/respiration.py` (ADR-07)
1. Per frame: append vertical displacement of shoulder/upper-chest pose landmarks (filtered)
   to a ring at frame ts.
2. Every `[tunable: 1 s]`, take last `[tunable: 20 s]` (15–30 s), band-pass **`[0.08–0.5 Hz]`**
   (rev 1: low edge widened from 0.1 Hz so the 6-brpm = 0.1 Hz paced block isn't attenuated at
   the filter edge), FFT/peak-count → RR (brpm).
3. **Indicative — no belt (rev 2).** No respiratory belt is purchased, so there is no valid RR
   ground truth; the metronome (6/10/15 brpm) is a **coarse plausibility check only**. Report the
   estimate + plausibility; **make no MAE accuracy claim**.
4. Secondary: rPPG-derived RR (RSA/amplitude modulation via NeuroKit2 EDR) as cross-check;
   report agreement, prefer motion estimate.
5. SQI from band power ratio + pose-landmark visibility; gate on motion.

### 6.4 Blink / PERCLOS — `extractors/blink.py` (ADR-08)
1. Compute **EAR** from FaceMesh eye landmarks each frame.
2. **Per-session adaptive threshold:** auto-calibrate open-eye baseline in the first
   `[tunable: 10 s]`; threshold = `[tunable: 0.6 × baseline]`; blink = EAR < threshold for
   ≥ `[tunable: 2–3 frames]` (~250 ms at 30 fps). **Baseline guard (rev 1):** require a
   minimum count of valid open-eye frames before arming; if the face is absent or eyes closed
   during the baseline window, stay un-calibrated and flag (don't emit garbage thresholds).
   **Eyewear is a recorded covariate** — the F1 ≥ 0.90 bar applies to the no-glasses condition.
3. Emit blink **events** (IRREGULAR_RATE) with duration; **blink rate** as a windowed scalar.
4. **PERCLOS (P80)** over a `[tunable: 60–90 s]` window.
5. Requires 30 fps (assert; warn if effective fps < `[tunable: 25]`).
   *Optional Phase-1 upgrade (deferrable):* MobileNetV2 eye-state CNN for glasses/pose
   robustness.

### 6.5 Gaze (zones) — `extractors/gaze.py` (ADR-09)
1. Front-end: MediaPipe iris + head-pose for eye crops & presence.
2. Gaze vector: **default L2CS-Net (ResNet-18)** via ONNXRuntime (CPU); MobileGaze is a stretch
   alternative. **Not drop-in (rev 1):** the two differ in input preprocessing and output
   convention (pitch/yaw sign, degrees vs radians, gaze origin) — the zone-mapping layer is
   model-specific, so a config swap also swaps the mapping calibration.
3. Map vector → **zones**: horizontal {left, center, right}, screen {on, off}, attention
   {present, absent}. Emit zone + confidence; **do not** emit pixel coordinates. Validation
   reports a **3×3 confusion matrix + accuracy above 33% chance** at fixed target geometry.
4. Optional 5–9-point calibration routine writes a per-user affine correction (Phase-1
   upgrade); without it, expect ~6–10° → zones only.

### 6.6 Affect (live) — `extractors/affect_live.py` (ADR-11)
1. Blendshapes from backbone → **neutral-subtract + z-score per session** → a small set of
   interpretable cues (brow, smile, eye-widen). Coarse, low-latency.
2. **HSEmotion** (`enet_b0_8_va_mtl`, ONNX) on the aligned face crop → 8-class emotion +
   continuous valence/arousal at **`[tunable: 10–15 Hz]`** (rev 1: not a hard 30 Hz — it
   competes with the backbone for CPU). Emit both with confidence as SQI.
3. Frame as "estimate, confidence X" — never a verdict (PRD §5 principle). **Validation is a
   model face-validity check only** (posed ≠ felt affect — ADR-11), reported descriptively.

### 6.7 Facial AUs (research) — `extractors/affect_au.py` decoupled worker (ADR-12, rev 2)
- **LibreFace** (best 0–5 AU intensity, PCC 0.63) — or **OpenFace 3.0** — at `[tunable: 5–15
  fps]` on a decoupled worker, or **offline** on logged frames. License is not a constraint
  (rev 2), so use the best estimator directly; no quarantine.
- Emits 0–5 AU intensities on IRREGULAR_RATE streams; flagged `meta={"research": True}`.
- If the AU model pulls a heavy DL stack (PyTorch), run it via the §4 multiprocessing escape
  hatch so it doesn't drag the live path; ONNX export (LibreFace ships one) avoids this.

### 6.8 Head/posture motion & fidget — `extractors/motion.py` (ADR-02)
- Posture (lean, slump) from torso pose geometry; **fidget index** = aggregate landmark jitter
  energy over a `[tunable: 5 s]` window. **Use lightly-/pre-filtered landmarks for fidget
  (rev 1):** the aggressive One-Euro filter used for posture would suppress the very
  high-frequency jitter fidget measures. Descriptive/exploratory.

---

## 7. Signal Quality (ADR-06)

- **De Haan spectral SNR** (primary): power in the HR-fundamental band (±`[6 bpm]`) + first
  harmonic ÷ **out-of-band power only** — the signal bins are **excluded** from the denominator
  (rev 1; otherwise it is not the De Haan ratio) — over the band-pass support `[0.7–4 Hz]`.
  Normalize to 0..1 for `sqi`.
- **Skewness / perfusion index** (supplemental sanity).
- **Orphanidou bSQI** (HRV beat acceptance, §6.2).
- Low-SQI records are emitted **with a flag**, not dropped (consumers decide); HRV is the
  exception — low-bSQI beats are dropped before IBI computation.

---

## 8. Motion quality gate (ADR-14)

Inputs (reusing FaceMesh + head pose — no second tracker):
- `FM_X`, `FM_Y` = mean landmark displacement between frames.
- `FSM` = ROI-area change (≈ z-motion).
- Head pose via `cv2.solvePnP` on canonical landmarks; weight **pitch ×2** vs yaw.
- Pulse-side: De Haan SNR + skewness; HR plausibility `[42–240 bpm]`, jump limit.

Normalize motion metrics by **inter-ocular distance** (scale-invariance). Combine into a
3-tier light with **hysteresis** `[tunable: enter/exit thresholds, dwell ≥ 1 s]`:
- 🟢 **GOOD:** low motion AND SNR ≥ `[3 dB]` → HR + HRV + RR emitted.
- 🟡 **USABLE:** mild motion OR SNR `[0–3 dB]` → HR only (widened CI); suppress HRV/RR.
- 🔴 **REJECT:** high motion OR SNR < `[0 dB]` → drop window, hold last good value, dashboard
  shows "re-acquiring."

**HRV does not require 100% GOOD (rev 1):** per §6.2, HRV computes when ≥80% of a window's beats
are GOOD-gated/bSQI-accepted — the per-frame gate flickering to USABLE on a swallow must not
starve the 60–120 s HRV window forever.

The gate state rides on every `SignalRecord.gate` and is itself published as a stream. **Phase 0
ships an always-`"unknown"` stub gate** (the real gate is built in build-step 3); HR records in
Phase 0 carry `gate="unknown"`.

---

## 9. Covariate layer (ADR-18, PRD FR-15)

Published as their own LSL streams alongside signals:
- **Skin tone:** self-reported Monk/Fitzpatrick at session start (FR-23) + an automated
  cheek-reflectance estimate (**unvalidated/descriptive, n=1**). Recorded **for Phase-2
  forward-compat only** — at n=1 there is nothing to stratify (rev 1).
- **Eyewear (glasses)** and **facial hair:** self-reported at session start (rev 1). Eyewear
  gates blink/gaze interpretation; facial hair drives the cheek-ROI drop (§5).
- **Lighting:** a **relative brightness index** from mean frame luma (rev 1 — **uncalibrated,
  NOT lux**: luma depends on exposure/gain/white-balance, not scene illuminance) with under/
  over-exposure flags, **plus a measured lux value per block** entered from a lux meter/phone
  app (good light = ≥300 lux; lighting_A ≈ 150 lux; lighting_B ≈ 500 lux at the face).
- **Global motion:** the gate's normalized motion magnitude.
- **Face presence / tracking confidence.**

---

## 10. The bus (ADR-15)

- **LSL outlets** (`bus/lsl_out.py`): one `StreamOutlet` per signal & covariate; types/units
  in stream XML. Regular-rate streams set `nominal_srate`; events use `IRREGULAR_RATE`.
- **Recording:** **LabRecorder** captures all Argus streams + the Polar H10 stream into one
  **XDF** with clock-offset metadata (FR-17). A thin CLI wrapper starts/stops a labeled
  recording and injects protocol markers.
- **Art bridge** (`bus/bridge_osc.py`, `bridge_ws.py`): subscribe to selected LSL streams,
  re-emit as OSC addresses (e.g. `/argus/hr`, `/argus/valence`) or WebSocket JSON, with a
  small forward-sync look-ahead `[tunable: 50 ms]` for jitter-free visuals. **Canvas = TouchDesigner
  over OSC (rev 2 — default).** `bridge_osc.py` is the primary art leg; `bridge_ws.py` exists for
  a future p5.js/web-shareable consumer (flip via config, no redesign).

---

## 11. Ground truth & time sync (ADR-16, ADR-17)

- **Polar H10** (`groundtruth/polar_h10.py`): `bleak` notify on `0x2A37`; parse with the
  **full flag-driven offset algorithm (ADR-17 — bit0 16-bit HR, bit3 energy-expended, bit4 RR)**,
  RR in 1/1024 s; **iterate all RR per packet**; reconstruct beat times by cumulative sum from a
  single anchored `local_clock()`; publish HR + beat-times to LSL. macOS: discover by **name**
  (UUID not MAC); reconnect retries. **On BLE reconnect (rev 1): re-anchor the cumsum and insert
  an XDF discontinuity marker**; the report excludes the gap from yield (else beat times drift/
  jump across the gap and corrupt HRV). Kubios artifact correction before HRV stats.
- **Frame timestamps:** stamped at grab; constant exposure→host offset measured once via an
  **LED-flash calibration** (`capture/calibration.py`) and subtracted.
- **Offline alignment** (in `validation/report.py`): load XDF with `proc_clocksync` +
  `proc_dejitter`. **Estimate the residual lag on a motion/talking block where HR varies**
  (rev 1 — at rest HR is flat and cross-correlation degenerates): resample instantaneous-HR to
  4 Hz, **cross-correlate** for the lag. **For HR-rate** agreement, apply that lag. **For HRV
  timing**, apply a **single fixed median-PTT lag, not a per-segment re-optimized lag** (rev 1 —
  re-optimizing would absorb the very IBI jitter HRV measures); then beat-match within
  `[±50–100 ms]` and report matched-beat yield. DTW only as a reported metric, never the aligner.

---

## 12. Validation harness (ADR-18)

- **Protocol runner** (`validation/protocol.py`): scripted blocks with on-screen prompts and
  XDF markers — `rest` ×2, `paced_breathing` (6/10/15 brpm metronome, **belt-referenced**),
  `light_motion` (talk/head-turn), `lighting_A` (~150 lux) / `lighting_B` (~500 lux), each with
  a **measured lux** entry (free phone app, rev 2), plus optional `gaze_targets` (fixed geometry)
  and `eye_closure` (commanded closure-fraction) blocks. ≥ 2 repeats each. **HRV stats taken from
  `rest` blocks only; the 6-brpm block is excluded from all HRV stats** (0.1 Hz resonance inflates
  SDNN/RMSSD). **Respiration is Indicative (rev 2, no belt)** — the paced blocks are a plausibility
  check, not an accuracy measurement.
- **Report generator** (`validation/report.py`): per condition & signal →
  Bland-Altman (bias + 95% LoA), MAE, RMSE, MAPE, Pearson r, **Lin's CCC**, SNR; HR also
  reported at a 60 s average **and** a 4–10 s quasi-instantaneous window; **HR pass/fail vs the
  EC13 + CTA-2065 *numeric thresholds* (feasibility, not conformance)**. HRV (rev 1): **SDNN BA
  in ms vs ±15 ms band + SDNN MAE ≤ 12 ms bar + Lin's CCC, over length-matched windows**;
  **ln-RMSSD BA in log-units/ratio (NOT ms, no band), indicative only**; IBI/matched-beat yield
  + SNR. The H10 is reported as a **reference (its own seated RMSSD bias is included in the
  reported bias, not attributed to the camera)**. Gaze → 3×3 confusion matrix vs chance. Affect
  → descriptive face-validity summary (no pass/fail). Output: a single HTML/PDF report.
  **Every report carries the feasibility banner:** *"Single-subject results are
  hypothesis-generating; they establish neither limits of agreement nor fairness. If the
  subject is Fitzpatrick V–VI, treat rPPG bars as near-worst-case; if I–III, near-best-case."*

---

## 13. Configuration

All `[tunable]`s live in a typed `config.py` (pydantic) with a shipped default profile and a
`validation` profile. Key knobs: camera (fps, resolution), rPPG (window_s, update_hz, method
POS/CHROM, bands, ROI indices, yaw cutoff), HRV (upsample_hz, bSQI threshold, window_s),
respiration (window_s, band), blink (baseline_s, threshold_ratio, min_frames, perclos_window),
gaze (model, calibration_points), motion-gate (enter/exit thresholds, pitch weight, dwell),
bus (OSC/WS targets, look-ahead), validation (block durations, repeats).

---

## 14. Test plan

| Layer | Test | Method |
|---|---|---|
| Capture | Frame ts monotonic; drop-oldest works under induced lag | Unit + injected sleep |
| Backbone | One pass/frame; FrameContext fields populated; 30 fps sustained | Bench on target laptop |
| rPPG/POS | POS output matches pyVHR on a **synthetic** RGB trace with an injected sinusoid | Golden-file unit test (CI) |
| HRV | SDNN/RMSSD vs NeuroKit2 on a synthetic BVP with known IBIs | Unit (CI) |
| Resp | RR **plausibility** vs metronome (Indicative; no accuracy claim, rev 2) | Protocol session (HIL) |
| Blink | F1 ≥ 0.90 vs frame-level manual annotation (±N-frame match) | Labeled clip (HIL/local) |
| Gaze | 3×3 confusion matrix vs scripted look-targets (fixed geometry) | Protocol session (HIL) |
| Motion gate | REJECT triggers under induced head motion; hysteresis (no flicker) | Scripted motion (HIL) |
| Sync | Cross-correlation lag < `[tunable]`; XDF clock metadata present | Validation session (HIL) |
| Bus | XDF replays losslessly; OSC/WS consumer receives expected rates | Integration (HIL) |
| ~~Licensing~~ | *Removed (rev 2 — no licensing firewall)* | — |
| End-to-end | One command → session → report with full stat set | Smoke + report diff (HIL) |

**Acceptance-test traceability:** each test maps to an FR/acceptance criterion in
[04_FEATURES_ACCEPTANCE.md](04_FEATURES_ACCEPTANCE.md).

## 14a. CI vs hardware-in-the-loop, and test fixtures (rev 1)

A headless CI runner has no camera, no Polar H10, and no target-laptop GPU, so the §14 matrix
splits in two:
- **CI (headless, every PR):** POS on a **synthetic RGB trace** (injected known sinusoid — no
  real face → avoids privacy concerns); HRV on a **synthetic BVP** with known IBIs; all pure unit
  tests; config validation. Deterministic and fixture-free (fixtures generated in-test). *(rev 2:
  the licensing import-lint is removed — no firewall.)*
- **Hardware-in-the-loop (self-hosted runner / local, gated runs):** throughput/30 fps bench,
  camera/backbone, BLE/Polar, motion gate, sync, protocol sessions, end-to-end report.
- **Fixtures with faces are never committed to the repo** (privacy, NFR-5): the blink-annotation
  clip + its annotation JSON live in **git-LFS / out-of-band storage with restricted access**,
  not CI history. Licensed rPPG datasets (UBFC/PURE/MMPD) stay out of the repo entirely; they
  are used only on the developer's machine for the offline rPPG-Toolbox reference.

---

## 15. Build order (implementation sequence)

0. **Spike-0 (before any feature work):** resolve the dependency matrix (§2a / B2) and the
   macOS Camera+Bluetooth permission path (§2a / B5). Output: a frozen `pyproject.toml` and a
   working camera+BLE self-check. Nothing below proceeds until Spike-0 is green.
1. **Skeleton + contracts:** `FrameContext` (read-only frame), `SignalRecord` (gate incl.
   `"unknown"`), `Extractor` ABC (`-> list`), config, capture thread + LatestFrameSlot, LSL
   outlet, null dashboard; `argus fetch-models` (§17a).
2. **Phase 0 vertical slice:** FaceLandmarker backbone → rPPG ROI → POS HR → SNR SQI → LSL →
   minimal dashboard; Polar H10 ingestion + XDF recording; first BA/MAE report. *Hit Phase 0
   exit criteria before widening.*
3. **Motion gate + covariates** (so every later signal is born gated).
4. **HRV** (upsample → bSQI → NeuroKit2), rest-only.
5. **Respiration** (pose-primary + rPPG cross-check).
6. **Blink/PERCLOS**, then **gaze zones**, then **live affect** (HSEmotion + blendshapes).
7. **Pose motion/fidget**; **Py-Feat AU worker** (decoupled); optional quarantined AU ref.
8. **Full bus** (OSC/WS bridge), **richer dashboard**, **validation harness** complete.
9. **Offline rPPG-Toolbox reference** for batch re-analysis.
10. Harden: tests, latency/throughput benchmarks, licensing CI gate, docs.

---

## 16. Known limitations carried into Phase 0/1 (state them in the UI/reports)

- rPPG/HRV degrade with motion, lighting, and (later) skin tone; HRV is rest-only, RMSSD
  indicative, no LF/HF.
- **Cheek rPPG ROI assumes hair-free skin** — beard/stubble forces forehead-only and lowers
  HRV yield (recorded covariate).
- Gaze is zone-level, not pixel-accurate, for uncalibrated users; reported as a confusion
  matrix above 33% chance, not a single accuracy number.
- Affect/AUs are estimates, not ground-truth emotion; what we run is a **model face-validity
  check on posed expressions** (posed ≠ felt affect), single-subject only.
- **Respiration is "Committed" only with a respiratory belt** as reference; against a metronome
  alone it is Indicative (metronome measures compliance, not breathing).
- **The H10 is a reference, not gold truth** (its own seated RMSSD bias is in the reported bias).
- Single-subject results are feasibility, not accuracy/fairness clearance; **a dark-skinned sole
  subject could fail/barely-pass rPPG bars for skin-tone reasons that won't surface at n=1**.
- Latency "feels alive" threshold for biofeedback is an open tuning question (NFR-1).

---

## 17a. Model assets & offline operation (rev 1)

First-run model downloads are permitted (NFR-5 "no cloud" refers to runtime data, not setup):
- Assets: MediaPipe `.task` files (face, pose), HSEmotion ONNX, L2CS-Net/MobileGaze ONNX,
  Py-Feat weights. Each is **pinned by URL + SHA-256 checksum** in a manifest.
- `argus fetch-models` downloads, verifies checksums, and caches under a configurable
  `ARGUS_MODEL_DIR`. After fetch, the system runs **fully offline**; CI uses cached/vendored
  small models or skips model-dependent tests (§14a).
- Each asset's license is recorded in the manifest; the licensing lint (§14a) also checks the
  manifest contains no non-commercial asset in the product set.

---

## 18. Failure-mode handling (rev 1 — explicit specs)

| Failure | Detection | Behavior |
|---|---|---|
| **Camera disconnect/unplug** | `read()` returns False repeatedly | Pipeline emits a `source_lost` state; gate → `reject`; XDF recording continues with a **gap marker**; dashboard shows "camera disconnected"; auto-resume on reconnect. |
| **No face in frame** | `FrameContext.face is None` | Face extractors emit nothing (no stale value); presence covariate = 0; dashboard shows "no face." |
| **Low light / over-exposure** | brightness index + exposure flags | Covariate flag raised; SQI drops naturally; report annotates the affected blocks. |
| **BLE dropout (Polar)** | notify stops / disconnect callback | Auto-reconnect with retries; **re-anchor cumsum + XDF discontinuity marker** (§11); HRV yield excludes the gap. |
| **Blink baseline invalid** | too few valid open-eye frames in baseline window | Stay un-calibrated, flag; do not emit blink metrics until armed. |
| **HRV starved (no GOOD windows)** | zero qualifying windows in a rest block | Report "insufficient clean data" for HRV (not a silent pass/fail). |
| **Effective fps < 25** | frame-rate monitor | Warn; mark blink/PERCLOS results invalid (30 fps required). |
