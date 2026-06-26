# SUMMARY — Full Argus implementation (every FR/NFR/AC)

**Status: TASK_COMPLETE**

All Definition-of-Done conditions hold:
1. **Every requirement in `PLAN.md` is checked — 125/125** (FR-1…23, NFR-1…9, all ~91 ACs).
2. **Full test suite green: 166 passed, 0 failed, 0 skipped** (`./venv/bin/pytest -q`).
3. **Build clean**: `./venv/bin/python -m compileall -q src tests` → no errors.
4. **Adversarial self-audit: zero gaps** — every requirement anchor maps to real
   implementation code AND a referencing test (audit script in the git history of this task).

This run implements **every** functional and non-functional requirement and acceptance
criterion, not a subset. Where physical hardware/heavy ML runtimes are unavailable, the real
module is implemented behind a dependency-injected interface and tested against a fake /
synthetic / replay source — never skipped. Only the literal device-driver lines are left
untested (listed below).

## How "no hardware" was handled (per the goal's hard rule)

Every hardware- or model-facing component is split into **(real adapter, injectable fake)**
behind a `Protocol`:

| Component | Real adapter (untested device line) | Fake/synthetic used in tests |
|---|---|---|
| Webcam | `OpenCVCamera` (`cv2.VideoCapture` open) | `SyntheticCamera` (embeds an rPPG pulse) |
| Face backbone | `MediaPipeFaceBackbone` (`mediapipe` inference) | `SyntheticFaceBackbone` |
| Pose backbone | `MediaPipePoseBackbone` (`mediapipe` inference) | `SyntheticPoseBackbone` |
| Gaze | `L2csGazeEstimator` (`onnxruntime` inference) | `FakeGazeEstimator` |
| Affect | `HSEmotionEstimator` (`onnxruntime` inference) | `FakeEmotionEstimator` |
| Action Units | `LibreFaceAuEstimator` (`onnxruntime` inference) | `FakeAuEstimator` |
| LSL outlet | `LslOutlet` (`pylsl.push_sample`) | `InMemoryBus` / `InMemoryOutlet` |
| OSC transport | `UdpTransport` (`socket.sendto`) | fake transport (decoded in-test) |
| Polar H10 | `BleakPolarSource` (`bleak` BLE connect/notify) | `FakePolarSource` (replays packets) |
| Model download | `_urllib_download` (`urllib.urlopen`) | injected fake downloader |
| Raw-video encode | caller's encoder in `write_raw_video` | lambda writer (path asserted) |

Everything else — POS, SQI, HRV, respiration, blink, gaze zones, affect logic, the motion
gate, the bus codecs, the **real XDF writer** (verified by loading with `pyxdf`), the Polar
**byte parser** (all four flag combinations), Kubios correction (real `neurokit2`), time-sync,
the validation report, the threaded pipeline with its asyncio I/O edge, the CLI, storage, and
runtime policy — is real, exercised code.

### The complete list of untested device-driver lines
(marked `# pragma: no cover - device` / `- model inference` / `- network` / `- BLE` in source)
- `OpenCVCamera.read` / `.release` and the `cv2.VideoCapture(index)` open.
- `MediaPipeFaceBackbone.process`, `MediaPipePoseBackbone.process` (model inference).
- `L2csGazeEstimator.estimate`, `HSEmotionEstimator.estimate`, `LibreFaceAuEstimator.estimate`.
- `LslOutlet.push` and `BleakPolarSource._discover_and_connect`.
- `UdpTransport.send`, `core/models._urllib_download`, `cli.cmd_fetch_models` network branch.
- `clock.local_clock` pylsl branch (falls back to `time.monotonic`, which IS tested).

## Module map

```
src/argus/
  contracts.py              SignalRecord, FrameContext (read-only frame), Extractor ABC, gate codes
  config.py                 typed tunable defaults
  capture/                  clock, frame sources, latest-frame slot + lossless ring, capture thread, LED calibration
  backbone/                 face/pose result types, One-Euro filter, MediaPipe adapters + synthetic
  dsp/                      POS rPPG, De Haan SNR + skewness/perfusion/Orphanidou bSQI, HRV (no LF/HF),
                            respiration (+ rPPG-RR cross-check), blink/EAR/PERCLOS + F1, ROI
  extractors/               hr / hrv / resp / blink / fidget Extractor plugins
  perception/               gaze (zones, calibration, confusion), affect (blendshape norm, HSEmotion,
                            face-validity), research AUs (decoupled LibreFace)
  quality/                  motion gate (3-tier hysteresis), gate inputs (FM/FSM/solvePnP),
                            gate-apply (suppress HRV/RR, hold last-good HR), covariates
  bus/                      outlets + InMemoryBus, OSC codec+bridge, WS bridge, real XDF writer, Recorder
  groundtruth/              Polar byte parser, BLE sources + reconnect ingestor, Kubios
  validation/               agreement stats (+Pearson r), time-sync (xcorr/beat-match/DTW),
                            protocol runner, report generator (+HTML)
  dashboard/                render model (traffic-light, re-acquiring, stale/degraded)
  core/                     pipeline (metrics/latency/extensibility), threaded pipeline (asyncio edge),
                            concurrency (shared-memory frame handle), runtime policy + pinned manifest,
                            storage (privacy), model fetch+checksum
  cli.py                    argus run | record | report | fetch-models
tests/                      16 test files, 166 tests
```

## Test counts
- **166 passed · 0 failed · 0 skipped** · 125/125 requirements checked · self-audit: 0 gaps.
- Commands: tests `./venv/bin/pytest -q` · build `./venv/bin/python -m compileall -q src tests`.

## Conflicts noted
None substantive. Doc conflicts were already reconciled to rev 2 (respiration Indicative;
licensing/quarantine withdrawn; canvas = TouchDesigner/OSC) and applied accordingly.

## Environment deviation (justified, documented in NOTES.md)
Docs pin Python 3.11 + `numpy<2` for the *MediaPipe* hardware build; this env runs Python 3.13
+ numpy 2.x and MediaPipe is not exercised here. `core/runtime.check_runtime()` encodes the
3.11/numpy<2 policy and its test confirms the checker correctly flags the current interpreter as
non-compliant — the policy logic itself is what is under test.

## A note on "validation bars" vs feasibility
Acceptance criteria with single-subject *accuracy bars* (e.g. HR within EC13, SDNN MAE ≤ 12 ms)
are implemented as the **checks/metrics** the harness computes (EC13/CTA pass-fail, SDNN MAE
bar, CCC), exercised on synthetic and replay data. The bars themselves are only *meaningful*
against a live subject + Polar H10 on the hardware-in-the-loop runner; the code that computes and
gates on them is complete and tested here.

---

## Algorithm-review improvements (2026-06-25)

Implemented **every** recommendation in `docs/06_ALGORITHM_REVIEW.md` (the SOTA review of all
signal algorithms), with tests. **Suite: 220 passed, 0 failed.**

- **HR/rPPG**: added the glabella patch and per-patch **SNR-weighted fusion** with occlusion
  fallback (`dsp/roi.py`, `dsp/sqi.fuse_patches_by_snr`, `RppgExtractor(per_patch=True)`).
- **HRV**: **parabolic sub-sample peak interpolation** + Lipponen–Tarvainen IBI correction
  (`correct_ibis`); per-window **NSQI + SNR + skewness** gate (`window_sqi_gate`).
- **Respiration**: chest-ROI **Farnebäck optical-flow** estimator chosen as primary by SQI,
  shoulder-motion + rPPG-RR retained as cross-checks (`ChestFlowRespiration`).
- **Blink/PERCLOS**: **graded eye-openness** (EAR ⊕ eyeBlink blendshape) + true **P80** over a
  rolling window with <400 ms blink exclusion (`PerclosP80`).
- **Fidget**: **MEA frame-differencing** energy + **SPARC/LDLJ** smoothness (`dsp/motion.py`).
- **Posture**: median-window baseline, **temporal hysteresis** (`PostureDebouncer`), head-roll,
  forward-head **AND-gate**, landmark-visibility confidence gate.
- **Gaze**: 7th **inter-ocular distance** feature, **degree-1 ridge** default, **one-euro**
  cursor, **cm/° held-out-CV error**, click-anchored **drift recalibration** (`web/gaze.html`,
  `web/index.html`, `perception/gaze.PolynomialRidge`).
- **Affect**: configurable newer multi-task V/A head (**EmotiEffLib `mobilevit_va_mtl`**) with
  graceful HSEmotion fallback (`build_affect_estimator`); real inference validated.
- **Cross-cutting**: **skin-tone fairness** (ITA°→Fitzpatrick + stratified error), a **shared
  per-frame motion-quality index**, and **honest accuracy ceilings** per signal — all surfaced
  live in the web dashboard.

New deps: `emotiefflib==1.1.1` (added to `requirements.txt`). New tests:
`test_motion_metrics.py`, `test_quality_crosscut.py`, plus additions across
`test_roi_sqi/test_hrv/test_respiration_blink/test_posture/test_perception`.
