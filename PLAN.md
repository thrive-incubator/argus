# PLAN — Full PRD/FEATURES Requirement Checklist

Derived **directly** from `docs/02_PRD.md` (FR-1..FR-23, NFR-1..NFR-9) and
`docs/04_FEATURES_ACCEPTANCE.md` (every acceptance criterion). One checkbox per FR, per
NFR, and per AC. Checked **only** when real implementation code exists AND a test
exercises it (fakes/synthetic sources where hardware is absent). See SUMMARY.md for the
explicit list of untested device-driver lines.

## Functional requirements (PRD §6)
- [ ] FR-1  Webcam capture ≥30fps/720p, CAP_PROP_BUFFERSIZE=1, frame ts via local_clock at grab
- [ ] FR-2  One MediaPipe Face Landmarker pass/frame → FrameContext (landmarks/iris/blendshapes/head-pose/ts)
- [ ] FR-3  One MediaPipe Pose pass/frame (separate Task), One-Euro filtered
- [ ] FR-4  Real-time throughput; drop frames for per-frame extractors, never drop lossless time-series ring
- [ ] FR-5  HR via POS over rolling 8–15 s window, ~1 Hz update
- [ ] FR-6  HRV (SDNN,RMSSD): upsample ≥256Hz → peaks → IBI → NeuroKit2, bSQI-gated, rest-only flag
- [ ] FR-7  Respiration: chest/shoulder displacement band-pass 0.08–0.5Hz, 15–30s; rPPG-RR cross-check
- [ ] FR-8  Blink (rate,duration), PERCLOS via adaptive-threshold EAR
- [ ] FR-9  Gaze zone (L/C/R, screen on/off, attention) via learned head + iris front-end; optional calibration
- [ ] FR-10 Live affect: blendshapes + HSEmotion emotion + valence/arousal
- [ ] FR-11 Research AUs: decoupled 5–15fps (LibreFace/OpenFace3, rev2) or offline
- [ ] FR-12 Head/posture motion + fidget index from BlazePose
- [ ] FR-13 Per-window SQI (De Haan SNR + skewness; Orphanidou bSQI for HRV beats)
- [ ] FR-14 Motion quality gate: 3-tier hysteretic GOOD/USABLE/REJECT from landmark motion + head pose + SNR
- [ ] FR-15 Publish covariates (skin-tone, lighting, global motion, face presence) as streams
- [ ] FR-16 Publish every signal/covariate on an LSL StreamOutlet (nominal_srate / IRREGULAR_RATE)
- [ ] FR-17 Record all LSL streams + Polar to one synchronized XDF with clock-offset metadata
- [ ] FR-18 Bridge LSL→OSC (and LSL→WebSocket) with forward-sync look-ahead
- [ ] FR-19 Live dashboard: each signal value+SQI+traffic-light; P0 HR-only, P1 full
- [ ] FR-20 Ingest Polar H10 over BLE 0x2A37, parse RR (1/1024s), beat-time cumsum, Kubios correction
- [ ] FR-21 Validation protocol runner: scripted blocks + XDF markers
- [ ] FR-22 Validation report generator: BA, MAE, RMSE, MAPE, Pearson r, CCC, SNR; SDNN/ln-RMSSD; EC13/CTA
- [ ] FR-23 Record subject Monk/Fitzpatrick skin tone per session

## Non-functional requirements (PRD §7)
- [ ] NFR-1 Latency = capture→first-emit ≤2s (HR); display path ≤150ms; excludes intrinsic window lag
- [ ] NFR-2 Throughput: sustain ≥30fps backbone on CPU
- [ ] NFR-3 Platforms: macOS + Linux; Windows unsupported
- [ ] NFR-4 Reproducibility: pinned deps; deterministic POS; one-command session + report
- [ ] NFR-5 Privacy: derived+XDF by default, no raw video unless opt-in flag, never committed
- [ ] NFR-6 Licensing: WITHDRAWN (rev2) — assert no firewall/quarantine machinery exists
- [ ] NFR-7 Extensibility: new Extractor self-registers, no backbone/bus change needed
- [ ] NFR-8 Observability: per-extractor timing/health + dropped-frame counters + bus rates exposed
- [ ] NFR-9 Honesty/UX: degraded/rejected signals visibly marked + "re-acquiring", never silently frozen

## Acceptance criteria (FEATURES)
### A — Capture & Backbone
- [ ] A1.AC1 frames ≥30fps/≥720p with buffersize=1
- [ ] A1.AC2 frame ts from local_clock at grab; strictly monotonic
- [ ] A1.AC3 latest-frame slot drops older frames; no unbounded queue; capture never blocks
- [ ] A1.AC4 exposure→host LED-flash calibration → constant offset recorded + subtracted
- [ ] A2.AC1 one FaceLandmarker pass/frame → 478 landmarks,10 iris,52 blendshapes,4×4 head-pose
- [ ] A2.AC2 num_faces=1; temporal smoothing; monotonic ts fed to Task
- [ ] A2.AC3 no face → FrameContext.face is None; extractors emit nothing (no crash/stale)
- [ ] A3.AC1 separate PoseLandmarker Task (Holistic not used)
- [ ] A3.AC2 One-Euro filter reduces jitter on static-subject clip (measurable)
- [ ] A3.AC3 per-landmark visibility/presence exposed
- [ ] A4.AC1 backbone sustains ≥30fps without unbounded memory growth
- [ ] A4.AC2 under load per-frame extractors drop frames; lossless ring loses zero samples
- [ ] A4.AC3 dropped-frame counters + per-extractor timing exposed
### B — Bus, Logging, Dashboard
- [ ] B1.AC1 each signal/covariate own outlet; nominal_srate vs IRREGULAR_RATE
- [ ] B1.AC2 stream metadata carries units/method/window; channels include sqi + gate_code
- [ ] B1.AC3 a second consumer can resolve and read every stream
- [ ] B2.AC1 one command records all streams + Polar to one XDF
- [ ] B2.AC2 XDF loads via pyxdf with clocksync+dejitter; clock-offset metadata present
- [ ] B2.AC3 replay reproduces per-signal sample counts within rounding
- [ ] B3.AC1 P0 dashboard shows live HR + SQI + gate, ≥1Hz
- [ ] B3.AC2 P1 dashboard renders every signal value+SQI+traffic-light
- [ ] B3.AC3 REJECT/degraded visibly marked "re-acquiring", never silently frozen
- [ ] B4.AC1 selected streams re-emitted as OSC addresses / WebSocket JSON
- [ ] B4.AC2 test consumer receives values at expected rate with forward-sync look-ahead
- [ ] B4.AC3 bridge adds no back-pressure if consumer disconnects
### C — Ground Truth & Time Sync
- [ ] C1.AC1 BLE connect by device name; auto-reconnect after dropout
- [ ] C1.AC2 full flag-driven 0x2A37 parse; all RR per packet; 4 flag-combo test vectors
- [ ] C1.AC3 beat times = cumsum from one anchor (not packet arrival); published HR+beat-times
- [ ] C1.AC4 RR artifacts corrected with NeuroKit2 Kubios before HRV
- [ ] C2.AC1 instantaneous-HR resampled 4Hz + cross-correlated; residual lag reported
- [ ] C2.AC2 camera beats matched to H10 within ±50–100ms; unmatched counted (yield)
- [ ] C2.AC3 DTW only a reported metric, never the aligner
### D — Cardiac
- [ ] D1.AC1 ROI multi-patch mean (forehead+cheeks) excluding eyes/mouth/brows
- [ ] D1.AC2 yaw beyond cutoff drops occluded cheek patch; ROI still valid
- [ ] D1.AC3 landmark jitter doesn't spike ROI mean (patch averaging) on static clip
- [ ] D2.AC1 POS rolling 8–15s window, ~1Hz update, output [42–240] bpm
- [ ] D2.AC2 POS matches pyVHR reference on a fixed clip within tolerance
- [ ] D2.AC3 HR vs Polar meets EC13 numeric threshold + MAPE<10% at rest ≥300lux; capture→emit ≤2s
- [ ] D2.AC4 each HR record carries De Haan SNR as sqi + gate; P0 gate = "unknown" stub
- [ ] D3.AC1 BVP cubic-spline upsampled ≥256Hz before peak detection
- [ ] D3.AC2 Orphanidou bSQI gates beats at a calibrated (ROC vs H10) threshold; low dropped
- [ ] D3.AC3 SDNN/RMSSD over fixed length-matched window when ≥80% beats GOOD+accepted; else "insufficient clean data"
- [ ] D3.AC4 SDNN MAE≤12ms + BA + CCC; ln-RMSSD BA in log-units, indicative
- [ ] D3.AC5 LF/HF / frequency-domain HRV not produced anywhere
### E — Respiration
- [ ] E1.AC1 RR from band-passed 0.08–0.5Hz chest/shoulder displacement, 15–30s, FFT
- [ ] E1.AC2 rPPG-derived RR secondary cross-check; motion is primary
- [ ] E1.AC3 Indicative (no belt): plausibility check vs metronome, no pass/fail
- [ ] E1.AC4 RR carries SQI from band-power ratio + pose visibility; motion-gated
### F — Ocular
- [ ] F1.AC1 open-eye baseline auto-calibrates; personalized threshold
- [ ] F1.AC2 blink = EAR<thr for ≥2–3 frames; emits events(+duration) and windowed rate
- [ ] F1.AC3 blink F1≥0.90 vs frame-level annotation with ±N-frame match tolerance
- [ ] F1.AC4 PERCLOS-P80 tracks commanded closure-fraction blocks
- [ ] F1.AC5 warn if fps<25; withhold metrics until baseline armed
- [ ] F2.AC1 gaze vector from learned head; iris front-end only
- [ ] F2.AC2 output zones, no pixel coordinates
- [ ] F2.AC3 3×3 confusion matrix + accuracy above 33% chance
- [ ] F2.AC4 optional 5–9-point calibration improves zone accuracy
- [ ] F3.AC1 no pupillometry feature; arousal proxy used instead
### G — Facial Affect
- [ ] G1.AC1 blendshapes neutral-subtracted + z-scored per session
- [ ] G1.AC2 HSEmotion emits 8-class emotion + V/A at 10–15Hz tunable, confidence as sqi
- [ ] G1.AC3 outputs labeled "estimate, confidence X", never a verdict
- [ ] G1.AC4 face-validity report: posed-happy V>posed-sad V + effect sizes + confusion matrix
- [ ] G2.AC1 AU model decoupled 5–15fps, doesn't drop live path below 30fps
- [ ] G2.AC2 AU streams emit 0–5 intensities, flagged research, with metadata
- [ ] G2.AC3 ONNX preferred; PyTorch path isolated via escape hatch
- [ ] G3 WITHDRAWN — assert no quarantine/firewall
### H — Motion, Quality, Covariates
- [ ] H1.AC1 De Haan SNR per window normalized 0..1 as sqi
- [ ] H1.AC2 skewness/perfusion supplements + Orphanidou bSQI implemented
- [ ] H1.AC3 low-SQI records emitted with a flag (HRV beats excepted — dropped)
- [ ] H2.AC1 gate inputs FM_X/FM_Y, FSM, solvePnP head pose (pitch×2), pulse SNR/skewness from FaceMesh
- [ ] H2.AC2 motion metrics normalized by inter-ocular distance
- [ ] H2.AC3 GOOD/USABLE/REJECT hysteresis (dwell≥1s) — no flicker on borderline
- [ ] H2.AC4 induced motion → REJECT, HRV/RR suppressed, HR holds last-good, "re-acquiring"; recover to GOOD
- [ ] H2.AC5 gate rides on every SignalRecord.gate + published as its own stream; P0 "unknown" stub
- [ ] H3.AC1 covariates (skin-tone+cheek estimate, eyewear, facial hair, lighting index+lux+exposure, motion, presence) as streams
- [ ] H3.AC2 skin tone/eyewear/facial hair recorded once per session, stored with XDF
### I — Validation Harness
- [ ] I1.AC1 scripted blocks (rest×2, paced-breathing, light-motion, lighting A/B, optional gaze/eye-closure, ≥2 repeats) + XDF markers
- [ ] I1.AC2 measured lux per lighting block; HRV from rest only (excl 6-brpm); respiration from paced blocks
- [ ] I2.AC1 report: BA, MAE, RMSE, MAPE, Pearson r, CCC, SNR; H10 labeled reference
- [ ] I2.AC2 HR reported at 60s average and 4–10s quasi-instantaneous window
- [ ] I2.AC3 SDNN MAE≤12 + BA + CCC length-matched; ln-RMSSD log-units; HR vs EC13+CTA numeric thresholds
- [ ] I2.AC4 feasibility banner + Fitzpatrick best/worst-case caveat
- [ ] I2.AC5 one command produces the full report from an XDF
### J — Cross-cutting
- [ ] J1.AC1 threading: capture + backbone + N extractor threads + asyncio I/O edge; no asyncio on CV hot path
- [ ] J1.AC2 multiprocessing+shared_memory escape hatch passes frame handles (not pickled)
- [ ] J1.AC3 runtime policy check: CPython 3.11, numpy<2, CPU-only torch
- [ ] J2 WITHDRAWN — assert no licensing firewall
- [ ] J3.AC1 deterministic POS on a fixed clip; pinned deps manifest
- [ ] J3.AC2 `argus record --session NAME` runs a session to a synchronized XDF
- [ ] J3.AC3 `argus report --xdf FILE` produces the report
- [ ] J3.AC4 `argus run` launches live pipeline + dashboard + bus
- [ ] J4.AC1 default storage = derived + XDF; no raw video unless opt-in flag
- [ ] J4.AC2 raw-video opt-in stored locally + labeled
