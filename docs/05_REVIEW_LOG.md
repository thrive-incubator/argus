# Argus — Review Log (Iteration 1)

*Record of the adversarial review pass over the Phase 0/1 document set and how each issue was resolved. Maintained so the engineering team can see what was challenged and why the docs say what they say.*

**Date:** 2026-06-24 · **Reviewers:** engineering-implementability lens + scientific/scope lens (independent).

> **Rev 2 scope change (owner directive, supersedes parts of this log):** (1) **no hardware
> purchases** → B3 respiration GT moot: respiration is now **Indicative** (motion-only, no belt),
> no MAE bar; lux via free phone app. (2) **Productification/licensing out of scope** → the whole
> licensing-firewall track (the `argus/research/` quarantine, the M7 import-lint, ADR-13, NFR-6,
> G3, J2) is **removed**; AUs use the best model (**LibreFace/OpenFace 3.0**) directly, which also
> eases B2 (no MIT-only constraint). (3) **Accuracy/fairness clearance** explicitly out of scope.
> (4) Canvas = **TouchDesigner/OSC**. Items B1, B4, B5, M1–M5, M8, M9, C2, C3, I1–I8 remain in
> force. **One open question:** confirm a Polar H10 (cardiac ground truth) is already on hand.

Severity: **BLOCKER/CRITICAL** = corrupts results or blocks day-1 work · **MAJOR/IMPORTANT** = must fix before the relevant feature · **MINOR** = tighten wording/precision.

| # | Severity | Issue (short) | Resolution | Docs touched |
|---|---|---|---|---|
| B1 | BLOCKER | "Latency ≤ 2 s" ambiguous vs 10 s rPPG window | Defined latency = capture→emit (pipeline lag), explicitly **excluding** intrinsic ~½-window physiological smoothing | PRD NFR-1, §4.1; TECH §6.1, §16 |
| B2 | BLOCKER | MediaPipe + ONNX + Py-Feat/Torch env may be unsatisfiable | Added dependency-compatibility spec: py3.11, `numpy<2`, CPU-only torch, **Py-Feat isolated in a subprocess/venv**; marked as Spike-0 | TECH §2, §17; PRD §9 |
| B3/C1 | BLOCKER | Respiration GT was a metronome (compliance ≠ accuracy) | **Respiratory belt (RIP) is now the committed RR ground truth**; metronome only spans the rate range; if no belt, respiration downgrades to *Indicative* | ADR-07; PRD §5; TECH §11,§12; FEAT E1 |
| B4 | BLOCKER | Polar RR parse omitted flag-driven offsets (bit0 16-bit HR, bit3 energy) → silent HRV corruption | Replaced with full flag-driven offset algorithm + test vectors for all 4 flag combos | ADR-17; TECH §11; FEAT C1 |
| B5 | BLOCKER | macOS camera/Bluetooth TCC permissions unmentioned → silent first-run failure | Added macOS permissions subsection + loud startup self-check | TECH §2a; PRD §9 |
| M1 | MAJOR | `consume()` return type `Optional` vs `list` contradiction | Standardized on `list[SignalRecord]` everywhere | ADR-19; PRD NFR-7; TECH §3.3 |
| M2 | MAJOR | `FrameContext.frame` shared+mutable across fan-out (data race) | Declared frame **read-only**; `writeable=False`; copy-before-write rule | TECH §3.1, §4 |
| M3 | MAJOR | Phase 0 must emit required `gate` field before gate exists | Added `"unknown"` to gate enum; Phase 0 ships always-`unknown` stub | TECH §3.2; FEAT §3.2, D2 |
| M4 | MAJOR | "Good light"/A/B had no lux; "lux proxy" can't yield lux | Defined lux targets (≥300 / ~150 / ~500 lux via cheap meter); relabeled luma as **relative brightness index (uncalibrated)** | PRD §5; TECH §9; FEAT I1 |
| M5 | MAJOR | Camera-disconnect / BLE-reanchor / blink-baseline failure modes unspecified | Added explicit failure-mode behaviors incl. BLE re-anchor + XDF gap marker | TECH §11, §6.4, new §18; FEAT C1 |
| M6 | MAJOR | Golden/blink/gaze fixtures had no source; dataset+privacy conflict | POS test uses **synthetic** trace; face clips are local/LFS only, never committed; CI vs local split | TECH §14, §14a |
| M7 | MAJOR | "CI gate" conflated headless vs hardware-in-loop; import-lint "simple" but isn't | Split CI (headless) vs HIL (self-hosted); specified AST-based import lint | TECH §14a; FEAT J2 |
| M8 | MAJOR | Affect/AU ACs unfalsifiable ("move in expected direction") | Reworded as **descriptive deliverables**, no pass/fail | FEAT G1,G2; PRD §5 |
| M9 | MAJOR | HRV `gate==good` + 60–120 s window → may never emit (deadlock) | GOOD-fraction policy (≥80% of beats GOOD/bSQI-accepted); defined zero-window reporting | TECH §6.2, §8; FEAT D3 |
| C2 | CRITICAL | SDNN committed bar was `|bias|` — weak statistic, passes for wrong reason | Changed to **SDNN MAE ≤ 12 ms** + report LoA + Lin's CCC (bias alone dropped) | ADR-05,18; PRD §5; TECH §12; FEAT D3 |
| C3 | CRITICAL | Affect "validation vs posed expressions" invalid for the construct | Reframed as **model face-validity check**; "posed ≠ felt affect" stated; word "validation" removed from affect rows | ADR-11; PRD §5; TECH §16; FEAT G1 |
| I1 | IMPORTANT | Blink GT annotation protocol + eyewear unrecorded | Specified frame-level annotation + ±N-frame match tolerance + κ; **eyewear added as covariate** | TECH §9; FEAT F1 |
| I2 | IMPORTANT | Gaze ≥80% meaningless without zone geometry; 33% chance | Specified target eccentricities/zone half-widths; report **3×3 confusion matrix + above-chance** | ADR-09; PRD §5; FEAT F2 |
| I3 | IMPORTANT | Beard destroys cheek rPPG ROI; facial hair unrecorded | **Facial hair added as covariate**; cheek-drop fallback to forehead-only | ADR-03; TECH §5,§9; FEAT D1 |
| I4 | IMPORTANT | H10 treated as truth; its own seated RMSSD bias not propagated | Report states H10 is *reference not gold truth*; its posture bias is included in reported HRV bias | ADR-18; TECH §12; FEAT I2 |
| I5 | IMPORTANT | bSQI 0.86 borrowed from contact PPG, unproven for rPPG | Threshold is **provisional, calibrated on this subject's rPPG beats** vs H10-confirmed beats | ADR-06; TECH §6.2; FEAT D3 |
| I6 | IMPORTANT | Cross-corr per-segment lag absorbs real IBI jitter (HRV confound) | HR-rate uses cross-corr; **HRV uses a single fixed median-PTT lag** to preserve beat jitter | ADR-16; TECH §11; FEAT C2 |
| I7 | IMPORTANT | PERCLOS "monotonic" check is near-tautological | Reworded to commanded closure-fraction blocks; no drowsiness GT in scope | PRD §5; FEAT F1 |
| I8 | IMPORTANT | Skin-tone "for stratification" meaningless at n=1; dark-skin could sink validation invisibly | Stated covariate is **forward-compat only**; added single-subject Fitzpatrick best/worst-case caveat | ADR-18; PRD §5; TECH §9,§16 |
| m1 | MINOR | `gate_code` numeric mapping undefined | Defined `good=0,usable=1,reject=2,unknown=3` | TECH §10 |
| m2 | MINOR | One-Euro filter would suppress the fidget signal it measures | Fidget uses lightly-/pre-filtered landmarks | TECH §6.8 |
| m3 | MINOR | Model-weight download/caching + "no cloud" conflict | Added model-assets section: pinned URLs+checksums, `argus fetch-models`, offline after fetch | TECH §17a; PRD NFR-5 note |
| m4 | MINOR | Sync cross-corr on flat rest HR degenerates | Run alignment on a motion/talking block where HR varies; fallback stated | ADR-16; TECH §11 |
| m5 | MINOR | CMS50E SpO₂ vs SpO₂-out-of-scope mixed message | Clarified CMS50E provides PPG-waveform reference only; SpO₂ field ignored | ADR-17 |
| m6 | MINOR | HSEmotion at 30 Hz asserted but competes for CPU | Made affect cadence a tunable (default 10–15 Hz) | TECH §6.6; FEAT G1 |
| m7 | MINOR | L2CS vs MobileGaze not drop-in (different I/O conventions) | Zone-map layer is model-specific; one is default, other a stretch | ADR-09; TECH §6.5 |
| m8 | MINOR | Windows "best-effort" undefined | Windows declared **unsupported** in Phase 0/1 | PRD NFR-3 |
| sM1 | MINOR | Resp band 0.1 Hz clips the 6-brpm metronome at the filter edge | Low edge widened to ~0.08 Hz for paced block | ADR-07; TECH §6.3 |
| sM2 | MINOR | De Haan SNR denominator must exclude signal bins | Denominator = out-of-band only; align to [0.7–4 Hz] support | ADR-06; TECH §7 |
| sM4 | MINOR | BA on ln-RMSSD can't be in "ms"; ±15 ms band misapplied | ±15 ms band applies to **SDNN only**; ln-RMSSD BA reported in log-units/ratio | ADR-18; TECH §12; FEAT I2 |
| sM5 | MINOR | "Meets EC13" reads like conformance (needs N subjects) | Reworded "meets EC13 *numeric threshold* (feasibility, not conformance)" | PRD §4.1; FEAT D2 |
| sM6 | MINOR | SDNN is window-length-dependent; must length-match camera vs H10 | Added: HRV stats computed over identical time-aligned windows | TECH §12 |

**Net effect on commitments:** Respiration stays *Committed* **only if** a respiratory belt is used (else *Indicative*). SDNN bar moved from bias to MAE+LoA+CCC. Affect/AU explicitly downgraded from "validation" language to "face-validity / descriptive." Two new recorded covariates (eyewear, facial hair). No change to the HRV scoping (SDNN/RMSSD/no-LF-HF) or the pupillometry cut — both were already correct.
