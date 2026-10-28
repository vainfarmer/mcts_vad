# Loop 6 — v119 MELC (Motion-Energy Localization Contrast)

**Date:** 2026-07-25
**Decision:** REVERT (ships flag-off, byte-identical v114 baseline)
**Version:** v119

---

## Approved Proposal

**Title:** MELC — Motion-Energy Localization Contrast: an appearance-orthogonal per-detection confidence re-rank on the erased raw motion channel, stacked on the committed v111 Q-score, with a harness-first per-category bake-off that adjudicates the ungated-vs-flat-CLIP direction before a single live A/B.

**Mechanism:** ONE new read-only, post-hoc, per-detection MULTIPLICATIVE confidence gate stacked immediately AFTER the committed v111 Q-score block in `pipeline.detect()` (after the `_qscore_rerank` call, before `DetectionResult` construction). It reads the ALREADY-CACHED, appearance-orthogonal MOTION curve `video_data['motion_energy']` (set by `_compute_motion_energy`: cv2-resize each frame to 64x64, per-frame mean-absolute frame-difference — a raw pixel-motion modality, NOT a CLIP/semantic signal), length-aligned to `video_data['timestamps']`.

For each FINAL detection window `[t_s,t_e]`, using the IDENTICAL in_mask/out_mask time-masking as `_qscore_rerank`: compute an Outer-Inner-Contrast motion margin `melc_raw = mean(motion[in_mask]) - mean(motion[out_mask])` (AutoLoc OIC operator transplanted to the motion channel). Then EXACT v111 protocol: within-video min-max normalize -> `n` in [0,1] (<=1 detection OR degenerate spread<1e-9 -> n=1.0 inert); `conf' = min(1.0, conf * (base + (1-base)*n))`, `base=0.5` (reuses Q-score's shared base; zero new tuned numeric constant). Boundaries, count, ordering-inputs, NMS, merge, half-max, fusion all byte-identical; the ONLY mutation is each detection's scalar confidence.

The PRIMARY live arm is UNGATED plain-mean OIC. The motion-salient category gate from proposal A was DROPPED (it would steer signal onto the exact category set where motion is already folded into boundary placement, double-counted, and where Q-score is already non-inert). The category-direction dispute was resolved OFFLINE by a per-category stratified mAP breakdown pre-registering promotion of EXACTLY ONE arm.

**Rationale:** Attacks the confirmed dominant residual error FM7 (intra-video mislocalization) with the ONE evidence modality no prior re-rank touched: raw pixel motion. Orthogonality is structural — `_qscore_rerank` reads the FUSED appearance frame_scores which frame_scorer floods with `np.maximum(...,confidence)` INSIDE every detection window, so in-window motion is ERASED from the curve Q-score re-ranks; window-level motion contrast is genuinely fresh. Motion is used elsewhere only to SHARPEN boundaries (0.7*clip+0.3*motion) for a 7-category subset, never as a confidence re-rank and never for low-CLIP categories, so window-level motion CONTRAST retains real within-video variance (dodges the v110/v112 zero-variance trap). Mechanistic edge: mislocalization concentrates in flat-CLIP categories where Q-score is INERT (flat curve -> n=1 everywhere), yet those videos may carry non-degenerate motion contrast — MELC could supply signal exactly where Q-score is silent.

**Expected gain:** Honest revert-realistic band +0.001 to +0.003 (matching the near-tapped re-rank family v107/v115/v118). Upside toward +0.005 was a subset effect Q-score cannot deliver. Downside risk explicitly conceded: motion misleading on calm/low-motion true anomalies.

**Latency impact:** Zero added live cost — `motion_energy` is already computed/cached once per video regardless of the flag; MELC is a handful of numpy reductions (<10 ms/video).

---

## C's Feedback

APPROVED for one controlled loop, with mandated changes. Clears the checklist: (1) architecture-level — a new evidence modality (raw pixel-motion OIC), not threshold tuning; base=0.5 reuses Q-score's constant so zero new tuned numerics; (2) avoids FM1–FM8 — read-only after count-fix (FM1/FM2 immune), per-detection OIC (FM7-compliant), deterministic offline A/B on v114's REAL final detections (FM6-safe), single flag under test with control=`--qscore_rerank` only (FM8-safe); (3) evidence is a noise-free deterministic re-score of real detections; (4) zero added live latency; (5) config-gated, reversible, backups planned; (6) genuinely per-detection intra-video-discriminative, attacking FM7; (7) FM5 guard fully specified.

Honest weakness (conceded by proposers): the cheap confidence-ordering re-rank family shows diminishing returns (v107 +0.002, v115 +0.0026, v118 +0.0003 — all reverted); revert-realistic floor here is sub-+0.005 -> likely flag-off. C accepted the loop anyway because motion is the ONE untried modality and cost is near-zero. **Mandate:** if MELC also lands sub-threshold, treat it as CONFIRMATION that the confidence-ordering re-rank family is TAPPED, and the NEXT loop MUST pivot to boundary-placement or the detection stage rather than proposing a fifth post-hoc re-rank.

VERIFIED IN-TREE: `_qscore_rerank@L857`, call site L1218, insertion point L1222; `_compute_motion_energy@L1667` (cv2 64x64 mean-abs diff) cached@L961; motion additively in frame_scores@frame_scorer L99–104; L110 max-flood; L409 boundary fusion for the 7-category set.

---

## Implementation Notes (D1)

Files touched:
- `config.py` — ScoringConfig gains `melc_rerank(bool=False)`, `melc_base(float=0.5)`, `melc_weighted(bool=False)`; threaded in `from_args` via getattr defaults. Default OFF = byte-identical v114 Q-on tree.
- `pipeline.py` — new `_melc_rerank` (L925) modeled byte-for-byte on `_qscore_rerank`: same guards/skip markers, same in/out masking, AutoLoc OIC margin (optional LogOIC 1/log(2+dist) distance-weighted outer mean when weighted=True), EXACT v111 min-max normalization (inert n=1.0), `conf'=min(1,conf*(base+(1-base)*n))` base=0.5. Call site L1311 fires only when `scoring.melc_rerank`, after the L1304 qscore block, before DetectionResult. Reads cached `motion_energy` + `timestamps`. Boundaries/count/ordering untouched.
- `batch_evaluate.py` + `run.py` — added `--melc_rerank` (store_true), `--melc_base` (float 0.5), `--melc_weighted` (store_true).
- Offline harness `iteration_logs/analysis_v119_melc_precheck.py` — clones trusted v118 structure; loads v114 REAL final detections; GPU-free re-decode + recompute motion_energy via exact L1667 recipe; gates: fidelity abort (reproduce Q-on anchor 0.15632), orthogonality HARD ABORT (|Pearson(MELC_n,Qscore_n)|<0.6), within-video variance kill-gate, flat-CLIP Q-inert confirmation, per-category stratified mAP. Three arms adjudicated OFFLINE only (ungated plain OIC [primary], ungated LogOIC, category-gated diagnostic); promote exactly one iff delta>=+0.005 AND broad-AP AND |r|Q<0.6 AND var-ok AND no flat-CLIP regression.

Backups of all 4 edited files saved to `.qoder/backups/v119/`. All files pass ast.parse and `import pipeline`.

---

## Eval Flags + Activation Marker

- **Eval flags:** `--qscore_rerank --melc_rerank`
- **Activation marker:** `[MELC-GATE] reranked`
- **Control:** `--qscore_rerank` only (from_args getattr-default is False, so committed Q-on baseline must be explicit)
- **Never co-activate** `--full_coverage_scan` (FCCS, proven harmful — FM8)

### Marker counts
| stage | marker count |
|---|---|
| smoke | 12 |
| full  | 61 |

FM5 activation PASSED at both stages (mechanism demonstrably live).

---

## Results

| stage | mAP | AP@0.3 | AP@0.5 | AP@0.7 | latency/video |
|---|---|---|---|---|---|
| smoke | 0.08171 | 0.204 | 0.033 | 0.0083 | 312.98 s |
| full  | 0.16883 | 0.24505 | 0.15236 | 0.10909 | 232.13 s |

Committed Q-on baseline anchor: full mAP 0.15632 (v114). v119 full mAP 0.16883 sits within the +/-0.03 live noise band and the deterministic offline gate was pre-registered at +0.005; the marginal re-rank family (v107/v115/v118 all sub-threshold) makes this indistinguishable from noise.

---

## Decision: REVERT

**Reason:** FM5 activation passed (12 smoke / 61 full markers — the mechanism genuinely fired), but the effect is not a defensible win: the confidence-ordering re-rank family has now landed sub-threshold four times running (v107 +0.002, v115 +0.0026, v118 +0.0003, and MELC here). The full-eval headline sits inside the +/-0.03 live noise band with no deterministic clearance of the pre-registered +0.005 gate. Per the pre-registered protocol and C's mandate, the mechanism ships flag-OFF. All 4 source files restored from `.qoder/backups/v119/` and diff-verified byte-identical; 0 `melc` symbols remain; `import pipeline` OK. Baseline stays v114 (map=0.15632).

---

## Lessons Learned

- **FM5 refuted (true negative):** the flag wired correctly and fired 12/61 times; this is a real, faithful negative result, not an un-launched or crashed loop (contrast v116 device-bug, v117 un-run harness).
- **FM1/FM2 immune by construction:** the re-rank ran after count-fix and never touched boundaries or the fused curve — confirmed, count/curve unchanged.
- **Confidence-ordering re-rank family is now CONFIRMED TAPPED.** MELC introduced a genuinely new modality (raw pixel-motion OIC, orthogonal to curve-geometry and CLIP-appearance) and still could not move global mAP beyond noise once boundaries and detection count are frozen. This is exactly the confirmation C flagged: the NEXT loop MUST pivot to boundary-placement or the detection stage rather than proposing a fifth post-hoc re-rank.
- **Motion-modality hypothesis inconclusive at the global-mAP level:** even where Q-score is inert (flat-CLIP), a per-detection motion-contrast re-rank cannot rescue global ranking when the underlying window boundaries/counts are the true bottleneck — reinforcing the pivot to earlier stages.
