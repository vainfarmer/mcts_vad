# Loop 7 — v120 BREF (Boundary-REFinement bake-off)

**Decision: REVERT** (ships flag-off, byte-identical to v114 baseline map=0.15632)

Date: 2026-07-25

---

## 1. Approved Proposal (summary)

**Title:** BREF — Boundary-REFinement bake-off: a harness-first, per-detection, count-frozen
boundary-PLACEMENT stage that adjudicates TWO orthogonal snap anchors offline on v114's real
detections (Arm-1 fused-curve derivative-extremum, Arm-2 motion-gradient half-max edge) and
promotes EXACTLY ONE to a single live A/B stacked on the committed v111 Q-score.

**Why this pivot:** The confidence-ordering post-hoc re-rank family is CONFIRMED TAPPED
(v107 ECE +0.002, v115 PROM +0.0026, v118 FSL +0.0003, v119 MELC noise — four consecutive
sub-threshold reverts). The standing v119 mandate required the next loop pivot to
boundary-PLACEMENT or the detection stage. Under greedy matching mAP is dominated by
LOCALIZATION (IoU with GT), not by re-ordering already-committed detections (FM7). The
project's own IoU decomposition puts ~39% of detection-quality loss in the IoU 0.1-0.5
"boundary-imprecise / long-event-shrunk" bucket — reachable ONLY by moving endpoints.

**Mechanism:** A read-only, post-hoc, per-detection boundary-PLACEMENT stage
`_boundary_refine(...)` inserted as the ABSOLUTE LAST transform in `detect()` — AFTER the
committed v111 Q-score block and BEFORE `DetectionResult`. Confidence stays byte-identical to
the Q-on baseline; BREF mutates ONLY the [t_s, t_e] offsets. Two arms, one promoted live via
`boundary_refine_mode`:
- **ARM-1 `fused_deriv`** — np.gradient of the cached fused frame_scores curve; within ±W snap
  start→argmax(+grad) (steepest RISE = BSN start-actionness peak), end→argmin(grad) (steepest
  FALL). Reads the curve's FIRST DERIVATIVE, orthogonal to Q-score (curve LEVEL).
- **ARM-2 `motion_edge`** — cached `video_data['motion_energy']` smoothed with the same 1s
  kernel as `_motion_temporal_localization`, signed gradient, v51 half-max LEVEL formula
  (max+p20)/2 on |grad|; ±W snap start→dominant rising edge, end→dominant falling edge.
  Appearance-orthogonal motion modality.

Both arms share ECE discipline (category resolution, searchsorted grid-snap,
CATEGORY_DURATION_PRIORS reject-cap, no-edge inert). W pinned to the EXISTING derived quantity
W = (t_e - t_s) * 0.3 (no new numeric constant). Expected gain: +0.005 to +0.015 mAP if arms
are event-aligned; floor near-zero (no-edge inert keeps v51 boundaries).

## 2. C's Feedback

APPROVE. The correct, disciplined pivot the v119 mandate demands. Genuinely per-detection,
intra-video-discriminative, orthogonal to v111 Q-score. Causal, not wishful. Evidence protocol
is the trusted class (deterministic re-derivation on the run's OWN real detections; Arm-1 reads
cached frame_scores; Arm-2's motion recompute is a deterministic pixel function) — NOT a
candidate-pool simulation, so FM6 does not apply. FM1/FM2 immune by construction. FM5 answered
with a real --boundary_refine flag, [BREF] per-detection + per-video markers on ALL return
paths, and an exit-2 hard gate mirroring the ECE gate. FM7 satisfied. FM8 clean (single live
flag, arm pre-selected offline, control leg = --qscore_rerank only). Not TSG. Latency
near-zero. Verified load-bearing code refs (L961/L1218-1219/L1223/L1237/L1667). Two mandated
corrections applied by D1: (1) W pinned to existing (t_e-t_s)*0.3; (2/3/4) harness repointed to
v114 real detections, count==65 assert, exactly-one promotion gate.

## 3. Implementation Notes (D1)

Files touched: pipeline.py (`_boundary_refine`, call site after qscore block before
DetectionResult, pre_bref_boundaries on all three return sites incl. both early returns),
config.py (ScoringConfig.boundary_refine=False, boundary_refine_mode='fused_deriv' placeholder),
models.py (DetectionResult.pre_bref_boundaries + conditional serialization),
batch_evaluate.py (--boundary_refine, --boundary_refine_mode, [BREF-GATE] exit-2 hard gate),
run.py (both flags), iteration_logs/analysis_v120_boundary_precheck.py (STEP-0 harness).
Backups in .qoder/backups/v120/ (4 files: batch_evaluate.py, config.py, pipeline.py, run.py).
All ast.parse OK; import pipeline OK; --help shows both flags; synthetic tests confirmed
fused_deriv snaps 11→10 / 19→20, motion_edge edge detection + reject-cap fire, flat-curve and
Normal-category inert, [BREF] markers emit correctly. D1 explicitly did NOT run STEP-0 (an eval)
and flagged that the finalizer MUST set the promoted arm before any live run.

## 4. Eval Configuration

- **Eval flags (treatment leg):** `--qscore_rerank --boundary_refine --boundary_refine_mode fused_deriv`
- **Control leg:** `--qscore_rerank` only
- **Activation marker:** `[BREF] summary:`
- **Marker counts:** smoke 12 (== 12 smoke videos); full 64 (== 64 succeeded videos). FM5 guard PASSED both legs.
- **Arm run live:** `fused_deriv` (ARM-1). NOTE: this was the *placeholder default*, NOT a harness-promoted arm — see §7.

## 5. Results

| leg | mAP | AP@0.3 | AP@0.5 | AP@0.7 | latency (s/video) | markers |
|-----|-----|--------|--------|--------|-------------------|---------|
| smoke (12 vids) | 0.13889 | 0.212 | 0.114 | 0.091 | 351.67 | 12 |
| full (64 vids)  | 0.17488 | 0.23303 | 0.17177 | 0.11983 | 158.35 | 64 |

Committed v114 deterministic baseline (VLM-noise-free A/B): map=0.15632.
Full headline 0.17488 sits inside the ±0.03 live noise band (cf. v107 0.17291, v115 0.17559,
v119 0.16883) and does not represent a deterministic clearance of the pre-registered +0.005
boundary A/B gate with AP@0.7 non-regression.

## 6. Decision + Reason

**REVERT.** (1) The headline full mAP 0.17488 is inside the ±0.03 live noise band — not a signal.
(2) The pre-registered LIVE VERDICT required a deterministic pre_bref/post_bref boundary A/B
delta ≥ +0.005 with AP@0.7 non-regression; that offline gate was NOT cleared/demonstrated.
(3) Critically, the STEP-0 harness (`analysis_v120_boundary_precheck.py`) was never executed —
`analysis_v120_boundary_precheck.json` does NOT exist — so no arm was ever legitimately promoted;
the live run used the placeholder default arm, violating the harness-first protocol. Per D1's
explicit instruction, absence of a promoted arm ⇒ REVERT at zero live cost.

Enforcement performed by finalizer:
- Restored all 4 backup files (batch_evaluate.py, config.py, pipeline.py, run.py) — diff-verified byte-identical.
- Removed the residual `pre_bref_boundaries` field + its conditional serialization from models.py (not backed up; inert with flag-off but removed for a clean revert).
- Confirmed 0 `boundary_refine`/`_boundary_refine`/`BREF`/`pre_bref` symbols remain in the 5 code files; `import pipeline` OK.
- Baseline stays v114 map=0.15632. The BREF harness .py is left in place (offline-only, inert).

## 7. Lessons Learned

- **FM5 guard: REFUTED as a false-positive risk** — the `[BREF] summary:` marker fired 12/12
  smoke and 64/64 full, so the mechanism genuinely activated and ran live; this is a true
  negative, not a plumbing failure.
- **FM7 pivot on-target but effect null at headline:** moving to boundary-PLACEMENT is the
  correct mandated pivot off the tapped re-rank family, but the fused_deriv arm did not produce
  a demonstrable deterministic mAP gain. The IoU-0.1-0.5 bucket movement was never adjudicated
  because the offline harness was not run.
- **PROCESS FAILURE (confirmed, repeat of v117):** harness-first was violated. v117 already
  reverted for exactly this reason (harness never run ⇒ arm undetermined). Here the harness was
  ALSO never run, but the live eval launched anyway on the placeholder default arm — worse than
  v117 because it consumed a full-eval budget on an unvalidated arm. Future finalizers/launchers
  MUST gate the live launch on the existence of the precheck JSON with a promoted arm; do not
  launch on the config default.
- **FM1/FM2: IMMUNE by construction** (fused curve untouched; count/order/confidence/NMS/merge
  frozen) — no evidence of curve-shape or duplicate-FP regression; consistent with the ECE/EUBC
  change-class.
- The confidence-re-rank family remains tapped; the boundary-placement family remains
  UN-adjudicated (not refuted) because the offline gate was skipped. A re-run of the STEP-0
  harness to actually promote/reject an arm is the clean next step before any further live BREF eval.
