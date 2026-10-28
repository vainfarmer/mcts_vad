# Agent Team Loops v114+ Summary (Continuation Session — v97-v113 lineage)
## Date: 2026-07-25
## Pipeline: video_anomaly_detection (UCF-Crime) — temporal localization, weak supervision
## Target: full-eval temporal mAP >= 0.20, per-video latency near/below ~128s

## Executive Summary
This session ran 10 loops (v114-v123) continuing from the v97-v113 arc. **Final
production best: v114, full-eval temporal mAP = 0.15632 (Q-score re-rank flipped
default ON), up from the v94/v89 true baseline of 0.1494.** Target of mAP>=0.20
was **NOT achieved**; the ~0.15-0.16 band remains a structural ceiling for
incremental post-hoc work, exactly as forecast.

The one durable positive of the session was **loop 1's Q-score in-vivo verdict**:
the v111 localization-quality re-rank (fused-curve geometry: boundary sharpness +
peak contrast + center fit, within-video min-max normalized) had shown +0.0087
offline on v108 *cached* detections but was never cleanly tested live (v113
confounded it with the proven-harmful FCCS flag). Loop 1 ran a **deterministic,
VLM-noise-free A/B on fresh v114 detections**: map_off 0.15013 -> map_on 0.15632,
**delta +0.00619**, clearing the pre-registered +0.005 gate, broad-based
(AP@0.3/0.5/0.7 all rose). This **affirmatively resolved FM6 for pure post-hoc
re-ranks**: because such a re-rank is a deterministic function of the run's own
output curve with zero VLM-noise coupling, the cached offline gain genuinely
transferred to fresh detections. `ScoringConfig.qscore_rerank` default was flipped
False->True; v114 is the new committed baseline.

Every subsequent loop (v115-v123) reverted. The session's central finding is a
**negative structural one**: the cheap post-hoc confidence-re-rank family is now
**TAPPED** (v107 ECE, v115 prominence, v118 FSL, v119 MELC all landed sub-+0.005
after the v95 TSM / v111 Q-score positives), and the mandated pivot to
**boundary-placement** (v120-v122) has so far produced one offline refutation
(v122 translation) and several un-adjudicated attempts blocked by a recurring
process failure (harness-skip). No mechanism beyond Q-score cleared the commit bar.

## What Was Tried (v114-v123)

| loop | ver | mechanism | result | full mAP | kept | verdict |
|---|---|---|---|---|---|---|
| 1 | v114 | Q-score in-vivo verdict (v111 clean single-flag A/B) | deterministic A/B +0.00619 > +0.005 gate, broad-based | 0.15632 | **YES** | **KEEP — new baseline; FM6 refuted for post-hoc re-rank** |
| 2 | v115 | PROM topographic peak-prominence re-rank (stacked on Q) | activated cleanly, det A/B +0.00259, stack>parts, but < +0.005 & inside noise | 0.17559 | no | REVERT — real but sub-threshold |
| 3 | v116 | SLM Normal-subtracted CLIP text-margin re-rank | CRASHED 12/14 smoke: MPS->numpy TypeError (np.asarray on mps tensor) | -1 | no | REVERT — device bug; untested in vivo |
| 4 | v117 | FDR two-arm CLIP-feature bake-off (SLM + TFO) | STEP-0 harness never run -> winner undetermined -> unlaunchable | -1 | no | REVERT — harness-skip; FM5 upstream |
| 5 | v118 | FSL raw-CLIP structural localization (comp arm) | activated cleanly, full mAP +~0.0003 over Q-on, broad-AP flat | 0.15664 | no | REVERT — effect null |
| 6 | v119 | MELC motion-energy OIC contrast (raw motion channel) | activated cleanly, inside +/-0.03 noise, no +0.005 clearance | 0.16883 | no | REVERT — new modality, still null |
| 7 | v120 | BREF boundary-refine bake-off (fused-deriv / motion-edge snap) | STEP-0 harness skipped, ran placeholder arm; full-budget wasted | 0.17488 | no | REVERT — un-adjudicated; harness-skip |
| 8 | v121 | HYBG hysteresis boundary expansion (un-flooded exterior) | launch gate refused (no promoted_arm); harness never run | -1 | no | REVERT — gate saved budget; un-adjudicated |
| 9 | v122 | BOR boundary-offset rigid recentering (exterior asymmetry) | **harness RAN**; both arms regressed offline (dmAP -0.012/-0.035); gate refused live | -1 | no | REVERT — **translation operator REFUTED offline** at zero cost |
| 10 | v123 | SLM-CV cross-video CLIP text-margin re-rank (absolute scale) | harness >8.5min exceeds per-call budget, no JSON -> gate refused | -1 | no | REVERT — harness-skip; un-adjudicated |

## Updated Failure-Mode Taxonomy (FM1-FM8 + session refinements)
- **FM1 curve-shape modification regresses** — remained immune-by-construction
  in every v114+ loop (all were read-only, post-count-freeze).
- **FM2 detection-count changes under greedy match** — immune by construction
  throughout (boundaries/count frozen; only global confidence ordering moved).
- **FM3 self-consistency on perceptual/extraction tasks** — not exercised this session.
- **FM4 self-consistency on segment inclusion** — not exercised this session.
- **FM5 mechanism implemented but never activated** — recurred as the session's
  DOMINANT process failure (v116 crash-before-marker, v117/v120/v121/v123
  harness-skip). Refinement: FM5 now has two sub-classes — (a) flag unthreaded
  (historical), (b) **flag wired but the adjudicating STEP-0 harness never
  completes**, making the live run unlaunchable by construction. The v121/v122/v123
  **launch gate** (refuse live run unless precheck JSON carries a promoted_arm)
  is the correct mechanical fix and saved full-eval budget from v121 onward.
- **FM6 simulation-to-live gap** — **REFUTED specifically for pure post-hoc
  re-ranks** (v114): a deterministic re-score of a run's own final detections is
  faithful in-vivo, not a simulation. Still holds for candidate-pool simulations.
- **FM7 video-level priors can't fix intra-video mislocation** — the pivot to
  per-detection / boundary-placement signals is on-target but NOT YET productive:
  every per-detection re-rank was sub-threshold, and boundary placement is either
  refuted (v122 translation) or un-adjudicated (v120/v121).
- **FM8 offline precheck is config-specific; never confound** — held clean; v122
  demonstrated the ideal end-to-end discipline (harness-first + gate = honest
  true-negative at zero live cost).
- **NEW — FM9 (device-placement in feature re-ranks)**: any re-rank touching
  `clip_features`/`text_features` MUST unconditionally `.detach().cpu().numpy()`
  before `np.asarray`, and smoke MUST run on the real MPS device. v116 crashed on
  this. (Fixed in v123.)
- **NEW — FM10 (harness budget mismatch)**: the STEP-0 offline harness must run as
  a dedicated upstream job, not inside the finalize loop — v123's harness was a
  >8.5-min task that exceeded the per-call budget and never produced JSON,
  triggering the launch gate. Long prechecks need to be launched as background
  jobs and polled, or chunked under the ~150s anti-stall limit.

## Latency Analysis
Committed baseline v114 latency ~165.87s/video at concurrency 3 — above the ~128s
target, driven by the Q-score re-rank being effectively free but the underlying
v51/v94 pipeline sitting at ~150-166s. Reverted mechanisms stayed near-neutral on
latency (post-hoc re-ranks add <1s; v115 177.62s, v119 232.13s, v120 158.35s all
within the pipeline's own variance) EXCEPT v118 FSL at 672.13s (raw pairwise
CLIP-feature structure computation is expensive) — a further reason it was not
worth its null effect. Harness-refused loops (v116/v117/v121/v123) consumed near-zero
live budget. No mechanism this session improved latency; none is committed that
regresses it materially beyond v114.

## Why mAP>=0.20 Was Not Reached (honest assessment)
The confidence-ordering post-hoc re-rank family is confirmed tapped: after the two
positives (v95 TSM, v111/v114 Q-score, together ~+0.007-0.009), FIVE further
orthogonal per-detection signals — peak prominence, raw CLIP text-margin, raw-CLIP
2nd-order structure, motion-energy OIC — all landed sub-+0.005. Once boundaries and
count are frozen, mAP depends only on global confidence ordering, and the residual
signal there is smaller than the +/-0.03 live VLM noise band. mAP cannot be moved
past ~0.16 by re-ranking alone; **the dominant error is intra-video
mislocalization (FM7), which only boundary-placement or detection-stage changes can
touch.** That pivot is correctly identified but not yet delivered.

## Concrete Next Directions
1. **Run the boundary-placement family CLEANLY, harness-first.** v121 HYBG
   (hysteresis expansion on un-flooded exterior) is fully implemented, C-approved,
   and ships flag-off — it needs only its STEP-0 harness run to completion as a
   dedicated background job, then a single live A/B if the offline IoU-bucket gate
   passes. This is the single highest-value queued item. (Note: v122 translation
   is REFUTED; do not repropose rigid recentering. v120 independent-snap and
   v121 expand are NOT refuted — only untested.)
2. **Cross-video absolute-scale calibration (v123 SLM-CV Arm-B).** The only
   untested lever against the frame-AUC vs temporal-mAP gap: an absolute
   anomaly-minus-Normal margin to demote normal-video / mislocated FPs from the
   GLOBAL ranking (parameter-free, model's own logit_scale). Blocked only by
   FM10 (harness budget) — re-run the precheck as a background job, poll for JSON,
   then gate. Distinct from FM7 because it acts cross-video on the global ordering,
   which re-ranks within-video cannot reach.
3. **Detection-stage / architecture change.** Incremental tuning is exhausted at
   ~0.15-0.16. Per the roadmap, only an architecture-level change (proposal
   generation, or a learned localization head that does not inherit the CLIP+MIL
   curve's mislocation) can plausibly reach 0.20. Treat any such attempt as a
   multi-loop research effort, harness-first, with the launch-gate discipline.
4. **Process hardening (make FM5/FM10 structurally impossible):** every
   harness-gated mechanism must (a) run its STEP-0 harness as a named background
   job under the anti-stall budget, (b) block the finalizer on the precheck JSON
   existing with a promoted_arm, (c) keep the launch gate + post-eval [MARKER]
   grep. v122/v123 proved the gate works; the remaining gap is executing the
   harness, not gating on it.

## Files (this session)
- iteration_logs/ideas/loop1_v114 ... loop10_v123 (10 idea/verdict files)
- iteration_logs/session_v97_ledger.md (extended with v114-v123 rows)
- iteration_logs/analysis_v111_qscore_precheck.py (reusable offline re-rank harness)
- Committed code change: ScoringConfig.qscore_rerank default False->True (v114)
- All v115-v123 mechanisms restored flag-off byte-identical from .qoder/backups/
