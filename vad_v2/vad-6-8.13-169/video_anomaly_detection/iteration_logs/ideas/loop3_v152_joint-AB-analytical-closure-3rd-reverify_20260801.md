# v152 — JOINT A+B ANALYTICAL-CLOSURE (3rd independent re-verify)

Date: 2026-08-01 | Loop 3 | Verdict: **ANALYSIS-ONLY CLOSURE — both leads CLOSED, ZERO eval/API spent** | Kept: **false** (nothing shipped; tree stays committed BUE-on 0.21932)

## Mechanism (NONE — this is a closure, not a shippable change)
Re-audited to closure the two genuinely-unrefuted optimization leads for raising full-eval temporal mAP on the fixed 65-video harness. No source/config/CLI/model/marker files touched; working tree byte-identical to committed v137+BUE-on (mAP 0.21932). Both leads re-run TODAY as zero-API offline oracles on the committed BUE-on cache `results/batch_eval_v144c_bue65ref`.

- **Lead A** — cross-video CONFIDENCE-LEVEL calibration: set each video's confidence LEVEL to a target while preserving intra-video order, using a verdict-independent supervised anomaly prior (model_mil_v6.pth / model_mil.pth video max/mean). NOT banned-4 (that normalizes level AWAY), NOT banned-1 (that is within-video re-rank).
- **Lead B** — finer-resolution (dense fps=2 CLIP/MIL) localizer substrate for the after-BUE relocation TARGET only, to recover the 50/99 zero-overlap dets.

## Oracle pre-check result (MANDATORY FM13, re-run 2026-08-01)
BASE mAP reprinted EXACTLY 0.21932 (AP@0.3 0.32097 / AP@0.5 0.19014 / AP@0.7 0.14685) => substrate fidelity confirmed (FM6).

**Lead A** (`analysis_v150_xvlevel_orch.py`): PASS gate = perfect level oracle > +0.03 AND >=1 real prior positive.
- PERFECT level oracle: meanIoU mAP 0.23823 (d=**+0.01891**), maxIoU mAP 0.21615 (d=**-0.00317**, NON-MONOTONE).
- EVERY real verdict-independent prior REGRESSES: v6 max -0.04206, v6 mean -0.14296, v6 blend -0.05593, v1 max -0.10106, v1 mean -0.13285, v1 blend -0.05405.
- spearman(MILv6max, maxIoU)=+0.1411; spearman(MILv1max, maxIoU)=+0.2722 => anomaly-PRESENCE vs localization-QUALITY orthogonality (FM7).
- **FAIL both conditions.** Perfect ceiling +0.01891 < +0.03 and inside +/-0.025 CI.

**Lead B** (`analysis_v150_denseoracle_B.py`): PASS gate = VI selector beats ~0/17 wrong-loc AND maps to > +0.03.
- 59 anomaly videos w/GT. @IoU>=0.3: GT-aware 30/59 | VI-peak 22/59 | VI-mean 22/59 | committed 27/59.
- @IoU>=0.5: GT-aware 21/59 | VI-peak 15 | VI-mean 14 | committed 16/59.
- 32 wrong-loc videos: VI-peak recovers 3, VI-mean 5, GT-aware ceiling 7 @0.3 (VI-peak 1 / VI-mean 3 / GT-aware 4 @0.5).
- **FAIL.** GT-aware 30/59 BREAKS the v149 32-snippet 0/17 reach wall (recall info EXISTS on denser track), but every VI selector is NET-WORSE overall (22 < 27/59); achievable dmAP ~+0.00985 (loop-17 fps=2), GT-aware ceiling +0.03222 is UNREACHABLE and sits at the noise floor. FM14 selector-quality ceiling.

## Effect-size vs actual
- Expected map gain (proposal): 0 (analytical closure).
- Perfect-A +0.01891 + achievable-B +0.00985 = **+0.0288 < +0.03**; achievable-A is NEGATIVE so realizable joint < achievable-B alone.
- Decomposition: v138 full re-rank oracle +0.05180 = banned within-video Q-reorder (~+0.006, family 1) + pure cross-video-LEVEL (perfect ceiling +0.01891). No un-banned combination clears +0.03.

## BOOTSTRAP CI + ABSOLUTE mAP vs floor (HEADLINE)
No eval spent (both prechecks fail the +0.03 gate, FM11/FM13). No bootstrap. **ABSOLUTE mAP stays 0.21932 > keep floor 0.21432.** Nothing shipped.

## FM1-FM19 table
| FM | Status |
|----|--------|
| FM1 curve-shape | IMMUNE (no edit) |
| FM2 additive/count/match-set | IMMUNE (no code path) |
| FM3 VLM context-bleed | IMMUNE (no VLM) |
| FM4 self-consistency | IMMUNE |
| FM5 never-activated | N/A (no flag/marker) |
| FM6 offline-doesnt-transfer | RESPECTED — oracles on fidelity-checked cache (BASE reprinted 0.21932); even oracle fails so no materialization risk |
| FM7 video-priors-cant-fix-intra | BINDING for A (presence vs quality orthogonality) |
| FM8 one-change | N/A (zero change) |
| FM9 MPS detach | N/A (CPU; MIL already .cpu().numpy()) |
| FM10 wall-clock | N/A |
| FM11 one-eval economy | RESPECTED — zero eval on predetermined null |
| FM12 | N/A |
| FM13 structural/resolution cap | THE CLOSURE (A oracle-capped +0.01891 / B selector-capped) |
| FM14 verdict-visibility/selector-quality | BINDING for B (3-5/32 vs GT-aware 7/32) |
| FM15 substrate infidelity | AVOIDED (features/test + cached fps=2 = correct MIL substrate) |
| FM16 noise floor | even perfect-A +0.01891 and achievable-B +0.00985 inside +/-0.025 |
| FM17 empty-video marker | N/A (no marker) |
| FM18 paired-vs-absolute floor | no arm below floor; tree stays 0.21932 |
| FM19 count-neutral-not-floor-safe | acknowledged (why v148 position-moving is banned-10) |

## Lead status
- **Lead A (cross-video LEVEL calibration): CLOSED** — hard PERFECT oracle cap +0.01891 (non-monotone), every real verdict-independent prior regresses (spearman<=0.27 presence-vs-quality orthogonality). No verdict-independent prior tracks the level oracle.
- **Lead B (denser-substrate recall): CLOSED at current selector** — FM14 selector-quality ceiling; recall INFO exists (GT-aware 30/59 breaks v149 wall, fps=2 GT-aware +0.03222) but no verdict-independent selector converts it (3-5/32 wrong-loc, net-worse 22<27/59, achievable +0.00985 inside noise floor).

## What the next loop MUST NOT repeat
1. Do NOT re-propose Lead A: no verdict-independent supervised prior tracks the localization-QUALITY level oracle (perfect ceiling only +0.01891, all real priors negative).
2. Do NOT re-open Lead B with the half-max / argmax heuristic (caps 3-5/17-32 wrong-loc -> +0.00985) or treat +0.079/+0.03222 GT-aware AP@0.3 as reachable without GT.
3. Do NOT re-propose 32-snippet supervised-MIL relocation (v143-v149 closed) — signal ceiling FM13.
4. Do NOT spend any eval on a pre-measured null (v135/v140/v149/v150/v151/v152 all zero-eval closures).
5. Any Lead B re-open MUST first raise a NEW trained verdict-independent localization-QUALITY selector (e.g. CVPR2025 Bridge-the-Gap proposal-level pseudo-label regression) MATERIALLY above 3-5/32 wrong-loc on `analysis_v150_denseoracle_B.py` AND lift the GT-aware reach ceiling clear of the +/-0.025 noise floor BEFORE any config/eval. This is a multi-loop upstream head, not a drop-in pre-calibratable prior; escalate to owner.

## Artifacts
- Precheck: `results/v152_precheck.txt`
- Oracles (reused, unchanged): `analysis_v150_xvlevel_orch.py`, `analysis_v150_denseoracle_B.py`
- No config/pipeline/run/batch_evaluate/models edits; no `.qoder/backups/v152` needed; no git checkout/reset/stash/clean used.
