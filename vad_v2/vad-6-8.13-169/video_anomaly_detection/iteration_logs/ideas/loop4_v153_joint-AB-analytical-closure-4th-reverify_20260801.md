# v153 — JOINT A+B ANALYTICAL CLOSURE (4th independent re-verification)

**Loop:** 4 (session v97 ledger) · **Date:** 2026-08-01 · **Verdict:** ANALYSIS-ONLY closure, ZERO eval/API spent (valid outcome, cf. v135/v140/v149/v150/v151/v152)
**Baseline (committed v137+BUE-on):** mAP **0.21932** (AP@0.3 0.32097 / AP@0.5 0.19014 / AP@0.7 0.14685) · absolute keep floor 0.21432
**Headline:** No admissible >= +0.03 verdict-independent, floor-safe lever remains. Both remaining unrefuted leads re-verified on the live committed BUE-on cache THIS SESSION and both FAIL the +0.03 admissibility gate. Working tree byte-identical to committed. NOT kept (nothing to keep — no mechanism shipped).

## Mechanism
No shippable change proposed. This loop is a joint closure of the two remaining unrefuted leads, exercised as zero-API deterministic oracles on the live committed BUE-on cache (`results/batch_eval_v144c_bue65ref` + `features/test`):
- **Lead A — cross-video CONFIDENCE-LEVEL calibration:** set each video's confidence LEVEL to a target while preserving intra-video order, using a verdict-independent supervised MIL anomaly prior.
- **Lead B — finer-resolution recall:** use a denser fps=2 CLIP/MIL track to relocate the zero-overlap dets that the 32-snippet curve could not (v149's 0/17 wall).

Both were exercised through their PERFECT GT-aware oracle AND every achievable verdict-independent selector.

## Oracle pre-check result (re-run THIS session, bit-for-bit reproduced)
BASE reprinted EXACTLY 0.21932 on both scripts (FM6 substrate fidelity).

**Lead A** (`analysis_v150_xvlevel_orch.py`):
- PERFECT level oracle: meanIoU mAP 0.23823 (**d=+0.01891**); maxIoU mAP 0.21615 (**d=-0.00317**, NON-MONOTONE/NEGATIVE) — ceiling < +0.03 and inside +/-0.025 CI.
- EVERY realizable verdict-independent prior REGRESSES: v6 max -0.04206, v6 mean -0.14296, v6 blend -0.05593, v1 max -0.10106, v1 mean -0.13285, v1 blend -0.05405.
- spearman(MILv6max, maxIoU)=+0.1411, (MILv1max, maxIoU)=+0.2722 (all <=0.2722) => MIL is an anomaly-PRESENCE detector, near-orthogonal to localization-QUALITY (FM7 binding). **CLOSED.**

**Lead B** (`analysis_v150_denseoracle_B.py`, 59 anomaly videos w/GT):
- @IoU>=0.3: GT-aware 30/59 | VI-peak 22/59 | VI-mean 22/59 | committed 27/59
- @IoU>=0.5: GT-aware 21/59 | VI-peak 15/59 | VI-mean 14/59 | committed 16/59
- of 32 wrong-loc (committed<0.3): VI-peak recovers 3, VI-mean 5, GT-aware ceiling 7 @0.3 (1/3/4 @0.5)
- GT-aware 30/59 BREAKS the v149 32-snippet 0/17 reach wall (recall INFO exists on denser track) but every verdict-independent selector is NET-WORSE (22<27/59); achievable dmAP ~+0.00985 (inside noise floor); GT-aware ceiling +0.03222 sits AT the +/-0.025 floor and needs GT. FM14 selector-quality ceiling. **CLOSED at current selector.**

**JOINT:** perfect-A +0.01891 + achievable-B +0.00985 = +0.0288 < +0.03, and achievable-A is NEGATIVE => realizable joint < achievable-B alone. Neither precheck clears +0.03 => ZERO eval spent (FM11/FM13).

## Effect-size vs actual
Claimed/expected gain: 0 (closure). Actual: no mechanism, no eval. Decomposition proof: v138 +0.05180 re-rank oracle = banned within-video Q-reorder (family 1, ~+0.006) + pure cross-video-LEVEL component (perfect +0.01891). All other levers are one of the 10 banned families.

## Bootstrap CI + ABSOLUTE mAP vs floor
No bootstrap (zero eval). Working tree ABSOLUTE mAP stays committed 0.21932 > floor 0.21432. No arm below floor because nothing ran.

## FM1–FM19 table
- FM1 curve-shape: IMMUNE (no edit). FM2 additive/count/match-set: IMMUNE (no code path; additive-append escape for B explicitly rejected as banned #5). FM3 VLM context-bleed: IMMUNE. FM4 self-consistency: IMMUNE. FM5 never-activated: N/A (no flag/marker). FM6 offline-doesnt-transfer: RESPECTED — oracles on fidelity-checked cache, BASE reprinted 0.21932 THIS session; oracle itself fails so no materialization risk. FM7 video-priors-cant-fix-intra: BINDING for A (presence vs quality orthogonality, spearman<=0.2722) and B (dense argmax off-GT). FM8 one-change: N/A (0 changes). FM9 MPS detach: N/A. FM10 wall-clock: N/A. FM11 one-eval economy: HONORED — zero eval on predetermined null. FM12: N/A. FM13 structural/resolution cap: THE CLOSURE (A oracle-capped +0.01891, B selector-capped). FM14 selector-quality: BINDING for B (3-5/32 vs GT-aware 7/32; net 22<27/59). FM15 substrate infidelity: AVOIDED (features/test + cached fps=2). FM16 noise floor: perfect-A +0.01891 and achievable-B +0.00985 both inside +/-0.025. FM17 empty-video marker: N/A (no marker). FM18 paired-vs-absolute floor: no arm below floor; tree stays 0.21932. FM19 count-neutral-not-floor-safe: acknowledged, demonstrated by B (count-neutral relocation demotes matched dets, 22<27).

## Lead status
- **Lead A (cross-video LEVEL calibration): CLOSED** — perfect oracle +0.01891 < +0.03; every realizable verdict-independent prior regresses (presence/quality orthogonality).
- **Lead B (dense fps=2 substrate): CLOSED at current selectors** — recall info genuinely exists (GT-aware 30/59 > committed 27/59) but no verdict-independent selector converts it net-positively; achievable +0.00985 and even GT-aware ceiling +0.03222 sits at the noise floor.

## What the next loop should NOT repeat
1. Do NOT schedule a 6th A/B re-verification — oracle-closed and reproduced bit-for-bit 4x (v150–v153).
2. Do NOT re-propose Lead A: no verdict-independent prior tracks the level oracle (all six regress; spearman<=0.2722).
3. Do NOT re-open Lead B with half-max/argmax/percentile-swept selectors (caps 3-5/32 -> +0.00985); do NOT treat GT-aware AP@0.3 (+0.079) or GT-aware ceiling (+0.03222) as reachable.
4. Do NOT re-propose 32-snippet relocation (v149 closed) or any banned family.
5. Do NOT spend eval on a pre-measured null (FM11).

## Only residual door (escalate to owner)
A NEWLY-TRAINED verdict-independent localization-QUALITY head (proposal-level pseudo-label completeness regression, PseudoFormer / Bridge-the-Gap CVPR2025 style) — a multi-loop upstream training project, not a drop-in pre-calibratable prior for this loop, and even a perfect one only reaches the +/-0.025 noise-floor ceiling. Alternatively, a larger/denser eval harness to lower the +/-0.025 noise floor so sub-0.03 levers become falsifiable.

## Artifacts (read-only, re-run today)
- `iteration_logs/analysis_v150_xvlevel_orch.py` (Lead A) — /tmp/v153_leadA.log
- `iteration_logs/analysis_v150_denseoracle_B.py` (Lead B) — /tmp/v153_leadB.log
- `iteration_logs/results/v153_precheck.txt`
