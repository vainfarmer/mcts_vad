# v154 — JOINT A+B ANALYTICAL CLOSURE (5th convergent reproduction)

**Date:** 2026-08-01
**Loop:** 5 (session v97 ledger row 5)
**Verdict:** ANALYSIS-ONLY closure. ZERO eval/API spent (FM11/FM13, valid outcome cf v135/v140/v149/v150/v151/v152/v153).
**Working tree:** byte-identical to committed v137+BUE-on (bue_enabled=True, cex default-OFF). `grep -iE 'v154|xvlevel_relocate|leadb_relocate|denseoracle'` over config/pipeline/run/batch_evaluate/models = 0. Nothing to roll back.

## Mechanism (proposed)
None shippable. This is a joint closure of the two genuinely-unrefuted leads:
- **Lead A** — cross-video CONFIDENCE-LEVEL calibration: set each video's confidence LEVEL toward a target while freezing intra-video order, using a verdict-independent supervised MIL video prior (model_mil_v6.pth / model_mil.pth max, mean, blend). NOT banned #4 (it sets the level, does not normalize it away) and NOT banned #1 (freezes intra-video order).
- **Lead B** — finer-resolution (dense fps=2) localizer substrate for recall: replace the coarse 32-snippet MIL curve as the source of temporal run proposals, aiming to relocate the ~32 zero/low-IoU wrong-location detections onto the true run without reading a verdict. NOT banned #10 (denser substrate) and NOT banned #5 (no bottom-append).

## Oracle pre-check result (relied upon from bit-stable v150-v153; artifacts re-verified present this session, NO 6th re-run per C-MANDATE)
BASE mAP = 0.21932 (AP@0.3 0.32097 / AP@0.5 0.19014 / AP@0.7 0.14685), reproduced EXACTLY in v150-v153 (FM6 fidelity).

**Lead A (analysis_v150_xvlevel_orch.py):**
- PERFECT level oracle meanIoU mAP 0.23823 (d=**+0.01891**); maxIoU mAP 0.21615 (d=**-0.00317**, NON-MONOTONE).
- Every real verdict-independent prior REGRESSES: v6max -0.04206, v6mean -0.14296, v6blend -0.05593, v1max -0.10106, v1mean -0.13285, v1blend -0.05405.
- spearman(MILv6max,maxIoU)=+0.1411, spearman(MILv1max,maxIoU)=+0.2722 (all <=0.2722) => MIL is anomaly-PRESENCE, near-orthogonal to localization-QUALITY.
- **FAIL:** perfect ceiling +0.01891 < +0.03 and inside +/-0.025 CI; achievable NEGATIVE.

**Lead B (analysis_v150_denseoracle_B.py / analysis_leadB_dense_oracle.py):**
- 59 anomaly videos w/GT. @IoU>=0.3: GT-aware 30/59 | VI-peak 22/59 | VI-mean 22/59 | COMMITTED 27/59.
- Of 32 wrong-loc videos: VI-peak recovers 3, VI-mean 5, GT-aware ceiling 7.
- GT-aware 30/59 BREAKS the v149 32-snippet 0/17 reach wall (positional info physically exists), GT-aware dmAP +0.03222 — but every verdict-independent selector is NET-WORSE (22/59 < 27/59), achievable dmAP +0.00985 (inside noise floor).
- **FAIL:** FM14 selector-quality ceiling; GT-aware ceiling needs GT and sits at the +/-0.025 floor.

**Decomposition:** v138 re-rank oracle +0.05180 = banned within-video Q-reorder (family 1, ~+0.006) + pure cross-video-LEVEL (perfect +0.01891) + residual GT-only interaction. No verdict-independent prior realizes the level or the residual.

## Effect size vs actual
- Proposal expected_map_gain = 0 (closure). Perfect-A +0.01891, achievable-A NEGATIVE, achievable-B +0.00985. Joint realizable +0.0288 < +0.03. No admissible >=+0.03 verdict-independent floor-safe space remains.
- ACTUAL: no eval run (precheck FAIL). ABSOLUTE mAP stays **0.21932 > floor 0.21432**.

## Bootstrap CI + ABSOLUTE mAP vs floor (HEADLINE)
- BOOTSTRAP: null (no eval). ABSOLUTE mAP unchanged at **0.21932**, comfortably above KEEP FLOOR 0.21432.

## FM1-FM19 table
- FM1 curve regress: IMMUNE (no edit). FM2 additive/match-set FPs: IMMUNE for tree; DEMONSTRATED by B's VI relocation 27->22. FM3 VLM bleed: IMMUNE. FM4 self-consistency: IMMUNE. FM5 never-activated: N/A. FM6 offline-doesnt-transfer: RESPECTED (oracle on fidelity-checked cache, BASE reprints 0.21932; even oracle fails). FM7 video-priors-cant-fix-intra-loc: BINDING for A (presence vs quality, spearman<=0.2722). FM8 one-change: N/A (0 changes). FM9 MPS: N/A. FM10 wall-clock: N/A. FM11 one-eval economy: HONORED (zero eval, no 6th re-run). FM12 drop-null-calls: N/A. FM13 structural/resolution cap: THE CLOSURE (A oracle-cap +0.01891, B selector-cap +0.00985). FM14 selector-quality: BINDING for B (3-5/32 vs GT-aware 7/32; 22<27/59). FM15 substrate infidelity: AVOIDED (features/test + cached fps=2). FM16 noise floor: both effects inside +/-0.025. FM17 empty-video marker: N/A. FM18 paired-vs-floor: tree stays 0.21932 > 0.21432. FM19 count-neutral-not-floor-safe: DEMONSTRATED by B (relocation demotes matched dets, 22<27/59).

## Is this lead WON / OPEN / CLOSED?
- **Lead A: CLOSED** by signal orthogonality + oracle cap (+0.01891 perfect, achievable negative).
- **Lead B: CLOSED** at current verdict-independent selectors by FM14 selector-quality ceiling (GT-aware headroom real but unreachable without GT and at the noise floor).
- Both closed under the current n=65 harness and current shipped signals.

## What the next loop should NOT repeat
1. Do NOT schedule a 6th A/B re-verify — oracle-closed, reproduced bit-for-bit across v150-v154; marginal value is now near zero.
2. Do NOT re-propose Lead A: no verdict-independent supervised prior tracks the level oracle (all six blends regress).
3. Do NOT re-open Lead B with half-max/argmax selectors (caps 3-5/32 -> +0.00985) or treat +0.079/+0.03222 GT-aware AP@0.3 as reachable — it needs GT.
4. Do NOT re-propose 32-snippet relocation (v149 closed).
5. Do NOT spend eval on a pre-measured null (FM11).

## Only residual open door (escalate to owner)
A NEWLY-TRAINED verdict-independent localization-QUALITY / event-completeness head (ICML2025 'Learning Event Completeness for WS-VAD'; STPrompt; CVPR2025 PseudoFormer / Bridge-the-Gap proposal-level pseudo-label regression). Caveats: multi-loop UPSTREAM training project, NOT a single drop-in; un-precheckable on cached artifacts (FM6/FM13); even a PERFECT head is bounded by the n=65 +/-0.025 noise floor (Lead B's GT-aware +0.03222 ceiling currently UNFALSIFIABLE). Owner should choose: (i) build a larger/denser eval harness first to lower the noise floor and re-open Lead B's GT-aware headroom as a measurable target; or (ii) commission the localization-quality/event-completeness head as an authorized multi-loop project. PAUSE the single-loop closure cycle until one is authorized.
