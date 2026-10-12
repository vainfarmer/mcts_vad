# v148 — TSAR: Tri-Signal Agreement Relocation (REVERT)

Date: 2026-08-01
Family: supervised-MIL count-neutral relocation (recall/candidate-generation)
Verdict: **REVERT — FM18 ABSOLUTE FLOOR BREACHED + all three KEEP gates FAILED.**
Committed baseline unchanged: v137 + BUE (default-ON), absolute mAP 0.21932, floor 0.21432.

## HEADLINE (bootstrap CI + absolute mAP vs floor 0.21432)
Paired-json bootstrap on `pre_tsar_boundaries` (n=65 common videos, 1000 resamples, 37 of 101 index-aligned pre-mechanism rows changed):
- **point_delta = -0.08550** (REF within-run arm 0.21138 -> NEW 0.12588)
- **CI90 [-0.13862, -0.00193]** (entirely below zero)
- **P(delta>0) = 0.037**
- **NEW absolute mAP = 0.12588 << floor 0.21432** (short by 0.08844) and far under committed 0.21932.
- GATE verbatim: point_delta>=0.01 **False (-0.08550)** / P>=0.9 **False (0.037)** / mAP>=floor **False (0.12588)**. Three of three fail.

NEW arm regressed on EVERY IoU threshold vs REF: AP@0.3 0.2375 vs 0.3107, AP@0.5 0.0758 vs 0.1795, AP@0.7 0.0644 vs 0.1439.

## MECHANISM
Adds ONE genuinely-new verdict-independent localizer — snippet-to-snippet CLIP temporal-delta d[i]=||f[i]-f[i-1]|| (motion/novelty) — as an INDEPENDENT third signal alongside the cached v1 MIL curve (model_mil.pth, video_data['mil_scores']) and a read-only v6 MIL curve (model_mil_v6.pth, second read-only inference, no fusion into scoring). In an after-BUE terminal slot, a far zero-overlap detection (center-gap/dur >= tsar_far_gap_frac) is width-preservingly TRANSLATED onto an agreed run only when >=2 of the 3 signals' max-mean above-fraction run centers cluster within tsar_agree_k snippets. conf/count/rank/type frozen. Params (config.py, default OFF): tsar_enabled=False, tsar_agree_k=4, tsar_far_gap_frac=0.15, tsar_min_agree=2, tsar_run_frac=0.25. CLI --tsar/--no_tsar in run.py + batch_evaluate.py.

## EFFECT-SIZE ARITHMETIC: CLAIMED vs ACTUAL
- Proposer's own honest pre-registration: expected +0.005 to +0.015 (below the +0.03 mandate and below the +/-0.025 n=65 noise floor). Analytic ceiling REFUTED +0.03: on the 17 profiled wrong-loc videos the GT-aware best-IoU-run oracle over all 3 signals = 11/17 @IoU>=0.3, but the verdict-independent (no-GT) max-mean selector — 2-of-3 agreement AND max-prominence — collapses to **0/17 @IoU>=0.3**. temporal-delta as a run target: argmax 7/17 but half-max run only 2/17 (unusable run shape).
- ACTUAL: not merely null — **actively negative (-0.08550)**. The mechanism FIRED (37 of 101 rows moved) and moved windows the WRONG way: the agreement selector placed windows off-GT far more often than the profiling implied, degrading AP at every threshold. The floor-safe monotone guarantee was ANNULLED because the paired REF was the within-run BUE-off/pre-mechanism arm (0.21138), NOT the committed BUE-on pipeline — so "after-BUE frozen-rank" did not make NEW >= 0.21932; instead the relocations demoted real matches.

## WHY THE "MONOTONE / SAFE NO-OP" GUARANTEE DID NOT HOLD
The proposal claimed after-BUE + frozen count/conf/rank => global greedy AP monotone-nondecreasing => absolute mAP >= 0.21932 by construction. That is FALSE when the mechanism actually MOVES windows: translating a detection off a run it (even partially) overlapped onto an "agreed" but wrong run DEMOTES a matched detection to unmatched. Monotonicity would only hold if moves were restricted to truly-zero-overlap dets AND every target strictly improved IoU — neither was enforced in vivo. 37 moves produced a -0.085 collapse. The construction argument was wrong: count/conf-freezing does NOT protect AP when window POSITIONS change adversarially.

## FM1–FM18 TABLE (as realized)
- FM1 curve-shape regress: IMMUNE (curves read-only, scoring untouched) — held.
- FM2 additive-FP: count/conf frozen, no NEW dets — held, but IRRELEVANT: harm came from MOVING existing matched dets off GT, not from added FPs.
- FM3 VLM bleed: N/A (no VLM).
- FM4 normalization: IMMUNE (within-video peak geometry) — held.
- FM5 activation: CLEAN — 14 smoke / 65 full [TSAR] summary markers == succeeded videos; grep gate passed.
- FM6 sim->live transfer: **MATERIALIZED, worst case.** offline verdict-independent 0/17 predicted null; live delivered -0.085 (worse than null — the agreement gate placed windows destructively).
- FM7 video-prior: N/A.
- FM8 one change: single relocation method (delta as its input).
- FM9 MPS: v6 forward used .detach().cpu().numpy(); v1 already .cpu().numpy() — held.
- FM10 wall-clock: one extra cached MIL forward, negligible (95.22s/vid full).
- FM11 one-full-eval: full eval run (against proposer's own abort recommendation).
- FM12: N/A.
- FM13 magnitude: CONFIRMED FATAL — GT-oracle 11/17 collapses to 0/17 verdict-independent; the +0.062 snap-oracle needs GT to place runs.
- FM14 verdict-visibility: N/A (count-neutral).
- FM15 substrate: MIL + delta on features/test is their true substrate — not the LIVE-CURVE caveat.
- FM16 noise floor: BINDING — honest estimate below +/-0.025 CI half-width; the realized effect was large but NEGATIVE.
- FM17 empty-video marker: FIXED & VERIFIED — [TSAR] summary fires on no_dets/no_mil/flat_curve/no_agree/no_far incl. Robbery050; report written, no [GATE] deadlock.
- FM18 paired-vs-absolute-floor: **KILLER, again.** NEW absolute mAP 0.12588 < floor 0.21432. The "after-BUE => absolute >= 0.21932 by construction" claim was refuted in practice — moving windows broke the monotone guarantee.

## SENTINELS
The pre-registered AGREEMENT-FIRE-BAND sentinel was supposed to ABORT before eval (proposer predicted 0/17 verdict-independent fire at useful locations). In vivo the agreement gate FIRED at scale (37 moves) rather than aborting — the offline 17-video profile did NOT predict the in-vivo fire population, and the MONOTONE-CHECK per-video sentinel (per-video AP >= pre-move) evidently was not enforced as a hard abort (a -0.085 outcome should have tripped it). Process gap: the honest "abort before eval" recommendation was overridden and a full eval was spent to confirm harm.

## WEAKNESSES / ROOT CAUSE
The binding constraint is TEMPORAL RESOLUTION of the relocation localizer on 32-snippet features: the correct GT-overlapping run EXISTS (GT-aware oracle 11/17) but cannot be IDENTIFIED without GT (verdict-independent 0/17). No model/gate/ensemble over v1+v6+delta 32-snippet curves fixes this. Worse, an aggressive agreement gate that DOES fire relocates matched dets destructively (-0.085). The supervised-MIL count-neutral relocation family is now empirically CLOSED over ALL available 32-snippet signals: v143 (additive floor-breach -0.018), v144 (crash), v145 (null, 3 fires), v146 (+0.036 point but P=0.876 + below floor), v147 (0 fires, artifact delta), v148 (fires at scale, -0.085 destructive).

## WHAT THE NEXT LOOP MUST NOT REPEAT
1. Do NOT bring a fourth ensemble/gate/prominence trick over the SAME 32-snippet v1/v6/delta curves — all now measured null-or-harmful.
2. Do NOT trust "count/conf-freeze => monotone AP => >= floor by construction": MOVING window POSITIONS breaks monotonicity when targets are wrong-but-agreed. The construction only holds if every move strictly non-decreases per-video IoU, which cannot be guaranteed verdict-independently on coarse features.
3. Do NOT proceed to a full eval when the proposer's own pre-registered sentinel says ABORT and the offline verdict-independent oracle is ~0 — the eval only confirmed (here, worse than) a pre-measured null.
4. Do NOT pair the mechanism field against a within-run pre-BUE/low arm and read a positive paired delta as a win (FM18): target ABSOLUTE mAP > 0.21932.
5. Do NOT re-open Normal-video FP suppression (banned family 9, oracle +0.00253).

## PIVOT
The only unrefuted in-family lever is a FINER-RESOLUTION localizer substrate for the after-BUE relocation TARGET ONLY (denser >32-snippet feature/actionness track), main pipeline 32-snippet scoring byte-identical. MANDATORY FIRST STEP (FM13): re-run the 17-video (ideally full-65) GT-aware-vs-verdict-independent run-IoU oracle on the DENSER substrate. Only propose a full eval if the no-GT selector clears enough conversions to argue >=+0.03 absolute over 0.21932 AND large enough for P>=0.9 at n=65. If the denser-substrate verdict-independent oracle is still ~0, DECLARE THE RELOCATION FAMILY CLOSED and pivot out of relocation entirely. Reuse this scaffold verbatim (after-BUE slot, pre_tsar_boundaries-style paired field, FM17 empty-path marker, count-neutral freeze) — swap ONLY the substrate resolution — BUT add a hard per-video MONOTONE-abort so a destructive move like -0.085 cannot ship.

## ROLLBACK CONFIRMATION
Restored all 5 files from /Users/zzl/PycharmProjects/vad/.qoder/backups/v148/ (diff -q byte-identical x5). grep -iE 'tsar|TSAR|pre_tsar|_tsar_relocate|mil_scores_v6' == 0 across all 5. py_compile PASS x5. bue_enabled=True, cex_extend=False intact. Nothing shipped; baseline stays v137+BUE (0.21932, floor 0.21432).
