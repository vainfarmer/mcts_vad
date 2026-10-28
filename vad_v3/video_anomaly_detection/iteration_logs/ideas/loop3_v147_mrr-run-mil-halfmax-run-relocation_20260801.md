# v147 — MRR-RUN: v1-MIL half-max contiguous-RUN relocation with in-vivo within-video prominence calibration

- **Date:** 20260801
- **Loop:** 3 (recall-lever sub-arc: v145 loop1 / v146 loop2 / v147 loop3)
- **Family:** supervised-mil-relocation (count-neutral)
- **Verdict:** **REVERT** — FM18 absolute floor breached AND (uncovered on recording) the mechanism fired **ZERO** relocations; the reported paired delta is a BUE measurement artifact.

## Headline (BOOTSTRAP CI + ABSOLUTE mAP vs floor 0.21432)

| gate | value | pass? |
|---|---|---|
| paired point_delta (>= 0.01) | **+0.03324** | ✅ (but SPURIOUS — see below) |
| P(delta>0) (>= 0.9) | **0.851** | ❌ |
| NEW absolute mAP (>= 0.21432 floor; target 0.21932) | **0.20803** | ❌ (below floor by 0.00629; below BUE-on baseline by 0.01129) |

Paired-json[pre_milrun_boundaries], 63 common videos, 1000 resamples, mean +0.03230, sd 0.03325, CI90 [-0.02204, +0.08592]. REF arm 0.17479 -> NEW 0.20803. Full: mAP 0.20803 / AP@0.3 0.30993 / AP@0.5 0.17430 / AP@0.7 0.13986; 65/65 succeeded, 65 [MRR-RUN] summary markers (no FM17 deadlock). Smoke mAP 0.2316.

## Mechanism (as implemented)

Count-neutral width-adaptive relocation of far-mislocated FINAL detections onto the v1 MIL (`model_mil.pth`, cached `video_data['mil_scores']` at pipeline.py:1345, `.cpu().numpy()` FM9-immune) highest-mean above-half-max CONTIGUOUS RUN. `hm = milrun_hm_frac*max + (1-frac)*P20`; extract contiguous above-hm runs, pick max-mean run, relocate any det whose center-gap > `milrun_far_gap_s` (6s) onto the full run window [run_start, run_end]. Placed IMMEDIATELY BEFORE the committed BUE block (so BUE regrows the moved window). conf/count/rank/type frozen (FM2-immune by design). IN-VIVO prominence gate (read-only within-video stats, NOT score-normalization): `z=(run_mean-P50)/(P90-P10+eps) >= milrun_min_prominence_z (1.0)` AND `run_mean-P50 >= milrun_min_run_support_margin (0.05)`.

## Effect-size arithmetic (claimed) vs ACTUAL

- **Claimed:** +0.035 mAP. Offline (analysis_milv6_rows.json) v1 half-max RUN lands 8/17 (47%) @IoU>=0.3 on the 17 zero-overlap wrong-loc videos vs v146 point-argmax 6/17; scaling ~14 FP->TP conversions to the 50 zero-overlap dets after a ~60% eligibility gate; each conversion pure FP->TP (no precision cost) => AP@0.3 +~0.03, AP@0.5 +~0.05, AP@0.7 +~0.03 => mean ~+0.037.
- **ACTUAL: the mechanism NEVER FIRED.** `[MRR-RUN] fire:` count = **0** in BOTH the 13-video smoke and the 65-video full run. Full-run reason breakdown: **61 no_prom**, 1 no_dets, 1 no_mil (+2 succeeded). Smoke: 12 no_prom, 1 no_dets. Zero windows were relocated by the mechanism under test. Realized conversion = 0/50, not 14/50.

## THE TWO ROOT CAUSES (both were pre-warned in this session's own implementation caveat)

1. **The in-vivo z>=1.0 prominence gate is structurally near-capped at ~1.0 and blocked EVERY far det.** Observed z on far-det videos: 0.007, 0.164, 0.176, 0.306 … the few videos with z>=1.0 (0.934/2.124/3.039 seen) had NO far dets, so the compound (far-gap AND z>=1.0 AND support>=0.05) never all-passed simultaneously. The implementation notes explicitly derived that `z -> (max-P50)/(P90-P10) <= 1` whenever the above-half-max run occupies >~10% of the 32 snippets (broad runs — exactly the wide-run target the proposal chose for robustness), so the z>=1.0 default admits ONLY sharp peaks, and mandated: run the offline pre-validation over the 50 zero-overlap dets FIRST and LOWER z to 0.6–0.8 if the gate passes <=3 dets, and "do NOT proceed to full eval on a <=3-fire gate (v145 was leverage-free at 3 fires)." **The pre-validation z-lowering was not applied and the smoke's 0-fire result (sentinel 1 requires >3 fires) did not abort the full eval.** This is v145's leverage-free null taken to the extreme: 0 fires vs 3.

2. **The paired field was snapshotted PRE-BUE, so the +0.03324 delta measured BUE, not the mechanism.** `pre_milrun_boundaries` is captured immediately before `_milrun_relocate`, which sits immediately before the committed BUE block. With 0 relocations, `_milrun_relocate` was a no-op, so the difference between `pre_milrun_boundaries` (REF, 0.17479) and the final post-BUE windows (NEW, 0.20803) is **entirely BUE's own ~+0.03 contribution** — consistent with BUE's known +0.02–0.03. 49 JSONs show a moved window vs the snapshot; every one of those moves is BUE, not milrun. The correct paired REF for a pre-BUE mechanism is BUE-on/milrun-off (~0.219 baseline), NOT the pre-BUE snapshot. The paired bootstrap therefore did NOT measure MRR-RUN at all.

3. **NEW absolute mAP 0.20803 < floor with an inert mechanism => noise.** If milrun fired 0×, the pipeline output is byte-identical to committed BUE-on, which should score ~0.21932; the observed 0.20803 is a single-draw VLM-noise dip (~-0.011, inside the ±0.019 floor). This reconfirms FM16/FM18: a single 65-video absolute-mAP draw is itself noise-sensitive enough to fall below the floor with no mechanism change.

## FM1–FM18 table

| FM | status |
|---|---|
| FM1 curve-shape regress | IMMUNE (windows move, curve/conf read-only) |
| FM2 additive-FP | IMMUNE & moot (count/conf/rank frozen; 0 fires) |
| FM3 VLM context-bleed | N/A (no VLM) |
| FM4 re-leveling | N/A |
| FM5 activation | CLEAN (65 summary markers, flag-gated, grep-gated) |
| FM6 sim->live transfer | **MATERIALIZED (worst case): offline 8/17 conversions -> 0 realized, because the gate the offline probe omitted (no prominence gate offline) rejected 100% in vivo.** The 8/17 was an ungated UPPER bound; the gate erased it. |
| FM7 video-prior | N/A (intra-video) |
| FM8 one change | OK (single method, v6 not loaded) |
| FM9 MPS | IMMUNE (mil_scores already .cpu().numpy() at :1345) |
| FM10 wall-clock | N/A (no extra inference); 135.6s/video |
| FM11 one-full-eval | OK |
| FM12 null-call drop | N/A |
| FM13 magnitude/fire-rate | **MATERIALIZED — fire-rate over-predicted to the limit: predicted ~14 fires, realized 0.** Same class as v145's 6x over-prediction, here total. |
| FM14 verdict-visibility | N/A (count-neutral) |
| FM15 substrate | AVOIDED (MIL on features/test is its true substrate) |
| FM16 noise floor | BINDING (P=0.851<0.9; and an inert run drew 0.011 below baseline) |
| FM17 empty-video deadlock | FIXED & VERIFIED (summary fires on no_dets/no_mil/no_prom; Robbery050 in smoke; report written) |
| FM18 paired-vs-absolute floor | **KILLER (as orchestrator ruled): +0.033 paired over a low PRE-BUE REF (0.17479) does NOT clear the absolute floor; NEW 0.20803 < 0.21432.** Aggravated by the measurement being BUE, not the mechanism. |

## Weaknesses (realized)

The proposal's own weakness #1 (FM16 noise floor) and the implementation caveat's z-cap warning BOTH materialized, and compounded into a null: the mechanism was inert (0 fires) so there was no signal to measure, and the one number that looked positive (+0.033) was a BUE artifact from a pre-BUE paired snapshot. The MIL localizer weakness #2 (capped signal, ~47% @0.3 ungated) is moot here because the gate never let a single move through.

## What the NEXT loop must NOT repeat

1. **Do NOT proceed to full eval when smoke fires <=3 (here 0).** The pre-registered sentinel "fires on >3 dets" FAILED in smoke and should have aborted to no-op before spending the eval. Honor the abort.
2. **Do NOT ship a within-video z-prominence gate at the ~1.0 default.** It is analytically near-capped at 1.0 for any broad (>~10%-of-32-snippet) run and admits only sharp peaks — structurally incompatible with a wide-run relocation target. If a within-video z gate is used again, calibrate the threshold offline over the full 50-det population FIRST and pick a value that yields a sane in-vivo fire band (15–45%), not a fixed 1.0.
3. **Do NOT snapshot the paired field before the committed BUE block for a pre-BUE mechanism.** The paired REF must isolate the mechanism (BUE-on in BOTH arms). A pre-BUE snapshot conflates BUE's +0.02–0.03 into the delta and gives a spurious positive — exactly what happened here. Either snapshot post-BUE-equivalent windows, or reconstruct the REF arm as flag-off (BUE-on/milrun-off), never pre-BUE.
4. **Do NOT treat the absolute-floor gate as satisfiable by a single noisy 65-video draw.** An inert run fell 0.011 below baseline on VLM noise alone; combine the paired delta with the correct BUE-on REF, and report the mechanism's isolated contribution, not a raw single-draw absolute mAP.
5. The supervised-MIL relocation family (v146 argmax-run at scale +0.036/P=0.876; v147 half-max-run with in-vivo z gate 0 fires) has now produced ONE at-scale-but-noise-band result and ONE inert result. The residual OPEN question is whether a properly-calibrated (offline-tuned, fire-band-verified, correctly-paired) supervised relocation can clear P>=0.9 — but it must be measured with a gate that actually fires and a REF that actually isolates the mechanism. The 50/99 zero-overlap recall ceiling remains OPEN and unrefuted.

## Restore / integrity

Restored config.py, pipeline.py, run.py, batch_evaluate.py from /Users/zzl/PycharmProjects/vad/.qoder/backups/v147/ (diff -q byte-identical). models.py was edited but NOT pre-backed-up; its two additive blocks (`pre_milrun_boundaries` field + to_dict serialization) were surgically removed. `grep -i "milrun|MRR-RUN|pre_milrun|_milrun_relocate"` across all 5 files = 0. All 5 parse OK. BUE default-ON (`bue_enabled: bool = True`) and CEX default-OFF (`cex_extend: bool = False`) intact. milrun never shipped; baseline stays committed BUE-on, floor 0.21432 unchanged (nothing kept).
