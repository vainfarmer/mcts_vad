# v168 — CCR-SE: count-neutral CLIP-category-curve semantic candidate RELOCATION (Shoplifting + Shooting)

**Loop:** 2 (session v97) · **Date:** 20260810 · **Verdict: REVERT** (absolute floor breached)
**n_videos = 88** (hard harness, `--categories Shooting,RoadAccidents,Shoplifting,Explosion`, BUG-1 fixed — confirmed `timing.num_videos == 88`)

## Headline (read this first)
- **PAIRED bootstrap** `pre_ccr_boundaries` (n=88, 1000 resamples): **point_delta +0.05301**, 90% CI **[+0.00977, +0.10681]**, **P(delta>0)=0.979**.
- **ABSOLUTE hard new_map = 0.10977** — **BELOW the KEEP floor (0.11306 task / 0.11746 adjudicated) AND below the committed v167 CGR-M baseline 0.12246 (−0.0127).** FM18 is decisive: paired delta + P both PASS, absolute mAP FAILS → **REVERT.**
- **Count-neutrality CONFIRMED:** mean detections/video **1.500 == 1.500 (132 == 132)** NEW vs pre_ccr arm; 0 violations across 88 videos.
- **The two GATED target cats did essentially NOTHING:** Shooting paired **+0.00013**, Shoplifting paired **+0.00082**. CCR-SE completely failed its stated objective (relocate wasted candidates onto the semantic GT peak for the low-motion cand-miss cats).

## Mechanism
Byte-for-byte structural clone of the committed v167 `_cgr_motion_relocate` with the SINGLE difference that the signal source is swapped from `video_data['motion_energy']` to the cached read-only per-frame relative CLIP category curve `rel = self.frame_scorer.get_category_scores(video_data, category)` (cat_prob/(cat_prob+normal_prob), `None` for Normal). New method `_ccr_curve_relocate`: category gate (`cfg.ccr_categories=('Shoplifting','Shooting')`) → smooth `rel` (window=max(2,int(ccr_smooth_win_sec*fps)), box kernel) → `t_peak=t[argmax(smoothed)]` → coverage guard (NO-OP if any det covers t_peak) → reclaim lowest-confidence det, in-place width-preserving rewrite of `t_start/t_end`, fire only if start moves. Runs pre-BUE **immediately after** the committed CGR-M call, **before** the `pre_bue` snapshot, so BUE widens / CBT tightens the relocated window exactly as for CGR-M. No curve normalization (raw smoothed argmax — deliberately avoids banned family #4). Flag `ccr_enabled` default OFF; `--ccr_relocate`/`--no_ccr_relocate` in run.py + batch_evaluate.py.

## Per-category cand-miss diagnosis (nGT / hit@.3 / cand-miss)
Shooting 25/9/5 · RoadAccidents 23/6/4 · Shoplifting 25/3/11 (target) · Explosion 22/10/1.
Gated: Shoplifting (cand-miss 11, lowest-motion) + Shooting (5, brief off-screen). RoadAccidents + Explosion deliberately EXCLUDED (protect the floor cats CGR-M already carries).

## Per-category PAIRED deltas (NEW / pre_ccr)
- **Shooting** (n23) **+0.00013** [.09867 / .09854] — GATED, flat.
- **Shoplifting** (n21) **+0.00082** [.05031 / .04949] — GATED, flat.
- RoadAccidents (n23) +0.16617 [.22473 / .05856] — NOT gated.
- Explosion (n21) +0.05363 [.14382 / .09019] — NOT gated.

**Interpretation of the paired-delta artifact:** the +0.053 point_delta and P=0.979 are carried ENTIRELY by RoadAccidents + Explosion — the two cats CCR does not touch. Their large paired deltas are NOT CCR gains; they arise because the `pre_ccr` reference arm reconstructs detections from **pre-BUE / pre-CBT** boundaries (paired-ref mAP 0.05676, far below the committed 0.12246), so the paired test conflates downstream BUE+CBT refinement with the relocation. On the ONLY cats CCR actually moves (Shooting/Shoplifting) the paired delta is ~0. This is exactly why **absolute new_map is the binding honest metric (FM18)** and it fails: 0.10977 < committed 0.12246. Adding the semantic relocation on the two gated cats net-HURT absolute mAP.

## Instrument self-test — CLEAN this loop (BUG-1 + BUG-2 both fixed; NOT a void)
- **BUG-1 (harness):** `--categories` baked into every eval; `timing.num_videos == 88`. No 290 leak.
- **FM20 (paired field written + GREPPED):** `pre_ccr_boundaries` present + length-matched in all 88 full JSONs. Smoke evidence pasted by D1 — Shooting032 `pre_ccr=[[273.0,284.0],[308.5,316.5],[720.0,722.7]]` len_match=True; Shoplifting001 `pre_ccr=[[14.0,35.5],[125.25,133.75]]` len_match=True.
- **FM21 (marker):** `grep -cF '[CCR-VID]'` == 88 == num_videos; unconditional on both empty/0-det early-return paths.
- **FM2 count-neutral:** every `[CCR-VID]` line `ndet_before==ndet_after`; mean 1.500==1.500 (132==132). 0 violations.
- **FM5 FIRED:** smoke Shooting032 relocated a wasted det parked at 720.0–722.7 onto semantic peak t_peak=298 (conf 0.3908 frozen); Shoplifting001 correctly NO-OP (peak_covered).

So this is a **VALID measured negative**, not an instrument void (unlike v164 selftest-fail / v165 unpersisted-field / v166 length-mismatch). The category-CLIP curve genuinely does not produce KEEP-worthy relocations on the low-motion hard cats even when the paired test is clean.

## FM table
- FM1 immune (curve read-only, only endpoints move). FM2 immune (count-neutral, 132==132, 0 adds). FM5 FIRED (not inert). FM6 judged on live 88-run, not offline. FM8 one lever (only signal source swapped). FM9 numpy CPU (`get_category_scores` already `.cpu().numpy()`). FM13 coverage-guard bounds to uncovered peaks. FM16 paired cancels VLM noise → P=0.979 clean. FM17 marker on empty paths. **FM18 = THE KILLER: absolute new_map 0.10977 < floor → REVERT.** **FM19 REALIZED: count-neutral position move not floor-safe — the gated relocations moved dets in a way that net-lowered absolute mAP below committed, while the gated cats' paired delta was ~0.** FM20/FM21 clean.

## WON / OPEN / CLOSED
- **WON:** nothing (no KEEP). Positive: instrument discipline fully clean — BUG-1/BUG-2 fixed, a genuinely valid measurement.
- **CLOSED (reaffirmed, now the 3rd category-CLIP failure):** category-CLIP-curve (`get_category_scores`) as a candidate-RELOCATION target — v164 (flat, selftest-fail), v166 (P=0.593), **v168 (paired PASS but abs-mAP fail; gated target cats flat +0.0001/+0.0008)**. The semantic curve does NOT yield KEEP-worthy online relocations on the low-motion hard cats. **Do NOT re-propose category-CLIP-curve-peak relocation on this substrate.** (Motion-energy relocation = the ONLY relocation signal that survived → committed as v167 CGR-M.)
- **OPEN:** Shoplifting cand-miss 11/25 STILL unaddressed — neither motion (CGR-M paired −0.038 there) nor category-curve (CCR paired +0.0008) reach its GT. Needs a fundamentally different source that peaks ON concealment AND survives committed BUE+CBT.

## Next-loop guidance
1. STOP proposing category-CLIP-curve relocation (thrice dead). The remaining Shoplifting residual is a SIGNAL-QUALITY problem (CLIP prompt / CoOp for concealment), not a relocation-target problem.
2. Any new cand-gen must clear the **ABSOLUTE floor** (committed v167 = 0.12246), not just the paired delta — a clean paired PASS on a pre-BUE reference is necessary but NOT sufficient (v168 proves this).
3. Do NOT re-touch CBT anchor/seed (v161–163) or PMX (v159).
4. Committed hard baseline stays v167 CGR-M **0.12246**; BUE / CBT / recal / CGR defaults all intact, CEX default-OFF, CCR left default-OFF (not promoted).

## Revert
Soft-revert per the proposal rollback plan: `ccr_enabled` left **default OFF** (no promotion, no default flip). Flag-off is byte-identical (`VADConfig.from_args(Namespace()).scoring == ScoringConfig()` → True). NOTE: the v168 backups were taken with the CCR impl already in the working tree, and no clean CGR-promoted backup exists (v167 backup is v160-era, pre-CGR-promotion); git revert is forbidden and would destroy the uncommitted committed-pipeline work. Therefore the CCR method + flag remain present-but-inert in the tree exactly like the existing default-OFF gated features — the committed CGR-M+BUE+CBT+recal pipeline is fully preserved at runtime. Verified: `anomaly_discovery.py` 0 CCR symbols; ScoringConfig() bue=True cbt=True recal=True cgr=True ccr=False; ccr_categories=('Shoplifting','Shooting'); flag-off byte-identical.

## Artifacts
- iteration_logs/results/v168_hard_eval.json
- iteration_logs/results/v168_hard_run.log
- iteration_logs/results/v168_bootstrap_ab.json

## SMOKE / FULL
- SMOKE (4 videos): mAP 0.12727 (AP@0.3 0.24545 / AP@0.5 0.13636 / AP@0.7 0), 190.85 s/vid, markers 4, pairedJSONs 4.
- FULL (88): mAP 0.10977 (AP@0.3 0.18659 / AP@0.5 0.11999 / AP@0.7 0.02273), 85.34 s/vid, markers 88, pairedJSONs 88.
