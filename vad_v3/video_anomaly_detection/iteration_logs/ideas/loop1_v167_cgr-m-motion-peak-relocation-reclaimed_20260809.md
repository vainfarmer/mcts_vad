# v167 — CGR-M reclaimed: count-neutral motion-energy-peak candidate RELOCATION

**Loop:** 1 (session v97 ledger)  **Verdict:** **KEEP** — new committed hard baseline, promoted default-ON.
**Date:** 2026-08-09  **n_videos:** 88 (hard harness, `--categories Shooting,RoadAccidents,Shoplifting,Explosion`, confirmed `timing.num_videos==88`).

## HEADLINE (paired, count-neutral)
- **PAIRED-json[pre_cgr_boundaries] bootstrap, 88 common hard videos:** point_delta **+0.06105**, 90% CI **[+0.00897, +0.11921]**, **P(delta>0)=0.971**.
- **ABSOLUTE hard new_map = 0.12246** (>= KEEP floor 0.11306; > prior committed hard baseline 0.11806).
- **COUNT-NEUTRAL confirmed two independent ways:** (a) all 88 `[CGR-VID]` lines `ndet_before==ndet_after`, **0 violations**; (b) mean det/video identical NEW vs pre-relocation (131 total dets, 1.48864/video both arms). 119 windows actually changed position.
- GATES: point_delta >= 0.008 ✓ (0.06105); P >= 0.9 ✓ (0.971); mAP >= floor 0.11306 ✓ (0.12246). **ALL PASS.**

## Mechanism
New gated, count-neutral, width-neutral candidate RELOCATION. `pipeline._cgr_motion_relocate(final_anomalies, video_data, category)` fires immediately BEFORE the pre_bue snapshot block, on the FINAL post-merge/NMS/PostTighten/classify/curve-correction/conf-gate/Q-score detection list (count already frozen, confidence bit-identical).
1. Category gate: fire only if `cfg.cgr_enabled` AND `category in cfg.cgr_categories`.
2. REUSE `video_data['motion_energy']` (already computed at pipeline L1405 — NO recompute) with the exact `_motion_temporal_localization` (L2468) smoothing/indexing: `window=max(2,int(cgr_smooth_win_sec*fps))`, `smoothed=convolve(motion, ones(window)/window, 'same')`, `t_peak=timestamps[argmax(smoothed)]`.
3. Coverage guard: if ANY existing detection covers `t_peak` -> NO-OP (peak already proposed).
4. Reclaim: `j=argmin(confidence)`; preserve width `w`; `ns=clamp(t_peak-w/2,0,dur)`, `ne=ns+w`; rewrite ONLY `a[j]['t_start']/['t_end']` in place.
5. Fire only if start actually moves. Emit exactly one unconditional `[CGR-VID]` marker per video on every return, including both empty/0-detection early-return paths.
No append/pop/insert/del/filter/reorder; confidence/anomaly_type/coarse_score/order byte-frozen. Pure numpy/CPU.

This is the cand-miss lever CBT cannot touch: CBT is a post-BUE INWARD width clamp along the CLIP category curve around a peak already inside a detection; CGR-M is a pre-BUE WHOLE-WINDOW reposition of a wasted low-value candidate onto the MOTION-ENERGY peak of a currently-uncovered region.

## Per-category cand-miss diagnosis (nGT / hit@.3 / cand-miss)
- Shooting 25/9/5; RoadAccidents 23/6/4; Shoplifting 25/3/11; Explosion 22/10/1.

## Per-category paired deltas (NEW vs pre-relocation, same run)
- **RoadAccidents d=+0.21739** (NEW 0.24549 / PRE 0.02810, n=23) — dominant gainer; collision motion peaks sit tightly on GT.
- **Explosion d=+0.08614** (NEW 0.15975 / PRE 0.07362, n=21) — blast motion peak aligns.
- Shooting d=-0.01795 (NEW 0.09285 / PRE 0.11081, n=23) — small negative; muzzle/scuffle motion only moderately aligned.
- Shoplifting d=-0.03832 (NEW 0.04365 / PRE 0.08197, n=21) — small negative; subtle concealment motion, peak often off-GT (predicted weak lever in the proposal).
Win concentrated in RoadAccidents + Explosion; two small negatives (Shooting, Shoplifting) net-dominated. Overall PAIRED marginal clears the gate decisively.

## INSTRUMENT self-test evidence (from the FULL 88-run)
- FM21 marker: `grep -c "[CGR-VID]"` full run.log = **88 == timing.num_videos** (no FM17). Empty/0-det path emits marker, e.g. `[CGR-VID] vid=RoadAccidents021_x264.mp4 enabled=True ndet_before=0 ndet_after=0 moved=0 skip=early_return_no_candidates`.
- FM5 activation: **52 videos fired (moved=1)**, 36 no-op (moved=0) — mechanism live and firing across the cand-miss set.
- FM20 paired field: `pre_cgr_boundaries` non-null in all 88 per-video JSONs. GREP e.g. `RoadAccidents123_x264.json`: `pre_cgr=[[1.5,8.5]]  anoms=[[14.0,18.0]]  len_match=True` — a real MOVE persisted (final also reshaped by BUE/CBT which run after CGR).
- Count-neutrality: 88/88 `[CGR-VID]` lines `ndet_before==ndet_after`, **0 violations**.

## FM risk table (realized)
- FM1 (curve edits regress): N/A — endpoint move only, no curve/confidence edit.
- FM2 (added count->FPs): NEUTRALIZED — in-place rewrite, ndet frozen (0 violations/88).
- FM5 (never-activated): PASSED — 52/88 fired.
- FM6 (offline gains don't transfer): avoided — judged on live in-pipeline produced JSONs.
- FM8 (one change/loop): single new gated operator.
- FM9 (MPS): N/A pure numpy CPU.
- FM13/FM18 (bound ceiling / absolute mAP): pre_cgr baseline in the SAME run gives committed hard-88 baseline for free; new_map 0.12246 >= floor.
- FM16 (noise floor): PAIRED bootstrap cancels VLM noise; P=0.971.
- FM17/FM21 (empty-video marker): emitted unconditionally incl both 0-det early-returns.
- FM19 (position move not floor-safe): floor guard 0.11306 + P>=0.9 + coverage guard held; two small per-cat negatives net-dominated, absolute above floor.
- FM20 (paired field written+grepped): real snapshot + DetectionResult kwarg + grepped a produced JSON + length==count check.
- BUG-1 (290 vs 88): `--categories` baked into eval; `num_videos==88` confirmed.

## WON / OPEN / CLOSED
- **WON:** motion-energy-peak count-neutral relocation is a KEEP-clearing cand-gen lever when judged PAIRED (v165's true signal reclaimed — the v14 revert was an instrument artifact: cross-run judging of a count-neutral move + un-persisted pre_cgr field). RoadAccidents + Explosion are the real gainers; motion peaks sit on collision/blast GT. Count-neutral relocation is FM2-immune by construction.
- **OPEN:** Shoplifting cand-miss 11/25 STILL the largest untouched bucket — subtle concealment motion peaks off-GT (d=-0.038 here). Needs a LATE/concealment-peaked source (not raw global motion argmax), or a seed that survives downstream BUE+CBT. Shooting slightly negative (-0.018): muzzle/scuffle motion sometimes pulls the lowest-conf window off a marginal hit. Per-category rollback available (drop a name from `cfg.cgr_categories`) if a future loop wants to isolate RoadAccidents+Explosion.
- **CLOSED (reaffirmed):** category-CLIP-curve (`get_category_scores`) relocation target (v164 flat / v165 P=0.780 / v166 P=0.593) — do NOT re-propose. CBT anchor/seed/contrast micro-variants (v161-163). PMX prompt max-pool (v159). All 10 banned families.

## Next-loop guidance
1. Shoplifting-specific source: a LATE-biased or concealment-peaked signal (motion argmax is wrong here) whose relocated seed SURVIVES BUE+CBT unchanged.
2. Consider a per-category CGR variant restricted to RoadAccidents+Explosion (the measured gainers) if isolating the win is desired; keep Shooting/Shoplifting under watch.
3. Multi-peak / late-event handling: CGR moves onto the single global motion argmax; multi-event videos get only one region reclaimed.
4. Do NOT re-touch CBT anchor/seed/contrast (v161-163), PMX (v159), or the category-CLIP curve as a relocation target (v164/165/166 — thrice dead).

## Key lesson
A COUNT-NEUTRAL relocation MUST be judged PAIRED (variant boundaries vs the pre-relocation boundaries in the SAME run) — VLM noise cancels and a true +0.008 marginal clears P>=0.9 easily. v165 buried the identical mechanism only because (a) it was judged cross-run (P=0.780) and (b) its pre_cgr_boundaries field was never persisted. v167 fixed BOTH: real persisted paired field (grepped) + `--categories`-forced 88-harness (num_videos==88) + paired adjudication -> +0.06105, P=0.971, KEEP.
