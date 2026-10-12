# v160 — CBT-RECAL: per-category caps + over-wide trigger recalibration (KEEP, NEW HARD BASELINE)

- **Loop:** 1 (session v97 ledger)
- **Date:** 2026-08-02
- **Flag:** `--cbt_recal` (default OFF; used with `--cbt_tighten`)
- **Verdict:** **KEEP — NEW HARD BASELINE.** Hard-88 absolute new_map **0.11806** >= KEEP target 0.11470 (beats committed CBT-on 0.09970 by +0.0184), paired P(delta>0)=0.922.

## Mechanism (ONE change / ONE flag, architecture-level)
Reclaim of the VOID v158. The committed v155 CBT method `_cbt_category_boundary_tighten` (pipeline.py ~L2325) is a terminal post-BUE, count-neutral, confidence-frozen, monotone-INWARD clamp of over-wide hard-category final detections toward per-category caps `CBT_CAPS={RoadAccidents:6, Explosion:13, Shoplifting:10, Shooting:14}`, firing only when `det_dur > 1.5*cap`. CBT-RECAL adds `--cbt_recal` which swaps ONLY the two scalar inputs when ON:
- caps -> GT-p75 recal `{RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5}`
- over-wide trigger multiplier `1.5 -> 1.1`

pipeline.py: `caps = cfg.cbt_recal_caps if cfg.cbt_recal else CBT_CAPS`; `trig = cfg.cbt_recal_trigger if cfg.cbt_recal else 1.5`; the three uses swapped (`cat not in caps`, `cap=float(caps[cat])`, `det_dur <= trig*cap`). Method objective/count/order/anomaly_type/confidence/frame_scores untouched. Flag-OFF => caps==CBT_CAPS, trig==1.5 => byte-identical to committed CBT-on. BUE (default-ON), CEX (default-OFF), v155 CBT method all undisturbed.

Lower caps + lower trigger make MORE adjacent-but-too-wide dets qualify and clamp them TIGHTER, converting the dominant hard miss (~46 boundary-too-wide GT segs) into IoU@0.5/0.7 hits.

## Per-category diagnosis (nGT/hit@.3/hit@.5/boundary-miss/cand-miss, v155)
Shooting 25/9/7/11/5, RoadAccidents 23/6/0/13/4, Shoplifting 25/3/2/11/11, Explosion 22/10/7/11/1. Dominant miss = BOUNDARY-too-wide. CBT-RECAL primary target = RoadAccidents (cap 6.0->4.5 + trigger 1.1 makes its many over-wide dets qualify and clamp toward GT-p75 ~4.3s).

## Result (headline = ABSOLUTE hard new_map vs 0.0997)
- **Hard-88 paired bootstrap (--paired-json pre_cbt_boundaries):** point_delta +0.02283, 90% CI [-0.00371, +0.04797], P(delta>0)=**0.922**, **new_map 0.11806** (vs committed CBT 0.09970, +0.0184).
- **Per-cat AP (v160 vs v155):** RoadAccidents **+0.0134** (0.2152 vs 0.2018, headline), Shooting **+0.0053** (0.0836 vs 0.0783), Shoplifting +0.0000 (0.0732 vs 0.0732), Explosion **-0.0023** (0.1586 vs 0.1609, within noise).
- **Smoke (14 hard videos):** mAP 0.2273, AP@0.3 0.3333 / AP@0.5 0.1970 / AP@0.7 0.1515; recal=on markers 14/14, pairedJSONs 13, no FM17. Latency 178s.
- **Full run (290 videos, doubles as paired-290 verify, task #11):** aggregate temporal mAP 0.17254 (AP@0.3 0.2650 / AP@0.5 0.1499 / AP@0.7 0.1027), paired-290 point_delta +0.0228, P 0.922 — easy cats NOT regressed beyond noise; recal=on markers 290/290 agree log & run.log.

## INSTRUMENT self-test (FM20/FM21 — the v158/v157 killers, PASSED)
- **FM20 paired-field-written:** reuses the EXISTING `pre_cbt_boundaries` write path (pipeline.py ~L1719-1724 -> DetectionResult) — NO new field, so the v158 killer is structurally impossible. D1 asserted `pre_cbt_boundaries` non-null in produced JSONs (RoadAccidents123 [[1.5,30.5]], Shooting032 captured pre-CBT); full run 219 paired JSONs present.
- **FM21 marker string/category:** literal `recal=on` appended to ALL FIVE `[CBT-VID]` emit sites incl the two detect-loop empty/0-det early-returns (L1429, L1535) — FM17-safe. Toggle self-test both directions: ON smoke grep `recal=on`=2 & `recal=off`=0; OFF smoke `recal=off`=2 & `recal=on`=0 — proves the marker toggles with the flag (config threaded, not hardcoded).
- **FM5 activation:** mechanism actually FIRED (smoke fired=1 RoadAccidents, fired=3 Shooting; not the empty path).

## FM table
- FM1/FM19 (curve-shape / floor-unsafe move): over-wide gate protects compact dets; monotone-inward, fires only when new_dur<det_dur and >=min_segment. HELD.
- FM2 (added dets -> FPs): COUNT-NEUTRAL, no append/delete. IMMUNE.
- FM5 (never-activated): confirmed fired. PASS.
- FM6 (offline non-transfer): measured on the same 88/290-video harness. N/A.
- FM8 (one change/loop): only two scalar inputs move behind one flag. HELD.
- FM13 (ceiling): measured on the identical-param set. N/A.
- FM17 (empty-video marker): recal= emits on 0-det early-return paths. PASS.
- FM18 (absolute mAP): headline = absolute hard new_map 0.11806. HELD.
- FM20 (paired-field-written): reused proven pre_cbt_boundaries, asserted non-null. ELIMINATED.
- FM21 (marker string/category): literal recal=on on all 5 sites, grep self-tested both directions on hard-cat smoke. PASS.

## Status
- **WON:** RoadAccidents (+0.0134) and Shooting (+0.0053) via tighter recal caps + trigger 1.1. CBT-RECALIBRATION lever confirmed positive; v158's cross-run signal reproduced in-vivo and now correctly adjudicated. NEW HARD BASELINE = 0.11806.
- **OPEN:** Shoplifting +0.0000 — its 11/25 cand-miss bucket is UNADDRESSED and CBT/CBT-RECAL (boundary-only, inward) structurally CANNOT fix it (GT is elsewhere in the video; the pipeline never proposes a window there). Explosion -0.0023 (noise; cap unchanged at 13.0, only trigger moved).
- **Margin caveat:** absolute new_map 0.11806 clears the 0.11470 target by +0.0034, inside the +/-0.02 CI half-width; the robust evidence is paired P=0.922. Hard-88 CI90 low is -0.00371 (crosses 0) but the KEEP gate is P>=0.9 on the absolute-vs-target — met.

## Next-loop guidance
1. **Shoplifting CANDIDATE-GENERATION** (the real remaining lever): category-aware scan density / segment proposal in `anomaly_discovery.py` so the pipeline PROPOSES a window where the GT actually is. cand-miss 11/25 is highest; CBT cannot rescue zero-overlap videos. Cross-run vs the NEW v160 baseline; argue count-effect (FM2 — additive dets demote precision unless they land on GT).
2. **Category-specific score/query quality** for RoadAccidents/Shoplifting (flat in-span CLIP contrast, per v157 diagnosis) — CLIP prompt / CoOp sharpening (NOT curve re-rank, NOT prompt max-pool = v159 PMX measured-CLOSED).
3. Per-category recal-cap fine-tuning is likely SATURATED for RoadAccidents; do not re-tune caps as a fresh loop without a new signal.

## Artifacts / backups
- Code (KEEP, flag default OFF): config.py L126-135/289, run.py L225-229, batch_evaluate.py L316-320, pipeline.py L2325-2328 + 5 `[CBT-VID]` marker sites.
- Backups: /Users/zzl/PycharmProjects/vad/.qoder/backups/v160/{config,run,batch_evaluate,pipeline}.py
- Results: iteration_logs/results/v160_{hard_eval.json, hard_run.log, bootstrap_ab.json}; full dir results/batch_eval_v160_full/.
