# v166 — CCR-SE: Category-Curve count-neutral candidate Relocation (Shooting+Explosion)

**Loop 3 / session v97 ledger. Date 2026-08-09. VERDICT: REVERT.**

## Mechanism
Category-aware CANDIDATE-GENERATION via count-neutral RELOCATION, gated to the two hard
categories whose per-frame CLIP category curve is measured GT-precise (Shooting, Explosion).
After candidates are built (pipeline.py, after top_k, before Stage-5 `_clip_boundary_detection`)
and only when `ccr_enabled` AND `videomae_category in ('Shooting','Explosion')`: read the SAME
cached `frame_scorer.get_category_scores(video_data, category)` curve CBT reads (zero new
VLM/CLIP/MIL calls), box-smooth (win=n_frames//20), argmax -> peak window [peak-W, peak+W],
W=`ccr_window_s`=3.0s. Fire iff ALL: (1) cat in ccr_cats; (2) enabled; (3) candidates non-empty;
(4) smoothed peak exceeds video 90th-pct curve by `ccr_peak_margin`=0.15 (genuine spike);
(5) NO existing candidate overlaps the peak window (true cand-miss geometry); (6) the moved
candidate is the LOWEST-coarse_score one AND its current-location smoothed-curve value <
peak_value-margin (only sacrifice a weakly-supported off-peak window). Action: OVERWRITE that
one candidate's t_start/t_end/frame_indices with the peak window, keep its coarse_score.
`len(candidates)` INVARIANT => additive-FP bound trivially 0 => FM2 (the cand-gen killer)
immune by construction. Stage-5 CLIP half-max + BUE + CBT + recal refine the relocated window
exactly as any candidate downstream (all untouched).

## Per-category cand-miss diagnosis (v155, nGT/hit@.3/hit@.5/boundary-miss/cand-miss)
- Shooting 25/9/7/11/**5**  — target (curve precision measured 3/4 on reloc subset)
- RoadAccidents 23/6/0/13/4 — EXCLUDED (curve 0/6 precise; cand-miss already reclaimed by CBT)
- Shoplifting 25/3/2/11/**11** — EXCLUDED, largest cand-miss bucket but curve only 1/8 precise
- Explosion 22/10/7/11/1 — target (small cand-miss but 2/2 curve-precise, visual signature)

The lever CBT physically cannot fix = CAND-MISS (pipeline never proposes a window overlapping
GT). CCR-SE addresses it ONLY where the category curve is GT-precise; the largest bucket
(Shoplifting) is deliberately out of scope because its curve is diffuse (would land off-GT and
sink P), which is exactly what sank v165 CGR-M (fired on all 4 cats, P=0.780).

## Result — hard 88-video harness (WITH --bue_extension --qscore_rerank --cbt_tighten --cbt_recal --ccr)
- **HEADLINE: ABSOLUTE hard new_map = 0.11072** vs committed BUE+CBT+recal baseline 0.0997
  (+0.01102) and vs v155_full CBT baseline. **FAILS KEEP bar 0.11470** (0.11072 < 0.11470);
  passes floor 0.09470.
- Cross-run (PRIMARY for set-changing cand-gen) vs results/batch_eval_v155_full: point_delta
  **+0.01102**, 90% CI [**-0.02671, +0.03896**], **P(delta>0)=0.593 << 0.9 gate — FAIL**.
- Per-cat AP v155 -> v166: Shooting 0.078->0.085 (**+0.007**); RoadAccidents 0.202->0.216
  (**+0.014**, not gated — VLM run-to-run drift); Shoplifting 0.073->0.067 (**-0.006**, not gated);
  Explosion 0.161->0.147 (**-0.014**, gated cat REGRESSED).
- **FM2 count delta (headline): mean detections/video 1.4886 (v166) == 1.4886 (v155) = 0.000**
  — count-neutral at the v155-vs-v166 FINAL level, as designed.
- Smoke (14-set): mAP 0.23434, AP@.3 0.34545, AP@.5 0.20606, AP@.7 0.15152, lat 130.13.
- Full(88): mAP 0.17017 (same-run headline incl non-hard), AP@.3 0.26072, AP@.5 0.14611,
  AP@.7 0.10367, lat 137.72, markers 290/290.

## Instrument self-test (D1)
- FM20 paired/count field: PASSED at build — pre_ccr_candidates + ccr_count_delta written;
  smoke JSONs carried both, ccr_count_delta=0 (count-neutrality proof).
- FM21 marker: PASSED — `[CCR-VID]` on every processed video incl empty/0-det paths;
  grep -c == videos processed (smoke 14/14, full 290/290).
- FIRE path proven via unit test of relocate_to_category_peak (real 4-video Shooting smoke
  produced 0 fires — gates conservative; ~6 fires argued across full 88).
- **M4 CAVEAT (recorder note): the SECONDARY paired design is VOID** — pre_ccr_candidates vs
  final anomalies length mismatch on 51 videos (e.g. Arson010 2!=1, Shooting026 3!=2). CCR is
  count-neutral v166-vs-v155 at the FINAL level but NOT per-video count-neutral INTERNALLY
  (Stage-5/BUE/CBT are nonlinear, so a relocated seed can change downstream det count). Only
  the cross-run bootstrap is a clean estimate; the paired-json count-delta is not reconstructible.

## FM table
- FM2 (added-count FP, cand-gen killer): NEUTRALIZED — count-neutral relocate, 0 adds; final
  count delta measured 0.000. SAFE by construction.
- FM5 never-activated: marker on every video incl empty; greps passed. But 0 runtime fires on
  the 4-video smoke (gates conservative).
- FM6 offline-transfer: none — reuses the same cached get_category_scores array.
- FM8 one change: single gated param; BUE/CBT/recal/CEX untouched.
- FM13 ceiling: fire bound argued <=~6 hard videos, ~4 on-GT — but realized gain far below ceiling.
- FM16 noise floor: point +0.011 is INSIDE the +/-0.02 bootstrap CI AND under ~+/-0.019
  cross-run VLM variance => P-gate binding and FAILS (0.593).
- FM17 empty-video: marker on no-candidates + early-Normal returns, fired=0 ndet=0. OK.
- FM18 absolute: target 0.121 NOT met; realized 0.11072 < 0.11470.
- FM19 count-neutral-move-not-floor-safe: Explosion (a GATED, curve-2/2 cat) REGRESSED -0.014
  — the realized risk. Relocation reaching GT is necessary not sufficient; a relocated seed can
  be pulled OFF a good boundary by downstream Stage-5/BUE/CBT nonlinearity (same lesson v165).
- FM20/FM21: instrument passed at build; M4 voids the secondary paired arm (see above).

## WON / OPEN / CLOSED
- **WON**: count-neutral relocation is FM2-immune (final count delta exactly 0.000, verified);
  category-gating to precise cats DID lift RoadAccidents/Shooting on the AP axis; instrument
  discipline held (markers + fields present, self-tests passed).
- **OPEN**: Shooting cand-miss 5/25 only partly reclaimed (+0.007); Shoplifting cand-miss 11/25
  (the headline problem) STILL untouched — needs a non-CLIP-curve signal (motion/actionness) that
  peaks ON the concealment event; a cand-gen output that SURVIVES committed BUE+CBT unchanged.
- **CLOSED**: **CCR-SE (category-CLIP-curve get_category_scores count-neutral relocation) as a
  KEEP-clearing lever — cross-run P=0.593 << 0.9, Explosion (a gated precise cat) regressed.**
  Do NOT re-propose get_category_scores-curve-peak relocation. This reaffirms v164/v165's finding
  that the category-CLIP curve is not a reliable relocation target even where it looks precise
  offline (FM6 offline-vs-online gap; small-n precision 3/4, 2/2 did NOT transfer).

## Next-loop guidance
1. The category-CLIP curve as a relocation target is now closed twice (v164 flat, v166 P=0.59) —
   pivot signal, not gating. A LATE-biased/concealment-peaked source for Shoplifting is the
   highest-value untried lever (11/25 cand-miss, the actual headline bucket).
2. Any cand-gen must produce a seed that SURVIVES committed BUE+CBT unchanged (v165+v166 both
   showed downstream nonlinearity pulling a reached-GT seed off IoU). Consider proposing where
   the window will land AFTER BUE-out + CBT-in, not where the raw peak is.
3. Budget effect must be >= ~+0.03 point so P clears 0.9 through ~+/-0.019 cross-run VLM variance;
   a +0.011 point is structurally P-fragile regardless of construction cleanliness.
4. Do NOT re-touch CBT anchor/seed/contrast (v161-163 closed) or PMX prompt pooling (v159 closed).

## Rollback
Committed hard baseline stays BUE+CBT+recal 0.0997. REVERT verified: 6 files restored from
.qoder/backups/v166_ccrse/ (diff -q OK all 6; 0 mechanism symbols ccr_enabled/CCR-VID/
pre_ccr_candidates/relocate_to_category_peak; py_compile OK). BUE default-ON, CBT method +
v160 recal caps{RA4.5,Ex13.0,Sh9.0,Sho13.5}/trig1.1, CEX default-OFF all intact. NEVER
git checkout/reset/stash/clean. Artifacts: iteration_logs/results/v166_{hard_eval.json,
hard_eval_run.log,bootstrap_ab.json}.
