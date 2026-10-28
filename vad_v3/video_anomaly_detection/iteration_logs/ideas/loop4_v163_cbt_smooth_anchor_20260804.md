# Loop4 v163 — CBT-SP: Smoothed-Peak anchor for the CBT inward clamp

Date: 2026-08-04
Verdict: **REVERT** (CLOSED for this measured variant)
Family: cbt-recalibration (anchor/position axis)

## Mechanism
ONE change stacked on committed BUE + CBT (+`--cbt_recal`): gate the ANCHOR that
`_cbt_category_boundary_tighten` (pipeline.py) uses to center its inward contrast clamp.
Committed anchor is `peak_idx = lo + argmax(rel[lo:hi])` off the RAW category curve. On flat/noisy
Explosion & Shooting curves that argmax lands on a noise spike, mis-centering a correctly-SIZED clamped
window and stalling detections at mid-IoU (hit@.3 but not @.5/@.7). Behind `--cbt_smooth_anchor`
(default OFF), `peak_idx` is taken from a category-aware moving-average smoothing of `rel`
(np.convolve, kernel from CATEGORY_DURATION_PRIORS: Explosion 1.0s, Shooting 1.5s). Only the anchor
source changes; contrast metric, hw search, inward monotone clamp, confidence-freeze, per-cat caps and
recal over-wide trigger byte-identical. Count-neutral, confidence-frozen, terminal, post-BUE.
Implementation gated the smoothed branch to `cat in {'Explosion','Shooting'}` (C-mandate applied in
initial impl) so RoadAccidents/Shoplifting never enter it.

## Per-cat diagnosis (v155 CBT-ON, nGT/hit@.3/hit@.5/boundary-miss/cand-miss)
- Shooting 25/9/7/11/5 — 2 mid-IoU dets (.3<IoU<.5), secondary target.
- RoadAccidents 23/6/0/13/4 — untouched (already IoU-fixed by CBT/recal).
- Shoplifting 25/3/2/11/11 — untouched; its 11/25 cand-miss is a CANDIDATE-GENERATION problem CBT cannot fix.
- Explosion 22/10/7/11/1 — primary target; ~3 dets in .3<IoU<.5 band from mis-centering.

## Headline result (88-video hard harness)
ABSOLUTE hard **new_map = 0.06626** (AP@0.3 0.12761 / AP@0.5 0.06528 / AP@0.7 0.00589).
Paired-json[pre_cbt_boundaries]: ref(BUE-only reconstruction) mAP 0.04931; point_delta +0.01695;
CI90 [-0.00375, +0.05315]; P(delta>0) = 0.923; n=88.

vs targets:
- KEEP floor new_map >= **0.11470** → **FAIL** (0.06626 is far below).
- absolute floor 0.09470 → **FAIL**.
- committed CBT-ON baseline 0.0997 → **FAIL** (regression in absolute terms).
- vs recal-alone stack base 0.11806 → **FAIL**.
Even though the paired P=0.923 clears 0.9 (the smooth/recal variant beats BUE-only reconstruction),
the ABSOLUTE hard mAP (FM18) is decisive and fails hard. → REVERT.

### IMPORTANT flag-drift caveat (record honestly)
`results/batch_eval_v163_full/COMMAND.txt` shows the full 88-video eval was launched with
`--cbt_anchor_centroid` (a centroid-anchor variant), NOT the proposed/implemented
`--cbt_smooth_anchor`. So the archived full-eval numbers above correspond to the CENTROID-anchor
variant, not the smoothed-peak mechanism the IMPL self-test validated. The `--cbt_smooth_anchor`
smoke was BLOCKED (PID 95685 `batch_eval_v13best_verify290` held the GPU; per GPU-serialize protocol
the smoke was not launched). Net: neither variant produced a passing absolute hard mAP; the one full
number we have (centroid anchor) is 0.066 << floor.

## INSTRUMENT self-test (from IMPL, smooth_anchor variant)
- FM20 paired field: REUSED proven `pre_cbt_boundaries`, written unconditionally whenever `--cbt_tighten`.
  Self-test PAIRED_OK: 2/2 `*_x264.json` contained `pre_cbt_boundaries`. Structurally impossible v158 killer.
- FM21 marker: exact ` sa=on`/` sa=off` token on all 5 [CBT-VID] f-strings incl empty/short-curve paths.
  `grep -c "\[CBT-VID\].* sa=on"` = 2 on Explosion/Shooting smoke; both-direction `sa=off`=1, `sa=on`=0 when flag off.
- BOUNDARY-MOVED: Explosion002 det2 [51.0,76.0] → sa=off [52.5,61.5] (raw argmax) vs sa=on [51.5,60.5]
  (smoothed): same count, same ~9-10s width, DIFFERENT center = anchor-only re-centering as designed.
- Validation: py_compile OK; flag-off byte-identical (VADConfig.from_args(Namespace()).scoring==ScoringConfig()).

## FM table
- FM1 curve-shape regress: anchor-only, no confidence/rank edit → low risk. Held.
- FM2 added→FP: count-neutral, N/A.
- FM5 never-activated: sa=on grep self-tested both directions on hard cats. OK.
- FM6 offline→no transfer: **REALIZED as the killer** — the mid-IoU→high-IoU conversion argument did
  not materialize into absolute mAP; measured full run (centroid variant) 0.066 << 0.133 honest central.
- FM8 one change/loop: single gated anchor-source line. OK.
- FM9 MPS: rel already np.float64; np.convolve pure numpy. OK.
- FM17 empty path: sa= token on empty/short-curve markers. OK.
- FM18 absolute mAP: enforced → REVERT.
- FM20 paired field: OK (reused, present).
- FM21 marker: OK (both-direction).
- Execution/GPU-serialize (v162): the required `--cbt_smooth_anchor` full eval was never cleanly run
  (GPU held by verify290); the only archived full run used a drifted flag `--cbt_anchor_centroid`.

## WON / OPEN / CLOSED
- CLOSED: CBT smoothed-peak anchor AND (measured) CBT centroid anchor as absolute-mAP levers on the
  hard harness — neither cleared the 0.11470 floor; centroid variant regressed to 0.066.
- OPEN: Shoplifting CANDIDATE-GENERATION (11/25 cand-miss) remains the highest untouched headroom; CBT
  anchor moves are count-neutral and structurally cannot rescue zero-overlap videos.
- OPEN: CBT RECALIBRATION reclaim (v158 per-cat GT-p75 caps hit 0.12085 cross-run but was VOID on paired
  field) — the paired field is now proven; re-run recal cleanly WITHOUT stacking an anchor change.

## Next-loop guidance
1. Do NOT re-propose CBT anchor re-centering (smoothed-peak or centroid) — both measured-CLOSED here.
2. Re-adjudicate v158 CBT-recal ALONE (per-cat GT-p75 caps + trigger 1.1) cleanly on the hard harness
   with the now-proven `pre_cbt_boundaries` paired field; it is the strongest un-void'd recal signal.
3. Attack Shoplifting candidate-generation in anomaly_discovery.py (category-aware scan density) — argue
   the count effect (FM2): additive dets demote precision unless they land on GT.
4. Enforce GPU-serialize BEFORE launch and confirm the launched flag set matches the proposed mechanism
   verbatim (this loop's full run drifted to --cbt_anchor_centroid).
