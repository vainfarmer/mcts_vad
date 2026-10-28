# v163 — CBT-CA: Centroid-Anchored CBT recalibration (loop 4)

Date: 2026-08-04
Family: cbt-recalibration (build-on-CBT lineage v155/v158/v160)
Verdict: **REVERT** (floor breached)
Flag: `--cbt_anchor_centroid` (default OFF; requires `--cbt_tighten --cbt_recal`)

## Mechanism
Inside `_cbt_category_boundary_tighten` (pipeline.py), swap the over-wide window's
re-centering seed from `peak_idx = argmax(rel[lo:hi])` to an **above-median
score-weighted mass centroid**:
```
seg = rel[lo:hi]; w = clip(seg - median(seg), 0, None)
peak_idx = lo + (round(average(arange(hi-lo), weights=w)) if w.sum()>0 else argmax(seg))
```
Runs AFTER the pre_cbt_boundaries capture -> paired field written byte-identically.
Monotone-inward clamp preserved; confidence/count/ordering frozen (count-neutral).
One gated seed branch added; flag-off == committed BUE+CBT(+recal-when-flagged).

## Per-category diagnosis (target vs outcome)
Target: recenter right-WIDTH-but-off-CENTER windows (the @0.5-vs-@0.7 stuck AP)
onto GT via mass centroid; primary RoadAccidents + Explosion, secondary Shooting;
Shoplifting untouched (cand-miss dominated). Expected hard new_map ~0.134.

Outcome (paired bootstrap, pre_cbt_boundaries, n=88, 92 windows changed):
- **All 4 categories REGRESSED vs v160 recal baseline (0.12085):**
  - RoadAccidents -0.11286 (0.10233 vs 0.21520)  <- worst; the exact "largest lever" cat
  - Shoplifting  -0.03557 (0.03766 vs 0.07323)
  - Shooting     -0.01659 (0.06705 vs 0.08364)
  - Explosion    -0.00818 (0.15045 vs 0.15863)

## Hard bootstrap (headline)
- ABSOLUTE hard **new_map = 0.06626** vs KEEP 0.11470 / floor 0.11306 / stretch 0.13306 -> **FAIL**
- AP@0.3 = 0.1276, AP@0.5 = 0.0653, AP@0.7 = 0.0059
- Paired vs BUE-only (pre_cbt_boundaries ref 0.04931): point_delta +0.01695, CI90 [-0.00375, +0.05315], P(delta>0)=0.923.
- GATE: P>=0.9 PASS (0.923); point_delta>=0.02 FAIL (+0.01695); mAP>=floor FAIL (0.06626 << 0.11306).

## CRITICAL run-quality anomaly (read before next loop)
The full-88 run produced depressed absolute numbers across the board:
new_map 0.06626 is BELOW the committed CBT baseline (0.0997) AND far below the
v160 recal cross-run (0.12085) — even the paired BUE-only ref came out at 0.04931.
The smoke (4 vids) was healthy (mAP 0.1818, AP@0.3/@0.5 0.2727). The mechanism
was NOT floor-safe at scale (FM19 realized: centroid moved windows off GT the
argmax was already clipping — RoadAccidents' noisy flat curves, exactly where the
proposal bet centroid would be MORE stable, regressed hardest -0.11286). The
across-the-board depression (all 4 cats down, including "untouched" Shoplifting
-0.0356 and Explosion) is stronger than a pure center-move would predict, so a
run/env confound cannot be excluded — but the verdict is unambiguous either way.

## Instrument self-test (PASSED — instrumentation was clean this loop)
- FM20 paired field: pre_cbt_boundaries present + non-null in produced JSONs (86/88 top-level; smoke 4/4). Reused committed write path — no new field.
- FM21 marker: `anchor=centroid` on all 5 [CBT-VID] sites incl 2 empty/0-det early-return paths; smoke markers=4/4, full markers=88/88 (outer log + inner run.log agree). Off->argmax byte-identity structurally guaranteed by the if/else gate.
- No FM17 (markers == succeeded videos).

## FM table
- FM1 curve read-only: OK (only seed index changed).
- FM2 count: N/A (count-neutral, 0 added dets).
- FM5 activation: fired (per-cat 92 windows changed; markers 88/88).
- FM6 offline->transfer: adjudicated in-vivo paired, no offline gap.
- FM8 one change: OK (single seed branch).
- FM17 empty marker: OK.
- FM18 absolute headline: OK (new_map = headline).
- **FM19 position-move-not-floor-safe: REALIZED — this killed the loop.** Centroid recenter un-hit currently-clipping windows; RoadAccidents (noisy curves) worst.
- FM20/FM21: instrumentation clean (no v158/v157-style void).

## Classification
- **CLOSED**: centroid / mass-center seed for CBT re-centering. Regressed all 4 hard cats vs recal; the "flat-curve stability" thesis for RoadAccidents/Shooting is falsified — argmax on the noisy curve beat the above-median centroid. Do not re-propose center-of-mass / weighted-centroid / centerness-style seed swaps on this substrate.
- **OPEN / still-standing baseline**: v160 per-category recal (caps {RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5}, trigger 1.1) remains the strongest cross-run lever (0.12085) but was VOID on paired-field grounds — worth a clean paired re-run of recal ALONE (no centroid) to bank it properly.
- Argmax seed is the CBT-recal seed of record; do not change it.

## Next-loop guidance
1. Re-run v160 recal ALONE with the proven pre_cbt_boundaries paired field to legitimately bank the +0.0197 RoadAccidents / 0.12085 mAP that was VOID before — highest-EV move, instrumentation now proven clean.
2. If touching center at all, gate the move by a GT-agnostic confidence/contrast test so only high-contrast unimodal curves move (FM19 guard), and cap the shift magnitude — but center-move on RoadAccidents is now measured-hostile, deprioritize.
3. Shift attention to the UNTOUCHED lever: Shoplifting CANDIDATE-GENERATION (cand-miss 11/25) — CBT boundary-only cannot reach it; needs category-aware scan density / proposal in anomaly_discovery.py.
4. Investigate the full-run absolute-level depression (0.06626 vs expected ~0.099+ baseline) before trusting any close-call full-88 CBT run — smoke was fine, full was not.
