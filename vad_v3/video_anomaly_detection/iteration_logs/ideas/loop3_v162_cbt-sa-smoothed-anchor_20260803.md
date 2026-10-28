# Loop 3 / v162 — CBT-SA (Smoothed-Anchor recalibration) — VERDICT: REVERT (VOID, no valid measurement)

## Mechanism
Single new flag `--cbt_anchor_smooth` (default OFF), stacked on the committed `--cbt_recal` (v160) + `--cbt_tighten` (v155). Inside `_cbt_category_boundary_tighten` (pipeline.py), before the CBT inward-clamp CENTER-seed argmax, box-car smooth the in-window segment `rel[lo:hi]` with a kernel of the category's own `min_segment` width (odd-forced), then `peak_idx = lo + argmax(smoothed_seg)`. The width-contrast hw-search and the monotone inward clamp to original `[t_s,t_e]` continue to use RAW rel — UNCHANGED. Pure center-seed refinement: no width, count, confidence, or curve-shape edit. Flag-off restores `peak_idx = lo + argmax(rel[lo:hi])` byte-identically.

Goal: convert already-tight v160-recal windows from IoU~0.5 to IoU>=0.7 — target the AP@0.7 ceiling (v160 stacked baseline AP@0.7=0.0303 was the sole binding free variable; AP@0.3/.5 near-saturated).

## Per-category diagnosis (targeting rationale)
- Explosion (blast = sharp event; cand-miss only 1/22, hit@.5=7 windows already FOUND but centers loose) — primary contributor.
- RoadAccidents (impact moment sharp; v160 recal lifted AP@.5 0.067->0.202, center-tightening would convert found windows to AP@.7) — primary.
- Shooting (gunshot sharp, hit@.5=7) — secondary/marginal.
- Shoplifting explicitly NOT targeted: cand-miss 11/25 is candidate-generation (GT never proposed); CBT cannot touch zero-overlap videos.

## Measurement — VOID / NOT COMPLETED (this is why REVERT)
- **BOOTSTRAP: null.** No paired adjudication was ever produced.
- **FULL(88): NOT completed.** `results/batch_eval_v162_full/` reached only 6/88 videos, no `evaluation_report.json`. INVALID. Worse, its `run.log` [CBT-VID] markers read `recal=off mpa=on` — i.e. the interrupted full run was launched WITHOUT `--cbt_recal` (recal=off) and against an EARLIER marker token (`mpa=on`, not the source `smooth=on`), so it would not even have reflected the committed stacked baseline the proposal requires.
- **SMOKE: CRASHED / partial.** `results/batch_eval_v162_smoke/` = 4 videos (1/cat), only **2 succeeded (Explosion002, RoadAccidents123), 2 FAILED (Shooting, Shoplifting)**. The reported temporal mAP 0.5152 (AP@.3 1.0 / AP@.5 0.545 / AP@.7 0.0) is over the 2 easy survivors only — NOT a hard-harness number and NOT comparable to the v160 0.11806 / v161 0.09448 14-video smoke figures. Treated as smoke_map = -1 (invalid).

**Adjudication:** KEEP requires new_map >= 0.11470 (and vs committed v160 0.11806) AND paired P(delta>0) >= 0.9. Neither the absolute new_map nor the paired P could be computed — the loop produced NO valid 88-video result and NO bootstrap. Per session discipline a loop with no valid measurement is VOID and defaults to REVERT. The idea itself was NOT measured-negative (contrast v161, which was a clean measured regression); CBT-SA remains UNTESTED.

## Instrument self-test (per D1 IMPL, code-level, PASSED before eval)
- FM20 paired field: reused the EXISTING `pre_cbt_boundaries` write path (pipeline.py L1730-1765) verbatim — no new field, so the v158 killer is structurally impossible. Smoke JSONs confirmed non-null (Explosion002, RoadAccidents123).
- FM21 marker: source token `smooth=on/off` threaded to all 5 [CBT-VID] sites incl the 2 empty/0-detection early-returns (FM17). NOTE the interrupted FULL run.log shows `mpa=on` instead — a marker-token inconsistency between the code path used for the full launch and the committed source; this alone would have failed an FM21 grep for `smooth=on` on the full run.

## FM table
- FM1 curve-shape: OK — rel untouched for scoring, only the argmax seed smoothed.
- FM2 count-neutral: OK — no det added/removed/split, confidence frozen (non-additive).
- FM5 not-activated: N/A — mechanism fired in the 2-video smoke, but never reached a valid full eval.
- FM6 offline-no-transfer: intended in-vivo via paired pre_cbt_boundaries — but the paired bootstrap was never run.
- FM8 one change: OK — single flag; recal is a pre-committed flag.
- FM9 MPS: OK — rel is `.detach().cpu().numpy()` ndarray before np.convolve.
- FM13 ceiling: bound first (AP@0.7=0.0303 confirmed binding).
- FM17 empty-video marker: covered at L1440/L1547.
- FM18 absolute mAP headline: intended new_map — never obtained.
- FM19 count-neutral POSITION move NOT auto floor-safe: THE MAIN THEORETICAL RISK (a fixed-width center shift can drop a .3/.5 hit; cf. v161's measured .5 give-back). Untested here.
- **FM (process): INSTRUMENTATION/EXECUTION FAILURE** — the loop was lost to an incomplete full run (6/88) launched with the wrong flags (recal=off) and an inconsistent marker (mpa vs smooth), plus a crashed smoke (2/4 videos failed). This is the v12 failure pattern (loops lost to instrumentation, not ideas). The idea was never adjudicated.

## WON / OPEN / CLOSED
- **OPEN (UNTESTED):** CBT-SA / smoothed center-seed for the CBT inward clamp. NOT closed — no valid measurement exists. Re-runnable if execution is fixed.
- WON: nothing.
- CLOSED: nothing new. (v161 already CLOSED decoupled tight-target width-cap; broader CBT width lever remains SATURATED per v158+v160+v161.)

## Next-loop guidance
1. If re-attempting CBT-SA: launch the FULL 88-video eval with the EXACT stacked flags `--bue_extension --qscore_rerank --cbt_tighten --cbt_recal --cbt_anchor_smooth --concurrency 2` and DRIVE IT TO COMPLETION (evaluation_report.json present, all 88 done) BEFORE running the paired bootstrap `analysis_bootstrap_ab.py --new results/batch_eval_v162_full --paired-json pre_cbt_boundaries`. Verify the marker on the FULL run.log matches the committed source token (`smooth=on`, recal=on) — the aborted run showed `mpa=on recal=off`, a launch/flag defect. Fix the crashing Shooting/Shoplifting smoke videos first so the smoke actually validates the hard cats.
2. Higher-value untried lever (per v160/v161 NEXT): Shoplifting CANDIDATE-GENERATION in anomaly_discovery.py (cand-miss 11/25 highest; boundary ops provably cannot fix zero-overlap; cross-run vs v160 baseline; argue FM2 count-effect).
3. RoadAccidents/Shoplifting category-curve QUALITY via calibrated CLIP prompt/CoOp (NOT curve re-rank, NOT the CLOSED v159 PMX max-pool).
4. CBT width lever is SATURATED — do not re-tune caps/trigger/max_hw.

## Committed state (unchanged) + REVERT verification
Committed hard baseline stays v160 (`--cbt_recal`) new_map 0.11806. REVERT verified: restored 4 files (pipeline.py, config.py, run.py, batch_evaluate.py) from `.qoder/backups/v162/`; `diff -q` silent (byte-identical); `grep cbt_anchor_smooth|smooth=on|smooth=off|mpa=` = 0; `py_compile` OK; `bue_enabled=True` & `cbt_enabled=False` & `cbt_recal=False` & `cex_extend=False` intact; CBT method `_cbt_category_boundary_tighten` + `pre_cbt_boundaries` present. Backups at `.qoder/backups/v162/`. Artifacts: `iteration_logs/results/v162_{smoke_eval.json, smoke_run.log, full_run.log}` (no hard_eval/bootstrap — never produced).
