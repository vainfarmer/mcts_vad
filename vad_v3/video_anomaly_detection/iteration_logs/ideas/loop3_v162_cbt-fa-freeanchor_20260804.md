# loop3 / v162 — CBT-FA (Free-Anchor top-K smoothed-peak contrast-window selection)

**Date:** 2026-08-04
**Verdict:** REVERT — VOID / BLOCKED (no valid measurement; the v12 instrumentation/execution-loss pattern, this time GPU-contention, NOT a measured-negative).
**Version-collision note:** This is a distinct idea from the earlier loop3/v162 (CBT-SA, smoothed-anchor) and loop4/v163 (CBT-CA, centroid) already in the ledger. The orchestrator re-used version tag "v162" for this CBT-FA re-attempt; chronologically it is the LATEST loop (backup 2026-08-04 21:22, self-test 21:55 — after both CBT-SA Aug-3 and CBT-CA Aug-4 records). Recorded honestly under the tag given; artifacts namespaced `v162_cbtfa_*` to avoid clobbering the prior v162 (CBT-SA) archives.

## Mechanism (as implemented, then reverted)
Single new flag `--cbt_freeanchor` (default OFF, byte-identical off), stacked on committed v160 `--cbt_recal` inside `_cbt_category_boundary_tighten` (pipeline.py). The committed CBT clamp fixes the contrast-window ANCHOR to a single raw peak `peak_idx = lo + argmax(rel[lo:hi])`, then sweeps half-width `hw` maximizing inner-minus-log-weighted-outer contrast. CBT-FA replaces the single argmax with a candidate-anchor set = top-K (K=3) local maxima of a mean-smoothed `rel` (odd window `w=max(3, round(min_seg*fps)|1)`), runs the EXISTING `hw` contrast sweep (extracted verbatim into `_sweep(anchor)`) from each candidate, and keeps the global best `(anchor,hw)`. Per-cat caps, over-wide trigger, monotone-inward clamp to original `[t_s,t_e]`, min_seg floor, frozen confidence, count — all UNCHANGED. Anchor-POSITION axis, orthogonal to the SATURATED CBT-width/caps axis.

## Per-category diagnosis (target rationale)
Diagnosis (nGT/hit@.3/hit@.5/boundary-miss/cand-miss): Shooting 25/9/7/11/5, RoadAccidents 23/6/0/13/4, Shoplifting 25/3/2/11/11, Explosion 22/10/7/11/1. Bet: v155/v160 CBT fires plentifully on Shooting/Explosion but AP barely rises because the single global argmax inside a wide det is a noise spike OFF the true event on flat/noisy curves; top-K smoothed anchors were hypothesized to re-anchor ~5-7 videos from 0-credit to a hit. RoadAccidents preserved (raw argmax already wins global-contrast search → same window re-selected). Shoplifting cand-miss 11/25 unreachable by any boundary lever (GT outside any proposal).

## Headline (ABSOLUTE hard new_map vs committed 0.11806)
**UNMEASURED.** Smoke BLOCKED, full never ran, bootstrap null. No absolute new_map, no per-cat AP, no paired P computable. KEEP gate (new_map >= 0.11470 AND paired P(delta>0) >= 0.9 AND >= 0.09470) UNADJUDICABLE. Central estimate in proposal was 0.133 (recal-only floor ~0.121); none of it was verified in-vivo.

## INSTRUMENT self-test result (the ONE thing that DID pass)
Ran on 2-video CBT-FA self-test set (`results/batch_eval_v162_selftest`, Explosion002 + RoadAccidents123; 3rd/4th videos hung in VLM scan and were stopped after mandatory gates satisfied):
- **FM21 marker:** `grep -cF "freeanchor=on"` = 2; both `[CBT-VID]` lines carry it and mechanism FIRED (fired=1 each). Empty/0-det path verified deterministically by calling the method with empty `final_anomalies` under both toggles → `freeanchor=on`/`freeanchor=off` on the `skip=no_dets` line (FM17-safe).
- **FM20 paired field:** reused the EXISTING `pre_cbt_boundaries` write path (no new field → v158 killer structurally impossible); both self-test JSONs non-null (Explosion002=[[52,62],[51,76]], RoadAccidents123=[[1.5,30.5]]).
- **FM5:** mechanism actually fired (not inert).

The instrument was sound. The loop died on EXECUTION, not instrumentation and not idea quality.

## Why VOID (execution failure, GPU contention)
Smoke launch was BLOCKED: a different full-dataset run `batch_eval_v13best_verify290` (PID 95685, `--max_per_category 999 --bue_extension --qscore_rerank --cbt_tighten`, since ~21:12) was occupying the GPU/MPS. D-smoke waited 3× ~135s cycles per procedure and correctly declined to launch v162 CBT-FA to avoid MPS contention (would corrupt latency + risk OOM). The pre-existing `results/batch_eval_v162_full` is STALE and WRONG-FLAGS — its COMMAND.txt shows `--cbt_mpa` (a prior CBT-SA-era marker), missing the required `--cbt_recal --cbt_freeanchor`; only 6/88 videos, no `evaluation_report.json`. Cannot adopt. So: smoke_map -1, full_map -1, bootstrap null.

## FM table
- FM1 (curve edits regress): N/A — curve read-only, only anchor SELECTION changes.
- FM2 (added→FP): N/A — count-neutral, conf frozen, one det in / one out.
- FM5 (never-activated): CLEARED on self-test (freeanchor=on fired 2/2).
- FM6 (offline≠in-vivo): UNTESTED — never adjudicated in-vivo.
- FM8 (one change): held — single new anchor-set axis on the in-tree recal machinery.
- FM9 (MPS): safe — `rel` already `.detach().cpu().numpy()`; smoothing is pure numpy on that array.
- FM13 (ceiling): boundary-too-wide is the dominant hard miss (~46 GT segs); anchor-position is a plausible residual, unproven.
- FM17 (empty-video marker): CLEARED — marker on the 0-det path both toggles.
- FM18 (absolute mAP): would-be headline; never produced.
- FM19 (position move not floor-safe): THE MAIN UNTESTED RISK — a count-neutral position move; historically NOT floor-safe (cf v161 give-back, v163 centroid regression). Bounded (over-wide-only, monotone-inward, RoadAccidents self-preserving) but never verified.
- FM20 (paired field written): CLEARED (reused existing `pre_cbt_boundaries`).
- FM21 (marker/category mismatch): CLEARED (literal `freeanchor=on` on all 5 [CBT-VID] sites incl empty).
- **Process FM = EXECUTION/RESOURCE-CONTENTION FAILURE** (smoke blocked by a concurrent full-dataset eval; stale wrong-flag full dir). Same VOID class as v12's 3 lost loops, but caused by GPU scheduling, not by the code.

## Status: WON / OPEN / CLOSED
- **WON:** none.
- **OPEN (UNTESTED, re-runnable):** CBT-FA top-K smoothed free-anchor — NOT closed. Instrument is proven-clean; only a completed run is missing.
- **CLOSED:** none new. (CBT-width lever stays SATURATED per v158+v160+v161; centroid center-seed CLOSED per v163; PMX prompt max-pool CLOSED per v159.)

## Next-loop guidance
1. **Serialize GPU access first.** The recurring failure this session (v162 CBT-SA, this CBT-FA) is concurrent full-dataset evals starving the loop's smoke/full. Before launching, `ps aux | grep batch_eval` and WAIT for any full run to finish, or coordinate a single GPU lease. Do NOT launch a second MPS job.
2. **If re-running CBT-FA:** exact stacked flags `--bue_extension --qscore_rerank --cbt_tighten --cbt_recal --cbt_freeanchor --concurrency 2` → `results/batch_eval_v162_full` (clean dir, delete the stale `--cbt_mpa` one first); verify FULL run.log token `recal=on freeanchor=on` on all 88; then `analysis_bootstrap_ab.py --new results/batch_eval_v162_full --paired-json pre_cbt_boundaries`. KEEP iff new_map>=0.11470 AND P>=0.9 AND >=0.09470. BUT: given v161 (CBT-TW) and v163 (CBT-CA) both showed count-neutral position/width moves on these noisy curves regress (FM19 realized twice), EV on CBT-FA is questionable — the flat-curve premise that a smoothed top-3 anchor lands better than argmax is the same premise CBT-SA/CBT-CA tested and lost. Consider it lower-priority than (3).
3. **Higher-EV pivot:** Shoplifting CANDIDATE-GENERATION in `anomaly_discovery.py` (cand-miss 11/25 highest; boundary ops provably cannot fix zero-overlap); cross-run vs v160 baseline; argue FM2 count-effect.
4. **RoadAccidents/Shoplifting category-curve QUALITY** via calibrated CLIP prompt/CoOp validated offline for peak-prominence (NOT curve re-rank, NOT CLOSED v159 PMX).

## Committed baseline unchanged
Committed hard baseline stays v160 (`--cbt_recal`) 0.11806. BUE default-ON / CBT method + recal intact / CEX default-OFF undisturbed.

## REVERT verification
Restored 4 files from `.qoder/backups/v162/` (config.py, run.py, batch_evaluate.py, pipeline.py); `diff -q` silent all 4; `grep cbt_freeanchor|freeanchor` = 0 across the 4 files; py_compile OK; `ScoringConfig()` → bue_enabled=True, cbt_enabled=False, cbt_recal=False, cex_extend=False; CBT method + `pre_cbt_boundaries` present (8 refs). No git checkout/reset/stash/clean used.

Artifacts (CBT-FA self-test only; no hard_eval/bootstrap ever produced): `iteration_logs/results/v162_cbtfa_selftest_run.log`, `v162_cbtfa_selftest_Explosion002.json`, `v162_cbtfa_selftest_RoadAccidents123.json`.
