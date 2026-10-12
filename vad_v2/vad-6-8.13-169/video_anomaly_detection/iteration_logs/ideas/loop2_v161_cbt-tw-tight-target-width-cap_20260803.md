# v161 — CBT-TW: Tight-Target Width Cap (decouple CBT search ceiling from over-wide fire cap)

**Loop 2, session v97. VERDICT: REVERT — floor breached + P<0.9 gate fail. Real measured regression (NOT an activation/instrument void this time).**

## Mechanism
ONE behavioral change stacked on the committed v160 CBT-recal (flag `--cbt_recal`, default OFF). The committed `_cbt_category_boundary_tighten` (pipeline.py) used ONE per-category scalar `cap` for BOTH (a) the over-wide fire gate `det_dur <= trig*cap` and (b) the ceiling of the contrast-window half-width search `max_hw = max(min_hw, int(cap*fps*0.5))`. CBT-TW decoupled the two: fire gate stays at the recal cap+trigger (SAME dets fire — count-neutral), search ceiling shrinks to a tighter per-category target so fired dets clamp closer to GT-median and cross IoU 0.7. New flag `--cbt_tight_target`, config `cbt_tight_caps` (GT-p50 from actual UCF-Crime test parquet: RoadAccidents 3.0, Explosion 7.0, Shooting 9.0; Shoplifting intentionally ABSENT = `.get(cat,cap)` no-op fallback). Flag-off => `tw==cap` => byte-identical to committed BUE + (recal when `--cbt_recal`).

## Per-category diagnosis (hit@.3/hit@.5 targets)
Motivation: v160 recal lifted AP@0.5 0.030->0.125 but AP@0.7 stalled at 0.0303 — the clamped sub-window was allowed to be as wide as the full p75 cap, so the contrast objective settled ~1.3-1.5x GT (clears .5, not .7). CBT-TW aimed to move ~4-6 mid-IoU segs from the .5-band into the .7-band. C-MANDATE ceiling bound (from v160 post-CBT widths) showed material headroom: RoadAccidents frac(post>3.0)=0.25, Explosion frac(post>7.0)=0.32, Shooting frac(post>9.0)=0.10 — NOT a no-op, warranted full eval.

## Instrument self-test (the v156-v158 killers — ALL PASSED this loop)
- **FM20 paired field**: REUSED existing `pre_cbt_boundaries` write path (no new field => v158 killer structurally impossible). Smoke JSONs confirmed non-null lists (Explosion002 pre=[[52,62],[51,76]] final=[[52,62],[53.5,60.5]]; RoadAccidents123 pre=[[1.5,30.5]] final=[[14.5,17.5]]). Full run: 216 paired JSONs written.
- **FM21 marker**: literal `tight=on` appended to ALL FIVE [CBT-VID] sites incl 2 empty/0-det early-returns (FM17-safe). Smoke marker_count=14/14, full marker_count=290/290. Toggle self-tested both directions (config-threaded, not hardcoded).
- **FM5 mechanism FIRED**: smoke [CBT] Explosion002 25->7 (TW tightened v160's 9.0->7.0), RoadAccidents123 29->3, conf frozen, ndet unchanged, count-neutral confirmed.

The instrument was clean — this loop's REVERT is a genuine mAP verdict, not a gate artifact.

## Result — HARD 88-video (headline)
- **SMOKE(14)**: mAP 0.2343 (AP@0.3 0.3455 / AP@0.5 0.2061 / AP@0.7 0.1515), lat 164.96s, marker 14/14.
- **FULL(88) aggregate**: mAP 0.16852 (AP@0.3 0.2597 / AP@0.5 0.1444 / AP@0.7 0.1014), lat 246.74s, marker 290/290, pairedJSONs=216.
- **BOOTSTRAP** (paired-json[pre_cbt_boundaries], n=290 common, 1000 resamples):
  - point_delta **+0.02003**, 90% CI **[-0.00762, +0.03858]**, **P(delta>0)=0.858**.
  - **ABSOLUTE hard new_map = 0.09448** (CBT-TW-on arm vs reconstructed BUE-only paired baseline).
- **Gates**: point_delta clears +0.02, but P=0.858 < 0.9 FAILS; new_map 0.09448 < KEEP floor 0.11470 AND < committed v160 headline 0.11806 (-0.02358) AND < session floor 0.11306. **BOTH gates + headline => REVERT.**
- **Per-category deltas (v161 vs v160, ALL NEGATIVE)**: Shooting -0.0006; RoadAccidents **-0.0446** (dominant); Shoplifting -0.0061; Explosion **-0.0304**.

## Interpretation
The tighter search ceiling did NOT convert mid-IoU into high-IoU hits — it OVER-SHRANK. RoadAccidents (-0.0446) and Explosion (-0.0304) drove the regression: forcing the contrast objective to a sub-GT-median ceiling pulled already-good v160 windows below their peak-centred optimum, dropping .5-band hits faster than it added .7-band hits. The predicted "AP@0.7 roughly doubles while AP@0.5 holds" did not materialize; the AP@0.5 give-back (Weakness #1) dominated. The contrast objective in v160 was already picking near-GT windows for the healthy-contrast cats, so tightening the ceiling only removed degrees of freedom it needed.

## FM table
- **FM1** curve read-only — held (no curve edit).
- **FM2/FM19** count-neutral (over-wide gate + trigger unchanged, monotone-inward clamp + min_seg floor) — held; but FM19 floor-safety was about ADDED segs, not the width-shrink give-back that actually regressed AP@0.5. New lesson: count-neutral inward *width* moves are NOT AP-safe when the pre-move window was already near-optimal.
- **FM5** mechanism FIRED (not inert) — clean.
- **FM6** offline GT-p50 caps did NOT transfer (predicted risk realized): parquet GT-median caps degraded in-vivo because post-BUE/post-recal windows are not centred to permit symmetric shrink to GT-median.
- **FM8** exactly ONE behavioral change (max_hw ceiling) — held.
- **FM13** ceiling bound BEFORE eval (C-MANDATE #1) showed 25-32% headroom — headroom existed but was the WRONG direction (shrinking hurt).
- **FM16** effect vs noise: point_delta +0.020 sat ON the CI edge (CI straddled 0), classic noise-floor ambiguity; the ABSOLUTE new_map is the honest headline and it regressed.
- **FM17** marker on all 5 sites incl empties — safe.
- **FM18** headline = absolute hard new_map 0.09448 < floor => REVERT.
- **FM20/FM21** instrument self-tests PASSED (first CBT-family loop this session with a fully clean instrument that produced a real verdict).

## Status: CLOSED
**CBT-TW / decoupled tight search-ceiling is CLOSED (measured-negative, -0.0446 RoadAccidents / -0.0304 Explosion, P=0.858).** Tightening the contrast-window search ceiling below the recal p75 cap over-shrinks near-optimal windows. Do NOT re-propose ceiling-tightening on the CBT-recal method — v160 recal caps are the tightening saturation point for the boundary lever.

## Next-loop guidance
1. **CBT-recalibration boundary lever is SATURATED** (v158 dir + v160 KEEP + v161 over-shrink all triangulate: p75 caps ~optimal, p50 too tight). Stop tuning CBT width — the ~46 boundary-too-wide residual that CBT could reach is now converted; remaining boundary misses are on flat-contrast videos CBT lands arbitrarily (v157 finding).
2. **Shoplifting CANDIDATE-GENERATION** (anomaly_discovery.py): cand-miss 11/25 is the highest unaddressed bucket; boundary ops (incl v155/v160/v161) provably CANNOT fix zero-overlap videos. Category-aware scan density / proposal where GT actually is; argue FM2 count-effect (additive dets demote precision unless they land on GT). Cross-run vs the v160 baseline (NOT v155).
3. **RoadAccidents/Shoplifting category-curve QUALITY** (CLIP prompt/CoOp calibration, validated offline for peak-prominence BEFORE eval — NOT curve re-rank, NOT v159 PMX max-pool which is CLOSED). Three loops (v157/v158/v159) plus this one point at flat in-span contrast as the true residual for the subtle cats.

**Committed hard baseline remains v160 CBT-recal (--cbt_recal), hard mAP 0.11806. BUE default-ON, CBT method + recal intact, CEX default-OFF, all undisturbed.**

## Revert verification
Restored pipeline.py/config.py/run.py/batch_evaluate.py from `.qoder/backups/v161/`; `diff -q` silent (all 4 identical to backup); `grep cbt_tight_target/cbt_tight_caps/tight= = 0` across all 4 files; py_compile OK; `bue_enabled=True & cbt_enabled=False & cex_extend=False & cbt_recal=False` intact; `_cbt_category_boundary_tighten` present; `pre_cbt_boundaries` write path present. models.py was never touched (paired field reused existing path). Backups: `.qoder/backups/v161/`. Artifacts: `iteration_logs/results/v161_{hard_eval.json,hard_run.log,bootstrap_ab.json}`.
