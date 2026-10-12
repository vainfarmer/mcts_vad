# v156 — CPA-G: Contrast-Guarded Category Peak Alignment (REVERT)

**Date 2026-08-02. Loop 2 of session v12. Verdict: REVERT — floor breached + mechanism was a functional NO-OP.**

## Mechanism
Terminal (post-BUE, post-CBT, pre-CEX) **count-neutral re-centering** of brief-hard-category
detections (BRIEF_CPA = {Shooting, RoadAccidents, Explosion}) onto the category-curve prominence
peak, fired ONLY when the AutoLoc inner-minus-outer contrast at the candidate peak window strictly
exceeds the contrast at the detection's current window by `cpa_margin`. Synthesis of peak
re-centering (Idea A) + contrast objective as a safety gate (Idea B). Confidence and detection
COUNT frozen; only `(t_start, t_end)` of an existing detection translates. New terminal method
`_apply_cpa` copied from `_cbt_category_boundary_tighten`, differing in that the chosen window may
sit OUTSIDE the original span (CBT's inward clamp provably cannot re-center outside `[t_s,t_e]`).

Gates (all required to fire): (a) prominence `>= cpa_tau` (1.0); (b) peak OUTSIDE current window;
(c) edge-overshoot `<= cpa_radius_frac * det_dur` (0.5); (d) contrast(peak) `>= contrast(current) +
cpa_margin` (0.05); plus current-window-support gate d1 (`cpa_cur_max` 0.0).

## Per-category diagnosis (motivation, from full-290 BUE taxonomy, cache-based, CBT-OFF)
- **Explosion**: boundary-bucket center offset **+6.0s** — largest displacement, curve peak sits
  ~6s off detection center, CBT-untouchable. Primary target (~12 boundary videos).
- **Shooting**: offset **+2.5s**, ~9 boundary videos. Secondary.
- **RoadAccidents**: offset **+0.7s** — width-dominated, not offset; CPA-G helps least. ~14 boundary videos.
- **Shoplifting**: NOT targeted — genuine signal/candidate-generation (NOGEN) failure, wrong lever for a boundary re-center.

Claimed honest conversion: ~7-11 videos gaining higher-IoU hits; point estimate +0.022 hard mAP.

## Result — HEADLINE
| Metric | Smoke (4 vid) | Full (88 vid) | Committed baseline | Floor |
|---|---|---|---|---|
| mAP | 0.10227 | **0.04339** | 0.04928 | 0.09470 |
| AP@0.3 | 0.13636 | 0.09259 | 0.106 | — |
| AP@0.5 | 0.13636 | 0.02932 | 0.035 | — |
| AP@0.7 | 0.03409 | 0.00826 | 0.006 | — |

**Bootstrap: PAIRED VERDICT VOID.** `pre_cpa_boundaries` was present (133 index-aligned rows), but
**0 of 133 detections had a window that actually changed** — both arms identical, no delta
distribution. point_delta / CI90 / P(delta>0) all degenerate 0.

**Per-category cross-run AP (v156 vs v155, IoU {0.3,0.5,0.7}) — ALL 4 regressed:**
Shooting −0.0148 (0.0636 vs 0.0783); RoadAccidents −0.1304 (0.0714 vs 0.2018);
Shoplifting −0.0556 (0.0177 vs 0.0732); Explosion −0.0298 (0.1311 vs 0.1609).

## Interpretation (load-bearing)
The mechanism changed **zero windows**, so the flag-on run is functionally identical to flag-off.
Therefore the full-run mAP gap (0.04339 vs committed 0.04928) is **run-to-run VLM nondeterminism
noise, NOT the mechanism**. The regression is a red herring in the sense that CPA-G did no work;
the real finding is that the gate stack (edge-overshoot at radius_frac=0.5 + cpa_margin=0.05 +
current-support gate d1) is too tight to admit ANY move. Predicted in IMPL weaknesses: a +6s
Explosion re-center at radius_frac=0.5 requires the post-BUE window to be wider than ~2*offset
(~12s); post-BUE brief-cat windows are evidently narrower, so the edge-overshoot cap blocked every
candidate. Markers fired 88/88 ([CPA-VID], FM17-safe) but [CPA] fire-count on the full set was
effectively 0 real moves — a "logged-but-inert" outcome (FM5-adjacent: activation marker present,
mechanism did no work).

## FM table
- **FM2** (adds→FPs): N/A — count-neutral, 0 adds. Clean.
- **FM5** (never-activated): [CPA-VID] fired 88/88, BUT the CORE re-center fired 0 real moves → the
  mechanism was effectively never-activated in-vivo. The per-video marker masked the inert core.
  LESSON: also assert a nonzero real-move count in smoke, not just marker presence.
- **FM16** (noise floor ±0.02): the observed 0.04339 vs 0.04928 gap (~0.006) is well inside
  run-to-run variance for a 0-move mechanism → confirms harness nondeterminism at this scale.
- **FM17** (empty-video deadlock): SAFE — 88/88 markers, both 0-det early-returns covered.
- **FM18** (absolute mAP): 0.04339 < floor 0.09470 → REVERT regardless of VOID paired verdict.
- **FM19** (count-neutral position moves not floor-safe): the headline risk, but UNTESTED — the
  contrast-margin + radius gates were so conservative that no move ever fired, so the FM19 hardening
  was never exercised.

## Category-lever status: **OPEN (untested), NOT closed**
The "category-aware boundary re-center outside CBT's inward clamp" lever is NOT closed — it was
never actually exercised. The specific v156 gate configuration is a no-op. The underlying structural
gap (CBT's monotone-inward clamp cannot move a center onto an out-of-window curve peak) remains real
and unaddressed. Distinct from banned-1 (curve confidence re-rank) and banned-7 (BUE outward union).

## Next-loop guidance
1. **First, empirically measure the post-BUE/post-CBT window-width distribution** for brief-cat
   detections on the 88 harness (from the archived v156 result JSONs — count-neutral, so widths ==
   committed). If widths are << 2*offset, the edge-overshoot gate at radius_frac<=0.5 is
   structurally incompatible with a +6s move and the whole radius-clamp design must be rethought
   (e.g. gate on the ABSOLUTE re-center distance vs a category duration prior, not vs det_dur).
2. **Instrument the real-move count** as a hard smoke gate: if 0 moves across the smoke set, DO NOT
   launch the 88-eval — relax radius_frac / margin first or abort. This loop wasted a full 88-eval
   on an inert mechanism because the smoke only checked [CPA-VID] presence, not real fires.
3. If widths permit, re-run with cpa_margin=0.0, radius_frac large (or distance-vs-prior gate) to
   let Explosion +6s moves through, THEN re-add the contrast margin to control FM19 — but ONLY after
   confirming nonzero fires.
4. Shoplifting stays deferred to a query-quality / prompt_weights (CoOp-lite) candidate-gen lever.
