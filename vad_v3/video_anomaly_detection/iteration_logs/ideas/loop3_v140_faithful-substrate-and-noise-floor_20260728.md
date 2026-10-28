# Loop 3 — v140: live-faithful curve substrate, and the harness NOISE FLOOR

**Date:** 2026-07-28
**Type:** measurement-infrastructure loop. **ZERO API cost** (local CLIP only), **ZERO pipeline code
changed** — working tree remains v137, byte-identical to the committed baseline.
**Outcome:** the substrate problem is SOLVED and validated; but the same tooling then proved that the
harness cannot measure the effect sizes this arc has been chasing. **Recommend halting the
+0.005-scale re-rank programme.**

---

## 1. What was built

`analysis_v140_livecurve_lib.py` — reproduces the in-vivo verdict-independent curve exactly, by
driving the pipeline's OWN classes with the batch runner's config and no LLM client:

```
VideoPreprocessor(fps=2, seg=6.0, motion_thresh=1.0, min_motion_int=2.0, overlap=0.25)
  -> VideoAnomalyExpert(ViT-L-14/openai).precompute_video_features()   # == pipeline.py:987
  -> FrameScorer.get_category_scores()                                 # == frame_scorer.py:41-61
```
Curves cached to `analysis_v140_livecurves/*.npz` (58 videos). No VLM/LLM client is constructible in
this path, so an API call is impossible by construction.

**FIDELITY GATE — passed exactly.** Validated against the live peak times recorded in v138's `[GPC]`
markers: **58/58 video+category pairs reproduced, |peak delta| median = mean = max = 0.000 s.** This
is the check the v138/v139 battery never had.

## 2. Q1 — the substrate retro-predicts BOTH in-vivo refutations

| arm | OLD substrate (features/test) | NEW faithful substrate | actual in vivo |
|---|---|---|---|
| dense `peak_in` (v138) | +0.0191, 8/8 positive | **-0.00775, 0/8** | **-0.00287** |
| pooled-32 `peak_in` (v139) | +0.0191, 8/8 positive | **-0.00412, 2/8** | **-0.00149** |

Correct sign on both, and correct ORDERING (pooled better than dense). The v138/v139 offline
`+0.0191` is therefore confirmed to have been **100% substrate artifact**, and FM15 is now closed
with a working remedy rather than just a warning.

## 3. Q2 — screening, and why its apparent winner is not real

Best arms on the faithful substrate (w=0.2, 8 caches): `peak_in` at 20 bins **+0.01016 (7/8)**,
16 bins +0.00630 (7/8), 8 bins +0.00612 (8/8); `contrast_norm` +0.00395; `act_max_norm` +0.00156
(8/8). `peak_in_p16` also has the best IoU>=0.3 AUC of any feature (**0.7990** vs cached confidence
0.7701). Superficially promotable. It is not, for two independent reasons.

**(a) The pooling parameter has no plateau — it flips sign between adjacent settings.**

| bins | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 | 48 | dense |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean dmAP | -0.0023 | +0.0068 | +0.0061 | +0.0069 | +0.0032 | +0.0063 | **+0.0102** | +0.0053 | **-0.0041** | +0.0065 | -0.0078 |

24 → 32 → 48 goes +0.0053 → **-0.0041** → +0.0065. No mechanism behaves that way. And mean AP@0.7 is
NEGATIVE at every single setting (-0.0009 to -0.0066).

**(b) "8/8 caches positive" was never 8 independent confirmations.** Pairwise Jaccard of detection
windows across the 8 caches: **mean 0.813, median 0.806, min 0.739**. The caches are re-runs over the
same 65 videos and share ~81% of their detection windows, so a deterministic rule agrees across them
almost automatically. The dual-cache rule I introduced in loop 1 (and the 8-cache battery built on
it) measures far less than it appears to.

## 4. THE LOAD-BEARING RESULT — the harness noise floor (video-level bootstrap)

Bootstrapping over VIDEOS (the true sampling unit), 400 resamples, faithful substrate:

| arm | v124 mean | v124 90% CI | P(d>0) | v130 mean | v130 90% CI | P(d>0) |
|---|---|---|---|---|---|---|
| bins=8 | +0.00709 | [-0.0191, +0.0304] | 0.71 | +0.00780 | [-0.0175, +0.0290] | 0.74 |
| **bins=20** | +0.01968 | [-0.0042, +0.0429] | **0.93** | +0.01030 | [-0.0106, +0.0331] | 0.78 |
| bins=32 | +0.00526 | [-0.0221, +0.0346] | 0.64 | +0.00001 | [-0.0259, +0.0257] | 0.52 |
| dense | -0.00763 | [-0.0345, +0.0184] | 0.33 | -0.01410 | [-0.0400, +0.0099] | 0.21 |

**Every CI straddles zero, including the best arm.** The CI half-width is **~±0.025** — five times the
+0.005 promotion gate this project has used since v114.

**Consequences (these apply retroactively to the whole arc, not just to v138-v140):**
1. **No +0.005-scale re-rank lever can be validated on this 65-video harness, ever.** The gate sits
   5x below the measurement's resolution. This is a property of the harness (n_gt=66, 11-point
   interpolated AP, max recall 0.379 @IoU0.3 ⇒ mAP is decided by precision at just 4 hit
   checkpoints, so one detection crossing one checkpoint moves mAP by ~0.005-0.010).
2. **The committed Q-score baseline win (v111/v114, +0.00619) is itself inside this noise band** and
   is not statistically distinguishable from zero on this harness. It should be regarded as
   unfalsified rather than as a demonstrated gain.
3. The v115-v123 nine-rejection arc, the sporadic keeps, and the offline/in-vivo mismatches are all
   consistent with sampling noise around zero. That is a simpler explanation than nine distinct
   mechanistic failures.
4. Scaling: CI half-width shrinks ~1/sqrt(n_videos). Detecting +0.005 at 90% needs ~25x more videos
   (~1600 — not available). The full 290-video test split gives ~±0.012, so **even with every test
   video, only effects >= ~+0.03 are reliably measurable.**

## 5. Recommendations (decision needed from the owner — these cost API budget)

1. **STOP the +0.005-scale re-rank programme.** It is unfalsifiable on this harness. Do not spend
   further evals on curve-derived confidence re-ranking.
2. **Only pursue mechanisms with an expected effect >= +0.03**, i.e. things that raise the recall
   ceiling / localization quality (currently max recall 0.379 @0.3, 0.212 @0.5; 50/99 detections have
   ZERO GT overlap) or supervised score quality — not ranking refinements. The pure-re-rank oracle
   ceiling is 0.27273, and within-video re-ranking is provably exhausted (oracle +0.006).
3. **Enlarge the harness before any further accuracy verdict.** Raising `--max_per_category` from 5 to
   ~15 (~195 anomaly videos) cuts the CI to ~±0.014; the full 290-video split (which also restores
   the ~51.7% Normal population that the latency session found missing) gives ~±0.012. Cost estimate
   at the observed ~200 s/video and concurrency 2: ~5.5 h for 195 videos, ~8 h for 290.
4. **Adopt the video-level bootstrap CI as the standard verdict statistic**, replacing point-estimate
   deltas and multi-cache agreement. Both of the latter are now shown to be misleading here.
5. **Land the raw-curve caching in the pipeline** (~2 lines: also save `get_category_scores` alongside
   the flooded `frame_scores`) so future prechecks are faithful without re-encoding. This loop's
   library makes it optional, not urgent.

## 6. Assets

- `analysis_v140_livecurve_lib.py` (+ `analysis_v140_livecurves/` cache, 58 videos) — faithful
  substrate, fidelity-gated.
- `analysis_v140_faithful_battery.py` / `.json` — retro-prediction test + feature screening.
- `v140_fidelity.log` — the 58/58 fidelity record.
- Prior loop assets unchanged: `analysis_v138_paired_invivo.py` (paired in-vivo A/B — still the right
  tool, and the only one that produced correct verdicts), `analysis_v138_rankdiag_orch.py`.

## 7. What loop 3 did NOT do

No pipeline code was changed; no evaluation was run; no API call was made. The working tree is v137
(committed baseline, mAP 0.15632, -7.5% VLM calls), verified byte-identical.
