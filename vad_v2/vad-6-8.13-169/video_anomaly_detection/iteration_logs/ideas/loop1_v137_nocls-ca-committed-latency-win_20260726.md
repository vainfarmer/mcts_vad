# v137 — NOCLS-CA classify-path DROP: CAPTURED (manual commit, default-ON)

**Date:** 2026-07-26
**Decision:** KEEP (default-ON, committed baseline). Opt out via `--no_classify_skip_class_agnostic`.
**Origin:** v132 (NOCLS-CA-v2) was reverted in the v6 session ONLY because it cleared the
arbitrary 15% single-lever gate at just -7.5%. Under the user's "reduce latency, protect
accuracy" directive, a proven-zero-accuracy-cost -7.5% VLM-call reduction at trivial
complexity is worth keeping. Captured manually after the v6 session closed.

## Mechanism
First-statement short-circuit in `pipeline.py::_ensemble_classify`: when
`config.scoring.classify_skip_class_agnostic` is True (now default), log `[NOCLS-CA]` and
return the upstream VideoMAE label (`videomae_cat if videomae_cat and !='Normal' else
'unknown'`) WITHOUT issuing the Stage-7 `qwen3-vl-plus` classify VLM calls
(`_classify_clip_guided` <=1 call + confusion-pair vote 3 calls => <=3 calls/video).

## mAP-neutrality — RE-VERIFIED against live source (not just trusting v132)
- `compute_temporal_map` (evaluate.py:306-380) builds predictions from ONLY
  `{video_id, segment (t_start/t_end), confidence}`; `anomaly_type` is never read.
- The classify result is assigned only to `a['anomaly_type']` (pipeline.py:1208) and carried
  into the output dict (pipeline.py:1263) — read by `compute_category_accuracy` (a SEPARATE
  Level-4 metric), not by temporal mAP.
- Everything geometric downstream keys off the UPSTREAM `videomae_category` (passed INTO
  classify), NOT the classify result: confidence gate (pipeline.py:1234 uses
  `CATEGORY_CONF_GATES[videomae_category]`), clip_score_curve_correction (1219), frame
  scores (1271). So t_start/t_end/confidence/count are all invariant to skipping classify.
- The only per-detection `anomaly_type` consumers (cat-skips at pipeline.py:1352 / 1524) live
  inside `_event_continuation_extension` / `_event_unit_boundary_contraction`, both default-OFF
  inert. => Skipping classify is mAP-NEUTRAL BY CONSTRUCTION.

## Files changed (backups in .qoder/backups/v137/)
- config.py: `ScoringConfig.classify_skip_class_agnostic = True` + from_args threading (default True).
- batch_evaluate.py, run.py: `--classify_skip_class_agnostic` (store_true, default True) +
  `--no_classify_skip_class_agnostic` (store_false opt-out), mirroring the qscore pattern.
- pipeline.py: `_ensemble_classify` short-circuit + `[NOCLS-CA]` marker; docstring bumped to v137.

## Validation
- AST OK x4; `import pipeline` OK; both flags in `--help`; `ScoringConfig().classify_skip_class_agnostic == True`.
- Confirmation smoke (14 videos, flags on): `[NOCLS-CA]` fired 10/14 (returns VideoMAE label, no
  classify call); [COST] logs intact; smoke mAP 0.20184 (>= floor, smoke-noisy). The 9 log
  tracebacks were all transient VLM API network errors in `_directed_llm_scan` (detect path,
  httpx/openai ConnectTimeout) — UNRELATED to this change (NOCLS path in zero tracebacks).
- Call reduction: v132 in-vivo full eval measured mean_vlm_calls 29.5846 vs REF 31.969 = **-7.5%**
  (identical mechanism). This commit reproduces that behavior with default-ON.

## Tradeoff (documented)
Degrades the separate Level-4 category-accuracy metric (returns coarse VideoMAE label instead
of the VLM-refined category). Does NOT affect temporal mAP (the headline metric). Reversible
with `--no_classify_skip_class_agnostic`.

## Net
Committed latency win: ~-7.5% per-video VLM calls at zero temporal-mAP cost (by construction).
Baseline temporal mAP stays 0.15632. This is the proven-safe ceiling of by-construction-neutral
call-dropping (v6/v135 proved the detect path cannot be losslessly reduced under the current
coarse_score-coupled confidence formula).
