# Loop 2 (v110) — TVR: Trained-VideoMAE per-detection confidence re-rank

## Proposal (Agent B, deferred from v109)
Re-rank each detection's confidence by the TRAINED VideoMAE sliding-window score
(`videomae_window_scores`, Stage 2b) sampled over its `[t_s,t_e]` support, min-max
normalized within video: `conf' = conf*(1-α + α*mae_norm)`. Rationale: unlike v109's
video-level MIL prior, this is PER-DETECTION → can fix the intra-video mislocation
that v109 identified as the dominant error, on the 34/65 multi-detection videos.

## C review: APPROVE-with-mandatory-offline-precheck
Same safe pattern as v109 (pure post-hoc re-rank; FM1/FM2 immune; FM3/FM4 N/A;
FM6-safe via deterministic re-score of v108 real detections). B's own devil's-advocate
flagged the kill-risk: VideoMAE windows may be saturated / poorly aligned in the
flat-CLIP categories. Gate: offline pre-check must show the signal has discriminative
content before any VLM spend.

## D implementation + result
- Built `analysis_v110_tvr_precheck.py`: reuses the REAL VideoPreprocessor(fps=2,
  motion=1.0) + VideoMAEClassifier to recompute window scores for all 65 videos
  (no VLM), maps to each detection's support, re-ranks, re-evaluates.
- Sanity anchor PASSED (baseline reproduces v108 0.15834).

### KILL-GATE FAILED — the trained VideoMAE window signal is SATURATED
All 64 successfully-processed videos have window-score ranges inside [0.990, 1.000]:
- **No within-video variation** (e.g. Abuse028 [1.000,1.000], Arrest024 [1.000,1.000],
  Shooting032 88 windows all [1.000,1.000]) → min-max normalization amplifies pure
  numerical noise, not localization signal.
- **No normal/anomaly discrimination**: the 5 Normal videos score just as high
  (Normal_Videos_801 [1.000,1.000], 831/905 [1.000,1.000]) as anomaly videos.
- 3-video smoke re-rank delta was +0.00047 (noise, driven by amplified saturation noise).

## Decision: REJECT at offline kill-gate (no full eval spent; pipeline byte-identical)

## Lessons
- The VideoMAE-large (UCF-Crime-finetuned) sliding-window signal, at 32-frame windows
  on fps=2 sampling, SATURATES to ~1.0 (P(anomaly)≈1) on essentially every window of
  every test video — including normals. This directly explains why the pipeline only
  uses it at 0.15 additive weight and why it cannot contribute to temporal localization
  or video-level discrimination. Do NOT build future signals on the raw window score.
- Confirms the v109 meta-lesson from the other side: BOTH available trained signals
  (MIL video-level; VideoMAE window) lack intra-video localization discrimination on
  this benchmark. A useful re-rank signal must come from the score-curve GEOMETRY
  (sharpness/contrast/centering), not from a trained classifier head → motivates v111.
- Reusable asset: `analysis_v110_tvr_precheck.py` caches window scores to
  `analysis_v110_mae_cache.json` for any future VideoMAE-signal experiment.
