# Loop 12: Score Calibration + Ensemble Research
## Date: 2026-07-10
## Target Version: V77

## Critical Research Insight
AP (Average Precision) depends ENTIRELY on detection RANKING, not absolute
confidence values. Monotonic calibration (Platt/isotonic scaling) CANNOT
improve AP. Only NON-MONOTONIC re-ranking (changing relative order between
detections) can help - specifically, boosting high-IoU detections above
low-IoU ones using signal properties that correlate with localization quality.

## Top Idea (Priority 1): IoU-Aware Score Re-Ranking
Combine 3 zero-shot-computable signals into a "localization quality" score Q:
1. Boundary sharpness: |CLIP gradient| at t_start/t_end (already computed
   via `gradient` array in _score_boundary_proposal)
2. Peak-to-boundary contrast: (peak_score - boundary_score) / peak_score
3. Center score (Idea 5, bonus): how close is the CLIP peak to the segment's
   temporal center - closer to center = better localized
   center_score = 1 - 2*|peak_pos - segment_center| / segment_duration

final_confidence = raw_confidence * (0.4 + 0.6 * Q)  where Q = weighted
combination of the 3 signals, normalized to [0,1]

## Priority 2 (deferred): Weighted Segment Fusion of multiple boundary estimates
## Priority 3 (deferred): Soft-NMS with category-specific sigma
## Priority 4 (deferred): Bayesian duration-prior posterior correction

## V77 Implementation Plan
Add boundary_contrast (already exists as `best_score`) + center_score +
gradient sharpness into the confidence formula in _clip_boundary_detection
and _clip_boundary_single_scale. This is a SCORING-ONLY change - does not
touch boundary detection logic itself, so it's safe/additive (can only help
or be neutral, cannot make boundaries worse since positions are unchanged).

Expected impact: +1.5 to +3 mAP points per research (theoretical), but our
prior experience shows zero-shot pipeline changes often underperform theory
due to VLM stochasticity masking small effects. Full eval needed to confirm.
