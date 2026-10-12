# Normal-video recognition schemes — design memo (v155, design-only)

**Date 2026-08-02. DESIGN ONLY — none evaluated on mAP this session. Owner will test separately.**

Measured fact: perfectly deleting ALL Normal-video FPs is only **+0.00253 mAP** (this is banned family 9 for the mAP objective). BUT it matters for deployment false-alarm rate and frame-level / video-level AUC. Each scheme below carries a **sub-floor-only invariant** as its mAP-neutrality proof, so it can raise Normal recognition WITHOUT moving the pooled PR curve.

## (a) Supervised sub-floor normalcy gate
Calibrate a `model_mil_v6` video-score threshold **tau on TRAIN Normals**. For a test video with `mil_score < tau`, suppress ONLY detections whose confidence is below the LOWEST confidence ever matched as a TP in the committed run (the "sub-floor band").

**mAP-neutral by construction:** deleting only detections ranked strictly below the last matched TP moves no point on the pooled PR curve — sub-floor dets contribute 0 recall, and their removal cannot raise or lower precision at any recall the curve actually reaches. Strictly cuts deployment false-alarm rate on Normals. Verdict-independent, pre-calibratable, count-reducing only in the sub-floor band.

**Risk / discipline:** tau MUST be calibrated TRAIN-only (no test leakage). The "lowest-matched-TP confidence" is a committed-run constant read once. If tau were tuned on test, or the sub-floor threshold set above the last-TP confidence, neutrality breaks.

## (b) Cheap normalcy VLM re-verify
Issue ONE binary "is this clip normal?" VLM call ONLY on videos where ALL detections are already sub-floor confidence (rare -> low incremental cost). A confident-normal verdict drops those sub-floor dets.

**mAP-neutral:** touches only the sub-floor band, adds no detection (FM2-safe). Reduces false alarms on ambiguous Normals.

**Risk:** adds VLM calls on a small subset; gate the call strictly on the all-sub-floor condition to bound cost. Do not let a "normal" verdict touch any above-floor detection.

## (c) Measurability (additive reporting, no objective change)
Add frame-level and video-level ROC-AUC to `evaluate.py`, reported ALONGSIDE temporal AP:
- **frame-level**: build a per-frame normal/anomaly label vector from GT segments and integrate the existing `frame_scores` (computed at pipeline.py:1639, BEFORE BUE/CBT, so it is flag-off-identical and unaffected by boundary operators).
- **video-level**: max-score-per-video AUC over the Normal-vs-anomaly split.

This makes (a) and (b) measurable without perturbing the mAP objective. (c) is purely additive reporting.

## Common invariant
Each of (a)/(b) carries the sub-floor-only invariant as its neutrality proof; (c) is purely additive reporting. All three are the owner's separate test later — no code shipped this session.
