# Loop 7 (v92): Borderline-Case Majority-Vote Revoting
## Date: 2026-07-12

## Motivation (building on V90's failure analysis)
V90's median-of-3 VLM boundary self-consistency regressed mAP (0.1494->0.1075).
Root cause: self-consistency helps REASONING/CATEGORICAL tasks (multiple valid
paths converge on truth) but hurts PERCEPTUAL/EXTRACTION tasks (timestamp
reading has no "alternative valid path", just noise added via diversity).

## New Hypothesis
`scan_segment()`'s `is_anomalous` decision (binary: does this segment show
anomalous activity?) is a CATEGORICAL decision with genuine interpretive
ambiguity ("nervous behavior" could reasonably be judged either way) - the
task type where self-consistency SHOULD help per literature, unlike the
numerical timestamp extraction that failed in V90.

## Latency-Conscious Design
Given tripling ALL ~10-18 segment-scan VLM calls per video would meaningfully
hurt the "minimize latency" goal, only re-vote (2 extra calls, 3 total) for
BORDERLINE cases where confidence lands in [0.15, 0.45] - straddling the 0.3
accept threshold. Clear-cut cases (very confident yes/no) skip extra calls
entirely, keeping latency close to baseline for the majority of segments.

## Implementation
In `anomaly_discovery.py`'s `scan_segment()`: after the first VLM call, check
if `result['anomaly_score']` is in the borderline zone. If so, fire 2 more
calls via `asyncio.gather`, take majority vote (>=50%) for `is_suspicious`
and mean of all valid scores for `anomaly_score`.

## Next Steps
Smoke test first (given V86's lesson: smoke test success didn't predict full
eval outcome for Soft-NMS, so treat smoke test as a coarse crash-detector
only, not strong evidence of correctness). Full eval regardless of smoke
result, given resource investment already made and the theoretical
distinction from V90's failure mode.
