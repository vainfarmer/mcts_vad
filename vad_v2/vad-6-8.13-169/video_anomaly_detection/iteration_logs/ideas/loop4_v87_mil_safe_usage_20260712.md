# Loop 4 (v87): Safe MIL v6 Usage Patterns - Research + Empirical Rejection
## Date: 2026-07-12

## Research Findings (5 ranked ideas, all claimed "decoupled from curve-shape")
1. Video-level pre-filtering / early-exit cascade (Cerberus 2025, cascading
   multi-agent VAD) - claimed "provably mAP-neutral" since it's a binary
   process/skip gate before the expensive CLIP+VLM pipeline runs.
2. Post-hoc confidence adjustment (RPC 2025) - MIL score within an ALREADY-
   fixed detection window, adjusts confidence only, not boundaries.
3. MIL as VLM classification context (narrow, safe, low impact)
4. Gated Mixture-of-Experts / per-video routing (GS-MoE, GCR) - highest
   potential but highest complexity, deferred
5. Confidence calibration via auxiliary MIL signal - needs calibration data

## CRITICAL EMPIRICAL REJECTION of Idea 1 (early-exit)
Before implementing, checked MIL v1's ACTUAL per-video max-score distribution
on the UCF-Crime test set (140 abnormal + 150 normal videos):

  Abnormal videos max-score: p1=0.0030, p5=0.1971, median=1.0000
  Normal videos max-score:   p95=0.9071, p99=0.9992, median=0.0012

Despite AUC=0.977 (aggregate), the distributions have a MEANINGFUL OVERLAP:
5% of genuinely ABNORMAL videos have max score < 0.197 (long left tail -
these are the HARD cases: weak CLIP/visual signal, subtle anomalies).

Threshold sweep (normal videos correctly skipped vs abnormal videos WRONGLY
skipped / false-negatived):
  threshold=0.05: 85% normal skipped, but 2.9% of ABNORMAL VIDEOS MISSED ENTIRELY
  threshold=0.10: 89% normal skipped, but 4.3% of ABNORMAL VIDEOS MISSED
  threshold=0.15: 89% normal skipped, but 5.0% of ABNORMAL VIDEOS MISSED

A missed video = 0 recall for that video = directly hurts mAP (worse than
any boundary imprecision, since it's a complete miss). Given current recall
is already 92-98%, ADDING 3-5% more false negatives via pre-filtering is a
real risk to mAP, contradicting the literature's "provably neutral" framing
(which likely assumed better score separation than this specific MIL model +
dataset combination actually has).

## Decision: REJECT hard early-exit filtering
The theoretical safety argument ("it's just a binary gate, doesn't touch
scoring for videos that proceed") is correct in principle, but the PRACTICAL
false-negative rate on THIS dataset with THIS model is too high to accept.
This is exactly the kind of "sounds safe in theory, verify empirically before
implementing" lesson this project has learned repeatedly.

## Note: existing VideoMAE-based early-exit (already in V51 baseline) is
DIFFERENT and already proven safe - it requires  BOTH category=='Normal' AND
confidence>50% from a DIFFERENT, independently-calibrated classifier. Adding
a SECOND, less-reliable gate (MIL) on top would only ADD risk, not reduce it,
since gates in series only need ONE to misfire for a false negative.

## Pivot
Redirect Loop 4-5 budget to Loop 6's original latency plan (safer): optimize
the CLIP encoding step itself (already identified as the dominant per-video
cost in prior session's profiling: ~13 fps on MPS, compute-bound) without
changing WHICH videos/frames get processed or excluding any video from full
processing.
