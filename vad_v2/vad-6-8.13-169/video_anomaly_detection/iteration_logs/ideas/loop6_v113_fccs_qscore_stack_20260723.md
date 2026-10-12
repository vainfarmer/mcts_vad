# Loop 6 (v113) — FCCS + Q-score Stacking: CATASTROPHIC REGRESSION

## Configuration
--full_coverage_scan --qscore_rerank (both flags ON)

## Result
mAP=0.0524 (vs v97 baseline 0.1615) — WORST result since v103's crash.
AP@0.3=0.0815, AP@0.5=0.0477, AP@0.7=0.0280

## Root Cause Analysis
The Q-score reranking function (_qscore_rerank) uses THREE signals:
boundary_sharpness, peak_contrast, center_fit (equal weight, averaged).

The OFFLINE precheck on v97's CACHED detections showed:
- all-equal (3 signals): +0.005 mAP
- contrast+center only: -0.010 mAP  
- sharp+contrast only: +0.004 mAP

But the LIVE full eval showed -0.109 mAP (catastrophic).

This is FM6 (simulation-to-live gap) at its WORST — the offline precheck
on cached detections gave a +0.005 signal, but on FRESH stochastic VLM
detections the same Q-score mechanism destroyed mAP. The reason: FCCS
produces a DIFFERENT distribution of detection geometries than the standard
scan (more candidates, different boundary positions), and the Q-score
signals (especially center_fit) interact badly with FCCS's specific
candidate geometry.

center_fit assumes anomaly peaks should be centered in the detection
interval, but FCCS's comparative scoring can produce legitimately
asymmetric detections where the VLM flagged an onset/offset region rather
than a centered peak. Re-ranking these as "low quality" demotes true
positives.

## Decision: REVERT Q-score entirely. v97 (FCCS only) remains the best.

## New Failure Mode
FM8: Offline precheck on one scan mode's detections does NOT generalize to
a different scan mode's detections. Q-score was validated on v108 (standard
scan) detections and showed +0.009; it was also validated on v97 (FCCS)
cached detections and showed +0.005; but the LIVE FCCS detections have
different statistical properties than either cache, and the gain vanished.
The precheck harness is only valid for the EXACT same pipeline configuration
that produced the cached detections.
