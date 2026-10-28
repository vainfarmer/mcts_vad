# Loop 1 (v109) — MIL Cross-Video Confidence Prior

## A's idea
Cross-Video MIL Confidence Prior: `conf' = conf*(0.7+0.3*mil_prior)`, mil_prior = top-k mean of the video's MIL snippet scores. Exploits the metric fact that `evaluate.compute_temporal_map` pools ALL detections into one GLOBAL confidence-sorted list (greedy single-match), yet no current confidence term is cross-video-calibrated. MIL is trained for exactly video-level cross-video discrimination.

## B's idea (deferred to v110)
Trained-VideoMAE Confidence Re-ranking (TVR): use the VideoMAE sliding-window score sampled over EACH detection's support `[t_s,t_e]` — a per-detection, within-video signal. Distinguishes correctly- vs wrongly-located detections inside the same video.

## A+B debate → C convergence
Both target the same cross-video-sort insight and both fit the confirmed-safe PURE POST-HOC CONFIDENCE RE-RANKING pattern (V89/V95 lineage). C selected A's as v109 (single per-video scalar → provably preserves within-video order, minimal code, exact-axis). Both admit a cheap OFFLINE pre-check: re-scoring v108's REAL saved detections is faithful in-vivo (re-ranking cannot change boundaries/count) → FM6-safe, ~zero cost, no VLM.

## C review: APPROVE-with-mandatory-offline-precheck
- FM1 avoided (never touches fused curve / t_start / t_end). FM2 avoided (count identical). FM3/FM4 N/A (deterministic, no VLM sampling, no inclusion change). FM5 addressed (would need [MILCONF-GATE] IF wired). FM6 addressed (offline re-score of real detections).
- Mandatory gate: offline pre-check must show gain BEFORE wiring the flag + spending a full eval.

## D implementation + result
- Built `iteration_logs/analysis_v109_milprior_precheck.py`: loads MIL (model_mil.pth), computes per-video prior from cached `features/test/*.npy` (32×768, exactly MIL's input), re-scores v108 detections, re-runs `evaluate.evaluate`.
- **Sanity anchor PASSED**: baseline re-score reproduces v108 mAP=0.15834 exactly.
- Prior separation excellent: anomaly mean=0.956 / median=1.000; normal mean=0.076 / median=0.000.

| formula (base+..*prior) | mAP | AP@0.3 | AP@0.5 | AP@0.7 | delta |
|---|---|---|---|---|---|
| baseline (v108) | 0.15834 | 0.21233 | 0.14905 | 0.11364 | — |
| 0.8 + 0.2·p | 0.15476 | 0.20294 | 0.14687 | 0.11448 | -0.00357 |
| 0.7 + 0.3·p | 0.15412 | 0.20115 | 0.14673 | 0.11448 | -0.00421 |
| 0.6 + 0.4·p | 0.15282 | 0.19991 | 0.14408 | 0.11448 | -0.00551 |
| 0.5 + 0.5·p | 0.15042 | 0.19712 | 0.13966 | 0.11448 | -0.00792 |

## Decision: REJECT at offline gate (no full eval spent, pipeline byte-identical)

## Lessons
- A VIDEO-LEVEL prior cannot help the DOMINANT error (wrong-location detections WITHIN anomaly videos — they all share the same high prior). It only demotes cross-video, but normal-video FPs already rank low, so no envelope gain; meanwhile it HURTS the ~few true anomalies with low MIL (collapse tail), netting negative. AP@0.7 rose marginally (+0.0008) while AP@0.3/0.5 fell — a location-precision reshuffle, not a gain.
- New generalizable lesson (FM7 candidate): **cross-video confidence calibration via a video-level prior does not improve greedy-single-match mAP when the dominant error is intra-video mislocalization.** The re-ranking signal must be PER-DETECTION (score WHERE the detection sits), not per-video.
- The offline re-score harness (faithful, ~0 cost, no VLM) is the key reusable asset — reuse for all future re-ranking loops. It reproduced the exact baseline, so its verdicts are trustworthy.
- Next (v110): test B's PER-DETECTION VideoMAE-over-support signal offline, which directly targets intra-video location.
