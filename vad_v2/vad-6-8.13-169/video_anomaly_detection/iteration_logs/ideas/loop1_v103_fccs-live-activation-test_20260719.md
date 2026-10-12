# v103 — FCCS Live Activation Test (v97 mechanism, first in-vivo evaluation)

## Approved Proposal

- **Title:** FCCS Live Activation Test (v97 mechanism, first in-vivo evaluation)
- **Mechanism:** Full-Coverage Comparative VLM Scanning (already implemented in the tree behind `--full_coverage_scan`): all 6s segments packed into contiguous batches of 8 (16 for >60-segment videos), comparative 0–1 rating against in-video normal baseline, 0.3 threshold, `_merge_adjacent`, unchanged downstream (top_k=3, v51 half-max, v95/v96 modulators, hard NMS).
- **Rationale:** C-approved in v97 with independent verification (dead-but-intact v56/v57 comparative scan code resurrected + truncation guard; v56 isolated comparative judging as mAP-positive +0.006; coverage traces prove wrong-location-by-construction). The v97 eval never activated it (FM5) — this loop runs the actual experiment.
- **Implementation plan:** Verify-only: confirm `--full_coverage_scan` exists in batch_evaluate.py (verified by orchestrator: batch_evaluate.py:189, config.py:86+168), confirm `[FCCS]` INFO markers in `anomaly_discovery.scan_full_coverage`, then evaluate with the flag ON.
- **Expected gain:** +0.01..0.03 (floor band per v97 analysis) if comparative calibration helps; true coverage recovery was the open question.
- **Risks:** FM2-category risk bounded by unchanged top_k/merge/NMS; FM1 untouched; FM3/FM4 absent by construction; FM5 eliminated by this loop's activation gate.
- **Latency impact:** v97 flag-off run measured 141.47s; live run expected similar (+16 images/call but 3–4x fewer calls).
- **Literature:** TimeChat/VTimeLLM (CVPR24), Holmes-VAU (CVPR25), TSPO (AAAI26), T3AL (CVPR24), LLM-as-a-judge calibration (Zheng 2023); see v97 idea file.

## C's Feedback

Pre-approved as v97 (C verdict, previous run). This loop only activates and measures it.

## Implementation Notes (D1)

Files changed: none — verify only. v97 FCCS already implemented and verified present in the tree; evaluated with `--full_coverage_scan`.

## Eval Flags & Activation Marker

- Flags: `--full_coverage_scan`
- Activation marker: `[FCCS]`
- Marker counts: smoke log = 46; full log = 176 (both confirm full_coverage_scan was live during inference).

## Results

| Metric | Smoke | Full |
|---|---|---|
| mAP | 0.19390878481787574 | 0.05861589530449584 |
| AP@0.3 | — | 0.11406317412897572 |
| AP@0.5 | — | 0.039057239057239054 |
| AP@0.7 | — | 0.022727272727272728 |
| Avg latency (s) | — | 191 |

Smoke note: AP@0.7 = 0.0 in smoke. Full-eval report required a ~6 min post-run wait before appearing.

## Decision: REVERT

**Reason:** Full-eval mAP of 0.0586 is far below the baseline band, and AP@0.7 collapsed to 0.0227 (smoke AP@0.7 = 0.0). Despite correct activation (46/176 `[FCCS]` markers, FM5 eliminated), full-coverage comparative scanning degraded precision at every IoU tier instead of delivering the expected +0.01..0.03 gain. Latency also rose to 191s vs the 141.47s flag-off reference. Verify-only loop: no backup directory existed for v103, so no code restore was required — the tree remains untouched.

## Lessons Learned

- **FM5 refuted for good:** the activation gate worked — `[FCCS]` markers confirm FCCS ran live in both smoke (46x) and full (176x) evals, so the v97 no-op outcome is not repeated; this is a genuine negative result for the mechanism, not an instrumentation artifact.
- **v56's +0.006 comparative-judging gain did not transfer to full-coverage packing:** batching all 6s segments (8 or 16 per call) with an in-video normal baseline appears to dilute anomaly saliency — comparative calibration against "normal" peers within the same video suppresses exactly the weak-signal anomalies the pipeline needs, collapsing high-IoU precision (AP@0.7 ≈ 0).
- **FM2-category risk was not bounded by unchanged top_k/merge/NMS:** the failure entered upstream at the scoring stage (0.3 threshold over comparative 0–1 ratings), so downstream invariants could not protect mAP — future proposals must treat score-distribution shifts as first-class risk even when merge/NMS are frozen.
- **Coverage recovery hypothesis is dead in this form:** wrong-location-by-construction is not fixed by scanning everything comparatively; the cost is +50s latency and a ~3x mAP regression.
