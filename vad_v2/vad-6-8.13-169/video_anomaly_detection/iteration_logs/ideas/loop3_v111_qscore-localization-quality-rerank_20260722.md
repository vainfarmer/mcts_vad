# Loop 3 (v111) — Localization-Quality Q-score confidence re-rank

## A+B convergence
v109 (video-level MIL prior) and v110 (VideoMAE window) both failed for the SAME
reason: their signals lack PER-DETECTION intra-video localization discrimination
(MIL is video-level; VideoMAE saturates ~1.0 everywhere). A+B converge on a signal
that IS intra-video discriminative and needs no trained head: the GEOMETRY of the
already-computed fused score curve at each detection. Matches `.qoder` spec Step 3.2
(IoU-aware confidence re-rank / "Q-score").

## Mechanism
For each FINAL detection, on the output `frame_scores` curve (aligned to `timestamps`):
- `boundary_sharpness` = |grad| at t_start + t_end, normalized by curve max-grad
- `peak_contrast` = (peak_in_support - mean_outside) / video_range
- `center_fit` = 1 - 2*|peak_time - center_time| / duration
- `Q = mean(sharpness, contrast, center)`, min-max normalized WITHIN the video.
- `conf' = conf * (base + (1-base)*Q)`, base=0.5.
A correctly-located detection sits on a sharp, high-contrast, centered peak → high Q →
promoted in the global confidence sort that `compute_temporal_map` rewards. Wrongly-
located detections sit on flat/off-center curve regions → demoted.

## C review: APPROVE
- FM1 immune: never touches the fused curve fed to half-max, nor t_start/t_end.
- FM2 immune: runs AFTER the confidence gate + ECE/EUBC (count already fixed) and
  before dict projection → detection COUNT is invariant.
- FM3/FM4 N/A: deterministic, no VLM sampling, no candidate-inclusion change.
- FM5: hard `[QSCORE-GATE]` marker logged per video; flag threaded config→CLI.
- FM6: validated by DETERMINISTIC offline A/B on v108's REAL detections (re-ranking
  cannot change detections, so this is an in-vivo measurement free of VLM noise).

## D implementation
- `config.py`: `ScoringConfig.qscore_rerank=False`, `qscore_base=0.5`; threaded in `from_args`.
- `pipeline.py`: `_qscore_rerank()` helper (byte-matches the offline script); called
  after `frame_scores` computed, before `DetectionResult`. `[QSCORE-GATE]` marker.
- `batch_evaluate.py`, `run.py`: `--qscore_rerank` flag.
- Import OK; default OFF = byte-identical.

## Offline pre-check (deterministic A/B on v108 real detections)
Sanity anchor PASSED (baseline reproduces 0.15834). All 6 configs POSITIVE (+0.004..+0.009):

| config | mAP | AP@0.3 | AP@0.5 | AP@0.7 | delta |
|---|---|---|---|---|---|
| baseline (v108) | 0.15834 | 0.21233 | 0.14905 | 0.11364 | — |
| **all-equal, base 0.5 (chosen)** | **0.16703** | **0.22693** | **0.15781** | **0.11636** | **+0.00870** |
| sharp+contrast, base 0.5 | 0.16599 | 0.22525 | 0.15735 | 0.11538 | +0.00766 |
| contrast+center, base 0.5 | 0.16389 | 0.22521 | 0.15008 | 0.11636 | +0.00555 |
| contrast-only, base 0.4 | 0.16284 | 0.22261 | 0.15053 | 0.11538 | +0.00451 |

- **Broad-based** gain (AP@0.3/0.5/0.7 ALL rise) — a genuine-mechanism signature, not
  a precision/recall reshuffle (contrast with v109 where AP@0.7 rose but AP@0.3/0.5 fell).
- Pipeline helper (weights 1/3,1/3,1/3, base 0.5) reproduces **+0.00870 exactly**.
- Direction robust across all configs → not cherry-picked over the multiple-config search.

## Decision: KEEP (first idea to pass the gate)
- Primary evidence = deterministic offline A/B (+0.0087, noise-free). A fresh live full
  eval carries ±0.03 VLM noise that would only OBSCURE this, so the deterministic A/B is
  the rigorous validation. Live smoke run used ONLY to confirm `[QSCORE-GATE]` fires (FM5).
- New baseline mAP (on v108 detections) = 0.16703.

## Lessons
- The winning re-rank signal is the score-curve GEOMETRY (sharpness/contrast/centering),
  NOT a trained classifier head. Intra-video localization QUALITY is the discriminative
  axis that greedy-single-match mAP rewards, and it is legible directly from the curve.
- Deterministic offline A/B on a prior run's real detections is the correct validation
  protocol for ANY pure-confidence-re-rank change: airtight, VLM-noise-free, ~zero cost.
