# v192 - CQM (Candidate-Quality MIL): NO-GO

Date: 2026-08-20. Research review and mandatory committed-baseline smoke completed. No production code was changed.

## Proposal

CQM was the strongest score-quality proposal outside the closed curve-rerank, whole-video-veto, relocation, scan-prompt/granularity, and raw-backbone-curve families. It would pool the existing frozen 32x768 train CLIP cells over candidate windows, train normal-video windows as negatives, and train anomaly videos with a noisy-OR/top-k bag objective. At inference it would calibrate each final detection independently from its pooled embedding rather than normalize or rank detections within a video.

If a valid train-only gate existed, the exact production contract would be:

- Default-OFF flag `cqm_score`, wired through `config.py`, `run.py`, and `batch_evaluate.py`.
- Candidate scoring in the terminal confidence slot in `pipeline.py`, with no count or boundary changes.
- Unconditional `[CQM-VID]` once per video and `[CQM-FIRE]` when confidence materially changes.
- Index-stable `pre_cqm_confidence` on every changed detection for paired bootstrap restore.
- Frozen train-produced model artifact loaded only when the flag is enabled.

## Review And NO-GO

The proposal has no defensible train-video-label-only localization gate. The official train split supplies 1,610 video labels but no temporal labels. Therefore it identifies positive bags, not positive candidate windows. OOF bag AUC, hard-category video recall, normal-window suppression, augmentation stability, and changed-top-k rates can all pass while the model consistently selects background from anomaly videos.

This is not a hypothetical concern:

- v191 obtained OOF video AUC 0.9540, but its causal temporal-context gain was only +0.00452 against the fixed +0.15 gate.
- v190 VideoMAE localization reached only 11/54 missed intervals (20.37%) at own-category z >= 2, with median z 0.6823 and a failed non-GT guard p90 of 12.177.
- v183's train-supervised MIL confidence scaler regressed hard-88 by -0.01253, with all four hard categories negative.
- Earlier realizable MIL quality priors regressed despite a small cross-video score oracle, and their score-quality rank correlation was weak.

Using test temporal GT to select a CQM checkpoint or threshold would leak verdict data. Inventing train window positives from the current MIL/CLIP/VLM selectors would be circular pseudo-labeling and would repeat signals already shown to fail on brief hard events. Consequently no numerical train-only GO gate can measure the claimed candidate-local quality property, so the mechanism was rejected before implementation. `eval_flags=""`, marker=`[COST]`; hard-88 and bootstrap were not armed.

## Mandatory Smoke

The required flag-off committed baseline ran from `results/batch_eval_v192_smoke/COMMAND.txt`. It completed 14/14 videos with 0 failed and 0 skipped. Both the top-level log and inner `run.log` contain exactly 14 `[COST]` lines.

| mAP | AP@0.3 | AP@0.5 | AP@0.7 | Avg/video |
|---:|---:|---:|---:|---:|
| 0.2272727273 | 0.3333333333 | 0.1969696970 | 0.1515151515 | 108.69s |

Mandatory artifacts:

- `results/batch_eval_v192_smoke.log`
- `results/batch_eval_v192_smoke/COMMAND.txt`
- `results/batch_eval_v192_smoke/evaluation_report.json`

## Decision And Closure

**NO-GO; production remains unchanged.** Do not re-propose candidate/window quality learning from video bag labels alone with MIL-selected positives or selectors derived from the current frozen signals. This direction can reopen only with an independent train-only temporal supervision source or a proxy whose candidate-local correlation is directly measurable without test GT. A materially different self-supervised temporal objective may remain open, but video-level discrimination is not sufficient evidence for localization.
