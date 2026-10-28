# v193 - UCA-TAL Supervised Temporal Localizer: Research NO-GO

Date: 2026-08-20. One proposal was reviewed; no production code was changed.

## Proposal And Dual-Angle Research

UCA-TAL was the sole proposal: train a compact anchor-free temporal localizer on frozen CLIP sequences using UCA timestamp annotations restricted to videos in the official UCF-Crime training split. It is outside the closed curve-rerank, proposal-append, scan/prompt/granularity, boundary-growth/relocation, whole-video-veto, and frozen VideoMAE MIL families because it learns candidate-local start/end and quality targets from independent dense supervision.

- Data angle: the local official train parquet has 1,610 unique videos and all temporal annotation fields are `-1`. No UCA annotation artifact exists in the workspace or dataset tree. Prior research records UCA as 1,854 train+test videos and notes that Kaggle authentication is unavailable.
- Architecture angle: frozen per-frame CLIP features could feed a small temporal transformer with center, duration, and IoU-quality heads. If approved, the contract would be default-OFF `uca_tal`, both CLIs and `from_args`, unconditional `[UCA-TAL-VID]`, material `[UCA-TAL-FIRE]`, and index-stable `pre_uca_tal_detections`.

## Fixed Train-Only Gate And Review

The pre-registered gate would require all conditions on grouped five-fold validation over the audited official-train intersection only: (1) 100% of annotation video IDs map to official train and 0 map to official test; (2) evaluator-faithful temporal mAP at IoU 0.3/0.5/0.7 improves by at least +0.030 over the frozen committed detector replay; (3) each hard category is non-negative; and (4) the 90% video-bootstrap probability of positive delta is at least 0.90. Test videos and test GT are forbidden from fitting, checkpoint selection, thresholds, or gate computation.

Skeptical review: **REJECT / Research NO-GO.** The required annotation file is absent, its train/test provenance cannot be audited, and the localized gate therefore cannot be executed. Substituting official video labels, synthetic boundaries, or current-pipeline pseudo-labels would repeat v191/v192's non-local evidence problem. No precheck result can honestly report GO, so no flag, model, backup of production files, or mechanism marker was introduced. Hard-88 was not armed.

## Mandatory Smoke

The committed baseline command in `results/batch_eval_v193_smoke/COMMAND.txt` completed 14/14 videos with 0 failed and 0 skipped. The top-level and inner logs each contain exactly 14 `[COST]` lines; `[UCA-TAL-VID]` and `[UCA-TAL-FIRE]` each occur 0 times, as expected for a rejected mechanism.

| mAP | AP@0.3 | AP@0.5 | AP@0.7 | Avg/video | Wall clock |
|---:|---:|---:|---:|---:|---:|
| 0.2222222222 | 0.3246753247 | 0.1904761905 | 0.1515151515 | 248.18s | 2203.42s |

Artifacts:

- `results/batch_eval_v193_smoke.log`
- `results/batch_eval_v193_smoke/run.log`
- `results/batch_eval_v193_smoke/COMMAND.txt`
- `results/batch_eval_v193_smoke/evaluation_report.json`
- 14 per-video JSON files in `results/batch_eval_v193_smoke/`

## Decision

**Research NO-GO; production remains unchanged.** UCA-supervised localization remains open only after an annotation artifact is acquired and its official-train-only provenance passes the fixed audit above. Do not approximate the missing timestamps with test GT or selectors derived from the current pipeline.
