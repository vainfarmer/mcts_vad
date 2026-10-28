# v191 - VTL (VideoMAE Temporal Learner): NO-GO

Date: 2026-08-20. Train-only precheck and mandatory committed-baseline smoke completed. No production code was changed.

## Prior Version

v190 completed the unfinished VMC sliding-window VideoMAE class-probability-curve precheck and closed NO-GO: only 11/54 missed GTs reached own-category z >= 2 (20.37% < 40%), median own-category z was 0.6823 < 1.0, and the worse non-GT guard p90 was 12.177 > 4.0. Artifacts: `results/v190_vmc_precheck.json`, `results/v190_vmc_precheck.out`, and `iteration_logs/analysis_v190_vmc_precheck.py`.

## VTL Precheck

VTL used the official train split only (1,610 unique videos: 810 anomaly, 800 Normal), with no temporal GT and no test data or GT. Frozen `OPear/videomae-large-finetuned-UCF-Crime` hidden features formed an eight-cell temporal grid; a lightweight depthwise causal-convolution head used top-2 MIL with category-stratified, video-grouped five-fold OOF evaluation and train-only inner-fold thresholds.

The aggregate learned head was strong, but the causal temporal context itself was not material:

- OOF AUC: 0.9540416667, PASS >= 0.90.
- Raw frozen-head AUC: 0.6703302469; gain: +0.2837114198, PASS >= +0.05.
- Pooled Normal FPR: 0.1775, PASS <= 0.20.
- Hard-category pooled recall: 0.9009433962, PASS >= 0.70; each hard category also passed >= 0.50.
- Source top-2 changed: 1,521/1,610 = 0.9447204969, PASS >= 0.10, with the eight-cell/top-2 call budget invariant.
- No-context ablation AUC: 0.9495185185; **causal temporal context AUC gain: 0.0045231481 < 0.15, FAIL**.

Because every gate was conjunctive, the context-gain failure makes the precheck NO-GO. No VTL flag, marker, model, or production-path change was introduced; hard-88 and bootstrap adjudication were not armed.

Artifacts: `iteration_logs/analysis_v191_vtl_precheck.py`, `results/v191_vtl_precheck/precheck_result.json`, `results/v191_vtl_precheck/precheck.log`, `results/v191_vtl_precheck/oof_predictions.csv`, and `results/v191_vtl_precheck/manifest.json`.

## Mandatory Smoke

The required flag-off committed baseline ran from `results/batch_eval_v191_smoke/COMMAND.txt` and completed 14/14 videos with 0 failed and 0 skipped. The top-level `results/batch_eval_v191_smoke.log` contains exactly 14 `[COST]` lines. Reported temporal metrics from `results/batch_eval_v191_smoke/evaluation_report.json`:

| mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---:|---:|---:|---:|
| 0.2222222222 | 0.3246753247 | 0.1904761905 | 0.1515151515 |

Mandatory artifacts:

- `results/batch_eval_v191_smoke.log`
- `results/batch_eval_v191_smoke/COMMAND.txt`
- `results/batch_eval_v191_smoke/evaluation_report.json`

## Decision And Closure

**NO-GO; production remains unchanged.** v191 closes the tested VTL scope: frozen VideoMAE final hidden tokens reduced to eight uniform temporal cells, spatially averaged, then learned with the registered 1024-to-64 projection, depthwise causal Conv1d(k=3), and top-2 MIL objective under video-level train labels. Its apparent advantage is almost entirely present without lagged context, so this design does not establish the required temporal-context effect.

Next versions must not re-propose this same eight-cell frozen-VideoMAE causal-convolution/top-2-MIL formulation or claim its aggregate AUC gain as temporal-context evidence. A materially different temporal representation, objective, or causal-context test remains outside this closure if it has a new pre-registered train-only gate.
