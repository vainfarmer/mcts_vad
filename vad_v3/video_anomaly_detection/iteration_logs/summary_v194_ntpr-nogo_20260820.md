# v194 - NTPR Normal Temporal Prediction Residual: Research NO-GO

Date: 2026-08-20. One outside-family proposal was reviewed; no production code was changed.

## Proposal And Dual-Angle Research

NTPR was the sole proposal: learn normal temporal dynamics from the 800 official-train Normal videos, then use frozen-feature next-step prediction residual as an anomaly-localization curve. This is outside the closed curve-rerank, proposal-append, VLM scan/prompt/granularity, boundary-growth/relocation, whole-video-veto, CLIP/SigLIP curve, and frozen-VideoMAE MIL families because it learns a normal-only temporal generative objective rather than anomaly-bag discrimination or post-processing an existing curve.

- Data angle: the official train parquet has 1,610 rows (800 Normal, 810 anomaly), but all four temporal annotation columns are negative for every row. There are therefore zero train-localized examples on which to measure whether a high residual lands on a real crime interval.
- Architecture angle: a small causal predictor over frozen temporal features could emit a per-cell residual, threshold contiguous high-residual runs, and preserve the committed detector when disabled. If approved, the contract would be default-OFF `ntpr_localizer`, both CLIs plus `from_args`, unconditional `[NTPR-VID]`, material `[NTPR-FIRE]`, and index-stable `pre_ntpr_detections`.
- Literature angle: future-frame prediction establishes prediction error as an anomaly signal (Liu et al., CVPR 2018); MemAE warns that unconstrained reconstruction can also reproduce anomalies (Gong et al., ICCV 2019); MNAD uses normal-pattern memory and compactness/separateness losses (Park et al., CVPR 2020). These results support the family but do not supply a UCF-Crime train-only localization validator.

Sources: CVF Open Access pages for [Future Frame Prediction](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html), [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html), and [MNAD](https://openaccess.thecvf.com/content_CVPR_2020/html/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.html).

## Fixed Train-Only Gate And Review

The pre-registered gate required all conditions on official-train data only: (1) at least 50 independently annotated anomaly videos, including at least five per hard category, map 100% to official train and 0% to official test; (2) grouped five-fold evaluator-faithful temporal mAP improves by at least +0.030 over the frozen committed replay; (3) every hard category is non-negative; (4) video-bootstrap P(delta>0) is at least 0.90; and (5) held-out Normal false-positive-video rate is at most 10%. Test GT is forbidden from fitting, checkpoint selection, thresholds, or this gate.

Skeptical review: **REJECT / Research NO-GO.** Gate prerequisite (1) fails with zero localized train rows. Synthetic shuffles, splices, or dropped frames would create proxy frame labels forbidden by the workflow and could validate only corruption detection, not crime localization. Video-level anomaly labels cannot validate where the residual peaks, repeating the localization-identifiability problem closed in v192. No honest train-only localization verdict is available, so no precheck training, flag, backup, production edit, mechanism marker, paired field, or hard-88 run was armed.

## Mandatory Smoke

The unchanged committed baseline command in `results/batch_eval_v194_smoke/COMMAND.txt` completed 14/14 videos with 0 failed and 0 skipped. The top-level and inner logs each contain exactly 14 `[COST]` lines. `[NTPR-VID]` and `[NTPR-FIRE]` each occur 0 times, as required for a rejected mechanism.

| mAP | AP@0.3 | AP@0.5 | AP@0.7 | Avg/video | Wall clock |
|---:|---:|---:|---:|---:|---:|
| 0.2272727273 | 0.3333333333 | 0.1969696970 | 0.1515151515 | 169.01s | 1201.22s |

Artifacts:

- `results/batch_eval_v194_smoke.log`
- `results/batch_eval_v194_smoke/run.log`
- `results/batch_eval_v194_smoke/COMMAND.txt`
- `results/batch_eval_v194_smoke/evaluation_report.json`
- 14 per-video JSON files in `results/batch_eval_v194_smoke/`

## Decision

**Research NO-GO; production remains unchanged.** Normal-only temporal prediction remains scientifically open only after an independent official-train localization set can execute the fixed gate. Do not substitute synthetic corruption locations, video labels, test GT, or current-pipeline pseudo-labels.
