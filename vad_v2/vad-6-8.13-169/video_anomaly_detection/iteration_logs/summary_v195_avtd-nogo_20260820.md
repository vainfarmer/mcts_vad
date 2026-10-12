# v195 - AVTD Audio-Visual Temporal Discordance: Research NO-GO

Date: 2026-08-20. One outside-family proposal was reviewed; no production code was changed.

## Proposal And Dual-Angle Research

AVTD was the sole proposal: learn audio-visual correspondence on official-train Normal videos and use local synchronization discordance as a temporal anomaly curve. This is outside the closed visual curve transforms/backbones, VLM presentation, boundary/relocation, whole-video veto, frozen-VideoMAE MIL, candidate-quality MIL, supervised-localizer, and visual prediction-residual families because it introduces an audio modality and a cross-modal synchronization objective.

- Data angle: `ffprobe` confirms AAC stereo audio in sampled `Shooting001_x264.mp4`, so the local corpus can contain usable audio. The authoritative official train split nevertheless has 1,610 video-labeled rows and zero temporal annotations; no independent train-localized annotation artifact is available.
- Architecture angle: frozen audio and visual encoders could score fixed temporal cells by correspondence error and propose contiguous high-error runs. If approved, the contract would be default-OFF `avtd_localizer`, both CLIs plus `from_args`, unconditional `[AVTD-VID]`, material `[AVTD-FIRE]`, and index-stable `pre_avtd_detections`.
- Literature angle: audio-visual correspondence can learn self-supervised representations (Arandjelovic and Zisserman, ICCV 2017), AVSlowFast demonstrates integrated audio-visual temporal modeling (Xiao et al., 2020), and audio-visual anomaly scoring has localized cross-modal inconsistencies in video forensics (Feng et al., CVPR 2023). None establishes UCF-Crime crime-interval localization from video labels alone.

Sources: [Look, Listen and Learn](https://arxiv.org/abs/1705.08168), [Audiovisual SlowFast Networks](https://arxiv.org/abs/2001.08740), and [Self-Supervised Video Forensics by Audio-Visual Anomaly Detection](https://arxiv.org/abs/2301.01767).

## Fixed Train-Only Gate And Review

The pre-registered gate required all conditions using official train only: (1) at least 50 independently localized anomaly videos, including at least five per hard category, map 100% to official train and 0% to test; (2) at least 90% of those videos and held-out Normal controls have valid continuous audio; (3) grouped five-fold evaluator-faithful temporal mAP improves by at least +0.030 over frozen committed replay; (4) every hard category is non-negative; (5) video-bootstrap P(delta>0) is at least 0.90; and (6) held-out Normal false-positive-video rate is at most 10%. Test videos and test GT are forbidden from fitting, thresholding, checkpoint selection, or gate computation.

Skeptical review: **REJECT / Research NO-GO.** Gate prerequisite (1) fails because official train has zero localized rows and no independent localization artifact. Audio presence in a sample proves input feasibility, not crime-interval localization. Offset-shift corruption would create synchronization labels but validate edit detection rather than real crime localization; video labels or current-pipeline pseudo-labels would repeat the localization-identifiability and circularity failures closed in v192-v194. No precheck fit, production implementation, backup, flag, paired field, or hard-88 was armed.

## Mandatory Smoke

The unchanged committed baseline command in `results/batch_eval_v195_smoke/COMMAND.txt` completed 14/14 videos with 0 failed and 0 skipped. There are 14 per-video JSON files. The top-level and inner logs each contain exactly 14 `[COST]` lines; `[AVTD-VID]` and `[AVTD-FIRE]` each occur 0 times, as required for a rejected mechanism.

| mAP | AP@0.3 | AP@0.5 | AP@0.7 | Avg/video | Wall clock |
|---:|---:|---:|---:|---:|---:|
| 0.2272727273 | 0.3333333333 | 0.1969696970 | 0.1515151515 | 169.10s | 1200.65s |

Artifacts:

- `results/batch_eval_v195_smoke.log`
- `results/batch_eval_v195_smoke/run.log`
- `results/batch_eval_v195_smoke/COMMAND.txt`
- `results/batch_eval_v195_smoke/evaluation_report.json`
- 14 per-video JSON files in `results/batch_eval_v195_smoke/`

## Decision

**Research NO-GO; production remains unchanged.** AVTD remains open only after independently localized official-train examples and corpus-wide audio coverage permit the fixed gate. Do not substitute synthetic offsets, test GT, video-label-only discrimination, or current-pipeline pseudo-labels.
