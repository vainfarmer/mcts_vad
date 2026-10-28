# Agent Team Orchestration — Live Run (starting v109)

## Source workflow: /Users/zzl/Documents/md/agent_team_loop_prompt_v108.md
## Started: 2026-07-22
## Baseline: V94 mAP=0.1492 (AP@0.3=0.2046, AP@0.5=0.1335, AP@0.7=0.1096)
## Live pipeline header: v105 (TSG reverted; byte-equivalent to v94 behavior)
## Target: mAP >= 0.20; secondary: minimize latency
## Min iterations: 10 loops (v109 .. v118), autonomous

## Environment (verified)
- Interpreter: `python3.14` (has numpy/torch/open_clip; plain `python3` does NOT)
- API key: `source video_anomaly_detection/.env` exports DASHSCOPE_API_KEY (never hardcode)
- Dataset: /Users/zzl/Movies/ucf-crime
- Smoke: `python3.14 -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir results/batch_eval_vXXX_smoke --max_per_category 1 --concurrency 3`
- Full:  same with `--output_dir results/batch_eval_vXXX --max_per_category 5 --concurrency 3`

## Role mapping in this live run
- A, B = parallel research subagents (independent angles) → converge to 1 proposal
- C = orchestrator applies FM1-FM6 gate + realism/latency/feasibility
- D = orchestrator implements, runs smoke→full, saves artifacts

## 6 Failure Modes (hard gate — any hit needs extraordinary justification)
- FM1 curve-shape modification (fused CLIP+MIL curve for half-max) — 10+ fails
- FM2 detection-count change under greedy single-match mAP — Soft-NMS fail
- FM3 self-consistency for numerical/perceptual extraction (timestamp) — noise
- FM4 self-consistency changing candidate/segment INCLUSION — = FM1 risk
- FM5 flag never actually enabled (mechanism never ran) — require [GATE] marker
- FM6 simulation gains don't transfer to live — require in-vivo only

## Safe patterns (confirmed non-regressing)
- Strictly UPSTREAM of fusion/boundary/candidate logic (V94 learned prompt wts)
- PURE post-hoc confidence re-ranking (never touch t_start/t_end or count) (V89)

## Loop ledger
| loop | ver | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | decision |
|---|---|---|---|---|---|---|---|
| 1 | v109 | MIL cross-video confidence prior | — | 0.15412 (rerank of v108) | 0.14673 | 0.11448 | REJECT@precheck (-0.0042; video-level prior can't fix intra-video mislocation) |
| 2 | v110 | TVR: per-detection VideoMAE-over-support re-rank | — | — | — | — | REJECT@precheck (VideoMAE window signal saturated ~1.0 everywhere incl. normals; no intra-video/normal-anomaly discrimination) |
| 3 | v111 | Localization-quality Q-score re-rank | — | 0.16703 (det. rerank of v108, +0.0087) | 0.15781 | 0.11636 | **KEEP** (broad-based gain across all IoU; helper reproduces +0.0087 exactly; FM1/FM2 immune) |
| 4 | v112 | Stack duration-prior fit onto Q-score | — | 0.16703 (Q4 neutral) | 0.15781 | 0.11636 | REJECT (dur-fit already baked into pipeline boundary priors → non-discriminative; separate mult regresses -0.008) |

## Loop 1 (v109) outcome
A's MIL prior REJECTED by offline pre-check (faithful re-score of v108 real detections; baseline reproduced 0.15834 exactly). Regressed -0.0042. Lesson: re-rank signal must be PER-DETECTION, not per-video. Pipeline untouched (byte-identical). Reusable asset: `analysis_v109_milprior_precheck.py` offline re-score harness.
