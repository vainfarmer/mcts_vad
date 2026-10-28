# v183 — CLC (Cross-video confidence Level Calibration via supervised MIL likelihood): NO-GO at Phase-0 offline precheck (direction CLOSED, zero eval spend)

Date: 2026-08-18. Precheck agent run (17 tool calls, ~8 min). NO batch_evaluate launched, NO VLM calls, NO pipeline code touched. Artifacts: iteration_logs/analysis_v183_clc_precheck.py, results/v183_clc_precheck.json, results/v183_clc_precheck.out.

## 1. What was proposed

The last un-attacked lead from the v11 audit: the ranking diagnostic measured a **+0.05180 cross-video-level oracle** (set each video's confidence LEVEL right, keep intra-video order). CLC = scale every detection confidence of video i by a supervised video-level anomaly likelihood from the pipeline's own MIL signal:

- s_i = mean of top-3 of the 32 MIL snippet scores (model_mil.pth)
- p_i = sigmoid(a*s_i + b), (a,b) fitted by logistic regression on the TRAIN split ONLY (filename label: contains 'Normal' -> 0 else 1)
- conf' = conf * (0.5 + 0.5*p_i) — count-frozen, window-frozen, preserves intra-video order, is NOT per-video normalization (banned 4 destroys level; CLC sets cross-video levels from an absolute supervised signal) and NOT curve-derived within-video re-rank (banned 1).

## 2. Phase-0 offline precheck (the whole loop — NO-GO, so no eval)

- Baseline results dir: results/batch_eval_v160_full (committed BUE+CBT+CBT-recal full-290). **Validity**: recomputed baseline mAP 0.17254429 == report 0.17254429 exactly; 290/290 videos matched; evaluate.py real metric functions reused (IoU [0.3,0.5,0.7], 11-point AP, global conf ranking, per-video greedy match).
- Train fit: a=8.9097, b=-4.5756, train AUC = 1.0000 (perfect separation — expected, model_mil.pth was trained on this same train split; sanity-only, in-distribution caveat).
- Test p_i spread: min 0.010 / mean 0.484 / max 0.987 — the signal is diffuse, not degenerate.

## 3. Pre-registered GO gate — RESULT

Gate: full-290 delta >= +0.005 AND hard-88 delta >= +0.008.

| slice | baseline | CLC | delta | verdict |
|---|---|---|---|---|
| full-290 | 0.172544 | 0.175394 | **+0.00285** | below +0.005 gate — FAIL |
| hard-88 | 0.118060 | 0.105525 | **-0.01253** | NEGATIVE — FAIL |

Per-category (all 4 hard cats negative — uniformly harmful there, not concentrated): Shooting -0.00132, RoadAccidents -0.02147, Shoplifting -0.00425, Explosion -0.00999.
Normal-video FA unchanged by construction (conf stays >0): 77/150 = 0.5133 both arms (verified).

=> **NO-GO**. Zero eval spend, zero code.

## 4. Why it fails (the lesson)

The supervised MIL likelihood carries a weakly-useful GLOBAL level signal (+0.0029 on 290) but it **mis-ranks exactly the hard categories**: its P(anomaly) is lower for the brief-event hard-cat videos (RoadAccidents/Explosion GT 1-3s inside long normal footage — the same short-clip/MIL-blindness measured in v180) than for Normal/other-category videos with spurious motion. Scaling confidences by it pushes true hard-cat detections below Normal-video FPs in the global ranking. The +0.0518 oracle exists, but the MIL video-level signal does not point at it.

## 5. CLOSED directions (do not re-propose)

- **Supervised MIL video-level likelihood as a cross-video confidence-level scaler** — measured offline, hard-88 -0.01253. This includes the model_mil_v6 swap variant: the failure is structural (MIL is blind on exactly the brief-event hard cats — v180 showed 3/5 harm videos have NO valid MIL scores at all), not a checkpoint-quality issue; a v6 variant would need to argue why the signal flips sign on hard cats, which the v180/v183 evidence contradicts.
- The cross-video-level oracle (+0.0518) remains un-captured; any future attempt must use a NON-MIL level signal and must first show offline that it ranks hard-cat true positives ABOVE Normal-video FPs (the exact failure measured here).

## 6. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON); clc_level was NEVER implemented; all other flags default-OFF.
