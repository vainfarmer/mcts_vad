# v184 — TSC (Temporal-Smoothed category Curves, Gaussian sigma=1.5 frames): NO-GO at Phase-0 offline precheck (direction CLOSED per pre-registered gate, zero eval spend)

Date: 2026-08-18. Precheck agent run (7 tool calls, ~5 min). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v184_tsc_precheck.py, results/v184_tsc_precheck.json (+ .out).

## 1. What was proposed

Frame-side counterpart to v181 PTE (text side, closed): the per-category CLIP curves from FrameScorer.get_category_scores (frame_scorer.py L41) are RAW — only the fused frame_scores get Gaussian smoothing today. Curve noise perturbs the boundary operators (BUE grows along them, CBT clamps along them) directly. TSC = apply ONE fixed Gaussian smooth (sigma = 1.5 frames, pre-registered, no fitting; scipy gaussian_filter1d mode=reflect truncate=4.0) to each category curve before any downstream consumer.

## 2. Phase-0 offline precheck

Reused the v181 cache (results/v181_pte_cache/*.npz: per-frame CLIP feats for 20 TRAIN videos — 4 per hard cat + 4 Training_Normal) and the v181 recipe (prominence = (max-in-GT - median)/(1.4826*MAD); train split has no frame GT so GT = full video for anomaly videos). Committed prompts (NOT PTE paraphrases).

## 3. Pre-registered GO gate — RESULT (identical shape to v181)

Gate: mean prominence delta >= +0.3 z on >= 3/4 hard cats AND no cat regresses > 0.1 AND normal max-z inflation < +0.5.

| category | raw z | tsc z | delta |
|---|---|---|---|
| Shooting | 4.09 | 3.02 | **-1.07** (regresses) |
| RoadAccidents | 3.57 | 3.22 | **-0.35** (regresses) |
| Shoplifting | 1.40 | 1.40 | +0.01 |
| Explosion | 1.09 | 1.56 | **+0.46** (only PASS) |

Normal max-z13 inflation: **-22.30** (smoothing crushes normal false peaks, e.g. Normal008 85.9 -> 19.9; FP guard passes easily).

Only 1/4 cats reach +0.3 AND two cats regress beyond -0.1 => **NO-GO**.

## 4. Why it fails (the lesson)

The median/MAD z recipe penalizes smoothing on videos whose raw curve already has one tall clean peak over a flat low-MAD baseline (Shooting017: 10.81->5.36; RoadAccidents003: 9.51->7.76) — max falls faster than MAD shrinks. TSC nets positive only on noisy-flat curves (Explosion015: 1.76->3.53). The committed curves' problem is NOT excess high-frequency noise on the videos that matter; their useful peaks are already narrow, and sigma=1.5-frame smoothing destroys exactly that narrowness.

Recorded caveat (from the precheck agent): this proxy measures peak-vs-MAD net only; it cannot credit TSC's intended boundary-op stability benefit. The closure therefore stands **per the pre-registered gate**; a revival would need a different pre-registered boundary-stability metric, not a re-run of this one.

## 5. CLOSED directions (do not re-propose)

- **Fixed Gaussian temporal smoothing of the CLIP category curves (sigma=1.5 frames)** — measured NO-GO on the v18 prominence recipe.
- Combined with v181 PTE (text side): BOTH sides of the "improve the curve signal by cheap deterministic transforms" hypothesis are now closed — neither paraphrase ensembling the text embeddings nor smoothing the frame axis improves the curve substrate under the registered prominence metric. The curve-quality hypothesis remains open only to QUALITATIVELY different instruments (e.g., learned temporal context, different visual backbone), not to parameter variants of these two.

## 6. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON); TSC was NEVER implemented; all other flags default-OFF.
