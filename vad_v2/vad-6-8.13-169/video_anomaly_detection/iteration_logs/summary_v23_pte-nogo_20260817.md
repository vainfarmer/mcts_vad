# v23 / v181 — PTE (Per-category prompt-Text Ensemble for CLIP category curves): NO-GO at Phase-0 offline precheck (direction CLOSED, zero eval spend)

Date: 2026-08-17. Workflow: `vad-team-map20-v23` (runId wf_fb7143aa-1e6), 1 agent (precheck), 20 tool calls, ~15 min. NO batch_evaluate launched, NO VLM calls, NO code touched.

## 1. What was proposed

Signal-quality lever on the curve substrate that BUE/CBT/frame_scores consume (the v16 record: "next levers must attack SIGNAL quality"). For EACH of the 13 anomaly categories replace the committed single text embedding with the uniform mean of K=4 pre-registered paraphrase embeddings (each L2-normalized, averaged, renormalized). Frame features unchanged (cached clip_features reused); zero new VLM calls. Frozen 13x4 prompt table written BEFORE any eval: iteration_logs/v181_pte_prompts.json.

Composition detail (verified by the precheck agent): the committed text features are 13 cats x 6 hand-crafted prompts combined by LEARNED per-cat weights (prompt_weights.json, v93, trained on train video-level labels). PTE with len-4 != len-6 prompts cleanly triggers the built-in uniform-mean fallback in anomaly_expert._precompute_text_features — it REPLACES the weighted 6-prompt ensemble for the target category column; Normal + non-target columns stay committed.

## 2. Phase-0 offline precheck (the whole loop — NO-GO, so no eval)

Train-split sample: 4 videos per hard cat (Shooting/RoadAccidents/Shoplifting/Explosion) + 4 Training_Normal, per-frame CLIP ViT-L-14 feats computed once locally and cached (results/v181_pte_cache/*.npz). Per-video curves built with committed vs PTE text embeddings; prominence = (max(curve)-median)/(1.4826*MAD) over the GT interval (the v18 recipe). GT caveat verified in parquet: train split has NO frame annotations, so GT interval = full video for anomaly videos (video-level label).

Full per-video numbers: results/v181_pte_precheck.json; script: iteration_logs/analysis_v181_pte_precheck.py. n=20 train videos.

## 3. Pre-registered GO gate — RESULT

Gate: mean GT-window prominence improves >= +0.3 z on >= 3/4 hard cats AND no hard cat regresses > 0.1 AND normal-video curve max-z inflation < +0.5.

| category | committed z | PTE z | delta | >= +0.3? |
|---|---|---|---|---|
| Shooting | +4.091 | +5.476 | **+1.384** | PASS |
| RoadAccidents | +3.569 | +18.929 | **+15.360** | PASS |
| Shoplifting | +1.396 | +1.348 | -0.048 | FAIL (within -0.1 floor) |
| Explosion | +1.095 | +1.027 | -0.068 | FAIL (within -0.1 floor) |

Normal-video max-z inflation: **-1.2884** (the <+0.5 FP guard passes easily — PTE does NOT inflate normal curves).

Only **2/4** hard cats clear the bar (need >= 3) => **NO-GO**.

## 4. Read (why it lands this way)

PTE's short diverse phrasings massively sharpen RoadAccidents (+15.4 z — the committed long descriptive prompts were diluting the "car crash" core) and help Shooting (+1.4 z), but the committed long-form prompts already match Explosion's flash/debris visuals and Shoplifting's concealment gestures, so the ensemble is flat-to-slightly-negative there. A per-category best-of-both mix would be test-tuning-adjacent and was NOT proposed (pre-registered uniform K=4 only).

## 5. CLOSED directions (do not re-propose)

- **Uniform per-category prompt-text ensembling (PTE) of the CLIP category curves** — refuted offline at the pre-registered gate: gains concentrated in 2/4 cats, flat elsewhere. A non-uniform / per-category-selective variant would violate the no-test-side-tuning discipline of this design and is NOT a free re-proposal.
- The curve signal-quality hypothesis itself remains open, but text-side paraphrase ensembling is not its lever.

## 6. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON, full-290 ~0.173); pte_curves was NEVER implemented (NO-GO before code); all other flags default-OFF. Artifacts: iteration_logs/analysis_v181_pte_precheck.py, iteration_logs/v181_pte_prompts.json (frozen table), results/v181_pte_precheck.json, results/v181_pte_cache/.
