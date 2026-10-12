# v185 — DSO (Dense-Substrate Oracle: is there curve signal at pipeline-missed GTs?): NO-GO, decisive (the v11 "denser localizer substrate" recall pivot is CLOSED)

Date: 2026-08-18. Precheck agent run (36 tool calls, ~26 min local CLIP recompute). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v185_dso_oracle.py, results/v185_dso_oracle.json (full per-GT table), .out.

## 1. Question

On hard-category videos where the committed pipeline's candidate generation MISSED the GT entirely, does the DENSE per-frame CLIP category curve contain a usable peak at the missed GT? A GT-aware oracle (FM13-compliant: both GT-aware and verdict-independent variants reported).

## 2. Setup

- Detections: results/batch_eval_v182_ref88 (fresh committed run, hard-88). Missed = no detection with IoU >= 0.3 against the GT interval.
- Coverage: Explosion 12 covered / 10 missed, RoadAccidents 11/12, Shooting 11/14, Shoplifting 7/18. **missed_total = 54 intervals across 51/88 videos** — the miss magnitude exceeds the old per-video "Shoplifting cand-miss 11/25" framing (18 intervals / 16 videos for Shoplifting alone). Expectation mismatch flagged, not a rule deviation.
- Curves: pipeline-exact recompute (VideoPreprocessor fps=2/motion filter -> CLIP ViT-L-14 mps -> committed text features with prompt_weights.json -> softmax cat/(cat+normal), logit_scale clamp 100). **Validity: recomputed kept-frame timestamps matched the committed run EXACTLY (max_abs_ts_diff = 0.0, 0 mismatches)**; MAD fallback never triggered.
- Proposable rule (pre-registered): contiguous region R with curve >= median + 1.0*MAD, IoU(R,GT) >= 0.3, AND peak z = (max-in-GT - median)/(1.4826*MAD) >= 2.0. Loose variant: peak z >= 2.0 anywhere in GT.

## 3. Results — GO gate (>= 40% strict GT-own-cat AND >= 30% verdict-independent): FAILED by >20x

| variant | strict proposable | loose peak-z |
|---|---|---|
| GT-own-category (GT-aware) | **1/54 (1.9%)** | 8/54 (14.8%) |
| max-over-13-cats (verdict-independent) | **0/54 (0%)** | 3/54 (5.6%) |

Notable rows: RoadAccidents002 GT [8.0-10.0] is the only strict hit (z=3.36, IoU 0.56); Shooting046 GT [158.7-169.6] has a real peak (z=3.07) but region IoU only 0.24 (peaks too narrow/misaligned); Explosion022 GT [74.3-80.7] max-curve region IoU 0.93 driven by ARSON (fire confusion, z=1.37) — overlap without a peak, illustrating cross-category failure of the GT-agnostic variant.

## 4. Conclusion — CLOSED

The dense CLIP category-curve substrate contains essentially **no usable signal at pipeline-missed GTs**. The "finer-resolution localizer substrate for recall" pivot (v11 lead B) is honestly CLOSED: the missed events are invisible to CLIP-ViT-L/14 at any density, not merely under-sampled at 32 snippets. Combined with the measured recognition ceiling (v18 frames, v20 prompt, v182 granularity) and the curve-transform closures (v181 PTE text side, v184 TSC frame side), the recall residual is now closed from EVERY cheap direction: the signal is not in the frames-as-shown, not in the prompt, not in the window size, not in the curve text embeddings, not in curve smoothing, and not in the dense curve substrate at all.

What this leaves open (expensive, qualitatively different instruments only): (i) a supervised temporal localizer trained with frame-level supervision (train split has none — would need pseudo-labeling or external data); (ii) a different visual backbone (the failure may be ViT-L/14's, not the pipeline's); (iii) spatial-zoom VLM presentation (change WHAT the VLM sees spatially, not temporally). None are cheap-loop material.

## 5. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON). No code touched.
