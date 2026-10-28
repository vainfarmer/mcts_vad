# v189 — BBS (Backbone-Swap oracle: SigLIP-SO400M-384 curves at missed GTs): NO-GO (labels-free curve hypotheses ALL refuted; failure is NOT backbone-specific)

Date: 2026-08-18. Precheck agent run (65 tool calls, ~70 min local incl. 3.5GB weight download via hf-mirror — huggingface.co unreachable from this network, HF_ENDPOINT=https://hf-mirror.com worked). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v189_bbs_precheck.py, results/v189_bbs_precheck.json (+ .out).

## 1. Question

After v185 (native-scale dense curves: 1/54 proposable) and v188 (motion-zoom: 11.1%, fallback-dominated), the last labels-free recall hypothesis: the failure is specific to ViT-L/14's representation. BBS recomputes the same curves with `ViT-SO400M-14-SigLIP-384` (webli; 384px input; HF timm repo commit ac16108d) — verbatim committed prompt table (new backbone tokenizer), prompt_weights.json as-is, identical curve recipe (z-scores are in-video relative so backbone calibration cancels).

## 2. Results — GO gate ((a) >= 40% of 54 reach z >= 2.0; (b) median uplift >= +1.0; (c) non-GT p90 < 4.0)

- (a) **1/54 = 1.9% — FAIL**
- (b) **median uplift -0.294 — FAIL**
- (c) guard 1.51 < 4.0 — PASS

Top uplifts: Explosion027 0.24 -> 3.56 (+3.32, the only z>=2), RoadAccidents011 -1.65 -> 0.89 (+2.54), RoadAccidents010 +1.04. Validity clean: all 51 videos max_abs_ts_diff = 0.0 vs committed run; zero errors.

## 3. Conclusion — CLOSED

The 54 missed GTs are invisible to SigLIP-SO400M-384 too. **The labels-free curve-signal hypotheses are now ALL refuted**: native density (v185), spatial zoom (v188), backbone swap (v189). Zero-shot image-encoder text-similarity curves do not contain the missed events, full stop.

The single untested representation family left: **video-native supervised features** — VideoMAE (UCF-Crime fine-tuned, models motion; the pipeline already uses OPear/videomae-large-finetuned-UCF-Crime as a classifier, weights cached locally) used as a CURVE substrate (sliding-window class probabilities). That is v190.

## 4. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON). No code touched.
