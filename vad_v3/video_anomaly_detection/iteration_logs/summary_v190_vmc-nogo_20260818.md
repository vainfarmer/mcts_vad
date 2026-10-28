# v190 — VMC (VideoMAE sliding-window class-probability curves at missed GTs): NO-GO (last representation family closed; labels-free recall exhaustively refuted)

Date: 2026-08-18. Precheck agent run (32 tool calls, ~45 min MPS, zero downloads — local cache, HF_HUB_OFFLINE=1). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v190_vmc_precheck.py, results/v190_vmc_precheck.json (+ .out).

## 1. Question

The final untested representation family for the 54 pipeline-missed GTs: video-native SUPERVISED motion-aware features — VideoMAE (OPear/videomae-large-finetuned-UCF-Crime, 14 classes) as a sliding-window class-probability curve substrate (16 kept-frame windows, stride 4 = 2s, prob assigned to window center).

## 2. Results — GO gate ((a) GT-aware >= 40% reach z>=2.0; (b) median >= +1.0; (c) guard non-GT p90 < 4.0): ALL FAIL

- (a) **11/54 = 20.4%** (vs CLIP's 1.9% — a real 10x lift, but half the gate)
- (b) median GT-aware z **+0.68**
- (c) guard own-cat non-GT p90 **12.18 — FAIL** (Explosion-curve saturation: near-zero MAD inflates z to 961.6/183.6 — calibration artifacts, not detections; the same curves fire off-GT). Verdict-independent maxcat variant: 9.3% reach, median 0.67 — worse.
- Notable: Shoplifting029 own z=5.99 vs CLIP 1.03 — the one genuinely category-specific VideoMAE response on a motion-gesture event; isolated.

Validity: timestamps == committed run (max_abs_ts_diff=0.0, all 51); 2 short videos (<16 kept frames) single padded window, documented; 1-235 windows/video (median 26).

## 3. Exhaustive closure map for the recall residual (all offline-measured on the SAME 54 missed GTs)

| instrument | result |
|---|---|
| Dense CLIP curve (v185) | 1.9% proposable |
| Motion-zoom CLIP (v188) | 11.1%, fallback-dominated |
| SigLIP-SO400M backbone (v189) | 1.9% — not backbone-specific |
| VideoMAE supervised curves (v190) | 20.4%, guard-failing calibration artifacts |
| VLM frames (v18) / prompt (v20) / window granularity (v182) | recognition ceiling, 3x measured |

The missed events are not recoverable by any signal family currently in the tree. Remaining recall paths require NEW SUPERVISION (frame-level labels the train split lacks) or NEW SENSING (different model class entirely) — both outside cheap-loop scope.

## 4. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON). No code touched.
