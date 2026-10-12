# v22 / v180 — SNH-P (MIL-argmax-peak-PROTECTED SNH veto): NO-GO at Phase-0 offline precheck (direction CLOSED, zero eval spend)

Date: 2026-08-17. Workflow: `vad-team-map20-v22` (runId wf_3f8dd5ac-be0), 1 agent (precheck), 48 tool calls, ~37 min. NO batch_evaluate launched, NO VLM calls, NO code touched.

## 1. What was proposed

v179 SNH (video-level P(normal)>=tau confidence veto) PASSED the FA objective hugely (Normal-video FA 0.50667 -> 0.16667, **-67.1%**) and held abs mAP (0.17324 >= 0.16454) but FAILED G4 GT-harm decisively: 5 GT-overlapping (IoU>=0.1) detections vetoed (limit <= 1):
RoadAccidents128 3.00-8.00 (IoU 0.7333, conf 0.9167), Explosion025 10.00-16.50 (0.5106), Shooting048 46.00-52.00 (0.4286), Shoplifting022 45.50-55.00 (0.4211), RoadAccidents002 9.00-11.50 (0.2857).

SNH-P = ONE pre-registered structural protection rule: when the video-level gate fires, veto every detection EXCEPT any whose [t_start,t_end] covers the video's MIL argmax snippet time t_peak=(argmax(mil_scores)+0.5)/32*duration (single strongest snippet-level evidence; model_mil.pth). tau untouched (no numeric test-fitting).

## 2. Phase-0 offline precheck (the whole loop — NO-GO, so no eval)

Method: parsed all 92 [SNH-FIRE] lines (60 distinct videos) from results/batch_eval_v179_snh290.log; recomputed 32-snippet MIL scores LOCALLY and pipeline-exactly (VideoPreprocessor fps=2/motion_threshold=1.0/min_motion_interval=2.0 -> CLIP ViT-L-14/openai on mps -> model_mil.pth, 32 consecutive-chunk mean-pool, one forward) via NEW script iteration_logs/analysis_v180_snhp_precheck.py. Cache: results/v180_mil_argmax.json; report: results/v180_snhp_precheck_result.json.

Validity checks: recomputed durations == eval-JSON durations all 60 videos; all 6 undefined-argmax videos exactly reproduce the v179 eval's own "[MIL] Too few frames" lines with identical kept-frame counts (RoadAccidents128=22, Explosion025=21, RoadAccidents002=24, Explosion016=23, RoadAccidents004=26, Normal_Videos_889=18) — treated as NOT protected per pre-registered caveat.

## 3. Pre-registered GO gate — RESULT

- Gate (a) >= 4/5 GT-harm dets protected: **FAIL — 0/5**.
  - RoadAccidents128 (harm window 3.00-8.00): MIL argmax UNDEFINED (22 kept frames < 32) -> not protected.
  - Explosion025 (10.00-16.50): MIL argmax UNDEFINED (21 < 32) -> not protected.
  - RoadAccidents002 (9.00-11.50): MIL argmax UNDEFINED (24 < 32) -> not protected.
  - Shooting048 (46.00-52.00): MIL argmax t=61.39s — OUTSIDE the harm window -> not protected.
  - Shoplifting022 (45.50-55.00): MIL argmax t=3.42s — OUTSIDE the harm window -> not protected.
- Gate (b) est FA rel reduction under protection >= 25%: PASS — 0.5789. Of 51 fully-vetoed Normal videos, 7 resurrect (Normal_Videos_003/010/041/606/873/882/898): est_fa=25+7=32, fa_rel_est=(76-32)/76=0.5789.
- Overall fires_protected=13/92 (12 Normal-video fires + Shoplifting022 idx0 t=0.50-6.50, NOT a GT-harm det).

Both gates required => **NO-GO**.

## 4. Why the rule fails structurally (the lesson)

The MIL-peak protection rule cannot rescue the G4 GT-harm regressions for two independent structural reasons:
1. **Short-clip degeneracy**: 3 of the 5 harm videos (all the brief-event RoadAccidents/Explosion ones) have < 32 kept frames after motion filtering, so the 32-snippet MIL recipe produces NO scores at all — the eval's own log shows "[MIL] Too few frames". The strongest snippet-level evidence simply does not exist on exactly the videos where the video-level head misfires.
2. **Peak misplacement where MIL exists**: on the 2 harm videos with valid MIL, the argmax sits far outside the vetoed true-positive window (Shooting048: peak 61.4s vs harm 46-52s; Shoplifting022: peak 3.4s vs harm 45.5-55s) — MIL argmax does not localize the v179-vetoed TP windows.

## 5. CLOSED directions (do not re-propose)

- **MIL-argmax-peak protection of the SNH video-level veto** — refuted offline, 0/5 harm coverage. Zero eval spend.
- Combined with v21 (SNH whole-video veto fails G4) and v17 (CNV): the **video-level null-hypothesis gate family** is now closed from both sides — the blunt veto harms GT, and the natural snippet-level protection signal (MIL argmax) does not exist or does not align on exactly the harmed videos. Any future Normal-FP work must NOT gate at whole-video granularity with MIL-peak protection.

## 6. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON, full-290 ~0.173); cgr/ccr/cnv/kfd/tcp/snh_veto all default-OFF; snh_protect was NEVER implemented (NO-GO before code). Artifacts: iteration_logs/analysis_v180_snhp_precheck.py, results/v180_mil_argmax.json, results/v180_snhp_precheck_result.json, results/v180_snhp_precheck.out.
