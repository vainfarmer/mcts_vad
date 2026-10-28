# v188 — SZM (Spatial-ZooM: motion-bbox-cropped frames -> CLIP curves at missed GTs): NO-GO (spatial direction CLOSED; validity clean)

Date: 2026-08-18. Precheck agent run (24 tool calls, ~24 min local). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v188_szm_precheck.py, results/v188_szm_precheck.json (per-GT table + per-video curves/fallback counts), .out.

## 1. Question

v185 proved the dense CLIP curve has no signal at the 54 pipeline-missed GT intervals at NATIVE frame scale. SZM tested the spatial hypothesis: does a motion-bbox crop (zoom) make the missed events visible to the same encoder? Crop rule (pre-registered): abs frame-diff vs previous kept frame at 64x64, mask > 25, largest connected component, map back, dilate 25%, clamp, min side 96px / max area 90%, else FULL-FRAME fallback.

## 2. Results — GO gate ((a) >= 40% of 54 missed GTs reach z_crop >= 2.0; (b) median uplift >= +1.0; (c) non-GT p90 max-z < 4.0)

- (a) **6/54 = 11.1% — FAIL** (worse than the 8/54 full-frame baseline; 5 of the 6 hits already had z_full >= 2.0; only RoadAccidents004 newly crossed 1.26 -> 2.33)
- (b) **median uplift -0.029 — FAIL**
- (c) guard 2.02 < 4.0 — PASS

**Fallback rate 68.8%** (5604/8148 kept frames): at 64x64 most frames have no motion component > 64 px — the "anomalies are small/distant" premise that motivated zooming is EXACTLY what defeats the motion-bbox cropper. Where crops did fire, gains were unsystematic (median ~0). Top uplifts: RoadAccidents011 -1.65 -> 0.91 (+2.56, still sub-threshold), Explosion027 +1.32, RoadAccidents004 +1.07.

Validity: timestamps matched the committed run exactly on all 51 videos (max_abs_ts_diff = 0.0); zero errors; zero inf-z.

## 3. CLOSED directions

- **Motion-bbox spatial zoom as a signal-repair mechanism** — refuted offline on the exact missed-GT population. The spatial variant of "change what the model sees" is closed alongside v185's temporal/dense closure.
- Status of the recall residual after v185+v188: the missed events are invisible to CLIP ViT-L/14 at native scale, at 32-snippet density, at full-frame 2fps density, and under motion-guided spatial zoom. The ONE remaining hypothesis class for recall-without-new-labels: **a different visual backbone** (the failure may be ViT-L/14's representation, not the pipeline's use of it).

## 4. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON). No code touched.
