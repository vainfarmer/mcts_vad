# v146 — MARL-D: MIL-v1 Actionness-Run Localization with Decisiveness Gate

**Loop:** 2 (session v97) · **Date:** 2026-08-01 · **Verdict: REVERT (ACCURACY FLOOR BREACHED)**

## Headline verdict (the binding statistic)
Paired-json bootstrap on field `pre_marl_boundaries` (BUE held ON in both arms, 1000 resamples, n=65 common videos, 100 pre-mechanism rows, **73 windows actually relocated** by MARL):
- **point_delta = +0.03639** (REF mAP 0.17551 → NEW mAP 0.21190), mean +0.03746, sd 0.03432
- **90% CI = [-0.01774, +0.09582]** — **crosses zero**
- **P(delta>0) = 0.876** (< 0.90 gate)
- **NEW full mAP = 0.21190 < floor 0.21432** (short by 0.00242)

GATE: point_delta≥0.01 **True** · P(delta>0)≥0.9 **FALSE (0.876)** · mAP≥floor **FALSE**. Two of three gates fail. **VERDICT: REVERT.**

Full eval: mAP 0.21190, AP@0.3 0.31975, AP@0.5 0.17607, AP@0.7 0.13986, latency 130.01 s/video, [MARL] markers 69 (outer==inner, FM5 clean). Smoke: mAP 0.24151, 13/13 done, 15 markers.

## Mechanism
Count-neutral, width-adaptive **relocation** of far-mislocated FINAL detections onto the production **v1 MIL** (`model_mil.pth`) argmax-run, fired ONLY on **decisively-peaked** v1 MIL curves. Localizer = `video_data['mil_scores']` (verdict-independent, cached at pipeline.py:1345, already `.cpu().numpy()`). Placed immediately before committed BUE so BUE regrows the moved window. Count/conf/rank/type frozen. Decisiveness gate: prominence = max − P20 ≥ `marl_min_prominence` (0.0141) AND selected-run mean − video mean ≥ `marl_min_run_margin` (0.0081), both frozen at the **P90 of the NORMAL-train distribution** (train has no temporal segment GT — weak supervision — so IoU-max calibration was infeasible; NORMAL-P90 is a legitimate segment-GT-free, label-informed proxy). Far-gap trigger gap>6.0 s, half-max frac 0.5.

## Effect-size arithmetic — claim vs. reality
- **Claimed +0.03**: ~4–6 zero-overlap far-gap dets on decisively-peaked curves convert to IoU≥0.3 at the realized argmax-run rate (v1 = 6/17=35%, mean dIoU +0.234, never negative on the zero-overlap population); the 3/17 reaching IoU≥0.5 populate the sparse @0.5/@0.7 buckets. FM2 FP demotion structurally zero (count+conf frozen). Ship gate = full-65 offline mAP proxy ≥ +0.03.
- **What happened**: The offline proxy / in-vivo run DID produce a **+0.036 point win** — the largest point delta of the recent recall-family arc, and MORE than the claimed +0.03. But the win is **inside the noise band** (CI half-width ±0.057 here, crossing 0; P=0.876 < 0.9) and NEW mAP landed **0.00242 below the floor**. 73 windows relocated (far more than v145's 3), so the mechanism FIRED at scale — unlike v145 (null, 3 fires) this one had leverage — but the leverage was not statistically separable from run-to-run VLM noise and did not clear the absolute floor.
- **Contrast with the arc**: v143 additive → −0.018 (FP demotion, floor breach). v144 crash (no verdict). v145 count-neutral curve-peak → +0.000 (3 fires, no leverage). v146 count-neutral **MIL** relocation → +0.036 point but noise-band + below floor. The supervised MIL localizer moved the point estimate the right direction and fired at scale, but not decisively.

## FM1–FM17 table
- FM1 curve-shape: IMMUNE — moved window, not curve.
- FM2 added-FP: IMMUNE and CONFIRMED (count+conf frozen; no FP demotion; the +0.036 direction is positive, not the v143 −0.018 collapse). This is the decisive win vs v143.
- FM3 context-bleed: IMMUNE — no VLM.
- FM4 re-leveling: N/A — only windows move.
- FM5 activation: CLEAN — 69 full / 15 smoke [MARL] markers, outer==inner; per-video summary fired on all paths.
- FM6 sim-transfer: proxy used realized argmax-run (no oracle); the +0.036 point matched the proxy direction, so no gross over-prediction this time (unlike v143's TP-only oracle).
- FM7 video-prior: N/A (intra-video).
- FM8 one change: single relocation method.
- FM9 MPS: `mil_scores` already `.cpu().numpy()` at 1345 — handled.
- FM13 magnitude: bounded offline; fired at scale (73 windows) so NOT the failure cause here — unlike v145.
- FM15 substrate infidelity: AVOIDED — supervised MIL on its true substrate (features/*.npy → model_mil.pth) / the pipeline's own cached array.
- **FM16 NOISE FLOOR — the binding failure.** The +0.036 point win sits inside a 90% CI that crosses zero (P=0.876). Even a paired design (which removes ~±0.019 VLM run-to-run noise) could not push P≥0.9 on n=65. Confirms v140's closure: the 65-harness CI half-width (~±0.025 nominal, ±0.057 realized here on a high-variance mechanism) means ~+0.036 point effects remain UNFALSIFIABLE. This is the second gate failure.
- **FM17 empty-video deadlock: FIXED and VERIFIED** — summary marker fired on mil_scores==None / 0-det / flat-curve paths; smoke included Robbery050; report written cleanly, no [GATE] deadlock.

## Honest weaknesses (realized)
1. **Decisiveness gate weakly discriminative.** NORMAL-train prominence P50=0.0006; v1 MIL curves are very flat overall, so only ~12% of anomaly videos pass the decisive gate. Yet 73 windows relocated in vivo — the gate was looser in practice than the 12% train figure suggested, so it did NOT structurally suppress the flat-curve TP-regression it was designed to prevent. The point delta being positive (+0.036) means net conversions outweighed regressions, but not decisively.
2. **Below the floor.** Even a +0.036 point win landed 0.00242 under the 0.21432 floor because absolute NEW mAP (0.21190) is what the floor gates on; a positive paired delta vs a below-baseline REF (0.17551) does not lift the absolute number above the committed BUE-on level. The REF here (0.17551) is the pre-marl arm within THIS run, not the committed baseline — a positive within-run delta over a run that happened to start low cannot rescue an absolute mAP below floor.
3. Train segment-GT-free calibration (NORMAL-P90) is a proxy, not an IoU-optimal fit; the thresholds may be mis-set relative to what would maximize conversions.

## What the NEXT loop must NOT repeat
- Do NOT re-propose MIL-v1 argmax-run relocation gated on a NORMAL-P90 prominence threshold — CLOSED: fires at scale but the win is noise-band and below floor.
- Do NOT trust a positive PAIRED point delta over a low within-run REF to clear an ABSOLUTE mAP floor — a positive paired delta and a below-floor NEW mAP can coexist (they did). The floor gates on absolute NEW mAP.
- Do NOT expect a count-neutral relocation (even a supervised, high-firing one) to move the 65-harness verdict past P≥0.9 — v146 joins v145 in showing count-neutral relocation is safe (FM2 immune, confirmed twice) but cannot clear the noise floor on n=65.
- The candidate-generation/recall ceiling (50/99 dets IoU=0, max recall 0.379@0.3) remains OPEN after v143(additive floor-breach)/v144(crash)/v145(null)/v146(noise-band+below-floor). The supervised-localizer path is now shown to FIRE and move the point estimate the right way — the residual blocker is the 65-harness noise floor + absolute floor, not the localizer signal. Any further recall-family proposal must either (a) clear +0.03 with P≥0.9 on a properly-powered harness (v140: needs ~1600 videos for +0.005, ~290 for +0.03), or (b) lift ABSOLUTE mAP above the committed BUE-on floor, not merely beat a within-run low REF.

## Artifacts
- Full eval: iteration_logs/results/v146_full_eval.json · run log: v146_full_run.log · smoke: v146_smoke_eval.json · bootstrap: v146_bootstrap_ab.json
- Calibration script: iteration_logs/analysis_marl_train_calib.py (train-only, no test read)
- Backups restored from /Users/zzl/PycharmProjects/vad/.qoder/backups/v146/ (5 files, diff -q byte-identical, grep -i marl = 0)
