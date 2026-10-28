# Loop 5 — v107: ECE-Live (first in-vivo evaluation of the Event-Continuation Extension)

Date: 2026-07-21
Decision: **REVERT** (flag stays OFF; backups restored; no functional source change existed)

## Approved proposal (joint A+B consensus)

**Title:** ECE-Live (v107, joint A+B consensus): first in-vivo evaluation of the dormant Event-Continuation Extension — T3AL-style whole-video v51-half-max level-set growth of under-extended final detections, adjudicated by a deterministic intra-run pre/post A/B immune to VLM noise.

**Mechanism:** Single controlled change: activate (flag-ON) the already-implemented, dormant `--event_continuation_extension` (FCCS and EUBC stay OFF). At the final post-processing slot — strictly downstream of ALL scoring, thresholding, top_k, merge, NMS — for each FINAL detection whose duration < CATEGORY_DURATION_PRIORS[cat] max: read the cached whole-video relative category CLIP curve (read-only, zero new model calls), compute the exact v51 half-max level hm = (max(rel) + p20(rel)) / 2 on the whole-video curve, region-grow left/right from detection edges while rel >= hm (T3AL-style level-set growth). AutoLoc-OIC-style REJECT-CAP: grown span exceeding the category duration prior max keeps original boundaries; otherwise snap to the timestamp grid (pinned v99 semantics). Detection count, rank order, confidences, categories, and every upstream stage remain byte-identical; only the IoU of already-fixed detections can change. Flag-on runs record `pre_ece_boundaries` per detection (models.py:131-153), enabling a deterministic same-run pre/post mAP A/B through the official greedy matcher (evaluate.compute_temporal_map) as the keep/revert criterion, immune to the +/-0.03 VLM noise band.

**Rationale (abridged):** v97's IoU decomposition put ~39% of detection-quality loss in the IoU 0.1-0.5 boundary-imprecision bucket dominated by long-event-shrunk (v51 local-window "cage" — an architecture limitation). Strongest deterministic evidence in project history: v99 simulation through the OFFICIAL evaluator gave +0.0307 mAP; agent C independently reproduced it to the last digit and re-ran a live-faithful grid at +0.0311 with zero threshold-crossing TP losses; a +/-10% level-perturbation band lower-bounded the gain at +0.015. The mechanism had NEVER produced an in-vivo result (v99 = pure FM5, 0 [ECE] lines; v105/v106 = orchestrator crashes). ECE sits in the only change class never observed to regress (v89/v94/v95/v96: read-only-curve, post-hoc modulation of fixed detections).

**Expected gain:** +0.0307/+0.0311 deterministic simulation delta; perturbation floor +0.015; projected full-eval mAP ~0.165-0.180 vs 0.1494 true baseline, concentrated in AP@0.5/AP@0.7.

**Decision rule (pre-registered):** KEEP iff deterministic intra-run pre/post delta >= +0.015 mAP; REVERT if delta < +0.005; +0.005..+0.015 = hold/inconclusive. Headline mAP used only for sanity, never for adjudication.

**Risks (abridged):** FM1 avoided by construction (read-only curve, post-hoc on fixed detections); FM2 impossible (count/order/confidence/category frozen); FM3/FM4 absent (zero VLM calls); FM5 answered by five-link verified chain (real CLI flag, from_args threading, per-video '[ECE] summary:' marker on all paths, in-process exit-2 gate at batch_evaluate.py:162-170, post-run grep). Residual real risks: (a) whole-video hm uses max(rel) — if the global max lies outside a detection, growth may no-op; (b) unconditional grid-snap slightly moves boundaries of all sub-max_dur detections; (c) simulation-to-live gap (fresh stochastic VLM detections differ from the v99 cache) — bounded by the deterministic same-run pre/post A/B, the sole keep/revert criterion.

**Latency:** neutral (<50ms/video numpy walk on cached curve).

**Literature:** T3AL (CVPR 2024, arXiv 2404.05426) — whole-video adaptive-threshold level-set localization; LEC-VAD (ICML 2025, arXiv 2506.13095) — incomplete localization as THE core WS-VAD defect; AutoLoc (ECCV 2018) — OIC rationale for the reject-cap; BSN/BMN — proposal completeness; EventVAD (ACM MM 2025) — training-free event extent on UCF-Crime.

## C's feedback (APPROVE)

C independently re-verified the entire five-link activation chain in the current tree by execution: live `--help` flag; `VADConfig.from_args` threading (flag-off byte-identical to default); `_event_continuation_extension` (pipeline.py:1165-1298) exactly implements pinned v99 semantics, invoked at 1121-1125 as the literal last stage before dict projection with `pre_ece_boundaries` recorded; '[ECE] summary:' fires on both early-return paths (912-913, 1004-1005) plus main path with the in-process exit-2 hard gate; `analysis_v106_ece_ab.py` correctly scores pre/post through the official compute_temporal_map. Evidence quality rated best in project history. Minor caveat (not blocker): compute_frame_scores runs AFTER ECE, so frame AUC shifts slightly on extended detections — metric-cosmetic; record frame AUC in the handoff.

## Implementation notes (D1)

v107 = verify-and-activate loop; ZERO functional source edits. The single controlled change was running the eval with the ECE flag ON. Backed up pipeline.py/config.py/batch_evaluate.py/models.py to `.qoder/backups/v107/`; verified the five-link activation chain by execution; re-ran the synthetic-curve unit exercise (material growth [22,28]->[20,39.5] at hm=0.5000, reject-cap, and skip paths all exercised, both '[ECE] pre=' and '[ECE] summary:' lines emitted); py_compile clean on all 11 main modules; only edit was a doc-only pipeline.py docstring bump v105->v107.

## Eval configuration

- Eval flags: `--event_continuation_extension` (both smoke and full)
- Activation marker: `[ECE] summary:`
- Marker counts: smoke log = **14** ; full log = **63** (= succeeded videos; 63/65 succeeded, 2 failed) — FM5 guard ACTIVE, in-process gate passed
- ECE action breakdown (full run, aggregated from summary lines): **34 material / 45 grid-snap / 9 reject-cap / 9 skip** across 63 videos — the mechanism fired materially (not a no-op)

## Results

| Run | mAP | AP@0.3 | AP@0.5 | AP@0.7 | avg latency (s/video) |
|---|---|---|---|---|---|
| smoke (v107, flag-ON) | 0.147441 | — | — | — | — |
| full (v107, flag-ON) | 0.172905 | 0.262952 | 0.138340 | 0.117424 | 166.42 |

Frame AUC (full, recorded per C's caveat; computed after ECE): 0.71648, AUPR 0.47945.

### Deterministic intra-run pre/post A/B (adjudication criterion)

`analysis_v106_ece_ab.py --results_dir results/batch_eval_v107_full --dataset_dir /Users/zzl/Movies/ucf-crime`

- videos scanned: 63; lacking pre_ece_boundaries: 0; activation anomalies: 0; videos in A/B: 63
- A (pre-ECE):  mAP 0.10008 (AP@0.3 0.11842, AP@0.5 0.09091, AP@0.7 0.09091)
- B (post-ECE): mAP 0.10211 (AP@0.3 0.12451, AP@0.5 0.09091, AP@0.7 0.09091)
- **DELTA = +0.00203** (entirely from AP@0.3; AP@0.5 and AP@0.7 unmoved)

## Decision: REVERT

Pre-registered rule: revert if deterministic delta < +0.005. Measured delta = **+0.00203 < +0.005** → REVERT. The headline full mAP of 0.1729 (vs 0.1494 baseline) is within/above the ±0.03 VLM noise band but was explicitly pre-registered as sanity-only, never adjudication. Revert action: flag stays OFF; all four backed-up files restored from `.qoder/backups/v107/` and verified identical by diff (the restore is a formality — no functional source edits existed).

## Lessons learned

1. **FM5 refuted for ECE (finally an in-vivo result):** the five-link activation chain worked exactly as designed — 63/63 markers, in-process gate passed, 0 activation anomalies. The mechanism ran live for the first time in project history. The dominant historical killer of this mechanism is now fully answered.
2. **Simulation-to-live gap CONFIRMED as the real failure mode (risk (c)):** the v99 cached-candidate simulation promised +0.0307/+0.0311 with a +0.015 perturbation floor, but on fresh stochastic VLM detections the same-run deterministic delta was only +0.00203. The v99 cache's under-extended long events were not reproduced in this run's detection realization; ECE's benefit is contingent on the specific boundary-error distribution of the cached run, not a stable property of the pipeline. Deterministic simulations on cached outputs upper-bound, and do not transfer to, live gains.
3. **The gain that did materialize was shallow:** +0.006 at AP@0.3 only, with AP@0.5/AP@0.7 completely unmoved — whole-video level-set growth extended detections, but not enough to cross the higher IoU thresholds where the v97 bottleneck analysis located the loss. The long-event-shrunk bucket is either upstream (detection placement) rather than boundary-extension-fixable, or requires per-detection (not whole-video-max) level statistics.
4. **The intra-run pre/post A/B design worked perfectly:** it delivered a noise-immune verdict on a run whose headline (0.1729) would otherwise have looked like a +0.023 win and triggered a false KEEP. Pre-registering the adjudication channel and refusing the headline was the single most valuable methodological decision of this loop. FM1/FM2 confirmed absent as predicted (count/order/confidence frozen; no regression in the A/B).

## Post-hoc full-eval analysis (supplement, 2026-07-21)

### Cross-version comparison (full eval)

| Version | mAP | AP@0.3 | AP@0.5 | AP@0.7 | frame AUC | avg lat (s) | succeeded |
|---|---|---|---|---|---|---|---|
| v94 (true baseline) | 0.1492 | 0.2046 | 0.1335 | 0.1096 | 0.6914 | 127.6 | 63/65 |
| v97 | 0.1615 | 0.2182 | 0.1500 | 0.1164 | 0.7081 | 141.5 | 65/65 |
| v100 | 0.1571 | 0.2153 | 0.1386 | 0.1174 | 0.7078 | 133.3 | 65/65 |
| **v107** | **0.1729** | **0.2630** | 0.1383 | 0.1174 | 0.7165 | 166.4 | 63/65 |

### Headline decomposition

- v107 headline 0.1729 is the highest full mAP of the session, but the entire +0.0237 delta vs v94 sits in AP@0.3 (+0.0584); AP@0.5 (0.1383) and AP@0.7 (0.1174) match v100 exactly. The deterministic A/B attributes only +0.00203 of it to ECE — the remaining ~+0.02 is favorable stochastic VLM detection realization at loose IoU, i.e. noise, not mechanism.
- AP@0.7 = 0.117424 is bit-identical across v99/v100/v102/v107: a stable core of well-localized TPs that NO post-hoc boundary mechanism tested so far has touched. The high-IoU bottleneck is upstream detection placement, not boundary refinement.
- ECE fired materially (34 material growths out of 97 ECE actions, 35%) yet moved mAP by only +0.002 — material growths either too small to cross IoU 0.5/0.7 or applied to detections that were not the ones costing IoU. Consistent with lesson 3.

### Category-level movement (frame AUC vs v94)

Variance is concentrated in four categories: Shooting +0.140, Vandalism +0.083, Arson +0.062, Robbery −0.078; Assault/Burglary/RoadAccidents/Stealing are essentially unchanged. Robbery is also the worst classification category this run (1/5 correct) — category confusion there, not localization, is the failure. Shoplifting frame AUC remains pathological (0.217) across runs.

### Run health

- 2/65 failures, both `openai.APITimeoutError` (Shooting034_x264 at 306.0s, Stealing062_x264 at 172.5s) — infrastructure, not mechanism.
- 166.4 s/video is the session's slowest, driven by API latency/timeouts; ECE itself is <50 ms/video, so latency is NOT attributable to the flag.

### A/B absolute-level caveat

`analysis_v106_ece_ab.py` scores against the FULL GT set (`load_ground_truth`), while the official report matches GT∩pred (`gt_matched`, evaluate.py:504-505). The 63-video run therefore carries the missing videos' GT segments as pure FNs in the A/B, deflating absolute pre/post mAP to ~0.100 vs headline 0.173. The pre/post DELTA remains internally valid (both sides share identical GT and identical detection sets), but future A/B scripts should restrict GT to matched videos so absolute levels are also comparable.

### Supplementary conclusions

1. True baseline remains v94 full mAP 0.1494; v107's 0.1729 must NOT be adopted as a new baseline — it is a flag-reverted run whose surplus is unadjudicated noise.
2. Whole-video post-hoc boundary editing is now measured (not simulated) at ≈+0.002 mAP. The boundary-extension family (ECE and variants) should be deprioritized unless redesigned with per-detection level statistics.
3. The remaining path to mAP≥0.20 points at detection placement/recall at IoU≥0.5 (upstream scoring/proposal quality), where v97-style analysis located the loss and where AP@0.5/0.7 have been flat across five versions.

## Artifacts

- `iteration_logs/results/v107_smoke_run.log`, `v107_smoke_eval.json`
- `iteration_logs/results/v107_full_run.log`, `v107_full_eval.json`
- Backups (restored): `/Users/zzl/PycharmProjects/vad/.qoder/backups/v107/`
