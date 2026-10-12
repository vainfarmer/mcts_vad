# Session v24 Summary — v182 FSD (Fine Segment Division) hard-88 A/B → REVERT, no promotion

Date: 2026-08-18 (runs executed 2026-08-17/18). Agent C session record. Plain text.
Verdict: **DO NOT PROMOTE. fsd_cats stays DEFAULT OFF. Committed pipeline (v137 + BUE + CBT + CBT-recal, all default-ON; directed scan; full_coverage_comparative_scan=False) untouched.**

---

## 1. Motivation — the third attack on the brief-event recognition residual

The brief-event residual (brief 1-3s events inside 6s directed-scan windows rejected by the VLM) has now been attacked from two sides and CLOSED twice:

1. **Frames (v18 §10):** RA009 (GT 7-8s, inside segment 4.5-10.5) and RA021 (GT 1-3s, inside segment 0-5.2) were REJECTED with all 11-12 frames of the segment shown — the committed directed path was never frame-blind.
2. **Prompt (v20, v177 TCP):** the anti-clip-averaging temporal-contrast prompt block did NOT convert the purpose population (RA009 identical miss in both arms; RA021 REGRESSED from REF hit IoU=0.6 to zero detections). Clip-averaging hypothesis refuted; whole-segment prompt rewording closed.

Residual open problem: **Shoplifting cand-miss 11/25** (v155/v160 diagnosis, still unaddressed — boundary ops provably cannot fix zero-overlap) and the RA009/RA021 recognition-bound pair. v182 FSD tested the remaining geometric side of the same hypothesis: **if text cannot steer the VLM to brief events inside a 6s window, shrink the window** — split each SELECTED 6s segment into its 2 chronological 3s halves at the scan-iteration step only, so the VLM judges ~6-frame halves instead of ~12-frame segments, with everything else (call shape, prompt, JSON schema, candidate selection on the original 6s grid, merge/NMS/classify/BUE/CBT) untouched.

## 2. The mechanism — FSD, grid-side ONLY, gated to Shoplifting/RoadAccidents

Single switch `--fsd_cats` (default OFF = byte-identical committed behavior). When ON AND the pipeline-resolved `videomae_category ∈ FSD_CATEGORIES = {Shoplifting, RoadAccidents}` (pipeline.py L1084), the DIRECTED Stage-4 scan (BOTH `scan_segment` paths: `_directed_llm_scan` and the `discovery.discover` fallback) iterates 3s half-segments instead of the 6s grid. The split is applied AFTER segment selection, at the scan-iteration step only (anomaly_discovery.py `fsd_split_segment` L51, single source; halves >=3 frames kept, <3 dropped; both halves <3 frames → original segment kept). Candidate build (CLIP-loc targeting + generic pre-screen + min_pass/max_scan caps) is computed on the ORIGINAL 6s grid BEFORE splitting — selection bit-identical to flag-off. `video_data['segments']` never mutated; the >60-segment long-video rule (comparative_long_segments) keeps seeing the 6s grid; `_fccs` untouched and stays OFF (FSD never armed on the comparative path). Per-video arming: `discovery.fsd_split` (inert default False), set at Stage 4 (pipeline.py L1993-1996).

**Gating source actually used:** the pipeline-resolved VideoMAE `videomae_category` (Stage-2 result, incl. the "Normal override -> CLIP" label-space fallback when armed). **The label-space fallback did NOT fire** — grepped: `grep -c "Normal override -> CLIP" batch_eval_v182_fsd88.log = 0` and `batch_eval_v182_ref88.log = 0`. Gating therefore reduced to the raw VideoMAE category. Resolved-category gating produced 43 gated videos: 24 resolved RoadAccidents + 19 resolved Shoplifting, including three cross-gated videos (Explosion043→RoadAccidents, Shooting043→RoadAccidents, Shoplifting007→RoadAccidents) — as designed (gating is by resolved category, not filename).

Markers (FM5/FM17/FM21): `[FSD-FIRE]` exactly once per GATED video (vid, category, n_segments_6s -> n_segments_3s); `[FSD-VID]` unconditional per video on every path (gated=0|1, n_segments).

## 3. Loop 1 — hard-88 cross-run A/B vs FRESH same-session committed REF

Commands (COMMAND.txt both dirs, `--categories Shooting,RoadAccidents,Shoplifting,Explosion`, `--concurrency 2`; FSD arm adds `--fsd_cats`):
`python3.14 -u -m video_anomaly_detection.batch_evaluate --dataset_dir ... --output_dir .../batch_eval_v182_{fsd88,ref88} --max_per_category 999 --concurrency 2 --categories Shooting,RoadAccidents,Shoplifting,Explosion [--fsd_cats]`

Watcher check: 88/88 succeeded both arms ("Succeeded: 88, Skipped: 0, Failed: 0" in both logs); `num_videos` FSD 88 / REF 88.

**FCCS==0 discipline held (FM20 grep-not-claim):**

```
grep -c "FCCS" batch_eval_v182_fsd88.log -> 0
grep -c "FCCS" batch_eval_v182_ref88.log -> 0
```

Both arms ran the identical committed directed scan; the v173 confound did not recur.

**Verdict statistics (results/batch_eval_v182_fsd88/bootstrap_ab.json, mode "cross-run", n_videos=88) — GREPPED from the produced artifact:**

| metric | NEW (FSD) | REF (committed) |
|---|---|---|
| mAP (mean AP@0.3/0.5/0.7) | 0.08967937558853943 | 0.11050929920980064 |
| AP@0.3 | 0.17313283731194176 | 0.19390949998426635 |
| AP@0.5 | 0.08417508417508417 | 0.11489112491786288 |
| AP@0.7 | 0.011730205278592375 | 0.022727272727272728 |

- point_delta = **-0.02082992362126121**; CI90 = [-0.039925435269741086, +0.008945443252604702]; **P(delta>0) = 0.16**; artifact verdict field: **REVERT**.
- Log-tail reproduction: FSD mAP 0.0897 (AP 0.1731/0.0842/0.0117), REF mAP 0.1105 (AP 0.1939/0.1149/0.0227) — consistent.

**Gate evaluation (pre-registered: promote-to-290 iff point_delta >= +0.008 AND P(delta>0) >= 0.9 AND new hard mAP >= floor):** ALL THREE legs FAIL — point_delta is NEGATIVE (-0.0208), P(delta>0)=0.16 << 0.9, and new mAP 0.08968 sits below every floor reading in force. FSD is a measured net-negative, not merely a non-robust positive.

**=> Loop 1 gate FAILED. Loop 2 (full-290) never armed (290 evidence: null).**

**Caveat (verbatim, in force):** "cross-run mode still carries VLM run-to-run variance (~+/-0.019 unpaired in this ledger) ON TOP of the bootstrap CI. Prefer a paired design when the mechanism is post-hoc." Under this caveat the -0.0208 point delta cannot be attributed to FSD with certainty (same magnitude as unpaired run noise); promotion however requires POSITIVE evidence, and there is none — P(delta>0)=0.16 is a decisive absence-of-benefit measurement.

**FM2 watch:** det/video FSD 142/88 = 1.6136 vs REF 132/88 = 1.5000 (+0.1136, far below the +1 flag; the +10 detections are the 3s-grid's extra acceptances, not count-inflation machinery). fa_rel_inc: -1 (not measurable — hard-88 contains no Normal videos; the Normal-FA gate was a Loop-2 gate and Loop 2 never ran).

## 4. Purpose population — NOT converted (GREPPED from produced per-video JSONs)

| video | GT | REF (committed) | FSD | IoU>=0.1 conversion? |
|---|---|---|---|---|
| RoadAccidents009 | 7-8s | 1 det: 20.0-22.0s (conf 0.6495), IoU=0.0 | 2 dets: 0.5-2.0s (conf 0.6443) + 20.0-22.0s (conf 0.3196), BOTH IoU=0.0 | NO — FSD's EXTRA 0.5-2.0s detection does not hit GT; no IoU>=0.1 gain |
| RoadAccidents021 | 1-3s | 0 detections | 0 detections | NO — zero detections in BOTH arms |

The 3s window did not make the VLM accept RA009's brief event (the extra FSD detection at 0.5-2.0s is off-GT, and the 20-22s false positive persists in both arms), and RA021 stayed empty in both arms. The purpose population did not convert.

## 5. Per-category (temporal mAP, recomputed evaluator-faithful from the produced per-video JSONs + GT, n=88)

| category | FSD | REF | delta (FSD-REF) | n |
|---|---|---|---|---|
| Shooting | 0.07594 | 0.08936 | **-0.01342** | 23 |
| RoadAccidents | 0.16109 | 0.19119 | **-0.03010** | 23 |
| Shoplifting | 0.09243 | 0.06703 | **+0.02539** | 21 |
| Explosion | 0.15863 | 0.13041 | **+0.02823** | 21 |

FSD gains on Shoplifting/Explosion but loses more on RoadAccidents/Shooting. The RoadAccidents loss (-0.0301) lands on the very category the mechanism was built for (brief crash events); the Shoplifting gain did not come from the cand-miss bucket converting (purpose population NOT converted, §4).

## 6. Cost delta (GREPPED [COST] lines in both produced logs)

| population | vlm_calls FSD | vlm_calls REF | ratio | clip_calls FSD | clip_calls REF | ratio |
|---|---|---|---|---|---|---|
| Gated videos only (n=43, gated=1) | 1300 | 657 | **1.98x (~2x, as expected)** | 462 | 416 | 1.11x |
| All 88 videos | 2180 | 1545 | **1.41x** | 1010 | 1010 | 1.00x |

The ~2x VLM cost on gated videos is the designed cost of scanning 3s halves instead of 6s segments (2x scan calls on the selected segments of gated videos; CLIP pass count is grid-independent). Wall-clock: FSD 81.4 s/video avg vs REF 331.5 s/video avg — cross-session API drift dominates wall time (the two runs executed ~19h apart); call count is the honest cost metric.

## 7. FM evidence (all GREPPED in produced artifacts)

- FM5/FM21 (mechanism armed): `grep -c "FSD-FIRE" batch_eval_v182_fsd88.log = 43` == number of gated=1 videos; sample: `[FSD-FIRE] vid=RoadAccidents009_x264.mp4 category=RoadAccidents n_segments_6s=7 -> n_segments_3s=14`.
- FM17 (unconditional per-video marker): `grep -c "FSD-VID" batch_eval_v182_fsd88.log = 88` == videos processed (incl. 0-detection paths); `grep -c "FSD-VID" batch_eval_v182_ref88.log = 0` (flag OFF, markers correctly absent by byte-identity).
- Gating split: `gated=1` 43, `gated=0` 45 (43+45=88); FSD-FIRE resolved-category split: 24 RoadAccidents / 19 Shoplifting.
- Label-space fallback: `grep -c "Normal override -> CLIP"` = 0 in BOTH logs — the fallback did not fire this run.
- Smoke (2 videos: Shoplifting001 + Normal_Videos_782, both succeeded): ON — FSD-FIRE=1 (Shoplifting001 32->64 segments), FSD-VID=2, Normal gated=0 (arming correct); OFF (x2 independent runs) — 0 FSD markers; off-vs-off2 JSONs differ ONLY in VLM confidence jitter on identical windows (33.5-35.5 / 48.0-56.5 both runs; Normal video zero dets both) — a live demonstration of the unpaired cross-run noise band behind the §3 caveat. Smoke mAP 0.0000 both arms (2-video smoke, uninformative by design; markers are the evidence).
- No concurrent batch_evaluate processes were run at any point (serial execution; no 16 GB swap-thrash recurrence).

## 8. Decision

**REVERT / DO NOT PROMOTE (workflow decision: promote=false).** `fsd_cats` stays DEFAULT OFF at every layer — verified and pasted post-session:

- config.py L326: `fsd_cats: bool = False  # DEFAULT OFF (v182 FSD); --fsd_cats to enable`
- config.py L327 (closure comment added this session): `# v182 CLOSED 2026-08-18 (session v24): hard-88 cross-run REVERT — point_delta -0.02083, P(delta>0)=0.16, purpose population NOT converted; flag stays OFF in-tree inert.`
- config.py L478: `fsd_cats=getattr(args, "fsd_cats", False),`
- run.py L381: `parser.set_defaults(fsd_cats=False)` (with --fsd_cats / --no_fsd_cats pair at L357-380)
- batch_evaluate.py L472: `parser.set_defaults(fsd_cats=False)` (pair at L448-471)

Runtime verification (this session): `DiscoveryConfig().fsd_cats = False`, `VADConfig().discovery.fsd_cats = False`, `VADConfig.from_args(Namespace()).discovery.fsd_cats = False`, and `from_args(empty).discovery == DiscoveryConfig()` = True (flag-off byte-equivalence at the dataclass level). `py_compile` OK on config.py / run.py / batch_evaluate.py / pipeline.py / anomaly_discovery.py. The flag, CLI, markers, and `fsd_split_segment` remain in the tree inert (flag-off byte-identical code path: `fsd_split=False` short-circuits the split comprehension, anomaly_discovery.py L643-646). No backups required (no promote executed); nothing reverted (no committed default ever changed).

## 9. Family update — is scan GRANULARITY the answer, or is the recognition ceiling confirmed a third time?

**The recognition ceiling is confirmed a THIRD time.** The directed-scan brief-event residual has now been measured unreachable from all three sides of the same whole-window absolute call:

1. **Frames (v18 §10):** rejected with all 11-12 frames shown — not a frame-visibility problem.
2. **Prompt (v20, v177 TCP):** rejected/regressed under an explicit anti-averaging temporal-contrast instruction — not a prompt-wording problem.
3. **Granularity (v24, v182 FSD):** NOT converted under a 2x finer 3s window — not a window-size problem. The 3s sub-segment containing RA009's GT onset still produced an off-GT acceptance pattern; RA021 stayed empty in both arms. And the cost nearly doubled on gated videos (1.98x VLM calls) for a net-negative mAP.

Closure read: re-slicing the SAME VLM call (same prompt shape, same JSON schema, same absolute judge-a-clip task) at finer granularity does not move the brief-event recognition residual. Scan GRANULARITY as-wired (6s→3s half-split on the directed path) is CLOSED — measured-negative (point -0.0208, P(gain)=0.16), not merely non-robust. Combined with v177, the whole "steer the committed whole-segment VLM call to brief events" design space (wording AND window size) is measured closed. Do NOT re-propose: whole-segment prompt rewording (v177), finer same-call window splits (v182), or mixed variants thereof for this residual. The residual, if addressable at all, needs a DIFFERENT instrument — a supervised localizer or a change in WHAT the VLM sees (e.g., motion-cued frame selection into the call, a cropped/zoomed sub-window presentation), pre-registered and offline-validated before any eval spend. Note also the interaction ledger: v177's RA021 REF-side hit (IoU=0.6) did not reproduce in v182's REF arm (zero detections) — the residual sits inside the run-to-run VLM noise band, reinforcing that only paired instruments can adjudicate sub-0.02 effects on this population.

Updated CLOSED list (scan-side / recognition family): v16 relocation; v17 CNV (harm); v18 KFD-U (as-wired confounded; directed path already shows all frames); v19 SNH (gate); v20/v177 TCP prompt (P-gate, purpose population not converted); v21 SNH-P (offline NO-GO); v22/v181 PTE (offline NO-GO); v23/v183 CLC (offline NO-GO); **v24/v182 FSD scan granularity (hard-88 REVERT — point -0.02083, P=0.16, purpose population NOT converted, recognition ceiling measured a third time).**

Evidence: results/batch_eval_v182_fsd88/{bootstrap_ab.json,evaluation_report.json,COMMAND.txt}, results/batch_eval_v182_ref88/{evaluation_report.json,COMMAND.txt}, logs batch_eval_v182_{fsd88,ref88}.log, smokes batch_eval_v182_smoke_{on,off,off2}.
