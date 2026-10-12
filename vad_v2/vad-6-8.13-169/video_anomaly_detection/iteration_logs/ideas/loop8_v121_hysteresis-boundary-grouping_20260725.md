# Loop 8 — v121 HYBG (Hysteresis Boundary Grouping)

Date: 2026-07-25
Decision: **REVERT** (never legitimately launched — STEP-0 offline harness not run; live launch gate refused)
Baseline unchanged: v114 Q-on, full mAP = 0.15013 (evaluation_report actual; proposal's 0.15632 was stale)

---

## Approved proposal (summary)

**Title.** HYBG — Hysteresis Boundary Grouping: a harness-first, count-frozen, per-detection boundary-EXPANSION stage that re-groups over-shrunk detections along the UN-FLOODED fused-curve exterior via dual-threshold hysteresis with internal-dip bridging, adjudicated deterministically OFFLINE on v114's real detections (with a mandatory un-flooded-exterior FIDELITY gate) before any live A/B (single controlled change, stacked on committed v111 Q-score).

**Mechanism.** Read-only, post-hoc, per-detection boundary-PLACEMENT stage `_hysteresis_boundary(...)` inserted as the ABSOLUTE-LAST transform in `detect()` — after the committed v111 Q-score block (pipeline.py ~L1218-1221) and before `DetectionResult` (~L1223). Fusion, confidence, count, ordering-inputs, NMS, merge stay byte-identical to the Q-on baseline; HYBG mutates ONLY each detection's `[t_start, t_end]`. Canny/TAG-style dual-threshold hysteresis on the UN-FLOODED exterior of the cached fused curve `frame_scores`:
- HIGH threshold H = v51 half-max LEVEL `(peak_val+baseline)/2` (pipeline.py L2040), baseline = percentile(smoothed, 20) (L2039).
- LOW (hysteresis exit) threshold L = baseline.
- Walk start leftward from `lo` while `fs>=L`, tolerating contiguous below-L runs shorter than `CATEGORY_DURATION_PRIORS[cat]['min_segment']` (dip-bridge); stop when a below-L run exceeds min_segment or the exterior never re-crosses H (no-actionness guard → keep v51 boundary). Symmetric rightward.
- Grid-snap via searchsorted; reject-cap: regrouped duration > category max → keep v51 boundary. Flat/degenerate/Normal/no-edge/no-re-cross-H → inert.
- Exactly ONE arm promoted live by `hysteresis_mode` AFTER the offline harness adjudicates {expand, expand_nodip, (category live-only)}. CVSA NOT bundled — single controlled change.

**Rationale.** Under greedy IoU matching, mAP (esp. AP@0.7=0.1096) is governed by localization IoU, not by re-ordering already-committed detections. Confirmed root cause: the pipeline's SHRINK-ONLY half-max correction (`clip_score_curve_correction`, v52, frame_scorer.py L120 "v52起只收缩不扩张") over-cuts multi-phase long events at internal actionness dips, populating the IoU-0.1-0.5 "boundary-imprecise / long-event-shrunk" bucket reachable ONLY by moving endpoints (FM7). HYBG = training-free TAG/watershed grouping (SSN, Zhao ICCV2017) + Canny hysteresis, re-annexing later phases that live in the un-flooded exterior. Distinct from v107 ECE (+0.002): adds the LOW-exit threshold + dip-bridge on the motion-inclusive fused curve. Ledger-mandated pivot off the now-tapped confidence-reorder family (v107/v115/v118/v119).

**Expected gain.** Offline on v114 real detections: +0.005 to +0.015 mAP concentrated in AP@0.5/AP@0.7 if dip-bridging recovers over-shrunk multi-phase events; tempered prior "likely borderline" (only novel lever vs ECE is LOW-exit + dip-bridge). Floor near-zero by construction (reject-cap + no-re-cross-H guard). Latency negligible (<10 ms/video numpy walk over cached curve).

---

## C's feedback (reviewer)

APPROVE with mandated changes. Verified every load-bearing claim against the tree: flood is inside-windows-only (frame_scorer L110-111), shrink-only correction confirmed (L120), half-max/baseline formulae correct (L2039-2040), insertion point correct, and ECE reads the un-flooded CATEGORY curve whereas HYBG reads the FUSED `frame_scores` — genuinely a different curve than the +0.002 ECE. Right pivot: boundary placement (FM7). Evidence class is the trusted one (deterministic transform on the run's own cached fused curve → live re-derives byte-identically, FM6 does not apply). FM1/FM2 immune by construction.

THE ONE SUBSTANTIVE FLAW: the un-flooded-exterior fidelity gate was mis-specified. `frame_scores` is flooded inside windows AND THEN Gaussian-smoothed (frame_scorer L114-116, sigma=max(1.0,1.5*fps_est)); smoothing spreads the interior flood OUTWARD ~sigma frames, so the walk's first frames read the flood-blur tail, not genuine pre-flood evidence — a systematic outward-expansion bias. STEP-0 GATE(2) as written (`assert exterior != max(curve,conf)`) is a TRIVIAL PASS under smoothing and does NOT detect the contamination. Not fatal (binding protection = the deterministic +0.005 offline kill-gate at zero live cost), but the gate must be fixed so adjudication is honest. Expected gain borderline; the offline bake-off must be able to tell real fused-curve signal from flood-blur.

---

## Implementation notes (D1)

Fully implemented, default-OFF (byte-identical to v114 Q-on tree). Backups at `.qoder/backups/v121/` (4 files: config.py, pipeline.py, batch_evaluate.py, run.py; no models.py change).

C's THREE mandated changes implemented:
1. **FM6 guard-band fidelity gate** — hysteresis walk begins `ceil(2*sigma)` frames OUTSIDE `[lo,hi]` (sigma = frame_scorer smoothing scale) so it never reads the Gaussian flood-blur tail. In BOTH the live method and harness; harness GATE(2) asserts the walk-outside-window property on ≥3 detections and REVERTS on violation. Replaces the trivial no-op check.
2. **Category control arm** — `mode='category'` reads `get_category_scores` (the exact un-flooded curve ECE reads). LIMITATION: cached v114 JSONs carry no category curve (no schema change, per v120 clean-revert discipline), so this arm is NOT re-derivable offline; promotable LIVE only via `--hysteresis_mode category`. Harness adjudicates fused arms {expand, expand_nodip}.
3. **Launch gate** — both legs pass `--qscore_rerank`; launcher asserts `analysis_v121_hysteresis_precheck.json` exists WITH a non-null `promoted_arm` AND `--hysteresis_mode` matches it (batch_evaluate.py, `sys.exit(2)` otherwise); post-eval `[HYBG-GATE]` greps `[HYBG] summary:` markers (count < succeeded ⇒ FM5 revert).

Harness `analysis_v121_hysteresis_precheck.py`: clones v111 structure; loads v114 real detections; GATE(1) anchor+round-trip fidelity (self-anchors to v114 dir's own mAP — proposal 0.15632 stale, actual 0.15013); GATE(2) guard-band fidelity; pre-registered promotion iff ΔmAP≥+0.005 AND AP@0.7 non-regression AND per-video count unchanged. VERIFIED: AST+import OK on 5 files; --help shows both flags; default-OFF byte-identity confirmed; pure hysteresis logic smoke-tested on synthetic curve. Did NOT run any evaluation (offline harness left for eval agents).

---

## Eval flags + activation marker

- Treatment CLI: `--qscore_rerank --hysteresis_boundary --hysteresis_mode expand`
- Control leg: `--qscore_rerank` only
- Activation marker (per-video INFO): `[HYBG] summary: vid=<name> n=<ndet> regrouped=<k> mode=<arm>`
- Launch gate marker (failure): `[HYBG-GATE] LAUNCH: ... missing a promoted_arm`

## Marker counts

| stage | `[HYBG] summary:` markers | notes |
|---|---|---|
| smoke | 0 | launch refused by gate; process exited immediately |
| full  | n/a | skipped (gate) |

## Results

| metric | value |
|---|---|
| smoke mAP | -1 (never ran; marker_count = -1) |
| full mAP | -1 (skipped, gate) |
| AP@0.3 / AP@0.5 / AP@0.7 | -1 / -1 / -1 |
| full avg latency | -1 s |
| offline harness A/B (map_off/map_on/delta) | NOT RUN (STEP-0 harness never executed) |

Smoke log (`results/batch_eval_v121_smoke.log`) contents:
```
[INFO ] VAD batch evaluation started | output_dir=.../batch_eval_v121_smoke
[ERROR] [HYBG-GATE] LAUNCH: analysis_v121_hysteresis_precheck.json missing a promoted_arm — refusing live HYBG run (run the offline harness first).
```

## Decision + reason

**REVERT.** The mandatory STEP-0 offline harness (`analysis_v121_hysteresis_precheck.py`) was never executed, so `analysis_v121_hysteresis_precheck.json` carries no `promoted_arm`. The launch gate (correctly, as designed) refused the live smoke run with `sys.exit(2)` — no arm was legitimately adjudicated, no smoke/full metrics exist. This is the SAME harness-skip process failure as v117 and v120. The gate did its job (no wasted full-eval budget this time — the launch died immediately), but the mechanism's effect on mAP remains UN-adjudicated. Restored all 4 backup files, diff-verified byte-identical; ships flag-OFF, byte-identical to the v114 Q-on baseline. Baseline stays v114 full mAP = 0.15013.

## Lessons learned (failure modes)

- **FM5 (activation) — confirmed, upstream/by-construction.** 0 `[HYBG] summary:` markers; the launch gate refused the run before any video processed. The gate worked exactly as intended (a hard launch precondition), turning a would-be silent placeholder run (v120's failure) into an immediate, cheap, loud refusal.
- **Process failure (harness-skip) — confirmed AGAIN (3rd time: v117, v120, v121).** The single recurring blocker across the boundary/feature-rerank loops is that the offline STEP-0 harness is implemented and C-approved but never run before launch. The v121 launch gate is the correct mechanical fix — it prevents the wasted-budget variant (v120) — but the meta-lesson stands: the eval workflow MUST run the offline harness as its first action for any harness-first proposal, or the loop produces zero signal.
- **FM6 (sim-to-live gap) — NOT reached.** C's real merge-in (guard-band fidelity gate replacing the trivial `!= conf` check, defending against Gaussian flood-blur outward bias) was implemented in both the live method and harness but never exercised, since the harness never ran.
- **FM1 / FM2 — immune by construction** (only `[t_start,t_end]` move; count/fusion/confidence/NMS/merge untouched); untested but not at risk.
- **FM7 (mislocalization) — pivot on-target but effect UN-adjudicated.** Boundary-placement family remains NOT refuted; HYBG needs a clean harness re-run to determine whether fused-curve hysteresis + dip-bridging clears +0.005. The mechanism ships flag-off, fully implemented, ready for a harness-first re-attempt.
