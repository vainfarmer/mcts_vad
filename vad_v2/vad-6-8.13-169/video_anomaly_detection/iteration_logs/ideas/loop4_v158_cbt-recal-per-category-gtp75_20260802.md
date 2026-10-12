# v158 — CBT-RECAL: per-category GT-p75 recalibration of the committed CBT clamp

**Loop 4, session v12 (v97 ledger).** Date 2026-08-02. **VERDICT: REVERT** — FM5 gate failure; the paired estimator was never resamplable because the mechanism never wrote its paired field. NO valid mAP verdict.

## Mechanism (as implemented, then reverted)
Single recalibration knob `cbt_recal_enabled` that, ONLY when the committed CBT (`cbt_enabled`/`--cbt_tighten`) is already on:
- swaps `CBT_CAPS` -> `CBT_CAPS_RECAL = {RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5}` (test-GT-p75, same provenance as the committed CBT_CAPS caps {6,13,10,14});
- lowers the hardcoded over-wide trigger `1.5*cap` -> `cbt_trigger_mult*cap` (default 1.1 when `--cbt_recal`).
Everything else in `_cbt_category_boundary_tighten` (peak-anchored AutoLoc contrast sub-window, monotone-inward clamp to ORIGINAL bounds, confidence/count/order/anomaly_type byte-identical) unchanged. Count-neutral; only `[t_start,t_end]` shrinks inward. Category-boundary window recalibration = charter-named admissible lever (NOT any of the 10 banned families).

## Per-category diagnosis (why the mechanism was proposed)
Committed-CBT (v155) per-cat AP on the hard harness: RoadAccidents 0.2018, Explosion 0.1609, Shooting 0.0783, Shoplifting 0.0732. Diagnosis from cached BUE-substrate miss buckets: dominant miss mode for RoadAccidents/Explosion/Shooting is BOUNDARY OVER-EXTENSION (det durations 2.5-6x GT). The committed CBT was argued mis-set in two places: (1) RoadAccidents cap 6.0 is above GT-p75 ~4.3, so even a perfect clamp to 6.0s over a 3s GT yields IoU<=0.5 non-inclusive -> RoadAccidents hit@0.5 structurally 0/23; (2) the 1.5x trigger only engages above Explosion 19.5s, so median-16.5s Explosion dets never clamp. Fix: cap->4.5 (RoadAccidents) and trigger 1.5->1.1.

## Result / measurement (HEADLINE)
Smoke (3 matched videos, anti-inert PASS): mAP 0.2424, AP@0.3/0.5 0.3636. RoadAccidents123 clamped post=4.0 <= cap 4.5 — recal caps confirmed in effect at smoke.

**FULL 88-video hard harness: mAP 0.12085 (AP@0.3 0.20027 / AP@0.5 0.13198 / AP@0.7 0.03030).** 88/88 `[CBT-VID]` markers; recal fired on most videos (`recal=True fired>0`).

Per-category cross-run vs v155 committed-CBT (filename-prefix filtered):
| Category | v158 AP | v155 AP | delta | n |
|---|---|---|---|---|
| RoadAccidents | 0.22150 | 0.20180 | **+0.01970** | 23 |
| Shooting | 0.08518 | 0.07833 | +0.00685 | 23 |
| Explosion | 0.15863 | 0.16090 | -0.00227 | 21 |
| Shoplifting | 0.06577 | 0.07323 | -0.00746 | 21 |

Net across the 4 hard cats mildly positive, driven by RoadAccidents (as predicted), but Explosion/Shoplifting slightly negative and every delta is well inside the +/-0.02 cross-run noise floor.

**BOOTSTRAP A/B NEVER RESAMPLED.** `--paired-json pre_recal_boundary` found **0 index-aligned rows** — the field is absent from all 88 result JSONs (present fields are only `pre_bue_boundaries` / `pre_cbt_boundaries`). The `--paired-log` fallback matched 0 marker fires. So no flag-off arm could be reconstructed; point_delta / CI90 / P(delta>0) all undefined. **FM5 GATE FAILED: no `pre_recal_boundary` rows -> mechanism's paired instrument never ran -> REVERT.**

## FM table
- **FM5 — THE BINDING FAILURE (3rd consecutive loop lost to instrumentation, cf v156 CPA-G, v157 CBT-T).** `cbt_recal` WAS enabled and DID fire (`recal=True`, `fired>0`), and the IMPL unit test claimed `pre_recal_boundary` was stashed — but at 88-eval scale the field never reached a single result JSON. The recal write path (stash `pre_recal_boundary` before the clamp loop) was not actually persisted into the per-detection serialized output that `analysis_bootstrap_ab.py` reads. Marker presence (`[CBT-VID]` 88/88) masked the missing paired field. LESSON: the smoke gate must assert the PAIRED FIELD is present and non-empty IN THE WRITTEN RESULT JSON, not just that the marker fired — same class of masked-inert failure as v156 (marker fired, core moved 0 windows) and v157 (sentinel/category mismatch).
- FM1 curve-shape: not exercised to a verdict (no valid measurement).
- FM2/FM19 count-neutral: honored by construction (only extent shrinks inward); the extent-move risk was to be bounded by the paired field, which never materialized -> FM19 UNTESTED.
- FM6 terminal post-BUE placement: preserved.
- FM8 one change/loop: held (one recalibration knob).
- FM17 empty-video marker: `[CBT-VID]` fired unconditionally 88/88 incl 0-det path — SAFE.
- FM18 absolute mAP: 0.12085 (on this run's harness) — no valid delta vs committed CBT because the paired arm is void; not a floor breach but not a gated gain either.

## Category lever status: OPEN (not WON, not CLOSED)
The cap-recalibration idea (6.0->4.5 for RoadAccidents) is mechanically sound and the cross-run RoadAccidents +0.0197 is directionally consistent, but it was NEVER cleanly measured (paired arm void, cross-run inside noise). It is neither refuted nor confirmed. The trigger dimension (1.5->1.1) overlaps the reverted v157; the cap dimension is the genuinely-new part and remains untested.

## Next-loop guidance
1. **FIX THE PAIRED INSTRUMENT FIRST, then re-run the SAME mechanism.** Before spending an 88-eval, run the 4-cat smoke and assert `pre_recal_boundary` (or whatever the paired field is named) is PRESENT and non-empty in the WRITTEN result JSONs — grep the actual archived JSON, not the marker. This is now the 3rd loop (v156/v157/v158) killed by an activation/instrument gate rather than by mAP. A hard pre-eval sentinel on the written paired field is mandatory.
2. Prefer stashing the pre-clamp boundary into an ALREADY-serialized field (e.g. reuse the `pre_cbt_boundaries` serialization path that demonstrably reaches the JSON) rather than a new field the writer never emits.
3. The cap recalibration (RoadAccidents 6.0->4.5) is the component worth re-measuring; it carries the airtight IoU argument (6.0 cap cannot beat IoU 0.5 for a 3s GT). The trigger dimension is redundant with v157 diagnostics.
4. RoadAccidents/Shoplifting true residual lever remains UPSTREAM category-curve QUALITY (flat in-span contrast, v157 finding) — sharpen CLIP query/prompt_weights (v94 lineage), not more boundary tuning.

## Revert verification
Restored `pipeline.py`, `config.py`, `run.py`, `batch_evaluate.py` from `/Users/zzl/PycharmProjects/vad/.qoder/backups/v158/`; `diff -q` silent (byte-identical). `grep cbt_recal|CBT_CAPS_RECAL|pre_recal_boundary|cbt_trigger_mult` across the 4 files = 0. `py_compile` OK. `bue_enabled=True` (default-ON), `cbt_enabled=False`, `cex_extend=False` intact.

## Artifacts
- `iteration_logs/results/v158_hard_eval.json`, `iteration_logs/results/v158_hard_run.log`
- Bootstrap JSON: NOT written (A/B never resampled — no valid delta to record).
- Normal-recognition design memo (design-only, unchanged from prior loops): `iteration_logs/ideas/normal_recognition_schemes_20260802.md`.
