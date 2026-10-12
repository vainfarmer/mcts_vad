# v203 — CBT-ALL: extend the CBT category cap key-set from 4 to 13 categories

**Date:** 2026-08-25
**Loop:** v203 (single mechanism: `cbt_all_cats`)
**Verdict: REVERT / lever MEASURED-CLOSED.** Not a null — an actively harmful,
statistically decisive negative on an exact paired end-to-end A/B at full-290.
Code reverted; the committed pipeline is unchanged.

---

## 1. Mechanism

`_cbt_category_boundary_tighten` (CBT, committed default-ON together with
`cbt_recal`) resolves its inward-clamp cap from the **detection's own resolved
`anomaly_type`**, looking that category up in a `caps` dict that keys only the
**4 hard categories** (`RoadAccidents 4.5 / Explosion 13.0 / Shoplifting 9.0 /
Shooting 13.5`). The other **9 anomaly categories (52 test videos)** therefore
never see the inward clamp at all.

CBT-ALL was a **one-line key-set extension** and nothing else:

```python
if cfg.cbt_all_cats:
    caps = {**CBT_ALL_EXTRA_CAPS, **caps}   # existing keys win => 4 committed caps bit-identical
```

`CBT_ALL_EXTRA_CAPS` was **derived in code** from
`CATEGORY_DURATION_PRIORS[cat]['expected_duration'][1]` (the same
"max plausible event duration" prior ECE already uses as its reject-cap), for
the 9 categories absent from `CBT_CAPS`. **No test GT was read anywhere in the
shipped path** — a GT-derived cap would have been test-fitting and would have
voided the loop. Method logic (peak anchor, AutoLoc outer/inner contrast
objective, monotone-inward clamp, `min_segment` floor, `cbt_recal_trigger=1.1`,
count-neutrality, confidence freeze) was **untouched**.

## 2. Why this was re-tested now (and not invented now)

v183-v200 = 18 consecutive versions with zero promotions, most dying at
research/precheck. The higher-yield move was to re-adjudicate operators already
in the tree whose original verdicts are **not evidence**: v16 measured the
**same-config cross-run spread at 0.02503** on hard-88 (v167_full 0.12246 vs
v169_ab88 0.09743, byte-identical flags) — roughly **3x the +0.008 target**. Any
mechanism reverted on a cross-run delta below ~0.02 was never actually tested.
CBT's category gate was *demoted, not killed*, by the roster, so it qualified.

FM13 first: `analysis_v203_cbtall_oracle.py`, rebuilt on the authoritative
`batch_eval_v26_full290` (baseline 0.16682, 0 missing anomaly videos), bounded
the ceiling at **+0.01843** (0.18525) from only **14 improvable dets**. The
method-width-constrained oracle was **also +0.01843**, i.e. the cap *width* was
never the binding constraint — the **addressable population** was (31 dets with
`dur > 1.1*cap`, 21 on anomaly videos, against CBT's own 103 fires). Ceiling
cleared the gate 2.3x, so the loop was worth an eval; at the measured
0.137-0.164 transfer factor the *expectation* was already only +0.0025-+0.0030.

## 3. Instrument and harness (and why this harness)

**Instrument:** the v27-RIG **exact paired dual-arm inside ONE eval**
(`--cbt_all_cats --ab_ablate cbt_all`). CBT is a terminal-slot operator
downstream of every VLM call, so arm-OFF is a deterministic replay of the
remaining chain on a pre-mechanism snapshot: same videos, same VLM sampling,
same detection count, arms differing by exactly one function call. The
`cbt_` family prefix resets `cbt_all_cats` to `False`, so **arm-OFF == the
committed CBT+recal calibration** with no rig edit. **Zero cross-run VLM noise
— the 0.02503 noise floor is never invoked in this adjudication.**

**Harness: full-290, and it had to be.** CBT-ALL can only fire where a
detection's resolved `anomaly_type` is one of the 9 extended categories, which
are absent from hard-88 as *video* categories. The roster's claim that hard-88
would be an *exact zero* was **wrong but in a way that only made it worse**:
4 of the 31 addressable dets sit on Shooting/RoadAccidents/Explosion videos
whose detection resolved to Vandalism/Abuse, so hard-88 would have been **~5x
under-powered rather than void** — the classic silently-voided loop. Full-290
was mandatory.

**Harness defect found and worked around (record it — it will bite again):**
`batch_evaluate.py --max_per_category` **defaults to 5**, so the documented
"full-290" command *without* that flag selects **65 videos**, not 290
(`select_videos(max_per_category=5) -> 65`, `=1000 -> 290`), and would still
emit an `evaluation_report.json` with no completeness field — the 8.4 inflation
trap in a new disguise. The run therefore carried `--max_per_category 1000`.

## 4. Self-test (4 gates, all PASS) — real smoke, not offline

`results/batch_eval_v203_smoke/` (full-290 slice, concurrency 1, 4 videos,
190.4s, succeeded=4 failed=0), verified by
`iteration_logs/analysis_v203_selftest.py`:

- **(a) marker/activation** — `[CBT-VID] vid=` on 4/4, carrying `allcats=on
  ncats=13`; `[AB-VID]` 4/4; `[AB-ERR]` 0.
- **(b) paired field len-match** — `ab_arm_off_boundaries` in all 4 JSONs,
  `len == len(anomalies)` everywhere.
- **(c) arm independence** (the in-place-mutation hazard:
  `_cbt_category_boundary_tighten` mutates `a["t_start"]/a["t_end"]`) — fires
  per video `(2,2,0,1)` **exactly equal** to window-diffs per video `(2,2,0,1)`;
  the 0-fire video `Fighting018` is byte-identical across arms.
- **(d) FM2 count/confidence freeze** — `armoff_len == ndet` on every video and
  `conf_diff == 0` on every det.

`[AB-COLLISION]=2` appeared (two Abuse dets share a rounded final window). This
only breaks `--paired-log`; `--paired-json` is index-aligned and immune, which
is the mode used for adjudication.

## 5. Coverage gate (PROJECT_DOCUMENTATION 8.4) — PASS

From `results/batch_eval_v203_cbtall_full290/evaluation_report.json`:

```
timing: num_videos 290, succeeded 290, failed 0, skipped 0
wall_clock 19448.1s (5.40h), avg_per_video 134.01s, concurrency 2
```

`failed == 0` and `num_videos == 290 == expected harness size` -> **no resume
needed, no mAP inflation**. FM20 verified by grep, not by claim: **290/290
per-video JSONs carry `ab_arm_off_boundaries`**, 339 index-aligned rows, zero
length-mismatch VOIDs. Marker counts: `[CBT-VID] vid=` **290**, `[AB-VID]
mech=cbt_all` **290** (the 77 lines without the `allfired=` suffix are the FM17
`early return: no candidates` 0-detection path — 213 + 77 = 290).

## 6. Firing population — the loop is LIVE, not void

From `allfired=` in the run log: **18 videos fired once, 5 fired twice ->
23/290 videos, 28 of 339 detections had their window actually changed.** This
matches the adjudicator's 28 changed windows independently.

## 7. Result

Adjudicated with the untouched tool:

```
python3.14 iteration_logs/analysis_bootstrap_ab.py \
  --new results/batch_eval_v203_cbtall_full290 \
  --paired-json ab_arm_off_boundaries --min-delta 0.008 --min-p 0.9
```

| | arm-OFF (committed CBT) | arm-ON (CBT-ALL) | delta |
|---|---|---|---|
| **mAP** | **0.17053** | **0.15599** | **-0.01454** |
| AP@0.3 | 0.26144 | 0.23273 | -0.02871 |
| AP@0.5 | 0.14678 | 0.13801 | -0.00877 |
| AP@0.7 | 0.10337 | 0.09723 | -0.00614 |

- **point_delta = -0.014542**
- **90% CI = [-0.048528, -0.003862]** — entirely below zero
- **P(delta>0) = 0.011** — 98.9% of 1000 video-level resamples show harm
- n_videos = 290, mode `paired-json[ab_arm_off_boundaries]`, tool verdict REVERT

Both gate legs fail **in the adverse direction** (needed `>= +0.008` and
`P >= 0.9`). Every IoU threshold regressed, so this is not a threshold artefact.
Arm-ON mAP stays above the 0.15132 floor only because the operator's damage is
bounded — that is not a defense.

### Count-neutrality: YES

**339 detections in both arms, mean 1.169/video identical.** The operator only
moves boundaries; it never adds or drops a detection. The entire loss is
therefore **boundary misplacement**, not a recall/precision trade. A
28-detection edit cost **1.45 mAP points**.

### Per-category mAP, arm-OFF -> arm-ON

| category | n | OFF | ON | delta |
|---|---|---|---|---|
| Stealing | 5 | 0.1818 | 0.0000 | **-0.1818** |
| Fighting | 5 | 0.1083 | 0.0000 | **-0.1083** |
| Vandalism | 5 | 0.2284 | 0.1281 | **-0.1003** |
| Burglary | 13 | 0.2021 | 0.1103 | **-0.0918** |
| Robbery | 5 | 0.0303 | 0.0909 | +0.0606 |
| Abuse | 2 | — | — | 0.0000 |
| Arrest | 5 | 0.8182 | 0.8182 | 0.0000 |
| Arson | 9 | 0.3157 | 0.3157 | 0.0000 |
| Assault | 3 | 0.3030 | 0.3030 | 0.0000 |
| Explosion | 21 | 0.1380 | 0.1380 | 0.0000 |
| Normal | 150 | 0.0000 | 0.0000 | 0.0000 |
| RoadAccidents | 23 | 0.2215 | 0.2215 | 0.0000 |
| Shooting | 23 | 0.0790 | 0.0790 | 0.0000 |
| Shoplifting | 21 | 0.0659 | 0.0659 | 0.0000 |

4 categories harmed vs 1 helped. The 4 hard categories are **exactly 0.0000**,
confirming the committed caps were preserved bit-identically as designed. Abuse,
Arson and Explosion contain *fired* videos yet show delta 0 — those edits did
not cross an IoU threshold — so the measured harm comes from a **narrower set
than the 23 fired videos**, concentrated in small-n categories where the
operator zeroed or halved AP outright.

## 8. Root cause (why the +0.01843 oracle headroom is unreachable by this method)

The oracle can pick **wide inward sub-windows**; the shipped AutoLoc contrast
objective on the 9 extended categories **systematically picks the narrowest
admissible window** (the 2.0s `min_segment` floor). Per-det IoU vs GT,
arm-OFF -> arm-ON from the smoke:

- `Fighting047` det1: 37.0-59.0s IoU **0.405** (a TP at 0.3) -> 40.5-42.5s IoU
  **0.037** — TP destroyed; that single det is the whole smoke-scale loss.
- `Abuse028`: 0.5-35.0s IoU 0.072 -> 32.5-34.5s IoU **0.000** (peak found at the
  wrong end of the video).
- `Abuse030`: 0.210 -> 0.216 (negligible).

The 4 categories where CBT works are **short bursts** (caps 4.5-13.5s). The 9
extended categories have **long GT events** (Fighting047 GT = 54.3s,
Fighting003 = 42.8s, plus Arson/Assault/Arrest), so an inward clamp toward a 2s
contrast peak is **directionally wrong** there. The pre-run note predicted a
negative, not merely a sub-gate, result; the 290-video run confirmed it.

## 9. Decision and disposition

**REVERT. Lever MEASURED-CLOSED — do not re-litigate.** The instrument was an
**exact paired end-to-end A/B inside a single 290-video run** (arms differ by
one function call, identical VLM sampling, count-frozen, confidence-frozen),
**not a noisy cross-run proxy**. The 0.02503 cross-run noise floor that
invalidated the earlier CBT-scope verdict does not apply here and cannot be
invoked to reopen this one. Combined with the FM13 oracle finding that the cap
*width* is not the binding constraint, **extending CBT's category key-set is
closed from both sides**: the addressable population is tiny *and* the clamp
direction is wrong for long-event categories.

**Code reverted** from `.qoder/backups/v203/`; `diff -q` clean on all four
files. Committed defaults re-verified after revert:

```
VADConfig.from_args(Namespace()).scoring == ScoringConfig()  -> True
qscore_rerank=True  bue_enabled=True  bue_closure_gate=True
cbt_enabled=True    cbt_recal=True    cbt_recal_trigger=1.1
cbt_recal_caps={RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5}
classify_skip_class_agnostic=True
ccr_enabled / cex_extend / cgr_enabled / cnv_veto /
event_continuation_extension / event_unit_boundary_contraction /
snh_veto = False    ab_ablate=''
hasattr(ScoringConfig(),'cbt_all_cats') -> False
hasattr(pipeline,'CBT_ALL_EXTRA_CAPS')  -> False
--cbt_all_cats absent from --help of both runners
import MultiAnomalyVAD OK; py_compile OK on all four files
```

`evaluate.py` and `analysis_bootstrap_ab.py` were never touched.

## 10. What a future loop would have to pre-register instead

Not a scalar retune. The failure is in the **objective**, so the only honest
follow-up is a **per-category minimum emitted width** for the long-event
categories (e.g. an `hw` floor tied to `CATEGORY_DURATION_PRIORS
expected_duration[0]` instead of the global `min_segment`). That is a change to
the *method*, not to its inputs, and belongs to its own pre-registration with
its own oracle. It was deliberately **not** smuggled into this loop.

## 11. Artifacts

- Run: `results/batch_eval_v203_cbtall_full290/` (290 per-video JSONs +
  `evaluation_report.json` + `bootstrap_ab.json`)
- Stdout log: `results/batch_eval_v203_ab.log`
- **Path warning:** the launch command duplicated
  `--dataset_dir/--output_dir/--max_per_category/--concurrency`, and argparse
  last-wins silently redirected output to
  `results/batch_eval_v203_cbtall_full290/`, leaving the nominal
  `results/batch_eval_v203_ab/` holding **only `COMMAND.txt`**. Same single run,
  same process — **not** a cross-run substitute. Any consumer hardcoding
  `batch_eval_v203_ab/evaluation_report.json` will hit a missing file.
  **De-duplicate the launch flags before the next loop** so the output dir
  matches the V_tag (this is the second consecutive loop with this defect —
  v201 hit it too).
- Smoke: `results/batch_eval_v203_smoke/` (+ `.log`)
- Analysis: `iteration_logs/analysis_v203_cbtall_oracle.py`,
  `iteration_logs/analysis_v203_selftest.py`
- Backups: `.qoder/backups/v203/` (pre-edit, restored from) and
  `.qoder/backups/v203/post_v203_snapshot/` (the reverted v203 code, retained
  so the closed lever is auditable without re-implementation)
