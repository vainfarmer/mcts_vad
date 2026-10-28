# v201 — EUBC (Event-Unit Boundary Contraction) re-adjudication on the exact paired dual-arm rig

**DECISION: CLOSE / REVERT. Lever MEASURED-CLOSED.**
Paired point delta **+0.00439** (gate >= +0.008 -> FAIL), **P(delta>0) = 0.622** (gate >= 0.9 -> FAIL),
90% CI **[+0.00000, +0.02388]** (floors at zero). Not promoted. No code changed; the flag
`event_unit_boundary_contraction` remains default-OFF and in-tree inert.

This is a **clean measured negative on the mechanism's own harness with a valid, non-void instrument** —
i.e. a successful loop that closes a lever, not a void run.

---

## 1. Mechanism and why it was re-tested NOW

**EUBC** (`_eubc_*` in `pipeline.py`, terminal post-VLM slot, step 2 of the operator chain per
PROJECT_DOCUMENTATION 5) contracts an event-unit window inward toward a high-energy span when the
detection is judged materially over-long. It runs **before** the dict projection / qscore / BUE / CBT,
so its contraction is subsequently re-processed by every committed downstream operator.

**Why re-tested rather than re-invented.** v183-v200 produced 18 consecutive versions with zero
promotions, most dying at research/precheck. Meanwhile **v16 measured the SAME-CONFIG cross-run spread
at 0.02503 on hard-88** (`batch_eval_v167_full` 0.12246 vs `batch_eval_v169_ab88` 0.09743, byte-identical
flags) — roughly **3x the +0.008 promotion target**. EUBC's original verdict was a cross-run delta well
inside that band, so **it had never actually been tested**: the historical revert was not evidence.
Re-adjudicating it on the exact paired rig was therefore cheap (zero new code) and decisive.

**Files changed: NONE.** The generalized v27-RIG dual-arm already ships `eubc` as a first-class MECH
end-to-end, so nothing needed wiring. Verified byte-identical to the pre-session tree via `cmp`
against `/Users/zzl/PycharmProjects/vad/.qoder/backups/v201/`:
`pipeline.py`, `config.py`, `run.py`, `batch_evaluate.py`, `models.py` all **UNCHANGED**.

Pre-existing load-bearing sites (read-only this loop):
- `config.py:432` — `_AB_ABLATE_FAMILIES` entry `'eubc': ('event_unit_boundary_contraction','eubc_')`
- `config.py:449` — `_ab_ablate_scoring`
- `pipeline.py:2456-2465` — single pre-ECE deepcopy snapshot
- `pipeline.py:2997-3155` — `_ab_ablate_arm_off_replay` (EUBC replay step at 3095-3096)
- `pipeline.py:2867-2951` — `[AB-FIRE]` / `[AB-COLLISION]` / `[AB-VID]` emission
- `pipeline.py:2979-2995` + `1979` + `2252` — early-return marker paths (FM17)
- `models.py:187,201,243-249` — `ab_arm_off_boundaries` / `ab_arm_off_confidences` field decl + JSON write

---

## 2. The instrument (exact paired, zero VLM noise)

EUBC is deterministic post-VLM processing, so both arms are computable inside **one** eval
(PROJECT_DOCUMENTATION 8.3.2): arm-OFF replays the remaining committed chain from a snapshot taken
before EUBC; arm-ON is the run's real output. **Same videos, same VLM sampling, same detection count;
the arms differ by exactly one function call.** Confirmed empirically —
`cfg_changed=event_unit_boundary_contraction` on **88/88** `[AB-VID]` lines, i.e. the only field
differing between arms is the EUBC flag while `qscore_rerank` / `bue_enabled` / `cbt_enabled` /
`cbt_recal` stay committed-ON in both arms. **VLM noise is genuinely zero.**

Adjudication command (tool untouched, per hard constraint):

```
python3.14 video_anomaly_detection/iteration_logs/analysis_bootstrap_ab.py \
  --new results/batch_eval_v201_eubc88 \
  --paired-json ab_arm_off_boundaries --min-delta 0.008 --min-p 0.9
```

**Dual-arm applicability:** EUBC is a verified-compatible mechanism (0 VLM refs). It is a pure boundary
operator, so `paired_field = ab_arm_off_boundaries` is the correct field. (`ab_arm_off_confidences` was
also written index-aligned at full precision — see the robustness cross-check in 7.)

**Arm-independence proof (in-place mutation hazard).** `_bue_boundary_union_extension` /
`_cbt_category_boundary_tighten` mutate `a["t_start"]`/`a["t_end"]` in place, so aliasing would silently
produce plausible numbers. Proven both directions in the pre-run self-test
(`results/batch_eval_v201_eubc_selftest/`, 12 videos, real completed smoke):
- 11/12 zero-fire videos: arm-OFF **byte-identical** to final in both boundaries and confidences —
  the strong test, since those finals are products of qscore+BUE+CBT.
- 1/12 firing video `Shooting002_x264`: arm-ON final 32.5-37.5 (dur 5.0) vs arm-OFF final 28.0-39.0
  (dur 11.0) — no dict aliasing possible.

---

## 3. Harness and WHY this harness

**hard-88** = `--categories Shooting,RoadAccidents,Shoplifting,Explosion` -> Explosion 21 /
RoadAccidents 23 / Shooting 23 / Shoplifting 21 = **88 videos**.

EUBC fires on **any anomaly detection** whose window is judged materially over-long, so it is live on
every one of these 88 videos — and it did fire (5). hard-88 is the designated harness for EUBC per the
harness-selection rule. It is *not* a mechanism like CBT-ALL (fires only on the 9 non-hard categories) or
SNH (fires only on the 150 Normal videos), both of which would return an exact-zero VOID on hard-88.
**This loop is NOT void.**

Eval flags actually run:
```
--dataset_dir /Users/zzl/Movies/ucf-crime
--output_dir results/batch_eval_v201_eubc88 --concurrency 2
--categories Shooting,RoadAccidents,Shoplifting,Explosion
--event_unit_boundary_contraction --ab_ablate eubc
```

**PATH CAVEAT for future readers.** The launch `COMMAND.txt` under `results/batch_eval_v201_ab/`
contains a **duplicated `--output_dir`**; argparse keeps the last one, so every per-video JSON and the
report landed in **`results/batch_eval_v201_eubc88/`**. `results/batch_eval_v201_ab/` holds only
`COMMAND.txt`. Authoritative artifacts:
- report: `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_eubc88/evaluation_report.json`
- log: `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_eubc88/run.log` (mirrored at
  `results/batch_eval_v201_ab.log`; both grep 88 markers, they agree)
- bootstrap: `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_eubc88/bootstrap_ab.json`

---

## 4. COVERAGE GATE (PROJECT_DOCUMENTATION 8.4) — **PASS**

`evaluation_report.json` carries no completeness field and `evaluate()` drops videos without a
prediction **together with their GT**, which inflates mAP (v26: 24/290 failures reported 0.17673 vs a
true 0.16682, +0.00991 — enough to flip a -0.0057 regression into a +0.0042 "gain"). Asserted before
adjudicating:

| check | value | verdict |
|---|---|---|
| `timing.failed` | **0** | PASS |
| `timing.succeeded` | **88** | PASS |
| `timing.skipped` | 0 | PASS |
| `num_videos` (report) | **88** = expected harness size | PASS |
| `grep -cF '[AB-VID] vid='` | **88** = num_videos, 1:1 (FM5/FM21) | PASS |
| per-video JSONs carrying `ab_arm_off_boundaries` | **88/88**, `bad_videos=[]`, `len(field)==len(anomalies)` on all 88 (FM20) | PASS |
| tool `common videos` | **88** | PASS |
| process exit | normal, no traceback, no resume needed | PASS |

Wall clock 11528.95 s (3.20 h), 261.61 s/video at `--concurrency 2`. Exactly one eval ran at a time.

---

## 5. Fired videos — the mechanism genuinely fired, and was largely absorbed downstream

| quantity | value |
|---|---|
| EUBC contractions (`[EUBC] ... FIRED`) | **23** |
| distinct videos with >=1 contraction | **22 / 88 (25.0%)** |
| videos whose FINAL output differs between arms (`fired=1`) | **9 / 88** |
| detections whose FINAL **window** differs | **6 / 133 (4.5%)** |
| detections whose FINAL **confidence** differs | 10 / 133 |
| `[AB-FIRE]` detection-level lines (window union conf changed) | 14 |
| `[AB-ERR]` / `[AB-COLLISION]` | **0 / 0** |

**Key structural finding: the committed BUE->CBT chain absorbs most EUBC contractions.** 23 contractions
fired, but only **6** still differ from the pre-EUBC snapshot in the final window — **17/23 were
re-expanded back to pre-EUBC extent** by downstream BUE growth and CBT's per-category cap (13.5 s for
Shooting, 4.5 RoadAccidents, 13 Explosion, 9 Shoplifting), which acts as a strong attractor. Traced
example (`Shooting019`, self-test): EUBC 37.5-48.5 -> 39.0-47.0 FIRED, BUE grew it to 38.0-54.5, CBT
clamped to 39.00-52.00 — and the no-EUBC arm lands on **39.00-52.00 too**, so `n_win_changed=0`.

Net-effective population is therefore **6/133 detections (4.5%)**, roughly an order of magnitude smaller
than the oracle's addressable-population assumption. That, not the point estimate alone, is why this
lever is closed.

Secondary finding on the oracle's prescribed remedy: over the self-test's 19 detections the no-op tally
was **14 materiality (74%) / 2 min_segment (10.5%) / 3 FIRED / 0 flat/cat/curve/signal**. `min_segment`
is **not** the binding constraint — EUBC's own `eubc_materiality = 0.75` gate is, followed by downstream
absorption. A min_segment relaxation could unlock at most 2/19 extra candidates. It was **not**
implemented (the brief forbids modifying the mechanism, and it would be a second change under
one-mechanism-per-loop).

---

## 6. Result — absolute mAP for BOTH arms (FM18), not only the delta

Tool output verbatim, `mode=paired-json[ab_arm_off_boundaries]`, `common videos=88`,
`133 index-aligned pre-mechanism rows, 6 detections whose window actually changed`:

| arm | mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|---:|
| **REF (arm-OFF, EUBC absent)** | **0.11429** | 0.19938 | 0.12076 | 0.02273 |
| **NEW (arm-ON, EUBC active)** | **0.11867** | 0.20635 | 0.12695 | 0.02273 |
| delta | **+0.00439** | +0.00696 | +0.00619 | **+0.00000** |

Video-level bootstrap (1000 resamples, n = 88 videos):
**mean +0.00661, 90% CI [+0.00000, +0.02388], P(delta>0) = 0.622, sd 0.00797.**

```
GATE  point_delta >= 0.008: False (+0.00439)   P(delta>0) >= 0.9: False (0.622)
VERDICT: REVERT
```

Note **AP@0.7 delta is exactly 0.00000** — the contraction bought nothing at the strict IoU tier, which
is precisely where a boundary-tightening operator should pay off if it were working.

### Per-category paired deltas (independently recomputed read-only via `evaluate.compute_temporal_map`, fps=30, 88 common; exact match to the adjudication)

| category | n | arm-OFF mAP | arm-ON mAP | delta |
|---|---:|---:|---:|---:|
| Explosion | 21 | 0.14311 | 0.14311 | **+0.00000** |
| RoadAccidents | 23 | 0.17675 | 0.17675 | **+0.00000** |
| Shooting | 23 | 0.10710 | 0.12833 | **+0.02123** |
| Shoplifting | 21 | 0.06703 | 0.06703 | **+0.00000** |

**The entire pooled effect is one category, carried by ~3 videos** (`Shooting002`, `Shooting026`,
`Shooting034`). The other three categories are **bit-identical in AP**. `Explosion004` and
`Shoplifting022` had window changes that produced no AP movement at all. A pooled +0.00439 resting on 3
videos in 1 of 4 categories is exactly the profile the 0.02503 noise floor was discovered to punish.

### Count-neutrality (FM2, count-frozen) — **PASS exactly**
133 detections in **both** arms (1.5114 / video). `ndet == armoff_len` on **88/88** `[AB-VID]` lines,
`[AB-ERR]` count-mismatch = 0. EUBC changes boundaries only, never the detection count.

---

## 7. Honest caveats

1. **`ab_arm_off_boundaries` is the carried-forward pre-mechanism snapshot, not a full re-run OFF arm.**
   `restore_pre_json` (`analysis_bootstrap_ab.py:110-150`) restores boundaries only and keeps arm-ON
   confidences. Since qscore sits downstream of EUBC, that is in principle a *partial* restore.
   **Measured impact: negligible.** Restoring boundaries **and** `ab_arm_off_confidences` together gives
   mAP OFF 0.11425 / ON 0.11867 -> **delta +0.00443** (vs +0.00439 boundary-only, difference 0.00004).
   The caveat is real but does not bite; the verdict is unchanged under either restore.
   (The pre-run self-test saw `n_conf_changed=0` on all 12 videos; at full hard-88 scale 10/133
   detections do shift confidence, so the self-test's "conf never moves" claim was optimistic —
   recorded here so no future session repeats it.)
2. **Downstream reaction is modeled for the ON arm only.** `Explosion004`: EUBC 2.5-10.5 -> 3.5-7.5, ON
   then lands 3.5-9.5 after downstream, while OFF stays 2.5-10.5. So "including downstream interactions"
   holds for arm-ON; arm-OFF is a replay approximation. This is the accepted semantics of the v27 rig.
3. **The tool's printed line `mAP >= floor 0.15132: False` is NOT a regression signal.** That floor is a
   full-290 number; this is a 4-category 88-video subset — apples-to-oranges. The verdict rests on
   `point_delta` and `P(delta>0)` alone.
4. **No full-290 dual-arm confirm was run**, and correctly so: a full-290 confirm is required only
   *after* a KEEP. This loop is a CLOSE.

---

## 8. Decision and closure state

**CLOSE / REVERT.** Gate is a conjunction and both legs fail: +0.00439 < +0.008 and P = 0.622 < 0.9,
with a 90% CI whose lower bound is exactly zero.

**Lever marked MEASURED-CLOSED.** Future sessions must not re-litigate EUBC. State the reason precisely:
the instrument was an **exact paired end-to-end A/B with zero VLM noise** on the mechanism's own harness
(88/88, failed = 0, 133 detections both arms, arms differing by exactly one function call), **not a noisy
cross-run proxy**. Unlike the pre-v16 verdict — which sat inside the 0.02503 same-config spread and was
therefore no evidence at all — this measurement is decisive at the loop's resolution. The mechanistic
reason is also now known and is not a tuning problem: **17 of 23 contractions are re-expanded by the
committed BUE->CBT chain**, whose per-category cap is a strong attractor, leaving a net-effective
population of 6/133 detections; and the dominant no-op gate is EUBC's own `eubc_materiality=0.75`, not
`min_segment`, so the oracle's prescribed relaxation is aimed at the wrong gate.

**Committed defaults verified untouched** (`python3.14`, read-only):
`ScoringConfig()` -> `qscore_rerank=True`, `bue_enabled=True`, `cbt_enabled=True`, `cbt_recal=True`,
`classify_skip_class_agnostic=True`; unshipped `ccr_enabled / cex_extend / cgr_enabled / cnv_veto /
event_continuation_extension / event_unit_boundary_contraction / snh_veto` all **False**;
`ab_ablate=''`. Flag-off byte-identity: `VADConfig.from_args(Namespace()).scoring == ScoringConfig()`
-> **True**. `evaluate.py` and `analysis_bootstrap_ab.py` untouched. Nothing to restore — zero edits were
made this loop, `cmp` clean against `.qoder/backups/v201/` on all five files.

**Corollary validated (positive result worth keeping):** the committed **BUE-then-CBT ordering with
`cbt_recal` per-category caps is robust to an upstream contraction** — it re-derives essentially the same
final boundary whether or not EUBC pre-shrinks the window. That is a stability property of the shipped
chain, measured, not assumed.

---

## 9. Artifacts

- `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_eubc88/` — 88 per-video JSONs (all with
  `ab_arm_off_boundaries` + `ab_arm_off_confidences`), `run.log`, `evaluation_report.json`,
  `bootstrap_ab.json`
- `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_ab.log` — stdout mirror (88 markers)
- `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_ab/COMMAND.txt` — launch command (duplicated
  `--output_dir`; see 3)
- `/Users/zzl/PycharmProjects/vad/results/batch_eval_v201_eubc_selftest/` +
  `/Users/zzl/PycharmProjects/vad/results/v201_eubc_selftest.out` — 12-video arm-independence self-test
- `/Users/zzl/PycharmProjects/vad/.qoder/backups/v201/` — pre-session copies of the 5 core files
  (`cmp`-verified identical to the working tree; no restore required)
- `/Users/zzl/PycharmProjects/vad/video_anomaly_detection/iteration_logs/summary_v201_eubc_20260825.md`
  — this file
