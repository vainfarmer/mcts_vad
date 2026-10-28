# v98 — Verified FCCS Baseline Establishment (flag-only controlled switch, zero code delta)

- **Loop:** 1
- **Version:** v98
- **Date:** 2026-07-18
- **Decision:** PENDING — mandatory log-verification gate; evals executed by later agents
- **Smoke mAP:** PENDING
- **Full mAP:** PENDING (AP@0.3 / AP@0.5 / AP@0.7 PENDING)
- **Full avg latency:** PENDING

---

## Root Cause Being Corrected (the failure this loop fixes)

The recorded "v97 FCCS" baseline **never ran FCCS**. Re-verified independently
in this session (and by C): `results/batch_eval_v97_full/run.log` contains 59
`[Directed scan]` lines and ZERO `[FCCS]` lines; `batch_eval_v97_smoke/run.log`
contains 13 and ZERO; the colocated `evaluation_report.json` files contain
mAP=0.16154893439350917 (full) and 0.15034965034965034 (smoke), exactly
matching the v97 ledger — actual run outputs, not mislabeled copies.
`[Directed scan]` is emitted only at pipeline.py:1332 inside
`_directed_llm_scan`, unreachable when the FCCS branch (pipeline.py:875-885)
fires. The wiring itself is correct and complete in the current tree (all five
points re-verified: config.py:80 defines `full_coverage_comparative_scan=False`;
config.py:153 threads `getattr(args,'full_coverage_scan',False)`;
batch_evaluate.py:189-191 defines `--full_coverage_scan`; pipeline.py:854-885
branches on it with `clip_regions=[]` verified side-effect-free;
pipeline.py:692-705 passes all four `comparative_*` params;
anomaly_discovery.py:403-462 implements `scan_full_coverage` emitting
`[FCCS] Full-coverage comparative scan` (:436) and `[FCCS] N/n flagged` (:461)).

**Conclusion: the v97 runs were executed WITHOUT the flag — an operator/script
omission, not a code bug.** The recorded 0.16155 is a directed-scan (v96-config)
re-run whose +0.0123 over v94's 0.1494 sits inside the ±0.03 noise band. C's
audit added the structural finding: `evaluation_report.json` persists NO run
args/flags (only timing + evaluation), so the mislabeling was undetectable from
the report alone — the run.log grep is the only activation evidence, which is
why the gate below is necessary and why the invocation must be persisted.

## The Controlled Change (this loop)

Exactly one architecture-level switch, flag-only, zero new code: run the full
65-video eval and the 14-video smoke with `--full_coverage_scan` ON against the
byte-identical post-v97 working tree. This replaces Stage-4 candidate
generation wholesale: directed partial-coverage independent scanning (10-18 of
23-120 segments, absolute per-segment judging, up to ~12 frames/segment via
scan_segment) -> full-coverage batched comparative scanning (100% of segments,
relative judging within contiguous batches of 8 segments x 2 keyframes, or
16 x 1 for >60-segment videos, ~4 calls/video). Everything downstream is
untouched: same 0.3 threshold, `_merge_adjacent`, candidate_top_k=3, v51
half-max boundary detection on the 0.75*CLIP+0.25*MIL curve (computed
independently in frame_scorer), v95/v96 confidence modulators, hard NMS, gates.

## C's Review Verdict: APPROVE, with 3 mandated procedural changes

C independently re-verified every load-bearing claim (premise TRUE; all five
wiring points correct; the head-slice defect real — `_comparative_call`
truncates keyframes[:2] while the directed path's scan_segment shows all ~12
frames, so Scenario B regression is genuinely live; `clip_regions=[]`
side-effect-free; FM2 correctly demoted to UNDER TEST; FCCS acts at the
segment-inclusion layer but as single-pass wholesale replacement, not revote).
Three small procedural mandates (zero code edits to the evaluated system):

1. **PERSIST THE INVOCATION** — at launch of BOTH runs, the exact command line
   (including `--full_coverage_scan` and all defaults explicitly) must be in
   `results/batch_eval_v98_{smoke,full}/COMMAND.txt`. Rationale:
   evaluation_report.json records no args/flags; the flag state must be
   independently auditable. (D1 has pre-written both files with the exact
   mandated commands; the runner executes exactly those and overwrites
   COMMAND.txt with the actually-invoked line if it deviates.)
2. **STRENGTHEN THE GATE FROM ">0" TO COUNT-CONSISTENT** — see the gate
   section below. A partially-falling-back or partially-skipped run must FAIL
   the gate, not pass on a single line. Gate failure -> STOP, diagnose, record
   nothing.
3. **EXTEND THE RECORD CORRECTION to the durable artifact** — in addition to
   annotating `session_v97_ledger.md`, annotate the v97 idea file
   (`loop1_v97_full-coverage-comparative-scan_20260718.md`, previously marked
   "Decision: KEEP") with the flag-omission root cause and a pointer to the
   v98 verified numbers. (Both done by D1 this session.)

Non-blocking notes from C: the v98-vs-v97 comparison carries an API-drift
confound (different sessions), adequately mitigated by the within-noise
adoption rule plus per-video conversion analysis against the existing v97
per-video JSONs; a same-session directed re-run would fully de-confound but
doubles eval cost for low marginal value — optional, not mandated.

## Mandatory Verification Gate (standing methodology for ALL future flagged runs)

Before accepting any number, for EACH run (smoke then full):

1. `grep -c "\[FCCS\] Full-coverage comparative scan" <output_dir>/run.log`
   == number of videos that reached Stage 4 (total videos minus zero-segment
   early exits; cross-check against the per-video JSONs in the output dir —
   expect ~14 smoke / ~65 full minus early exits). Count-consistent, NOT ">0".
2. `grep -c "\[FCCS\].*segments flagged before merge" <output_dir>/run.log`
   == same count.
3. `grep -c "\[Directed scan\]" <output_dir>/run.log` == 0.
4. `<output_dir>/COMMAND.txt` exists and contains `--full_coverage_scan`.

Gate failure -> STOP, diagnose wiring/environment, record nothing.

## Run Plan (executed by eval agents, NOT by D1)

- Smoke: exact command in `results/batch_eval_v98_smoke/COMMAND.txt`
  (`--max_per_category 1`, 14 videos, `--full_coverage_scan`, all other flags
  identical to the recorded v97 smoke: concurrency 3, detect qwen-vl-plus,
  classify qwen3-vl-plus, fps_sample 3.0, fps 2, segment_duration 6.0,
  segment_overlap 0.25, dataset_dir /Users/zzl/Movies/ucf-crime). Gate first.
- Full: exact command in `results/batch_eval_v98_full/COMMAND.txt`
  (`--max_per_category 5`, 65 videos, otherwise identical). Gate first.
- Per-video analysis against the v97 (directed-scan) per-video JSONs in
  `results/batch_eval_v97_full/`: candidate counts per video, wrong-location
  conversions among the ~15 known zero-IoU videos (Stealing036, Robbery050,
  Vandalism015 in smoke), TP regressions on currently-correct videos,
  per-video latency (first true FCCS latency number — the recorded 141.47s
  was directed-scan latency under API drift).

## Decision Rule

- FCCS full mAP >= 0.16155 - 0.03 (within noise or better) -> FCCS becomes the
  new baseline; proceed to the DGEA loop measuring delta against THIS fresh
  same-session number (not 0.16155), with per-video wrong-location conversion
  analysis as the primary keep/revert criterion (raw deltas inside ±0.03 are
  not decisive).
- FCCS full mAP < 0.13155 -> the "FM2 refuted for wholesale replacement"
  lesson is itself refuted; DGEA still proceeds (approved-in-principle) but
  its risk section is rewritten against the refuted category and its
  stratified-keyframe mechanism is re-framed as the remedy for FCCS's
  per-segment evidence deficit (2 head frames vs ~12 in the directed path).

## Expected Gain (honest, bidirectional)

Primary deliverable is a valid baseline and a tested safety lesson, not a
guaranteed delta. Scenario A (comparative calibration dominates): +0.005 to
+0.015 over the directed-scan 0.16155 -> 0.165-0.175, re-validating the
v56/v97 comparative-judging hypothesis for real and giving DGEA a solid floor.
Scenario B (per-segment evidence reduction dominates — 2 head frames vs ~12):
regression toward 0.13-0.15, refuting the wholesale-replacement safety lesson
and re-framing DGEA's stratified keyframe mechanism as the required fix.
Either outcome unblocks the path to 0.20 and makes the next loop's DGEA delta
attributable for the first time.

## Sequenced Next Loop (approved-in-principle by C, documented here, NOT executed this loop)

DGEA (Dynamics-Guided Evidence Allocation) on the verified FCCS baseline,
delta measured against the fresh same-session FCCS number, primary decision
criterion = per-video wrong-location conversion analysis, with C's two
mandated refinements:
(a) keyframe budget allocated WITHIN each comparative batch (exactly
    batch_size x k_base images per call, e.g. 16) via within-batch
    water-filling — makes the latency-neutrality claim true per-call and
    eliminates 32-image straggler calls;
(b) spike-robust placement for the k_base=1 long-video rule (argmax of
    3-frame moving-average-smoothed d(t), or median position of top-3
    dynamics frames) instead of raw global argmax;
keep R1 (stratified argmax for k>=2) and R2 (even-spacing fallback); add a
>60-segment video to the smoke recall check.

---

## Implementation Notes (D1)

**Zero code delta to the evaluated system** — the working tree stays
byte-identical post-v97 in every behavior-affecting file; no git mutations of
any kind. No prior partial v98 implementation existed (repo-wide grep found
v98 only in a workflow journal). Pre-flight wiring re-verification passed at
implementation time: config.py:80/:153, batch_evaluate.py:189-191,
pipeline.py:854-885/:692-705/:1332, anomaly_discovery.py:403-462 (:436/:461)
all present and unmodified; v97 log evidence re-counted (full 59 directed /
0 FCCS; smoke 13 / 0).

Changes made (all procedural/documentation, per the approved plan + C's 3
mandates):

1. `pipeline.py` — docstring header bumped to v98 with a concise version-note
   entry (documentation-only, zero behavioral effect; mandated by standing
   version-note convention). No code lines touched.
2. `iteration_logs/session_v97_ledger.md` — record correction banner added;
   v97 row relabeled "Directed-scan re-run (v96 config) — flag omission, NOT
   FCCS"; v98 row added (PENDING, pending gate); count-consistent gate and
   decision rule recorded as standing methodology.
3. `iteration_logs/ideas/loop1_v97_full-coverage-comparative-scan_20260718.md`
   — RECORD CORRECTION banner inserted at top (flag-omission root cause +
   pointer to v98); "Decision: KEEP" annotated (void as FCCS validation);
   FM2-refutation lesson demoted VALIDATED -> UNDER TEST.
4. `results/batch_eval_v98_smoke/COMMAND.txt` and
   `results/batch_eval_v98_full/COMMAND.txt` — exact invocations persisted
   (C mandate 1), including `--full_coverage_scan` and all defaults
   explicitly, mirroring the recorded v97 runs (dataset_dir
   /Users/zzl/Movies/ucf-crime per run_v50.sh; smoke --max_per_category 1,
   full --max_per_category 5).
5. This idea file — flag-omission root cause + the count-consistent
   log-verification gate documented as standing methodology for all future
   flagged runs.

Backups before editing: `.qoder/backups/v98/` (pipeline.py,
session_v97_ledger.md, v97 idea file).

**Validation:** ast.parse OK on pipeline.py; `import pipeline` OK. No
behavior-affecting file modified, so the v98 eval measures exactly one
controlled variable: the `--full_coverage_scan` flag.

**Deviations:** one deliberate, documentation-only deviation from the
proposal's literal "working tree stays byte-identical" wording — the
pipeline.py docstring header v98 entry (standing version-note convention;
cannot affect evaluation behavior; the evaluated code paths are
byte-identical post-v97).
