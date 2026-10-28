# Loop 1 / v143 — CGR: Coverage-Guided Recall Rescue

**VERDICT: REVERT.** Full mAP 0.13937 breached the hard accuracy floor (0.15132), and the
video-matched bootstrap point delta was NEGATIVE (-0.018) with P(delta>0)=0.174.

## Headline bootstrap verdict (the deciding statistic)

- Mode: cross-run (61 common videos, 1000 resamples) vs REF `batch_eval_v124_full`.
- **point_delta = -0.01800** (NEW v143 mAP 0.13937 vs REF v124 mAP 0.15738).
- **CI90 = [-0.03660, +0.00621]**, **P(delta>0) = 0.174**, bootstrap mean -0.01146.
- Gate FAILED on all three pre-registered criteria: point_delta>=0.02 (NO), P>=0.9 (NO),
  full mAP>=floor 0.15132 (NO, 0.13937).
- CAVEAT (tool): cross-run mode still carries ~+/-0.019 VLM run-to-run variance on top of the
  bootstrap CI. A paired-drop-flag design was built for exactly this (see below) but the
  outcome was so far below the floor that the paired refinement was moot — the ON run is
  strictly worse than the OFF baseline, not merely noisy.

## Mechanism (as implemented)

Candidate-generation family (NOT re-ranking, NOT boundary growth). Per-video ADDITIVE second
pass, triggered only when first-pass max VLM confidence < `cgr_trigger_conf` (0.70). On trigger,
partition the timeline into `cgr_strata` (6) contiguous strata, find strata that contain NO
first-pass scanned segment (true coverage gaps), and issue one EXISTING independent
`scan_segment` VLM call on each gap stratum's top-CLIP segment. A rescued segment becomes a new
candidate iff conf > first_pass_max + `cgr_margin` (0.05) AND conf >= `cgr_abs_floor` (0.55).
Rescued candidates ran through an ISOLATED copy of the downstream (`_cgr_process_rescued`) and
were appended tagged `cgr_added=True`; first-pass detections were byte-identical to flag-off,
making the OFF arm exactly reconstructible (drop `cgr_added`, cross-check `pre_cgr_ndet`).

## Effect-size arithmetic — CLAIMED vs WHAT ACTUALLY HAPPENED

- CLAIMED +0.035. Argument: AP@0.3 decided by precision at 4 attainable hit checkpoints
  (1st/7th/14th/20th); converting ~6-8 all-miss videos into TPs pulls the 14th/20th-hit ranks
  earlier and can activate the currently-zero recall=0.4 checkpoint (need only +2 over today's
  25 hits). Oracle recompute on v124 cache: +0.021 (K=4@0.90) .. +0.080 (K=16@0.55).
- ACTUAL: **mAP FELL by 0.018** (full, cross-run) and by ~0.011 (smoke: 0.14502 vs the harness's
  effective baseline). AP@0.5 collapsed 0.1464 -> 0.1098, AP@0.7 0.1044 -> 0.0909; AP@0.3 roughly
  flat (0.2213 -> 0.2174). So the additive detections did NOT land as clean TPs — they behaved as
  FALSE POSITIVES that, under GLOBAL AP ranking, DEMOTED precision at the very 14th/20th
  checkpoints the mechanism aimed to raise. This is exactly the FM2 over-fire pathology the
  proposal itself flagged as its dominant risk.
- The oracle table was an UPPER bound assuming rescued strata land ON GT at VLM locator accuracy
  ~0.638. In vivo the accept gate (`scan_segment`'s uncalibrated "even subtle signs count" prompt)
  admitted off-GT high-confidence clips, so realized on-GT rate was well below the oracle
  assumption and the FP cost dominated the TP benefit.

## FM1-FM16 table (as-run)

- FM1 curve-shape: IMMUNE (no boundary/half-max/curve modification). CONFIRMED — AP@0.3 flat.
- FM2 detection-count/FP: **MATERIALIZED — this is the killer.** Additive detections acted as
  high-ranked FPs; under global AP they demoted precision at the attainable checkpoints. The
  dual gate (rel conf>firstmax+margin AND abs>=0.55) did NOT prevent it: the second-pass VLM
  returns high conf on gap strata (uncalibrated absolute threshold), so the gate let FPs through.
- FM3 self-consistency/context-bleed: IMMUNE (independent scan_segment per stratum). Held.
- FM4 inclusion self-consistency: N/A (first-pass membership untouched).
- FM5 never-activated: CLEAN — 61 full-eval markers, 13 smoke markers; `[CGR] summary: vid=`
  emitted per flag-on video. Activation is NOT in doubt; the mechanism ran and hurt.
- FM6 sim-transfer: partial. The oracle recompute was on the live v124 cache (not a simulator),
  yet still over-predicted because it assumed correct placement; the VLM-placement + FP-cost term
  was the unmodeled gap. Lesson: an oracle that only ADDS correct TPs is not a faithful model of
  a mechanism whose real distribution includes off-GT admits.
- FM7 video-level prior: N/A (structural intra-video coverage, not a prior).
- FM8 one change/loop: respected (one additive stage).
- FM9 MPS: safe (batch_score_segments returns numpy).
- FM10 wall-clock: full avg 220.83s/video (extra calls only in triggered low-conf videos); not
  the verdict.
- FM11 one-eval validatable: yes (single full eval + bootstrap).
- FM12 call-dropping: N/A (adds calls, never merges/batches — independence preserved).
- FM13 magnitude cap: the oracle ceiling (+0.07-0.08) was above +0.03, so magnitude was not the
  blocker; DIRECTION was (net negative).
- FM14 verdict-visibility: N/A.
- FM15 substrate infidelity: AVOIDED (no offline curve precheck drove the decision; VLM judged
  live pixels via the existing path).
- FM16 noise floor: the full 290-split was NOT reached — the 65-harness result was so far below
  floor (breach of 0.012+) that the run self-terminated as a REVERT before any 290 adjudication.
  The cross-run point delta -0.018 is larger in magnitude than the ~+/-0.019 VLM noise but
  NEGATIVE, so noise cannot rescue it.

## Honest weaknesses that came true

The proposal's stated #1 weakness — FM2 over-fire driven by `scan_segment`'s uncalibrated
absolute confidence — is precisely what sank it. The absolute floor (0.55) and margin (0.05)
were picked WITHOUT in-vivo calibration; the pre-eval over-fire sentinel (accept-rate / conf-pin
check) and the smoke-65 GT-aware binding sentinel were listed as STILL PENDING and were not run
before spending the eval. The mechanism was shipped with blind gate values and evaluated
directly, so the untested knob was the failure point.

## What the NEXT loop must NOT repeat

1. **Do NOT re-propose CGR or any additive second-pass VLM scan gated on an uncalibrated
   ABSOLUTE confidence threshold.** `scan_segment`'s prompt pins high on isolated clips; any
   accept rule keyed on its absolute conf will inject high-ranked FPs that global AP punishes.
   This is the same uncalibrated-absolute-threshold pathology that motivated the (banned)
   comparative FCCS. Candidate-generation is still the right family (50/99 detections have IoU=0),
   but a rescue must use a COMPARATIVE / relative-to-first-pass criterion that is calibrated
   in vivo BEFORE any eval, or a non-VLM localizer (MIL/actionness), not the raw scan prompt.
2. **Do NOT spend a full eval before running the pre-registered smoke sentinels.** The over-fire
   sentinel (accept-rate <= 0.5, conf not pinned ~0.9) and a GT-aware on-GT-vs-off-GT check on
   smoke-65 were designed exactly to catch this cheaply and were skipped. Any future
   candidate-adding mechanism must pass a smoke-65 accept-rate + on-GT sentinel first.
3. **Do NOT trust an oracle that only adds correct TPs** as an effect-size argument for a
   mechanism whose realized output includes off-GT admits. Model the FP cost (net TP-FP under
   global AP), not the best case.
4. The candidate-generation HEADROOM is real and untouched (BUE only lengthens existing hits);
   the next attempt should locate NEW events with a signal that is verdict-independent and
   pre-calibratable (e.g. MIL/model_mil_v6 actionness peaks in coverage gaps), NOT a fresh
   uncalibrated VLM absolute-threshold gate.

## Artifacts

- Full eval report: `iteration_logs/results/v143_full_eval.json` (mAP 0.13937, AP@0.3 0.21735,
  AP@0.5 0.10985, AP@0.7 0.09091, 61 videos).
- Smoke eval: `iteration_logs/results/v143_smoke_eval.json` (mAP 0.14502, 13 videos).
- Full run log: `iteration_logs/results/v143_full_run.log` (61 FM5 markers).
- Bootstrap: `iteration_logs/results/bootstrap_ab.json` (and source dir
  `results/batch_eval_v143_full/bootstrap_ab.json`).

## Revert record

Restored from `.qoder/backups/v143/`: pipeline.py, config.py, run.py, batch_evaluate.py,
anomaly_discovery.py (diff -q byte-identical). models.py: backup .ORIG was HEAD (predates the
pre-session ece/cex/bue working-tree edits), so the v143 CGR blocks were removed surgically
(cgr_added field + serialization + from_dict + pre_cgr_ndet field/serialization) while
preserving pre_ece/pre_cex/pre_bue. analysis_bootstrap_ab.py: .ORIG backup was 0 bytes (backup
failed); removed the additive `restore_pre_drop_flag` function + `--paired-drop-flag` arg +
dispatch block. VERIFIED: py_compile PASS on all 7 files; `grep -i cgr` across all mechanism
files returns 0 matches; `from_args(Namespace()).discovery == DiscoveryConfig()` and `.scoring
== ScoringConfig()` both True; pipeline header back at v142 (BUE, default-OFF). Baseline stays
v124/v137 (committed mAP 0.15632). NO destructive git commands used.
