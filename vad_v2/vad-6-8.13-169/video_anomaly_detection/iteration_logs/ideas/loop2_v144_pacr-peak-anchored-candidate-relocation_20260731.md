# v144 — PACR: Peak-Anchored Candidate Relocation

**Loop:** 2 (session v8)  **Version:** v144  **Date:** 20260731
**Family:** candidate-generation (relocation, count-neutral)
**Verdict:** REVERT — did NOT reach a real verdict. The run CRASHED at the FM5 activation gate before any accuracy eval; no full eval, no bootstrap CI. Reverted by orchestrator decision.
**Baseline unchanged:** committed mAP 0.15632 (ref full v124 0.17076). Hard floor 0.15132 untouched (nothing shipped).

---

## Mechanism (as implemented)

For each FINAL detection whose window does NOT contain the video's in-vivo,
category-conditioned CLIP-curve global peak (`frame_scorer.get_category_scores`
— the exact cached, verdict-independent signal the boundary logic already
trusts; ZERO new CLIP cost), PACR issues ONE fresh INDEPENDENT `scan_segment`
VLM call on the peak segment. If the peak segment's confidence exceeds the
detection's own first-pass confidence by a margin (`pacr_conf_margin=0.05`, a
COMPARATIVE gate in post-code — never a merged/contact-sheet prompt), it
TRANSLATES the detection's window to be centered on the peak, **preserving
width, confidence, anomaly_type, and rank exactly**. It never adds a detection,
never grows a boundary outward, never unions old+new, never does local OIC
relocation. Anti-duplicate: skip if the peak is within `pacr_min_peak_gap_sec=6.0`
of any OTHER detection's original window.

Placement: terminal slot AFTER `_qscore_rerank`, BEFORE the `pre_bue_boundaries`
capture, so it composes with BUE (PACR translates far-misses first; BUE then
grows the relocated window). Paired-measurable via `pre_pacr_boundaries`
persisted per result JSON.

It is the count-neutral repair of the refuted CGR (v143): CGR ADDED off-GT clips
as high-ranked FPs and measured -0.018; PACR translates instead of adds, so the
intended worst case is neutral, not negative.

---

## Effect-size arithmetic — CLAIMED vs WHAT ACTUALLY HAPPENED

**Claimed +0.035 (range +0.03..+0.05).** AP@0.3 is decided by precision at 4
attainable hit checkpoints (1st/7th/14th/20th hit; v124 ranks 1/9/24/48 =
precision 1.0/0.778/0.583/0.417). Analytic ceiling (FM13): of 60 anomaly videos,
37 have zero detection hitting GT@0.3; the in-vivo peak lies inside GT in 15/37
(41%) → 15 recoverable videos. Because PACR is count-neutral, each recovery
converts a miss into a hit AT ITS EXISTING RANK with NO new FP, so every
checkpoint term is upward. Realizing 6–8 of 15 → 20th-hit rank 48→~40
(precision 0.417→0.50, +0.083), 14th-hit similar, +2 net hits activates the
recall=0.4 checkpoint (~+0.036) → honest +0.03..+0.05.

**What actually happened: THE CLAIM WAS NEVER TESTED.** The run crashed at the
FM5 `[PACR-GATE]` (see below) before `evaluation_report.json` was written and
before any full/290 eval or paired bootstrap ran. All the load-bearing
PRE-EVAL prechecks the proposal itself declared MANDATORY (N_recoverable>=12,
TP_EXPOSURE<N_recoverable, IoU-realized width ceiling, and above all the
smoke-65 degeneracy sentinel) were NOT executed. **No accuracy evidence exists
for or against PACR.** The only number produced is an offline-recomputed
14-video sentinel-smoke mAP with the flag ON = **0.16852** (AP@0.3 0.2899,
AP@0.5 0.12475, AP@0.7 0.09091), which is a 14-video subset, far below the noise
floor, and is NOT a verdict.

---

## Headline verdict statistic (BOOTSTRAP CI)

**None.** No full eval → no `bootstrap_ab.json`. Point delta, CI90, and
P(delta>0) are all **absent (recorded as -1 / null)**. The mechanism died at an
activation-plumbing bug, not at the accuracy gate. This is a build/gate failure,
not a refutation of the idea.

---

## The crash — FM5 gate bug (the actual cause of REVERT)

The 14-video sentinel smoke finished 14/14 (Succeeded 14, Failed 0), but the
FM5 `[PACR-GATE]` refused to write `evaluation_report.json`:
markers=13 < succeeded=14. `Robbery050_x264` succeeded with **0 anomalies** but
the empty/0-detection code path SKIPPED the per-video `[PACR] summary:` marker,
violating the FM5 spec ("emit once per succeeded video, incl. empty/normal").
Relaunch would NOT help: all 14 result JSONs already exist, so the runner skips
those videos and the gate re-fails deterministically. The one-line fix (emit the
PACR summary even when ndet_before=0) was never applied because the orchestrator
called REVERT. **Latency 244.2 s avg/video (wall-clock 1183.6 s, 2.89x
parallelism).**

Secondary red flag surfaced by the zero-VLM functional smoke and by the mocked
run: **accept_rate hit 1.0 on a degenerate mock** where the peak scan pinned
~0.9 — exactly the VLM-gate-degeneracy pathology the proposal flagged as its #1
weakness. This was never disambiguated on real VLM calls because the run crashed
first.

---

## FM1–FM16 table (design intent vs realized)

| FM | Design claim | Realized this loop |
|----|--------------|--------------------|
| FM1 curve-shape | IMMUNE — curve/frame_scores never modified; get_category_scores read-only | Not contradicted (no accuracy run) |
| FM2 count/FP | count FROZEN (translate not add) → no new FP in ranking | Untested; this was CGR's killer, PACR's whole premise — UNVERIFIED |
| FM3 context-bleed | IMMUNE — one INDEPENDENT scan_segment/fire, compared in post-code | Design preserved |
| FM4 inclusion self-consistency | N/A | N/A |
| FM5 activation | unique per-fire + per-video markers + grep gate | **FAILED HERE** — empty/0-det video emitted no summary marker; gate blocked report |
| FM6 sim-transfer | accept decision is a LIVE VLM verdict, not offline-curve transfer | Not reached |
| FM7 video-level prior | N/A (per-detection intra-video) | N/A |
| FM8 one change | single terminal-slot method | Held |
| FM9 MPS | scan_segment/get_category_scores return numpy | Held |
| FM10 wall-clock | <=1 extra VLM call/fire | 244 s/video observed; API-drift dominated |
| FM11 one-eval-validatable | one full/290 eval + paired bootstrap | Never got to spend it |
| FM12 no merge/batch | adds calls, never merges | Held |
| FM13 magnitude | ceiling 15 recoverable, 6–8 clears +0.03 | Unverified (no eval); analytic only |
| FM14 verdict-visibility | N/A (confidence frozen) | N/A |
| FM15 substrate infidelity | get_category_scores computed IN-VIVO == fidelity-gated substrate | Design correct; irrelevant since no offline precheck ran |
| FM16 noise floor | PAIRED via pre_pacr_boundaries + 290 final verdict | Never measured |

---

## Honest weaknesses (confirmed and outstanding)

1. **VLM-gate degeneracy — the dominant live threat, and it showed teeth.**
   `scan_segment`'s "even subtle signs count" prompt pins ~0.9 on isolated
   clips. The mocked/degenerate smoke drove accept_rate to 1.0. Count-neutrality
   caps the downside at ~neutral, but ALSO erodes the upside: relocations land
   on GT only ~41% of the time, so a degenerate gate realizes far below +0.03.
   The pre-registered smoke sentinel (accept_rate<=0.5, no uniform 0.9 conf-pin,
   on-GT>=off-GT) is LOAD-BEARING and was never run against real VLM.
2. **IoU realization / width binding.** A width-preserved window centered on the
   peak reaches IoU>=0.3 only if the original is wide enough (~>=0.3*GT);
   zero-overlap dets skew short (dur ratio ~0.42), so some of the 15 ceiling
   converts to AP@0.5/0.7 recall rather than AP@0.3 hits — pulling honest
   realization toward +0.02..+0.03. Re-deriving width at the new anchor would
   help but is a SECOND change (FM8 forbids it).
3. **The FM5 empty-video summary bug** — a plumbing defect, cheap to fix, but it
   cost the entire loop. Any future mechanism with a per-video activation gate
   MUST emit its summary marker on the 0-detection / normal-video path too, or
   the gate deadlocks after the JSONs are written.

---

## What the NEXT loop must NOT repeat

- **Do NOT ship a per-video activation gate that only emits its summary marker
  when ndet>0.** Emit the summary on EVERY succeeded video (incl. empty/normal)
  or the `[GATE]` deadlocks and no report is written — and relaunch cannot
  recover because JSONs already exist. Test the 0-detection path in the
  functional smoke BEFORE spending an API smoke.
- **Do NOT trust a comparative VLM gate on raw `scan_segment` confidence without
  first running the accept-rate + conf-pin + on-GT/off-GT sentinels in vivo.**
  This is the third loop (v143 CGR, v144 PACR) whose thesis leans on
  `scan_segment` producing calibrated, discriminative confidence on isolated
  clips; the mock again pinned ~0.9. Either calibrate that signal first, or use
  a verdict-independent, pre-calibratable localizer (MIL/actionness) instead of
  a fresh uncalibrated VLM threshold.
- **Do NOT re-propose boundary outward-growth / UNION / OIC relocation** —
  OWNED by v142 BUE, pending its 290 verdict.
- The candidate-generation headroom (50/99 dets IoU=0, median GT distance ~20s,
  which BUE cannot reach) **remains OPEN and unrefuted** — PACR neither confirmed
  nor killed it. A count-neutral relocation is still a structurally sound idea;
  it needs (a) a bug-free empty-video activation path, (b) an in-vivo-calibrated
  or verdict-independent relocation gate, and (c) the pre-registered degeneracy
  sentinel actually run before any eval spend.

---

## Artifacts

- `iteration_logs/results/v144_smoke_run.log` (14-video sentinel smoke, crashed at gate)
- `iteration_logs/results/v144_smoke_eval.json` (offline-recomputed metrics; NOT a verdict)
- No `v144_full_eval.json`, no `v144_full_run.log`, no `bootstrap_ab.json` (full eval skipped by smoke-gate crash).
- 14 per-video result JSONs remain at `results/batch_eval_v144_smoke/`.

## Revert verification

All 5 core files restored to the pre-edit state and diff-verified byte-identical
to `/Users/zzl/PycharmProjects/vad/.qoder/backups/v144/`:
`pipeline.py`, `config.py`, `run.py`, `batch_evaluate.py`, `models.py` — all
`diff -q` clean. `grep -i pacr` = **0 matches** across all 5. `py_compile` PASS
x5. `from_args(Namespace()).scoring == ScoringConfig()` and `.discovery ==
DiscoveryConfig()` both True. Pipeline header back at **v142** (BUE/CEX
default-OFF). No destructive git commands used.
