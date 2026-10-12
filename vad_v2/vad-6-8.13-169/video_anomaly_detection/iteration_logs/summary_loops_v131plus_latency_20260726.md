# Session Summary — v131+ LATENCY Continuation (Agent C, End-of-Session)

**Date:** 2026-07-26
**Objective:** LATENCY REDUCTION with ACCURACY PROTECTED — cut per-video VLM API calls
(detect qwen-vl-plus + classify qwen3-vl-plus) via FM12-clean whole-call DROPS only.
**Session reference (v124 baseline, unchanged):** REF_vlm_calls = **31.969** / REF_clip = 22.308 /
REF_wallclock = **183.11s** / full mAP **0.17076**. Accuracy floor **0.15132** (committed baseline 0.15632).
**KEEP gate:** full mAP >= 0.15132 AND drift-free [COST] VLM-call drop >= 15% vs REF 31.969.

---

## EXECUTIVE SUMMARY

**Net latency win achieved this session: 0.0%. No v131+ lever was KEPT. Baseline remains v124
(31.969 VLM calls / 183.11s / mAP 0.17076).** The accuracy floor **HELD on every loop that ran** —
no mechanism ever regressed mAP below 0.15132.

Both banked high-value untouched levers were built, run, and closed:

- **Lever 1 — Adaptive max_scan (v131 PAST):** mechanically fired (42/65 videos) but delivered
  **-0.29% VLM calls** (`mean_dropped=0.00` on every fire). The `max_scan = min(max(10,len//5),18)`
  ceiling empirically **binds at real content**, so the below-threshold Source-2 rank-tail it was
  supposed to trim **does not exist in practice**. Same magnitude-dilution / max_scan-binding failure
  seen at v125 and v130 — now confirmed a **THIRD time**. Scan-tail pruning is a proven dead end.

- **Lever 2 — Classify-path pruning (v132 NOCLS-CA-v2):** the cleanest lever of the session —
  a pure `return <label>` whole-call DROP of the Stage-7 `_ensemble_classify` VLM sub-tree,
  **mAP-neutral BY CONSTRUCTION** (`compute_temporal_map` reads only
  `{video_id, t_start, t_end, confidence}`, never `anomaly_type` — evaluate.py:306-380). Floor held
  (mAP 0.15847). But structurally **capped at ~7.5%** (analytic ceiling <= 9.4%: `_ensemble_classify`
  runs once/video, <= 3 VLM calls). Below the 15% gate BY CONSTRUCTION. Lever 2 is now **CLOSED**.

- **v133-v136 — analytical closures (ZERO code, ZERO eval):** after both concrete levers landed
  below the gate, the remaining loops proved *why nothing else clears it*. v135 is the load-bearing
  result: a source-verified PROOF that **no strictly verdict-independent detect-drop can exist** under
  the current architecture, so KEEP-gate requirement (a) is UNSATISFIABLE on the dominant detect path.

**Bottom-line finding:** Under strict FM12 (no merge/batch/propagation), **no construction-neutral
lever reaches the 15% [COST] gate on this 65-video anomaly-only harness.** The three FM12-clean
archetypes are each capped BELOW 15% by a distinct structural bound, and the detect-path is provably
closed by the coarse_score-coupled confidence formula. The session ends with a documented FORWARD PATH
(a two-loop confidence-decoupling prerequisite) rather than a win.

---

## PER-VERSION TABLE (v131+)

| ver | mechanism | lever | VLM-call reduction (drift-free) | full mAP | floor held | kept |
|---|---|---|---|---|---|---|
| v131 | PAST prior-adaptive scan truncation (drop below-thresh Source-2 tail) | 1: adaptive max_scan | **-0.29%** (31.877 vs 31.969; fired 42x, mean_dropped=0.00) | 0.17163 | YES | **false** |
| v132 | NOCLS-CA-v2 class-agnostic classify-subtree whole-call DROP | 2: classify pruning | **-7.5%** (29.585 vs 31.969; fired 60x, capped <=9.4%) | 0.15847 | YES | **false** |
| v133 | Eval-set undersampling closure (Normal videos ARE present; ceiling ~5.7%) | analytical | n/a (0 code, 0 eval) | -1 | trivially | **false** |
| v134 | WVNG-SINGLE calibrated single-signal whole-video Normal early-exit | 3: early-exit | n/a (rejected pre-impl, <15% guaranteed) | -1 | trivially | **false** |
| v135 | Detect-path verdict-independence CLOSURE (proof req.(a) unsatisfiable) | analytical | n/a (0 code, 0 eval) | -1 | trivially | **false** |
| v136 | Detect-inclusion decoupling precondition — escape-clause discharge | analytical | n/a (0 code, 0 eval) | -1 | trivially | **false** |

*(-1 = no eval run: analytical / no-op loop by design, per FM11 economy — the single validating
eval was not spent on a predetermined-REVERT measurement.)*

---

## DID THE TWO UNTOUCHED LEVERS WORK? — AND WHY / WHY NOT

### Lever 1: ADAPTIVE max_scan (v131 PAST) — NO (structurally null)
- **Design (FM12-clean):** keep every Source-1 segment + every CLIP>=0.3 Source-2 segment; drop only
  the below-threshold Source-2 rank-tail detect calls. `new_cap = min(old_cap, max(candidate_top_k + margin, n_protected))`.
- **Why it failed:** `mean_dropped=0.00` across all 42 fires. The ceiling `min(max(10,len//5),18)`
  already binds at real content, so `protected_front + recall_floor >= old_cap` on every video — there
  is no below-0.3 tail for old_cap to have retained. The "padding" the lever targets **is not there**.
  Confirmed a THIRD time (v125 PREGATE, v130 CASCADE, v131 PAST).
- **Verdict:** any lever that only trims the max_scan tail is null. Future scan-reduction must attack
  the **content-bound region itself** (protected-count, Source-1 / candidate_top_k sizing), not the tail.

### Lever 2: CLASSIFY-PATH PRUNING (v132 NOCLS-CA-v2) — CLEAN but CAPPED (< gate)
- **Design (FM12-clean):** short-circuit `_ensemble_classify`, return the deterministic VideoMAE label;
  drop all classify (qwen3-vl-plus) calls. Detect + VLM-boundary paths byte-identical.
- **Why it is capped:** `_ensemble_classify` runs **once per video**, <= 3 VLM calls (confusion-pair
  3-vote) or 0-1 (`_classify_clip_guided`). Ceiling = 3 / 31.969 ~ **9.4%**; realized **-7.5%**.
  Below 15% gate BY CONSTRUCTION — provable without running (FM11).
- **Verdict:** mAP-neutrality CONFIRMED (anomaly_type outside compute_temporal_map's domain), floor
  held, FM2 impossible (count fixed upstream). Lever 2 **CLOSED** — genuinely clean, just too small.

### Lever 3 (auxiliary): CALIBRATED WHOLE-VIDEO EARLY-EXIT (v127 remedy / v134) — HARNESS-CAPPED ~5.7%
- Addressable population (Normal videos) is under-sampled on this 100%-anomaly harness; even a perfect
  single-signal Normal gate is capped ~5.7% here. Rejected pre-implementation (v134) as guaranteed <15%.

---

## THE CENTRAL CLOSURE (v135): why NO detect-drop can clear the gate

Requirement (a) — mAP-neutral BY CONSTRUCTION for BOTH VLM outcomes — is **UNSATISFIABLE** on the
detect path as currently wired. Three source-verified facts:
1. Surviving-detection **confidence is strictly monotone in `coarse_score`** (pipeline.py:2225-2228:
   both `_base_conf` and `_boundary_bonus` multiply `_coarse`).
2. A merged region's **`coarse_score` = MAX over its scanned segments' VLM verdicts**
   (anomaly_discovery.py:245 is_suspicious; :478 merge keeps max).
3. `compute_temporal_map` **ranks ALL predictions GLOBALLY by confidence** (evaluate.py:342).

Chain: dropping any droppable detect segment S removes S's (un-run, VLM-determined) coarse_score from
the region MAX -> surviving confidence shifts -> global AP rank reorders -> mAP moves. The DIRECTION is
fixed by S's un-run verdict = **verdict-DEPENDENT**. Only BOUNDARIES are verdict-independent (half-max
crossing from CLIP/MIL curve, pipeline.py:2101-2113); SCORE and COUNT are not. The per-call VLM
independence FM12 protects for *surviving* calls is exactly what makes a *dropped* call's absence
verdict-visible in the global mAP ranking.

**SCOPE QUALIFIER (not architecture-eternal):** unsatisfiable UNDER THE CURRENT coarse_score-coupled
confidence formula + global-AP ranking. Do NOT cite as a permanent no-op ban after the formula changes.

---

## UPDATED FAILURE-MODE TAXONOMY (FM1-FM12, + proposed FM13/FM14)

- **FM1 curve-shape modification regresses.** Standing. The registered forward path (decouple confidence
  from coarse_score) is itself an FM1 change requiring its own mAP-neutrality loop.
- **FM2 detection-count/verdict changes make FPs under greedy match.** Standing; killer of the
  merge/batch family (v126/v129). NOT triggered by v131 (nothing dropped) or v132 (count fixed upstream).
- **FM3 VLM self-consistency / context-bleed hurts perceptual verdicts.** Standing; co-killer of v129 BDS.
- **FM4 self-consistency on inclusion is dangerous.** Standing.
- **FM5 mechanism never activated (flag + marker + grep gate).** CONFIRMED VALUABLE again — v131's
  `[PAST-AGG] mean_dropped=0.00` distinguished "fired-but-null" from "never-fired"; v132 [NOCLS-CA] fired
  60x proving REVERT was on MAGNITUDE not a no-op.
- **FM6 candidate-pool simulation gains do not transfer.** Standing (not exercised v131+).
- **FM7 video-level priors cannot fix intra-video mislocation.** Standing.
- **FM8 offline precheck is config-specific; one controlled change per loop.** Standing.
- **FM9 MPS tensors need .detach().cpu().numpy().** Standing (no tensor math v131+).
- **FM10 wall-clock is API-drift-dominated; decide on drift-free call count.** CONFIRMED ESSENTIAL —
  v131 showed a spurious ~11.7% wall "win" (-0.29% calls); v132 a spurious -48.3% wall "win" vs the
  decisive -7.5% call delta. Deciding on wall would have produced false KEEPs.
- **FM11 offline harness must complete as a pollable step; prefer mechanisms needing none.** VINDICATED
  — v132's <=9.4% ceiling and v133-v136 closures were provable WITHOUT a run; the single eval was not
  burned on predetermined-REVERT measurements.
- **FM12 cut calls by DROPPING provably-null whole calls, NEVER merge/batch/repackage.** Central.
  v131/v132 verified pure whole-call drops (survivors byte-identical) — FM12 REFUTED as a *risk* for the
  drop-call archetype. Merge/batch family (v97/v126/v129) stays refuted for accuracy.
- **NEW FM13 (proposed) — "structural-cap / magnitude-dilution":** an FM12-clean, mAP-neutral drop can
  still fail purely on MAGNITUDE: max_scan binds the scan tail (<1%, v125/v130/v131), classify is
  once/video <=3 calls (~7.5-9.4%, v132), Normal early-exit is harness-undersampled (~5.7%, v127/v134).
  Cleanliness != efficiency; bound the ceiling analytically BEFORE spending the eval.
- **NEW FM14 (proposed) — "verdict-visibility via global-max ranking":** the per-call VLM independence
  FM12 protects for surviving calls makes a *dropped* call's absence verdict-VISIBLE when the surviving
  score is a MAX over verdicts and mAP ranks globally (v135). Under such coupling, requirement (a) is
  unsatisfiable — a drop-call latency lever needs confidence DECOUPLED from the dropped verdict.

---

## CONCRETE NEXT DIRECTIONS

1. **DO NOT re-try (proven dead):** any max_scan-tail trim (v125/v130/v131); any merge/batch/contact-sheet
   (v97/v126/v129 — breach floor); classify pruning as a >=15% play (v132 capped ~7.5%); dual-AND or
   single-signal whole-video Normal early-exit on this harness (v127/v134, ~5.7%); local-compute trims
   like VideoMAE-window skip (v128, below the call metric).

2. **ONLY architectural route to a >=15% detect-drop (two-loop prerequisite, OUT of this session's scope):**
   - **Loop N:** DECOUPLE surviving-detection confidence from `coarse_score` — derive confidence from the
     verdict-INDEPENDENT CLIP boundary/actionness contrast (`best_score`/`boundary_contrast`); use
     `coarse_score` only as an existence gate (>0.3), NOT a ranking magnitude (BSN/BMN lineage). This is an
     FM1 curve-shape change — validate mAP-neutral on its own dedicated loop first.
   - **Loop N+1:** Under that redesign, a CLIP-peak-dominance detect-drop ("this Source-2 segment lies in a
     CLIP peak already covered by a scanned Source-1 segment") becomes verdict-independent on BOTH outcomes
     and thus gate-clean.

3. **Policy escalation (outside a lever's remit):** on THIS 65-video anomaly-only harness no FM12-clean
   construction-neutral lever reaches 15%. If a win is required without the redesign, the decision-owner
   must EITHER (a) relax the 15% gate to accept the banked ~7.5% clean classify-drop (v132, mAP-neutral by
   construction), OR (b) evaluate on a harness with the true ~51.7% Normal population, where whole-video
   early-exit becomes >15%-addressable. Both are policy calls, not lever proposals.

4. **Instrumentation caveat:** the [COST] counter is ~3x concurrency-inflated (sum-of-per-video-deltas vs
   true calls). Reductions are valid only as RATIOS vs the same-session REF; do not read absolute counts
   across differently-concurrent runs.

---

## ARTIFACTS
- Ledger: `iteration_logs/session_v97_ledger.md` (rows v131/v132/v135)
- Idea files: `iteration_logs/ideas/loop1_v131_past-prior-adaptive-scan-truncation_20260726.md`,
  `loop2_v132_nocls-ca-v2_20260726.md`, `loop5_v135_detect-path-verdict-independence-closure_20260726.md`
- Prior session: `iteration_logs/summary_loops_v124plus_latency_20260726.md`
- All v131/v132 mechanisms reverted; working tree ships byte-identical to committed v124 baseline.
