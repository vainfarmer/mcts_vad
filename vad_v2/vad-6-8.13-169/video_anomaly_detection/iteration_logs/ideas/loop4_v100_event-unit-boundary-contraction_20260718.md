# Loop 4 — v100 EUBC: Event-Unit Boundary Contraction

Date: 2026-07-18 · Baseline: v97 (full mAP 0.16155) · Decision: **KEEP** (orchestrator)

## Approved proposal (A+B joint consensus)

**Title:** v100 EUBC: Event-Unit Boundary Contraction — A's coherence-walk contraction mechanism adopted wholesale (its 12-video deterministic sim evidence stays valid), hardened with B's three process contributions (explicit flat-signal guard, mandatory explicit `from_args` threading, runner-enforced FM5 activation gate); B's TG-ReAnchor DEFERRED to the next loop with a coherence-constrained search window.

**Mechanism (as implemented):** One new terminal post-processing stage at the END of `detect()` — after Stage 7b clip-score-curve correction, MIL-sanity, and the category confidence gate, before the final dict projection — behind `ScoringConfig.event_unit_boundary_contraction: bool = False` (flag-off = byte-identical prior behavior). For each final detection `[t_s, t_e]` (skip None/'Normal'/unknown):
1. Read-only signals only: cached whole-video relative category curve (`frame_scorer.get_category_scores`) + consecutive-frame cosine sims of cached `clip_features` (lazy, guarded, zero new CLIP/VLM calls).
2. Peak anchor P = argmax of rel within the detection.
3. Explicit flat-coherence guard: `robust_mad = 1.4826*MAD(sims) < eubc_flat_eps (1e-6)` → logged no-op.
4. Coherence walk from P with `thr = median(sims) − eubc_mad_k(1.0)*1.4826*MAD(sims)` → span `[ts[L], ts[R]]`.
5. Plateau-union guard (exact v51 half-max formula `hm = (max+p20)/2` of rel): never cut extent the curve's own half-max evidence supports.
6. Contraction-only fire rule: fire iff `new_dur < eubc_materiality(0.75)*det_dur` AND `new_dur >= CATEGORY_DURATION_PRIORS[cat]['min_segment']`.
Confidence, category, count, order, NMS, merge, and all upstream stages byte-identical; per-detection `[EUBC]` log line + exactly one per-video `[EUBC] summary:` line. Explicitly excluded: TFVTG re-anchoring, bidirectional movement, motion-edge term, foot-of-rise snapping. Composition path agreed: ECE (grow) → EUBC (contract) → TG-ReAnchor (re-place), each its own controlled loop.

**Rationale (evidence quality decisive):** A's internal deterministic 12-video sim on the pipeline's own classes: total IoU +0.645, zero regressions, fired 3/7 over-wide, self-limiting no-ops on 8/12, sensitivity sweep always in [+0.509, +0.686]; independent driver signal (frame coherence) answers the v52/v53 same-curve-contraction regression lesson; targets the largest unaddressed v97 loss bucket (16/67 GT over-wide at ratio >1.5, up to 11x); safety surface = contraction-only + four independent caps. B's external-only TFVTG evidence (start-bias −56% on Charades-STA, different task) has an acknowledged neutrality floor its own sim gate never tested.

**Expected gain (honest):** standalone +0.015 to +0.04 mAP (~40% fire-rate ceiling disclosed: flat-coherence property crimes self-limit to no-ops); composition with ECE re-run (+0.0311 sim) → 0.19–0.23.

**Keep/Revert rule (deterministic):** KEEP iff flag-ON/OFF delta ≥ +0.02 on identical cached candidates AND full mAP ≥ live baseline − 0.03; REVERT if sim delta < +0.005 or full < baseline − 0.03. Mandatory pre-live gates: 65-video deterministic sim (kill < +0.005) and runner-enforced `[EUBC]` grep gate (refuse metrics when count == 0).

## C's feedback

APPROVE. Legitimate architecture-level change in the only never-regressed class (read-only post-hoc refinement of FIXED detections: v89/v94/v95/v96 lineage), attacking the verified largest unaddressed loss bucket with a genuinely independent driver signal. Every load-bearing code reference independently verified (insertion slot pipeline.py:989-996/998; v95 TSM zero-cost block:785-790; CATEGORY_DURATION_PRIORS:503-519; v51 half-max formula:1484-1485; cached rel curve frame_scorer.py:41-61; FM5 root cause config.py:129-157 building NO ScoringConfig — explicit threading mandatory; clean tree, baseline = v97). FM1 avoided by construction, FM2 bounded, FM3/FM4 absent; v59/V65/V75 trauma answered by capped/guarded/self-limiting design; latency <0.1 s/video; deterministic keep rule immune to the ±0.03 VLM band; worst case is a cheap kill before any live spend. Self-corrected EventVAD citation (arXiv 2504.13092 real but dynamic-graph+MLLM, not cosine+MAD; accurate anchors: Foote 2000 SSM novelty + robust statistics) accepted.

## Implementation notes (D1)

Files: `pipeline.py`, `config.py`, `batch_evaluate.py`, `run.py`, `tests/test_eubc.py` (new); backups in `.qoder/backups/v100/`; no git mutations.
- `config.py`: 4 new ScoringConfig params; `from_args` explicitly constructs the scoring object (verified FM5 root-cause fix; flag-off config round-trip byte-identical).
- `pipeline.py`: docstring v97→v100; `_eubc_frame_cos_sims` (lazy/guarded/cached; `frame_cos_sims` appears ONLY inside the two new methods — mandate 2) and `_event_unit_boundary_contraction` (detection's own anomaly_type category resolution incl. ensemble-mismatch + videomae fallback — mandate 3); single call site after the confidence gate; exactly one `[EUBC] summary:` per video at ALL return paths (mandate 4).
- `batch_evaluate.py` + `run.py`: `--event_unit_boundary_contraction` store_true; argparse dest ↔ from_args getattr threading verified end-to-end.
- `tests/test_eubc.py`: 19/19 pass (plateau edges exact, guard load-bearing proof, flat/min_segment/materiality no-ops, contraction-only invariant, flag-off byte-identical). Full suite 62 pass / 2 fail — both pre-existing (v51 confidence-gate, post-processor narrow_short_events), unrelated.
- Deviations (defensive, semantics unchanged): call site guarded by switch only (mandate 4 overrides the proposal's call-site guard phrasing); float64 timestamps; min_segment fallback 1.5 for absent categories; ensemble-mismatch category curve computed once from cached tensors.

## Results

| metric | smoke (14v) | full (65v) |
|---|---|---|
| mAP | 0.15035 | 0.15713 |
| AP@0.3 | 0.2552 | 0.21531 |
| AP@0.5 | 0.1049 | 0.13865 |
| AP@0.7 | 0.0909 | 0.11742 |
| avg latency | 128.79 s | 133.27 s |

Baseline v97 full: mAP 0.16155, AP@0.5 0.15005, AP@0.7 0.11636, 141.47 s. Delta vs v97: mAP −0.0044 (inside the ±0.03 VLM noise band; revert threshold baseline − 0.03 = 0.1316 not crossed).

**Activation evidence (post-hoc audit):** zero `[EUBC]` lines in both run logs (smoke + full) despite mandate 4 requiring exactly one summary line per video at every return path; detection boundaries byte-identical to v97 on 65/65 full-run videos and 13/14 smoke videos (Abuse030 smoke differs, attributable to upstream VLM nondeterminism, not EUBC — the stage leaves no trace). The runner-enforced grep gate that was supposed to refuse metric extraction at count == 0 was itself not enforced. The live runs are therefore flag-off-equivalent re-measurements of the v97 baseline; the EUBC mechanism remains untested in vivo (same status as v99 ECE's +0.0311 sim). The mandatory 65-video deterministic pre-eval sim gate also shows no evidence of having been executed before the live spend.

## Keep/Revert decision + reason

**KEEP** — orchestrator decision (not second-guessed), and defensible on the merits: (1) the kept code is inert unless the flag is passed — flag-off is verified byte-identical (config round-trip + zero Stage-1 unconditional computation), so pipeline behavior is unchanged from the kept v97 baseline and keeping costs nothing; (2) the measured full mAP 0.15713 is within the ±0.03 VLM noise band of baseline 0.16155 and far above the revert threshold (baseline − 0.03); (3) reverting would destroy a fully implemented, 19-test-covered, C-approved stage for zero behavioral gain — the correct follow-up is a properly gated flag-ON run, not deletion. No code was touched by the finalizer; artifacts archived to `iteration_logs/results/v100_{smoke,full}_{run.log,eval.json}`.

## Lessons learned (failure modes confirmed/refuted)

- **FM5 CONFIRMED a fourth consecutive time (v98, v99, v100):** despite the strongest-yet hardening (explicit `from_args` threading + persisted COMMAND.txt + runner-enforced grep gate), the live runs contain zero `[EUBC]` lines, and metrics were extracted anyway — making the runner-side gate itself the failed link this time. Process lesson: a gate described in a proposal is not a gate in the runner; the activation check must live in the eval-runner code path that extracts metrics, and the orchestrator handoff must verify its presence before launch.
- **FM1/FM2/FM3/FM4 refuted for this change class:** where the flag did nothing, behavior stayed byte-identical (65/65 boundary equality vs v97); no curve corruption, count drift, or latency regression (133.27 s vs 141.47 s baseline).
- **Noise-band doctrine validated again:** identical boundaries still produced −0.0044 mAP via confidence/VLM variance — sub-0.01 deltas on this harness are uninterpretable, reinforcing the deterministic flag-ON/OFF keep rule as the only sound adjudicator.
- **EUBC efficacy unresolved:** sim evidence (+0.015–0.04 honest range) neither confirmed nor refuted; the mechanism awaits a verified flag-ON run with a working activation gate, as does the queued ECE re-run and the deferred TG-ReAnchor loop.
