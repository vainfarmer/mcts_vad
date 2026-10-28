# v126 — TRLM+ : Temporal Run-Length Merge of contiguous near-duplicate targeted segments

**Loop:** 3 (session v97 ledger) | **Date:** 2026-07-25 | **Decision: REVERT**

## Proposal

**Title:** TRLM+ : Temporal Run-Length Merge of contiguous near-duplicate targeted segments before the directed VLM scan (with a mandatory smoke abort-gate).

**Mechanism:** Architecture-level control-flow change in `pipeline.py` `_directed_llm_scan` (line ~1904) — the dominant per-video VLM detect path, which issues ONE `discovery.scan_segment` -> `llm.generate` VLM call per targeted segment. AFTER `targeted_segments` is finalized and capped by `max_scan` and BEFORE the `tasks=[scan_segment(...)]` gather, collapse runs of temporally-contiguous, visually-near-duplicate targeted segments into a SINGLE VLM call and propagate that one verdict dict to every member:
1. sort targeted_segments by t_start;
2. walk them, starting a NEW run whenever the next segment is not temporally contiguous (t gap > eps) OR cosine of their mean CLIP features < tau (default 0.97);
3. per run pick the representative = the member with the highest CLIP anomaly_score (biases toward flagging = protects recall); issue ONE scan_segment call and assign the returned `{is_suspicious, anomaly_score, description}` to ALL members;
4. build the `suspicious` list per-member exactly as today, keeping each member's own t_start/t_end/frame_indices, so `_merge_adjacent` and every downstream boundary/curve path are byte-unchanged.

Run-length pruning (RLP, Pattern Recognition Letters 2026) lifted from token granularity to VLM-call granularity, training-free. Default-OFF; flag-off skips the block entirely (byte-for-byte prior behavior).

**Config params:** `trlm_merge: bool = False`, `trlm_cos_tau: float = 0.97`, `trlm_gap_eps: float = 0.1`.

## C's feedback (APPROVE)

Strongest proposal of the session and the first architecturally sound latency mechanism. Code claims verified against the working tree: `_directed_llm_scan` issues exactly one scan_segment VLM call per targeted segment; Source-1 segments overlapping a single clip_region are consecutive time-slices and thus contiguous-by-construction; max_scan cap and suspicious-build loop confirmed. Critical mis-profiling check PASSES: `full_coverage_comparative_scan` defaults False, so the `elif clip_regions` branch is the live path — `_directed_llm_scan` is on the common path, NOT the dead NGATE-style branch. Strengths: architecture-level run-length control-flow merge (not magic-number tuning); config-gated default-OFF byte-for-byte revert; real `--trlm_merge` flag + per-video `[TRLM]` marker + grep gate (FM5); FM9-safe CPU cosine via `.cpu().numpy()` reusing v100 pattern; FM11-clean (no offline harness); FM1 immune. Incorporates the v125 lesson via a hard smoke abort-gate that converts a would-be null into a zero-cost early stop. mAP risk honestly scoped to FM2 and argued down (tau=0.97 + mandatory contiguity + max-CLIP-score representative + hard floor gate at 0.15132).

## Implementation notes (D1)

Implemented as an architecture-level, config-gated (DEFAULT-OFF), byte-for-byte-revertible control-flow merge in `MultiAnomalyVAD._directed_llm_scan` (pipeline.py). Added: (1) `batch_evaluate.py` `--trlm_merge/--trlm_cos_tau/--trlm_gap_eps` args (verified in `--help`); (2) pipeline.py helpers `_trlm_mean_feat` (FM9-safe `.detach().cpu().numpy()`; L2-normalized mean CLIP feat; None on empty/failure) and `_trlm_runs` (sorts by t_start, splits on gap>eps OR cosine<tau OR unavailable feature — conservative); (3) merge block AFTER max_scan cap and BEFORE scan_segment gather, guarded by `trlm_merge and precomputed is not None and len(targeted_segments)>1`; representative = max CLIP anomaly_score member, with graceful fallback to first member (no NameError); each member keeps own t_start/t_end/frame_indices; single verdict fanned out; (4) per-video `[TRLM]` INFO marker. Verified: ast.parse + import OK; unit test of `_trlm_runs`. Q-score default-ON and v51 half-max boundary untouched.

**Files:** `pipeline.py`, `batch_evaluate.py`, `config.py`. Backups in `/Users/zzl/PycharmProjects/vad/.qoder/backups/v126/`.

## Eval flags, activation marker, marker counts

- **Eval flags:** `--trlm_merge --qscore_rerank`
- **Activation marker:** `[TRLM]`
- **Smoke marker count:** 13 (14-video smoke)
- **Full marker count:** 59 (65-video full; videos with <=1 targeted segment produce no merge line)
- Sample: `[TRLM] video=Abuse028 targeted=10 runs=2 calls_saved=8` — merges are large and real; the mechanism fired at scale (contradicting the anti-correlation null hypothesis).

## Results

### Accuracy (temporal mAP)

| Metric | Smoke (14 vid) | Full (65 vid) |
|---|---|---|
| mAP | 0.18855 | **0.12547** |
| AP@0.3 | 0.36364 | 0.16787 |
| AP@0.5 | 0.11111 | 0.11765 |
| AP@0.7 | 0.09091 | 0.09091 |
| frame AUC | 0.80477 | 0.69225 |
| video-level F1 | 0.96000 | 0.94017 |

**Accuracy floor = 0.15132. Full mAP 0.12547 < floor → FLOOR BREACHED.**

### Latency / cost (drift-free [COST], 65 videos)

| Metric | v126 (TRLM on) | Session REF | Win |
|---|---|---|---|
| mean_vlm_calls | 11.2 | 31.969 | **-65.0%** |
| mean_clip_calls | 21.815 | — | — |
| mean_elapsed_s | 134.694 | — | — |
| wall-clock/video | 134.7s | 183.112s | **-26.4%** |

## Decision: REVERT

**Reason:** The latency mechanism worked and worked well — VLM call count dropped 65.0% (31.97 → 11.2) and wall-clock dropped 26.4% (183.1s → 134.7s), both far exceeding the 15% KEEP threshold. BUT the ACCURACY GUARD failed: a latency change must not drop accuracy, and full temporal mAP fell to 0.12547, below the floor of 0.15132. The FM2 accuracy risk materialized — collapsing contiguous near-duplicate runs onto a single verdict lost real detections despite tau=0.97 + contiguity + max-score-representative mitigations. All three code files restored from `/Users/zzl/PycharmProjects/vad/.qoder/backups/v126/` and verified byte-identical (`diff -q` clean, no `trlm` references remain); pipeline AST-parses.

## Lessons

- **FM2 CONFIRMED (the killer):** propagating one VLM verdict across a contiguous near-duplicate run is NOT mAP-neutral in practice even at tau=0.97 + mandatory contiguity + recall-biased representative. The "mAP-neutral-by-construction" claim was too strong; run-length collapse at VLM-call granularity trades measurable detection accuracy for compute. Full mAP dropped ~0.026 below floor.
- **Efficacy/null risk REFUTED:** the anti-correlation critique (targeted segments least mergeable) did NOT hold — merges were large and pervasive (59/65 videos fired, median calls_saved high; 65% total VLM-call reduction). The smoke abort-gate correctly let it through because the efficiency was genuinely there; the failure was purely on the accuracy axis, which only the full eval could reveal.
- **Smoke was optimistic:** smoke mAP 0.18855 (above floor) did not predict full mAP 0.12547 (below floor). The 14-video smoke over-sampled easy/contiguous cases; accuracy verdicts require the full eval regardless of a passing smoke gate.
- **FM5/FM9/FM11/FM1 all clean:** marker fired and was greppable, no mps crash, no offline harness stall, boundary curve untouched. The engineering was sound; the idea's accuracy cost was the sole reason for revert.
