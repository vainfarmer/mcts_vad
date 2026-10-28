# Loop 5 — v128: VMW-SKIP (Config-gated prune of the saturated VideoMAE sliding-window stage)

**Date:** 2026-07-26
**Decision:** REVERT
**Version:** v128
**Slug:** vmw-skip-videomae-sliding-window-prune

---

## Proposal (summary)

**Title:** VMW-SKIP — Config-gated prune of the saturated VideoMAE sliding-window stage (Stage 2b), a pure post-detection LOCAL-compute sink whose output never touches the detection set (segments/count/boundaries byte-identical by construction), removing a documented near-zero-signal cost with the drift-free local wall-clock as the headline.

**Mechanism:** The DEFAULT path always runs Stage 2b (`pipeline.py:1069-1078`): `if sc.videomae_weight > 0:` calls `self.videomae.sliding_window_scores(...)`, one VideoMAE forward pass per sliding window (measured ~857 windows / 65 videos in v124 ≈ 13/video, up to 64). VMW-SKIP adds a single config gate: when `sc.skip_videomae_window` is True the whole Stage-2b block is bypassed (window centers/scores never populated) and one INFO marker `[VMWSKIP] windows_skipped=<N>` is emitted per video, with N computed by a new deterministic `VideoMAEClassifier.count_sliding_windows(...)` helper that replicates the real loop (range/break + the two 1-window fallbacks). The full-video VideoMAE classification (Stage 2 → `videomae_category`/`category_hint`/`conf_gate` → mAP) is KEPT verbatim; only the temporal sliding-window channel is pruned. Gate: `if sc.videomae_weight > 0 and not sc.skip_videomae_window:`. Flag-off = byte-identical prior behavior. NO VLM prompt, NO CLIP call, NO detection decision modified.

**mAP-neutrality by construction:** window scores are read at exactly one functional site (`frame_scorer.py:93-97`) inside `compute_frame_scores`, which runs only AFTER `final_anomalies` are fixed (Stage-5 boundaries, Stage-7b correction, conf_gate all done). With the flag on, `videomae_window_centers/scores` keys are never populated → `frame_scorer.py:95 if mae_centers and mae_scores...` no-ops gracefully → segments/count/boundaries byte-identical. The only residual mAP path is the Q-score confidence re-rank via the frame_scores curve, a 0.15-weighted near-constant (saturated ~1.0) term, largely canceled by (a) segment-overlay masking at the Q-score peak, (b) Q-score within-video min-max invariance to an additive constant.

**Expected gain:** 100% elimination of the deterministic VideoMAE sliding-window forward passes — LOCAL GPU compute only, drift-free. VLM and CLIP call counts UNCHANGED by design. mAP must hold ≥ 0.15132 floor or REVERT.

---

## C's feedback (summary)

Approved as the best-engineered, most FM-immune proposal of the session; verified in-repo that (1) window scores read at one functional site behind a graceful guard (no crash, no FM9 numpy op); (2) `compute_temporal_map` (evaluate.py:306+) is driven only by anomalies (segments+confidence), never frame_scores → detection-set byte-identity makes it mAP-neutral except via the disclosed Q-score residual; (3) config/from_args/CLI insertion points match the `qscore_rerank` pattern → clean, reversible gate. FM1/FM2 immune by construction, FM5 guarded (real flag + unique `[VMWSKIP]` marker), FM11-clean (no offline harness; one full eval validates).

**CRITICAL MAGNITUDE REALITY (pre-registered):** this mechanism changes ZERO VLM/CLIP calls — the session's primary drift-free win metric (call count) is FLAT by design. It only moves LOCAL VideoMAE compute (~17s/video mean ≈ 9.3% of the 183.1s reference), BELOW the 15% KEEP gate on the mean. By the literal KEEP rule it REVERTS on magnitude on the average, clearing 15% only on long compute-heavy CCTV videos. C approved anyway for documentary value (clean true-negative that closes the VideoMAE lever, cf. v122), but the loop must NOT be framed or reported as a call-count win.

---

## Implementation notes (D1)

Implemented per approved proposal + all of C's mandated fixes. Files edited (backups in `/Users/zzl/PycharmProjects/vad/.qoder/backups/v128/`):
- `config.py` — `ScoringConfig.skip_videomae_window: bool = False` + `from_args` threading via `getattr(args, "skip_videomae_window", False)`.
- `pipeline.py` — Stage-2b gate `if sc.videomae_weight > 0 and not sc.skip_videomae_window` + skip-branch emitting `[VMWSKIP] windows_skipped=<N>` with exact count; v128 docstring header.
- `videomae_classifier.py` — new `count_sliding_windows(nframes, window_size, stride)` mirroring `sliding_window_scores` verbatim (range/break loop + both n<num_frames and empty-windows fallbacks → exactly 1 window). Verified 200f→11 windows, 10f→<32 fallback→1.
- `batch_evaluate.py` / `run.py` — `--skip_videomae_window` argparse (visible in `--help`).

Docstring neutrality wording corrected: `compute_frame_scores` is called at 3 sites (~1039 early-Normal return, ~1131 no-candidates return — both zero-detection paths — and ~1271, the only detected-video Q-score site); dropped the inaccurate "EXACTLY ONE site" wording. All 5 edited files pass `ast.parse` and `import pipeline`. FM9 N/A (pure stage bypass + int arithmetic). VLM/CLIP call count unchanged by design.

---

## Eval configuration

- **Eval flags:** `--qscore_rerank --skip_videomae_window`
- **Activation marker:** `[VMWSKIP]`
- **Marker counts:** smoke = 14 (marker_count expected 14) ; full = 65 (fires on 100% of the 65 videos) — activation guard PASSED, no under-fire no-op.

---

## Results

### Accuracy

| Metric | Value |
|---|---|
| Smoke mAP | 0.17123617123617121 |
| Full mAP | 0.17590327830737038 |
| AP@0.3 | 0.2433620088351546 |
| AP@0.5 | 0.16181818181818183 |
| AP@0.7 | 0.1225296442687747 |
| Accuracy floor | 0.15132 |
| Floor held? | YES (0.17590 ≥ 0.15132) |
| Full wall-clock | 164.9 s |

### Cost — drift-free [COST] instrumentation (65 videos) vs session reference

| Cost metric | v128 (flag ON) | Session REF | Delta |
|---|---|---|---|
| mean_vlm_calls | 31.784615384615385 | 31.96923076923077 | -0.58% (0.6%, noise — flat by design) |
| mean_clip_calls | 22.476923076923075 | — | (unchanged by design) |
| mean_elapsed_s | 159.82461538461538 | 183.1123076923077 | -12.7% wall-clock |

---

## Decision: REVERT

**Reason:** Latency win insufficient. `max(VLM-call reduction 0.6%, wall-clock reduction 12.7%) < 15%` required by the KEEP rule. The mAP held at 0.17590327830737038 (≥ floor 0.15132) — accuracy is NOT the failure; the mechanism is mAP-neutral as designed. The revert is a clean true-negative on **magnitude**, exactly as pre-registered by both the proposal and C: the mechanism touches ZERO VLM/CLIP calls (call-count flat by design), and the LOCAL VideoMAE compute saving lands at ~12.7% wall-clock on the mean — below the 15% gate. Not worth the added config complexity for a sub-threshold, call-count-neutral local-compute trim.

Code reverted: all 6 files restored from `/Users/zzl/PycharmProjects/vad/.qoder/backups/v128/` and verified byte-identical via `diff`. `ast.parse` OK on all files post-revert; no `skip_videomae_window` / `VMWSKIP` / `count_sliding_windows` traces remain. Mechanism is fully removed (not merely default-OFF).

---

## Lessons — failure modes confirmed / refuted

- **CONFIRMED — FM10 magnitude ceiling on local-compute-only levers:** a lever that moves LOCAL GPU compute but issues no VLM/CLIP call is structurally capped below the session's primary drift-free win metric (call count is flat by design). Here it produced only a 12.7% wall-clock trim vs the 15% gate — the honest pre-registration held exactly, and the VideoMAE saturated-stage lever is now closed with documentary value (cf. v122).
- **REFUTED — accuracy-risk failure modes (FM1/FM2):** detection-set byte-identity by construction held; full mAP was identical to the held baseline (0.17590). Boundaries/count/segments never shifted; the Q-score confidence residual caused no measurable mAP change. FM1 (boundary), FM2 (greedy-match reorder) did not fire.
- **REFUTED — FM5 activation under-fire:** the real `--skip_videomae_window` flag + unique `[VMWSKIP]` marker fired on 100% of videos (smoke 14/14, full 65/65) — no silent no-op.
- **CONFIRMED — FM9 N/A:** pure stage bypass + int arithmetic, no MPS-tensor→numpy op; smoke ran clean on mps.
- **Takeaway:** for future latency levers, prioritize mechanisms that remove VLM/CLIP *calls* (the drift-free, gate-clearing metric) over LOCAL-compute pruning; local-compute trims should be reserved for compute-heavy long-video regimes where they can clear 15%, not proposed as mean-case wins.
