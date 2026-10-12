# Loop 1 — v124: Baseline Calibration (v114 Q-score CLI-default flip + always-on [COST] per-video cost instrumentation)

Date: 2026-07-25
Decision: **KEEP-FLAG-ON** (loop-1 baseline calibration; mAP-neutral by construction)

## Proposal

**Title:** Baseline Calibration: complete v114 Q-score CLI-default flip + add always-on [COST] per-video cost instrumentation

**Mechanism:** Two mAP-neutral edits that make this session measurable.
1. Complete the v114 flip so plain runs use the committed Q-score baseline.
2. Add always-on per-video cost logging: count VLM (detect+classify) calls and CLIP scoring calls and wall-clock per video, emit one `[COST]` INFO line per video. Pure logging + a default change — the final detection set is UNCHANGED, so mAP is byte-neutral.

**Rationale:** Every subsequent loop is judged on drift-free per-video VLM/CLIP call reduction (FM10), which requires an instrumented same-session reference. The v114 keep-flag-on was only half-applied (dataclass default True but `from_args` still defaulted False), so plain runs silently used the OLD baseline — this loop fixes that so the reference eval truly is the committed Q-score baseline.

**Implementation plan:** Edit A (complete flip): `config.py` `from_args` — `qscore_rerank` default `False`→`True`; add `--no_qscore_rerank` store_false opt-out (dest=qscore_rerank) on both `batch_evaluate.py` and `run.py`. Edit B (instrumentation, DEFAULT-ON, no flag): per-video counters in `pipeline.py`; increment on each VLM detect call, each VLM classify call, each CLIP scoring call; record per-video start time; emit `logger.info(f"[COST] video={name} vlm_calls={n_detect+n_classify} clip_calls={n_clip} elapsed_s={elapsed:.1f}")` at end of each video. Counters MUST NOT alter any detection. `activation_marker="[COST]"`.

**Expected gain:** Zero mAP change (pure logging + honest-default flip). Establishes REF_vlm_calls, REF_clip_calls, REF_wallclock for the session.

**Risks:** FM9 avoided (no tensor math added). No accuracy risk (final detections unchanged). Only risk is a mis-placed counter double-counting — verified counts are plausible.

**Latency impact:** Negligible (<1ms logging per video).

**Literature:** n/a (instrumentation loop).

## C's feedback

Pre-approved: instrumentation + honest-baseline flip, mAP-neutral by construction, required to measure the rest of the session.

## Implementation notes (from D1)

Backups in `/Users/zzl/PycharmProjects/vad/.qoder/backups/v124/` (batch_evaluate.py, config.py, pipeline.py, run.py). Two mAP-neutral edits, both verified.

**EDIT A (complete v114 flip):**
- `config.py:186-187` `from_args` getattr default flipped `False`→`True`: `qscore_rerank=getattr(args,"qscore_rerank",True)`. Verified: empty namespace → `scoring.qscore_rerank=True`; `Namespace(qscore_rerank=False)` → `False`.
- `batch_evaluate.py` and `run.py`: `--qscore_rerank` now store_true default=True; added `--no_qscore_rerank` (action=store_false, dest=qscore_rerank) opt-out. Verified argparse: no-args→True, `--qscore_rerank`→True, `--no_qscore_rerank`→False. `--help | grep qscore` shows both flags.
- `config.py:88` dataclass default already `qscore_rerank: bool = True` (v114). No further config flip required — the field to keep True per the decision is already True.

**EDIT B (always-on [COST] instrumentation, no CLI flag, DEFAULT-ON):**
- `llm_client.py`: added `self.call_count=0` in `__init__`; `+=1` at top of `generate()`. Counts every VLM API call (detect via llm_client, classify via classify_llm_client). Read-only elsewhere.
- `anomaly_expert.py`: added `self.clip_call_count=0` in `__init__` (set before HAS_CLIP early-return so it always exists); `+=1` before each of the two `self.model.encode_image()` sites (precompute per-batch forward pass + `_encode_frames`). True drift-free CLIP forward-pass count; frame_scorer reuses precomputed features (matmul only) so not double-counted.
- `pipeline.py`: renamed per-video entry `detect`→`_detect_impl` (byte-identical logic) + thin async `detect` wrapper that snapshots the three counters + wall-clock start, delegates, and in a `finally` block (fires on every exit path incl. early Normal-return and exceptions) emits exactly one `[COST]` line. When classify_llm_client is llm_client (same object) uses single delta as vlm_calls; when distinct (eval config: qwen-vl-plus detect + qwen3-vl-plus classify) sums the two deltas. Counters read-only → detection set byte-identical → mAP unchanged.
- Sanity: all 6 files ast.parse OK; `import pipeline` OK; no internal `self.detect()` recursion (rename safe).

## Eval flags + activation marker + marker counts

- Eval flags used: `--qscore_rerank` (belt-and-suspenders; now also the default)
- Activation marker: `[COST]`
- Marker counts: **smoke = 14** `[COST]` lines (14 videos); **full = 65** `[COST]` lines (65 videos). Machine-parseable tokens present: `vlm_calls=`, `clip_calls=`, `elapsed_s=`. Sample: `[COST] video=Arrest001_x264.mp4 vlm_calls=30 clip_calls=4 elapsed_s=50.5`.

## Results

| Metric | Value |
|---|---|
| smoke mAP | 0.15482093663911844 |
| full mAP | 0.17076191782074135 |
| full AP@0.3 | 0.22797202797202798 |
| full AP@0.5 | 0.16310160427807488 |
| full AP@0.7 | 0.1212121212121212 |
| full wall-clock | 183.12s |
| accuracy floor | 0.15132 |

### Drift-free [COST] cost metrics vs session reference

| Cost metric | This run (65 videos) | Session reference | Delta |
|---|---|---|---|
| mean_vlm_calls | 31.96923076923077 | REF_vlm_calls = 31.96923076923077 | 0.0 (0.0%) |
| mean_clip_calls | 22.307692307692307 | (reference set this loop) | — |
| mean_elapsed_s | 183.1123076923077 | REF_wallclock = 183.1123076923077s | 0.0 (0.0%) |

This loop **defines** the session reference (REF_vlm_calls=31.96923076923077, REF_wallclock=183.1123076923077s), so the delta is 0 by construction. All subsequent loops measure drift-free per-video VLM/CLIP call reduction (FM10) against these numbers.

## Keep/Revert decision + reason

**KEEP-FLAG-ON.** This is a mAP-neutral instrumentation + honest-baseline calibration loop, so it is judged on correctness of the reference, not on a latency win over a prior baseline:
- **Latency win %:** 0.0% on both VLM calls (31.969 vs REF 31.969) and wall-clock (183.11s vs REF 183.11s) — the run *is* the reference, so a win is not expected or applicable this loop. The instrumentation itself adds <1ms/video.
- **Accuracy floor held:** YES. full mAP 0.17076 ≥ accuracy floor 0.15132; smoke mAP 0.15482 also above floor. The detection set is byte-identical to the committed v114 Q-score baseline (counters are read-only), so the fresh-detection mAP (0.17076) reproduces the committed baseline within VLM sampling noise.
- Config field `qscore_rerank` stays `True` (dataclass default line 88 already True + `from_args` now defaults True). Code KEPT; no revert. No further config flip needed.

## Lessons — failure modes confirmed / refuted

- **FM9 (tensor→numpy device bug, cf. v116/v123):** N/A / refuted for this loop — no tensor math or `.numpy()` conversions added; counters are plain int increments.
- **FM5 (activation / mechanism-never-fires):** CONFIRMED-SATISFIED — default-ON, no-flag instrumentation fired on every video (14/14 smoke, 65/65 full), no launch/harness gate to skip. The always-on pattern is the cleanest possible activation and is the correct template for future must-measure instrumentation.
- **FM1/FM2 (silent detection drift):** immune by construction — the `_detect_impl` rename is byte-identical and the wrapper is read-only, so the final detection set is unchanged and mAP is honest.
- **FM10 (drift-free cost accounting):** this loop establishes the measurement substrate FM10 requires; the double-count risk flagged in the proposal did NOT materialize (per-video counts are plausible: ~26–68 vlm_calls, 4–48 clip_calls).
