# v116 — SLM (Semantic Localization Margin) — Loop 3

**Decision: REVERT** (smoke gate FAILED — mechanism crashed, activation marker never fired)

## Summary of outcome

Per-detection Normal-subtracted CLIP text-direction re-rank, stacked as an
independent multiplicative gate immediately AFTER the committed v114 Q-score
block in `pipeline.detect()`. Default OFF = byte-identical to v114.

The smoke run FAILED the FM5 activation gate: the required guard marker
`[SLM-GATE] reranked` was found **0 times**. Root cause: the SLM code path
crashed on 12/14 smoke videos with `TypeError: can't convert mps:0 device type
tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.` —
the `clip_features` (and/or text_features) tensors live on the MPS device and
`np.asarray(...)` was called on them without a `.cpu()` hop. The only 2 videos
that ran to completion had no detections and hit the benign
`[SLM-GATE] skipped (no detections)` path (1 occurrence at log line 348), so the
fired-path marker never appeared. temporal_map mAP = 0.0 (only 2/14 videos
succeeded, neither with a true-positive detection). Full eval was skipped by the
orchestrator gate. Code reverted to the v114 backup; the mechanism ships
flag-off byte-identical.

---

## Approved proposal

**Title:** SLM — Semantic Localization Margin: per-detection Normal-subtracted
CLIP text-direction re-rank stacked on the committed v111/v114 Q-score
(AnomalyCLIP/VadCLIP direction-projection + AutoLoc-OIC lineage).

**Mechanism:** Add ONE new per-detection, read-only, post-hoc confidence re-rank
factor stacked as an independent multiplicative gate immediately AFTER the
committed Q-score block. Reads the high-dimensional pre-overlay CLIP feature
space (`video_data['clip_features']`, unit-norm, 1:1 aligned to timestamps) and
the learned category text prototypes (`anomaly_expert.text_features`, Normal
prototype = last row). Per detection with resolved category c:
`margin = F @ T[c_idx] - F @ T[-1]` (anomaly-minus-Normal direction, common-mode
cancellation of generic salience). `slm_raw = mean(margin[in_mask]) -
mean(margin[out_mask])`; within-video min-max normalize -> `slm_n`;
`conf' = min(1, conf * (base + (1-base)*slm_n))`, base=0.5. Empty/unknown/Normal
windows demoted with a worst-value sentinel. Boundaries, count, order, NMS,
fusion, half-max all byte-identical; only the global confidence ordering moves.
Ships exactly one live aggregation (plain mean); an OIC top-k arm (k=3) is an
offline-only A/B ablation, not a second flag.

**Rationale:** Attacks the dominant residual error FM7 (intra-video
mislocalization). Curve-geometry re-ranks (Q-score, v115 prominence) structurally
cannot demote a wrongly-located window because `frame_scorer.py ~L110` overwrites
the in-window curve with `np.maximum(base_signal, conf)` — the VLM confidence
floods the region and erases raw CLIP micro-structure. SLM injects an orthogonal
evidence modality on the untouched pre-overlay feature space: does this window
project onto the category text direction MORE than the rest of the same video?

**Expected gain:** deterministic offline A/B (Q-on baseline 0.15632),
hypothesized +0.004 to +0.010 mAP; honest floor +0.002–0.004 (would ship
flag-off under the +0.005 gate). Latency effectively zero (two matmuls, no new
VLM/model call).

## C's feedback

APPROVE. All load-bearing code claims verified: (1) Normal prototype IS last row
of `text_features` (L219–229); (2) `clip_features` 1:1 aligned to timestamps
(identical motion mask applied to frames and timestamps, L98–99); (3)
`_qscore_rerank` (L857) + call site (L1218–1221) + within-video min-max/ones
fallback exactly as mirrored; (4) orthogonality argument CODE-CONFIRMED —
`frame_scorer.py L110` overwrites in-window curve, so Q-score re-reads VLM
confidence while SLM reads untouched CLIP feature space. Genuinely new evidence
modality attacking FM7, not a disguised tune. FM1/FM2 immune, FM7 satisfied, FM8
clean, evidence is a deterministic offline re-score of v114's own real detections.
Honest weakness: like v115 prominence (+0.0026) and v107 ECE (+0.002), the true
effect may land below +0.005 and ship flag-off — acceptable; pre-registers the
gate that would reject it.

## Implementation notes (D1)

Implemented per approved proposal + all 4 mandated harness checks. Default OFF =
byte-identical v114. `config.py`: `ScoringConfig` gains `slm_rerank(bool=False)`
/`slm_base(float=0.5)` + from_args threading. `pipeline.py`: `_slm_rerank` added
after `_qscore_rerank`; class consts `_SLM_AGG='mean'` (live) / `_SLM_OIC_K=3`
(offline-only); call site in `detect()` after the Q-score block gated by
`config.scoring.slm_rerank`. `batch_evaluate.py` + `run.py`: `--slm_rerank`
(store_true), `--slm_base` (float 0.5). Offline harness
`analysis_v116_slm_precheck.py` re-extracts clip_features via real AnomalyExpert.
All 5 files AST-parse; imports OK; `--help` shows `--slm_rerank`. Backups at
`.qoder/backups/v116/`. No eval run by D1. NOTE: from_args threads
`qscore_rerank` with `getattr(...,False)`, so the launcher MUST pass
`--qscore_rerank` explicitly to keep the committed baseline ON.

## Eval configuration

- **Eval flags used:** `--qscore_rerank --slm_rerank`
- **Activation marker:** `[SLM-GATE] reranked`
- **Marker counts:** smoke `[SLM-GATE] reranked` = **0** (FAIL); `[SLM-GATE]`
  (any) = 1 (one `skipped (no detections)` at log L348). Full: skipped (gate).

## Results

| metric | smoke | full |
|---|---|---|
| mAP | 0.0 | -1 (skipped) |
| AP@0.3 | 0.0 | -1 |
| AP@0.5 | 0.0 | -1 |
| AP@0.7 | 0.0 | -1 |
| avg latency (s/video) | 1256.88 | -1 |
| videos succeeded | 2/14 (12 FAIL) | — |

Deterministic offline A/B was never produced (harness not run; the live smoke
crashed before any fired-path activation).

## Decision & reason

**REVERT.** FM5 activation gate FAILED: `[SLM-GATE] reranked` count = 0 despite
the required `--qscore_rerank --slm_rerank` flags. The mechanism crashed on
12/14 videos with an MPS→numpy conversion error (`np.asarray` on an mps
`clip_features`/`text_features` tensor without `.cpu()`), so it never re-ranked
a single detection in vivo. Per the pre-registered rule (0 activation markers =
automatic REVERT), code restored from `.qoder/backups/v116/` (all 4 files
diff-identical); mechanism ships flag-off byte-identical to v114.

## Lessons learned

- **FM5 refuted (true negative, but for an implementation-bug reason, not a
  no-op-flag reason):** unlike prior FM5 misses where the flag simply wasn't
  threaded, here the flag WAS wired and DID enter the code path — but the code
  raised before emitting the fired-path marker. The 0-count gate correctly
  caught it either way.
- **New failure mode surfaced: device-placement bug.** `np.asarray()` /
  `.numpy()` on MPS tensors throws `can't convert mps:0 device type tensor to
  numpy`. Any future re-rank that touches `clip_features` or `text_features`
  must `.detach().cpu().numpy()` (or `np.asarray(t.cpu())`) unconditionally, and
  the smoke must be run on the actual target device (MPS), not assumed CPU. The
  D1 notes claimed `.detach().cpu().numpy()` for text_features but the crash
  shows at least one tensor (most likely `video_data['clip_features']`) reached
  `np.asarray` still on MPS.
- **FM1/FM2 remained immune** (read-only, post-count-fix) — the crash was a hard
  error, not a curve/count regression; on the 2 no-detection videos the guard
  path behaved correctly (`skipped (no detections)`).
- **Orthogonality thesis untested in vivo** — the mechanism's core claim (SLM as
  an evidence modality independent of the confidence-flooded curve) could not be
  evaluated because it never executed. If revisited, fix the device hop first,
  then run the deterministic offline harness (the cheap, faithful, decisive
  validation) BEFORE any live smoke.
