# v115 — PROM: Topographic Peak-Prominence Per-Detection Re-rank (REVERT)

Date: 2026-07-25
Loop: 2
Decision: **REVERT**

## Approved Proposal

**Title:** PROM: Topographic Peak-Prominence per-detection re-rank — a whole-video peak-isolation localization-quality signal stacked orthogonally on the committed v111 Q-score (single controlled change; BMS deferred to next loop).

**Mechanism:** Add ONE new per-detection, post-hoc confidence re-rank factor — the topographic PROMINENCE of each final detection's in-window peak on the already-computed fused `frame_scores` curve — stacked as an INDEPENDENT multiplicative gate immediately AFTER the committed Q-score block. Read-only on the same curve; boundaries, detection count, order, categories, NMS and every upstream stage byte-identical.

For each final detection `[t_s,t_e]` on output `frame_scores` (fs)/`timestamps` (ts):
- `in_mask = (ts>=t_s)&(ts<=t_e)`; if none -> `prom=0.0` (benign demotion).
- `peak_idx = np.where(in_mask)[0][np.argmax(fs[in_mask])]` — IDENTICAL in-window argmax to `_qscore_rerank` (avoids boundary-peak selection mismatch).
- `prom = clip(scipy.signal.peak_prominences(fs,[peak_idx])[0][0] / (fs.max()-fs.min()+1e-9), 0, 1)` — reuses Q-score's exact `vid_range` normalizer; zero new numeric constants.
- Min-max normalize `prom` WITHIN the video across that video's detections (identical protocol to Q-score); single-detection videos -> `prom_n=1.0` (inert).
- `conf' = min(1.0, conf*(prominence_base + (1-prominence_base)*prom_n))`, `prominence_base=0.5` (reuses Q-score's base; base-insensitive, zero tunable).

The peak-ISOLATION axis Q-score structurally cannot see: Q's `peak_contrast` = peak − global-out-mean (single-scale), whereas prominence = peak − nearest-HIGHER-saddle (whole-video isolation). Demotes a detection on a locally-sharp but SECONDARY bump when a taller competing peak exists elsewhere in the same video.

**Literature anchor:** AutoLoc OIC (Shou et al., ECCV 2018); BSN/BMN peak/boundary saliency ranking (Lin et al., ECCV 2018 / ICCV 2019); RTFM/MGFN feature-magnitude ranking (cited to explain WHY motion/feature-magnitude families were falsified); T3AL / EventVAD training-free curve-domain scoring.

## C's Feedback (APPROVED)

Independently reproduced the core claim via a deterministic re-score of the v114 real final detections (`analysis_v115_stack_precheck.py`). Confirmed: raw baseline 0.15013 -> Q-only 0.15632 (sanity anchor EXACT for committed v114) -> Q+prom STACK 0.15891 = **+0.00259** over Q-only, with AP@0.3 (0.23563->0.24116), AP@0.5 (0.12121->0.12273), AP@0.7 (0.11212->0.11285) ALL rising — broad-based genuine-mechanism signature. Kill-gate PASSED: 33/33 multi-detection videos have non-zero within-video prominence spread (up to 0.99) — real intra-video variance, NOT the v112 dur-fit / v110 VideoMAE zero-variance trap. Orthogonality confirmed: prom-only (0.15704) ≈ Q-only, yet the STACK (0.15891) exceeds both. FM1 immune (read-only output curve), FM2 immune (count/order frozen), FM7 satisfied (per-detection intra-video), FM8 clean (single new flag; committed `--qscore_rerank` is baseline, not confound). Modest gain, but real, noise-free, reproducible, and in the most productive family (v95 TSM / v111 Q-score lineage).

## Implementation Notes (D1)

Files: `pipeline.py`, `config.py`, `batch_evaluate.py`, `run.py`. Implemented as a pure post-hoc confidence gate stacked AFTER the committed v111 Q-score, default OFF (byte-identical to v114). Added `_prominence_rerank()` in `pipeline.py` immediately after `_qscore_rerank()`, structurally identical: same guards, IDENTICAL in-window argmax `peak_idx`, `scipy.signal.peak_prominences` on the output curve, same `vid_range` normalizer, within-video min-max normalized, `conf'=min(1, conf*(base+(1-base)*prom_n))`, base=0.5. Call site in `detect()` gated on `self.config.scoring.prominence_rerank`, placed immediately after the Q-score block. `config.py`: `ScoringConfig.prominence_rerank=False` + `prominence_base=0.5` with `from_args` getattr threading. CLI flag `--prominence_rerank` on BOTH `batch_evaluate.py` and `run.py`. INFO marker `[PROM-GATE]` emitted per video on both reranked and skipped paths (FM5 guard). All backed up to `/Users/zzl/PycharmProjects/vad/.qoder/backups/v115/`.

## Eval Flags + Activation Marker

- Eval flags: `--qscore_rerank --prominence_rerank` (qscore = committed v114 baseline; prominence = single change under test; never co-activated with `--full_coverage_scan`).
- Activation marker: `[PROM-GATE] reranked`
- Marker counts: **smoke = 12**, **full = 61** (>=1 per succeeded video; FM5 activation gate PASSED — mechanism genuinely fired in vivo).

## Results

### Deterministic offline A/B (v114 real final detections, VLM-noise-free)
| Config | mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---|---|---|
| raw baseline | 0.15013 | — | — | — |
| Q-only (v114 committed, sanity anchor) | 0.15632 | 0.23563 | 0.12121 | 0.11212 |
| Q + prominence STACK | 0.15891 | 0.24116 | 0.12273 | 0.11285 |
| **marginal delta (stack − Q-only)** | **+0.00259** | +0.00553 | +0.00152 | +0.00073 |

Kill-gate: 33/33 multi-detection videos non-zero within-video prominence spread (up to 0.99). Base-insensitive (0.3/0.5/0.7 identical mAP).

### Live eval
| Metric | Smoke | Full |
|---|---|---|
| mAP | 0.18344 | 0.17559 |
| AP@0.3 | 0.325 | 0.25019 |
| AP@0.5 | 0.135 | 0.15031 |
| AP@0.7 | 0.091 | 0.12626 |
| latency (s/video) | 230.48 | 177.62 |
| [PROM-GATE] reranked markers | 12 | 61 |

## Decision + Reason

**REVERT.** Per orchestrator decision. The deterministic offline marginal (+0.00259 over the Q-on baseline) is real and broad-based, but modest — below the strict live +0.005 headline gate, and the live full mAP (0.17559) sits inside the ±0.03 VLM-noise band, so no clear positive live signal justifies committing a new baseline this loop. The mechanism ships flag-off (byte-identical), and the code was reverted to the v114 backup so no residual surface remains. Prominence is queued as the committed-baseline foundation for the next-loop BMS signed onset/offset stack.

## Lessons Learned

- **FM1 (curve-shape regression) refuted / immune:** confirmed — a read-only post-hoc re-rank on the frozen output curve never touches boundaries/fusion; live full AP@0.5/@0.7 held (0.150/0.126) with no regression from activation.
- **FM2 (count change) immune:** count/order frozen upstream; only global confidence order moved. Confirmed by construction.
- **FM5 (activation) PASSED:** the `--prominence_rerank` flag genuinely fired (12 smoke / 61 full markers) — unlike v102/v104 opt-in-flag misfires, the launcher passed both required flags and the mechanism ran in vivo.
- **FM6 (sim-to-live gap):** the deterministic re-score is not a candidate-pool simulation (sanity anchor reproduces 0.15632 exactly), so the offline delta is faithful — but a +0.0026 offline delta is too small to resolve against ±0.03 live VLM noise, reaffirming that sub-+0.005 offline gains cannot be committed on live evidence alone. Same pattern as v107 ECE (+0.002 offline, reverted).
- **FM7 (per-detection intra-video) satisfied:** 33/33 videos showed real within-video prominence spread — NOT the v112 dur-fit / v110 VideoMAE zero-variance trap.
