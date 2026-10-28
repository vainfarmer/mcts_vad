# Loop 2 — v139 GPC pooled-peak fix: REVERT, and the closure that matters

**Date:** 2026-07-28
**Verdict:** REVERT. Paired in-vivo dmAP **-0.00149** (gate required >= +0.005).
**Disposition:** all 4 files restored from `.qoder/backups/v138/`, diff-verified byte-identical,
0 `gpc` symbols remain, `import pipeline` OK. Ships as **v137**; committed baseline mAP 0.15632
unchanged; floor 0.15132 held (0.16450).

---

## 1. What was changed (one controlled change vs v138)

`gpc_pool_bins` (default 32) average-pools the DENSE per-frame category curve into 32 bins and takes
the peak BIN's center, instead of the raw per-frame argmax. `gpc_pool_bins=0` reproduces the refuted
v138 behavior. Everything else identical to v138.

Motivation was measured, not guessed: in v138's own paired diagnostic, substituting a pooled peak on
the SAME live detections moved the delta from **-0.00287 to +0.01424**. A deterministic unit test
(synthetic curve: sustained 0.7 plateau at 10-40 s + lone 0.99 spike at 100 s) confirmed the
inversion — v138 demotes the plateau window and keeps the spike; v139 does the opposite.

## 2. Result — paired A/B on 99 live detections (66 videos)

| arm | mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---|---|---|
| OFF (conf_old restored from the log) | 0.16599 | 0.23015 | 0.14798 | 0.11983 |
| ON (pooled peak, bins=32) | 0.16450 | 0.23099 | 0.14713 | 0.11538 |
| **paired delta** | **-0.00149** | +0.00084 | -0.00085 | -0.00445 |

Activation clean: 99 `[GPC]` fires, every line carrying `bins=32`, 99/99 matched by the paired
harness (FM5 satisfied).

**The fix worked at the signal level and still did not move mAP:**

| | v138 (dense peak) | v139 (pooled peak) | offline probe |
|---|---|---|---|
| demotion precision | 0.815 | **0.850** | 0.887 |
| false-demotion rate on true hits | 54.5% (12/22) | **37.5% (9/24)** | 30.1% |
| paired dmAP | -0.00287 | -0.00149 | (+0.0191 predicted) |

Both signal-quality metrics moved toward the offline values, exactly as the diagnosis predicted, yet
the AP checkpoint precisions (the only four numbers 11-point AP at n_gt=66 can see) did not improve.

## 3. The closure — FM15 generalized: RESOLUTION was necessary, SUBSTRATE is the real gap

Matching the offline probe's temporal resolution was **necessary but not sufficient**. The residual
difference is the **feature substrate**:

- offline battery curve: `features/test/<vid>.npy` — 32 pre-extracted snippet embeddings, produced by
  a separate extraction script (unknown frame sampling / aggregation), then `get_category_scores`.
- in-vivo curve: live per-frame CLIP encoding of **motion-filtered, fps-sampled** frames, then the
  same `get_category_scores`.

Pooling the live curve to 32 bins does **not** reconstruct the snippet curve. Direct proof, same rule
and same live detections, only the peak's provenance differing:

| peak source | paired dmAP |
|---|---|
| features/test 32-snippet curve (v138 diagnostic) | **+0.01424** |
| live dense curve pooled to 32 bins (v139, in vivo) | **-0.00149** |

**Consequence — the load-bearing lesson of this session:** *an offline precheck of any CURVE-DERIVED
signal built on `features/test` is not predictive of in-vivo behavior, however many caches agree.*
Here 8/8 independent caches agreed at every weight, with two decisive placebos (random-binary
-0.0248 at matched fire rate, wrong-category -0.0046), and it was still wrong **twice**. Cache
agreement bounds noise; it says nothing about a substrate that is uniformly wrong.

**Concrete remedy for future loops (cheap, ~2 lines):** cache the live `get_category_scores` curve in
the per-video artifact during any run. It is already computed in vivo; what gets saved today is only
the detection-FLOODED `frame_scores` (`frame_scorer.compute_frame_scores`, frame_scorer.py:63+),
which is partly circular with confidence and useless as a re-rank precheck substrate. With the raw
live curve cached, every curve-signal precheck becomes faithful and this failure mode disappears.

## 4. Status of the accuracy arc after two loops

- **GPC / peak-containment family: CLOSED in vivo** (2 refutations, both on paired measurement).
- The peak-containment family was also shown **saturated offline** (no stacking variant clears
  +0.005 — see loop 1 memo §7b), so nothing remains to try inside it.
- The **offline-measured** cross-video-level headroom (+0.0518 oracle; GPC appeared to harvest 37%)
  is **NOT confirmed reachable** — it was measured on the same discredited substrate.
- What survives as trustworthy (measured on real cached detections, not on features/test):
  pure re-rank oracle ceiling **0.27273**; within-video oracle only **+0.006** and ~99% already spent
  by the committed Q-score (this explains the v115-v123 nine-rejection arc); global-ranking
  decomposition into 4 checkpoint precisions.
- **Recommended loop 3 (not started):** land the raw-live-curve caching change FIRST as a
  measurement-infrastructure loop (mAP-neutral by construction, no mechanism), then re-derive the
  cross-video-level candidates on a faithful substrate. Any further curve-signal lever evaluated
  before that infrastructure exists is untrustworthy by the finding above.

## 5. Run-integrity notes (disclosed)

- Part 1 of the eval stalled at 46/65 for ~70 min because the **system** exhausted swap (30.98 GB of
  31.74 GB used, ~135 MB RAM free; the process sat in uninterruptible sleep). The same video had
  completed in 740 s during v138. Not a code fault.
- Resumed at `--concurrency 2` (the runner skips videos whose JSON exists); all 65 videos completed.
  Both command lines are recorded in `results/batch_eval_v139_full/COMMAND.txt`.
- The user stopped the session before `evaluation_report.json` was written, so the figures above were
  computed offline from the 65 completed per-video artifacts plus the run log — the paired A/B needs
  exactly those two inputs, so the verdict is unaffected.
- Artifacts: `iteration_logs/results/v139_full_run.log`, `v139_paired_ab.json`,
  `v139_full_run_part1.log`.
