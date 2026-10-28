# v131 — PAST: Prior-Adaptive Scan Truncation

**Loop:** 1 | **Date:** 2026-07-26 | **Decision:** REVERT

## Proposal (summary)
Replace the blind `max_scan[:N]` truncation in `pipeline.py:_directed_llm_scan` (def line 1904) with a
provably-null truncation that keeps every Source-1 segment and every CLIP>=0.3 Source-2 segment, dropping
only the below-threshold Source-2 rank-tail detect (qwen-vl-plus) calls.

- **Mechanism:** Architecture-level DROP-CALL on the dominant detect path. Build an index-aligned
  `targeted_is_protected` list (Source-1 appends push True; Source-2 appends push
  `all_scores[i] >= clip_threshold`). Because Source-2 is appended in DESCENDING all_scores rank, all
  >=threshold segments occupy the FRONT of the list. Truncate to
  `new_cap = min(old_cap, max(candidate_top_k + scan_floor_margin, n_protected))` so the front
  (all Source-1 ∪ all Source-2>=0.3) + a >=5 recall floor is retained byte-identical; only proven
  below-0.3 rank-tail calls are declined. Flag off => byte-identical `targeted_segments[:old_max_scan]`.
- **old_cap** = `min(max(10, len(segments)//5), 18)` (a CEILING, not a floor).
- **Flags:** `--adaptive_scan_truncation` (master switch, default False) + `--scan_floor_margin` (default 2).
- **Markers:** per-fire `[PAST] video n_targeted=... n_protected=... old_cap=... new_cap=... dropped=...`
  and process-level `[PAST-AGG] fire_count=... mean_dropped=... mean_n_targeted=...`.
- **Literature:** FrugalGPT (Chen et al., TMLR 2024); speculative/latency-optimal cascades; SIFT
  frozen-gate routing — all motivate declining a provably-below-threshold expensive call.

## C's feedback (verification gate)
VERIFIED against code: FM12-clean (pure whole-call drop on detect path; survivors byte-identical; no
merge/batch/propagation); the session's endorsed Untouched Lever #1 (adaptive max_scan). Structural claims
confirmed: `_directed_llm_scan` runs once/video (pipeline.py:1121); Source-2 appended top-min_extra by
descending rank with NO threshold test (1934-1940), so all S2>=0.3 provably sit ahead of all S2<0.3.

**Mandated correction (checklist #3 FAILED as written):** the "mAP-neutral BY CONSTRUCTION" claim is FALSE.
`scan_segment` sets `is_suspicious` on the VLM's OWN confidence>0.3, INDEPENDENT of CLIP
(anomaly_discovery.py:245); candidates ranked by VLM confidence and cut to candidate_top_k=3
(pipeline.py:1149-1153). A dropped CLIP<0.3 tail segment could have been VLM-flagged into the top-3. So
mAP-neutrality is an EMPIRICAL bet (low-CLIP rarely yields top-k VLM verdicts), not "by construction";
the single full eval is the adjudicator. Approved with all 7 corrections applied by D1.

## Implementation notes (D1)
Implemented per approved proposal + all 7 of C's mandates. Removed "mAP-neutral by construction" language
(replaced with detection-contract-bet framing + explicit CLIP-independence note). Hard recall floor
`>=5` enforced via assert. Per-fire + atexit aggregate markers. Uses `self.config.discovery.clip_threshold`
(not hardcoded 0.3), logged in marker. Phantom-fire guard: fires only when the Source-2 scores block ran.
Backups made before editing. Q-score default-ON and v51 half-max boundary untouched; flag default False =
byte-identical baseline. Config threaded through `VADConfig.from_args` via getattr; CLI added to
batch_evaluate.py and run.py. AST-parse + import + `--help` validated (no eval run by D1).

## Flags / markers / marker counts
- Eval flags: `--adaptive_scan_truncation --qscore_rerank`
- Smoke: `[PAST]` per-fire markers = **11**; `[PAST-AGG] fire_count=11 mean_dropped=0.00 mean_n_targeted=34.64`
- Full: `[PAST]` per-fire markers = **42**; `[PAST-AGG] fire_count=42 mean_dropped=0.00 mean_n_targeted=27.10`
- **KEY FINDING:** the flag FIRED (42 videos) but `mean_dropped=0.00` — the truncation window never
  actually shed a call. old_cap already binds at real content on essentially every video, so
  `max(candidate_top_k+margin, n_protected)` >= old_cap. This is the v125/v130 max_scan-binding /
  magnitude-dilution failure mode, on otherwise virgin ground.

## Results

| Metric | Value | Reference |
|---|---|---|
| smoke mAP | 0.17340067340067342 (markers=11) | — |
| full mAP | 0.1716313096085242 | prior baseline 0.1716313096085242 (held) |
| AP@0.3 | 0.2305461027386161 | — |
| AP@0.5 | 0.16181818181818183 | — |
| AP@0.7 | 0.1225296442687747 | — |
| frame AUC | 0.7139672628311464 | — |
| wall-clock (avg/video) | 164.16s | — |
| **mean_vlm_calls (drift-free)** | **31.876923076923077** | **REF 31.969** |
| mean_clip_calls (drift-free) | 22.184615384615384 | — |
| mean_elapsed_s (drift-free) | 161.64 | — |
| accuracy floor | mAP 0.1716 >= 0.15132 (HELD) | floor 0.15132 |

## Decision: REVERT
- **VLM-call reduction:** (31.969 - 31.8769) / 31.969 = **0.29%** — far below the 15% required.
  wall-clock 11.7% is API-drift noise, NOT decisive (FM10).
- **Accuracy floor:** HELD — full mAP 0.1716313096085242 == prior baseline, well above 0.15132. No
  accuracy regression, but no cost win either.
- **Root cause:** `mean_dropped=0.00` across all 42 fires. old_cap (`min(max(10,len//5),18)`) already
  binds at real content, so the protected front + recall floor equals or exceeds old_cap on every video;
  there was no below-0.3 rank-tail left for old_cap to have kept in the first place.

## Lessons (FMs confirmed/refuted)
- **FM12 (merge/batch corruption): AVOIDED/refuted** — pure whole-call drop; survivors byte-identical;
  full mAP identical to baseline confirms no payload corruption.
- **FM2 (candidate-pool/verdict -> FPs): not triggered** — mAP held exactly; but moot since nothing dropped.
- **FM10 (decide on drift-free vlm_calls, not wall): confirmed essential** — wall-clock showed a spurious
  ~11.7% "win" that vlm_calls (0.29%) exposed as drift noise. Deciding on wall would have been a false KEEP.
- **FM5 (real flag + unique marker + grep gate): confirmed valuable** — the `[PAST-AGG] mean_dropped=0.00`
  line is exactly what distinguished "fired-but-null" from "never-fired", pinpointing the dilution.
- **Magnitude-dilution (v125/v130 max_scan-binding): CONFIRMED AGAIN** — the ceiling `min(max(10,len//5),18)`
  empirically binds at content; the below-threshold Source-2 tail it supposedly retained does not exist in
  practice, so there is nothing left to trim. This is now a three-time-confirmed dead end (v125, v130, v131):
  any lever that only trims the max_scan tail is null. Future scan-reduction ideas must attack the
  content-bound region itself (e.g. protected-count itself, or Source-1/candidate_top_k sizing), not the tail.

## Artifacts
- Report (this file): `/Users/zzl/PycharmProjects/vad/video_anomaly_detection/iteration_logs/ideas/loop1_v131_past-prior-adaptive-scan-truncation_20260726.md`
- Smoke log/eval: `iteration_logs/results/v131_smoke_run.log`, `v131_smoke_eval.json`
- Full log/eval: `iteration_logs/results/v131_full_run.log`, `v131_full_eval.json`
- Backups restored from `/Users/zzl/PycharmProjects/vad/.qoder/backups/v131/` (mechanism fully removed; verified byte-identical via diff).
