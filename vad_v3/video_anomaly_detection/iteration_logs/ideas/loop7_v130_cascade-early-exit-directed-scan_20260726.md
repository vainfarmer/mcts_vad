# v130 — CASCADE: Early-exit two-stage directed VLM scan

**Loop 7 · 2026-07-26 · Decision: REVERT**

## Proposal

**Title:** CASCADE — Early-exit two-stage directed VLM scan: skip the Source-2 CLIP
safety-net padding calls when Source-1 CLIP-localized segments already fill the
`candidate_top_k` budget (with a MANDATORY smoke fire-rate pre-empt gate).

**Mechanism:** Architecturally split the monolithic directed VLM scan
(`pipeline.py _directed_llm_scan`, ~1904-1978 — the dominant per-video VLM cost at
10-18 `scan_segment` detect calls) into a two-stage conditional-computation cascade
with a data-dependent early exit, replacing the single `asyncio.gather` over the
truncated Source-1 ∪ Source-2 union.

- **Stage A (high-prior):** build `source1_segments` exactly as today (the
  `for seg in segments: for region in clip_regions` overlap loop), cap to `max_scan`,
  and await `scan_segment` only over these. Count
  `n_susp = sum(r['is_suspicious'] for r in s1_results)`.
- **Stage B (safety-net padding):** issued ONLY IF `n_susp < candidate_top_k` (=3).
  Source-2 is the fixed-quota top-CLIP padding
  (`min_extra = max(2, int(len(segments)*0.2))`, deduped vs Source-1,
  budget-remaining). When Stage A already yields ≥ `candidate_top_k` suspicious
  candidates, the top-3 downstream budget is already filled by higher-prior
  CLIP-localized detections, so all Source-2 padding scans are skipped and their VLM
  detect calls are never issued.

Every surviving `scan_segment` call keeps its ORIGINAL independent single-segment
payload — no batching, no verdict propagation, no payload change — so it is immune to
the v129/v126/v97 batching killer. Default-OFF; byte-identical to the committed
baseline when off.

**Rationale:** Source-2 top-20% padding is frequently the majority of the budget yet
exists purely as a recall net. On videos where CLIP localization is strong (Source-1
yields ≥3 suspicious), the padding calls are eliminated. Positioned as the session's
highest-leverage lever versus the banked NOCLS classify-path lever (~9-10% of total).

**Literature:** Two-stage detector conditional computation (Faster/Mask R-CNN
proposal-gates-head); CoLT5 (Ainslie et al., EMNLP 2023); LLM cascades / FrugalGPT
cheap-then-expensive routing; SCSampler / AdaFrame salient-frame selection. Borrowed
mechanism: gate the expensive stage on a cheap high-prior stage's saturation of the
downstream budget.

## C's feedback (approved into MANDATED CHANGES)

CASCADE is a legitimate architecture-level conditional-computation cascade; code
claims verified against pipeline.py; avoids the batching killer; default-OFF; one
controlled change; FM5/FM9/FM11-compliant. Two understated issues gated approval:

1. **mAP-neutrality is weaker than claimed.** Source-2 selects the TOP-CLIP-anomaly
   segments — exactly those most likely to earn a high VLM anomaly_score and out-rank
   Source-1 candidates in the coarse_score top-3. Displacement (FM2) is *plausible,
   not rare*. Fine ONLY because the full-eval floor 0.15132 is the true gate; stop
   leaning on the neutrality argument as the safety net.
2. **Structural efficacy anti-correlation (v125 no-op risk).** `max_scan` truncates
   the union: when Source-1 is large (→ likelier to fire), Source-2's real
   contribution (`max_scan − len(source1)`) is small → firing saves few calls; when
   Source-1 is small (big savings) the gate rarely fires. `fired=1` alone does NOT
   imply call reduction. The smoke pre-empt must gate on SAVINGS MAGNITUDE
   (`src2_skipped` + `[COST]` vlm delta), not merely fire-rate.

## Implementation notes (D1)

Implemented faithfully per approved proposal + all 5 mandated changes. Default-OFF;
byte-identical to committed v114/v124 Q-score baseline when off. Files:
`config.py`, `batch_evaluate.py`, `run.py`, `pipeline.py` (backups in
`.qoder/backups/v130/`).

- **config.py:** `DiscoveryConfig.cascade_directed_scan: bool = False`, threaded via
  `from_args` `getattr(args,"cascade_directed_scan",False)`.
- **batch_evaluate.py + run.py:** `--cascade_directed_scan` (action=store_true),
  verified in `--help` for both. Eval also passes `--qscore_rerank` (committed
  baseline on).
- **pipeline.py `_directed_llm_scan`:** split into `source1_segments` (CLIP-loc
  overlap loop, unchanged) and `source2_segments` (top-CLIP padding, deduped). OFF
  path byte-identical (union → max_scan → single gather → `_merge_adjacent`). ON path:
  Stage-A gathers only `source1[:max_scan]`; `n_susp` = raw pre-`_merge_adjacent`
  Source-1 suspicious; if `n_susp >= 3` skip Stage-B and emit `fired=1`, else gather
  `source2_budget` and emit `fired=0`.
- **C's mandates:** (1) `fired=1` logs `src2_skipped`; (2) docstring downgrades
  neutrality claim — floor 0.15132 is SOLE accuracy gate; (3) displacement diagnostics
  `max_skip_clip` / `min_kept_coarse`; (4) trigger uses raw pre-merge suspicious,
  non-firing path byte-equivalent to OFF; (5) flags verified in `--help`.

## Eval configuration

- **Flags:** `--cascade_directed_scan --qscore_rerank`
- **Activation marker:** `[CASCADE]` (always carries explicit `fired=0/1`)
- **Marker counts:** smoke = 13 markers (11 `fired=1`, 2 `fired=0`) over ~14 videos;
  full = 58 markers (48 `fired=1`, 10 `fired=0`) over 65 videos.

## Results

| metric | smoke | full | session REF (v124) |
|---|---|---|---|
| temporal mAP | 0.18095 | **0.17708** | 0.17076 |
| AP@0.3 | — | 0.25283 | — |
| AP@0.5 | — | 0.15858 | 0.16310 |
| AP@0.7 | — | 0.11983 | 0.12121 |
| wall-clock (s) | — | 208.6 | 183.11 |

**Drift-free [COST] (FM10 headline, 65 videos):**

| cost metric | v130 | REF | delta |
|---|---|---|---|
| mean_vlm_calls | 30.985 | 31.969 | **−3.1%** |
| mean_clip_calls | 22.292 | 22.308 | −0.1% |
| mean_elapsed_s | 205.386 | 183.112 | **+12.2%** (drift) |

## Decision: REVERT

**Reason:** The KEEP gate requires `max(VLM-call reduction, wall-clock reduction) ≥ 15%`.
- VLM-call reduction = **3.1%** (30.985 vs REF 31.969) — far below 15%.
- Wall-clock **rose** +12.2% (API-drift-dominated, FM10, not a real regression but
  certainly no latency win).
- Accuracy floor **HELD**: full mAP 0.17708 ≥ floor 0.15132 (and slightly above the
  v124 reference 0.17076 — within VLM noise; no accuracy cost). AP@0.5/0.7 within noise
  of REF.

Latency win insufficient; mAP merely held. Not worth the added architectural
complexity → REVERT. Baseline stays v124. 4 files restored from
`.qoder/backups/v130/`, diff-verified byte-identical, AST OK, ships flag-off/removed
byte-identical.

## Lessons — failure modes

- **C's concern #2 (structural efficacy anti-correlation, v125-class near-no-op)
  CONFIRMED as the killer.** Fire-rate was HIGH (48/58 full `fired=1`) yet call
  savings were tiny (3.1%): many firing videos logged `src2_skipped=0` (e.g.
  Arrest007 `fired=1 src2_skipped=0`, Arrest039 `fired=1 src2_skipped=0`) because
  `max_scan` already binds and Source-1 fills the cap — so firing did NOT imply
  savings. `fired=1` count is the WRONG headline; `src2_skipped` × fleet size is the
  real lever, and it was structurally small.
- **FM2 (displacement) REFUTED as a materialized risk this loop** — C's mandated
  reliance on the floor rather than the neutrality claim was vindicated in the safe
  direction: mAP actually held/rose. Displacement did not bite because the few skipped
  segments rarely out-ranked the CLIP-localized top-3.
- **FM10 essential again:** the drift-free −3.1% call count is the authoritative
  verdict; the +12.2% wall-clock is noise and would have mis-signalled a regression.
- **FM1/FM3/FM4 immune CONFIRMED** (boundary/CLIP curve untouched; independent
  single-segment payloads preserved — no batching-family accuracy breach unlike
  v126/v129).
- Directed-scan Source-2 padding-pruning family now shares v125's fate: the addressable
  savings are capped by `max_scan` binding on exactly the high-prior videos where the
  gate fires. NOCLS (orthogonal classify-path lever) remains banked.
