# v125 PREGATE — Adaptive CLIP-confidence cascade gate on directed-scan VLM padding

**Loop:** 2 | **Version:** v125 | **Date:** 2026-07-25 | **Decision:** REVERT

## Proposal

**Title:** PREGATE: adaptive CLIP-confidence cascade gate on the directed-scan VLM padding — skip per-segment VLM detect calls on Source-2 segments the CLIP pre-screen already ranks below the video's own adaptive-normal threshold (fixed-quota padding -> signal-adaptive candidate count).

**Mechanism:** In `pipeline.py _directed_llm_scan` (Stage-4 directed detect scan, the dominant per-video VLM sink at ~10-18 of REF ~32 calls), the targeted segment set = Source 1 (segments overlapping category CLIP-loc `clip_regions`) UNION Source 2 (top-CLIP-anomaly padding, `min_extra=max(2,int(n*0.2))` by rank), capped at `max_scan=min(max(10,n//5),18)`. PREGATE converts Source 2 from fixed-quota into a FrugalGPT-style cheap->expensive cascade gate: a Source-2 segment is escalated to the qwen-vl-plus `scan_segment` detect call only if `all_scores[i] >= adaptive_thresh` (`=max(0.3, mean_s-0.5*std_s)`, the code's own already-computed normal cutoff), with an unconditional top-2-by-score recall floor. Source 1 untouched; per-kept-segment `scan_segment>0.3` rule byte-identical; Stage-5 `_clip_boundary_detection` curve untouched (mAP-neutral by construction, immune to FM1).

**Rationale:** Classic model-cascade / confidence-routing — route with a free local scorer (precomputed CLIP anomaly score, zero extra encode/API cost), spend the expensive qwen-vl-plus detect call only where the cheap scorer is not already confident the segment is normal. Source-2 currently ignores the adaptive normal cutoff it computes one block above, padding purely by rank.

**Expected gain:** ~10-20% fewer total VLM calls/video (HONEST estimate; magnitude unproven — gate only prunes segments simultaneously top-rank AND below-mean, a narrow intersection, and `max_scan` caps the total). mAP designed ~0.000; floor `>= 0.15132`.

## Reviewer C's feedback

APPROVE. All code claims independently verified against pipeline.py (1904-1944), config.py, batch_evaluate.py, [COST] marker (962). Legitimate architecture-level cascade gate, not magic-number tuning. mAP-neutrality sound (Stage-5 curve independent of scan breadth; Source-1 untouched; scan_segment>0.3 byte-identical). FM5/FM9/FM10/FM11 all handled. Default-off, reversible.

**SUBSTANTIVE CONCERN (proved correct):** the >=15% call-count KEEP gate is at real risk of a **structural no-op**. `adaptive_thresh = max(0.3, mean-0.5*std)`. The `mean-0.5*std` term sits BELOW the mean, so on a symmetric distribution >50% of segments clear it while `min_extra` only selects the top ~20% by rank — meaning the gate skips ~0 and prunes nothing. The gate only bites when the 0.3 floor BINDS (most segments < 0.3). Compounded by `max_scan` already trimming calls. Efficacy hinges on an unverified property of the CLIP anomaly-score distribution. Fine to test (mAP-neutral by construction) but must not waste two full evals on a no-op.

## Implementation notes (D1)

Faithful to proposal + C's mandated diagnostics. `config.py` DiscoveryConfig `adaptive_clip_scan_gate: bool = False` + from_args getattr threading; `pipeline.py` gated Source-2 loop + `[PREGATE]` marker; `batch_evaluate.py`/`run.py` `--adaptive_clip_scan_gate` argparse. gate=False byte-identical to v124. FM9 N/A (plain python floats). Marker logs kept, skipped, adaptive_thresh, and `top{min_extra}_below_thresh` so the smoke run alone reveals floor-binding before spending paired evals. Backups: `.qoder/backups/v125/{pipeline,config,batch_evaluate,run}.py`.

## Eval flags, activation marker, marker counts

- **Flags used:** `--qscore_rerank --adaptive_clip_scan_gate`
- **Activation marker:** `[PREGATE]`
- **Marker count smoke:** 13 | **Marker count full:** 59
- **Observed markers:** every `[PREGATE]` line logged `skipped 0 CLIP-normal seg` and `top{min_extra}_below_thresh=0` — the gate NEVER pruned a single segment. C's structural-no-op prediction confirmed exactly.

## Results

| Metric | Value | Session reference |
|---|---|---|
| Smoke mAP | 0.1764309764309764 (13 markers) | — |
| Full mAP | 0.16802435254738168 | floor 0.15132 (held) |
| AP@0.3 | 0.23481685929503768 | — |
| AP@0.5 | 0.14942148760330579 | — |
| AP@0.7 | 0.11983471074380164 | — |
| Full wall-clock | 275.5s | REF 183.11s |
| [COST] mean_vlm_calls | 32.04615384615385 | REF 31.96923076923077 |
| [COST] mean_clip_calls | 22.46153846153846 | — |
| [COST] mean_elapsed_s | 271.2492307692308 | — |

## Decision: REVERT

**Reason:** Latency win insufficient. Drift-free [COST] mean_vlm_calls = 32.046 vs same-session reference 31.969 -> **-0.2%** (essentially zero; the gate skipped 0 segments on every video). Wall-clock -48.1% is API-drift-dominated corroboration only, not the primary signal, and still far from the 15% call-count KEEP gate. `max(-0.2%, -48.1% wall-clock) < 15%` required. mAP HELD at 0.16802 (>= floor 0.15132), so no regression — this is a clean no-op REVERT, not a quality loss. The added candidate-generation complexity is not worth zero call savings.

## Lessons — failure modes

- **NEAR-NO-OP RISK: CONFIRMED.** C's exact prediction. `adaptive_thresh = max(0.3, mean-0.5*std)` puts the bar below the mean while `min_extra` only pads the top ~20% by rank; the top-ranked padding segments are the highest scorers, which always clear a below-mean threshold. Result: `skipped=0`, `top{min_extra}_below_thresh=0` on 100% of videos. The 0.3 floor never bound in this eval set. The gate is structurally a no-op unless applied to a genuinely normal-heavy, low-score-distribution regime that UCF-Crime here does not exhibit.
- **FM10 handled correctly.** Paired same-session [COST] vlm_calls (32.046 vs 31.969) was the honest arbiter and cleanly showed -0.2%, immune to the -48.1% wall-clock API-drift illusion. Trusting wall-clock would have falsely "kept" a no-op.
- **FM1/FM4 REFUTED as risks (as designed).** mAP held at 0.16802 exactly at floor-clearing level; no boundary/confidence movement — the mAP-neutrality-by-construction argument held.
- **Diagnostic-marker value proven.** The `skipped`/`top_below_thresh` fields in the [PREGATE] marker made the no-op visible in the log itself; ideally the smoke `skipped ~0` tripwire should have pre-empted the full paired eval per D1's protocol.
