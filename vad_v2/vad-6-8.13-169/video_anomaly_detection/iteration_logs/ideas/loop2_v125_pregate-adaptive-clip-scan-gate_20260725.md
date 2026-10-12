# v125 — PREGATE: adaptive CLIP-confidence cascade gate on directed-scan VLM padding

**Loop:** 2 (session v97 ledger) · **Date:** 2026-07-25 · **Decision:** REVERT · **Baseline stays:** v124 (`qscore_rerank=True` default, [COST] instrumentation on)

---

## Proposal

**Title:** PREGATE — adaptive CLIP-confidence cascade gate on the directed-scan VLM padding: skip per-segment VLM detect calls on Source-2 segments the CLIP pre-screen already ranks below the video's own adaptive-normal threshold (fixed-quota padding → signal-adaptive candidate count).

**Mechanism.** Target = `pipeline.py:_directed_llm_scan` (line 1904), Stage-4 directed detect scan — the dominant per-video VLM sink (10–18 of REF ~32 calls; classify ensemble ~3; FCCS default-OFF). One `discovery.scan_segment` qwen-vl-plus detect call fires PER targeted segment. Targeted set = Source 1 (segments overlapping category CLIP-loc `clip_regions`) UNION Source 2 (top-CLIP-anomaly padding, `min_extra=max(2,int(n*0.2))` by rank), capped at `max_scan=min(max(10,n//5),18)`.

The waste is Source 2: it force-adds the top-ranked padding segments by rank only, ignoring `adaptive_thresh = max(0.3, mean_s - 0.5*std_s)` already computed one block above — so on normal-heavy clips it VLM-scans segments CLIP itself scores below its own "possibly-anomalous" bar. PREGATE converts Source 2 from fixed-quota into a cheap→expensive cascade gate (FrugalGPT-style): a Source-2 segment is escalated to the VLM only if `all_scores[i] >= adaptive_thresh`, with an UNCONDITIONAL top-2-by-score recall floor. Source 1 untouched; per-kept-segment `scan_segment>0.3` judgment byte-identical; Stage-5 `_clip_boundary_detection` curve + fusion untouched.

**mAP-neutrality thesis:** (i) final boundaries+confidence come from Stage-5 on the continuous CLIP curve, independent of which segments were VLM-probed (immune to FM1); (ii) a below-threshold flagged segment becomes a low-CLIP-contrast candidate Stage-5 scores low and the category conf-gate usually drops; (iii) temporal mAP is class-agnostic and confidence-ranked, a function of surviving high-CLIP candidate geometry (all retained). 0.005 slack + top-2 floor + untouched Source-1 preserve the sensitive path.

**Expected gain (honest):** ~10–20% fewer total VLM calls/video vs same-session flag-off reference, magnitude UNPROVEN because the gate only prunes segments simultaneously top-rank AND below-mean (narrow intersection) and `max_scan` caps the total. mAP designed ~0.000; MUST hold full temporal mAP ≥ 0.15132.

## C's feedback

**APPROVE.** All code claims verified against pipeline.py (1904–1944), config.py (DiscoveryConfig ~90–100, from_args ~189), batch_evaluate.py (231–236), [COST] marker (962). Legitimate architecture-level cheap→expensive cascade gate, not magic-number tuning. mAP-neutrality sound (FM1-immune, Source-1 untouched, scan_segment>0.3 byte-identical → FM4 avoided). FM5/FM9/FM10/FM11 all handled.

**SUBSTANTIVE CONCERN (prophetic):** the ≥15% call-count KEEP gate is at real risk of a **structural no-op**. `adaptive_thresh = max(0.3, mean-0.5*std)`; the `mean-0.5*std` term is BELOW the mean, so on a symmetric distribution >50% of segments clear it while `min_extra` selects only the top ~20% by rank → gate skips ~0, prunes nothing. Gate only bites when the 0.3 floor BINDS (most segments < 0.3), the asserted-but-unverified "normal-heavy" regime. Compounded by `max_scan`: savings only materialize when N1 + gated-Source-2 sits BELOW the cap. mAP-neutral by construction so worst case is a clean sub-15% REVERT — but must NOT waste two full evals discovering a no-op.

## Implementation notes (D1)

Faithful to approved proposal + all C mandates. Files edited: `config.py` (`DiscoveryConfig.adaptive_clip_scan_gate: bool = False` + from_args getattr threading), `pipeline.py` (Source-2 padding loop → gated cascade + `[PREGATE]` marker + v125 docstring), `batch_evaluate.py` and `run.py` (`--adaptive_clip_scan_gate` argparse). gate=False path byte-identical to v124. `[PREGATE]` marker logs kept, skipped, adaptive_thresh, and top{min_extra}_below_thresh so the smoke run alone reveals whether the 0.3 floor binds. FM9 N/A (plain python floats from `batch_score_segments`). Backups: `.qoder/backups/v125/{pipeline,config,batch_evaluate,run}.py`.

## Eval flags, activation marker, marker counts

- **Flags used:** `--qscore_rerank --adaptive_clip_scan_gate`
- **Activation marker:** `[PREGATE]`
- **Marker counts:** smoke = **13** · full = **59** (activation confirmed: FM5 PASSED, gate fired in vivo; not a no-op at the code level — but a near-no-op at the savings level, see below)

## Results

| Metric | v125 (gate ON) | Reference (v124 session, gate-off) |
|---|---|---|
| smoke mAP (marker_count=13) | 0.1764309764309764 | — |
| full mAP | 0.16802435254738168 | 0.17076 (v124) |
| AP@0.3 | 0.23481685929503768 | — |
| AP@0.5 | 0.14942148760330579 | 0.16310 (v124) |
| AP@0.7 | 0.11983471074380164 | 0.12121 (v124) |
| full wall-clock | 275.5 s | REF 183.11 s |

### Drift-free [COST] (over 65 videos)

| [COST] metric | v125 (gate ON) | SESSION REFERENCE (v124) | Delta |
|---|---|---|---|
| mean_vlm_calls | 32.04615384615385 | 31.96923076923077 | **+0.24% (−0.2% "reduction" ⇒ effectively zero / slightly more calls)** |
| mean_clip_calls | 22.46153846153846 | 22.308 | ~flat |
| mean_elapsed_s | 271.2492307692308 | REF_wallclock 183.1123076923077 | −48.1% wall-clock (API-drift-dominated, corroboration only) |

## Decision + reason

**REVERT.** Latency win insufficient: `max(VLM-call −0.2%, wall-clock −48.1%) < 15%` required. The drift-free primary signal (mean_vlm_calls) did NOT drop — it rose +0.24% vs the same-session v124 reference, so the ≥15% call-reduction KEEP gate FAILED decisively. The −48.1% wall-clock is API-drift-dominated (FM10) and cannot be trusted as the headline. Accuracy floor HELD (full mAP 0.16802 ≥ floor 0.15132) — the mechanism is mAP-neutral exactly as designed — but neutrality alone does not justify the added candidate-generation complexity without a real efficiency win. Code restored from `.qoder/backups/v125/`, all 4 files diff-verified byte-identical; ships flag-off inert; baseline stays v124.

## Lessons — failure modes confirmed / refuted

- **C's structural no-op concern CONFIRMED (the decisive outcome).** The gate activated (13 smoke / 59 full `[PREGATE]` markers — FM5 refuted as a false positive, real activation) yet pruned essentially nothing: mean_vlm_calls +0.24%, not the hoped ≥15% drop. The `mean-0.5*std` threshold sitting below the mean while `min_extra` selects only the top ~20% by rank means the top-rank ∧ below-threshold intersection is near-empty, and `max_scan` already binds the call count. The narrow-intersection near-no-op risk both reviewers flagged materialized.
- **FM1 immune — CONFIRMED.** Stage-5 boundary curve independent of scan breadth; full mAP held at 0.16802, no geometry regression.
- **FM2 (detection-count TP/FP flip) — refuted / did not fire.** Accuracy floor held; top-2 recall floor + untouched Source-1 kept the detection set stable.
- **FM10 — CONFIRMED essential.** Wall-clock showed a spurious −48.1% "win" that the drift-free [COST] vlm_calls headline correctly overrode; without the same-session paired reference this would have been a false KEEP.
- **Process win:** the `[PREGATE]` kept/skipped/top-below-thresh diagnostic + the mandated smoke tripwire gave the honest arbiter; this was a clean mAP-neutral true-negative REVERT (like v122), not a regression.
- **Forward:** efficiency levers on the directed-scan sink that gate on the code's own `adaptive_thresh` are structurally near-no-op given the `mean-0.5*std` vs top-20%-rank geometry and the `max_scan` cap. A future call-reduction attempt must either (a) lower/replace the threshold to actually bind, or (b) target `max_scan` / Source-1 breadth directly — not pad-gate the already-capped Source-2 tail.
