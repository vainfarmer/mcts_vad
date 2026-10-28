# Loop 6 — v129 BDS (Batched Directed Scan) — REVERT

**Date:** 2026-07-26
**Decision:** REVERT (ACCURACY GUARD failed — full mAP 0.11980 < floor 0.15132; a latency change must not drop accuracy)
**Baseline:** v124 Q-score (committed), session reference REF_vlm_calls=31.969 / REF_wallclock=183.11s
**Eval flags:** `--batched_directed_scan --qscore_rerank` | **Activation marker:** `[BDS]` (marker_count=26 on smoke)

---

## Proposal

**Title:** BDS — Batched Directed Scan: collapse the directed-scan's N per-segment VLM detect calls into ceil(N/B) contact-sheet calls with ABSOLUTE per-segment verdicts (K:1 call collapse over the SAME targeted set).

**Mechanism:** The dominant per-video VLM sink is `pipeline.py::_directed_llm_scan` (L1904-1975): after selecting `targeted_segments` = Source-1 (segments overlapping CLIP-loc `clip_regions`) UNION Source-2 (top-CLIP-ranked padding), truncated to `max_scan=min(max(10,len//5),18)`, it issues ONE `discovery.scan_segment` `.generate` VLM detect call PER targeted segment (asyncio.gather L1957), i.e. N=10-18 calls. BDS replaces that N-call loop with ceil(N/B) batched contact-sheet calls (default B=4): a new `AnomalyDiscovery.scan_segments_batched_independent()` chunks the IDENTICAL targeted set into batches of B, packs each segment's keyframes (reusing `_sample_keyframes`) into one `.generate` call labelled by segment index + time range, and requests a JSON ARRAY of INDEPENDENT per-segment verdicts using the byte-identical ABSOLUTE criteria of the current `scan_segment` prompt (anomaly-type bullet list + `{is_anomalous, confidence, description}`, conf>0.3 = suspicious), NOT the relative "pick the standout" framing. Verdicts unpack back onto the same `targeted_segments` and flow through the UNCHANGED `_merge_adjacent` -> candidate -> refine -> filter -> Q-score path. Per-batch fallback to individual `scan_segment` on any short/unparseable array guarantees no silently-dropped verdict.

Two controlled distinctions from the reverted v97 FCCS: (1) batches ONLY the same Source-1∪Source-2 targeted set, not full coverage (candidate pool byte-unchanged); (2) absolute per-segment decision function, not comparative. Only the API packing changes.

**Expected gain:** mean per-video VLM calls REF 31.97 -> ~22-26 (directed-scan portion ~10-18 collapses to ceil(N/4)=3-5), ~22-41% total call reduction, coverage-INDEPENDENT (K:1 collapse). mAP expected NEUTRAL, required to hold >= 0.15132.

**Risk (materialized):** FM2/FM3 — packing B segments' keyframes into one payload can flip a per-segment verdict via cross-segment visual context bleed, changing the candidate set and moving mAP under greedy match. Gated by full-eval floor 0.15132 (hard revert).

**Literature:** GHR-VLM (2026) contact-sheet packing of grounded clips; NVIDIA VSS (2026) chunk-size<->call-count tradeoff; Speculative Cascades (2026) escalate-only-on-failure fallback; in-repo v56 `scan_segments_comparative` K:1 packing primitive adapted to absolute per-segment verdicts.

---

## C's feedback

APPROVE with mandated changes — strongest latency lever this session; clears the checklist where v125/v127 failed.

- **Verified against code:** `_directed_llm_scan` (L1904) issues one `scan_segment` `.generate` per targeted segment (L1955-1958 gather), N=10-18. Collapsing to ceil(N/4)=3-5 saves ~7-13 of REF ~32 VLM calls = ~22-41%, DETERMINISTIC and COVERAGE-INDEPENDENT. Directed scan runs on ~100% of videos with >=2 targeted segments so `[BDS]` fires ~everywhere — NOT the v125 threshold-intersection no-op nor v127 empty-population 0-fire. Checklist #1 PASS.
- Architecture-level API packing, not magic-number tuning (B=4 bounded knob). #3 PASS.
- No offline harness; one full eval + [COST] — avoids FM11 (killed 4 of last 10 loops). #4 PASS.
- Config-gated default-OFF, flag-off byte-identical, real CLI flag + unique `[BDS]` marker + grep gate. Keyframes already CPU numpy so FM9 N/A. #5/#6 PASS.
- **Why not a reject (vs v97/v126):** distinctions are genuine and code-confirmed — batches ONLY the identical Source-1∪Source-2 targeted set (candidate SELECTION unchanged) and uses ABSOLUTE per-segment verdicts with NO propagation (the v126 killer). Genuinely untested point.
- **Gate hard on accuracy:** latency win near-certain and drift-free; the only open question is whether packing flips enough verdicts to breach the floor. One full eval decides.

---

## Implementation notes (D1)

v129 BDS implemented faithfully with all of C's mandated changes across 4 files (backups in `.qoder/backups/v129/`):
- `config.py` DiscoveryConfig: `batched_directed_scan: bool = False`, `directed_scan_batch_size: int = 4`; threaded in from_args.
- `batch_evaluate.py` argparse: `--batched_directed_scan`, `--directed_scan_batch_size`.
- `anomaly_discovery.py`: `scan_segments_batched_independent()` — chunking via `_comparative_batch_ranges`, packing via `_sample_keyframes`, ABSOLUTE per-segment prompt (byte-identical scan_segment bullet list; conf>0.3=suspicious via reused `_parse_comparative_response_checked`); new `_scan_batch_independent_guarded` mirrors v97 truncation guard, falls back to individual `scan_segment()` on short/unparseable batch.
- `pipeline.py` `_directed_llm_scan`: flag-guarded branch; flag-off byte-identical.

C mandates honored: #1 docstring+config comments state NOT mAP-neutral by construction (segment SET identical, per-segment VERDICTS may shift via context bleed FM2/FM3; accuracy decided SOLELY by floor 0.15132). #4 added `[BDS]` verdict-shift diag (suspicious/targeted counts -> merged candidate count per video). #6 BDS strictly inside directed-scan branch, mutually exclusive with FCCS.

Batch sizing note: `_comparative_batch_ranges` folds small remainders into near-equal batches (max B+1=5), so N=10,B=4 -> 2 calls, N=18 -> 4 calls (fewer than naive ceil). `[COST]` vlm_calls captures the drop (batched calls use `self.llm.generate`, the counted detect client). AST parse + `import pipeline` OK; `--help` shows both flags; config threading confirmed. No eval run by D1.

---

## Results

| metric | smoke | full |
|---|---|---|
| mAP | 0.14268 | **0.11980** |
| AP@0.3 | — | 0.15985 |
| AP@0.5 | — | 0.10865 |
| AP@0.7 | — | 0.09091 |
| wall-clock (s) | — | 189.09 |
| `[BDS]` marker_count | 26 | fired ~all videos |

### Drift-free [COST] vs session reference

| [COST] metric | v129 | REF (v124) | delta |
|---|---|---|---|
| mean_vlm_calls | 9.769 | 31.969 | **-69.4%** (call_win) |
| mean_clip_calls | 20.923 | 22.308 | -6.2% |
| mean_elapsed_s | 189.095 | 183.112 | **+3.3%** (wall REGRESSED, wall_win=-3.3%) |
| n videos | 65 | 65 | — |

---

## Decision: REVERT

**Reason:** The latency mechanism WORKED spectacularly and drift-free — mean VLM calls collapsed 31.969 -> 9.769 = **-69.4%**, far exceeding the >=15% KEEP gate; `[BDS]` fired on ~all videos (FM5 PASSED, marker_count=26 on smoke). Wall-clock did NOT track the call drop (+3.3%, wall_win=-3.3%, API-drift-dominated FM10 corroboration only). **But the ACCURACY GUARD FAILED: full mAP 0.11980 < floor 0.15132** — a latency change must not drop accuracy. Smoke mAP 0.14268 was already at/below floor and did NOT rescue it; the full result confirmed the breach. Code reverted: all 4 files restored from `.qoder/backups/v129/` and diff-verified byte-identical; restored config has no `batched_directed_scan` symbol (clean pre-implementation baseline, default-OFF); `import pipeline` OK. Baseline stays v124.

---

## Lessons

- **FM2/FM3 CONFIRMED as the killer, again.** Just like v126 TRLM (verdict propagation), v129 shows that ANY change to VLM payload composition on the common directed-scan path is NOT mAP-neutral in practice — even with the two hardening distinctions (identical targeted SET + ABSOLUTE per-segment prompt + NO propagation + B=4). Cross-segment visual context bleed inside a single contact-sheet payload flips enough per-segment verdicts to move the candidate set and crater mAP under greedy match (0.11980, well below floor and below even v126's 0.12547). The candidate-pool-byte-identical argument protects SELECTION but not the VERDICTS the VLM returns on packed input.
- **The VLM-call-batching family is now effectively refuted for accuracy.** v97 FCCS (relative prompt, full coverage), v126 TRLM (propagation), and v129 BDS (absolute prompt, identical set) have all breached the floor. Collapsing per-segment VLM calls saves 65-69% of calls reliably but consistently costs accuracy; the per-call independence of scan_segment is load-bearing for mAP, not incidental.
- **Smoke does NOT clear the floor (v126 lesson re-confirmed).** Smoke mAP 0.14268 correctly foreshadowed the sub-floor full result this time — but the abort-gate can only ABORT, never approve; the full-eval floor is the sole hard verdict.
- **FM5/FM9/FM10/FM11 all clean.** Real CLI flag + guaranteed `[BDS]` marker, no MPS crash (keyframes CPU numpy), drift-free call count correctly reported the true -69.4% win independent of the +3.3% wall-clock noise, one full eval decided (no offline harness). Engineering was sound; the sole revert cause is the accuracy cost on the common path.
