# v135 — DETECT-PATH VERDICT-INDEPENDENCE CLOSURE (joint A+B)

Loop 5 (this block: v131..v135). Date: 2026-07-26.
Type: **SOURCE-VERIFIED ANALYTICAL SESSION-CLOSE. ZERO CODE. ZERO EVAL.**
Status: APPROVED closure (discharges C's escape clause). Baseline unchanged = v124 (full mAP 0.17076 / committed 0.15632). Floor 0.15132 trivially held (nothing changed).

---

## What this loop is

C's mandate for the latency session was a detect/scan-path DROP-CALL whose provably-null / redundancy
argument is **VERDICT-INDEPENDENT**: dropping the call must leave the FINAL detection set (count,
boundaries, AND scores) byte-identical for BOTH possible VLM outcomes, so mAP is neutral by
construction and the whole win is measured purely by the drift-free [COST] VLM-call drop.

C set an explicit escape clause: *"If no such verdict-independent dominance invariant can be
constructed, say so explicitly and stop proposing on this eval set rather than submitting another
sub-gate classify/early-exit measurement."*

This memo is that closure. It proves that **no strictly verdict-independent detect-drop can exist on
the detect path as currently wired**, so KEEP-gate requirement (a) is UNSATISFIABLE on that path.
No code is written and no eval is run (FM11 economy: the one full eval is NOT spent on a
predetermined-REVERT measurement).

---

## The three load-bearing source facts (VERIFIED against source, not just cited)

1. **Surviving-detection confidence is STRICTLY MONOTONE INCREASING in coarse_score.**
   `pipeline.py:2225-2228`
   ```
   _coarse = cand.get('coarse_score', 0.8)
   _base_conf = _coarse * 0.80
   _boundary_bonus = _coarse * 0.20 * min(1.0, best_score / 0.5) * _tsm_mod * _onset_mod
   result['confidence'] = min(1.0, _base_conf + _boundary_bonus)
   ```
   BOTH `_base_conf` and `_boundary_bonus` multiply `_coarse`, so `confidence` rises whenever
   `coarse_score` rises (the CLIP `best_score`, `_tsm_mod`, `_onset_mod` factors are all >= 0 and
   verdict-independent). Confidence = `coarse_score x (0.80 + 0.20 x CLIP_bonus)`.

2. **A merged region's coarse_score = MAX over its scanned segments' VLM verdicts.**
   `anomaly_discovery.py:245` — a segment becomes a candidate iff
   `'is_suspicious': is_anomalous and confidence > 0.3`, where the VLM-returned confidence in
   `(0.3, 1.0]` is the `coarse_score`.
   `anomaly_discovery.py:464-484` — `_merge_adjacent` merges gap<=2s neighbours, keeping at `:478`
   `'coarse_score': max(last['coarse_score'], cand['coarse_score'])`.

3. **compute_temporal_map ranks ALL predictions GLOBALLY by confidence before greedy IoU.**
   `evaluate.py:342` — `all_predictions.sort(key=lambda x: x['confidence'], reverse=True)`.

Detect call is issued at `pipeline.py:1956` (`self.discovery.scan_segment`) inside
`_directed_llm_scan` (1904-1975), one per targeted segment (Source-1 CLIP-loc overlap + Source-2
top-CLIP pre-screen), capped at `max_scan = min(max(10, len//5), 18)` (~line 1942). top_k=3 sort on
coarse_score at `pipeline.py:1149-1153`. Boundary half-max crossing at `pipeline.py:2101-2113`.

---

## The proof (why requirement (a) fails on the detect path)

A droppable detect call is `scan_segment` on a targeted segment S that we want to prove cannot alter
the final detection set. For mAP-neutrality-by-construction, the drop must be byte-identical on
COUNT, BOUNDARIES, and SCORE for BOTH of S's possible VLM outcomes (suspicious / not).

- **BOUNDARIES — SAFE (the only verdict-independent dimension).** The half-max crossing
  (`pipeline.py:2101-2113`) is derived purely from the CLIP/MIL score curve inside a search window,
  independent of `coarse_score`. A drop never moves boundaries.

- **SCORE — NOT SAFE.** Surviving confidence = `coarse_score x (0.80 + 0.20 x CLIP_bonus)`, monotone
  in `coarse_score` (fact 1), and a region's `coarse_score` = MAX over its merged scanned segments'
  VLM verdicts (fact 2). Dropping S removes S's `coarse_score` from that MAX. If S's un-run,
  VLM-determined verdict would have been the region maximum, the surviving confidence DROPS; because
  the AP ranking is GLOBAL across all videos (fact 3), that shift reorders the precision-recall curve
  and moves mAP. Whether it shifts is decided entirely by S's un-run VLM output — the definition of
  verdict-DEPENDENT.

- **COUNT — NOT SAFE.** The number of survivors after top_k=3 depends on how many scanned segments
  turn out suspicious, which is verdict-determined. Proving pre-call that S is dominated out of the
  top-3 would require >= 3 OTHER candidates guaranteed `coarse_score >= 1.0` — unknowable before their
  own VLM calls return. The lone theoretical exception (S cut by top_k=3 even at coarse_score=1.0)
  needs exactly that unknowable-pre-verdict condition.

The three families C pointed at all fail on this same chain:
(i) overlap/score dominance — fails (coarse_score IS the verdict, merge takes max, mAP ranks on it);
(ii) top_k=3 saturation dominance — fails (surviving count is verdict-determined);
(iii) CLIP-peak co-location — fails on SCORE for the same max-coarse reason.

Hence **requirement (a) is unsatisfiable for every droppable detect call** as the pipeline is wired.
This is mAP-neutral-by-construction turned against itself: the per-call VLM verdict independence that
FM12 protects for *surviving* calls is exactly what makes a *dropped* call's absence verdict-visible
in the global mAP ranking.

---

## SCOPE QUALIFIER (verbatim — this closure is NOT architecture-eternal)

> unsatisfiable UNDER THE CURRENT coarse_score-coupled confidence formula (pipeline.py:2225-2228)
> + global-AP ranking (evaluate.py:342), NOT architecture-eternal.

Do NOT cite this closure as a permanent no-op ban after the confidence formula changes. It is a
property of THIS pipeline's coarse_score-coupled confidence + global-ranked AP objective, not a
generic LLM-cascade result.

---

## Session consistency

- scan-tail dead: v125/v130/v131 (max_scan binds real content, savings capped)
- classify capped ~7.5%: v132 (Lever 2 whole-call drop, below 15% gate)
- whole-video early-exit harness-capped ~5.7%: v127 (dual-AND fired 0; addressable population tiny)
- call-MERGING/BATCHING family REFUTED for accuracy: v97 FCCS, v126 TRLM, v129 BDS all breach floor

No FM12-clean, construction-neutral lever reaches the >= 15% [COST] gate on this 65-video
anomaly-only eval set. Closure triggers NONE of FM1-FM12 (no code change), respects FM11 economy,
and is not a merge/batch/propagation.

---

## FORWARD PATH — registered as a TWO-LOOP PREREQUISITE (explicitly OUT OF SCOPE for this loop)

The one architectural precondition that would make a >= 15% detect-drop constructible in a FUTURE,
separate cycle:

- **Loop N (prerequisite, NOT this loop):** DECOUPLE surviving-detection confidence from
  coarse_score — derive confidence purely from the verdict-INDEPENDENT CLIP boundary/actionness
  contrast (`best_score` / `boundary_contrast`), and use coarse_score ONLY as the existence gate
  (>0.3), NOT as a ranking magnitude. This mirrors BSN (Lin et al., ECCV 2018) / BMN (ICCV 2019)
  temporal-proposal scoring, where proposal confidence comes from a boundary/actionness map separate
  from the trigger. This is itself an **FM1 curve-shape change** (it modifies the v51/v95/v96
  confidence mechanism the session is required to KEEP), so it must be validated as mAP-neutral on
  its own dedicated mAP loop BEFORE any dependent drop-call. It is NOT a pure drop-call.

- **Loop N+1 (the dependent drop-call):** Under that redesign, a region's mAP-ranking score is no
  longer a max over dropped segments' VLM verdicts, so a genuine CLIP-peak-dominance detect-drop
  ("this Source-2 segment lies inside a CLIP peak already covered by a scanned Source-1 segment; if
  suspicious it merges into that peak's region and its coarse_score no longer affects the score, and
  if not suspicious it contributes nothing") becomes verdict-independent on BOTH outcomes and thus
  gate-clean.

Neither half is implemented or evaluated this loop. If C can exhibit a concrete segment that is
byte-identical-safe on scores AND count for BOTH verdicts under the current formula, implement it;
the proof asserts that requires the confidence-decoupling prerequisite first.

---

## Literature (rules OUT, not confirms)

- FrugalGPT (Chen et al. 2023, arXiv 2305.05176) and cascade/early-exit inference skip downstream
  calls once a confidence criterion is met — but those settings score each item INDEPENDENTLY,
  whereas here the skipped call's verdict feeds a per-region MAX that a GLOBAL cross-video AP ranking
  (evaluate.py:342) is sensitive to. That is exactly why the FrugalGPT-style skip is NOT
  verdict-independent here.
- Adaptive-Consistency (Aggarwal et al., EMNLP 2023, arXiv 2305.11860) stops sampling once a majority
  is locked — that lossless-drop logic applies only to the classify ENSEMBLE (already closed ~7.5%,
  v132), not the single-vote detect path.
- BSN (Lin et al., ECCV 2018) / BMN (ICCV 2019): the constructive escape — deriving detection
  confidence from a verdict-INDEPENDENT boundary/actionness contrast rather than the detector's own
  score — is the documented precondition (registered above) under which a CLIP-peak-dominance
  detect-drop would become verdict-independent and gate-clean in a later loop.

(Built-in WebSearch was API-blocked this session; citations are from prior knowledge.)
