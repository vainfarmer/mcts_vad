# Loop 1 (v86): Search Storm - Low-Risk Isolated mAP Improvements
## Date: 2026-07-12

## Baseline Confirmed
Pipeline reverted to clean V51 (see archived_failed_v80-85b/SUMMARY.md for the
5 failed MIL-based attempts that preceded this). Confirmation smoke test (14
videos): mAP=0.1183, consistent with historical V51 noise band (0.12-0.15).

## Agent A Findings (post-processing, curve-shape-isolated)
Ranked by safety+impact:
1. **VLM self-consistency (3x median boundary estimation)**: for candidates
   ALREADY selected by CLIP, call VLM 3x for boundary estimation, take median
   of start/end estimates. Fully isolated (doesn't touch CLIP curve/candidate
   pool). +0.01-0.02 mAP expected. Cost: 3x VLM calls but only for the
   LOW_CLIP_CATEGORIES + low-confidence subset already using VLM boundary
   estimation (small fraction of videos) - manageable latency impact.
2. **Soft-NMS** (Bodla et al. ICCV 2017): decay overlapping detection scores
   by Gaussian tIoU decay instead of hard removal. Trivial (30min-1hr),
   negligible latency, +0.005-0.015 mAP.
3. **GAP Gaussian sub-snippet boundary refinement** (CVPR 2023): fit Gaussian
   to score curve at boundaries, Taylor-expand for sub-frame precision.
   Fully isolated post-processing. +0.005-0.015 mAP, 2-3hrs.
4. CLIP TTA (MTA-style multi-view averaging) - PARTIALLY isolated (smooths
   but doesn't fuse heterogeneous signal), deferred due to risk + latency cost.
5. Category disambiguation via ASK-HINT structured prompting - see Agent B.

## Agent B Findings (category disambiguation, video-level not temporal)
1. **Attribute-decomposed binary verification**: replace single multi-class
   VLM question with 7 structured yes/no questions (weapon? retail store?
   force used? etc.) mapped to categories via deterministic decision tree in
   Python. VLMs are more accurate at binary visual QA than multi-class
   selection among semantically-overlapping categories (Shoplifting/Stealing/
   Robbery/Burglary cluster). +5-10% category accuracy, ZERO extra API cost
   (same call count, restructured prompt + code-side decision tree).
2. **Diverse-prompt self-consistency**: replace 3 IDENTICAL VLM tie-break
   calls with 3 DIFFERENT-PERSPECTIVE prompts (scene-focused/action-focused/
   consequence-focused). Zero extra cost (same call count). +3-5% expected.
3. Segment-only classification (send only anomaly segment, not full video) -
   may also REDUCE latency. +2-5%.
4-6: smarter routing, post-classification verification, consistency checking
   - smaller incremental gains, deferred.

## V86 Implementation Decision
Stack the SAFEST, LOWEST-COST wins first (all isolated from boundary curve
shape, all zero-or-near-zero extra API cost):
1. Soft-NMS (replace hard NMS in post_processor.py)
2. Diverse-prompt self-consistency for category tie-breaking (zero extra
   cost - same 3 calls, different prompts)
3. Attribute-decomposed binary verification IF time permits (bigger change,
   defer to v87 if v86's simpler wins need isolated evaluation first)

Explicitly deferred to keep latency goal: VLM boundary self-consistency
(adds 3x calls even if small subset), CLIP TTA (adds 4-8x CLIP calls).

## POST-HOC UPDATE (Loop 8, 2026-07-12): V86 full eval REGRESSED
Full 65-video eval: mAP=0.0961 (vs V51 baseline 0.1492). Root cause confirmed
by reading evaluate.py: standard GREEDY matching in temporal mAP computation
(predictions sorted by confidence, each GT matchable once). Soft-NMS's
permissive survival threshold let multiple near-duplicate detections survive
per true event; only the highest-confidence one becomes a TP, the rest become
FALSE POSITIVES under greedy matching. This directly contradicts Agent A's
"fully isolated, safe" assessment - Soft-NMS IS isolated from the CLIP curve
shape (as claimed), but it is NOT isolated from the mAP EVALUATION PROTOCOL,
which the research didn't examine closely enough. Lesson: "isolated from
detection algorithm X" doesn't mean "isolated from evaluation metric Y" -
must check both.

DECISION: Reverted to Hard-NMS (temporal_nms). Kept diverse-perspective
self-consistency for category tie-breaking (V86 part 2) since it's
orthogonal to segment count/NMS - doesn't add/remove detections, only
changes which category LABEL is assigned. This becomes V89 (see loop8
research file for full isolated evaluation).
