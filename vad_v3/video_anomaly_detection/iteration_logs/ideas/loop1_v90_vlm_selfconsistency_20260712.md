# Loop 1 (v90): Search Storm - VLM Boundary Self-Consistency + GAP
## Date: 2026-07-12 (new 15-loop session, continuing from v89)

## Research Findings
### Idea 1: VLM Self-Consistency for Boundary Refinement - IMPLEMENTED
- GAP algorithm (Nag et al. CVPR 2023): obtained exact formula (Gaussian
  smoothing + log-space Taylor/Newton expansion for sub-snippet peak
  refinement). Assessed as POOR FIT: GAP solves snippet-quantization error,
  our boundaries are continuous-time and VLM errors are 2-10s scale -
  different problem entirely. NOT IMPLEMENTED.
- VLM self-consistency: N=3 recommended (RASC NAACL 2025), median aggregation
  (for N=3, median=trimmed-mean, unambiguous choice). CRITICAL RISK found:
  temperature=0 does NOT guarantee determinism (infra-level FP non-
  associativity) BUT diversity may be too LOW to be useful at temp=0 alone.
  Recommendation: temperature ladder (0.0, 0.3, 0.5) across the 3 calls for
  genuine diversity while keeping one deterministic anchor call.
- Confidence-weighted averaging (WBF-style) considered but deferred: with
  only N=3, marginal benefit over simple median, and VLM confidence scores
  aren't reliably calibrated.

## V90 Implementation
1. Added `temperature_override` optional param to `LLMClient.generate()`
   (backward compatible - all existing call sites unaffected, default None
   uses client's configured self.temperature as before)
2. `_vlm_boundary_estimation()`: replaced single VLM call with 3 concurrent
   calls (asyncio.gather) at temperatures [0.0, 0.3, 0.5], median aggregation
   of start_time/end_time/confidence across valid (anomaly_detected=true)
   responses.

## Respects both hard-won lessons from prior session
1. Does not touch CLIP/MIL fused curve shape - operates only on an already-
   selected candidate's boundary position.
2. Does not change detection COUNT - exactly 1 candidate in, 1 refined
   candidate out, so cannot trigger the greedy-matching false-positive
   problem that killed Soft-NMS (V86).

## Scope of effect
Only affects candidates in LOW_CLIP_CATEGORIES with clip_conf < 0.15 (the
V51-original VLM boundary estimation guard condition) - a SMALL SUBSET of
videos, not a pipeline-wide change. Cost: 3x VLM calls but only for this
subset (parallelized via asyncio.gather, so wall-clock cost is ~1x, not 3x).

## POST-HOC UPDATE: V90 full eval REGRESSED
Full 65-video eval: mAP=0.1075 (vs V89's 0.1494). This is a THIRD distinct
failure mode beyond the two known lessons (curve-shape sensitivity,
greedy-match false-positive risk) - self-consistency via temperature-
diversity doesn't transfer from reasoning tasks to perceptual/extraction
tasks (see v91 docstring in pipeline.py for full analysis). REVERTED.

## Running list of "isolation" categories that can STILL fail:
1. Curve-shape modification (affects half-max boundary detection) - FAILS
2. Detection-count changes under greedy-match eval (Soft-NMS) - FAILS
3. Self-consistency/sampling-diversity for perceptual tasks (this) - FAILS
Remaining untested category: post-hoc ensembling of COMPLETE detection sets
from independent full pipeline runs (Loop 7+, next).
