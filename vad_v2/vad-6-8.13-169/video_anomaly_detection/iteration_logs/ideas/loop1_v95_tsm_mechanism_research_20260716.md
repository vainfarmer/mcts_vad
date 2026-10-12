# Loop 1 (v95): How TAL Papers Construct Localization Signals - Deep Mechanism Study
## Date: 2026-07-16 (resumed after MCP connection interruption; research done 07-12)

## Key Finding: 4 Genuinely Different Signal Construction Paradigms
| Paradigm | Signal Type | Differs from our "1 curve + threshold" by |
|----------|-------------|---------------------------------------------|
| BSN/BMN | 3 curves (start/end/actionness) | Multiple complementary, directionally asymmetric curves |
| TAG/Watershed | 1 curve, multi-threshold sweep | Never commits to one threshold, joint multi-scale eval |
| ActionFormer/TriDet | 2-channel distance field | Directly regresses boundary distance, no thresholding |
| TSM+Checkerboard | 2D N x N frame-frame matrix | Genuinely different signal TYPE (not text-similarity at all) |

## Detailed mechanism notes (from Agent A + Agent B search storm)

### BSN/BMN (ECCV 2018 / ICCV 2019)
Three independent per-frame curves via 1D temporal conv (TEM): P_start(t),
P_end(t), P_actionness(t). Soft IoR labels over a +/-10% duration zone around
GT boundaries. Proposals = combinatorial (start_i, end_j) pairs where both
exceed threshold and end_j > start_i, scored via Boundary-Sensitive Proposal
(BSP) features sampled at start/center/end zones. BMN replaces this with a
dense 2D Boundary-Matching confidence map (duration x time).
Training-free analog: start_likelihood(t) = anomaly_score(t:t+w) -
anomaly_score(t-w:t) (asymmetric forward-looking contrast); end_likelihood(t)
= reverse. Directionally asymmetric, unlike our symmetric inner-outer
contrast.

### TAG / Watershed (SSN lineage, ICCV 2017)
Sweep threshold tau over ~11 levels x grouping ratio eta over ~40 levels
(~440 combos) on ONE actionness curve, extract contiguous regions above tau,
merge via eta, score by avg actionness, NMS at 0.95. Never commits to one
threshold - union across all "water levels." Training-free analog: sweep our
existing curve at multiple thresholds (0.1x-0.9x peak), generate one segment
per threshold, score by mean score, NMS to select - does NOT modify the
curve itself.

### ActionFormer / TriDet (ECCV 2022 / CVPR 2023)
Per-location (not per-boundary) regression of (distance_to_start,
distance_to_end) directly, decoded as [t-d_s, t+d_e]. Multi-scale pyramid
(6 levels, each handling different duration ranges). TriDet extends to a
probability DISTRIBUTION over boundary offsets (handles ambiguity), boundary
= expectation under this distribution. Fundamentally a 2-channel distance
FIELD, not a thresholded presence curve.

### OIC/LogOIC (AutoLoc ECCV 2018 -> FreeZAD 2025)
OIC(segment) = mean(inner_activation) - mean(outer_activation). LogOIC adds
log-decay weighting to outer contributions (1/log(1+d)), making score more
sensitive to immediate boundary contrast. FreeZAD's full pipeline: pseudo-
label -> similarity curve -> multi-threshold candidate generation -> LogOIC
scoring -> FFT-based "actionness calibration" (frequency-domain energy
multiplied into confidence) -> NMS. NOTE: LogOIC was ALREADY TRIED in a
prior session (V65-68) as a boundary-detection REPLACEMENT and it
over-narrowed long events while helping short ones - net negative. This
session's TAG-multi-threshold-generation-then-OIC-scoring pattern is subtly
different (generates MULTIPLE candidates from unmodified curve, scores
separately) vs the prior over-eager application (used LogOIC to directly
TIGHTEN detected boundaries) - worth distinguishing but carries similar risk.

### T3AL (CVPR 2024)
Frame-text similarity -> BYOL self-supervised test-time weight adaptation
(actually updates VLM weights per-video) -> 2D pairwise CAPTION similarity
matrix for proposal suppression (proposals textually dissimilar to the
majority get suppressed). The BASE signal is still 1D; novelty is in
refinement via a genuinely 2D structure (inter-proposal text-similarity, not
inter-frame visual similarity).

### VideoGEM / GEM (CVPR 2024/2025)
Self-self attention (SSA(X) = softmax(XX^T/tau)*X on Q/Q, K/K, V/V
projections) applied at INTERMEDIATE ViT layers (depth 3-7 best), producing
per-frame SPATIAL heatmaps (WHERE in frame), not temporal. Multi-layer
combination with exponentially-increasing-by-depth weights. Adaptable
training-free idea: combine similarity from MULTIPLE CLIP layers (not just
final layer) with fixed heuristic weights - different FEATURE extraction,
not a different post-hoc fusion. Caveat: still produces a 1D curve after
aggregation, so still carries curve-shape risk if fused into the existing
pipeline naively.

### Temporal Self-Similarity Matrix + Checkerboard Kernel (Foote's novelty
function; UBoCo CVPR 2022 lineage) - SELECTED FOR IMPLEMENTATION
SSM[i,j] = cosine_sim(CLIP_feat[i], CLIP_feat[j]) for ALL frame pairs (not
frame-to-text). Temporally coherent segments = high-similarity diagonal
blocks; boundaries = block-transitions. Checkerboard kernel K (+1 same-block
quadrants, -1 cross-block quadrants, Gaussian-tapered) correlated along the
SSM diagonal: N(t) = sum(K[i,j] * SSM[t-L+i, t-L+j]). Peaks in N(t) = visual
scene transitions, COMPLETELY INDEPENDENT of text semantics/category.

This is the ONLY approach among all researched that constructs a truly
different SIGNAL TYPE (2D matrix -> derived 1D novelty function) rather than
a different way to compute/threshold/score a 1D text-similarity curve.
Zero additional CLIP calls needed (reuses existing video_data['clip_features']).

## Risk Assessment Given 4 Known Failure Modes (from prior rounds)
1. Curve-shape modification (fusion weight/candidate pool changes) - regresses
   even when the added signal is individually validated as better.
2. Detection-count changes under greedy single-match mAP evaluation
   (Soft-NMS) - false positives from surviving near-duplicates.
3. Self-consistency for numerical/perceptual extraction tasks (VLM boundary
   timestamp reading) - adds noise, no "alternative valid path" exists.
4. Self-consistency changing candidate/segment INCLUSION decisions
   (borderline revoting) - same risk category as #1 via a different route.

TSM+checkerboard avoids all 4 IF applied as:
- Pure CONFIDENCE re-ranking (not boundary position modification) - same
  safe pattern as V89's diverse-perspective category tie-break (confirmed
  non-regressing) and V94's learned prompt weights (confirmed non-regressing,
  best-ever AP@0.7).
- NOT a VLM sampling/self-consistency technique at all - orthogonal risk
  profile entirely.
- Does NOT add/remove candidates or change which segments are scanned.

## V95 Implementation Plan
1. Compute N x N CLIP self-similarity matrix per video (reuse existing
   video_data['clip_features'], already extracted - zero extra CLIP/VLM calls)
2. Apply Gaussian-tapered checkerboard kernel correlation along the diagonal
   to get a novelty(t) curve (try kernel half-width L in {15, 30, 60} frames,
   pick per-category or take max across scales)
3. For each already-detected candidate's boundaries [t_start, t_end], compute
   a "TSM sharpness" score = novelty(t_start) + novelty(t_end), normalized
   to [0,1] relative to the video's own novelty curve range
4. Modulate confidence (NOT boundary position) by this TSM sharpness score,
   combined with existing boundary_contrast the same way V82's Q-score did
   (Q-score itself was bundled into a broader revert without fully isolated
   confirmation in the prior session - this gives it a clean, isolated retest
   using a genuinely different, more principled sharpness signal)
