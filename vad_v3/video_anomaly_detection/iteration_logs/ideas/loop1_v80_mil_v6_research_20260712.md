# Loop 1: Search Storm - MIL Collapse Prevention + Temporal Grounding Literature
## Date: 2026-07-12
## Target Version: V80

## Agent A Findings (MIL collapse-specific)
### Top picks for implementation:
1. **K-maxmin Inner Bag Range Loss** (TE-VTAF 2026, ICASSP 2019 IBL):
   ```
   s_topk = mean(topk(scores_abn, k=3))
   s_botk = mean(bottomk(scores_abn, k=3))
   L_range_abn = max(0, margin - (s_topk - s_botk))  # margin=0.3-0.5
   # For normal videos: minimize the gap instead (should be uniformly low)
   L_range_norm = mean(topk(scores_norm,3)) - mean(bottomk(scores_norm,3))
   ```
   Directly penalizes score range=0 (our exact collapse symptom). ~2hr implementation.

2. **VICReg-style variance hinge** (Bardes/Ponce/LeCun 2021, novel cross-pollination):
   ```
   L_variance = max(0, gamma - sqrt(scores_abn.var() + eps))  # gamma=0.1-0.15
   ```
   Simplest possible anti-collapse term, 3 lines, zero-risk to add alongside range loss.

3. AEM entropy maximization w/ cosine annealing (MICCAI 2025) - more complex, deferred
4. KMIL feature-magnitude margin loss - complements range loss at feature level, deferred to v2 if v1 insufficient
5. Two-stage pseudo-label self-training (MIST-style) - "nuclear option", deferred

### Key insight: no one else has formalized "collapse rate" as a metric -
our project's diagnostic (range<0.05 = collapsed) is a genuinely novel finding.

## Agent B Findings (broader temporal grounding + latency)
### Top picks:
1. **TFVTG dual dynamic+static scoring** (ECCV 2024): onset-detection via positive
   first-difference of smoothed score curve + contrast (inside vs outside).
   Reduces start-time bias by 56% in their benchmark. Pure signal processing,
   no training needed, ~50 lines.
2. **Box-Cox similarity distribution adjustment** (TAG, arXiv 2025): de-skew
   CLIP similarity distribution per category before thresholding - addresses
   the exact "cv<0.03, no discriminability" problem for low-CLIP categories.
3. **EVS-style VLM frame pruning** (2025, used in NVIDIA Nemotron): prune
   temporally-static frames before sending to VLM - up to 4x latency reduction,
   <1% accuracy loss. DIRECTLY addresses "minimize per-video latency" goal.
4. VadCLIP++ frame-difference features (feature-space L2, distinct from our
   already-tried score-space temporal_diff) - deferred, moderate complexity
5-10: FlashVTG pyramid, T* coarse-to-fine, layer-weighted CLIP, prompt
   decomposition, cross-category recurrence - all moderate+ complexity, deferred

### SOTA context confirmed: GCTG-CLIP (2025) reports avg mAP@0.1-0.5=12.30%,
best published WTAL result on UCF-Crime. Our 14% at IoU 0.3/0.5/0.7 (harder
thresholds) is genuinely competitive with SOTA trained methods.

## V80 Implementation Decision
Combine Agent A's Idea 1 (K-maxmin range loss) + Idea 5 (VICReg variance hinge)
on top of the PROVEN MILAnomalyScorer architecture (v1's TemporalBlock -
dilated convs + self-attention, explicitly NOT LGTAdapter/GCN which caused
75-83% collapse in v3/v4). Per-video range monitoring during training with
early-stop guard (per spec's Phase 1 Step 1.2 requirement).

Loss formula:
L = BCE + alpha*L_magnitude_margin + lam_smooth*L_smooth + lam_sparse*L_sparse
    + lam_range*L_range_hinge + lam_var*L_variance_hinge
