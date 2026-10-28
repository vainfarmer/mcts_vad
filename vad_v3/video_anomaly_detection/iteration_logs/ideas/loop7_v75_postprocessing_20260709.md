# Loop 7: Post-Processing & Boundary Refinement
## Date: 2026-07-09
## Version: V75

## Research Sources
- CoReVAD (arXiv 2026): Gaussian smoothing + softmax refinement
- GAP (CVPR 2023): Gaussian-fit sub-snippet boundary refinement
- BTM (MDPI 2022): Confidence-dependent boundary shrinkage
- Cascaded Boundary Regression: Iterative gradient-based refinement

## Implementation Plan for V75
1. Score-gradient boundary refinement:
   - After half-max crossing, check score gradient at boundaries
   - If gradient still rising at start, shift start earlier
   - If gradient still falling at end, shift end later
   - Walk boundary until gradient changes sign (inflection point)

2. Confidence-dependent boundary trimming:
   - Low-confidence detections are more likely over-wide
   - Scale boundary width by confidence: tighter for uncertain detections

3. Gaussian smoothing on score curve before boundary detection:
   - Replace box filter with Gaussian kernel for smoother curves
   - Preserves peaks better while reducing noise

## Expected Impact
- Target: improve IoU for the 39% of detections at IoU 0.1-0.5
- Score-gradient refinement could tighten boundaries by 10-20%
- Combined: potentially +0.03-0.08 mAP
