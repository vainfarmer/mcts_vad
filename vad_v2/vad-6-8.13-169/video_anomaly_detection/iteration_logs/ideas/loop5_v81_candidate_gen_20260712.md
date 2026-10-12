# Loop 5: MIL-Guided Candidate Generation for Low-CLIP Categories
## Date: 2026-07-12
## Target Version: V81

## Key Validation
Literature confirms our diagnosis: CLIP text embeddings for UCF-Crime property-
crime categories (Shoplifting/Stealing/Burglary/Robbery) form a nearly
indistinguishable semantic cluster - CLIP similarity to "shoplifting" prompt is
nearly identical between actual shoplifting and normal shopping frames. Best
VLM multiclass accuracy on UCF-Crime is only 65.31%. This is a fundamental
CLIP limitation, not a prompt engineering problem - validates using MIL
(learned from visual patterns, not text-matching) as primary signal for these
categories now that MIL v6 has real within-video discrimination.

## Ranked Ideas (by feasibility within few-hour budget)
1. **MIL-TAG multi-threshold grouping + Soft-NMS** (CTAP/TAG literature):
   sweep thresholds [0.1..0.9], group consecutive foreground snippets,
   confidence=mean score, Soft-NMS across all threshold outputs. ~2hrs.
2. **CV-based adaptive signal weighting** (DSANet-inspired): per-video,
   not per-category, continuous routing:
   ```
   cv = std(clip_scores) / (mean(clip_scores) + eps)
   mil_weight = 0.5 + 0.4 * sigmoid((0.05 - cv) * 100)  # ranges 0.1-0.9
   ```
   ~1.5hrs.
3. **Half-max boundary detection on MIL curve** (same algorithm we already
   use for CLIP, just applied to MIL peaks - zero new architecture). ~1hr.
4. AFF-style softmax modality weighting - simpler alternative to #2, deferred
5. **Gaussian smoothing of MIL scores before proposal generation** (sigma=1.5
   snippets, bridges 1-2 snippet gaps). ~30min, do first as a prerequisite.
6. Optical flow motion-reversal for covert actions - literature shows even
   specialized pose-based detectors only reach 69-77% AUC, MIL likely more
   impactful. DEFERRED (also adds latency, conflicts with speed goal).
7-8: PE-MIL boundary sharpening, ActionFormer single-shot - deferred (too
   complex for budget or marginal gain over half-max).

## V81 Implementation Plan
1. Gaussian-smooth MIL scores (sigma=1.5) as preprocessing
2. Implement MIL-TAG proposal generation (multi-threshold + Soft-NMS)
3. Route: use MIL-TAG as PRIMARY candidate source for Shoplifting/Stealing/
   Arson/Burglary; supplement (not replace) CLIP-peak candidates for other
   categories via union+dedup (same pattern as old V74, now viable since
   MIL v6 doesn't collapse)
4. CV-based adaptive weighting deferred to V82+ if V81 shows promise
