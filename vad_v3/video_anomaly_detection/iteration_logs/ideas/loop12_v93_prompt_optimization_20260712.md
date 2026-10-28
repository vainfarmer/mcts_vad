# Loop 12 (v93): CoOp-Style Prompt Optimization + Honest Ceiling Assessment
## Date: 2026-07-12

## Honest Ceiling Assessment (from research, verbatim key point)
"Your mAP of 0.149 is actually not bad for a zero-shot system." T3AL (CVPR
2024, SOTA zero-shot TAL) achieves comparable mAP ranges on standard TAL
benchmarks (THUMOS14: zero-shot ~9-14%, weakly-supervised 46-52%, fully-
supervised 66-72% - a 35-40pt gap from zero-shot to weakly-supervised).
The +34% relative improvement needed (0.149->0.20) is comparable in magnitude
to T3AL's ENTIRE contribution over naive CLIP baselines - i.e., what we need
is roughly "one complete research paper's worth" of improvement, not an
incremental tweak.

Combined best-case estimate across ALL remaining ideas: +0.04 to +0.095
(reaching 0.189-0.244), BUT this assumes independent additive composition,
which our failure history (5 distinct confirmed failure modes, no successful
compounding) strongly suggests will NOT hold in practice.

## Ranked Ideas (by mechanism-based confidence they avoid known failure modes)
1. **CoOp-style prompt optimization** (HIGHEST): learn continuous prompt
   context vectors via MIL loss on training set video-level labels. Operates
   UPSTREAM of fusion/boundary logic - only changes the TEXT EMBEDDING that
   CLIP compares against, everything downstream (get_category_scores,
   boundary detection) unchanged. Expected: +0.02-0.04 mAP.
2. **CLIP model ensembling** (ViT-H/14 LAION + ViT-L/14 OpenAI, geometric
   mean/min combination): also upstream of fusion. Expected +0.01-0.03, but
   doubles CLIP inference cost (latency concern).
3. Category-specific boundary tightening for short-duration categories
   (Arrest/Burglary) - small, safe, +0.005-0.015.
4. Motion-based FALSE-POSITIVE-ONLY suppression (never adds detections, only
   removes CLIP+MIL detections in zero-motion regions) - low confidence,
   could hurt recall.
5. Intersection/consensus candidate filtering - explicitly assessed as
   risky (could hurt recall more than precision, still perilously close to
   failure mode 1 despite theoretical framing).
NOT recommended: VLM consistency for confidence-weighting (5x cost, small
expected gain), any more MIL retraining, any more curve-fusion changes.

## V93 Implementation Decision
Full CoOp requires hooking into CLIP's internal token-embedding layer to
inject learnable continuous vectors - complex, requires careful open_clip
internals work. TRACTABLE SIMPLIFICATION: instead of learnable continuous
embeddings, learn a per-prompt WEIGHT VECTOR for the existing 6 hand-crafted
prompts per category (replacing simple mean-ensembling with a LEARNED
weighted combination), trained via MIL loss (top-k video-level pooling,
exactly analogous to our MIL classifier training) on the 1610-video training
set. This is a much smaller parameter space (6 weights x 13 categories = 78
learnable scalars, vs thousands of embedding dimensions for full CoOp) but
captures the core idea: use training signal to make text embeddings sharper/
better-discriminating, entirely upstream of fusion/boundary logic.
