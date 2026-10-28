# Loop 15: Candidate Recall Research - Negative Prompt Contrastive Scoring
## Date: 2026-07-10
## Target Version: V78

## Research Findings (single combined agent)
### Tier 1 (implementing now - zero cost, zero latency):
**Negative Prompt Contrastive Scoring + Multi-Prompt Ensembling**
- AnomalyCLIP (ICLR 2024), AA-CLIP (CVPR 2025): CLIP has "anomaly-unawareness" -
  shows higher similarity to "normal" prompts even for genuinely abnormal images
- Fix: define confusion negative prompts per category, compute contrastive score:
  score = mean(sim(frame, pos_prompts)) - alpha * max(sim(frame, neg_prompts))
- Expected: 15-30% reduction in wrong-location candidates
- Cost: zero (text embeddings precomputed once)

### Tier 2 (deferred - higher cost/complexity):
**Hybrid CLIP peaks + Uniform VLM Coarse Triage**
- Add parallel uniform-chunk VLM binary triage as candidate source
- Expected: 30-50% reduction but +$0.03-0.06/video, +20-40s latency
- DEFERRED: conflicts with "minimize per-video time" goal from user request

### Tier 3 (deferred - large scope):
- Flashback memory bank (requires offline generation, $100-200 one-time)
- T*-style iterative zoom-in

## V78 Implementation Plan
1. Define 2-3 confusion negative prompts per category in pipeline.py
2. Precompute their CLIP text embeddings once (alongside existing category
   text_features in AnomalyExpert)
3. Modify get_category_scores (frame_scorer.py) to subtract negative
   similarity: relative_scores = cat_scores - alpha * neg_scores (before
   the softmax-ratio normalization, or as an additive adjustment after)
4. This affects BOTH candidate generation (_clip_temporal_localization) AND
   boundary detection (_clip_boundary_detection) since both call
   get_category_scores - need careful validation this doesn't destabilize
   the proven half-max crossing boundary detection.
