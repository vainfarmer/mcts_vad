# Loop 14: Latency Optimization Notes
## Date: 2026-07-10

## Confirmed bottleneck (from Loop 8 profiling)
CLIP ViT-L/14 encode on MPS: ~13.4 frames/sec, compute-bound (not preprocessing-
bound - PIL resize is negligible at 0.18s/200 frames). For a 943-frame video,
CLIP precompute alone costs ~70-150s of the total ~125-250s per-video time.

## Per-video time breakdown (from V75 65-video eval, n=64)
mean=124.5s, median=80.0s, max=504.6s, min=11.4s (highly variable by video length)

## Optimization candidates (not yet implemented this session - time budget priority
## went to mAP-improving changes in Loops 9-13; latency work deferred)
1. **Smaller CLIP backbone for coarse localization pass**: switch multi-scale
   temporal localization (_clip_temporal_localization) to ViT-B/32 (4-8x faster
   than ViT-L/14), keep ViT-L/14 only for final candidate verification/VLM
   context. RISK: may reduce localization precision since ViT-B/32 has weaker
   zero-shot discrimination. Needs A/B test before adopting.
2. **Frame sampling rate reduction**: current motion-filter removes ~0-40%
   frames per video already; could add a secondary uniform downsample (e.g.
   max 1 fps for CLIP precompute stage specifically) since 13 fps encode speed
   × long videos (500+ frames after motion filter) is the main driver of
   the 500s tail latency.
3. **Skip MIL v1 recompute per video** - already loaded once in __init__,
   not a bottleneck (single forward pass on 32 snippets is <10ms).
4. **Reduce VLM call count**: _directed_llm_scan already limits to
   top-K segments (candidate_top_k config) - already optimized in earlier
   loops (V56-V58 candidate generation work).

## Decision for this session
Given time budget constraints and that mAP=0.50 target is far from reached,
latency optimization is DEFERRED to a future session. The single highest-
leverage latency fix (frame sampling reduction for long videos) is flagged
for Loop 17's final recommendations but not implemented/tested here to avoid
risking pipeline stability changes late in this session without eval budget
to verify them.
