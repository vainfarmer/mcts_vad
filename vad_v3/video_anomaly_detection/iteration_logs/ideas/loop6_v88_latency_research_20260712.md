# Loop 6 (v88): Latency Optimization Research
## Date: 2026-07-12

## Key Finding: Most "safe" wins are ALREADY implemented
Research identified 9 ranked latency ideas (CoreML conversion, MLX, concurrent
VLM calls, connection pooling, text-embedding caching, pipeline parallelism,
multi-image VLM batching, coarse-then-fine two-stage CLIP, pre-decode overlap).

Verification against current codebase showed MOST zero-risk items are already
in place:
- **5A (persistent connection/pooling)**: DashScopeLLMClient wraps AsyncOpenAI,
  instantiated ONCE per batch run (batch_evaluate.py L91/96), httpx connection
  pooling is default behavior. ALREADY DONE.
- **5B (concurrent VLM calls)**: `_directed_llm_scan` uses `asyncio.gather(*tasks)`
  for parallel segment scanning (line ~1028); `_ensemble_classify`'s 3-way vote
  also uses `asyncio.gather` (line ~1859). ALREADY DONE.
- **2A (cached text embeddings)**: `anomaly_expert.py`'s `_precompute_text_features()`
  called once in `__init__`, cached as `self.text_features`. ALREADY DONE.
- **4A (pipeline parallelism across videos)**: partially achieved via
  batch_evaluate.py's `concurrency=3` semaphore-based video-level parallelism.

## Remaining opportunity: CoreML conversion (highest impact, highest risk)
Idea 1A (CoreML conversion of CLIP vision encoder) offers 3-4x speedup on the
dominant bottleneck (CLIP encoding, ~13 fps -> potentially ~40-55 fps) with
claimed zero accuracy risk (numerically equivalent FP16 embeddings). However:
- Touches the MOST CRITICAL, MOST EXTENSIVELY-VALIDATED code path (CLIP
  encoding feeds BOTH candidate generation AND boundary detection)
- This project has repeatedly learned that "safe-seeming" CLIP-path changes
  (V72 feature re-centering, V78 confusion-negative contrastive scoring) had
  unexpected regressions despite sound theoretical justification
- Requires 4-8 hours of careful implementation + validation (fixed-shape
  batching, coremltools conversion, numerical equivalence testing)
- GPU contention observed empirically: running a benchmark script
  concurrently with the live batch eval slowed BOTH down (1049-frame video
  took 115s/9.1fps instead of the expected ~78s/13.4fps) - confirms MPS is a
  single shared GPU resource, any change to its usage pattern needs care.

## Decision: DEFER CoreML conversion
Given (a) the user's PRIMARY goal is mAP>=20% (latency is secondary
"minimize"), (b) the repeated lesson that CLIP-path changes carry more risk
than they appear, and (c) remaining time budget in this 10-loop session,
CoreML conversion is deferred rather than attempted blind. The already-
implemented optimizations (connection pooling, concurrent VLM calls, cached
embeddings, batch-level parallelism) represent the safe, already-captured
latency wins. Documented here for a FUTURE session with dedicated time to
implement + rigorously validate CoreML conversion's numerical equivalence
before trusting it in the full pipeline.

## What WAS validated as safe and already helps
Soft-NMS (V86) doesn't add computation (same segment count, just different
suppression math) - no latency cost, already deployed.
