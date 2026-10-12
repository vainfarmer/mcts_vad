# Loop 8: Data Feasibility + Latency Profiling
## Date: 2026-07-10

## UCA Dataset Investigation
- UCA (UCF-Crime Annotation, CVPR 2024) provides dense sentence-level + timestamp
  annotations for 1854 UCF-Crime videos (train+test) - would enable REAL supervised
  temporal training data (vs our current weak video-level labels only)
- Available via Kaggle (vigneshwar472/ucaucf-crime-annotation-dataset) - requires
  Kaggle API auth, not configured in this environment
- DECISION: Skip for now. VadCLIP's approach doesn't require dense temporal labels
  anyway (see below) - lower integration cost, comparable expected benefit.

## Key Insight: VadCLIP doesn't need new data
- VadCLIP (AAAI 2024) uses the SAME weakly-supervised setup as our current MIL:
  video-level labels for training, frame-level GT only for test evaluation.
- Its improvement over vanilla MIL comes from ARCHITECTURE: LGT-Adapter
  (local temporal conv + global graph conv + lightweight transformer) +
  dual-branch MIL-Align loss (coarse classification + fine-grained
  language-vision alignment), not from better labels.
- This means we can implement VadCLIP-style architecture directly on our
  existing 1610 train / 290 test CLIP feature files (features/train, features/test)
  with NO new data acquisition needed.

## Latency Profiling
- CLIP precompute is the dominant per-video cost: ViT-L/14 on MPS runs at
  ~13.4 frames/sec (measured: 200 frames = 14.88s pure encode, PIL preprocessing
  is negligible at 0.18s/200 frames - NOT the bottleneck).
- For long videos (943 frames) this alone costs ~70s+ of the 150s total.
- Batch size (32 vs 64) doesn't materially help - compute-bound on MPS, not
  transfer-bound.
- OPTIMIZATION CANDIDATE (Loop 14): use faster backbone (ViT-B/32, ~4-8x faster)
  for the coarse multi-scale localization pass, reserve ViT-L/14 only for
  final classification/VLM-selected segments. Risk: may reduce localization
  precision. To be tested in Loop 14 with careful A/B comparison.

## MIL v2 Device Bug (needs fix before Loop 10)
- V75 eval log showed repeated: "[MIL-v2] Scoring failed: Input type
  (MPSFloatType) and weight type (torch.FloatTensor) should be the same"
- Root cause: model_mil_v2.pth loaded with map_location='cpu' but
  _snippet_tensor is already on self.device (mps). Need .to(self.device)
  on the v2 model before inference.
- This bug silently disabled V74's MIL-guided candidate generation for the
  entire V75 eval run - MIL v2 candidates were NEVER actually used.
