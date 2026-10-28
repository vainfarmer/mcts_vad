# Loop 10-11: MIL v3/v4 Retraining - FAILED, Reverted to v1
## Date: 2026-07-10
## Version: V76 (final: MIL v1 used everywhere, bug fix only)

## What was tried
1. MIL v3: VadCLIP-style LGT-Adapter (local windowed attention + lightweight GCN)
   + loss ratio corrected to 1:1 sparsity:smoothness (8e-3:8e-3)
2. MIL v4: Same LGT-Adapter architecture, smoothness dialed back to 2e-3
   (4:1 sparsity:smoothness, between RTFM default 10:1 and v3's 1:1)

## Critical discovery: aggregate metrics are misleading
Both v3 and v4 looked EXCELLENT on aggregate stats:
| Model | AUC | Abn mean | Separation | Jitter |
|-------|-----|----------|------------|--------|
| v1    | 0.977 | 0.867  | 0.808      | 0.019  |
| v3    | 0.969 | 0.866  | 0.792      | 0.006  |
| v4    | 0.967 | 0.801  | 0.763      | 0.009  |

But a NEW metric - per-video within-video score RANGE (max-min per video) -
revealed both v3 and v4 had COLLAPSED to near-constant scores for most
abnormal videos (each video outputs a single flat value, either near 0 or
near 1, providing ZERO temporal information):

| Model | Median range | % videos "flat" (range<0.05) |
|-------|--------------|-------------------------------|
| v1    | 0.0671       | 45%                           |
| v3    | 0.0000       | 83%                           |
| v4    | 0.0000       | 75%                           |

Low aggregate "jitter" was NOT because curves were smoothly-varying-and-
discriminative (the goal) - it was because most videos individually
collapsed to a constant (trivially zero jitter, zero discriminative value).
This is a classic failure mode of MIL with strong smoothness regularization:
smoothness loss and BCE loss together can be satisfied by a degenerate
solution (constant-per-video output matching binary video label) without
ever learning WITHIN-video localization.

## Root cause hypothesis
The LGT-Adapter's GCN component (similarity + distance adjacency, softmax-
normalized) aggregates information across ALL 32 snippets per forward pass,
which may over-smooth/homogenize features before the classifier even sees
them - unlike v1's TemporalBlock (dilated convs at multiple scales + local
self-attention) which preserves more snippet-specific signal.

## Decision: V76 keeps MIL v1 for BOTH boundary fusion AND candidate generation
The original V75 bug (MIL v2 device mismatch silently disabling candidate
generation for all 65 videos) is fixed by simply routing V74's TAG-based
candidate generation to use `video_data['mil_scores']` (v1) instead of a
separate v2/v3 model. This is the safest fix: v1 is proven, and while its
45% collapse rate isn't ideal, it's the best of the three options tested.

## Lesson for future loops
When evaluating any new snippet-scoring model for TEMPORAL localization
(not just video-level classification), ALWAYS check per-video score RANGE
(max-min), not just pooled/aggregate statistics across all videos. A model
can have perfect video-level AUC while being completely useless for finding
WHERE in the video the anomaly is.
