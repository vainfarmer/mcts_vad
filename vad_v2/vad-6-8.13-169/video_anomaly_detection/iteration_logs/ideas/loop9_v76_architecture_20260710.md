# Loop 9: VadCLIP LGT-Adapter + Loss Rebalancing Research
## Date: 2026-07-10
## Target Version: V76

## Key Architecture Specs (Agent A: VadCLIP/PE-MIL deep dive)
- LGT-Adapter: Local windowed-attention transformer + lightweight GCN
  - Local: TransformerEncoderLayer, window=8, d_model=512, nhead=8
  - Global GCN: H_sim (cosine similarity adjacency) + H_dis (relative distance
    adjacency, sigma=1), softmax-normalized, single learnable d x d weight
  - Residual connections at each stage
- VadCLIP loss: L_bce + L_nce + 0.1*L_con (dual-branch, requires text prompts -
  too complex for our scope, SKIPPED)
- PE-MIL: Normal Context Prompts explicitly model context around anomalies to
  sharpen boundaries - conceptually validates our need for smooth-but-sharp scores

## Key Loss Tuning Insight (Agent B: GS-MoE + RTFM/MGFN tuning)
- RTFM default: lambda_sparse=8e-3, lambda_smooth=8e-4 (10:1 sparsity:smoothness)
- **Our V73 MIL v2 failure**: lam_sparse=0.1, lam_smooth=0.01 (10:1 ratio, SAME
  direction as RTFM default, just scaled up 12.5x) -> too volatile for boundary fusion
- **Root cause confirmed**: sparsity-dominant loss ratio produces peaky/volatile
  scores that don't fuse well with smooth CLIP curves
- **Fix**: Flip to smoothness-dominant or 1:1 ratio. Recipe: lambda_smooth=8e-3,
  lambda_sparse=8e-3 (1:1, 10x RTFM's smoothness weight)
- GS-MoE Temporal Gaussian Splatting: train-time continuous Gaussian pseudo-labels
  anchored at score peaks -- inherently smooth+peaked, but requires 2-stage training
  (SKIPPED for V76 due to complexity, candidate for V77+ if V76 promising)

## V76 Implementation Plan
1. New model class `LGTAdapterMIL` in mil_model.py:
   - Local windowed self-attention (window=8) via custom attention mask
   - Lightweight GCN (similarity + distance adjacency, single weight matrix)
   - Same RTFM-style classifier head + top-k magnitude selection
2. New loss: lambda_smooth=8e-3, lambda_sparse=8e-3 (1:1, smoothness-corrected)
   margin=100, alpha=0.0001 (RTFM defaults, unaffected by our failure mode)
3. Train on existing features/train (768-dim, no projection needed - adjust
   GCN weight to 768x768)
4. Evaluate: video AUC + score jitter metric (mean |s[t+1]-s[t]| on abnormal
   videos) to explicitly verify smoothness before pipeline integration
5. If smoothness+discrimination both good: integrate as MIL v3 in pipeline
   fusion (replacing v1 in _clip_boundary_detection), full 65-video eval
