# Loop 5 Research: MIL Retraining with Enhanced Regularization
## Date: 2026-07-09
## Version: V73

## Problem
Current MIL model (model_mil.pth) produces near-saturated scores:
- mean=0.994, max=1.000, min=0.978
- Every snippet gets high score -> useless for temporal localization
- Root cause: sigmoid saturates, weak sparsity/magnitude losses

## Research Sources
- RTFM (ICCV 2021): Feature magnitude learning, margin=100
- MGFN (AAAI 2023): Magnitude-Contrastive with dual-branch
- PE-MIL (CVPR 2024): Prompt-enhanced MIL for boundary clarity

## Implementation: Retrain with stronger regularization
1. Increase sparsity loss: lam_sparse 0.01 -> 0.1 (force most snippets to be LOW)
2. Increase smoothness loss: lam_smooth 0.001 -> 0.01 (enforce temporal consistency)
3. Add temperature scaling before sigmoid (sharper output distribution)
4. Reduce classifier capacity (less dropout to prevent memorization)
5. Add cosine annealing LR schedule
6. Train 100 epochs instead of 50

## Expected Impact
- MIL scores should have much wider range (not all 0.98-1.0)
- Better temporal discrimination -> better boundary detection input
- Combined with V72 re-centering -> potentially significant improvement
