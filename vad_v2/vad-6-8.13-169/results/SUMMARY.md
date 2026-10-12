# Archived: V80-V85b Failed Experiment Chain
## Date archived: 2026-07-12

## What happened
A sequence of experiments (V80 through V85b) all attempted to leverage a newly
retrained MIL model (v6, with anti-collapse range/variance loss - see
../ideas/loop1_v80_mil_v6_research_20260712.md for the successful training
research) to improve pipeline mAP. Despite MIL v6 being DRAMATICALLY better
by component-level metrics (collapse rate 45%->5%, median within-video score
range 0.067->0.495, AUC comparable at 0.973 vs 0.977), EVERY attempt to use
this improved model in the pipeline regressed end-to-end mAP:

| Version | mAP | vs V51 baseline (0.1492) | Change tested |
|---------|-----|--------------------------|----------------|
| V80 | 0.1106 | -26% | Dynamic MIL fusion weight (up to 0.5, from 0.25) |
| V83 | 0.0573 | -62% | Reverted dynamic weight to 0.25 + MIL-TAG candidates + Q-score reranking |
| V84 | 0.0818 | -45% | (variant, see batch_eval_v84.log for details) |
| V85 | 0.0926 | -38% | (variant, see batch_eval_v85.log for details) |
| V85b | 0.0432 | -71% | (variant, see batch_eval_v85b.log for details) |

## Root cause (confirmed via V80 analysis, see ../ideas/loop4_v80_regression_analysis_20260712.md)
The half-max boundary detection algorithm is sensitive to the SHAPE of the
score curve, not just each signal's individual discriminative quality. MIL
v6's peaks, even though more "real" (not collapsed to a constant), do not
necessarily align with CLIP's peaks or with the true ground-truth anomaly
location. Any change that increases MIL's influence on the fused curve shape
(via fusion weight) or adds MIL-derived candidates to the pool dilutes/shifts
away from the CLIP-based signal that V51's boundary detection was tuned
against, with no guarantee the shift is toward the true location.

This is the SAME lesson learned earlier in the project with MIL v2/v3/v4
(documented in loop10-11_v76_mil_retrain_failure_20260710.md) - now confirmed
a THIRD time with v6, despite v6 fixing the specific collapse problem that
v2/v3/v4 had. Component-level model quality (AUC, collapse rate) does NOT
transfer to pipeline-level mAP when the component feeds a shape-sensitive
downstream algorithm.

## Resolution
Pipeline.py fully reverted to the clean, proven V51 baseline (MIL v1 model,
fixed 0.25 fusion weight, no MIL-TAG candidates, no Q-score reranking).
New experimentation resumes at V86 with this clean baseline as the starting
point, per user instruction, targeting a more realistic mAP>=20% goal.

## What's still available for future reference
- model_mil_v6.pth (in video_anomaly_detection/) - the anti-collapse-trained
  MIL model itself remains on disk. Its FAILURE was in how it was USED in
  the pipeline (fusion weight, candidate generation), not necessarily a
  flaw in the model itself. Future ideas that use v6 differently (e.g. as
  a purely independent verification signal rather than modifying the
  boundary-detection curve shape) may still be worth exploring.
- train_mil_v6.py - the training script with anti-collapse range/variance
  loss + per-video collapse monitoring, in case useful for future MIL work.
