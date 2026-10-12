# Loop 4 (v91): BMN/BSN Proposal Networks - REJECTED (research-only, no implementation)
## Date: 2026-07-12

## Research Summary
Searched for weakly-supervised temporal proposal generation networks (BSN/BMN
descendants) as a genuinely different architecture family from MIL classifiers
(which we've tried 5x: v1/v2/v3/v4/v6).

### Key Findings
1. **T-RPN (ACML 2024) and PseudoFormer (CVPR 2025)**: both bootstrap boundary
   regression from MIL-derived pseudo-labels (CAS/attention-map peaks converted
   to pseudo start/end proposals). Neither trains from raw video-level labels
   directly - the regression head's quality ceiling is BOUNDED by the initial
   MIL scorer's quality. Neither evaluated on UCF-Crime (THUMOS-14/ActivityNet
   only).
2. **ARMS (IEEE TMM 2024)**: multi-phase self-training SPECIFICALLY on
   UCF-Crime, closest precedent, but still produces per-snippet SCORES not
   direct boundary regression.
3. **No paper has successfully done direct boundary regression for weakly-
   supervised VAD on UCF-Crime.** All SOTA methods (VadCLIP, GS-MoE, LEC-VAD,
   ARMS) still produce per-snippet scores + threshold/Gaussian-smooth them.

### Critical Risk Assessment (directly answering our specific concern)
Would a proposal network avoid our repeated "curve-shape integration" failure
mode by construction? Research concludes: PARTIALLY, NOT FULLY.
- Pseudo-labels bootstrapping the regression head STILL come from the same
  CLIP+MIL score curve - if that curve has the problems we've documented,
  the regression head inherits them (learns a parameterized version of our
  EXISTING half-max thresholding, not something fundamentally better).
- At inference, proposal-network boundaries would still need to be
  integrated/reconciled with CLIP verification - a NEW fusion problem in a
  different form, not elimination of the fusion problem.
- Genuine partial advantage: discrete proposal outputs have a smaller
  interaction surface than continuous per-snippet scores (fewer degrees of
  freedom to interact badly), but this is a risk REDUCTION, not risk
  ELIMINATION.

### Suggested Alternative (Temporal Gaussian Splatting from GS-MoE) - ALSO REJECTED
Agent recommended TGS (Gaussian kernels at score peaks to fill in
under-scored snippets of the same event) as a "low-risk, low-cost" alternative.
**We reject this too upon reflection**: TGS is fundamentally a SCORE-CURVE
SHAPE MODIFICATION - exactly the risk category that has failed 5+ times in
THIS pipeline (V72 feature re-centering, V78 confusion-negatives, V80 dynamic
MIL fusion, V81 MIL-TAG candidates, V83-85b chain). The fact that TGS helped
in GS-MoE's paper is in the context of an END-TO-END TRAINED model where the
whole scoring pipeline is jointly optimized WITH the Gaussian smoothing loss -
not as a post-hoc modification bolted onto an already-tuned zero-shot curve,
which is our actual situation. Applying it post-hoc carries the same risk
profile as our previous 5 failures, despite superficial novelty.

## DECISION: Do not implement either proposal networks or TGS.
Redirect remaining loop budget to approaches that operate STRICTLY on
already-finalized detections (post-hoc, detection-count-preserving) - the
ONLY category of change that has NOT regressed this entire session (V90's
VLM self-consistency, pending full-eval confirmation). Moving Loop
5-6 (originally slotted for proposal-network implementation) directly to
Loop 7's planned "final-detection-level ensembling" research instead.
