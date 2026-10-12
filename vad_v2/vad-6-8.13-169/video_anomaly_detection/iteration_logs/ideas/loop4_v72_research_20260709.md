# Loop 4 Research: Training-Free Temporal Modeling Innovation
## Date: 2026-07-09
## Version: V72 (to be implemented)

## Agent A Findings (Temporal Modeling Focus):
### Top Proposals:
1. **Feature Re-centering + Temporal Difference** [RANK 1]
   - Source: AnomalyCLIP (Zanella et al.) + FDA-CLIP (2025) + Sensors 2023
   - Subtract normal prototype from CLIP features -> anomaly = magnitude deviation
   - Temporal difference: ||feature[t+1] - feature[t]|| -> sharp onset/offset peaks
   - Training-free, pure numpy operations
   - Expected: +5-15% mAP

2. **LogOIC Multi-Scale** [RANK 2] - Already tried V65-V68, FAILED (over-narrows long events)

3. **Savitzky-Golay + Second Derivative** [RANK 3]
   - Classical signal processing for noise reduction + inflection point detection

4. **Flashback Memory Bank** [RANK 5]
   - Repulsive prompting to maximize normal/anomalous separation
   - 87.3% AUC on UCF-Crime (zero-shot SOTA)

## Agent B Findings (Anomaly Detection Focus):
### Top Proposals:
1. **CoReVAD Post-Processing** [RANK 1 - DO FIRST]
   - Gaussian smoothing + position weighting + softmax refinement
2. **RTFM/MGFN Magnitude Learning** - Retrain MIL with magnitude regression
3. **PE-MIL Prompt-Enhanced MIL** - CVPR 2024
4. **VadCLIP Dual-Branch** - AAAI 2024
5. **GS-MoE Temporal Gaussian Splatting** - ICCV 2025, 91.58% AUC

## Critical Discovery:
- UCF-Crime temporal localization mAP (IoU 0.1-0.5) = 6-9% even for supervised SOTA
- Our 0.14 at IoU 0.3/0.5/0.7 is competitive for zero-shot
- mAP=0.50 requires fundamental architectural change

## Debate Decision - V72 Implementation:
Feature Re-centering + Temporal Difference + CoReVAD Post-Processing (all training-free)
