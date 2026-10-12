# Loop 6: MIL-Guided Candidate Generation
## Date: 2026-07-09
## Version: V74

## Problem
~29% of anomalous videos have candidates in completely wrong temporal locations.
Current candidate generation relies solely on CLIP cosine similarity peaks.
When CLIP fails (subtle anomalies, visually ambiguous), there's no backup.

## Research Findings (Agent A + Agent B)
### Agent A top proposals:
1. **TSM (Temporal Self-Similarity Matrix)** — visual transition detection from CLIP features
2. **MIL Score-Guided Candidates** — use MIL v2 scores directly
3. **VADTree GEBD** — pre-trained event boundary detection (requires new model)

### Agent B top proposals:  
1. **TAG (Temporal Actionness Grouping)** — multi-threshold connected components on MIL scores
2. **Gaussian Splatting proposals** — model anomaly peaks as temporal Gaussians
3. **Hybrid MIL + CLIP** — parallel candidate sources with cross-source boosting

## Implementation Decision: V74
Add MIL v2 snippet scores as a PARALLEL candidate source alongside existing CLIP-peak candidates.

### Architecture Change:
```
BEFORE: CLIP features → similarity peaks → candidates → boundary detection
AFTER:  CLIP features → similarity peaks → CLIP candidates ─┐
        MIL scores → TAG thresholding → MIL candidates ─────┤→ Merge → boundary detection
```

### Key Components:
1. Multi-threshold TAG on MIL scores (Otsu + fixed thresholds)
2. Gap-filling (merge adjacent segments separated by ≤1 snippet)
3. Union-merge with existing CLIP candidates
4. Cross-source confidence boost when MIL and CLIP agree

## Expected Impact
- Should rescue the ~29% of videos where CLIP peaks are wrong
- MIL v2 has good discrimination (abn=0.64 vs nor=0.07)
- Minimal risk — adds candidates, doesn't remove existing ones
