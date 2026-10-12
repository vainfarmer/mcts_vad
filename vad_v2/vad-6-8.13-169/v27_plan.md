# V27 Pipeline Iteration Plan

## Context

V26 achieved 83.3% category accuracy (50/60), up from v24's 80%. But two major issues remain:
1. **Normal video false positives**: 4/5 Normal videos misclassified as anomalous (80% FP rate)
2. **Temporal localization**: AP@0.5 = 7.2%, mean IoU = 0.322
3. **10 remaining category errors**

## Changes (all in pipeline.py)

### 1. Remove "No LLM hits → CLIP regions" fallback (Normal FP fix)
### 2. Minimum confidence threshold 0.72 for final anomalies (Normal FP safety net)
### 3. CLIP-override threshold: <50% → <70% (Category accuracy)
### 4. VLM majority voting (3 calls) for confusion map arbitration (Category accuracy)
### 5. Expand LLM max_scan: 14 → proportional to video length, up to 24 (Temporal)
### 6. MCTS search range: ±50% → ±75% (Temporal)
### 7. Update docstring to v27

## Expected impact
- Normal FP: 4 → ≤2
- Category accuracy: 83% → ~85%
- Temporal AP@0.5: improvement from 0.072
