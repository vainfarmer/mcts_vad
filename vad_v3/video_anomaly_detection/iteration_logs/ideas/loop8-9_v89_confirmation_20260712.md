# Loop 8-9: V89 Isolated Full Eval - Diverse-Perspective Prompting
## Date: 2026-07-12

## Result
V89 (Hard-NMS restored + diverse-perspective self-consistency KEPT):
mAP=0.1494, AP@0.3=0.2258, AP@0.5=0.1313, AP@0.7=0.0909

## Comparison
| Version | mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---------|------|--------|--------|--------|
| V51 (historical best) | 0.1492 | 0.2931 | 0.1213 | 0.0331 |
| V69 (V51-equivalent, prior session) | 0.1410 | 0.1985 | 0.1337 | 0.0909 |
| V89 (diverse prompts) | 0.1494 | 0.2258 | 0.1313 | 0.0909 |

## Analysis
V89's mAP (0.1494) matches V51's (0.1492) almost exactly. The per-threshold
breakdown shows AP@0.7 identical to V69's 0.0909 (both far above V51's 0.0331)
while AP@0.3 sits between V51 and V69. This exact AP@0.7 value (0.0909)
matching a PREVIOUSLY-DOCUMENTED pure-noise comparison (V51 vs V69, same code,
different VLM API run) strongly suggests V89's profile is within normal VLM
stochasticity, not a genuine effect of diverse-perspective prompting.

## Conclusion
Diverse-perspective self-consistency for category tie-breaking is NEUTRAL:
- Does not regress mAP (confirmed via full eval, unlike Soft-NMS)
- Does not clearly improve mAP either (differences consistent with noise)
- Category accuracy remains solid (84.48%, matching historical ~83-87% range)
- Zero extra cost (same 3-call count), so even a neutral result has no
  downside - keeping it is a reasonable default since it's theoretically
  motivated (self-consistency literature) even if not empirically proven
  on this specific noisy benchmark.

DECISION: V89 becomes the session's best confirmed-safe production version.
