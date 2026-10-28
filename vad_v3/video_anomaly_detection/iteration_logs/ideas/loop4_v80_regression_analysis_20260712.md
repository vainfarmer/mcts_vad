# Loop 4: V80 Full Eval Regression Analysis
## Date: 2026-07-12

## Result
V80 (MIL v6 + dynamic fusion weight): mAP=0.1106, AP@0.3=0.2008, AP@0.5=0.1082,
AP@0.7=0.0227. REGRESSION from V51 baseline (0.1492) and session baseline (~0.13).

## Root Cause Analysis
The spec's Step 1.4 dynamic fusion formula:
  mil_weight = min(0.5, 0.25 + 0.5 * (mil_range / 0.3))

This was calibrated assuming MIL range would be a SPARSE high-confidence
signal (occasionally high range = trust more). But MIL v6's anti-collapse
training was SO effective that median range is now 0.495 - which means:
  mil_weight = min(0.5, 0.25 + 0.5*1.65) = min(0.5, 1.075) = 0.5 (the CAP)

So essentially ALL videos now hit the 0.5 cap (2x the original 0.25 weight).
This is not "dynamic" in practice - it's just "always 0.5" given v6's much
improved baseline range.

## Why higher MIL weight hurts despite MIL being "better"
Historical evidence from THIS project (documented across V67, V68, V73):
- V67: removing MIL fusion ENTIRELY (0% weight) changed Fighting IoU 0.574->0.176
- V68: restoring MIL fusion (25% weight) recovered this
- V73: MIL v2 (also improved discrimination vs v1) at 25% weight STILL caused
  mAP to drop 0.1492->0.0867 because MIL's temporal PEAKS don't necessarily
  align with CLIP's peaks, and fusing two curves with different peak
  LOCATIONS (not just different quality) shifts the half-max crossing points
  in ways that don't correlate with ground truth improvement.

The half-max boundary detection algorithm is sensitive to the SHAPE of the
fused curve, not just each component's individual quality/discrimination.
Even a "better" MIL model, at higher fusion weight, can shift boundaries in
directions uncorrelated with the true anomaly location, because MIL and
CLIP capture different (imperfectly correlated) notions of "anomalous."

## Decision: V83
Revert boundary-fusion MIL weight to fixed 0.25 (V51's proven value).
Keep MIL v6 model (still better for OTHER uses) but stop trying to increase
its influence on the CURVE SHAPE used for boundary detection. Use MIL v6's
improved discrimination ONLY via V81's MIL-TAG candidate generation, which
is an ADDITIVE mechanism (union of proposal regions) rather than a curve-
shape-modifying mechanism - fundamentally different risk profile.

## Lesson for future loops
"Better" component-level metrics (AUC, within-video range, collapse rate) do
NOT guarantee better end-to-end pipeline mAP when that component feeds into
a SHAPE-SENSITIVE algorithm (half-max crossing). Any change to fusion
weights/curve-shape must be evaluated via full pipeline eval, not just
component-level metrics - this is the SECOND time this exact lesson has
been learned in this project (first: V73 MIL v2, now: V80 dynamic weight).
