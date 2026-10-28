# Loop 4 (v112) — Stack duration-prior fit onto Q-score

## Proposal
Add category duration-prior fit (from CATEGORY_DURATION_PRIORS) as an INDEPENDENT
signal onto the kept v111 Q-score: a detection whose duration matches its category's
expected [lo,hi] is more likely correct. Two variants: (A) Q4 = mean(sharp, contrast,
center, dur_fit); (B) Q3 × separate dur multiplier.

## C review: APPROVE (offline pre-check, stacking on v111)
Same safe pattern. Loop baseline = v111 QSCORE (0.16703), not raw v108.

## D result (offline, deterministic on v108 detections)
Anchor: v111 Q3 reproduces 0.16703 ✓.

| config | mAP | vs v111 |
|---|---|---|
| v111 Q3 (base) | 0.16703 | — |
| Q4 (add dur_fit, base 0.5) | 0.16703 | +0.00000 |
| Q4 (base 0.3) | 0.16703 | +0.00000 |
| Q3 × dur (α=0.2) | 0.15911 | -0.00792 |
| Q3 × dur (α=0.3) | 0.15737 | -0.00967 |
| Q3 × dur (α=0.5) | 0.15589 | -0.01115 |

## Decision: REJECT (v111 stays the baseline)

## Lessons
- duration-fit is NON-DISCRIMINATIVE here: the pipeline ALREADY enforces the same
  CATEGORY_DURATION_PRIORS as boundary constraints (max_dur reject-cap, min_segment),
  so nearly every final detection already fits its prior → dur_fit ≈ 1.0 for all →
  zero variance → no effect after within-video min-max normalization (Q4 neutral).
- As a separate multiplier (Q3×Dur) it REGRESSES: the few out-of-prior detections it
  demotes are not systematically wrong, and the demotion disrupts v111's good ordering.
- Generalizable: a re-rank signal that is ALREADY baked into the pipeline (as a boundary
  constraint here) carries no fresh discriminative content. Stack only signals ORTHOGONAL
  to what the pipeline already uses.
