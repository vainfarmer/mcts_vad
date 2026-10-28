# v150 — JOINT A+B ANALYTICAL CLOSURE (loop 1, 20260801)

**Verdict: ANALYSIS-ONLY CLOSURE. Both genuinely-unattacked leads oracle-capped below +0.03. ZERO eval spent (valid outcome per FM11/FM13, cf. v135/v140/v149). Nothing shipped; working tree byte-identical to committed v137+BUE.**

Committed BUE-on 65-video baseline: mAP=0.21932 (AP@0.3=0.32097 / AP@0.5=0.19014 / AP@0.7=0.14685). Absolute floor 0.21432. REF = `results/batch_eval_v144c_bue65ref`.

## Mechanisms considered (neither shipped)
- **Lead A — cross-video LEVEL re-level.** `conf' = conf*(1-w + w*p(v))` with `p(v)` a global-comparable supervised MIL video prior. Multiplicative-monotone: preserves intra-video order/spread (Kendall-tau=1), only shifts each video's confidence band toward its correct cross-video level. NOT banned-4 (does not normalize the level away), NOT banned-1 (not within-video re-ranking).
- **Lead B — DSR dense-substrate relocation.** Keep 32-snippet scoring/counts/conf/BUE byte-identical; relocate only verdict-independently-unanchored detection WINDOWS onto the dense (fps=2) MIL-v1 half-max run, count-neutral + confidence-frozen, with a hard monotone abort (FM19). NOT banned-10 (that ban is verbatim scoped to the 32-snippet substrate, 0/17; B uses the denser substrate).

## ORACLE PRE-CHECK (all reproduced on the LIVE committed set; deterministic)
**Lead A** (`analysis_v150_xvlevel_orch.py`):
- Reproduces the v138 +0.0518 cross-video-LEVEL oracle (bounded-nudge +0.0496..+0.0593).
- Every realizable MIL prior REGRESSES: v6 blend w=0.5 **-0.05593**, v1 blend w=0.5 -0.05405, v6 MILmax level -0.04206, v1 MILmax level -0.10106.
- spearman(MIL-v6 max, per-video maxIoU) = **+0.1411**; spearman(MIL-v1 max, maxIoU) = +0.2722 → prior is near-**ORTHOGONAL** to the oracle target (FM7). MIL detects PRESENCE; the level oracle rewards localization QUALITY.
- frame_scores peak/std correlate +0.60/+0.56 with the target but are the flooded, verdict-DEPENDENT output curve (circular). No clean verdict-independent signal tracks the +0.0518. **LEAD A NULL (+0.0005).**

**Lead B** (`analysis_leadB_dense_oracle.py` GATE-1; `/tmp/gate2_proxy.py` GATE-2):
- GATE-1 (n=17): dense MIL-v1 verdict-independent **5/17 @IoU>=0.3**, 2/17 @0.5 (GT-aware 10/17, 5/17); mil6 3/17; motion 0/17. Beats the 0/17 32-snippet wall → family genuinely re-opened.
- GATE-2 (LIVE 65-video pred set, baseline 0.21932 reproduced):
  - vindep (realizable) relocation of all 16 wrong-loc videos → **mAP 0.22626 (+0.00694)**; AP@0.3 +0.0272, AP@0.5 +0.0006, AP@0.7 **-0.0070**.
  - GT-aware (unreachable) oracle → mAP 0.24460 (+0.02528); AP@0.3 +0.079, AP@0.5 +0.004, AP@0.7 -0.007.

## Effect-size vs actual
Authorization threshold was absolute mAP ceiling strictly clearing +0.03 (~0.249). Realistic arm 0.22626 and even the GT-aware oracle 0.24460 BOTH FAIL. The AP@0.3 headroom (+0.027 realistic / +0.079 GT-aware) **cannot propagate to +0.03 mAP** because AP@0.5 barely moves and AP@0.7 regresses (a tighter dense run loses loose-window @0.7 matches). Lead A moves no checkpoint (d ∈ [-0.0003, +0.0005]). No paired CI can clear zero on +0.0007 / +0.007 effects.

## BOOTSTRAP CI + ABSOLUTE mAP vs floor (headline)
No eval spent → no bootstrap CI. Absolute mAP stays **0.21932 = committed** (above floor 0.21432). Lead B's realizable 0.22626 would be above floor but its DELTA +0.007 is inside the ±0.025 noise floor and below the 0.01 paired KEEP threshold, so no eval was justified (FM11/FM16/FM18).

## FM1–FM19
- FM1 curve-shape: IMMUNE (both read the curve, don't reshape).
- FM2 count/match-set: IMMUNE (A monotone-multiplicative, B count-neutral) — irrelevant, effects null.
- FM3/FM4 VLM: N/A. FM5 never-activated: N/A (no ship).
- FM6 sim-transfer: N/A — GATE-2 ran on the LIVE committed pred set + live dense feats; the null is in-vivo.
- FM7 video-prior-can't-fix-localization: BINDING for Lead A (level oracle = maxIoU; prior sees presence; spearman +0.14).
- FM8 one-change: respected. FM9 MPS: MIL forward uses .cpu().numpy(). FM10: N/A.
- FM11 zero-eval-on-null: HONORED. FM12: N/A.
- FM13 structural cap: CONFIRMED FOR BOTH — A = signal-orthogonality; B = dense-substrate resolution ceiling (even GT-aware +0.0253 < +0.03).
- FM14: N/A. FM15 substrate: MIL on features/test correct; dense fps=2 feats are MIL's true substrate.
- FM16 noise floor: A +0.0005 and B +0.007 both << ±0.025 → unfalsifiable.
- FM17 empty-video marker: N/A (no ship). FM18 absolute floor: no ship, ABS mAP stays 0.21932.
- FM19 count-neutral-not-floor-safe: directly relevant to B — GATE-2 assumed PERFECT gating; a real gate firing on correctly-located videos would DEMOTE matched dets, eroding the sub-threshold +0.007. Closure is conservative.

## Status of each lead
- **Lead A (cross-video LEVEL calibration): CLOSED.** Oracle target (per-video maxIoU) is orthogonal to any verdict-independent supervised prior; every realizable re-level regresses mAP. This is a signal-orthogonality ceiling, not noise.
- **Lead B (dense-substrate relocation): CLOSED at the current selector.** Dense fps=2 carries real positional signal (GT-aware 10/17) — the ONLY remaining place with real positional signal — but the binding ceiling is verdict-independent SELECTION accuracy (5/17), which caps realizable mAP at +0.007. NOT closed by ban; closed by its own GATE-2 proxy.

## What the NEXT loop must NOT repeat
1. Do NOT re-propose Lead A as-is: no verdict-independent supervised prior tracks the level oracle (all MIL priors regress; spearman <=+0.27). A new IoU-regressor is a new model/training substrate (FM8 out of scope) and is Lead B's finer-substrate domain anyway.
2. Do NOT re-open Lead B with the half-max heuristic: it caps at 5/17 vindep → +0.007. Any re-open MUST first raise verdict-independent @IoU>=0.3 hit rate MATERIALLY above 5/17 with a NEW selector and re-clear GATE-1 AND GATE-2 (absolute ceiling > ~0.249) BEFORE touching config or spending eval.
3. Do NOT read the +0.079 GT-aware AP@0.3 as reachable — it needs GT.
4. Do NOT treat frame_scores peak/std (+0.60 corr) as a clean level signal — verdict-dependent, circular with confidence.

## Artifacts (offline, no pipeline import, no config/CLI change)
- `iteration_logs/analysis_v150_xvlevel_orch.py` (Lead A null)
- `iteration_logs/analysis_leadB_dense_oracle.py` (Lead B GATE-1: 5/17 vindep)
- `/tmp/gate2_proxy.py` (Lead B GATE-2: 0.22626 vindep / 0.24460 GT-aware; baseline 0.21932 reproduced)
- `iteration_logs/results/v150_precheck.txt` (this precheck, archived)
