# v151 — JOINT A+B ANALYTICAL CLOSURE (re-verified) — 20260801

## Headline
**ANALYSIS-ONLY CLOSURE. ZERO eval/API spent (FM11/FM13). Nothing shipped, nothing to roll back.**
Working tree byte-identical to committed v137+BUE (`bue_enabled=True`, `cex_extend=False`,
mAP **0.21932**, floor **0.21432**). Baseline reproduced to the digit on the committed
65-video BUE-on cache `results/batch_eval_v144c_bue65ref`.

BOOTSTRAP CI: **none** — no NEW arm was produced, so no video-matched paired CI exists.
ABSOLUTE mAP vs floor: unchanged at **0.21932 (> floor 0.21432)** because no code changed.

This is the successor / independent re-run of v150's closure. Both closures agree: the two
genuinely-unrefuted leads (A cross-video LEVEL calibration, B finer-resolution relocation)
each fail their OWN pre-registered oracle gate, for two DIFFERENT and now precisely-located
reasons.

## Mechanism (proposed, then refuted by its own oracle)
- **Lead A**: a verdict-independent, pre-calibratable supervised video-level prior
  (model_mil_v6.pth / model_mil.pth max & mean over 32 snippets on features/test) used to
  RE-SET each video's confidence LEVEL toward the correct cross-video order while FREEZING
  intra-video rank — generalizing the MIL-sanity gate at pipeline.py:1579. NOT banned-4
  (that normalizes the level AWAY; this SETS it), NOT banned-1 (curve-derived WITHIN-video).
- **Lead B**: count-neutral, conf-frozen, position-only relocation of after-BUE zero-overlap
  detections onto runs from a DENSER localizer substrate (fps=2 dense MIL-v1 curve,
  93–1434 frames/video, far finer than the 32-snippet MIL), verdict-independent selector =
  half-max contiguous run around global argmax. NOT banned-10 (denser substrate escapes the
  32-snippet resolution cap), NOT banned-8 (never scan_segment abs conf), BUE untouched.

## Oracle pre-check result (the decisive evidence)
- **LEAD A** (`analysis_v150_xvlevel_orch.py`): pure cross-video-LEVEL PERFECT oracle (intra
  order frozen) = **+0.01891** (meanIoU; mAP 0.23823) / **-0.00317** (maxIoU). Every real
  prior REGRESSES: v6 max -0.04206, v6 mean -0.14296, v6 blend -0.05593, v1 max -0.10106,
  v1 mean -0.13285, v1 blend -0.05405. spearman(MIL,maxIoU) v6=+0.1411, v1=+0.2722.
  -> anomaly-PRESENCE != localization-QUALITY (near-orthogonal). Perfect ceiling +0.019 < +0.03.
- **LEAD B GATE-1** (`analysis_leadB_dense_oracle.py`, n=17): dense MIL-v1 verdict-independent
  **5/17** @IoU>=0.3 (GT-aware 10/17); v6 3/17; motion 0/17. Reach wall (0/17 on 32-snippet)
  is genuinely BROKEN — density carries positional info. GATE-1 PASSES.
- **LEAD B GATE-2** (`/tmp/leadb_map_recompute.py`, 66 REF files): baseline 0.21932 exact;
  GT-aware relocation **+0.03222** (mAP 0.25154); verdict-independent argmax **+0.00985**
  (mAP 0.22916). The +0.02237 gap is PURE selector quality (FM14).

## Effect-size vs actual
| Lead | proposer claim | perfect/GT-aware oracle | achievable verdict-independent | gate (>=+0.03 & floor) |
|---|---|---|---|---|
| A | re-level toward +0.052 headroom | +0.01891 (meanIoU) / -0.00317 (maxIoU) | negative (-0.042 to -0.143) | FAIL |
| B | +0.03 recall recovery | +0.03222 (GT-aware, UNREACHABLE) | +0.00985 (inside +/-0.025 noise) | FAIL |

Neither reaches 0.24932 (=0.21932+0.03) verdict-independently.

## FM1–FM19 table
- FM1 curve-shape regress: IMMUNE (no curve edit).
- FM2 additive/count->FP: IMMUNE (no change / Lead B count-neutral).
- FM3 VLM bleed: IMMUNE (no VLM path). FM4 self-consistency: N/A. FM5 never-activated: N/A (no code).
- FM6 offline->live non-transfer: RESPECTED — oracles are UPPER bounds; even GT-aware/perfect
  (+0.032 / +0.019) are unreachable verdict-independently, so no optimistic gain transfers.
- FM7 priors can't fix intra-video mislocation but CAN fix cross-LEVEL: DIRECTLY TESTED —
  cross-level perfect caps +0.019, real priors regress. **Binding for Lead A.**
- FM8 one change/loop: HELD (zero changes). FM9 MPS detach: N/A (oracles .cpu().numpy()).
- FM10 wall-clock: N/A. FM11 zero eval on predetermined outcome: HONORED.
- FM12: N/A. FM13 structural/temporal-resolution cap bounded BEFORE eval: THIS IS THE CLOSURE
  (A signal-orthogonality cap, B selector-quality cap).
- FM14 verdict-visibility: **BINDING for Lead B** — recovered dets low-conf, no-GT argmax
  mis-ranks decoy runs; the decisive FM.
- FM15 substrate infidelity: AVOIDED (MIL on features/test; dense oracle on cached fps=2 track).
- FM16 noise floor +/-0.025: moot — realized effects (+0.019, +0.00985, negatives) below floor AND +0.03.
- FM17 empty-video marker: N/A (no marker added).
- FM18 paired-vs-ABSOLUTE floor: CENTRAL — Lead A realized mAP 0.08–0.17 (< floor 0.21432);
  Lead B verdict-independent tops at 0.22916 (> floor but < 0.21932+0.03 target, inside noise).
- FM19 count-neutral not floor-safe: Lead B position moves demote some matched dets; even
  GT-aware nets only +0.032, verdict-independent +0.00985; unsafe.

## Lead status
- **Lead A (cross-video LEVEL calibration): CLOSED** — signal orthogonality (spearman<=0.27;
  perfect ceiling +0.019 < +0.03; every real prior negative). Do NOT re-propose.
- **Lead B (finer-resolution relocation): CLOSED at current selector** — reach wall broken
  (GT-aware +0.03222, VI 5/17) but achievable verdict-independent +0.00985 (FM14 selector cap),
  inside noise. Only re-open with a NEW GT-free selector whose OWN oracle beats +0.03
  verdict-independently (max room is +0.02237 to the GT-aware ceiling — almost certainly < +0.03).

## What the next loop should NOT repeat
1. Do NOT re-propose Lead A — no verdict-independent supervised prior tracks the level oracle
   (presence != localization-quality).
2. Do NOT re-open Lead B with the half-max argmax heuristic (caps 5/17 -> +0.00985), nor treat
   the +0.079 GT-aware AP@0.3 as reachable.
3. Do NOT re-propose relocation on the 32-snippet substrate (v149 closed) or any of the 10
   banned families.
4. Do NOT spend eval on a pre-measured negative/+0.00985 null (the v135/v140/v149/v150 process
   failure — HONORED here).
5. The ONLY residual legitimate direction is a multi-loop upstream BSN/BMN-style
   GT-IoU-supervised localization-quality/boundary head (needs frame-level IoU supervision the
   weakly-supervised setup lacks), or a learned GT-free dense run-ranker for Lead B. Standing
   gate on ANY re-proposal: a NEW oracle showing a verdict-independent selector > +0.03. Escalate
   to owner; do not eval on the current substrate.

## Artifacts
- Precheck: `results/v151_precheck.txt`
- Oracles (read-only, re-run this loop):
  `iteration_logs/analysis_v150_xvlevel_orch.py`,
  `iteration_logs/analysis_leadB_dense_oracle.py`,
  `/tmp/leadb_map_recompute.py`
- No smoke, no full eval, no bootstrap (analysis-only).
