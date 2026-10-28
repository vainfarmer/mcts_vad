# v145 — CSR: Curve-Support Count-neutral Relocation

Date: 2026-08-01
Loop: 1 (of the v145 workflow)
Verdict: **REVERT** — mechanism ran but produced a NULL effect (Δ mAP = 0.00000); full mAP 0.20599 < floor 0.21432.

## Mechanism
Count-neutral, width-preserving TRANSLATION of low-curve-support, far-mislocated FINAL
detections onto the within-video CLIP category-curve peak, placed in the terminal slot
IMMEDIATELY BEFORE committed BUE so BUE could regrow the relocated window.

- Read-only gate on the LIVE verdict-independent category curve (`frame_scorer.get_category_scores`,
  the exact signal BUE trusts). ZERO new VLM/CLIP/MIL calls.
- Fire iff `curve_in < csr_support_frac(0.6) * peak_val` (window lacks category-curve evidence)
  AND `|peak_time - window_center| > csr_min_gap_sec(8.0)` (peak is FAR, so BUE growth cannot reach it).
- On fire: translate the window width-preserving to center on `peak_time`, clipped to [0,duration].
  confidence / anomaly_type / rank / membership / detection COUNT frozen bit-for-bit.
- Explicitly NOT banned family 8: gate is the pre-calibratable curve, never raw scan_segment abs conf.
  This was the count-neutral repair of v143 (additive) and the empty-marker-fixed repair of v144 (crashed).

## Effect-size arithmetic — CLAIMED vs WHAT HAPPENED
- CLAIMED +0.033 (honest band +0.03..+0.045): ~15-20 far-mislocated zero-overlap dets satisfy the gate;
  CLIP peak lands on GT ~41% strict / ~53% at ±2s → ~6-9 FP→TP conversions; +2 net hits activates the
  currently-zero recall≈0.4 checkpoint (~+0.036 dominant term); BUE composes after to recover IoU width.
- WHAT HAPPENED: mechanism fired on only **3 detections** across the full 65-video eval
  (Arrest030 / Burglary061 / Fighting033), and those 3 relocations produced **ZERO change in any AP metric**.
  REF (BUE-on-original windows) and NEW (BUE-on-relocated windows) arms are **bit-identical**:
  mAP 0.20599, AP@0.3 0.30329, AP@0.5 0.17483, AP@0.7 0.13986 in BOTH arms.
- The predicted "15-20 fires" was off by ~5-7x (actual 3). The relocated windows either (a) still missed
  GT after translation, or (b) moved between two off-GT positions that both score 0 under the ranking, or
  (c) BUE regrew them back over a set that did not cross any of the ~4 attainable hit checkpoints.
  Net: the count-neutral design worked exactly as advertised (no FP demotion — decisively unlike v143's
  −0.018) but delivered NO gain either. Count-neutrality bought safety at the cost of leverage.

## HEADLINE VERDICT — the bootstrap CI
Paired-json mode on field `pre_csr_boundaries` (BUE held ON in both arms, isolating CSR):
- 101 index-aligned pre-mechanism rows; **3** windows actually changed via CSR; conf held X→X on all.
- Video-level bootstrap, 1000 resamples, n=65 common: **point_delta = +0.00000, sd 0.00000, ALL resamples identical.**
- CI90 [0.00000, 0.00000]; P(delta>0) = 0.000.
- GATE (verbatim): `point_delta >= 0.01: False (+0.00000)   P(delta>0) >= 0.9: False (0.000)   mAP >= floor 0.21432: False (0.20599)`
- All three gates FAIL. VERDICT: REVERT.

Note: the initial `--paired-log/--marker` attempt returned 0 marker fires (the exact log-marker string
never appears in run.log for the paired harness); retried in `--paired-json pre_csr_boundaries` mode,
which is the authoritative measurement here.

## FM1–FM17 table
- FM1 curve-shape: IMMUNE — curve read-only, never modified/rescaled.
- FM2 additive/FP-demotion: IMMUNE and CONFIRMED-immune — count frozen; ZERO metric movement, no demotion
  (the decisive contrast with v143's −0.018). But immunity also meant zero upside on this run.
- FM3 context-bleed: IMMUNE — no VLM call.
- FM4 per-video normalization: N/A — gate is a within-video RATIO on windows, not scores/levels.
- FM5 activation: per-fire + per-video summary markers present; 63 per-video summaries on full (65/65 JSONs,
  63 = one per succeeded video; 2 non-succeeded); smoke 13/13 summaries incl. empty/guard paths.
  **FM17 (empty-video deadlock) FIXED and VERIFIED** — no [GATE] crash, report written cleanly (unlike v144).
- FM6 sim-transfer: N/A here — gate ran on the live faithful curve, no offline precheck involved in the verdict.
- FM7 video-level prior: N/A — per-detection intra-video relocation.
- FM8 one change: single terminal-slot relocation method.
- FM9 MPS: .detach().cpu().numpy() guard present on curve read.
- FM10 wall-clock: not API-driven (latency ~148s/video, in-band).
- FM11 one-eval validatable: yes.
- FM12 no merge/batch: satisfied.
- FM13 magnitude: the REAL failure — the fired population was tiny (3, not 15-20) and each fire had zero
  checkpoint leverage. Magnitude ceiling arithmetic (15/37 recoverable) DID NOT materialize in vivo.
- FM14 verdict-visibility: N/A conf frozen.
- FM15 substrate infidelity: AVOIDED — live faithful curve, not features/test.
- FM16 noise floor: the paired harness worked; delta is a hard 0, not noise-masked.
- FM17 empty-video marker deadlock: FIXED (the whole point vs v144); confirmed live.

## Honest weaknesses (materialized)
1. **The gate is far too selective in vivo.** It predicted 15-20 fires and got 3. The two-condition AND
   (`curve_in < 0.6*peak` AND `|peak−center| > 8s`) is much rarer on the real 65-harness than the thin
   n=17 offline calibration suggested. The 50/99 zero-overlap dets do NOT mostly present as "low local
   curve support + far peak" — many have moderate local support or a nearby (< 8s) peak, so they slip the gate.
2. **Even when it fired, count-neutral translation had no leverage.** Moving a window from one off-GT
   position to another (or onto a peak that is itself off-GT, or onto GT but at IoU too low to cross a
   checkpoint even after BUE regrowth) scores identically under global AP. The claimed "+2 net hits activate
   the recall≈0.4 checkpoint" never occurred: 3 relocations, 0 new hits.
3. Calibration was on n=17 wrong-location dets with n=2 TP controls — far too thin to predict fire-rate or
   on-GT fraction, exactly as the proposal's own weakness #2 warned. The load-bearing unknowns stayed unknown
   until the eval was spent.

## What the NEXT loop should NOT repeat
- Do NOT re-propose curve-support/curve-peak COUNT-NEUTRAL relocation in this form. It is now empirically
  shown to be either (a) too selective to fire enough, or (b) leverage-free when it does. Count-neutral
  relocation onto the live category-curve peak is effectively CLOSED for the far-miss population as gated here.
- Do NOT trust an offline n≈17 fire-set calibration to predict in-vivo fire-rate — v145 over-predicted fires ~6x.
- Do NOT assume BUE composition rescues IoU after relocation — it did not move any checkpoint here.
- The candidate-generation / recall ceiling (50/99 IoU=0 dets, max recall 0.379@0.3) remains OPEN and
  UNSOLVED after v143 (additive, floor-breach), v144 (crashed), v145 (null). The three refutations jointly
  imply: a purely-relocating count-neutral move onto the CLIP peak lacks leverage, and an additive VLM move
  demotes precision. The remaining unrefuted direction is putting detections on the RIGHT event via a
  supervised, pre-calibratable localizer (MIL snippet scores / model_mil_v6.pth / actionness on features/test)
  strong enough to change which windows survive/rank — NOT another category-curve-peak geometry trick, and
  NOT another raw scan_segment abs-conf gate.

## Rollback done
5 files restored from `/Users/zzl/PycharmProjects/vad/.qoder/backups/v145/` (pipeline.py, config.py,
models.py, run.py, batch_evaluate.py), diff -q byte-identical. `grep -i csr` = 0 matches. py_compile PASS.
`from_args(Namespace()).scoring == ScoringConfig()` → True; bue_enabled=True (untouched), cex_extend=False
(untouched). No destructive git used. Baseline stays committed BUE-on pipeline; floor 0.21432 held (nothing shipped).
