# v165 CGR-M — Category-gated MOTION-peak count-neutral candidate Relocation

**Loop 2 (session v97) · 2026-08-09 · VERDICT: REVERT** (cross-run P(delta>0)=0.780 < 0.9)

## Mechanism (ONE change on the wired v164 CGR-R scaffold)
PRE-BUE, count-neutral RELOCATE-only. v164 CGR-R wired a scaffold (`_cgr_relocate`, called PRE-BUE;
`cgr_enabled=False`; `--cgr` CLI; `pre_cgr_boundaries` paired field; `[CGR-VID]` marker on all paths incl
empty) that NEVER FIRED — FM5 death: its relocation TARGET was `frame_scorer.get_category_scores`, a
softmax cat-vs-normal curve that is NEARLY FLAT on diffuse Shoplifting → `prom < cgr_min_prom` →
`reason=weak_peak` → no fire. **v165's single change:** swap that flat target for the per-frame
MOTION-ENERGY curve (`video_data['motion_energy']`, computed unconditionally from decoded frames, read-only,
min-max scaled only for internal prominence/peak-selection math — confidences never touched), PLUS a
category-CLIP FLOOR: fire only when `cat_curve[peak_idx] >= median(cat_curve)` so the motion peak coincides
with baseline category-appearance evidence. All existing gates kept: real prominence, UNCOVERED peak-window,
SOURCE = lowest-conf off-signal det with `protected_local_peak` guard (FM19), at most ONE window moved.
Untouched BUE then grows the relocated seed outward, CBT clamps it inward. Zero detections added/deleted.

## Per-category cand-miss diagnosis (v155 cache; nGT/hit@.3/hit@.5/boundary-miss/cand-miss)
- Shooting 25/9/7/11/**5** · RoadAccidents 23/6/0/13/**4** · Shoplifting 25/3/2/11/**11** · Explosion 22/10/7/11/**1**
- CBT (inward clamp) already reclaimed the boundary-too-wide miss (RoadAccidents AP 0.067→0.202). Remaining
  and CBT-unfixable = CAND-MISS (pipeline never proposes a window overlapping GT). Shoplifting 11/25 is the
  primary leverage; theft actions were argued motion-prominent + late/uncovered.

## Result — HEADLINE
- **SMOKE(14):** mAP 0.2273 · AP@0.3 0.333 · AP@0.5 0.197 · AP@0.7 0.152 · 138.4 s/vid · markers 14/14 · pairedJSONs 14/14.
- **FULL(88 hard-harness):** all-set temporal mAP 0.1710 · markers 289/289 · pairedJSONs 289/289 · 176.0 s/vid.
- **PRIMARY cross-run vs `results/batch_eval_v155_full` (n=88 hard):**
  point_delta **+0.02143**, 90% CI [**-0.02027, +0.05906**], **P(delta>0)=0.780**, ABSOLUTE hard **new_map=0.12113**.
  → **KEEP gate FAILS**: new_map 0.12113 ≥ 0.11470 ✓ AND ≥ 0.09470 floor ✓ BUT P=0.780 < 0.9 ✗ → **REVERT**.
- **SECONDARY paired-json[pre_cgr_boundaries] (n=289):** delta +0.04445, 90% CI [+0.02261, +0.07937], P=1.000, mAP 0.17097 → KEEP.
  Adjudication: for a cand-gen mechanism that CHANGES the detection SET, cross-run is PRIMARY; paired P=1.0
  is inflated because it compares pre/post windows within the same VLM run. Cross-run carries VLM
  run-to-run variance (~±0.019) on top of the CI, which is exactly what pushes P below 0.9. DECISION = REVERT.
- **ABSOLUTE vs committed CBT-ON baseline 0.0997:** new_map 0.12113 is +0.0214 above — a positive point
  estimate, but not P-robust on the hard harness (the binding KEEP test).

## Per-category AP delta (v165 vs v155) + count delta
- Shooting **+0.0162** (0.094 vs 0.078) · RoadAccidents **+0.0165** (0.218 vs 0.202) ·
  Shoplifting **+0.0000** (0.073 vs 0.073) · Explosion **-0.0023** (0.159 vs 0.161, noise).
- **FM2 count delta (headline):** mean dets/video new 1.478 vs ref 1.456 = **+0.022/vid** (~count-neutral;
  271 detection windows RELOCATED, count near-flat). FM2 disarmed by construction — no added detections.

## Instrument self-test (D1 gates — ALL PASSED, incl. the exact test v164 FAILED)
- FM20 pre_cgr_boundaries: written index-aligned, same length, on main + BOTH empty/0-det paths; smoke JSONs
  asserted non-null with len==len(anomalies) (Shoplifting001 2==2, 033 1==1, 037 1==1). Full: 289/289.
- FM21 marker `[CGR-VID]`: unconditional incl empty path; smoke grep == #videos (FM17-safe); full 289/289 (log==inner).
- FM5 (v164 killer, FIXED): `grep 'fired=1 reason=relocated'` ≥ 1 on smoke — Shoplifting037 fired (motion
  curve is prominent where get_category_scores was flat). Mechanism ACTUALLY FIRES now.
- C-mandate off-GT source: Shoplifting037 relocated source OLD window 18.34–31.26 vs GT 38.0–40.0 = zero
  overlap → genuinely off-GT, no TP vacated. Count-neutral confirmed (conf 0.8038→0.8038 identical).

## FM table
- FM2 (added→FP, the cand-gen killer): DISARMED — relocate-only, len byte-identical, +0.022/vid, no adds.
- FM5 (never-activated, v164 killer): FIXED — motion_energy target fires where flat get_category_scores did not.
- FM6 (offline→in-vivo): motion_energy is the SAME cached array the pipeline computes; no skew.
- FM8: single lever (target-signal swap + one category floor gate).  FM9: numpy CPU array; cat_curve detached.
- FM13/FM16/FM18: absolute hard new_map 0.12113 is the honest headline; point +0.02143 sits INSIDE the
  ±0.02 CI half-width → the P(delta>0)≥0.9 gate (not the point estimate) is binding, and it FAILS at 0.780.
- FM17: [CGR-VID] on both 0-det paths.  FM19: source = lowest-conf off-signal + protected_local_peak; miss = net-zero.
- FM20/FM21: reused proven write path + exact marker grep; self-tested.
- **REALIZED interaction (new lesson):** the relocated seed, though it reaches GT, is subsequently pulled
  OFF a LATE GT by the downstream CBT inward-clamp. Shoplifting037: motion peak_t=35.0 sat ~3 s EARLY of GT
  38–40; relocated seed 31.50–38.50 touched GT, but CBT clamped the FINAL window to 35.0–37.0 (IoU=0). This
  fire netted ZERO — count-neutral, floor-safe, no regression, but no conversion. This BUE-grow/CBT-clamp
  interaction (not FP explosion) is the reason Shoplifting delta ended +0.0000 despite firing.

## WON / OPEN / CLOSED
- **WON:** motion_energy target resurrects the FM5-dead v164 scaffold — CGR now FIRES (reason=relocated).
  Count-neutral relocation is FM2-safe (+0.022/vid, 271 windows moved). Paired signal is strongly positive
  (P=1.0), and cross-run POINT estimate (+0.0214, new_map 0.12113) is above both the KEEP bar and 0.0997 —
  the mechanism is directionally right, just not P-robust on the noisy hard harness.
- **OPEN:** Shoplifting cand-miss 11/25 STILL unaddressed. Root cause is now sharper: CGR-M DOES relocate a
  window onto/near the late GT, but (a) the category-CLIP floor suppressed 2/3 smoke fires (undercuts fire
  rate), and (b) the motion peak is ~3 s EARLY of the concealment GT and CBT then clamps the grown seed OFF
  the GT. So relocation reaching GT is necessary but not sufficient — the seed must SURVIVE BUE+CBT on it.
- **CLOSED:** CGR-M (category-gated motion-peak count-neutral relocation) as a KEEP-clearing lever
  (cross-run P=0.780 < 0.9; do not re-propose motion-peak CGR relocation as-is). Reaffirms v164: the
  category-CLIP-curve (get_category_scores) target is flat → dead, already CLOSED.

## Next-loop guidance
1. The relocation LANDS but CBT clamps it OFF late Shoplifting GT (motion-peak-vs-GT offset ~3 s). A future
   variant would need the relocated seed SHIELDED from the CBT inward-clamp toward the early motion peak — but
   that stacks two mechanisms (violates one-change) and re-touches CBT (v161–163 CLOSED family). Prefer a
   cand-gen lever whose output SURVIVES the committed BUE+CBT unchanged.
2. Consider LATE-biased window placement for Shoplifting (motion spike = grab/approach; GT = later
   concealment), or a source-signal that peaks ON the concealment rather than the motion transient.
3. Cross-run VLM variance (~±0.019) inflated the CI and cost the P≥0.9 gate despite a +0.0214 point gain and
   new_map 0.12113 > 0.0997. For any cand-gen loop, this variance is unavoidable (detection set changes) —
   budget for a larger effect (target ≥ +0.03 point) so P clears 0.9 through the variance.
4. RoadAccidents +0.0165 and Shooting +0.0162 were the real (if modest) gainers here — motion-prominence
   helped brief-event categories; Explosion/Shoplifting flat. A brief-event-only CGR could be more P-robust.

## Revert record
Committed hard baseline stays committed BUE+CBT+recal (hard new_map 0.0997). REVERT verified:
- pipeline.py restored wholesale from `.qoder/backups/v162/pipeline.py` (delta vs current was CGR + version
  header only; 0 CGR symbols after). config.py/run.py/batch_evaluate.py surgically stripped of CGR ONLY
  (the v164 CBT/recal default-ON promotion in these files was PRESERVED — it is a separate KEPT change).
- Verified: `grep -ci cgr` = 0 across all 4 files; py_compile OK; `import MultiAnomalyVAD` OK;
  ScoringConfig() bue_enabled=True (ON), cbt_enabled=True, cbt_recal=True with v160 caps
  {RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5}/trigger 1.1, cex_extend=False (OFF),
  no cgr attr; `VADConfig.from_args(Namespace()).scoring == ScoringConfig()` True.
- Current-v165 working state backed up to `.qoder/backups/v165_cgrm/{pipeline,config,run,batch_evaluate}.py`.
- Artifacts: `iteration_logs/results/v165_{hard_eval.json,hard_run.log,bootstrap_ab.json}`
  (bootstrap_ab.json on disk is the SECONDARY paired KEEP; primary cross-run REVERT numbers above).
- NEVER git checkout/reset/stash.
