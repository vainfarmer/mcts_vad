# v157 — CBT-T: recalibrate the CBT inward-clamp fire gate (cbt_trigger_mult 1.5 -> 1.1)

**Session:** v12 loop3 (hard-harness objective: raise temporal mAP on Shooting/RoadAccidents/Shoplifting/Explosion, 88 videos).
**Baseline (committed v155 CBT-on):** hard-harness mAP 0.09970 (AP@0.3 0.2424 / AP@0.5 0.076 / AP@0.7 0.023). Do-not-regress floor 0.09470.
**VERDICT: REVERT — smoke-gate marker failure (FM5/FM17 per orchestrator), 88-video eval never launched. Lever OPEN (never measured), NOT closed.**
**KEPT: false. smoke_map: 0.1273. full_map: -1 (skipped).**

## Mechanism (one scalar)
CBT (v155, the shipped +0.05446 hard-harness win) is a terminal, post-BUE, count-neutral, confidence-frozen INWARD clamp: for FINAL dets of the 4 hard cats whose `det_dur > 1.5*cap` (CBT_CAPS = per-cat GT-p75: RoadAccidents 6.0, Explosion 13.0, Shoplifting 10.0, Shooting 14.0), it recomputes the category relative-score curve inside the span, centers on the in-span peak, grows an AutoLoc contrast-optimal window, and clamps monotone-inward to `[t_s,t_e]`.

CBT-T exposed the hardcoded `1.5` trigger as config `cbt_trigger_mult` (default 1.5 = byte-identical to v155) and set it to **1.1**, making dets merely 10%+ over GT-p75-cap eligible for the SAME already-shift-capable clamp. ONE change: the scalar `cbt_trigger_mult`. Implemented + validated at unit level, then reverted.

## Per-category diagnosis (from the archived hard-cat smoke, 4 videos, trig=1.1)
The smoke DID fire the mechanism genuinely (contradicting the orchestrator's marker_count=0 — see "Marker reconciliation"). All 6 `[CBT]` fires at trig=1.1:

| video | cat | det | pre_dur | post_dur | span move | cap | in-span contrast |
|---|---|---|---|---|---|---|---|
| Explosion002 | Explosion | 1 | 25.0 | 9.0 | 51.0-76.0 -> 52.5-61.5 | 13.0 | **0.6203** (healthy) |
| RoadAccidents123 | RoadAccidents | 0 | 29.0 | 6.0 | 1.5-30.5 -> 13.0-19.0 | 6.0 | **0.0155** (near-flat) |
| Shoplifting001 | Shoplifting | 0 | 23.5 | 2.0 | 12.0-35.5 -> 33.5-35.5 | 10.0 | **0.0046** (near-flat, clamped to min_segment tail) |
| Shooting032 | Shooting | 0/1/2 | 46/40/40 | 2/3/3 | -> 297-299 / 269-272 | 14.0 | 0.42 / 0.36 |

### Two diagnostic findings that reshape the hypothesis
1. **The smoke fires were on ALREADY-eligible wide dets, not the target 6-9s band.** Every fired det (23.5-46s wide) exceeds even `1.5*cap` (9-21s), so it would fire at mult=1.5 too. None of the 4 smoke videos contained a RoadAccidents det in the 6-9s "newly-eligible" band the proposal targeted. **Sentinel-b (a fire at 1.1 that is ABSENT at 1.5) was therefore NOT actually demonstrated on a target det.** The proposal's premise "RoadAccidents dets floor at 6-9s" is unconfirmed on this sample (RoadAccidents123 was 29s wide).
2. **RoadAccidents & Shoplifting in-span category-curve contrast is near-zero (0.0046-0.0155).** The category relative-score curve is nearly FLAT inside these spans, so CBT/CBT-T's "in-span peak" is noise, not a real event prominence — the clamp lands on an essentially arbitrary sub-window (RoadAccidents 29s -> 13-19s; Shoplifting 23.5s -> 2s tail). This is a signal-quality ceiling: for the two SUBTLE/flat-curve categories, tightening the trigger just clamps harder onto noise. Explosion/Shooting have healthy contrast (0.36-0.62) and clamp meaningfully.

**Implication for next loop:** the binding problem for RoadAccidents/Shoplifting is CATEGORY-CURVE QUALITY (flat in-span signal), not the trigger multiplier. A trigger recalibration cannot manufacture a peak where the curve is flat. The proven-safe lever for these two is upstream CLIP query / prompt_weights quality (v94 lineage) to sharpen the category curve, NOT further boundary-gate tuning.

## Marker reconciliation (why REVERT despite genuine fires)
The orchestrator's smoke-gate reported `marker_count=0` and "no [CBT] per-fire clamp lines" on a run whose note lists an **Abuse** video and only 1 `[CBT-VID]` line — i.e. a different/earlier smoke than the archived `batch_eval_v157_smoke` (correct 4 hard cats, 4 `[CBT-VID]`, 6 `[CBT]` fires, all carrying `trig=1.1`). The exact sentinel string the harness grepped did not appear in the run it gated on, tripping FM5/FM17 and blocking the 88-video launch. So the REVERT was triggered by a **smoke-harness / sentinel-string measurement gap, not a proven-inert mechanism**. The mechanism is demonstrably live; it was simply never measured on the 88-harness. This is the SECOND consecutive loop (cf v156 CPA-G) lost to a marker/activation-gate failure rather than a mAP verdict.

## FM table
- **FM1** (boundary edits regress): UNTESTED on 88. Real risk elevated by finding #2 — near-flat-contrast clamps (RoadAccidents/Shoplifting) place windows on noise; the contrast self-guard cannot protect when contrast itself is ~0.005. Explosion/Shooting long-tail-GT risk also untested.
- **FM2** (count->FP): immune, count-neutral by construction (confidence 0.7749->0.7749 etc frozen).
- **FM5** (never-activated): the BINDING failure — orchestrator sentinel_count=0 blocked the eval, though archived smoke shows real fires. LESSON: the smoke gate must grep the SAME run.log / SAME category set / SAME exact sentinel string that the eval will use; a category-set mismatch (Abuse vs the 4 hard cats) silently voids the gate.
- **FM6** (offline->transfer): terminal post-BUE placement preserved (the slot that made CBT survive).
- **FM8** (one change): held — one scalar.
- **FM9** (MPS): reuses CBT's existing `.detach().cpu().numpy()` path.
- **FM13** (bound ceiling first): analytic estimate +0.017-0.025 stands but is now undercut by finding #1 (target band absent) and #2 (flat curve).
- **FM16** (noise floor +/-0.02): moot — no 88-video measurement taken.
- **FM17** (empty-video marker): `trig=` added to all `[CBT-VID]` sites incl 0-det paths; verified firing in the archived smoke on all 4 videos, but the orchestrator gate keyed on a different string.
- **FM18** (absolute mAP): target was 0.09970 -> ~0.117; unmeasured.

## Category lever status: **OPEN** (not closed)
- The trigger-recalibration hypothesis was NEVER measured on the 88-harness. It is not refuted.
- BUT the smoke diagnostics weaken it: (a) the target RoadAccidents 6-9s band did not appear; (b) RoadAccidents/Shoplifting in-span contrast is ~0.005-0.015 (flat), so the clamp cannot reliably improve IoU for the two subtle cats regardless of trigger.
- Explosion/Shooting clamp with healthy contrast — if any category benefits from a laxer trigger it is these two, not the RoadAccidents beneficiary the proposal named. This inverts the proposal's targeting.

## Next-loop guidance
1. **Do NOT re-run CBT-T as-is** without first fixing the smoke gate: assert `grep -c 'trig=1.1'` == videos AND a REAL-MOVE fire on a target-band det that is ABSENT at 1.5, on the EXACT 4-hard-cat run.log the eval uses. (Same lesson v156 recorded: smoke must assert nonzero REAL new-move count on the TARGET population, not marker presence.)
2. **If re-attempting CBT-T:** gate the clamp on in-span contrast >= a floor (e.g. skip clamp when contrast < ~0.05) so flat-curve RoadAccidents/Shoplifting dets are NOT clamped onto noise — otherwise a laxer trigger actively harms the two subtle cats. This makes CBT-T effectively an Explosion/Shooting lever.
3. **RoadAccidents/Shoplifting true lever = category-curve QUALITY** (upstream): sharpen the CLIP text query / prompt_weights for these two so the in-span curve has real prominence; boundary gates are downstream of a signal that is currently flat. This is the endorsed "category-specific score/query quality" lever and is the higher-EV next move.
4. A larger smoke (>=2 RoadAccidents videos with 6-9s post-BUE dets) is needed to even confirm the target band exists before spending an 88-eval.

## Rollback (done)
`--cbt_trigger_mult 1.5` (or omit) is byte-identical to committed v155. Full revert executed: restored pipeline.py/config.py/run.py/batch_evaluate.py from `.qoder/backups/v157/`, hand-reverted the additive `pre_cbtT_boundaries` field from models.py. Verified: `diff -q` vs backup silent for all 4; `grep -c 'cbt_trigger_mult|pre_cbtT|CBT-T'` = 0 across all 5 files; py_compile OK; ScoringConfig() -> bue_enabled=True, cex_extend=False, cbt_enabled=False, no cbt_trigger_mult attr. Committed tree undisturbed.

## Artifacts
- `iteration_logs/results/v157_smoke_eval.json`, `iteration_logs/results/v157_smoke_run.log` (4-video hard-cat smoke; full-88 + bootstrap never produced).
- Normal-recognition design memo: `iteration_logs/ideas/normal_recognition_schemes_20260802.md` (unchanged, carried from v155/v156).
