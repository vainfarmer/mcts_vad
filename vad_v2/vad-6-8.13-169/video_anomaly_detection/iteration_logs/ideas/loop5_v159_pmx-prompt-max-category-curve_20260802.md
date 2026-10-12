# v159 — PMX: Prompt-Max Category Curve (brief hard cats) — VERDICT: REVERT (floor breached)

Loop 5 of session v12 (map20). Hard harness = 88 videos (Shooting/RoadAccidents/Shoplifting/Explosion).
Committed baseline this session = v155 CBT-on hard mAP **0.09970**; do-not-regress floor **0.09470**.

## Mechanism (one change)
Replace the single collapsed ensemble-vector CLIP category logit with a **per-frame MAX-over-the-6-prompt-embeddings** logit, injected into the category column of the `similarity` tensor BEFORE the existing softmax + `cat/(cat+normal)` ratio, for the 3 BRIEF hard cats only {Shooting, Explosion, RoadAccidents}. Shoplifting deliberately excluded (max-pool of a diffuse weak signal only lifts its noise floor). The sharpened curve feeds candidate seeding / VLM scan targeting AND `_clip_boundary_detection`.

- Root cause confirmed at `anomaly_expert.py:207-224`: every category curve driven by ONE collapsed text vector (v93 learned-weight or mean of the 6 prompts). For a brief multi-modal event this dilutes the one discriminative prompt at the true frame.
- Implementation: unused `text_features_perprompt` dict on the expert (3 cats), PMX branch in `frame_scorer.get_category_scores`, `pmx_enabled` flag (default OFF) + `--pmx_extension` in run.py/batch_evaluate.py. Flag-off byte-identical to committed. Markers PMX-FIRE (core) + PMX-VID (all 3 return paths incl 0-det).
- Banned-family: NOT banned-1 (changes the CLIP TEXT QUERY, not a re-rank of the same curve); NOT banned-3 (scan budget unchanged); NOT banned-4 (max over prompt axis per frame, no per-video rescale, Normal column identical). This is the admissible category-specific CLIP-query / score-quality lever.

## Activation (FM5/FM17 — clean this loop)
Prior 3 loops (v156/v157/v158) all died on activation-gate/instrument failures. v159 fixed the sentinel (bracket-free fixed-string tokens): smoke PMX-FIRE=3 (>0, core max-pool branch fired — inert-core trap ABSENT), PMX-VID=4 (==videos). Full 88: PMX-FIRE=60, PMX-VID=88 (all paths incl 0-det, no FM17 deadlock). **The mechanism genuinely fired and changed the detection set — this is a real measured mAP verdict, not another activation void.**

## HEADLINE — hard-harness bootstrap (cross-run vs fresh v155 REF this session)
- Smoke (4 vid): mAP 0.13333 (AP@0.3 0.218 / AP@0.5 0.091 / AP@0.7 0.091).
- **Full 88: mAP 0.06450** (AP@0.3 0.16193 / AP@0.5 0.02507 / AP@0.7 0.00649), 99.11 s/video.
- Bootstrap 1000 resamples, n=88 common: **point_delta -0.03520**, 90% CI **[-0.08610, -0.00163]** (entirely below 0), **P(delta>0)=0.044**.
- GATE: ALL FAIL. point -0.0352 < +0.02; P(d>0) 0.044 < 0.9; new mAP 0.06450 < floor 0.09470.

### Per-category AP deltas (v159 vs v155 REF), the diagnosis
| cat | n | v155 mAP | v159 mAP | delta |
|---|---|---|---|---|
| Shooting | 23 | 0.07833 | 0.07916 | +0.00083 |
| Explosion | 21 | 0.16090 | 0.15698 | -0.00392 |
| RoadAccidents | 23 | 0.20180 | 0.07359 | **-0.12820** |
| Shoplifting | 21 | 0.07323 | 0.06169 | -0.01154 (byte-identical target — this is pure VLM run-to-run drift) |

**RoadAccidents drives the entire regression** (-0.128 on 23 videos). Shooting/Explosion (the other 2 targeted brief cats) are flat within noise. Shoplifting was NOT touched by PMX (excluded), so its -0.0115 is unpaired VLM nondeterminism, quantifying the cross-run floor.

## Interpretation — why PMX hurt RoadAccidents
The proposal's per-category arithmetic assumed max-over-prompts SURFACES buried true peaks. For RoadAccidents the opposite happened: max-pooling over 6 prompts **raised the noise floor faster than the signal peak**, injecting spurious high peaks that re-seeded the VLM scan / boundary onto WRONG windows, collapsing the strong v155 RoadAccidents localization (0.202 -> 0.074). This is exactly the diffuse-cat FP-surge failure that Shoplifting was excluded to avoid — but RoadAccidents (nominally "brief") behaves like the subtle cat here: its category curve does not have a clean single discriminative sub-appearance prompt, so MAX picks up whichever prompt happens to fire on background. Consistent with the v157 finding that RoadAccidents in-span category-curve contrast is near-ZERO (flat curve) — max-pooling a flat curve amplifies noise, not signal.

Shooting/Explosion (healthy contrast per v157) were essentially unmoved: max-pool neither helped nor hurt, so even the intended brief-cat lift did NOT materialize — the offline "surface 6-9 missed events" premise did not transfer (Weakness #1 realized: CLIP is largely blind to the event regardless of prompt).

## FM table
- FM1 (curve-shape regress): REALIZED — sharpened curve mis-seeded RoadAccidents.
- FM2 (count/FP): additive path; RoadAccidents FP surge is the concrete manifestation.
- FM5 (activation): CLEAN this loop — PMX-FIRE=60/PMX-VID=88, core fired, no inert-core trap (fixed the v156-v158 gate bug with bracket-free fixed-string grep). This was a REAL verdict.
- FM6 (offline->online transfer): REALIZED — the assumed recall lift on brief cats did not transfer (Shooting/Explosion flat).
- FM8: held (one change).
- FM9 (MPS): .detach() present on pmx_logit path.
- FM13 (ceiling): boundary-only capped at AP@0.3; PMX aimed to lift loose recall but instead lowered it.
- FM16 (noise floor): cross-run half-width ~+/-0.02; point -0.0352 clears the floor on the DOWNSIDE, CI entirely <0 — regression is real, not noise.
- FM17: no deadlock (PMX-VID on 0-det path).
- FM18: absolute hard mAP 0.06450 < floor 0.09470 => REVERT.
- FM19: additive (not count-neutral); PMX-B count-neutral fallback was NOT triggered/wired (smoke did not breach the count-cap; the full eval regressed on set membership, which PMX-B avoids by construction — see next-loop).

## Category lever status: **CLOSED for RoadAccidents (max-pool); OPEN-but-weak for Shooting/Explosion**
- **Max-over-6-prompts category logit is CLOSED as a RoadAccidents lever**: it amplifies a flat/diffuse curve's noise and destroys strong existing localization (-0.128). Do NOT re-propose max-pool on RoadAccidents or Shoplifting (flat in-span contrast, cf v157).
- For Shooting/Explosion max-pool was inert (no gain, no harm) — the discriminative-prompt-dilution hypothesis did not pay off; likely CLIP-blindness, not ensembling dilution (Weakness #1).

## Next-loop guidance
1. **Do NOT re-run PMX max-pool.** It is measured-negative (RoadAccidents) or measured-null (Shooting/Explosion). The full 88-eval verdict is definitive (not an activation void).
2. If revisiting category-signal QUALITY, use the **count-neutral PMX-B** variant (max-pool feeds ONLY `_clip_boundary_detection`, candidate membership frozen, paired via logged pre_pmx_boundary) — it structurally cannot cause the RoadAccidents set-membership collapse that killed PMX-A, and is paired-measurable at ~+/-0.015 precision. But bound its ceiling first (boundary-only ~+0.01, likely inside floor).
3. **RoadAccidents/Shoplifting true residual = upstream category-curve QUALITY via better CLIP text queries in the prompt_weights.json (v94) lineage** — a CALIBRATED-weight ensemble that re-weights the 6 prompts per cat, NOT an un-tuned max. This is the third consecutive loop (v157, v158, v159) to point at query quality for the flat-curve subtle cats. Consider a CoOp-lite / prompt_weights re-fit on train for RoadAccidents specifically, validated offline for peak-prominence BEFORE spending an eval.
4. **v155 CBT remains the committed hard baseline (mAP 0.09970).** No change this session.

Artifacts: iteration_logs/results/v159_{hard_eval.json, hard_run.log, bootstrap_ab.json}. Backups: .qoder/backups/v159/. Normal-schemes memo: ideas/normal_recognition_schemes_20260802.md (unchanged — same (a)/(b)/(c) design).
