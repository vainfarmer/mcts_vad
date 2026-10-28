# Loop 10 — v123 — SLM-CV (Semantic Localization Margin, Cross-Video calibrated)

**Date:** 2026-07-25
**Decision:** REVERT (never legitimately launched — FM5 auto-revert / launch-gate refusal)
**Baseline retained:** v114 committed Q-on (full mAP 0.15632 / 0.15013 this cache set)

---

## Approved Proposal

**Title:** SLM-CV — Semantic Localization Margin, Cross-Video calibrated: a device-safe, harness-first per-detection anomaly-minus-Normal CLIP text-margin OIC re-rank whose within-video-vs-cross-video squash is adjudicated offline, stacked on the committed v111 Q-score (single controlled change).

**Mechanism:** One new read-only, post-hoc, per-detection MULTIPLICATIVE confidence re-rank `_slm_rerank(self, final_anomalies, video_data, base, mode)` inserted in `Pipeline.detect()` IMMEDIATELY AFTER the committed v111 Q-score block (pipeline.py:1218-1221) and BEFORE `DetectionResult(...)` (pipeline.py:1223), stacked on Q-score. Reads already-cached pre-overlay `video_data['clip_features']` (unit-norm) and `self.anomaly_expert.text_features` (Normal prototype = LAST row). Per FINAL detection with resolved category index c: per-frame cosine sims `sim = clip_features @ text_features.T`, anomaly-minus-Normal margin `m_t = sim[:,c] - sim[:,-1]`, squashed parameter-free `p_t = sigmoid(logit_scale * m_t)` (model's OWN existing constant, reused). OIC window contrast `slm_raw = mean(p[in]) - mean(p[out])` with the IDENTICAL in/out masking as `_qscore_rerank`. Single architectural DoF, adjudicated offline as exactly-one-of-two arms: (A) WITHIN_VIDEO min-max normalize (the honest control that finally measures the never-run v116); (B) CROSS_VIDEO `s = clip(slm_raw, 0, 1)` directly (the ceiling bet — no within-video normalization, so demotion is comparable ACROSS videos). Then `conf' = min(1.0, conf * (base + (1-base)*s))`, base=0.5. All tensor->numpy via `.detach().cpu().numpy()` (fixes the v116 MPS crash). Boundaries, count, ordering-inputs, NMS, merge, half-max, fused curve, and Q-score byte-identical; only mutation is each final detection's scalar confidence, applied AFTER count is frozen.

**Rationale:** Structural headroom is the frame-AUC(0.70) vs temporal-mAP(0.15) gap. Under UCF-Crime temporal mAP, detections are pooled and ranked GLOBALLY per category. Every prior post-hoc re-rank (v111 Q-score, v115 PROM, v118 FSL, v119 MELC) used WITHIN-VIDEO min-max normalization, which forces each video's best detection (incl. normal-only and mislocated windows) to s=1, leaving cross-video TP-vs-FP ordering equal to the raw VLM confidence — the precise reason the family looks tapped. SLM-CV arm B is the first re-rank injecting a per-detection signal on an ABSOLUTE cross-video scale (fixed CLIP embedding anomaly-minus-Normal margin, squashed by the model's own logit_scale), directly attacking the AUC-mAP gap. Normal-subtraction cancels common-mode CATEGORY DIFFICULTY (protects low-CLIP anomalies Shoplifting/Stealing/Arson/Burglary). OIC operator additionally demotes intra-video mislocated windows (FM7). Orthogonality to Q-score is code-confirmed (frame_scorer.py:110 np.maximum floods the in-window fused curve, erasing raw CLIP micro-structure that Q-score reads; SLM reads untouched pre-overlay feature space); gated at |Pearson|<0.6.

**Expected gain:** Arm A +0.001..+0.003 (matches tapped family, the control). Arm B +0.005..+0.015 mAP concentrated in AP@0.5/@0.7 IF cross-video FP demotion is real. Floor near zero (multiplicative gate bounded [0.5,1.0], inert guards, offline adjudication before live budget).

---

## C's Feedback (APPROVED)

Verified all load-bearing code claims against the tree: Normal prototype = last row of text_features (anomaly_expert.py:227); frame_scorer.py:110 np.maximum flooding confirms Q-score reads flooded curve while raw clip_features stay untouched (orthogonality basis real, not asserted); Q-score block at 1218-1221 with DetectionResult at 1223 (insert point correct); `_qscore_rerank` in/out masking is exactly the template SLM mirrors and runs post-count-freeze (FM1/FM2 immune by construction); clip_features cached at pipeline.py:917; `logit_scale = model.logit_scale.exp().detach().clamp(max=100.0)` exists and is reused (no new numeric).

Why it clears the bar despite the tapped family: NOT a fifth within-video re-rank. Arm B's cross-video calibration attacks the precise structural reason v107/v115/v118/v119 stalled. FM audit: FM5 clean (real flag + unique [SLM-CV] marker on fired AND every guard path AND both detect() early returns + post-eval grep + launch gate requiring precheck JSON with promoted==true and promoted_arm==--slm_mode — the correct mechanical fix for recurring v117/v120/v121 harness-skip). FM6 clean (deterministic offline A/B on v114's REAL final detections). FM7 clean (strictly per-detection OIC). FM8 clean (dropped fitted tau — uses model's own logit_scale + parameter-free clip). FM1/FM2 immune. v116 MPS crash fixed by unconditional .detach().cpu().numpy(). Ship flag-off if harness doesn't clear +0.005.

---

## Implementation Notes (D1)

Implemented v123 SLM-CV faithfully with all of C's mandated changes; built from scratch. Backups in `.qoder/backups/v123/`.
- **config.py:** ScoringConfig gains `slm_rerank`(False), `slm_mode`('cross_video'), `slm_base`(0.5); threaded in `from_args` via getattr defaults (flag-off byte-identical to v114/Q-on). No fitted tau field (FM8-clean).
- **pipeline.py:** `_slm_rerank(...)` modeled on `_qscore_rerank`. Reads clip_features + text_features (Normal=last row), `.detach().cpu().numpy()` unconditionally, reuses model.logit_scale. `m_t=sim[:,c]-sim[:,-1]` via `categories.index(anomaly_type)`; stable-sigmoid; `slm_raw=mean(p[in])-mean(p[out])`; arm A min-max, arm B clip(0,1); `conf'=min(1,conf*(base+(1-base)*s))`. Inert s=1.0 skip marker on no-detections/None features/model None/unknown label/misalignment. Call AFTER Q-score, BEFORE DetectionResult, guarded. Both detect() early returns emit guarded [SLM-CV] skip markers. Docstring bumped to v123.
- **batch_evaluate.py + run.py:** added `--slm_rerank`, `--slm_mode {within_video,cross_video}`, `--slm_base` (default 0.5). batch_evaluate adds `check_slm_activation_gate` (post-eval grep, count>=succeeded else sys.exit(2)) and `check_slm_launch_gate` (pre-launch sys.exit(2) unless precheck JSON exists AND promoted==true AND promoted_arm==--slm_mode).
- **analysis_v123_slmcv_precheck.py:** deterministic offline A/B on v114 REAL detections; re-extracts CLIP features on real MPS; gates: fidelity anchor 0.15632, orthogonality |r|<0.6, within-video variance kill-gate, saturation kill-gate, low-CLIP per-category non-regression, deterministic mAP A/B; promotes exactly one arm.

---

## Eval Flags + Activation Marker

- **Treatment CLI:** `--qscore_rerank --slm_rerank --slm_mode cross_video` (ceiling-bet arm; substitute promoted_arm from precheck JSON)
- **Control CLI:** `--qscore_rerank` only
- **Activation marker:** `[SLM-CV]`
- **NEVER** co-activate `--full_coverage_scan` (FM8)

## Marker Counts

- **smoke:** 0 `[SLM-CV] reranked` markers — eval never ran. Log shows only `[SLM-CV] LAUNCH: refusing — precheck JSON missing`.
- **full:** skipped (gate).

---

## Results Table

| metric | value |
|---|---|
| smoke mAP | -1 (marker_count = -1; launch refused) |
| full mAP | -1 |
| AP@0.3 | -1 |
| AP@0.5 | -1 |
| AP@0.7 | -1 |
| full avg latency (s/video) | -1 |
| deterministic A/B (map_off/map_on/delta) | not run (precheck harness never completed) |

The v123 smoke eval could NOT be launched with the required flags. The launch gate (correctly, per design) refused because `analysis_v123_slmcv_precheck.json` does not exist. The precheck script is a heavy offline job (>8.5 min, exceeds per-call budget; killed at 143, no JSON produced). The required flags were NOT dropped and the JSON was NOT fabricated. No batch_evaluate ran; full skipped.

---

## Decision + Reason

**REVERT.** v123 was never legitimately launched. The mechanism is implemented and C-approved, but its own launch gate physically blocked the live smoke because the mandatory STEP-0 offline harness (`analysis_v123_slmcv_precheck.py`) never ran to completion within the per-call budget, so no arm was promoted and no `analysis_v123_slmcv_precheck.json` exists. With zero `[SLM-CV]` activation markers this is an FM5 auto-revert. Per proposal design, the mechanism ships flag-off byte-identical to v114. The 4 backup files (batch_evaluate.py, config.py, pipeline.py, run.py) were restored from `.qoder/backups/v123/` and diff-verified byte-identical; the v123 precheck script was removed and the orphaned background precheck process (PID 86874) was killed. Restored config carries zero `slm` symbols; all 4 files AST-parse.

---

## Lessons Learned (failure modes)

- **FM5 CONFIRMED (harness-skip, 3rd repeat — v117/v120/v121 lineage):** the recurring failure is not the mechanism but the operational cost of the STEP-0 offline harness. The v123 launch gate did its job — it refused a live launch lacking a legitimately-promoted arm, wasting ZERO eval budget (like v121/v122, unlike v120's wasted full run). But the harness itself is a >8.5-min job that does not fit the per-call budget, so it must be run to completion (background, polled across turns) BEFORE the finalize loop, not inside it. Any future harness-gated mechanism MUST budget a dedicated completed harness run as an explicit upstream step.
- **FM1/FM2 immune by construction** (post-count-freeze read-only re-rank) — not reached.
- **FM7 (mislocation) / cross-video calibration hypothesis: UN-adjudicated.** The genuinely-new ceiling-bet lever (absolute cross-video anomaly-minus-Normal OIC margin) was never measured — neither refuted nor confirmed. It remains the strongest un-tested hypothesis against the frame-AUC/temporal-mAP gap and should be re-run once the precheck JSON exists.
- **v116 MPS crash fix VALIDATED at code level** (unconditional .detach().cpu().numpy(), AST-parses, imports) but not at runtime since eval never ran.
- **Process discipline held:** flags not dropped, JSON not fabricated — an honest true-negative launch at zero live cost, consistent with v121/v122.
