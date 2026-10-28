# Loop 4 — v117: FDR (Feature-space Deviance Re-rank) single-flag offline bake-off

**Date:** 2026-07-25
**Decision:** REVERT (never launched — offline bake-off winner undetermined; FM5 auto-REVERT)
**Kept:** false

---

## Approved Proposal (summary)

**Title:** FDR v117 — Feature-space Deviance Re-rank via a single-flag offline bake-off: implement TWO orthogonal per-detection CLIP-feature re-rank arms (SLM text-margin + TFO visual self-deviance), adjudicate DETERMINISTICALLY offline on v114's real detections, and promote EXACTLY ONE arm to the live A/B (single controlled change).

**Mechanism.** The dominant residual error (FM7 intra-video mislocalization) is invisible to every committed re-rank because `frame_scorer.py` L106-110 overwrites the in-window fused curve with `np.maximum(base_signal, a['confidence'])` — the VLM confidence FLOODS each detection window, so Q-score geometry (v111) and prominence (v115) read a shape that is an artifact of the confidence itself. Both arms independently converged on the same fix: read the UNCONTAMINATED pre-overlay CLIP feature space (`video_data['clip_features']`, unit-norm, 1:1 timestamp-aligned, computed at `pipeline.py` L917) that NO current re-rank touches. Each arm is a read-only, post-hoc, per-detection multiplicative gate stacked immediately AFTER the committed Q-score block, using the EXACT v111 protocol (within-video min-max normalize → n∈[0,1]; ≤1 detection → inert; conf' = min(1, conf·(base + (1-base)·n)), base=0.5; boundaries/count/order/NMS/fusion/half-max/categories byte-identical — only global confidence ordering, hence mAP, moves):

- **ARM-SLM (category-SEMANTIC text margin):** F = clip_features (N×D), T = anomaly_expert.text_features (K+1×D), Normal prototype = T[-1]. Per final detection with resolved category c: margin(t) = F[t]·T[c] − F[t]·T[-1] (Normal-subtracted). slm_raw = mean(margin[in]) − mean(margin[out]) (AutoLoc Outer-Inner-Contrast). Unknown/Normal/absent/empty in-mask → worst-value sentinel (benign demotion).
- **ARM-TFO (category-AGNOSTIC visual self-deviance):** per video once, bg = per-dimension MEDIAN of all frame features, L2-normalized. d_t = 1 − cos(F[t], bg). tfo_raw = mean(d[in]) (absolute within-window deviance — deliberately not a contrast form, to stay orthogonal to Q-score's peak_contrast). RTFM "anomaly = feature outlier" recast ANGULARLY (CLIP features unit-norm; magnitude dead — the reason v110 VideoMAE-magnitude saturated).

**Single controlled change:** the deterministic offline bake-off re-scores v114's OWN real final detections through each arm independently; EXACTLY ONE arm (the winner) is activated in the single live A/B. The loser ships inert flag-off.

**Pre-registered adjudication:** live candidate = arm that (a) clears offline stack delta ≥ +0.005 over the 0.15632 Q-on anchor AND (b) rises broad-based across AP@0.3/0.5/0.7 AND (c) is least Q-correlated of qualifiers. If both qualify → higher, more Q-orthogonal one; loser deferred. If NEITHER clears +0.005 → both ship flag-off and loop REVERTS to v114.

**Expected gain:** each arm hypothesized +0.004 to +0.010 mAP; honest floor +0.002-0.004 (sub-threshold, revert-realistic). Latency ~zero (matmuls/median on cached arrays, no new forward pass; stays in v114 128-166s band).

---

## Reviewer C Feedback (summary)

APPROVED. Every load-bearing claim code-verified in the current tree: (1) causal thesis is real — `frame_scorer.py` L106-110 literally overwrites each window with `np.maximum(base_signal, a['confidence'])`; text_features[-1] IS the Normal prototype; clip_features unit-norm at pipeline L917, upstream of all re-ranks → both arms read-only on cached arrays, zero new forward pass. (2) FM coverage complete: FM1/FM2 immune by construction; FM5 addressed at its real v116 root cause (MPS→numpy hop before marker) via unconditional `_to_np` + real-device smoke + per-video markers + log-grep gate; FM6 does not apply (deterministic re-score of v114's OWN real detections, verified cache at results/batch_eval_v114_full, 66 JSONs); FM7 per-detection variance kill-gate; FM8 respected (exactly one new live flag, never bundled with FCCS); TFO is angular (1-cos), NOT v110 magnitude. (3) two-arm structure not a confound — offline re-score is faithful in-vivo, one-change-per-loop binds only the LIVE run. Honest caveat: admitted floor (+0.002-0.004) sits BELOW the +0.005 bar; the two comparable re-rank loops (v115 prominence +0.0026, v107 ECE +0.002) both reverted — a revert is a realistic outcome, but the offline oracle is cheap and pre-registers it, so it is the right way to spend zero live budget. Most productive family available.

---

## Implementation Notes (D1)

v117 FDR fully implemented and verified in tree:
- **config.py** (L101-104, threaded from_args L205-208 via getattr defaults): slm_rerank/tfo_rerank bool=False, slm_base/tfo_base float=0.5. Default OFF = byte-identical v114. qscore_rerank stays committed baseline.
- **pipeline.py:** `_fdr_to_np` helper (unconditional `.detach().cpu().numpy()` — root-cause fix for v116 MPS→numpy crash); `_slm_rerank` (category-semantic margin, AutoLoc OIC, worst-value sentinel); `_tfo_rerank` (per-video L2-normed per-dim MEDIAN prototype, angular d=1−cos, mean(d[in])). Both read-only on `video_data['clip_features']`, mirror `_qscore_rerank` structure exactly. Two independently-gated call sites in detect() immediately after Q-score block.
- **CLI** (batch_evaluate.py, run.py): --slm_rerank/--tfo_rerank (store_true), --slm_base/--tfo_base (float default 0.5). Verified in --help.
- **FM5 guard:** unique `[SLM-GATE]`/`[TFO-GATE] reranked/skipped` markers on every path.
- **Offline harness** (analysis_v117_fdr_precheck.py, NOT run): clones v116, adds TFO arm, re-extracts clip_features via real AnomalyExpert.precompute_video_features with `_to_np` fix, re-scores v114 real cached detections (results/batch_eval_v114_full). Implements C's tightenings: hard fidelity abort-gate (reproduce raw baseline AND Q-only anchor to |Δ|≤0.0005 of 0.15632), quantified orthogonality (per-detection Pearson vs Q, reject if |corr|>0.6 or stack fails to beat both Q-only AND arm-only), pre-registered adjudication.
- All 5 files pass ast.parse; `import pipeline` OK. Backups at .qoder/backups/v117/. NO EVAL RUN.

---

## Eval Flags + Activation Marker

- **Live flag (intended):** `--qscore_rerank --slm_rerank` (ARM-SLM) OR `--qscore_rerank --tfo_rerank` (ARM-TFO) — EXACTLY ONE new flag alongside the committed baseline; NEVER both new flags together live.
- **Activation marker:** `[SLM-GATE] reranked` (if SLM wins) OR `[TFO-GATE] reranked` (if TFO wins).
- **Offline harness** analysis_v117_fdr_precheck.py needs NO live flag.

## Marker Counts

| log | SLM-GATE reranked | TFO-GATE reranked |
|---|---|---|
| smoke | N/A (never launched) | N/A (never launched) |
| full  | N/A (never launched) | N/A (never launched) |

0 markers → FM5 auto-REVERT.

---

## Results Table

| metric | value |
|---|---|
| smoke mAP | -1 (not run) |
| full mAP | -1 (not run) |
| AP@0.3 | -1 (not run) |
| AP@0.5 | -1 (not run) |
| AP@0.7 | -1 (not run) |
| full avg latency | -1s (not run) |
| offline map_off / map_on / delta | N/A — harness (analysis_v117_fdr_precheck.py) never run; analysis_v117_fdr_precheck.json MISSING |

---

## Keep/Revert Decision + Reason

**REVERT.** The single live A/B must activate EXACTLY ONE new flag — the offline bake-off winner (--slm_rerank OR --tfo_rerank, never both). But the offline harness `analysis_v117_fdr_precheck.py` was never run over v114's 65 real detections, so its output `analysis_v117_fdr_precheck.json` is MISSING and the winner is UNDETERMINED. The pre-registered verdict could even be "neither arm clears +0.005 → ship flag-off / revert to v114." Launching with a blindly-guessed flag would risk activating the loser arm or an unwarranted run — a violation of the one-change protocol. With no winner, no v117 smoke/full eval was launched; no report/log exists; 0 activation markers → FM5 auto-REVERT. Code restored from .qoder/backups/v117/ (the 4 edited files) and verified byte-identical via diff: config.py/pipeline.py/batch_evaluate.py/run.py no longer contain any slm_rerank/tfo_rerank/_slm_rerank/_tfo_rerank/_fdr_to_np/SLM-GATE/TFO-GATE symbols; committed qscore_rerank baseline intact; `import pipeline` OK. Baseline stays v114 (map=0.15632).

---

## Lessons Learned

- **FM5 (non-activation) — CONFIRMED again, upstream this time.** v116 died at an MPS→numpy crash *after* launch; v117 never even reached launch because the decisive offline gate (the bake-off) that determines WHICH flag is legal to activate was never executed. Lesson: a single-flag bake-off design is only as live-ready as its adjudicating harness — the harness must run and emit its JSON verdict BEFORE the finalizer stage, or the live run is un-launchable by construction and auto-reverts.
- **FM6 (sim-to-live gap) — NOT REACHED.** The offline oracle was never run, so the deterministic re-score fidelity (anchor reproduction of 0.15632) and the arm deltas remain unmeasured. The orthogonality thesis (both arms read the uncontaminated clip_features vs the flooded fused curve) stays untested in vivo — genuinely unmeasured, not refuted.
- **FM1/FM2 — immune by construction** (boundaries/count/order/fusion frozen; only global confidence reorders) — never at risk; not exercised this loop.
- **Process lesson:** the code and harness were both fully implemented and C-approved, but the loop stalled at the harness-execution step. The mechanism ships inert (reverted from tree); ARM-SLM (v116-approved, twice-unmeasured) and ARM-TFO (fresh axis) remain viable candidates for a future loop that actually runs analysis_v117_fdr_precheck.py first.
