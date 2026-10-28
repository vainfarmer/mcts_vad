# v118 — FSL: Feature-space Structural Localization Re-rank

Loop 5 | Version v118 | Date 2026-07-25 | Decision: **REVERT**

## Approved proposal (summary)
One new read-only, post-hoc, per-detection multiplicative confidence gate stacked
immediately AFTER the committed v111 Q-score block in `pipeline.detect()`. It reads the
UNCONTAMINATED pre-overlay `video_data['clip_features']` (unit-norm N×D, cached at
pipeline L917) — NOT the fused output curve that `frame_scorer.py` L106-110 floods via
`np.maximum(base_signal, a['confidence'])`. Angular cosine only (magnitude is dead on
unit-norm CLIP → sidesteps the v110 VideoMAE saturation).

Per final detection window it scores two orthogonal localization-quality axes:
- **Boundary-discontinuity** (sbd/ebd): margin `m = min(|I|//2, gap-to-edge)` frames just
  inside vs. equal margin just outside each boundary — does the edge land on a real content
  transition?
- **Segment-separability** (sep = cohesion − cross, `cohesion = ||c_in||²`,
  `cross = c_in·c_out`): is the window a visually distinct block vs the rest of the video?
  Note FBC's internal-homogeneity term == SEP's cohesion, so it is never double-counted.

THREE ARMS scored in the OFFLINE harness only:
- `arm_fbc = (sbd + ebd + cohesion)/3`
- `arm_sep = cohesion − cross`
- `arm_comp = mean( within-video-norm((sbd+ebd)/2), within-video-norm(sep) )`

Each arm follows the EXACT v111 protocol: within-video min-max → n∈[0,1] (≤1 detection → n=1,
inert), `conf' = min(1, conf·(base + (1-base)·n))`, base=0.5. Degenerate guards → worst-value
sentinel + skip-marker. The offline bake-off was to promote EXACTLY ONE arm to a single live
A/B stacked on the committed v111 Q-score.

**Full proposal JSON is preserved in the orchestration record; mechanism/rationale/risks
verbatim above.**

## C's feedback (APPROVED)
Verified load-bearing claims against the live tree: (1) `frame_scorer.py` L106-111 confirms the
fused curve is flooded via `np.maximum(base_signal, a['confidence'])` inside each detection
window, so committed Q-score/prominence are genuinely blind to raw-feature localization
structure — orthogonality is structural, not wishful. (2) `_qscore_rerank` reads flooded
`frame_scores` (1st-order); FSL reads raw `clip_features` (2nd-order pairwise) — a real distinct
data source. (3) v111 min-max/single-detection-inert protocol reproduced verbatim. (4) control
MUST pass `--qscore_rerank` explicitly (from_args uses getattr default False). Why it clears the
bar: pure post-hoc read-only re-rank (FM1/FM2 immune), per-detection with real within-video
variance (FM7-on-target), angular-cosine-only (sidesteps v110 saturation), raw-feature so not
baked into boundary placement (avoids v112/v110 zero-variance trap), FM8-clean. Residual
concern: offline CLIP re-extraction byte-matching live L917 (fidelity abort-gate handles it);
expected gain honestly revert-realistic (floor +0.002, cf. v107/v115 reverts).

## Implementation notes (D1)
- `config.py`: ScoringConfig gains `fsl_rerank:bool=False`, `fsl_mode:str="comp"`,
  `fsl_base:float=0.5`; threaded in from_args via getattr. Default OFF = byte-identical v114.
- `pipeline.py`: module-level `_to_np(x)` (.detach().cpu().numpy() guard, v116 MPS crash fix);
  staticmethod `_fsl_within_norm`; method `_fsl_rerank(final_anomalies, video_data, timestamps,
  mode, base)` after `_qscore_rerank`; call site in detect() immediately AFTER the qscore block,
  gated on `self.config.scoring.fsl_rerank`.
- `batch_evaluate.py` + `run.py`: `--fsl_rerank` (store_true), `--fsl_mode {fbc,sep,comp}`
  default comp, `--fsl_base` float default 0.5.
- Harness `iteration_logs/analysis_v118_fsl_precheck.py`: clone/extend of v111; re-extracts CLIP
  features per v114 video (same path as pipeline L917), caches to `analysis_v118_fsl_feats.pkl`;
  fidelity gate + orthogonality (Pearson vs Q and TSM) + within-video-variance + broad-AP +
  beats-both gates; pre-registered verdict picks single arm clearing all sub-gates.
- Backups: `.qoder/backups/v118/` (config.py, pipeline.py, batch_evaluate.py, run.py).

## Eval flags + activation marker
- Flags used: `--qscore_rerank --fsl_rerank --fsl_mode comp`
- Activation marker: `[FSL-GATE] reranked`
- Marker counts: **smoke = 12**, **full = 61** (guard satisfied / active in both).

## Results

| metric | value |
|---|---|
| smoke mAP | 0.08953168044077135 |
| full mAP | 0.15663679184480417 |
| AP@0.3 | 0.21869542886492035 |
| AP@0.5 | 0.14584304584304586 |
| AP@0.7 | 0.10537190082644629 |
| full avg latency | 672.13 s/video |

Reference anchors: v114 raw mAP = 0.15013; v111 Q-on anchor = 0.15632. Live full mAP 0.15664
vs the Q-on control ≈ +0.0003 — well below the pre-registered +0.005 promotion bar. No
`analysis_v118_fsl_precheck.json` was persisted in the tree at finalize time.

## Decision: REVERT
Reason: full mAP 0.15664 is only ~+0.0003 over the committed v111 Q-on baseline (0.15632),
far below the pre-registered +0.005 offline/live promotion gate. The proposal explicitly
pre-registered a flag-off revert as an acceptable outcome when no arm clears the bar; the FSL
`comp` arm did not move mAP meaningfully. All 4 edited files restored from
`.qoder/backups/v118/` and diff-verified identical; restored config has zero `fsl_rerank`
occurrences (byte-identical v114 baseline behavior). Latency also ballooned to 672 s/video
(outside the v114 128–166 s band) in this run — an additional reason not to keep, though the
mechanism itself adds ~zero compute (the latency is run-environment, not FSL arithmetic).

## Lessons learned
- **FM7 (mislocalization) confirmed hard**: a structurally-clean, orthogonal, raw-feature
  per-detection re-rank that reads the uncontaminated clip_features (not the flooded curve)
  still produced only a negligible +0.0003 mAP move. The dominant residual error resists even
  well-targeted post-hoc confidence re-ordering — the localization signal in raw CLIP structure
  is real but too weak to shift global mAP once boundaries/count are frozen.
- **FM1/FM2/FM5 refuted as risks**: read-only design kept boundaries/count/fusion byte-identical;
  the `[FSL-GATE] reranked` marker fired (12 smoke / 61 full), so activation was never in doubt —
  the null result is a genuine effect-size null, not a non-activation artifact.
- **Post-hoc re-rank family is near-tapped**: after v95 TSM and v111 Q-score (both positive),
  v107 ECE, v115 prominence, and now v118 FSL all landed sub-threshold and reverted. The cheap
  read-only re-rank lever appears to have diminishing returns; future gains likely need to move
  boundary placement or the detection stage itself, not just confidence ordering.
