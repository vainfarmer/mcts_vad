# Loop 1 — v138 GPC: Global-Peak-Containment confidence prior

**Date:** 2026-07-27
**Session:** accuracy arc resumed (previous session closed the latency arc; v137 banked -7.5% VLM calls).
**Objective:** PRIMARY target full-eval temporal mAP >= 0.20. Reference v124 full mAP 0.17076,
committed baseline 0.15632, accuracy floor 0.15132.
**Status:** implemented, flag `--gpc_rerank` (default OFF), smoke PASSED, full eval running.

---

## 0. The diagnostics that shaped this loop (orchestrator, zero API cost)

Harnesses: `analysis_v138_rankdiag_orch.py`, `analysis_v138_actionness_lib.py`,
`analysis_v138_actionness_precheck.py`, `analysis_v138_peakin_robust.py`.

1. **Pure re-ranking oracle ceiling = 0.27273** (conf := true IoU, detections/boundaries/count
   untouched) on both v130 and v132 caches. Headroom from v130's 0.17708 = **+0.0957**, so the
   0.20 target IS reachable by ranking alone — it needs ~25-30% of oracle skill. Conversely
   conf := const collapses mAP to 0.071/0.081, i.e. the existing formula already supplies ~55%
   of achieved mAP. Ranking is the highest-leverage surface, not a rounding error.
2. **mAP is decided by ~4 numbers.** n_gt=66, 11-point interpolated AP, max recall 0.379 @IoU0.3
   ⇒ only recall points 0.0/0.1/0.2/0.3 are attainable, i.e. precision at the 1st/7th/14th/20th
   hit. Verified exactly: 1.0, 7th hit at rank 9 (0.778), 14th at rank 24 (0.583), 20th at rank 48
   (0.417) ⇒ AP@0.3 = 2.78/11 = 0.2528 — matches the reported 0.25283 to 4 decimals.
3. **The top of the list is already saturated** (top10: 10/10 IoU>0, 7 with IoU>=0.3). All remaining
   gain must come from promoting the buried hits at ranks 35,37,39,45,47,48,50,53,68,82,87 above the
   ~30 misses outranking them. 50/99 detections have ZERO overlap with GT (median IoU = 0).
4. **Rank correlation is a MISLEADING objective.** Interior curve mean has HIGHER spearman-vs-IoU
   (+0.52) than cached confidence (+0.45), yet substituting/blending it REGRESSES mAP (-0.020 to
   -0.026). Only checkpoint precision matters. (This is the most likely reason the v115-v123
   re-rank arc produced nine straight rejections.)
5. **Cached artifacts are exhausted, and WHY.** LOO-CV logistic regression over 13 cached features
   scores AUC 0.744/0.717 for IoU>=0.3 — WORSE than cached confidence alone (0.771/0.742).
   Root cause found in source: cached `frame_scores` is NOT the raw CLIP curve; it is the FLOODED
   curve (`frame_scorer.compute_frame_scores`, frame_scorer.py:63+ overlays each detection's own
   confidence onto 0.35*cat_scores, then smooths), so every feature read off it is partly circular
   with confidence.
6. **The un-flooded curve is recoverable offline.** `frame_scorer.get_category_scores`
   (frame_scorer.py:41-61) is verdict-INDEPENDENT and is exactly reproducible from
   `features/test/<vid>.npy` (290 files, (32,768), L2-normalized ViT-L/14) with the pipeline's own
   text embeddings + learned `prompt_weights.json`. On that curve the features ARE discriminative:
   act_max AUC 0.785/0.747, peak_gap 0.778/0.722, **peak_in 0.772/0.742**, halfmax_recall
   0.758/0.757 — at parity with the VLM-derived confidence, and all 16 features keep the same
   direction on both caches.
7. **Dual-cache rule (new discipline).** Duration/count arms (`conf*min(1,dur/D)` for D=5..20,
   dur/vdur, /ndet) flip sign between caches (v130 -0.005..-0.031 vs v132 +0.003..+0.014) = noise,
   reproducing the v112 duration-prior rejection. No lever may be promoted on one cache again.

---

## 1. Mechanism (single controlled change)

```
conf' = conf * (1 - w + w * peak_in),      w = 0.2 (default)
peak_in = 1 iff argmax(get_category_scores(video, cat)) falls inside [t_start, t_end]
```

Runs as the **last** confidence touch, after `_qscore_rerank` (pipeline.py:1288-1291), where count
and boundaries are already final. Never reads a VLM verdict; never writes t_start/t_end/count.

**Why this is a different object from the nine rejected re-rank levers:** all of them modulated
confidence with a quantity derived from the FLOODED output curve (v111 Q-score itself uses
`frame_scores`) or from geometry (duration, prominence, margins). GPC is the first to use the
**raw, verdict-independent, category-conditioned semantic curve** — the same signal the boundary
logic trusts — and it asks a question none of them asked: *did this window capture the video's
single strongest piece of semantic evidence?*

**Literature lineage** (from this loop's sweeps): IoU-Net (ECCV 2018) — classification confidence
carries no localization confidence, predict quality explicitly; AutoLoc (ECCV 2018) OIC loss and
P-MIL (CVPR 2023) — inner/outer contrast as training-free proposal quality; and the ranking-score
family BSN (multiply), BMN (`sqrt(p_cc*p_cr)`), TadTR (`sqrt(p*g)`), DiffTAD (`(p_bc+p_c)/2`).
Notably, across the 2023-2026 cohort **only TadTR and DiffTAD fuse a localization-quality term into
the ranking score, and they are exactly the two that can drop NMS** — quality-aware scoring and
reduced suppression dependence are the same lever. GPC is a bounded multiplicative member of that
family. MGFN (AAAI 2023) states the complementary constraint we respect: magnitudes are not
comparable across scenes, so we must NOT re-level per video (see §3).

## 2. Offline evidence (8 independent cached full evals, zero API cost)

| axis | result |
|---|---|
| cache x weight | **8/8 caches positive at every w in {0.15,0.2,0.3,0.4,0.5}** (v114/v124/v125/v127/v128/v130/v131/v132) |
| best weight | w=0.2: mean dmAP **+0.01909**, median +0.02108, min +0.01009, max +0.02598 |
| high-IoU | mean **dAP@0.7 +0.01059** — the gain is not bought at IoU 0.7 |
| parameter sensitivity | mean dmAP varies only +0.0177..+0.0191 across w in [0.15,0.5] — not a knife edge |
| curve resolution | 32 snippets +0.0186 (8/8 pos), 16 +0.0096 (7/8), 8 +0.0090 (8/8) — monotone in resolution, and the in-vivo curve is DENSER (~130 frames) than the offline probe |
| continuous variants | 1+peak_gap +0.0101, act_max_norm +0.0101 (both 8/8 positive but weaker than the binary) |
| fire / broadness | peak outside window on **512/784** detections (65.3%) across the 8 caches |
| demotion quality | of those, 454 are IoU<0.3 (correct) vs 58 IoU>=0.3 ⇒ **demotion precision 0.887**; recall on misses 0.768; false-demotion rate on hits 0.301 |

**Placebo controls (both decisive):**
- random binary demotion at the matched 65.3% rate: **-0.0248** mean (20 seeds, sd 0.0068,
  best seed -0.0147; P(placebo >= real) = 0.000). So the gain is NOT "demoting 65% of detections".
- peak_in computed from a WRONG anomaly category: **-0.0046** mean (3 shuffles). So the gain
  requires the correct category-conditioned semantics, not merely any curve's peak.

## 3. Risk assessment (FM1-FM14)

| FM | verdict |
|---|---|
| FM1 curve-shape | IMMUNE — curve is read-only; no boundary/half-max computation touched |
| FM2 detection count | IMMUNE — runs after conf-gate/ECE/EUBC/NMS/merge; nothing added or filtered |
| FM3 / FM4 self-consistency | N/A — no VLM calls at all |
| FM5 never activated | mitigated: per-fire `[GPC]` marker + `[GPC-GATE]` summary + COMMAND.txt; smoke gate VERIFIED (19 fires, `in=1` leaves conf identical, `in=0` applies 0.8) |
| FM6 sim-doesn't-transfer | RESIDUAL RISK, mitigated: offline probe uses a 32-snippet curve, in vivo uses the dense curve. Resolution axis is monotone (8→32 improves), so the denser curve should be >= as good, but this is an extrapolation. The paired in-vivo A/B settles it. |
| FM7 video-level priors | Not applicable in the failure sense: this is per-detection. For 1-detection videos it acts as cross-video calibration, which is what a globally-ranked AP actually needs. |
| FM8 one change/loop | respected — one flag, one factor |
| FM9 MPS tensors | safe — `get_category_scores` already returns numpy (frame_scorer.py:56 `.cpu().numpy()`) |
| FM10 wall-clock drift | N/A — no latency claim; cost is zero (curve cached per video+category, already computed during boundary refinement) |
| FM11 offline-first economy | fully respected — 8-cache battery + 2 placebos before any API spend |
| FM12 | N/A — no call dropping |
| FM13 structural cap | effect +0.019 is above the +0.005 promotion bar used at v114; not a magnitude no-op |
| FM14 verdict-visibility | N/A — nothing dropped; and GPC is deliberately verdict-INDEPENDENT |

**Deliberately NOT done (measured to be harmful):** per-video re-leveling. Agent B measured on the
v124 cache: per-video z-score **-0.0898**, per-video rank/winner-take-all **-0.0931**, min-max
keeping level +0.0000. Global AP depends on the cross-video confidence LEVEL, so GPC only applies a
bounded 0.8-1.0 factor and never renormalizes.

## 4. Implementation

- `config.py`: `ScoringConfig.gpc_rerank = False`, `gpc_weight = 0.2` + explicit `from_args`
  threading; verified `from_args(Namespace()).scoring == ScoringConfig()`.
- `run.py`, `batch_evaluate.py`: `--gpc_rerank` (store_true, default False), `--gpc_weight` (float).
- `pipeline.py`: new `_global_peak_containment` (before `detect`), invoked after the Q-score block
  (pipeline.py:1288+); header bumped to v138. Category resolution follows the ECE/EUBC pattern
  (detection's own `anomaly_type`, fallback to `videomae_category`, skip Normal/unknown/unlisted).
- Backups: `.qoder/backups/v138/{pipeline,config,run,batch_evaluate}.py`.
- Validation: AST OK x4, `import pipeline` OK, both flags in `--help`, defaults confirmed
  (gpc_rerank=False, gpc_weight=0.2), flag-off config byte-equivalent to `ScoringConfig()`.

## 5. Measurement protocol — PAIRED, pre-registered

The ledger shows a **+/-0.019 run-to-run full-eval spread between configs that are mAP-identical by
construction** (v130 0.17708 / v131 0.17163 / v132 0.15847). The expected GPC effect is the same
magnitude, so an unpaired "v138 vs REF" comparison cannot resolve it. Because GPC is a pure
multiplicative factor applied last, the ON run's own `[GPC]` lines record `conf_old -> conf_new`
per detection, so restoring `conf_old` reconstructs the flag-OFF arm EXACTLY on identical
detections/boundaries/verdicts. `analysis_v138_paired_invivo.py` does this.

**Pre-registered gates:**
- ACTIVATION: `[GPC]` lines present in smoke AND full, `unmatched == 0` in the paired harness.
- KEEP iff **paired dmAP >= +0.005** AND paired AP@0.7 non-regressing AND full mAP >= floor 0.15132
  AND demotion precision in vivo >= 0.75 (guards against a curve that behaves differently dense).
- REVERT otherwise, restoring from `.qoder/backups/v138/`.

## 6. Results — VERDICT: REVERT (pre-registered paired gate failed), root cause diagnosed, fix = v139

**Full (65 videos, flags ON, `results/batch_eval_v138_full`):** 61/65 succeeded (4 transient VLM
API failures), 93 `[GPC]` fires, avg 224.3 s/video, unpaired mAP 0.15372, frame AUC 0.717665.

**The decisive measurement — PAIRED A/B on the SAME 93 live detections:**

| arm | mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---|---|---|
| OFF (conf_old restored from the log) | 0.15658 | 0.22673 | 0.13885 | 0.10417 |
| ON (as shipped: dense per-frame argmax) | 0.15372 | 0.21337 | 0.13759 | 0.11019 |
| **paired delta** | **-0.00287** | -0.01336 | -0.00126 | +0.00603 |

Gate required >= +0.005 ⇒ **FAILED**. 93/93 detections matched (activation clean, FM5 satisfied).
Floor held (0.15372 > 0.15132). In-vivo demotion precision 0.815 (offline 0.887), but the
false-demotion rate on true hits blew out to **54.5%** (12 of 22) versus 30.1% offline — that gap
is the whole story.

### Root cause: FM6 via SIGNAL-RESOLUTION INFIDELITY (proposed **FM15**)

The offline probe reconstructed the curve from `features/test` = **32 pooled snippets**; the pipeline
computes `get_category_scores` **per sampled frame (~130)**. The argmax of a dense curve is a
single-frame **SPIKE**; the argmax of a pooled curve marks a region of **SUSTAINED** evidence. Those
are different signals wearing the same name. Measured, paired, on the same live detections:

| variant (same mechanism, same detections) | mAP | paired d |
|---|---|---|
| OFF | 0.15658 | — |
| dense per-frame peak (v138 as shipped) | 0.15372 | **-0.00287** |
| **32-bin pooled peak (the signal the offline battery actually tested)** | **0.17082** | **+0.01424** |

Dense vs pooled `peak_in` disagree on **20/93** windows; on the 22 true hits the dense peak keeps
only **10** vs **15** pooled. A deterministic unit test reproduces the inversion exactly: with a
sustained 0.7 plateau at 10-40 s and a lone 0.99 spike at 100 s, v138 demotes the plateau window and
keeps the spike window; the pooled version does the opposite.

**Lesson (FM15, for the taxonomy):** an offline precheck must reproduce the in-vivo signal's
TEMPORAL RESOLUTION, not merely its formula. My §7 weakness #1 named this risk but treated the
resolution sweep (8→16→32 monotone improving) as reassurance; extrapolating that trend past 32 to
~130 was exactly backwards, because pooling is not a fidelity loss here — it IS the mechanism. Eight
sign-consistent caches and two decisive placebos could not catch it, because all eight shared the
same pooled substrate. **A dual-cache rule protects against noise, not against a substrate that is
uniformly wrong.**

**Disposition:** v138's dense-argmax arm is refuted and is NOT shipped. Rather than restoring the
backups and discarding a mechanism that measures +0.01424 paired on live data, the code is carried
forward as **v139** with `gpc_pool_bins` (default 32) pooling the dense curve before the argmax;
`gpc_pool_bins=0` reproduces the refuted v138 behavior for auditability. The flag remains default
OFF, so the shipped default is still byte-identical to v137. v139 is under full eval with the same
paired protocol.

- **Smoke (14 videos, flags ON):** 11/14 succeeded (3 failures = transient detect-path VLM API
  ConnectTimeouts, the same class seen in v137's smoke, unrelated to GPC — zero tracebacks in the
  GPC path). smoke mAP **0.20184**. `[GPC]` fired 19x, paired delta **0.00000** on the smoke subset
  with demotion precision **0.875** (vs 0.887 offline): the mechanism behaves as designed but 19
  detections / n_gt~14 is far too quantized for the AP checkpoints to move. Smoke is an
  activation/crash gate only.
- **Full (65 videos):** running — `results/batch_eval_v138_full`, COMMAND.txt recorded.

## 7. Honest weaknesses

1. **Resolution extrapolation (FM6).** Every offline number comes from a 32-snippet reconstruction,
   not the dense in-vivo curve. The dense curve's argmax can land on a narrow spike that the pooled
   probe smoothed away, which would change `peak_in` for borderline windows. The resolution sweep
   trends the right way but is not proof; the paired A/B is the real test.
2. **Arm selection on the evaluation set.** `peak_in` was chosen after inspecting ~20 arms on the
   same 65-video set that scores the result. The 8-cache/2-placebo battery and the w-insensitivity
   make a pure fluke unlikely, but the effect size should be treated as an upper estimate.
3. **The 31% false-demotion rate is real** — 58 of 193 true hits get demoted. GPC wins on net, not
   per detection; a bounded w (0.2) is precisely what keeps those wrong demotions survivable.

## 7b. Family saturation — measured, so loop 2 does NOT re-spend here (FM11)

Stacking probes on the same 8 caches, each stacked ON TOP of GPC w=0.2 (deltas include GPC's
+0.01909, so the incremental value is the difference):

| arm | mean dmAP | pos | incremental over GPC |
|---|---|---|---|
| GPC w=0.2 (reference) | +0.01909 | 8/8 | — |
| GPC + interior act_max_norm (w2=0.2) | +0.02043 | 8/8 | **+0.0013** |
| GPC + top-2-peak containment (w2=0.1) | +0.01893 | 8/8 | -0.0002 |
| GPC + multi-scale peak vote (w2=0.2) | +0.01626 | 8/8 | -0.0028 |
| GPC + peak-distance decay (w2=0.2) | +0.01399 | 8/8 | -0.0051 |
| multi-scale vote INSTEAD of peak_in | +0.01310 | 8/8 | (weaker alone) |
| top-2-peak INSTEAD of peak_in | +0.01543 | 8/8 | (weaker alone) |
| GPC at w=0.3 | +0.01886 | 8/8 | -0.0002 |

**Conclusion:** the binary single-peak gate captures essentially all of the extractable signal in
this family; every refinement (softening, multi-peak, multi-scale, distance weighting) is neutral or
negative, and no stack clears the +0.005 promotion bar. **v139 must attack a different signal**, not
a peak_in variant.

Agent B's independently measured ceiling decomposition (v124 / v114) frames where to go:

| bound | v124 d | v114 d |
|---|---|---|
| ORACLE-FULL (perfect global order) | +0.10197 | +0.09229 |
| ORACLE-CROSS-VIDEO (per-video level oracle, intra-order kept) | **+0.05180** | +0.05089 |
| ORACLE-WITHIN (permute each video's own confidences) | +0.00602 | +0.00623 |
| RANDOM rank (8 seeds) | -0.11072 | -0.08975 |

The **within-video** budget is ~99% already spent (committed Q-score's live A/B was +0.00619 against
a +0.006 oracle) — which explains the nine-rejection v115-v123 arc: PROM/FSL/MELC/SLM/BOR were all
within-video normalized and were competing for a +0.006 prize. GPC is an ABSOLUTE (cross-video)
quantity, so it draws on the +0.052 cross-video oracle and harvested ~37% of it, leaving **~+0.033
measured headroom in the cross-video-level family** — the registered v139 target. Note B also
re-tested and killed five other cross-video re-leveling candidates (MIL prior -0.0016..-0.0092;
absolute CLIP anomaly-vs-Normal margin, i.e. v123's never-run "strongest untested lever",
-0.0004..-0.0234; curve_std +0.0022 on v124 but -0.0350 on v130; det_dur +0.0024/-0.0086;
background level -0.006..-0.018) — none replicate. So the family is alive but narrow: only
semantically-conditioned, bounded, absolute curve properties have worked.

**Explicitly rejected for v139 (decided at orchestrator level): DPA / dense bottom-ranked proposal
append.** Agent B measured +0.0307 mean over 6 caches (6/6 same sign, AP@0.3/0.5/0.7 all up,
monotone in K to +0.0336 at K=100) by appending ~40-100 synthetic multi-scale grid proposals per
video strictly below the confidence floor (min real conf 0.25861 vs band top 0.02), which is
AP-monotone by construction under `_compute_ap`. It is nonetheless **metric gaming, not a
localization win, and will not be shipped**: frame AUC is bit-identical (0.708393 -> 0.708393),
~+0.024 of the +0.031 is reproduced by a signal-FREE random generator (so most of it is lottery
tickets, not score quality), the ordering signal's attribution does not survive scrutiny (q_abs
nearly doubles the grid's gain but HURTS random), and post-DPA numbers would not be comparable
either to the v51-v137 ledger or to external baselines whose submission density is unknown. B, who
proposed it, concurs and recommends against shipping it. Its one legitimate use is as
infrastructure: grid K=40 with ORACLE tail ordering scores +0.1147 while our q_abs ordering realizes
+0.0307 (27%), so the residual +0.084 is a pure ranking problem on a set that never touches the
half-max curve — the only FM1-immune landing site found for `model_mil_v6.pth` or a temporal-context
model trained on `features/train`. That is a future infrastructure decision requiring an explicit
protocol disclosure, not an mAP claim.

## 8. Process note

Agent A's literature leg died twice on API transport errors (`Connection closed mid-response`) and
its findings arrived only via a nested sweep; Agent B's own report body was lost the same way (only
its addendum arrived). Built-in WebSearch is disabled in this environment and several arXiv fetches
returned HTTP 403, so the literature grounding in §1 is from search-snippet-level sources and should
be treated as motivation, not as verified citation. The empirical case for v138 stands entirely on
on-disk measurement.
