# Loop 9 — v122 BOR: Boundary-Offset Recentering

**Date:** 2026-07-25
**Decision:** REVERT (ships flag-off, byte-identical v114 baseline)
**Kept:** false

---

## Approved Proposal

**Title:** BOR — Boundary-Offset Recentering: a duration-frozen rigid TRANSLATION of each detection window driven by un-flooded fused-curve exterior asymmetry (the operation missing from the boundary toolkit), harness-first offline A/B on v114's real cached detections, stacked on the committed v111 Q-score.

**Mechanism:** A read-only, count-frozen, per-detection boundary-PLACEMENT stage `_recenter_boundaries(final_anomalies, frame_scores, timestamps)` inserted in `Pipeline.detect()` immediately AFTER `compute_frame_scores` and BEFORE the committed v111 Q-score block, so Q-score re-ranks the recentered windows on the same cached fused curve. It RIGIDLY TRANSLATES each window by a whole-window shift while holding DURATION FIXED (v51 half-max width preserved). Per detection `[t_s,t_e]`, using ONLY the un-flooded EXTERIOR of the cached fused curve: `Lext = mean(fs over [t_s-m, t_s])`, `Rext = mean(fs over [t_e, t_e+m])`, margin `m = bor_margin_frac*dur`, clipped so the margin never enters a neighboring detection's window nor exceeds video bounds. Imbalance `I = (Lext - Rext)/(fs.max()-fs.min()+eps)`. If `|I| >= bor_gate_frac`, shift the whole window toward the higher-exterior side by `step = clip(bor_gain*I, -cap, +cap)*dur`, `cap = bor_max_shift_frac*dur`; grid-snap via `searchsorted`, re-clip to bounds holding dur. GUARDS -> INERT: Normal/no-detection, curve None/<3 samples/misaligned, flat curve, `|I|<gate`, or a shift running off-video or into a neighbor. After all shifts, re-apply the existing hard `temporal_nms(iou=0.3)` + `global_merge` so a translation can never manufacture a near-duplicate (count monotone NON-INCREASING). Fused curve, confidence, count-inputs, ordering, category are byte-identical to the Q-on baseline. Arm set pre-registered = arm SELECTION not tuning: `{balance_conservative, balance_moderate}` this loop (`balance_consensus` DEFERRED — requires per-channel curves v114 did not cache).

**Rationale (short):** Under greedy IoU matching, mAP is governed by localization IoU, not confidence order. The confidence-re-rank family is CONFIRMED TAPPED (v95/v111 the only wins; v107/v115/v118/v119 all sub-+0.005). FM7 proves a re-rank cannot lift a mislocated window's own IoU — only moving the window can. The existing boundary toolkit is provably incomplete: v52 shrink-only, v107/v121 expand-only, v120 independent-edge snap; NONE performs a rigid duration-preserving TRANSLATION. BOR targets the pure-OFFSET slice of the IoU-0.1-0.5 bucket via exterior-asymmetry (AutoLoc/SSN OIC + ActionFormer/G-TAD center-offset lineage), reduced to a training-free rule reading only the offline-visible un-flooded exterior.

**Expected gain:** Honestly BORDERLINE (+0.005 to +0.015 mAP concentrated in AP@0.5/0.7 IF off-center offset errors are a real slice); floor near zero by construction (gate+cap+inert guards+post-shift hard NMS/merge). Adjudicated deterministically OFFLINE before any live budget.

---

## C's Feedback

APPROVE with mandated changes. Structural claims verified against tree (ScoringConfig:55, qscore_rerank default-ON:88, insertion 1213->1218, temporal_nms/global_merge:581/1133, v114 cache lacks per-channel curves so deferring balance_consensus is honest). Clears the bar: architecture-level new operation (rigid duration-frozen translation), correctly executes the MANDATED pivot off the tapped confidence-re-rank family onto boundary PLACEMENT (FM7), FM1/FM2 immune by construction, FM8 clean, HARNESS-FIRST with v121-style LAUNCH GATE finally fixing the recurring v117/v120/v121 "harness built but never run" failure.

TRUST NOTE: BOR changes boundaries so it is NOT the same trusted class as Q-score (reorder-only); it IS trustworthy for a different reason — a deterministic transform of v114's REAL cached fused curve vs REAL GT (not FM6 candidate-pool sim). Residual-3 interior-flood mismatch (shifted window reads a curve flooded at the OLD location) makes shift-decision evidence exterior-only and partly synthetic — a real limitation the end-to-end mAP-on-real-GT harness absorbs.

MANDATED CHANGES: (#1) pin down the LIVE verdict protocol as a deterministic single-run A/B off the run's OWN fresh cached curve (v107 protocol), NOT two headline live runs (else the honest +0.005-0.015 effect drowns under +/-0.03 VLM noise, as killed v115). (#2) harness reports per-video localization delta distribution + broad_based_ok gate. (#3) CAUSALITY diagnostic: per-shifted-det best-IoU before-vs-after + causal_ok. (#4) channel serialization — DROP if byte-identity cannot be shown trivially.

---

## Implementation Notes (D1)

Files: config.py, pipeline.py, batch_evaluate.py, run.py, iteration_logs/analysis_v122_bor_precheck.py. Backups: `.qoder/backups/v122/`.
- pipeline.py: `MultiAnomalyVAD._recenter_boundaries(...)` implementing exterior-balance rigid shift + guards + neighbor-clipped margin + post-shift temporal_nms/global_merge. Arm table `_BOR_ARMS = {balance_conservative:(0.3,0.25), balance_moderate:(0.5,0.50)}`, shared `bor_gain=0.5, bor_gate_frac=0.1`. Inserted after compute_frame_scores, before Q-score block. Docstring bumped v114->v122.
- config.py: ScoringConfig gains `boundary_offset_recenter:bool=False, bor_mode:str='balance_moderate', bor_margin_frac=0.5, bor_max_shift_frac=0.5, bor_gain=0.5, bor_gate_frac=0.1`; threaded via from_args. Flag-off = byte-identical v114.
- CLI + GATES: `--boundary_offset_recenter` (store_true), `--bor_mode {balance_conservative,balance_moderate}`. `check_bor_launch_gate()` sys.exit(2) unless `analysis_v122_bor_precheck.json` exists with `promoted_arm==--bor_mode`. `check_bor_activation_gate()` post-eval grep gate on `[BOR] summary:` markers. Every processed video emits exactly one `[BOR] summary:` marker (fired path, all guard paths, both detect() early returns).
- HARNESS (analysis_v122_bor_precheck.py): clones v111 structure; identity arm sanity anchor round-trips cache exactly (max conf reconstruction err = 0.0); mandated broad_based + causality diagnostics. Pre-registered promotion: dmAP>=+0.005 AND AP@0.7 non-regression AND count monotone non-increasing AND broad-based AND causal-positive.
- MANDATED #4 (channel serialization): DROPPED as C authorized (byte-identity not trivially shown) — diff is BOR+gates ONLY, single controlled change uncofounded. balance_consensus remains deferred.

---

## Eval Flags + Activation Marker

- **Treatment CLI:** `--qscore_rerank --boundary_offset_recenter --bor_mode balance_moderate`
- **Control:** `--qscore_rerank` (v114 baseline; qscore default-ON post-v114)
- **Activation marker:** `[BOR] summary:`
- **Launch gate marker:** `[BOR-GATE] LAUNCH:`

### Marker counts
| stage | `[BOR] summary:` markers | outcome |
|---|---|---|
| smoke | 0 | launch gate refused (PID 57835 died <1s) |
| full  | n/a | skipped (gate) |

---

## Results

### Offline harness (analysis_v122_bor_precheck.json — the decisive adjudication)
| arm | mAP | AP@0.3 | AP@0.5 | AP@0.7 | dmAP | dAP@0.7 | broad_based_ok | causal_ok | passed |
|---|---|---|---|---|---|---|---|---|---|
| baseline (v114) | 0.15013 | — | — | 0.11019 | — | — | — | — | — |
| balance_conservative | 0.11541 | 0.21145 | 0.08106 | 0.05372 | **-0.03472** | -0.05647 | false | false | **false** |
| balance_moderate | 0.13764 | 0.23719 | 0.09585 | 0.07989 | **-0.01249** | -0.03030 | false | false | **false** |

`promoted_arm = null`. Causality diagnostic: moderate `iou_net_per_shifted_det = -0.00092` (22 shifted dets across 20 firing videos; videos_iou_improve=3 vs regress=4). conservative `iou_net = -0.00419` (19/19; improve=4 regress=3). Both arms REGRESS both globally and on the shifted detections themselves — the translation moved windows the WRONG way on net.

### Live eval
| metric | smoke | full |
|---|---|---|
| mAP | -1 (not run) | -1 (not run) |
| AP@0.3 / 0.5 / 0.7 | -1 | -1 |
| avg latency (s/video) | -1 | -1 |

Live run never executed: the launch gate correctly refused because the harness legitimately promoted no arm.

---

## Decision + Reason

**REVERT.** The STEP-0 offline harness RAN (unlike v117/v120/v121) and did its job: it adjudicated `promoted_arm=null` because BOTH pre-registered arms regressed — balance_moderate dmAP -0.0125 / dAP@0.7 -0.030, balance_conservative dmAP -0.0347 / dAP@0.7 -0.056, both broad_based_ok=false AND causal_ok=false (net-negative IoU on the very detections they shifted). The v121-style launch gate then correctly refused the flagged live smoke (`[BOR-GATE] LAUNCH: ...promoted_arm!=bor_mode (want=balance_moderate, got=None)`), so zero eval budget was wasted. The finalizer did NOT fabricate the precheck JSON to bypass the gate. Restored all 4 files from `.qoder/backups/v122/` (config.py, pipeline.py, batch_evaluate.py, run.py) — each diff-verified byte-identical; `ScoringConfig` no longer carries `boundary_offset_recenter` (confirmed via import). Baseline stays v114 (full mAP 0.15013 on this cache set; committed headline 0.15632).

---

## Lessons Learned

- **FM7 (mislocalization) — REFUTED for this operator.** Rigid duration-frozen TRANSLATION driven by fused-exterior asymmetry does NOT net-improve localization on v114's real detections: causality diagnostics show net-NEGATIVE IoU on the shifted detections themselves (moderate -0.00092/det, conservative -0.00419/det). Exterior-asymmetry is too noisy a shift-direction signal without the deferred cross-channel consensus gate — RESIDUAL RISK 2 (spurious single-channel exterior bump drives a wrong shift) materialized, and RESIDUAL RISK 3 (interior-flood mismatch) likely compounds it.
- **Process discipline WORKED.** Harness-first + launch gate is now validated end-to-end: the harness ran BEFORE any live budget, adjudicated an honest true-negative, and the gate blocked the wasted run. This is the correct resolution of the v117/v120/v121 harness-skip failure — a negative that cost zero eval budget beats an un-adjudicated full run.
- **Boundary-placement family: translation arm now REFUTED offline** (adds to v120 independent-snap, v121 expand — none cleared). Shrink (v52) and the v51 half-max width remain the only surviving boundary ops. Next boundary attempts need the per-channel consensus signal (requires caching motion/tsm/clip curves) or a fundamentally different localization signal.
- **FM1/FM2 immune by construction confirmed** (count monotone non-increasing: n_det 100->99 both arms; fused curve/confidence/order/category untouched). FM8 clean (single flag, never bundled).
