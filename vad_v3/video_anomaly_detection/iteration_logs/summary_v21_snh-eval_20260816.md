# Session v21 — v179 SNH (Supervised Normality Head) 290-eval: CLOSED on GT-HARM (G4); G1 paired VOID (M4)
# Agent C decision-execution record, 2026-08-16. One full-290 paired-log eval of the v19-GO-gated
# Normal-video false-alarm lever. Every load-bearing number below was re-grepped / recomputed from
# the produced artifacts THIS session (FM20; verification script + output banked, section 6).
# No eval was re-run by Agent C; the evaluator, GT labels, and analysis_bootstrap_ab.py are untouched.

Date: 2026-08-16
Scope: SNH — the v17-record-mandated, v19-GO-gated successor to CNV for the owner-directed
NORMAL-video false-positive objective: a video-level supervised P(normal) gate (tau=0.7894) that
vetoes ALL of a gated video's detections (confidence := 0.0, detections kept), judged by the four
pre-registered gates (paired mAP non-inferiority AND abs floor AND >=25% Normal-FA-rate reduction
AND <=1 GT-overlapping veto).
Answer: **CLOSE. `snh_veto` stays default False. The code stays in the tree inert (same disposition
as cnv). SNH closed on HARM (G4: 5 GT-overlapping true-positive vetoes, limit <= 1), and the
pre-registered paired adjudication is VOID on M4 exactly as the pipeline's own [SNH-COLLISION]
warnings predicted.**

## 0. HEADLINE

- **G1 paired mAP delta >= -0.003: VOID (M4)** on the pre-registered conf-restore path — 92
  [SNH-FIRE] lines collapse to **87 unique rounded (vid,t_start,t_end) keys** while 92 detections
  match them (4 [SNH-COLLISION] videos: Normal_Videos_641/696/909/932), so the flag-OFF arm is not
  faithfully restorable by the registered path and no delta may be reported from it. Supplementary
  exact idx= keyed restore (0/92 mismatches at the log's print precision; recomputed by Agent C):
  OFF mAP 0.17120316, **point delta +0.00204, 90% CI [-0.01023, +0.00671], P(delta>0)=0.579** —
  even the exact restore cannot establish non-inferiority (CI crosses -0.003).
- **G2 abs mAP(ON) >= 0.16454: PASS** — 0.17324230 (evaluation_report.json; recomputed via
  evaluate.compute_temporal_map, reproduces the report to 1e-12).
- **G3 Normal-video FA-rate relative reduction >= 25%: PASS — 67.1%** (0.50667 -> 0.16667;
  76 -> 25 of 150 Normal videos; surviving fp dets/Normal video 0.847 -> 0.300, i.e. 127 -> 45;
  51 Normal videos made detection-free).
- **G4 GT-harm <= 1 vetoed detection overlapping GT at IoU>=0.1: FAIL — 5** (max IoU 0.7333,
  two of them high-confidence true positives at conf 0.9167 / 0.9140). Section 4.
- VERDICT: **CLOSE** — G4 fails decisively at 5x the budget; G1 is VOID on the registered path.

## 1. THE MECHANISM, and the v19 GO context it stood on

SNH (v179) is a tiny supervised video-level P(normal) head — multinomial logistic regression over
16 aggregate features computed from the pipeline's OWN cached signal families (MIL snippet stats,
fused CLIP category-curve stats, CLIP Normal-prompt similarity, motion-energy outliers), ZERO new
VLM/CLIP/MIL calls at train and eval time (SHARED module snh_features.compute_snh_features,
no train/eval skew). Veto: at the TERMINAL slot, if P(normal) >= tau, every detection of the video
gets confidence := 0.0 and is KEPT in the list (count-frozen), one [SNH-FIRE] line per vetoed
detection in the v171-proven marker format, plus one unconditional [SNH-VID] summary per video on
EVERY path (FM17, including 0-detection and early-return-normal paths).

v19 (offline GO gate, summary_v19_snh-gate_20260816.md) trained the head on 102 TRAIN videos and
measured: stratified 5-fold CV **AUC 0.996** (model_snh.json: cv_auc 0.99615); **tau = 0.7894**
(model: 0.789417) = largest threshold with CV anomaly-flag <= 5% (measured 2/52 = 3.8%); train
normal-flag **47/50 = 94%** vs the 40% GO bar -> GO. Top-weighted features were ALL MIL-family
(mil_frac_gt_0.5, mil_top8_mean, mil_top3_mean, mil_max) — SNH is largely a calibrated threshold
on the trained WSAD MIL head. v19 carried an explicit IN-DISTRIBUTION CAVEAT: model_mil.pth was
trained on that same train split, so 94%/3.8% is an UPPER BOUND; transfer to TEST normals was the
open question v21 existed to answer. v21's answer: the gate fires on **58/150 = 38.7% of test
Normal videos** (less than half the train rate — the in-distribution optimism is confirmed) and on
**9/140 = 6.4% of anomaly test videos**, which translated directly into the G4 harm below.

## 2. EXECUTION — first-attempt stall failure; self-tests; queue discipline

The FIRST v21 attempt died on a process failure, not a result: a foreground Bash command exceeded
~150s and the runtime killed the stalled agent at 180s. The re-attempt ran under strict
discipline — ALL long work nohup-background + short polls, never two batch_evaluate processes
concurrently (16 GB swap-thrash) — and completed: implementation self-tests, one full-290 eval
(wall 19762.4s), paired adjudication, FA and GT-harm analyses.

Self-tests (main-session/agent, before the 290 spend; 14-video set; greppable artifacts):
- OFF: results/v179_selftest_off.out — temporal mAP **0.2238**; 0 [SNH-ERR], 0 Traceback.
- ON:  results/v179_selftest_on.out  — temporal mAP **0.2273**; 0 [SNH-ERR], 0 Traceback.
- FORCE (SNH_FORCE_FIRE=1): results/v179_selftest_force.out — every detection vetoed
  (force=1 on every [SNH-VID]), mAP **0.0457**; early-return path still emits its unconditional
  [SNH-VID] (skip=early_return_no_candidates); 0 [SNH-ERR], 0 Traceback. The force run already
  showed 3 [SNH-COLLISION] warnings on 14 videos — the M4 collision risk was visible BEFORE the
  290 spend and was correctly carried into the pre-registered adjudication as a VOID condition.
- 2-video liveness smoke (batch_eval_v179_smoke_on/off): 2/2 succeeded both arms; the gate fired
  on the smoke Normal video (p_normal=0.9933) with ndet=0 — nothing to veto.

Harness (full run, COMMAND.txt): `batch_evaluate --dataset_dir ~/Movies/ucf-crime
--output_dir results/batch_eval_v179_snh290 --max_per_category 999 --concurrency 2 --snh_veto`.
NO --categories filter (Normal videos ARE the subject). Result: **290/290 succeeded**, avg
136.17 s/video, wall 19762.4s (2026-08-16 21:12 -> 2026-08-17 02:43).

## 3. THE INSTRUMENT — paired-log conf restore, and WHY G1 is VOID (M4)

The pre-registered adjudication is
`analysis_bootstrap_ab.py --new <dir> --paired-log <log> --marker SNH-FIRE --paired-mode conf`,
whose restore keys each fire on the ROUNDED (vid, t_start, t_end) triple (2-decimal). SNH vetoes
whole videos, so videos whose two vetoed detections share a rounded window produce colliding keys.
The pipeline detects this at veto time and warns ([SNH-COLLISION], dup_veto_keys=n, "M4 VOID
expected"). On the full run: 92 [SNH-FIRE] lines but only **87 unique rounded keys** — collisions
on Normal_Videos_641 (2 fires 1 key), _696 (2:1), _909 (2:1), _932 (3:1). The registered restorer
would restore 92 detections from 87 keys (matched=92 != n_fires=87) — the two arms cannot both be
right, so per M4 **no delta may be reported from the registered path**. G1 is VOID, full stop.

Supplementary (non-registered, reported for information only): the [SNH-FIRE] lines also carry an
`idx=` field (the detection's index in the video's anomaly list). An exact idx-keyed restore
rebuilds the OFF arm unambiguously — **0/92 mismatches** at the log's print precision (Agent C
recompute: 88/92 bit-identical at 1e-6, 4 differ <=0.005s from the 2-decimal printed times) — and
yields OFF mAP 0.17120316, point delta **+0.00204**, video-level bootstrap (seed 0, 1000
resamples, same protocol as analysis_bootstrap_ab.py) 90% CI **[-0.01023, +0.00671]**,
**P(delta>0)=0.579**, sd 0.00524. So even setting the VOID aside, the exact restore cannot
establish non-inferiority: the CI lower bound crosses -0.003 and the sign is a coin flip.

## 4. THE NUMBERS (290; all recomputed from artifacts this session)

Arm table (compute_temporal_map, fps=30, GT test split, 290 common videos):
```
arm          mAP    AP@0.3    AP@0.5    AP@0.7
ON (SNH)  0.17324   0.27237   0.14559   0.10176     = evaluation_report.json to 1e-12 (num_videos=290)
OFF (idx) 0.17120   0.26403   0.14623   0.10334     exact idx= restore of the 92 vetoed confs
delta    +0.00204  +0.00834  -0.00064  -0.00158     CI90 [-0.01023, +0.00671]  P(delta>0)=0.579
```
(Full precision: ON 0.17324230348112937; OFF 0.17120316066442; delta +0.00203914.)

Objective (Normal-video false alarms; arm-ON = detections with conf>0 in the produced JSONs;
arm-OFF = idx-restored confs):
```
arm    FA videos (of 150)   FA rate    surviving fp dets   per Normal video
OFF        76               0.50667          127               0.847
ON         25               0.16667           45               0.300
```
Relative reduction **67.1%** (gate >= 25%); **51 Normal videos made detection-free**.

Harm (vetoed detections on the 9 gated anomaly videos, IoU vs GT at fps=30; 10 vetoes total):
```
video              t (s)        conf     IoU vs GT   counted (>=0.1)
RoadAccidents128   3.00-8.00    0.9167   0.7333      YES — high-conf true positive
Explosion025      10.00-16.50   0.8007   0.5106      YES
Shooting048       46.00-52.00   0.6076   0.4286      YES
Shoplifting022    45.50-55.00   0.5402   0.4211      YES
RoadAccidents002   9.00-11.50   0.9140   0.2857      YES — high-conf true positive
Explosion016      13.50-25.50   0.9285   0.0769      near-miss, NOT counted
```
**gt_vetoed = 5** (gate <= 1) — fails by 5x, and two of the five are confident true positives:
the video-level normal gate DOES veto real anomalies on held-out test videos.

Gate fire rates (from the 290 unconditional [SNH-VID] lines): gate=1 on 67/290 videos overall;
**58/150 = 38.7% of Normal test videos** (vs 94% on train — v19's in-distribution caveat
confirmed as an upper bound); **9/140 = 6.4% of anomaly test videos** (Explosion016/025,
RoadAccidents002/004/009/128, Shooting048, Shoplifting022/039); 60 videos had >= 1 veto; sum of
vetoed = 92 = [SNH-FIRE] count (cross-check).

## 5. THE 4-GATE TABLE (pre-registered in v19, identical thresholds to v17)

| gate | criterion | result | verdict |
|---|---|---|---|
| G1 | paired mAP delta >= -0.003 (registered conf restore) | VOID (M4): 92 fires -> 87 rounded keys, matched != n_fires; no delta reportable. Supplementary idx restore: +0.00204, CI90 [-0.01023, +0.00671], P=0.579 — CI crosses -0.003 | **VOID** (supplementary: not established) |
| G2 | abs mAP(ON) >= 0.16454 | 0.17324230 | **PASS** |
| G3 | Normal-FA-rate relative reduction >= 25% | 67.1% (0.50667 -> 0.16667) | **PASS** |
| G4 | gt_vetoed (IoU>=0.1) <= 1 | 5 (max IoU 0.7333) | **FAIL** |

VERDICT: **CLOSE** — G4 fails decisively; G1 VOID on the pre-registered path. G2/G3 passes do not
rescue a mechanism that vetoes confident true positives.

## 6. FM EVIDENCE — grepped / recomputed THIS session (FM20), not asserted

Raw marker greps on results/batch_eval_v179_snh290.log:
- `grep -c [SNH-VID]` = **290** (FM17 pass — one unconditional per-video summary on EVERY path,
  including 0-detection and early-return-normal paths; self-test force run shows
  skip=early_return_no_candidates).
- `grep -c [SNH-FIRE]` = **92** (FM5 pass — real mechanism activations, not a no-op; sum of
  vetoed= over [SNH-VID] lines = 92, cross-checked).
- `grep -c [SNH-COLLISION]` = **4** (Normal_Videos_641/696/909/932; the M4-VOID prediction).
- `grep -c [SNH-ERR]` = **0** (no model-file/ feature errors; never-crash path unused).
- Unique rounded (vid,t_start,t_end) keys among the 92 fires = **87** (recomputed).
- FM21: marker strings exactly [SNH-FIRE] / [SNH-VID] / [SNH-COLLISION] / [SNH-ERR].
- FM18: absolute mAPs read and reproduced, not just deltas (section 4).

Recompute (records-phase verification, artifacts only — ZERO model calls, ZERO eval):
`iteration_logs/analysis_v21_snh_records_verify.py`, output banked at
`results/v21_records_verify.out`. It reproduces, from the produced JSONs + log + GT parquet:
arm-ON mAP 0.17324230348113 (= evaluation_report.json), arm-OFF 0.17120316066442, delta
+0.00203914, bootstrap CI [-0.01023, +0.00671] / P=0.579 (exact match to the adjudication),
FA 76->25 / 127->45, the five G4 rows with their IoUs, Explosion016 near-miss 0.0769, gate rates
58/150 and 9/140, and the 92->87 key collapse with the four collision videos named.

## 7. THE DECISION — what exactly is refuted, what remains open

**CLOSE. Refuted: the video-level supervised P(normal) gate on cached MIL+CLIP+motion features at
tau=0.7894 as a Normal-FA lever under the pre-registered gates.** The refutation is on HARM, not
on under-fire or on mAP: at the tau that legitimately passed the v19 train GO (94% normal-flag at
3.8% anomaly-flag, CV AUC 0.996), the gate fires on 6.4% of held-out anomaly test videos and those
fires include confident TRUE detections (RoadAccidents128 conf 0.917 IoU 0.733; RoadAccidents002
conf 0.914 IoU 0.286). A whole-video normality decision cannot express "this video is normal
EXCEPT at 3-8s"; the veto is all-or-nothing per video, and on test data the head's errors are
exactly the pipeline's true positives. The FA objective passing at 67.1% does not rescue it —
the pre-registered <=1 harm gate exists precisely to catch this, and it caught it at 5x.

Secondary measured facts: (a) the train->test transfer gap is real — 94% train normal-flag vs
38.7% test normal-flag (in-distribution MIL optimism, as v19 flagged); yet even the reduced test
fire rate produced decisive harm, so "fire less" is not "fire safe". (b) The paired adjudication
path itself needs the idx= field (or JSON-side pre-mechanism rows) for whole-video veto mechanisms
— rounded-window keys collide whenever one video loses two same-window detections; the pipeline's
own [SNH-COLLISION] warning predicted the M4 VOID and was right. (c) Even the exact restore shows
no establishable mAP non-inferiority (CI crosses -0.003), so a post-hoc "it didn't hurt mAP"
rescue is not available either.

Remains OPEN (would need its own mechanism + train demonstration + pre-registered gates, per the
v17 rule): an ANOMALY-SHIELDED variant — e.g. veto only detections that are independently
weak on a second signal, or a per-detection (not per-video) normality score — anything that can
spare a confident localized true positive inside an otherwise normal-looking video. Also banked
from v19: the MIL family dominates normal/anomaly separation; CLIP-curve-z and motion are weak.

CONSEQUENCE for future sessions: do NOT re-run SNH at a different tau (a higher tau just re-finds
the same trade-off lower down; the 5 TP vetoes happened at the train-optimal operating point), and
do NOT propose any whole-video P(normal) veto on these cached features without an explicit
per-detection anomaly shield. Both are measured-closed by this record.

## 8. FLAG STATE — verified THIS session, nothing flipped

CLOSE branch executed (no backup needed — no default changed):
- `ScoringConfig()` re-instantiated and printed THIS session (2026-08-16, python3.14):
  snh_veto False; cnv_veto False; bue_enabled True; cbt_enabled True; cbt_recal True;
  qscore_rerank True; cgr_enabled False; cgr_ab_diag False; ccr_enabled False.
- `VADConfig.from_args(argparse.Namespace()).scoring.snh_veto` = False, and
  from_args(empty).scoring == ScoringConfig() (byte-level equality) — re-verified AFTER the edit.
- Both CLIs carry the opt-in AND opt-out with default False (grep): run.py L286 `--snh_veto`,
  L306 `--no_snh_veto`, L309 `set_defaults(snh_veto=False)`; batch_evaluate.py L377/L397/L400 same.
- `py_compile` clean on config.py, pipeline.py, run.py, batch_evaluate.py, snh_features.py.
- The ONLY code-tree change this session: a one-line closure comment at the snh_veto field in
  config.py recording the v21 CLOSE verdict (comment-only; defaults re-verified post-edit).
- Flag-off byte-identity at source: all three _snh_veto call sites guard on
  `self.config.scoring.snh_veto` (pipeline.py L1818 early-return-normal, L1980 no-candidates,
  L2355 terminal); when False no signal is read, model_snh.json is never loaded, no marker emits.
  The implementation stays in the tree, inert, default-OFF — same disposition as cnv/cgr/ccr/tcp.

## 9. UPDATED CLOSED / BANNED FAMILIES

- **NEW CLOSED — SNH / video-level supervised P(normal) gate veto (v179, tau=0.7894, MIL+CLIP+
  motion cached features):** G4 gt_vetoed 5 (>1), G1 paired VOID (M4 rounded-key collisions;
  supplementary exact restore delta +0.00204, CI crosses -0.003), G2/G3 pass (abs 0.17324; FA
  -67.1%). Closed ON HARM. Includes: any re-tune of tau, and any whole-video normality veto on
  these cached features WITHOUT a per-detection anomaly shield.
- Carried forward CLOSED: CNV cascade null-hypothesis confidence veto (v171/v172, closed on harm);
  candidate RELOCATION (v164/v166/v168/v169); CBT anchor/seed re-centering (v161-v163); PMX
  (v159); additive VLM rescue (v143); CSR (v145); MARL-D (v146); GPC (v138/v139); KFD-U (v173);
  TCP prompt-side temporal contrast (v177).
- BANNED families (auto C-REJECT) unchanged: 1 curve within-video re-rank; 2 VLM merge/batch;
  3 max_scan/scan-tail; 4 per-video normalization; 5 dense proposal APPEND; 6 compute trims;
  7 boundary OUTWARD-growth = BUE; 8 scan_segment-ABS-conf gates; 9 Normal-FP suppression for mAP;
  10 MIL-relocation on the 32-snippet substrate. (SNH did not violate family 9 — its objective is
  the FA rate with mAP as a non-inferiority constraint; its measured outcome now extends the
  closure to supervised whole-video normality vetoes on cached features, on harm evidence.)

## 10. COMMITTED PIPELINE STATE (end of v21)

Unchanged: v137 + BUE + CBT + CBT-recal + Q-score, ALL DEFAULT-ON; directed scan;
full_coverage_comparative_scan=False. Inert default-OFF code in tree: cnv_veto, cgr_enabled,
cgr_ab_diag, ccr_enabled, tcp_prompt, kfd flags, **snh_veto**. full-290 headline this run
0.17324230 (arm-ON; committed-pipeline equivalent = the idx-restored OFF arm 0.17120316, within
the usual cross-run VLM noise of v160_full 0.17254). Reusable assets banked: the idx= field in
veto markers + the exact idx-keyed restore recipe (the fix for M4 on whole-video veto mechanisms);
analysis_v21_snh_records_verify.py as the template for records-phase FM20 recompute-from-artifacts;
the measured transfer-gap datapoint (94% train -> 38.7% test normal-flag at fixed tau) for any
future train-gated mechanism. This session promoted nothing and deleted nothing.
