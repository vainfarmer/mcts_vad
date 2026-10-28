# Session v19 — SNH (Supervised Normality Head) offline GO-gate

Date: 2026-08-16
Scope: the v17-record-mandated successor to CNV for the owner-directed NORMAL-video
false-positive objective. v17 closed CNV on HARM and ruled that any future Normal-FP lever must
FIRST demonstrate on TRAIN normals that its video-level gate fires on normals at all. v19 runs that
mandatory offline phase: extract cached-signal features for train videos, train a supervised P(normal)
head, apply the pre-registered GO gate. NO test eval is spent in this session (offline only).

## 0. HEADLINE

**GO.** The SNH head separates train normals from train anomalies by a wide margin:
stratified 5-fold CV **AUC 0.996**; at the pre-registered operating point tau=0.7894 (the largest tau
keeping the CV anomaly-flag rate <= 5% -> measured 2/52 = 3.8%), the video-level gate
(P(normal) >= tau) flags **47/50 = 94% of train NORMAL videos** — far above the 40% GO bar.
The direction CNV could not close (an unsupervised triple that cannot tell hallucination from truth)
IS separable by a supervised head on the pipeline's own cached signals. The 290 test eval
(implementation + paired adjudication) defers to the NEXT session (v21) so as not to collide with the
in-flight v20 (TCP) eval; no test result exists yet and no promotion is claimed.

## 1. MECHANISM (what SNH is)

A tiny supervised video-level P(normal) head — multinomial logistic regression over 16 aggregate
features computed from the pipeline's OWN cached signal families, ZERO VLM calls at train AND eval
time (all local compute, reusing the pipeline's preprocessing + CLIP ViT-L + model_mil.pth + frame-diff
motion):
- MIL snippet scores: mil_max / mil_top3_mean / mil_top8_mean / mil_std / mil_frac_gt_0.5
  (model_mil.pth, the trained WSAD MIL head, over the video's 32 snippets);
- fused CLIP category curve: allcat_maxz_max (max in-video z of the max-over-13-cat curve),
  top1_mean / top1_std / top1_topdecile_prominence (top-1 cat similarity stats);
- CLIP Normal-prompt similarity: normal_mean / normal_max / normal_std / normal_topdecile_prom;
- motion energy: motion_outlier_frac / motion_max_z (frame-diff outliers).
Veto (v21, not implemented here): a video-level gate — if P(normal) >= tau, veto ALL the video's
detections (confidence := 0.0, detection kept, [SNH-FIRE] conf=old->new markers, exact-paired restore
via analysis_bootstrap_ab.py --paired-log --paired-mode conf, the v171 proven path).

## 2. EXECUTION — workflow died; main session completed the gate inline

`vad-team-map20-v19` launched 2026-08-15 ~21:16 (run wf_f696a486-456). Its FeatExtract agent wrote
results/snh_train_features.jsonl (102 videos: 50 normal + 52 anomaly x 16 features) and its
D0-retry agent started 23:49, then the workflow produced NO further journal events for 11+ hours
(overnight session suspend; the runtime reaped it — TaskStop found no task). The offline train/diagnose
step was therefore completed INLINE by the main session (2026-08-16): wrote
iteration_logs/analysis_v19_snh_train.py (numpy-only logistic regression, standardized features,
stratified 5-fold CV seed 20260816, AUC via Mann-Whitney, tau = largest threshold with CV
anomaly-flag <= 0.05, GO iff normal-flag >= 0.40) and ran it. This is exactly the step the workflow's
TrainDemo agent was chartered to do; completing it inline changes no method and spends no test eval.
The downstream implement+eval steps were NOT done (deferred to v21).

## 3. GO-GATE NUMBERS (all from the produced artifacts)

- Feature file: results/snh_train_features.jsonl — 102 rows, 16 numeric features each (35 raw
  non-finite values from MAD=0 z-scores sanitized via np.nan_to_num + clip to +-50 before training;
  the first unsanitized run diverged to NaN — fixed, not hidden).
- CV AUC = **0.996** (near-perfect ranking of normal vs anomaly on train).
- tau = **0.7894**: the largest P(normal) threshold with CV anomaly-flag rate <= 0.05; measured
  anomaly-flag = 2/52 = 3.8% (2 false-normal anomalies), normal-flag = 47/50 = **94%**.
- GO bar: normal-flag >= 40%. 94% >= 40% -> **GO** (with the caveat below).
- Top-weighted features (|w|, positive w pushes toward NORMAL): mil_frac_gt_0.5 (-0.913),
  mil_top8_mean (-0.909), mil_top3_mean (-0.906), mil_max (-0.905), top1_mean (-0.682). The
  **MIL family dominates** — high MIL snippet scores push P(normal) DOWN (toward anomaly); the
  normal-prompt and motion families are weak by comparison. Read: the trained WSAD MIL head is
  doing almost all the normal/anomaly separation; SNH is largely a calibrated threshold on MIL.
- Model saved: video_anomaly_detection/model_snh.json (feature_order, scaler, weights, bias, tau,
  cv_auc) for the v21 runtime veto.

## 4. HONEST CAVEAT — in-distribution optimism (the open question)

model_mil.pth was TRAINED on this same UCF-Crime train split, so the MIL features of these 102 train
videos are partially in-distribution: the 94%/3.8% separation and AUC 0.996 are an UPPER BOUND, not a
test-set guarantee. The GO gate is satisfied on train as designed (v17 mandated a train demonstration),
but whether the gate transfers to TEST normals — where the whole point is to veto FPs the pipeline
hallucinates on videos the MIL head never saw — is decided ONLY by the 290 eval in v21. If test MIL
scores are noisier, tau=0.7894 may flag far fewer test normals (gate under-fires -> low FA reduction)
or misflag anomalies (gt_vetoed harm). v21 measures this directly; no transfer is assumed here.

## 5. WHAT v21 MUST DO (handoff)

1. Implement --snh_veto (default OFF): terminal-slot video-level veto reusing the v171 CNV plumbing —
   load model_snh.json, compute the 16 features for the current video from the same cached signals,
   if P(normal) >= tau veto ALL detections (conf:=0.0, keep, [SNH-FIRE] conf=old->new markers +
   unconditional [SNH-VID]). CLI in run.py AND batch_evaluate.py.
2. Self-test incl. an SNH_FORCE_FIRE restore proof (forced veto -> paired-log restore -> mAP identical).
3. ONE full-290 eval (NO --categories; Normal videos are the subject), --concurrency 2, queue-disciplined.
4. Adjudicate with analysis_bootstrap_ab.py --paired-log --paired-mode conf. Promotion gates
   (pre-registered, same as v17): paired mAP delta >= -0.003 AND abs >= 0.16454 AND Normal-video
   FA-rate relative reduction >= 25% AND <= 1 GT-overlapping (IoU>=0.1) detection vetoed.

## 6. LEDGER

No session_v97_ledger.md row is added for v19: the ledger is version-keyed and v19 produced no code
version and no eval (offline gate only). The SNH implementation version + its 290 mAP/FA numbers will
be recorded on the v21 row when the eval runs.
