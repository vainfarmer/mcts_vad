export const meta = {
  name: 'vad-team-map20-v19',
  description: 'VAD agent team v19 — SNH (Supervised Normality Head): the v17-record-mandated successor to CNV for the owner-directed NORMAL-video false-positive objective. v17 closed CNV on HARM (v171 paired -0.00969, FA -2.6%, 13 GT vetoes; v172 tighten: FA 0%, 6 GT vetoes) and its C-record ruled: any future Normal-FP lever must be a NEW SUPERVISED mechanism that FIRST demonstrates on TRAIN normals that its video-level gate fires on normals at all. SNH = a tiny supervised video-level P(normal) head (logistic regression / tiny MLP) trained on TRAIN-split videos ONLY, over ~15-25 aggregate features computed from the pipeline\'s OWN cached signal families (MIL snippet scores max/topk-mean/std/frac>0.5; fused CLIP category-curve stats incl. max-z and top-decile prominence; CLIP Normal-prompt similarity stats; motion-energy outlier fraction/max-z) — zero VLM calls at train AND eval time, all local compute. TWO-PHASE HONESTY DESIGN: PHASE 0 (offline, NO eval spend) extracts features for ~50 train-normal + ~50 train-anomaly videos (standalone extractor reusing pipeline modules: video preprocessing + CLIP ViT-L + model_mil.pth + frame-diff motion — the v17 council proved this is cheap local compute), trains with cross-validation, and applies the PRE-REGISTERED GO GATE: video-level gate (P(normal) >= tau, tau chosen on TRAIN ONLY at the largest value keeping CV anomaly-flag-rate <= 5%) must flag >= 40% of train NORMAL videos. NO-GO -> write the record closing the video-level-gate direction and STOP (no eval). GO -> implement --snh_veto (default OFF) as a terminal-slot confidence veto REUSING the v171 CNV veto plumbing (confidence := 0.0, detection kept, [SNH-FIRE] conf=old->new marker lines in the analysis_bootstrap_ab.py restore_pre_marker conf-mode format + unconditional [SNH-VID] per video incl. empty/early-return paths), eval ONCE on the full 290 (NO --categories; Normal videos are the subject), adjudicate with the PROVEN paired-log conf restore (exact paired, zero VLM noise). PROMOTION GATES (same as v17, pre-registered): paired mAP delta >= -0.003 AND abs >= 0.16454 AND Normal-video FA-rate relative reduction >= 25% AND <= 1 GT-overlapping (IoU>=0.1) detection vetoed. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON, full-290 ~0.173); cgr/ccr/cnv/kfd all default-OFF and stay so. Concurrency 2. NONE of the 10 banned families; this is judged on the FA-rate objective with mAP only as a non-inferiority constraint — NOT Normal-FP-for-mAP (banned 9).',
  phases: [
    { title: 'FeatExtract', detail: 'offline: standalone extractor (no VLM) computes video-level feature vectors for ~50+50 TRAIN videos, nohup + poll' },
    { title: 'TrainDemo', detail: 'train the head with CV; PRE-REGISTERED GO gate (flag >=40% train normals AND <=5% train anomaly at TRAIN-chosen tau). NO-GO closes the direction with zero eval spend' },
    { title: 'Implement', detail: 'D1 wires --snh_veto (default OFF) reusing the CNV veto plumbing; self-test incl. forced-fire restore proof' },
    { title: 'Eval290', detail: 'one full-290 eval, paired-log conf bootstrap + Normal FA rates + GT-veto harm check' },
    { title: 'Decide', detail: 'promote default-ON iff all four legs pass; else archive-close; summary + ledger + state-of-the-union after three straight closures' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const CONCURRENCY = 2
const N_FULL = 290
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const SNH_FLAG = '--snh_veto'
const FIRE_MARKER = 'SNH-FIRE'
const VID_MARKER = 'SNH-VID'
const V_SNH = 'v176'
const FEAT_FILE = ROOT + '/results/snh_train_features.jsonl'
const MODEL_FILE = CODE + '/model_snh.json'
// pre-registered gates
const MAP_TOL = -0.003
const ABS_FLOOR_290 = 0.16454
const MIN_FA_REL_RED = 0.25
const MAX_GT_VETO = 1
const GO_MIN_NORMAL_FLAG = 0.40   // CV: fraction of train NORMAL videos the gate flags
const GO_MAX_ANOM_FLAG = 0.05     // CV: fraction of train ANOMALY videos the gate may flag

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE: v137 + BUE + CBT + CBT-recal ALL DEFAULT-ON (full-290 ~0.173). cgr/ccr/cnv/kfd_uniform all DEFAULT-OFF (v16 closed relocation; v17 closed CNV on HARM; v18 closed KFD-U — coverage was necessary not sufficient: 0/8 newly-visible GTs converted). Do NOT change any committed default. !!!',
'',
'!!! WHY SNH, AND THE LESSON IT MUST ENCODE !!!',
'v17 CNV failed because its gate was an UNSUPERVISED conjunct of weak signals: 30/41 of its vetoes hit ANOMALY videos (13 overlapped GT) while buying only 2.6% FA reduction. The v17 C-record mandate: a Normal-FP lever must be SUPERVISED and must FIRST demonstrate on TRAIN normals that its video-level gate fires on normals at all. SNH trains P(normal) on video-level labels (train split only) over the SAME signal families, so the veto points are chosen by supervision, not by a hand-built conjunct.',
'',
'!!! FEATURES (pre-registered, ~15-25 dims, ALL from cached/local signal families — zero VLM) !!!',
'Per video, from a STANDALONE extractor that reuses pipeline modules (VideoPreprocessor fps=2 + motion filter; anomaly_expert CLIP ViT-L text-sims incl. the 13 category prompts AND the Normal prompt; model_mil.pth MILAnomalyScorer; frame-diff motion energy):',
'  MIL snippet scores: max, top-3 mean, top-8 mean, std, fraction > 0.5',
'  fused top-1-category curve (cat/(cat+normal) of the video\'s top CLIP category): max-z vs median/MAD, top-decile prominence, mean, std',
'  max over ALL 13 categories of curve max-z',
'  CLIP Normal-prompt similarity curve: mean, min-z, bottom-decile',
'  motion energy: outlier fraction (median+3*MAD), max-z, mean',
'The SAME feature function is used at train-extraction time and inside the pipeline at eval time (shared module — no train/eval skew).',
'',
'!!! THE VETO MECHANICS (reuse v171 CNV plumbing verbatim) !!!',
'Terminal slot (after CBT, final list): if P(normal) >= tau -> veto ALL the video\'s detections: confidence := 0.0, detection KEPT (FM2 count frozen, index-aligned), one ' + FIRE_MARKER + ' line per vetoed detection in EXACTLY the restore_pre_marker conf-mode format (vid + t_start + t_end + conf=old->new — copy the CNV producer grep cnv_veto/' + 'CNV-FIRE in pipeline.py). ' + VID_MARKER + ' unconditional per video incl. 0-detection and early-return-normal paths (same call sites as CNV-VID). Stage-2 per-segment veto from CNV is NOT carried over — v17 showed segment-level vetoes on anomaly videos were the harm vector; SNH is video-level ONLY.',
'',
'!!! GATES (pre-registered) !!!',
'GO gate (offline, TRAIN split, cross-validation): tau = largest threshold with CV anomaly-flag-rate <= ' + (GO_MAX_ANOM_FLAG * 100) + '%; GO iff the gate then flags >= ' + (GO_MIN_NORMAL_FLAG * 100) + '% of train NORMAL videos (and report the ROC-AUC). NO-GO -> the video-level-gate direction is closed with zero eval spend — that is a VALID outcome.',
'PROMOTION gate (full-290, paired-log conf restore): paired mAP delta >= ' + MAP_TOL + ' AND abs >= ' + ABS_FLOOR_290 + ' AND Normal-video FA-rate relative reduction >= ' + (MIN_FA_REL_RED * 100) + '% AND GT-overlapping (IoU>=0.1) vetoes <= ' + MAX_GT_VETO + '.',
'',
'FM DISCIPLINE: FM2 count frozen. FM5 ' + FIRE_MARKER + ' must fire. FM17 ' + VID_MARKER + ' on EVERY path. FM20 grep not claim. FM21 exact marker strings. FM18 absolute mAP. NO TEST-SET TUNING: tau and the head are fixed by TRAIN data only; the 290 run is the first and only time test videos are scored. HARNESS: full 290, NO --categories, --concurrency ' + CONCURRENCY + '.',
'HARD CONSTRAINTS: flag default OFF (byte-identical off); CLI in BOTH run.py and batch_evaluate.py; StructuredOutput EXACTLY ONCE per agent; no Bash > ~150s except stated smokes/extraction (which run via nohup + poll).',
].join('\n')

const featSchema = { type: 'object', properties: { done: { type: 'boolean' }, n_videos: { type: 'number' }, n_normal: { type: 'number' }, n_anomaly: { type: 'number' }, feature_dims: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'n_videos', 'note'], additionalProperties: false }
const demoSchema = { type: 'object', properties: { trained: { type: 'boolean' }, go: { type: 'boolean' }, auc: { type: 'number' }, tau: { type: 'number' }, cv_normal_flag_rate: { type: 'number' }, cv_anomaly_flag_rate: { type: 'number' }, evidence: { type: 'string', description: 'PASTED CV output: per-fold flag rates, AUC, tau derivation — TRAIN data only' }, note: { type: 'string' } }, required: ['trained', 'go', 'evidence', 'note'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'PASTED: (a) ' + VID_MARKER + ' grep == #smoke videos incl empty/early-normal; (b) forced-fire (SNH_FORCE_FIRE=1) restore proof: analysis_bootstrap_ab.py --paired-log --paired-mode conf n_fires>0 matched==n_fires; (c) natural smoke: which videos vetoed + their P(normal); (d) FM2 len unchanged; (e) eval-time features == extractor features on the same video (no train/eval skew proof)' }, marker_fire: { type: 'string' }, marker_vid: { type: 'string' }, eval_flags: { type: 'string' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'instrument_selftest', 'marker_fire', 'marker_vid', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number' }, fires: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'fires', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, n_fires: { type: 'number' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, fa_video_rate_on: { type: 'number' }, fa_video_rate_off: { type: 'number' }, fa_rel_reduction: { type: 'number' }, fp_det_on: { type: 'number' }, fp_det_off: { type: 'number' }, gt_vetoed: { type: 'number' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'n_fires', 'point_delta', 'new_map', 'fa_video_rate_on', 'fa_video_rate_off', 'fa_rel_reduction', 'gt_vetoed', 'note'], additionalProperties: false }
const decideSchema = { type: 'object', properties: { promoted: { type: 'boolean' }, decision: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['promoted', 'decision', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

// ============================ MAIN ============================
log('v19 starting. SNH (Supervised Normality Head): the v17-mandated supervised successor for the Normal-video false-alarm objective. Offline GO gate FIRST (train-split CV: flag >= ' + (GO_MIN_NORMAL_FLAG * 100) + '% normals / <= ' + (GO_MAX_ANOM_FLAG * 100) + '% anomaly) — no eval spend unless supervision demonstrably separates. concurrency ' + CONCURRENCY + '.')

// ---------- PHASE 0a: feature extraction (offline, nohup + poll) ----------
phase('FeatExtract')
const feat = await retryAgent(() => agent([
'You are Agent D0. Build the SNH TRAIN-split feature file — OFFLINE, zero VLM calls, zero eval spend. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'DO:',
'1. Write ' + LOGS + '/analysis_v19_snh_featextract.py: a STANDALONE extractor reusing pipeline modules (follow the precedent of ' + LOGS + '/analysis_v18_pwtc_precheck.py, which already ran pipeline-exact CLIP curves locally on mps). Per video: preprocess (fps=2 + motion filter), CLIP ViT-L frame feats, the 13-category + Normal text-sim curves (pipeline-exact softmax -> cat/(cat+normal)), MIL snippet scores from model_mil.pth, frame-diff motion energy -> emit ONE JSONL line per video with {video, label (normal=1 iff a Training_Normal video, else 0), category, features{...the pre-registered families...}}.',
'2. Sample ~50 train-normal + ~50 train-anomaly videos (stratified across the 13 categories) from /Users/zzl/Movies/ucf-crime (TRAIN split only — find the train manifest/dir layout the repo already uses; if the dataset dir is test-only, say so loudly and use the training-videos location you find).',
'3. Launch it via nohup (it is ~1-2h of local mps compute), then POLL with sleep 140 Bash calls until ' + FEAT_FILE + ' has >= 90 lines (or the log shows completion). Report any per-video failures.',
'4. Sanity-paste 2 lines (one normal, one anomaly) showing the feature vector.',
'Return done, n_videos, n_normal, n_anomaly, feature_dims, note.',
].join('\n'), { schema: featSchema, label: 'D0-featextract', phase: 'FeatExtract' }))

let demo = null, impl = null, run290 = null, boot = null
const featOk = !!(feat && feat.done && feat.n_videos >= 80)
log('FeatExtract: ' + (featOk ? ('OK n=' + feat.n_videos + ' (normal=' + feat.n_normal + ' anomaly=' + feat.n_anomaly + ' dims=' + feat.feature_dims + ')') : 'FAILED/insufficient -> ' + (feat ? feat.note : 'null')))

// ---------- PHASE 0b: train + CV demo + GO gate ----------
phase('TrainDemo')
if (featOk) {
  demo = await retryAgent(() => agent([
'You are Agent D1. Train the SNH head on TRAIN-split features and apply the PRE-REGISTERED GO gate. No test data anywhere near this. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'DO:',
'1. Write ' + LOGS + '/analysis_v19_snh_train.py: load ' + FEAT_FILE + ', standardize features, train logistic regression (L2, balanced option) with 5-fold STRATIFIED cross-validation. Report CV ROC-AUC.',
'2. tau derivation (TRAIN only): the largest threshold whose CV anomaly-flag-rate (P(normal) >= tau on ANOMALY videos) <= ' + GO_MAX_ANOM_FLAG + '. Then the CV normal-flag-rate at that tau.',
'3. GO iff cv_normal_flag_rate >= ' + GO_MIN_NORMAL_FLAG + ' AND cv_anomaly_flag_rate <= ' + GO_MAX_ANOM_FLAG + '. Paste the per-fold numbers and the tau derivation.',
'4. If GO: refit on ALL train features and save ' + MODEL_FILE + ' as plain JSON {weights, bias, feature_order, scaler_mean, scaler_scale, tau} (dependency-free load at eval time).',
'5. Also paste the top-5 features by |weight| — the record wants to know WHAT supervision found (compare with CNV\'s failed hand-conjunct: MIL topk / curve-z / motion-fraction).',
'Return trained, go, auc, tau, cv_normal_flag_rate, cv_anomaly_flag_rate, evidence (pasted), note.',
  ].join('\n'), { schema: demoSchema, label: 'D1-traindemo', phase: 'TrainDemo' }))
}
const go = !!(demo && demo.trained && demo.go)
log('TrainDemo: ' + (demo ? ('trained=' + demo.trained + ' go=' + demo.go + (demo.auc != null ? (' AUC=' + demo.auc.toFixed(3) + ' tau=' + (demo.tau != null ? demo.tau : '?') + ' normFlag=' + (demo.cv_normal_flag_rate != null ? (demo.cv_normal_flag_rate * 100).toFixed(1) + '%' : '?') + ' anomFlag=' + (demo.cv_anomaly_flag_rate != null ? (demo.cv_anomaly_flag_rate * 100).toFixed(1) + '%' : '?')) : '')) : 'skipped (no features)'))

// ---------- PHASE 2: implement (only if GO) ----------
phase('Implement')
if (go) {
  impl = await retryAgent(() => agent([
'You are Agent D2. Wire SNH into the pipeline as --snh_veto (default OFF), reusing the v171 CNV veto plumbing. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'THE TRAINED HEAD: ' + MODEL_FILE + ' (JSON: weights, bias, feature_order, scaler stats, tau=' + (demo && demo.tau != null ? demo.tau : '?') + '). The offline GO gate passed: AUC=' + (demo && demo.auc != null ? demo.auc.toFixed(3) : '?') + ', CV normal-flag=' + (demo && demo.cv_normal_flag_rate != null ? (demo.cv_normal_flag_rate * 100).toFixed(1) + '%' : '?') + ' at anomaly-flag=' + (demo && demo.cv_anomaly_flag_rate != null ? (demo.cv_anomaly_flag_rate * 100).toFixed(1) + '%' : '?') + '.',
'DO:',
'1. Back up every file you touch to ' + ROOT + '/.qoder/backups/' + V_SNH + '/ FIRST; ls it.',
'2. Shared feature module (e.g. video_anomaly_detection/snh_features.py): ONE function computing the pre-registered feature vector from the pipeline\'s in-run cached signals (_mil_scores, the fused category curves, the Normal-prompt sim, motion energy). It MUST be the same computation the offline extractor used — import shared helpers, do not reimplement divergently.',
'3. Terminal slot (the CNV veto site — grep cnv_veto in pipeline.py): when snh_veto on, compute P(normal) from ' + MODEL_FILE + ' (dependency-free JSON load, cached); if >= tau veto ALL detections exactly like CNV (confidence := 0.0, kept, FM2 assert len unchanged), one ' + FIRE_MARKER + ' line per vetoed detection in the restore_pre_marker conf-mode format, ' + VID_MARKER + ' unconditional per video on EVERY path (same sites as CNV-VID) carrying P(normal) and the fire decision.',
'4. Config snh_veto default False + CLI --snh_veto in BOTH run.py and batch_evaluate.py. Dev-only SNH_FORCE_FIRE=1 env forcing the gate (for the restore proof). Flag-off byte-identical. Bump pipeline.py header to ' + V_SNH + '. Do NOT touch any committed default or models.py.',
'',
'HARD SELF-TEST (paste ACTUAL output):',
'  SMOKE A (natural, ~13-14 videos incl. Normal-category): ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V_SNH + '_selftest --max_per_category 1 --concurrency 1 ' + MECH_BASE + ' ' + SNH_FLAG + '  -> ' + VID_MARKER + ' grep == #videos; report which videos vetoed + their P(normal); FM2 len check vs flag-off.',
'  SMOKE B (forced): SNH_FORCE_FIRE=1 on 2 videos -> ' + FIRE_MARKER + ' lines AND ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-log <log> --marker ' + FIRE_MARKER + ' --paired-mode conf reports n_fires>0, matched==n_fires.',
'  SKEW CHECK: on 1-2 videos, the eval-time feature vector == the standalone extractor\'s vector for the same video (tolerance 1e-6) — paste both.',
'VALIDATE: py_compile; import MultiAnomalyVAD; --snh_veto in BOTH --help; ScoringConfig() defaults unchanged (bue/cbt/recal True; cgr/ccr/cnv/kfd False; snh_veto False).',
'Return implemented, selftest_passed, instrument_selftest (pasted), marker_fire, marker_vid, eval_flags, files_changed, validation, notes.',
].join('\n'), { schema: implSchema, label: 'D2-implement-' + V_SNH, phase: 'Implement' }))
} else {
  log('Implement: skipped — ' + (demo && demo.trained ? 'GO gate FAILED (supervision does not separate at the pre-registered bar)' : 'no trained head'))
}

// ---------- PHASE 3: Eval290 (only if implemented) ----------
phase('Eval290')
const snhOk = !!(impl && impl.implemented && impl.selftest_passed && impl.eval_flags)
if (snhOk) {
  const launch = await retryAgent(() => agent([
'You are Agent D-launcher for the SNH full-290 eval. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + ROOT + '/results/batch_eval_' + V_SNH + '_snh290 ; log=<dir>.log ; report=<dir>/evaluation_report.json',
'REQUIRED FLAGS (verbatim, NO --categories — Normal videos are the subject): ' + impl.eval_flags,
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. Running for this V_tag -> adopt if flags match, else kill+relaunch. 3. report exists AND log shows the flags AND ' + VID_MARKER + ' lines -> launched=true note="already complete". 4. A DIFFERENT eval running -> sleep 140 (up to 3x) then blocked.',
'5. Else write <dir>/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + ROOT + '/results/batch_eval_' + V_SNH + '_snh290' + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + '/results/batch_eval_' + V_SNH + '_snh290' + " --max_per_category 999 --concurrency " + CONCURRENCY + " " + impl.eval_flags + " > " + ROOT + '/results/batch_eval_' + V_SNH + '_snh290.log' + " 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 log. Return launched, note.',
  ].join('\n'), { schema: launchSchema, label: 'D-launch-snh290', phase: 'Eval290' }))
  if (launch && launch.launched) {
    const dir = ROOT + '/results/batch_eval_' + V_SNH + '_snh290'
    for (let w = 0; w < 130; w++) {
      const st = await retryAgent(() => agent([
'You are Agent D-watcher for the SNH 290 eval (expect ' + N_FULL + '). SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'1. If ' + dir + '/evaluation_report.json exists -> done=true; read mAP/AP@0.3/0.5/0.7/num_videos (' + PY + ' -c "import json;d=json.load(open(\'' + dir + '/evaluation_report.json\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'))"); num_videos MUST be ' + N_FULL + '. markers: VC=grep -cF "' + VID_MARKER + '" ' + dir + '.log; FC=grep -cF "' + FIRE_MARKER + '" ' + dir + '.log -> marker_count=VC (~290, FM17), fires=FC (>0 = mechanism fired, FM5).',
'2. Else pgrep -f "batch_evaluate.*' + V_SNH + '_snh290". Alive -> sleep 140 (up to 2x), done=false note="running <n>/290" (n = JSON count). STALLED >30min -> kill + relaunch same nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead no report -> tail -30 log; TRANSIENT -> relaunch; CODE BUG -> crashed=true + traceback.',
'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, marker_count, fires, note.',
      ].join('\n'), { schema: watchSchema, label: 'D-watch-snh290-' + w, phase: 'Eval290' }))
      if (!st) continue
      run290 = st
      if (st.done || st.crashed) break
    }
  } else {
    run290 = { done: false, crashed: true, mAP: -1, marker_count: -1, fires: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  }
  log('Eval290: done=' + run290.done + ' crashed=' + run290.crashed + ' mAP=' + run290.mAP + ' n=' + (run290.num_videos != null ? run290.num_videos : '?') + ' vidMarker=' + run290.marker_count + ' fires=' + run290.fires)
  if (run290.done && !run290.crashed && run290.marker_count > 0) {
    boot = await retryAgent(() => agent([
'You are Agent D-adjudicator for the SNH paired A/B on the FULL 290. arm-ON = the run as-is (vetoed detections conf 0.0); arm-OFF = committed, restored EXACTLY from ' + FIRE_MARKER + ' conf=old->new lines. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'STEP 1 (FM5/FM20): FC=$(grep -cF "' + FIRE_MARKER + '" ' + ROOT + '/results/batch_eval_' + V_SNH + '_snh290.log); VC=$(grep -cF "' + VID_MARKER + '" ' + ROOT + '/results/batch_eval_' + V_SNH + '_snh290.log). n_fires=FC. VC~290 else ran=false STOP. FC==0 is a TRUE zero-effect read (gate never fired on test) — still ran=true, continue.',
'STEP 2: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_' + V_SNH + '_snh290 --paired-log ' + ROOT + '/results/batch_eval_' + V_SNH + '_snh290.log --marker ' + FIRE_MARKER + ' --paired-mode conf --min-delta ' + MAP_TOL + ' --min-p 0.9 -> point_delta, ci90, p_gt0, new_map, ref_map.',
'STEP 3 (THE OBJECTIVE): Normal-category videos — fa_video_rate_on/off (>=1 surviving conf>0 detection; arm-OFF = restored confs), fa_rel_reduction=(off-on)/off, fp_det_on/off (mean surviving dets per Normal video). Paste code path + numbers.',
'STEP 4 (harm): gt_vetoed = vetoed detections with GT IoU>=0.1; also vetoed-on-anomaly-videos by category, and each vetoed video\'s P(normal).',
'STEP 5: per-category AP deltas (flag |d|>0.002) into per_category.',
'Report ran, n_fires, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, fa_video_rate_on, fa_video_rate_off, fa_rel_reduction, fp_det_on, fp_det_off, gt_vetoed, per_category, note (VERDICT verbatim).',
    ].join('\n'), { schema: bootSchema, label: 'D-boot-snh290', phase: 'Eval290' }))
  }
} else {
  log('Eval290: skipped')
}
const g = (boot && boot.ran) ? {
  promote: boot.point_delta >= MAP_TOL && boot.new_map >= ABS_FLOOR_290 && boot.fa_rel_reduction >= MIN_FA_REL_RED && boot.gt_vetoed <= MAX_GT_VETO,
  delta: boot.point_delta, abs: boot.new_map, fa: boot.fa_rel_reduction, gt: boot.gt_vetoed,
} : { promote: false }
log('SNH gate: ' + (boot && boot.ran ? ('delta=' + boot.point_delta.toFixed(5) + ' abs=' + boot.new_map.toFixed(5) + ' FA ' + boot.fa_video_rate_off.toFixed(3) + '->' + boot.fa_video_rate_on.toFixed(3) + ' (rel -' + (boot.fa_rel_reduction * 100).toFixed(1) + '%) gt_vetoed=' + boot.gt_vetoed + ' -> ' + (g.promote ? 'PROMOTE' : 'FAIL')) : 'no adjudication'))

// ---------- PHASE 4: decide ----------
phase('Decide')
const decision = await retryAgent(() => agent([
'You are Agent C. Write the v19 decision + session record. Scrupulously honest. Plain text.',
'', CONTEXT, '',
'EVIDENCE:',
'  FeatExtract: ' + JSON.stringify(feat),
'  TrainDemo (offline GO gate): ' + JSON.stringify(demo && { trained: demo.trained, go: demo.go, auc: demo.auc, tau: demo.tau, cv_normal_flag_rate: demo.cv_normal_flag_rate, cv_anomaly_flag_rate: demo.cv_anomaly_flag_rate }),
'  SNH 290: ' + JSON.stringify(boot),
'  computed promote=' + g.promote,
'',
'DO:',
 g.promote
 ? ('1. PROMOTE. Back up config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/snh_promote/ ; set snh_veto default True in ScoringConfig, from_args default, and set_defaults in BOTH entrypoints (add --no_snh_veto opt-out). VALIDATE + paste: py_compile; snh_veto True; bue/cbt/recal True; cgr/ccr/cnv/kfd False; import MultiAnomalyVAD.')
 : ('1. DO NOT PROMOTE — confirm snh_veto stays default False (paste defaults; committed pipeline untouched). Close honestly on the RIGHT branch: (a) if the offline GO gate failed -> the video-level-gate DIRECTION is closed: state the CV numbers and that supervision on these signal families cannot separate train normals from anomalies at the pre-registered bar — future Normal-FP work needs a NEW SIGNAL FAMILY (e.g. VideoMAE features, a frame-level trained head), not another gate on these families; (b) if GO passed but the 290 gate failed -> SNH is MEASURED-CLOSED on the test harness: state delta/CI, FA rates, gt_vetoed, and whether train->test transfer or the veto mechanics were the failure.'),
'2. date +%Y%m%d ; write ' + LOGS + '/summary_v19_' + V_SNH + '_<YYYYMMDD>.md covering: the v17 mandate; features + what supervision found (top weights vs CNV\'s failed conjunct); the offline CV demo + GO verdict (TRAIN numbers only); if it ran, the 290 numbers (delta/CI/P, FA rates, FP dets/video, gt_vetoed, per-category, vetoed videos\' P(normal)); the decision; updated CLOSED list.',
'3. STATE OF THE UNION (this is the 3rd straight measured closure — v16 relocation, v17 CNV, v18 KFD-U, plus v19 if failed): append to the summary a frank paragraph on what the closures collectively imply about the remaining headroom at the 0.173 plateau, which families still have ANY unrefuted lever (scan-prompt/context quality given v18\'s "VLM saw but did not act"; a frame-level trained head; MIL-v6), and what evidence would justify each.',
'4. APPEND a ledger row to ' + LOGS + '/session_v97_ledger.md for ' + V_SNH + ' (use -1 for absent values; if the eval never ran, verdict = the offline outcome). Do NOT rewrite existing rows.',
'5. Reply with a 12-line recap leading with the headline (offline GO result +, if ran, FA-rate change and paired mAP delta).',
'Return promoted, decision, validation (pasted), note.',
].join('\n'), { schema: decideSchema, label: 'C-decide-snh', phase: 'Decide' }))

log('v19 complete. SNH: ' + (g.promote ? 'PROMOTED' : 'NOT promoted') + ' | offline go=' + go + (demo && demo.auc != null ? (' AUC=' + demo.auc.toFixed(3)) : '') + ' | 290: ' + (boot && boot.ran ? ('delta=' + boot.point_delta.toFixed(5) + ' FA->' + boot.fa_video_rate_on.toFixed(3)) : 'n/a'))
return {
  feat, demo: demo && { trained: demo.trained, go: demo.go, auc: demo.auc, tau: demo.tau, cv_normal_flag_rate: demo.cv_normal_flag_rate, cv_anomaly_flag_rate: demo.cv_anomaly_flag_rate },
  go, snh290: boot, promote: g.promote,
  decision: decision ? decision.decision : null,
}
