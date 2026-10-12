export const meta = {
  name: 'vad-team-map20-v22',
  description: 'VAD agent team v22 — v180 SNH-P (MIL-peak-PROTECTED SNH veto): the targeted structural fix for the v179 G4 failure. v179 SNH (video-level P(normal)>=tau veto) PASSED the FA objective hugely (Normal-video FA 0.507->0.167, -67.1%) and held abs mAP (0.17324>=0.16454) but FAILED G4: 5 GT-overlapping detections vetoed (limit <=1) — the whole-video veto is too blunt on the ~3.5% of anomaly videos where the head misfires. SNH-P adds ONE pre-registered structural rule: when the video-level gate fires, veto every detection EXCEPT any detection whose [t_start,t_end] covers the video\'s MIL argmax snippet time (t_peak = (argmax(mil_scores)+0.5)/32 * duration — the single strongest snippet-level evidence of anomaly; model_mil.pth). Protected detections get NO [SNH-FIRE] line, so the PROVEN paired-log conf restore (--marker SNH-FIRE --paired-mode conf) works UNCHANGED. Protected detections are logged as [SNHP-SKIP] (vid, window, t_peak, reason) for observability. PHASE 0 (offline, ZERO eval spend): using the v179 artifacts (results/batch_eval_v179_snh290 + .log), for every one of the 92 [SNH-FIRE] lines determine whether the fired window covers its video\'s MIL argmax (recompute mil_scores locally with the analysis_v19_snh_featextract.py machinery — CLIP+MIL local, NO VLM API). PRE-REGISTERED GO GATE: (a) >= 4 of the 5 v179 GT-harm detections (RoadAccidents128 3.0-8.0 IoU0.733; Explosion025 10.0-16.5 IoU0.511; Shooting048 46.0-52.0 IoU0.429; Shoplifting022 45.5-55.0 IoU0.421; RoadAccidents002 9.0-11.5 IoU0.286) would have been PROTECTED (window covers MIL argmax), AND (b) the estimated Normal-video FA-rate relative reduction under protection stays >= 25% (recompute per Normal video: would any detection survive the veto via protection? that resurrects an FA video). NO-GO -> write the record closing the protection direction and STOP (no eval). GO -> implement (flag snh_protect default OFF, active only with --snh_veto; marker [SNHP-SKIP]; CLI in run.py+batch_evaluate.py), self-test (OFF byte-identical; ON smoke; SNH_FORCE_FIRE smoke showing the MIL-argmax-covering detection survives), ONE queue-disciplined full-290 eval (--snh_veto --snh_protect), paired-log adjudication, SAME 4 pre-registered gates as v179 (paired delta >= -0.003, abs >= 0.16454, FA rel reduction >= 25%, GT-harm <= 1), promote-or-close + records. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON); cgr/ccr/cnv/kfd/tcp/snh all default-OFF and stay so (snh_veto itself stays OFF regardless; only the snh_veto+snh_protect COMBINATION is promoted if all 4 gates pass). Concurrency 2, never two evals (16 GB swap-thrash). NOT a banned family: judged on the FA-rate objective with mAP as non-inferiority constraint; the protection rule is structural (no numeric test-fitting — tau is NOT touched).',
  phases: [
    { title: 'Precheck', detail: 'offline MIL-argmax coverage of the 92 v179 fires -> GO/NO-GO' },
    { title: 'Implement', detail: 'v180 --snh_protect + self-test' },
    { title: 'Eval290', detail: 'launcher + watcher loop, full-290 --snh_veto --snh_protect' },
    { title: 'Adjudicate', detail: 'paired-log conf restore + 4 gates' },
    { title: 'Decide', detail: 'promote-or-close + summary + ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const PKG = ROOT + '/video_anomaly_detection'
const ITER = PKG + '/iteration_logs'
const PY = 'python3.14'
const V = 'v180'
const V179_DIR = ROOT + '/results/batch_eval_v179_snh290'
const V179_LOG = V179_DIR + '.log'
const TAG = 'snhp290'
const DIR = ROOT + '/results/batch_eval_' + V + '_' + TAG
const LOG = DIR + '.log'
const REP = DIR + '/evaluation_report.json'
const N_FULL = 290

const RULES = [
  'NEVER edit the evaluator, GT labels, or ' + ITER + '/analysis_bootstrap_ab.py.',
  'NEVER run two batch_evaluate processes concurrently (16 GB swap-thrash).',
  'NO Bash command may run > 150s in the foreground — the runtime kills stalled agents at 180s. Long work = nohup background + short polls.',
  'grep claims must be GREPPED in produced artifacts, never asserted (FM20/FM21).',
  'Do NOT git-commit anything. StructuredOutput EXACTLY ONCE per agent (only where a schema is given).',
].join('\n')

const preSchema = { type: 'object', properties: {
  go: { type: 'boolean' },
  harm_protected: { type: 'integer', description: 'of the 5 v179 GT-harm dets, how many cover their video MIL argmax' },
  fires_total: { type: 'integer' }, fires_protected: { type: 'integer' },
  normal_fa_videos_v179: { type: 'integer' }, normal_fa_videos_est: { type: 'integer' },
  fa_rel_est: { type: 'number' },
  per_harm: { type: 'string', description: 'one line per harm det: vid window iou mil_argmax_t covered?' },
  note: { type: 'string' },
}, required: ['go', 'harm_protected', 'fires_total', 'fires_protected', 'fa_rel_est', 'per_harm', 'note'], additionalProperties: false }

const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, vid_count: { type: 'number' }, fire_count: { type: 'number' }, skip_count: { type: 'number' }, err_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'vid_count', 'fire_count', 'skip_count', 'err_count', 'note'], additionalProperties: false }
const adjSchema = { type: 'object', properties: {
  verdict: { type: 'string', enum: ['PROMOTE', 'CLOSE'] },
  paired_delta: { type: 'number' }, abs_map_on: { type: 'number' }, abs_map_off: { type: 'number' },
  ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_delta_pos: { type: 'number' },
  fa_rate_on: { type: 'number' }, fa_rate_off: { type: 'number' }, fa_rel_reduction: { type: 'number' },
  gt_vetoed: { type: 'integer' }, n_fires: { type: 'integer' }, matched: { type: 'integer' },
  n_skips: { type: 'integer' }, vid_count: { type: 'integer' }, gate_table: { type: 'string' }, summary: { type: 'string' },
}, required: ['verdict', 'paired_delta', 'abs_map_on', 'abs_map_off', 'ci90_low', 'ci90_high', 'p_delta_pos', 'fa_rate_on', 'fa_rate_off', 'fa_rel_reduction', 'gt_vetoed', 'n_fires', 'matched', 'n_skips', 'vid_count', 'gate_table', 'summary'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

// ============================ PHASE 0 — OFFLINE PRECHECK ============================
phase('Precheck')
log('v22 phase 0: offline MIL-argmax coverage precheck on the 92 v179 fires')

const PRE_PROMPT = [
  'You are Agent D-precheck for v180 SNH-P. OFFLINE ONLY — zero eval spend, NO VLM API, NO batch_evaluate launches. ' + RULES,
  'GOAL: decide GO/NO-GO for the MIL-peak protection rule using ONLY the v179 artifacts + local CLIP/MIL recomputation.',
  'Context: ' + V179_LOG + ' carries 92 [SNH-FIRE] lines (vid=<stem> idx cat t=<ts>-<te> conf=...) — detections the v179 video-level SNH veto zeroed. The 5 GT-harm detections (G4 failure): RoadAccidents128 3.00-8.00 IoU0.733; Explosion025 10.00-16.50 IoU0.511; Shooting048 46.00-52.00 IoU0.429; Shoplifting022 45.50-55.00 IoU0.421; RoadAccidents002 9.00-11.50 IoU0.286.',
  'PROTECTION RULE (pre-registered): a fired detection is PROTECTED (would survive) iff its [t_start,t_end] covers its video\'s MIL argmax snippet time t_peak=(argmax(mil_scores)+0.5)/32*duration.',
  'STEPS:',
  '1. Parse all 92 [SNH-FIRE] lines from ' + V179_LOG + ' -> (vid, t_start, t_end). Also note each video\'s [SNH-VID] gate=1 lines.',
  '2. For every DISTINCT fired video, recompute mil_scores LOCALLY: reuse ' + ITER + '/analysis_v19_snh_featextract.py machinery (it loads CLIP ViT-L + model_mil.pth and computes the 32-snippet MIL scores; run it as a library or adapt it into ' + ITER + '/analysis_v180_snhp_precheck.py — a NEW script, do not edit the v19 one). Cache results to ' + ROOT + '/results/v180_mil_argmax.json. This is local GPU/CPU compute, ~1-2 min/video; if the total distinct-video count is large (>60), you may run it as a nohup background job and poll (short Bash calls only).',
  '3. For each fired detection: covered = (t_start <= t_peak <= t_end).',
  '4. GO GATE (pre-registered): (a) >= 4 of the 5 GT-harm dets covered; AND (b) estimated Normal-video FA relative reduction >= 25%: for each Normal video that v179 made detection-free (all its dets fired), check whether ANY of its fired dets is protected (=> FA video resurrects). est_fa_videos = v179 FA-on count (25) + resurrected; fa_rel_est = (76 - est_fa_videos)/76 using the v179 numbers (FA off=76/150 -> on=25/150).',
  '5. Report go, harm_protected, fires_total, fires_protected, normal_fa_videos_v179 (=25), normal_fa_videos_est, fa_rel_est, per_harm (5 lines), note (method + any caveats, e.g. mil_scores all-zero videos where argmax is undefined -> treat as NOT protected and say so).',
].join('\n')

const pre = await retryAgent(() => agent(PRE_PROMPT, { schema: preSchema, label: 'D-precheck-' + V, phase: 'Precheck' }))
log('v22 precheck: go=' + (pre && pre.go) + ' harm_protected=' + (pre && pre.harm_protected) + '/5 fa_rel_est=' + (pre && pre.fa_rel_est))

if (!pre || !pre.go) {
  log('v22 NO-GO: protection rule fails its offline gate — closing direction without eval spend. precheck=' + JSON.stringify(pre))
} else {

// ============================ PHASE 1 — IMPLEMENT ============================
phase('Implement')

const IMPL_PROMPT = [
  'You are Agent D-implementer for v180 SNH-P. The offline precheck is GO (' + pre.harm_protected + '/5 harm dets protected, est FA rel reduction ' + pre.fa_rel_est.toFixed(3) + '). ' + RULES,
  'The v179 SNH implementation is IN THE TREE (default OFF): pipeline.py _snh_veto (~L3473) + _snh_p_normal (~L3369) + _snh_load_model (~L1144), call sites ~L1818/~L1980/~L2355, config snh_veto (~config.py L230), CLI in run.py+batch_evaluate.py. READ THEM FIRST.',
  'TASK — add flag snh_protect (default OFF), version tag v180, a MODIFIER active only when snh_veto is also on:',
  '1. BACKUP: cp pipeline.py config.py run.py batch_evaluate.py into ' + ROOT + '/.qoder/backups/' + V + '/ .',
  '2. config.py: snh_protect: bool = False next to snh_veto, comment: v180 MIL-peak protection; v22 GO record.',
  '3. pipeline.py _snh_veto: when the video-level gate fires AND self.config.scoring.snh_protect, compute t_peak = (argmax(mil_scores)+0.5)/len(mil_scores)*duration using the SAME video_data[\'mil_scores\'] array _snh_p_normal already reads (None/empty/all-equal -> NO protection, log why in [SNH-VID]); PROTECT (skip veto, keep confidence, NO [SNH-FIRE] line) any detection with t_start <= t_peak <= t_end, logging one [SNHP-SKIP] line per protected detection: vid=<stem> idx t=<ts:.2f>-<te:.2f> t_peak=<:.2f> conf=<:.4f>. All other behaviour (markers, counts, FM2 assert) unchanged. The [SNH-VID] line gains protected=<k>.',
  '4. CLI: --snh_protect store_true dest snh_protect set_defaults(snh_protect=False) in run.py AND batch_evaluate.py.',
  '5. VALIDATE: py_compile; ScoringConfig()/from_args(argparse.Namespace()) -> snh_protect False, snh_veto False, bue/cbt/recal True, cnv/cgr/ccr False; both --help show the flag.',
  '6. SELF-TEST SMOKES (queue-disciplined: pgrep -f batch_evaluate first; cwd=' + ROOT + '; the v179 smoke commands are the model):',
  '   a. OFF: --categories Normal,Arson --max_per_category 1, NO flags -> grep -c "SNH" log == 0.',
  '   b. ON without protect (--snh_veto only): same as v179 smoke — SNH-VID==2.',
  '   c. FORCE + protect: SNH_FORCE_FIRE=1 --snh_veto --snh_protect -> if the Arson video\'s detection covers its MIL argmax it MUST survive with a [SNHP-SKIP] line and no [SNH-FIRE] for it (check the log; either outcome is acceptable but MUST be consistent with the printed t_peak); grep -cF "[SNH-VID]" == 2.',
  '   Paste all commands + greps + the [SNHP-SKIP]/[SNH-VID] lines. End with a verdict line: READY or NOT-READY (+reasons).',
].join('\n')

const impl = await retryAgent(() => agent(IMPL_PROMPT, { label: 'D1-implement-' + V, phase: 'Implement' }))
if (!impl || /NOT-READY/.test(impl)) {
  log('v22 ABORT: implement NOT-READY (FM5 discipline — no eval spent). impl=' + (impl || 'null').slice(0, 400))
} else {

// ============================ PHASE 2 — EVAL290 ============================
phase('Eval290')

const LAUNCH_PROMPT = [
  'You are Agent D-launcher for the v180 SNH-P full-290 eval. Ensure the eval is running, then return IMMEDIATELY. SHORT. ' + RULES,
  '0. QUEUE: pgrep -f "batch_evaluate"; if another run is active, sleep 140 and re-check (up to 20x, each a SEPARATE short Bash call); launch only when clear.',
  '1. If ' + REP + ' exists -> launched=true note="already complete".',
  '2. Else write ' + DIR + '/COMMAND.txt and launch EXACTLY:',
  "bash -lc 'mkdir -p " + DIR + "; source " + PKG + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + DIR + " --max_per_category 999 --concurrency 2 --snh_veto --snh_protect > " + LOG + " 2>&1 & echo PID $!'",
  '   NO --categories (all 290). BUE/CBT/recal are default-ON — do not pass them.',
  '3. sleep 30; tail -5 ' + LOG + ' for an immediate Traceback. Return launched, note.',
].join('\n')

const WATCH_PROMPT = [
  'You are Agent D-watcher for the v180 full-290 eval (expect ' + N_FULL + ' videos). SHORT. ' + RULES,
  '1. If ' + REP + ' exists -> done=true. Metrics: ' + PY + ' -c "import json;d=json.load(open(\'' + REP + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'),ti.get(\'succeeded\'),ti.get(\'failed\'))" . num_videos MUST be 290.',
  '   markers: vid_count=$(grep -cF "[SNH-VID]" ' + LOG + ') (must be ~290, FM17); fire_count=$(grep -cF "[SNH-FIRE]" ' + LOG + '); skip_count=$(grep -cF "[SNHP-SKIP]" ' + LOG + ') (MUST be > 0 — the protection fired, FM5; if 0, scream in note); err_count=$(grep -cF "[SNH-ERR]" ' + LOG + ') (must be 0).',
  '2. Else pgrep -f "batch_evaluate.*' + V + '_' + TAG + '". Alive -> ONE sleep 140, count JSONs (ls ' + DIR + '/*.json | wc -l), done=false note="running <n>/290". >40 min no progress -> diagnose via tail -30, do NOT kill without evidence.',
  '3. Dead with no report -> tail -30 ' + LOG + '. TRANSIENT -> relaunch the SAME nohup (runner skips completed). CODE BUG -> crashed=true + traceback.',
  'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, vid_count, fire_count, skip_count, err_count, note.',
].join('\n')

const launch = await retryAgent(() => agent(LAUNCH_PROMPT, { schema: launchSchema, label: 'D-launch-' + V, phase: 'Eval290' }))
if (!launch || !launch.launched) {
  log('v22 ABORT: launch failed: ' + (launch ? launch.note : 'null'))
} else {
let st = null
if (launch.note && launch.note.indexOf('already complete') >= 0) {
  st = await retryAgent(() => agent(WATCH_PROMPT, { schema: watchSchema, label: 'D-watch-adopt-' + V, phase: 'Eval290' }))
}
for (let w = 0; w < 160 && !(st && (st.done || st.crashed)); w++) {
  st = await retryAgent(() => agent(WATCH_PROMPT, { schema: watchSchema, label: 'D-watch-' + V + '-' + w, phase: 'Eval290' }))
}
if (!st || !st.done) {
  log('v22 ABORT: eval incomplete: ' + (st ? st.note : 'watcher null'))
} else if (st.err_count > 0 || st.skip_count === 0) {
  log('v22 ABORT: instrument failure — err=' + st.err_count + ' skips=' + st.skip_count + ' (protection never fired => FM5 void). ' + st.note)
} else {
log('v22 eval done: mAP=' + st.mAP + ' fires=' + st.fire_count + ' skips=' + st.skip_count)

// ============================ PHASE 3 — ADJUDICATE ============================
phase('Adjudicate')

const ADJ_PROMPT = [
  'You are Agent C, the SKEPTICAL adjudicator for v180 SNH-P. Recompute everything from ' + DIR + ' + ' + LOG + '. Bash < 150s. ' + RULES,
  'PRE-REGISTERED GATES (identical to v179): paired mAP delta >= -0.003 AND abs mAP >= 0.16454 AND Normal-video FA-rate relative reduction >= 25% AND <= 1 GT-overlapping (IoU>=0.1) detection vetoed.',
  '1. FM GREPS: [SNH-VID]==290; [SNH-FIRE]; [SNHP-SKIP] (>0); [SNH-COLLISION]; [SNH-ERR]==0. Confirm run mAP from ' + REP + '.',
  '2. PAIRED: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + DIR + ' --paired-log ' + LOG + ' --marker SNH-FIRE --paired-mode conf --min-delta -0.003 --min-p 0.5 — paste FULL output. M4: matched != n_fires => paired VOID (say so; adjudicate on arm-ON absolute + FA).',
  '3. FA OBJECTIVE: Normal-video FA rate both arms (arm-OFF = restored), rel reduction, fp_dets/Normal-video. Compare with v179 (0.50667 -> 0.16667): did protection cost FA? (v179 artifacts: ' + V179_DIR + ').',
  '4. GT-HARM: vetoed dets overlapping GT at IoU>=0.1 — list each (vid, window, IoU). CRITICAL COMPARISON: the 5 v179 harm dets (RoadAccidents128 3-8; Explosion025 10-16.5; Shooting048 46-52; Shoplifting022 45.5-55; RoadAccidents002 9-11.5) — for each state whether it was vetoed again or PROTECTED ([SNHP-SKIP]) in this run.',
  '5. GATE TABLE + VERDICT: PROMOTE iff all 4 pass. summary: one honest paragraph incl. whether the MIL-peak rule fixed the exact failure it was aimed at, and the protection rate (skips/(skips+fires)).',
  'Return verdict, paired_delta, abs_map_on, abs_map_off, ci90_low, ci90_high, p_delta_pos, fa_rate_on, fa_rate_off, fa_rel_reduction, gt_vetoed, n_fires, matched, n_skips, vid_count, gate_table, summary.',
].join('\n')

const adj = await retryAgent(() => agent(ADJ_PROMPT, { schema: adjSchema, label: 'C-adjudicate-' + V, phase: 'Adjudicate' }))

// ============================ PHASE 4 — DECIDE + RECORDS ============================
phase('Decide')
log('v22 adjudication: ' + (adj && adj.verdict))

const DECIDE_PROMPT = [
  'You are Agent C (records) for v180 SNH-P. Write the session records from the adjudication below; re-grep artifacts to verify every number (FM20). Do NOT re-run evals. ' + RULES,
  'PRECHECK: ' + JSON.stringify(pre),
  'ADJUDICATION: ' + JSON.stringify(adj),
  'EVAL FINAL: mAP=' + st.mAP + ' AP@0.5=' + st.ap50 + ' AP@0.7=' + st.ap70 + ' fires=' + st.fire_count + ' skips=' + st.skip_count,
  '1. Write ' + ITER + '/summary_v22_snhp_20260817.md modeled on summary_v21_snh-eval_20260816.md: verdict; the v179 G4 failure recap; the protection rule; precheck numbers (harm protected ' + pre.harm_protected + '/5, fa_rel_est ' + pre.fa_rel_est.toFixed(3) + '); the 290 numbers (arms, paired delta/CI/P, FA rates, gt_vetoed with the per-harm-det protected-vs-vetoed table); gate table; FM grep evidence; decision; family update (precisely what is refuted or validated).',
  '2. Append ONE ledger row to ' + ITER + '/session_v97_ledger.md (do NOT modify existing rows): loop | ' + V + ' | SNH-P ... | kept=<true|false> with the 4 gate numbers, fires/skips/matched, FA rates, gt_vetoed, and revert/promote state.',
  '3. DECISION EXECUTION:',
  '   - CLOSE: snh_protect + snh_veto stay default False (verify + paste); code stays in-tree inert; one-line config.py closure comment at snh_protect.',
  '   - PROMOTE: backup ' + ROOT + '/.qoder/backups/' + V + '_promote/ FIRST; flip BOTH snh_veto and snh_protect default True (dataclass + from_args + set_defaults both CLIs), add --no_snh_veto/--no_snh_protect opt-outs, validate. NOTE in the record that this changes the committed pipeline\'s Normal-video behaviour (FA -67% class effect) even though mAP is non-inferior — the promotion criterion here is the FA objective as pre-registered.',
  '4. Report files written + final flag defaults.',
].join('\n')

await retryAgent(() => agent(DECIDE_PROMPT, { label: 'C-records-' + V, phase: 'Decide' }))
log('v22 complete: verdict=' + (adj && adj.verdict) + ' delta=' + (adj && adj.paired_delta) + ' fa_rel=' + (adj && adj.fa_rel_reduction) + ' gt_vetoed=' + (adj && adj.gt_vetoed))
}
}
}
}
