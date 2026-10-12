export const meta = {
  name: 'vad-team-map20-v21',
  description: 'VAD agent team v21 — SNH (Supervised Normality Head) EVAL session. The --snh_veto implementation (v179) is ALREADY LANDED AND VERIFIED by the main session after the first workflow attempt died on the 180s stall watchdog (foreground long-running Bash): _snh_veto terminal-slot video-level P(normal)>=tau veto (pipeline.py L3473, call sites L1818/L1980/L2355), model_snh.json + shared snh_features.py, [SNH-FIRE]/[SNH-VID] markers, SNH_FORCE_FIRE hook, CLI in run.py+batch_evaluate.py, flag default OFF byte-identical, py_compile+defaults verified. THIS RUN: (1) ONE queue-disciplined full-290 eval with --snh_veto (launcher + watcher-loop pattern — NO agent ever blocks on a long foreground command, the exact v20-proven idiom); (2) paired-log conf adjudication (analysis_bootstrap_ab.py --paired-log --marker SNH-FIRE --paired-mode conf, zero cross-run VLM noise); (3) the 4 pre-registered gates: paired mAP delta >= -0.003 AND abs mAP >= 0.16454 AND Normal-video FA-rate relative reduction >= 25% AND <= 1 GT-overlapping (IoU>=0.1) detection vetoed; (4) promote-or-close + records. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON, full-290 ~0.173, v160_full 0.17254); cgr/ccr/cnv/kfd/tcp all default-OFF and stay so. Concurrency 2, never two evals at once (16 GB swap-thrash). Judged on the FA-rate objective with mAP only as non-inferiority constraint — NOT Normal-FP-for-mAP (banned 9).',
  phases: [
    { title: 'Eval290', detail: 'launcher + watcher loop, full-290 --snh_veto, concurrency 2' },
    { title: 'Adjudicate', detail: 'paired-log conf restore + 4 gates + FA rate + FM greps' },
    { title: 'Decide', detail: 'promote-or-close + summary + ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const PKG = ROOT + '/video_anomaly_detection'
const ITER = PKG + '/iteration_logs'
const PY = 'python3.14'
const V_IMPL = 'v179'
const TAG = 'snh290'
const DIR = ROOT + '/results/batch_eval_' + V_IMPL + '_' + TAG
const LOG = DIR + '.log'
const REP = DIR + '/evaluation_report.json'
const N_FULL = 290

const RULES = [
  'NEVER edit the evaluator, GT labels, or ' + ITER + '/analysis_bootstrap_ab.py.',
  'NEVER run two batch_evaluate processes concurrently (16 GB swap-thrash).',
  'NO Bash command may run > 150s in the foreground — the runtime kills stalled agents at 180s (this exact failure killed the first v21 attempt). Long work = nohup background + short polls.',
  'grep claims must be GREPPED in produced artifacts, never asserted (FM20/FM21).',
  'Do NOT git-commit anything. StructuredOutput EXACTLY ONCE per agent.',
].join('\n')

const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, vid_count: { type: 'number' }, fire_count: { type: 'number' }, err_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'vid_count', 'fire_count', 'err_count', 'note'], additionalProperties: false }
const adjSchema = { type: 'object', properties: {
  verdict: { type: 'string', enum: ['PROMOTE', 'CLOSE'] },
  paired_delta: { type: 'number' }, abs_map_on: { type: 'number' }, abs_map_off: { type: 'number' },
  ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_delta_pos: { type: 'number' },
  fa_rate_on: { type: 'number' }, fa_rate_off: { type: 'number' }, fa_rel_reduction: { type: 'number' },
  fp_det_on: { type: 'number' }, fp_det_off: { type: 'number' },
  gt_vetoed: { type: 'integer' }, n_fires: { type: 'integer' }, matched: { type: 'integer' },
  vid_count: { type: 'integer' }, gate_table: { type: 'string' }, summary: { type: 'string' },
}, required: ['verdict', 'paired_delta', 'abs_map_on', 'abs_map_off', 'ci90_low', 'ci90_high', 'p_delta_pos', 'fa_rate_on', 'fa_rate_off', 'fa_rel_reduction', 'gt_vetoed', 'n_fires', 'matched', 'vid_count', 'gate_table', 'summary'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

const LAUNCH_PROMPT = [
  'You are Agent D-launcher for the v21 SNH full-290 eval (' + V_IMPL + '). Ensure the eval is running, then return IMMEDIATELY. SHORT. ' + RULES,
  '0. QUEUE DISCIPLINE FIRST: pgrep -f "batch_evaluate" ; if another run is active, sleep 140 and re-check (up to 20x, each check a SEPARATE short Bash call — never one long sleep chain); only launch when clear.',
  '1. If ' + REP + ' exists -> launched=true note="already complete".',
  '2. Else write ' + DIR + '/COMMAND.txt and launch EXACTLY:',
  "bash -lc 'mkdir -p " + DIR + "; source " + PKG + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + DIR + " --max_per_category 999 --concurrency 2 --snh_veto > " + LOG + " 2>&1 & echo PID $!'",
  '   NO --categories flag (all 290; Normal videos are the subject). The committed BUE/CBT/recal flags are default-ON — do not pass them.',
  '3. sleep 30; tail -5 ' + LOG + ' to catch an immediate Traceback. Return launched, note.',
].join('\n')

const WATCH_PROMPT = [
  'You are Agent D-watcher for the v21 SNH full-290 eval (expect ' + N_FULL + ' videos). SHORT. ' + RULES,
  '1. If ' + REP + ' exists -> done=true. Read metrics: ' + PY + ' -c "import json;d=json.load(open(\'' + REP + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'),ti.get(\'succeeded\'),ti.get(\'failed\'))" . num_videos MUST be ' + N_FULL + ' (else scream harness bug in note).',
  '   markers: VC=$(grep -cF "[SNH-VID]" ' + LOG + ' 2>/dev/null) -> vid_count (must be ~' + N_FULL + ', FM17); FC=$(grep -cF "[SNH-FIRE]" ' + LOG + ' 2>/dev/null) -> fire_count; EC=$(grep -cF "[SNH-ERR]" ' + LOG + ' 2>/dev/null) -> err_count (must be 0 or explain every hit in note).',
  '2. Else pgrep -f "batch_evaluate.*' + V_IMPL + '_' + TAG + '". Alive -> ONE sleep 140 (single Bash call, under the 150s cap), count done JSONs (ls ' + DIR + '/*.json | wc -l), done=false note="running <n>/' + N_FULL + '". If n has not advanced across checks for >40 min of wall time -> diagnose via tail -30 ' + LOG + ' (swap thrash? API stall?) and say so in note; do NOT kill without evidence.',
  '3. Dead with no report -> tail -30 ' + LOG + '. TRANSIENT (network/API) -> relaunch the SAME nohup command (the runner skips completed videos). CODE BUG -> crashed=true + paste the traceback.',
  'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, vid_count, fire_count, err_count, note.',
].join('\n')

// ============================ PHASE 1 — EVAL290 ============================
phase('Eval290')
log('v21: launch full-290 SNH eval (impl already landed + verified by main session)')

const launch = await retryAgent(() => agent(LAUNCH_PROMPT, { schema: launchSchema, label: 'D-launch-' + V_IMPL, phase: 'Eval290' }))
if (!launch || !launch.launched) {
  log('v21 ABORT: launch failed: ' + (launch ? launch.note : 'null'))
} else {
let st = null
if (launch.note && launch.note.indexOf('already complete') >= 0) {
  st = await retryAgent(() => agent(WATCH_PROMPT, { schema: watchSchema, label: 'D-watch-adopt-' + V_IMPL, phase: 'Eval290' }))
}
for (let w = 0; w < 160 && !(st && (st.done || st.crashed)); w++) {
  st = await retryAgent(() => agent(WATCH_PROMPT, { schema: watchSchema, label: 'D-watch-' + V_IMPL + '-' + w, phase: 'Eval290' }))
}
if (!st || !st.done) {
  log('v21 ABORT: eval did not complete: ' + (st ? st.note : 'watcher null'))
} else if (st.err_count > 0) {
  log('v21 ABORT: [SNH-ERR] x' + st.err_count + ' — feature path broken on test videos: ' + st.note)
} else {
log('v21 eval done: mAP=' + st.mAP + ' fires=' + st.fire_count + ' vid=' + st.vid_count)

// ============================ PHASE 2 — ADJUDICATE ============================
phase('Adjudicate')

const ADJ_PROMPT = [
  'You are Agent C, the SKEPTICAL adjudicator for v21 SNH. Trust nothing from the implementer; recompute everything from ' + DIR + ' + ' + LOG + '. Bash < 150s per command. ' + RULES,
  'The 4 promotion gates are PRE-REGISTERED (same as v17): paired mAP delta >= -0.003 AND abs mAP >= 0.16454 AND Normal-video FA-rate relative reduction >= 25% AND <= 1 GT-overlapping (IoU>=0.1) detection vetoed.',
  '1. FM GREPS (raw counts): [SNH-VID] must == ' + N_FULL + ' (FM17); [SNH-FIRE]; [SNH-COLLISION]; [SNH-ERR] must be 0. The run evaluator mAP = ' + st.mAP + ' — confirm from ' + REP + '.',
  '2. PAIRED ADJUDICATION (v171 proven path, zero cross-run VLM noise): ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + DIR + ' --paired-log ' + LOG + ' --marker SNH-FIRE --paired-mode conf --min-delta -0.003 --min-p 0.5  — paste the FULL output. M4: if matched != n_fires the paired verdict is VOID — say so and adjudicate on arm-ON absolute + FA objective only.',
  '3. FA-RATE OBJECTIVE: from the paired restore, compute in BOTH arms the fraction of category==Normal test videos with >= 1 detection at confidence > 0 (arm-ON vs arm-OFF-restored), relative reduction = (off-on)/off; plus fp_dets per Normal video both arms. Use the same convention the v171/v172 records used (grep their summary if unsure: ' + ITER + '/summary_v17_v171-v172_20260815.md).',
  '4. GT-HARM: among the vetoed detections (the [SNH-FIRE] lines), how many overlap a GT segment at IoU >= 0.1 (evaluator convention: GT frames/30 vs det seconds, evaluate.load_ground_truth)? List each: vid, window, IoU. Reuse an existing iteration_logs analysis script if one fits.',
  '5. GATE TABLE + VERDICT: PROMOTE iff all 4 pass, else CLOSE. Include in summary: the gate fire rate on the ~150 Normal test videos vs the 94% train normal-flag rate (the v19 in-distribution caveat — model_mil.pth was trained on the same train split), i.e. how many Normal videos had gate=1 (grep gate=1 among Normal_* [SNH-VID] lines) and how many anomaly videos were vetoed.',
  'Return verdict, paired_delta, abs_map_on, abs_map_off, ci90_low, ci90_high, p_delta_pos, fa_rate_on, fa_rate_off, fa_rel_reduction, fp_det_on, fp_det_off, gt_vetoed, n_fires, matched, vid_count, gate_table (one line per gate PASS/FAIL with numbers), summary (one honest paragraph).',
].join('\n')

const adj = await retryAgent(() => agent(ADJ_PROMPT, { schema: adjSchema, label: 'C-adjudicate-' + V_IMPL, phase: 'Adjudicate' }))

// ============================ PHASE 3 — DECIDE + RECORDS ============================
phase('Decide')
log('v21 adjudication: ' + (adj ? adj.verdict : 'NULL'))

const DECIDE_PROMPT = [
  'You are Agent C (records) for v21 SNH. The adjudication StructuredOutput is below. Write the session records; re-grep artifacts to verify every load-bearing number (FM20). Do NOT re-run any eval. ' + RULES,
  'ADJUDICATION: ' + JSON.stringify(adj),
  'EVAL WATCH FINAL: mAP=' + st.mAP + ' AP@0.3=' + st.ap30 + ' AP@0.5=' + st.ap50 + ' AP@0.7=' + st.ap70 + ' num_videos=' + st.num_videos + ' fires=' + st.fire_count + ' vid=' + st.vid_count,
  '1. Write ' + ITER + '/summary_v21_snh-eval_20260816.md modeled on ' + ITER + '/summary_v17_v171-v172_20260815.md: headline verdict; SNH mechanism recap + v19 GO context (CV AUC 0.996, tau 0.7894, 94% normal-flag @ 3.8% anomaly-flag, MIL-family dominance, in-distribution caveat — see ' + ITER + '/summary_v19_snh-gate_20260816.md); the 290 numbers (arm mAPs, paired delta, CI, P, FA rates, fp_dets/Normal-video, gt_vetoed list); the 4-gate table; FM20/FM17/FM5 grep evidence (raw counts); the first-attempt stall failure + main-session self-test note; the decision; CLOSED/BANNED family update (if CLOSE: state precisely what is refuted — the video-level supervised P(normal) gate on MIL+CLIP+motion features at tau=0.7894 — and what remains open).',
  '2. Append ONE row to ' + ITER + '/session_v97_ledger.md (do NOT modify existing rows): loop | ' + V_IMPL + ' | SNH (Supervised Normality Head, video-level P(normal)>=tau veto, v19-GO-gated) ... | smoke | mAP_on | AP@0.5 | AP@0.7 | latency | kept=<true|false>. Include the 4 gate numbers, fires/matched, FA rates on/off, gt_vetoed, the Normal-test-video gate fire rate vs 94% train, and revert/promote state.',
  '3. DECISION EXECUTION:',
  '   - verdict=CLOSE: snh_veto stays default False — VERIFY (ScoringConfig() + VADConfig.from_args(argparse.Namespace()) + both CLIs; paste checks); code STAYS in-tree inert (like cnv) — do NOT delete; add a one-line config.py comment at the snh_veto field recording the closure verdict.',
  '   - verdict=PROMOTE: backup to ' + ROOT + '/.qoder/backups/' + V_IMPL + '_promote/ FIRST, then flip snh_veto default True in config.py (dataclass + from_args getattr), set_defaults in run.py + batch_evaluate.py, add --no_snh_veto opt-outs, validate py_compile + defaults both ways.',
  '4. Report exact files written/edited + the final committed-pipeline state (all flag defaults).',
].join('\n')

await retryAgent(() => agent(DECIDE_PROMPT, { label: 'C-records-' + V_IMPL, phase: 'Decide' }))
log('v21 complete: verdict=' + (adj && adj.verdict) + ' paired_delta=' + (adj && adj.paired_delta) + ' fa_rel=' + (adj && adj.fa_rel_reduction))
}
}
