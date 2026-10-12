export const meta = {
  name: 'vad-team-map20-v20',
  description: 'VAD agent team v20 — TCP (Temporal-Contrast Prompt): a prompt-side recall lever on the COMMITTED directed scan path, grounded in the v18 post-hoc chain analysis. The v18 addendum (summary_v18 §10) established: (1) the committed pipeline uses the DIRECTED scan (config.py L246 full_coverage_comparative_scan=False) — scan_segment (anomaly_discovery.py L358) already shows the VLM ALL ~12 frames of each 6s segment with an absolute whole-segment prompt; (2) the v173 A/B was confounded (KFD also engaged FCCS) — any future scan-side A/B MUST hold the scan path constant (directed in BOTH arms, FCCS line count == 0 in BOTH logs, pre-registered VOID check); (3) the real residual on the directed path is RECOGNITION-BOUND: RA009 (GT 7-8s) and RA021 (GT 1-3s) were rejected by the VLM even with all frames shown — the whole-segment prompt ("Determine if this segment contains any abnormal activity") lets a brief 1-3s event dilute into a mostly-normal 6s window. MECHANISM: append a pre-registered temporal-contrast instruction block to the scan_segment prompt (frames are time-ordered and span the segment; a real anomaly may be BRIEF and occupy only a few mid-clip frames — judge by the single MOST anomalous moment, not the clip average; compare early vs late frames for sudden change). Same frames, same call count, same directed path, same JSON schema — only prompt TEXT changes; zero latency cost. Flag tcp_prompt default OFF, CLI in run.py AND batch_evaluate.py, marker TCP-VID unconditional per video on every path + the TCP block logged once per video. Cross-run A/B vs a FRESH same-session committed REF (VLM input text changes — no paired instrument possible). GATES (pre-registered): Loop 1 hard-88 — PROMOTE-TO-290 iff point_delta >= +0.008 AND P(delta>0) >= 0.9 AND new hard mAP >= 0.11306, FCCS-count==0 in both logs, det/video increase flagged at > +1.0. Loop 2 (only if Loop 1 passes) full-290 net-neutrality — delta >= -0.003 AND abs >= 0.16454 AND Normal-video FA-rate relative increase <= +2%. Purpose-population check: does TCP produce detections overlapping the RA009/RA021 GTs (the recognition-bound population the mechanism is built for)? Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON); cgr/ccr/cnv/kfd/snh all default-OFF and stay so. Concurrency 2. QUEUE DISCIPLINE: v19 (SNH) may still be running its 290 eval — EVERY batch_evaluate launch in this session (incl. the self-test smoke) must first verify no other batch_evaluate process is running, and sleep-poll until clear (up to ~90 min) — never run two evals concurrently (16 GB machine, owner-diagnosed swap thrash).',
  phases: [
    { title: 'Implement', detail: 'D1 appends the pre-registered TCP block to the scan_segment prompt (default OFF), self-tests prompt content on/off, queue-disciplined smoke' },
    { title: 'AB88', detail: 'fresh committed REF88 + TCP88 (both directed, FCCS==0 verified) -> cross-run bootstrap + RA009/RA021 purpose check' },
    { title: 'Confirm290', detail: 'if 88 passes: fresh REF290 + TCP290, net-neutrality + Normal-FA bound' },
    { title: 'Decide', detail: 'promote tcp_prompt default-ON iff both loops pass; else archive-close; summary + ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS = '--categories ' + HARD_CATS
const N_HARD = 88
const N_FULL = 290
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const TCP_FLAG = '--tcp_prompt'
const VID_MARKER = 'TCP-VID'
const V_IMPL = 'v177'
const V_290 = 'v178'
// pre-registered gates
const MIN_DELTA_88 = 0.008
const MIN_P = 0.90
const HARD_FLOOR = 0.11306
const MAP_TOL_290 = -0.003
const ABS_FLOOR_290 = 0.16454
const MAX_FA_REL_INC = 0.02
const MAX_DET_INC = 1.0

const TCP_BLOCK = [
'Temporal reasoning instructions:',
'- The frames are shown in time order and span the whole segment uniformly.',
'- A real anomaly may be BRIEF (1-3 seconds) and occupy only a few frames in the middle of the clip. Do NOT average the clip: judge by the single most anomalous moment you can find.',
'- Compare the early frames against the later frames: note any sudden change (impact, fall, crowd scatter, smoke or flame onset, concealment motion). A change that starts and ends inside the clip still counts fully.',
].join('\n')

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE: v137 + BUE + CBT + CBT-recal ALL DEFAULT-ON (full-290 ~0.173; hard-88 ~0.11806). cgr/ccr/cnv/kfd_uniform/snh all DEFAULT-OFF (v16 relocation closed; v17 CNV closed on harm; v18 KFD-U closed). full_coverage_comparative_scan=False is COMMITTED — the DIRECTED scan is the committed path. Do NOT change any committed default. !!!',
'',
'!!! THE v18 LESSONS THIS SESSION IS BUILT ON (summary_v18_v173-v174_20260815.md §10 — read it) !!!',
'1. The committed directed scan (scan_segment, anomaly_discovery.py L358-391) shows the VLM ALL ~12 frames per 6s segment with a whole-segment absolute prompt. The pipeline is NOT frame-blind on the committed path.',
'2. The v173 KFD A/B was CONFOUNDED: kfd_uniform_keyframes also engaged the FCCS comparative path (pipeline.py L1776-1777). Any scan-side A/B MUST hold the scan path constant: directed in BOTH arms, grep -c "FCCS" == 0 in BOTH produced logs — a nonzero count is a VOID-level confound (pre-registered).',
'3. The recognition-bound residual: RA009 (GT 7-8s, inside segment 4.5-10.5) and RA021 (GT 1-3s, inside segment 0-5.2) were REJECTED by the VLM with all 11-12 frames shown. Hypothesis: the whole-segment prompt ("Determine if this segment contains any abnormal or suspicious activity") invites clip-AVERAGING, so a brief 1-3s event dilutes into a mostly-normal window.',
'',
'!!! THE MECHANISM — TCP (Temporal-Contrast Prompt), prompt-side ONLY !!!',
'When tcp_prompt is ON, APPEND this exact block to the scan_segment prompt immediately BEFORE the "Respond ONLY in JSON format" line (context_prompt and everything else unchanged):',
'---',
TCP_BLOCK,
'---',
'Same frames, same call count, same directed path, same JSON response schema (is_anomalous/confidence/description) — ONLY prompt text changes. Zero latency cost. This is the admissible "query quality" lever; it is NOT a banned family (not an abs-conf gate, not merge/batch, not scan-density).',
'',
'!!! GATES (pre-registered) !!!',
'Loop 1 (hard-88, cross-run vs FRESH same-session committed REF): PROMOTE-TO-290 iff point_delta >= +' + MIN_DELTA_88 + ' AND P(delta>0) >= ' + MIN_P + ' AND new hard mAP >= ' + HARD_FLOOR + '. FM2 watch: flag if mean detections/video rises > +' + MAX_DET_INC + '. VOID if FCCS appears in either log.',
'Loop 2 (full-290, only if Loop 1 passes): net-neutrality — delta >= ' + MAP_TOL_290 + ' AND abs >= ' + ABS_FLOOR_290 + ' AND Normal-video FA-rate relative increase <= +' + (MAX_FA_REL_INC * 100) + '% (a "find the most anomalous moment" instruction may also raise Normal-video hallucination — measure it).',
'Purpose population: report whether TCP produces detections overlapping the RA009 (GT 7-8s) and RA021 (GT 1-3s) intervals at IoU>=0.1 where REF has none.',
'',
'FM DISCIPLINE: FM5 the TCP block must be GREPPABLE in the produced log (log the block once per video alongside ' + VID_MARKER + '). FM17 ' + VID_MARKER + ' unconditional per video on EVERY path incl. 0-detection and early-return-normal (mirror the CNV/SNH call sites). FM20 grep not claim. FM21 exact strings. FM18 absolute mAP. HARNESS BUG: --categories baked into every hard-88 command by the launcher; watcher verifies num_videos=' + N_HARD + '.',
'QUEUE DISCIPLINE (load-bearing): another session\'s eval may be running. BEFORE EVERY batch_evaluate launch (incl. the self-test smoke): pgrep -f batch_evaluate; if anything is running, sleep 140 and re-check, up to 40 times (~90 min); only launch when clear. NEVER run two evals at once (16 GB swap-thrash, owner-diagnosed).',
'HARD CONSTRAINTS: flag default OFF (flag-off byte-identical, prompt text bit-identical); CLI in BOTH run.py and batch_evaluate.py; evals FIXED --concurrency ' + CONCURRENCY + '; StructuredOutput EXACTLY ONCE per agent; no Bash > ~150s except stated smokes.',
].join('\n')

const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'PASTED: (a) flag-ON smoke log lines showing the TCP block inside the actual prompt AND ' + VID_MARKER + ' grep == #videos; (b) flag-OFF run on the same videos producing the bit-identical committed prompt (diff proof); (c) VLM call count identical on/off ([COST] lines); (d) FCCS grep == 0 in the smoke log' }, marker_vid: { type: 'string' }, eval_flags: { type: 'string' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'instrument_selftest', 'marker_vid', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number' }, fccs_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'fccs_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, det_per_video_new: { type: 'number' }, det_per_video_ref: { type: 'number' }, fa_rel_inc: { type: 'number' }, purpose_converted: { type: 'string', description: 'which of RA009/RA021 gained an IoU>=0.1 detection under TCP that REF lacked' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'ref_map', 'note'], additionalProperties: false }
const decideSchema = { type: 'object', properties: { promoted: { type: 'boolean' }, decision: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['promoted', 'decision', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

function evalLaunchPrompt(V, tag, evalFlags, forceHarness, vidMarker) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const harn = forceHarness ? (HARNESS + ' ') : ''
  return [
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + dir + ' ; log=' + dir + '.log ; report=' + dir + '/evaluation_report.json',
'REQUIRED FLAGS (verbatim): ' + harn + evalFlags,
'0. QUEUE DISCIPLINE FIRST: pgrep -f batch_evaluate; while anything is running, sleep 140 and re-check (up to 40x); report blocked only after that.',
'1. Already running for this V_tag -> adopt if the flags match, else kill and relaunch. 2. report exists AND the log shows the required flags -> launched=true, note="already complete". 3. A DIFFERENT version is running -> keep polling per step 0.',
'4. Otherwise write ' + dir + '/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + dir + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category 999 --concurrency " + CONCURRENCY + " " + harn + evalFlags + " > " + dir + ".log 2>&1 & echo PID $!'",
'5. sleep 30; tail -5 the log to catch an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, vidMarker, nTotal) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const rep = dir + '/evaluation_report.json'
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + ' (expect ' + nTotal + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE.',
'1. If ' + rep + ' exists -> done=true. Read mAP/AP@0.3/0.5/0.7/num_videos: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'))" . num_videos MUST be ' + nTotal + ' (else harness bug, say so loudly).',
'   markers: VC=$(grep -cF "' + vidMarker + '" ' + dir + '.log 2>/dev/null) -> marker_count=VC. FCCC=$(grep -cF "FCCS" ' + dir + '.log 2>/dev/null) -> fccs_count=FCCC (MUST be 0 — the directed-path-constant pre-registration; nonzero = VOID-level confound, scream it in note).' + (vidMarker === VID_MARKER ? ' VC must be ~' + nTotal + ' (FM17) and the log must show the TCP block (FM5).' : ' (REF arm: VC may be 0; FCCS must still be 0.)'),
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 (up to 2x), then done=false note="running <n>/' + nTotal + '" (n = JSON count). STALLED >30min -> kill and relaunch the SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead with no report -> tail -30 the log. TRANSIENT -> relaunch. CODE BUG -> crashed=true + traceback.',
'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, marker_count, fccs_count, note.',
  ].join('\n')
}
async function runEval(V, tag, evalFlags, vidMarker, maxWatch, nTotal, ph, forceHarness) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, evalFlags, forceHarness, vidMarker), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: ph }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, fccs_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  if (launch.note && launch.note.indexOf('already complete') >= 0) {
    const st0 = await retryAgent(() => agent(evalWatchPrompt(V, tag, vidMarker, nTotal), { schema: watchSchema, label: 'D-watch-adopt-' + tag + '-' + V, phase: ph }))
    if (st0 && st0.done) return st0
  }
  for (let w = 0; w < (maxWatch || 130); w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, vidMarker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: ph }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, marker_count: -1, fccs_count: -1, note: 'watcher cap reached' }
}
function boot88Prompt(tcpDir, refDir, ph) {
  return [
'You are Agent D-adjudicator for the TCP CROSS-RUN A/B on the hard-88. Bash < 150s per command. StructuredOutput EXACTLY ONCE.',
'STEP 1 (confound + FM gates): F1=$(grep -cF "FCCS" ' + tcpDir + '.log); F2=$(grep -cF "FCCS" ' + refDir + '.log); VC=$(grep -cF "' + VID_MARKER + '" ' + tcpDir + '.log); PB=$(grep -cF "Temporal reasoning instructions" ' + tcpDir + '.log). F1+F2 MUST be 0 (else ran=false, note="VOID: scan-path confound"), VC ~88, PB>0 (the TCP block was actually sent).',
'STEP 2: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + tcpDir + ' --ref ' + refDir + ' --min-delta ' + MIN_DELTA_88 + ' --min-p ' + MIN_P + ' -> point_delta, ci90, p_gt0, new_map, ref_map. Paste the cross-run CAVEAT verbatim.',
'STEP 3 (FM2 watch): mean detections/video both dirs; flag increase > +' + MAX_DET_INC + '.',
'STEP 4 (purpose population): for RoadAccidents009 (GT 7-8s) and RoadAccidents021 (GT 1-3s): does the TCP arm have a detection with IoU>=0.1 where the REF arm has none? purpose_converted lists them (GT from the parquet via evaluate.load_ground_truth, fps=30).',
'STEP 5: per-category AP deltas for the 4 hard cats into per_category.',
'Report ran, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, det_per_video_new, det_per_video_ref, purpose_converted, per_category, note (VERDICT verbatim + caveat).',
  ].join('\n')
}
function boot290Prompt(tcpDir, refDir, ph) {
  return [
'You are Agent D-adjudicator for the TCP net-neutrality confirm on the FULL 290. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'STEP 1 (confound + FM gates): FCCS grep MUST be 0 in BOTH logs; ' + VID_MARKER + ' ~290 in the TCP log; TCP block grep > 0. Else ran=false STOP.',
'STEP 2: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + tcpDir + ' --ref ' + refDir + ' --min-delta ' + MAP_TOL_290 + ' --min-p ' + MIN_P + ' -> point_delta, ci90, p_gt0, new_map, ref_map. CAVEAT verbatim.',
'STEP 3 (Normal-FA bound): fraction of Normal-category videos with >=1 detection and mean dets/video, both dirs; fa_rel_inc=(on-off)/off (0 if off==0). Bound +' + (MAX_FA_REL_INC * 100) + '% relative. Context: same-pipeline committed runs drift ~+/-2.6% relative on this metric (v171/v172) — report raw rates AND that drift band.',
'STEP 4: per-category AP deltas (flag |d|>0.005); detection-count means.',
'Report ran, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, det_per_video_new, det_per_video_ref, fa_rel_inc, per_category, note.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v20 starting. TCP (Temporal-Contrast Prompt): prompt-side recall lever on the COMMITTED directed path (v18 addendum lessons baked in: directed-in-both-arms, FCCS==0 VOID check). Gates: 88 delta >= +' + MIN_DELTA_88 + ' & P >= ' + MIN_P + ' & abs >= ' + HARD_FLOOR + '; 290 net-neutral & FA inc <= +' + (MAX_FA_REL_INC * 100) + '%. QUEUE-disciplined launches. concurrency ' + CONCURRENCY + '.')

// ---------- PHASE 1: implement ----------
phase('Implement')
const impl = await retryAgent(() => agent([
'You are Agent D1. Implement TCP exactly as pre-registered, then PROVE it. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'IMPLEMENT:',
'1. Back up every file you touch to ' + ROOT + '/.qoder/backups/' + V_IMPL + '/ FIRST; ls it.',
'2. DiscoveryConfig: tcp_prompt (default False) + CLI --tcp_prompt in BOTH run.py and batch_evaluate.py, threaded through VADConfig.from_args. Flag-off byte-identical (prompt text bit-identical to committed).',
'3. anomaly_discovery.py scan_segment (~L362): when the flag is on, APPEND the pre-registered TCP block (in CONTEXT) immediately BEFORE the "Respond ONLY in JSON format" line. Nothing else changes — same keyframes, same schema, same parser. Do NOT touch _comparative_call, the FCCS path, or kfd wiring.',
'4. MARKER: ' + VID_MARKER + ' unconditional per video on EVERY return path (segments scanned, tcp=on/off) incl. 0-detection and early-return-normal (mirror the CNV/SNH call sites in pipeline.py). When tcp_prompt is on, ALSO log the full TCP block once per video immediately after its ' + VID_MARKER + ' line (FM5 greppability of "Temporal reasoning instructions").',
'5. Bump the pipeline.py header to ' + V_IMPL + '. Do NOT touch any committed default.',
'',
'HARD SELF-TEST (QUEUE-DISCIPLINED; paste ACTUAL output):',
'  SMOKE ON: ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V_IMPL + '_selftest_on --max_per_category 1 --concurrency 1 ' + HARNESS + ' ' + MECH_BASE + ' ' + TCP_FLAG + '  -> (a) grep -cF ' + VID_MARKER + ' == #videos AND grep -cF "Temporal reasoning instructions" > 0 AND grep -cF FCCS == 0; (b) a log excerpt showing the TCP block inside an actual scan prompt.',
'  SMOKE OFF: same command minus ' + TCP_FLAG + ' (output ..._selftest_off) -> the scan prompt is bit-identical to committed (diff a prompt excerpt on the same video).',
'  (c) VLM call count identical on/off ([COST] lines).',
'VALIDATE: py_compile; import MultiAnomalyVAD; --tcp_prompt in BOTH --help; DiscoveryConfig() defaults unchanged; from_args(Namespace()) == defaults.',
'Return implemented, selftest_passed, instrument_selftest (pasted), marker_vid, eval_flags, files_changed, validation, notes.',
].join('\n'), { schema: implSchema, label: 'D1-implement-' + V_IMPL, phase: 'Implement' }))

const tcpOk = !!(impl && impl.implemented && impl.selftest_passed && impl.eval_flags)
let ref88 = null, tcp88 = null, boot88 = null, ref290 = null, tcp290 = null, boot290 = null
if (!tcpOk) {
  log('Phase 1: TCP NOT verified (selftest=' + (impl ? impl.selftest_passed : 'null') + ') -> stopping before eval spend')
} else {
  log('Phase 1: TCP verified. flags=' + impl.eval_flags)

  // ---------- PHASE 2: hard-88 cross-run A/B ----------
  phase('AB88')
  ref88 = await runEval(V_IMPL, 'ref88', MECH_BASE, 'QSCORE', 70, N_HARD, 'AB88', true)
  log('REF88: done=' + ref88.done + ' mAP=' + ref88.mAP + ' n=' + (ref88.num_videos != null ? ref88.num_videos : '?') + ' fccs=' + ref88.fccs_count + ' ' + (ref88.note || ''))
  if (ref88.done && !ref88.crashed) {
    tcp88 = await runEval(V_IMPL, 'tcp88', impl.eval_flags, VID_MARKER, 70, N_HARD, 'AB88', true)
    log('TCP88: done=' + tcp88.done + ' mAP=' + tcp88.mAP + ' n=' + (tcp88.num_videos != null ? tcp88.num_videos : '?') + ' vidMarker=' + tcp88.marker_count + ' fccs=' + tcp88.fccs_count)
  }
  const clean = !!(ref88 && ref88.done && !ref88.crashed && ref88.fccs_count === 0 && tcp88 && tcp88.done && !tcp88.crashed && tcp88.fccs_count === 0 && tcp88.marker_count > 0)
  if (clean) {
    boot88 = await retryAgent(() => agent(boot88Prompt(ROOT + '/results/batch_eval_' + V_IMPL + '_tcp88', ROOT + '/results/batch_eval_' + V_IMPL + '_ref88', 'AB88'), { schema: bootSchema, label: 'D-boot-88', phase: 'AB88' }))
  } else {
    log('AB88: eval unusable or FCCS confound -> no adjudication')
  }
}
const pass88 = !!(boot88 && boot88.ran && boot88.point_delta >= MIN_DELTA_88 && boot88.p_gt0 >= MIN_P && boot88.new_map >= HARD_FLOOR)
log('88 verdict: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3) + ' absTCP=' + boot88.new_map.toFixed(5) + ' absREF=' + boot88.ref_map.toFixed(5) + ' purpose=' + (boot88.purpose_converted || '?') + ' -> ' + (pass88 ? 'PASSES' : 'FAILS')) : 'no valid adjudication'))

// ---------- PHASE 3: 290 confirm ----------
phase('Confirm290')
if (pass88 && impl) {
  ref290 = await runEval(V_290, 'ref290', MECH_BASE, 'QSCORE', 130, N_FULL, 'Confirm290', false)
  log('REF290: done=' + ref290.done + ' mAP=' + ref290.mAP + ' n=' + (ref290.num_videos != null ? ref290.num_videos : '?') + ' fccs=' + ref290.fccs_count)
  if (ref290.done && !ref290.crashed) {
    tcp290 = await runEval(V_290, 'tcp290', impl.eval_flags, VID_MARKER, 130, N_FULL, 'Confirm290', false)
    log('TCP290: done=' + tcp290.done + ' mAP=' + tcp290.mAP + ' n=' + (tcp290.num_videos != null ? tcp290.num_videos : '?') + ' fccs=' + tcp290.fccs_count)
  }
  const clean290 = !!(ref290 && ref290.done && ref290.fccs_count === 0 && tcp290 && tcp290.done && tcp290.fccs_count === 0 && tcp290.marker_count > 0)
  if (clean290) {
    boot290 = await retryAgent(() => agent(boot290Prompt(ROOT + '/results/batch_eval_' + V_290 + '_tcp290', ROOT + '/results/batch_eval_' + V_290 + '_ref290', 'Confirm290'), { schema: bootSchema, label: 'D-boot-290', phase: 'Confirm290' }))
  }
} else {
  log('Confirm290: skipped — 88 gate not passed')
}
const ok290 = !!(boot290 && boot290.ran && boot290.point_delta >= MAP_TOL_290 && boot290.new_map >= ABS_FLOOR_290 && (boot290.fa_rel_inc == null || boot290.fa_rel_inc <= MAX_FA_REL_INC))
const promote = !!(pass88 && ok290)
log('290 verdict: ' + (boot290 && boot290.ran ? ('delta=' + boot290.point_delta.toFixed(5) + ' abs=' + boot290.new_map.toFixed(5) + ' FA inc=' + (boot290.fa_rel_inc != null ? (boot290.fa_rel_inc * 100).toFixed(1) + '%' : '?') + ' -> ' + (ok290 ? 'OK' : 'FAILS')) : 'n/a') + ' -> promote=' + promote)

// ---------- PHASE 4: decide ----------
phase('Decide')
const decision = await retryAgent(() => agent([
'You are Agent C. Execute the TCP promotion decision and write the session record. Scrupulously honest about cross-run noise (no paired instrument exists for prompt changes). Plain text.',
'', CONTEXT, '',
'EVIDENCE:',
'  88: ' + JSON.stringify(boot88),
'  290: ' + JSON.stringify(boot290),
'  gates: 88 delta >= +' + MIN_DELTA_88 + ' AND P >= ' + MIN_P + ' AND abs >= ' + HARD_FLOOR + '; 290 delta >= ' + MAP_TOL_290 + ' AND abs >= ' + ABS_FLOOR_290 + ' AND FA rel inc <= +' + (MAX_FA_REL_INC * 100) + '%',
'  computed: pass88=' + pass88 + ' ok290=' + ok290 + ' -> promote=' + promote,
'',
'DO:',
 promote
 ? ('1. PROMOTE tcp_prompt. Back up config.py/run.py/batch_evaluate.py/anomaly_discovery.py to ' + ROOT + '/.qoder/backups/tcp_promote/ ; set tcp_prompt default True in DiscoveryConfig, from_args default, and set_defaults in BOTH entrypoints (add --no_tcp_prompt opt-out). VALIDATE + paste: py_compile; tcp_prompt True; bue/cbt/recal True; cgr/ccr/cnv/kfd/snh False; import MultiAnomalyVAD.')
 : ('1. DO NOT PROMOTE — confirm tcp_prompt stays default False (paste defaults; committed pipeline untouched). Close honestly on the observed pattern: if the purpose population (RA009/RA021) converted but the global delta failed, say the recognition hypothesis held locally but symmetric-loss/drift elsewhere cancelled it (name the cancelling categories); if the purpose population did NOT convert, say the clip-averaging hypothesis is wrong for this VLM and the recognition residual is not prompt-addressable with whole-segment calls — the record should state that the directed-scan recognition ceiling is now measured twice (frames v18, prompt v20).'),
'2. date +%Y%m%d ; write ' + LOGS + '/summary_v20_' + V_IMPL + '-' + V_290 + '_<YYYYMMDD>.md covering: the v18-addendum basis (directed path committed; FCCS==0 discipline held — paste the counts); the TCP block verbatim; the 88 and (if ran) 290 numbers with CI/P/caveat; purpose-population conversion; per-category; Normal-FA read at 290; the decision; updated CLOSED list.',
'3. APPEND ledger rows to ' + LOGS + '/session_v97_ledger.md for ' + V_IMPL + ' (88) and, if it ran, ' + V_290 + ' (290): loop|version|title+verdict|smoke|mAP|AP@0.5|AP@0.7|lat|kept. -1 for absent. Do NOT rewrite existing rows.',
'4. Reply with a 12-line recap leading with the 88 delta + P and the decision.',
'Return promoted, decision, validation (pasted), note.',
].join('\n'), { schema: decideSchema, label: 'C-decide-tcp', phase: 'Decide' }))

log('v20 complete. TCP: ' + (promote ? 'PROMOTED' : 'NOT promoted') +
    ' | 88: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3)) : 'n/a') +
    ' | 290: ' + (boot290 && boot290.ran ? ('delta=' + boot290.point_delta.toFixed(5) + ' abs=' + boot290.new_map.toFixed(5)) : 'n/a'))
return { arm88: boot88, arm290: boot290, pass88, ok290, promote, decision: decision ? decision.decision : null }
