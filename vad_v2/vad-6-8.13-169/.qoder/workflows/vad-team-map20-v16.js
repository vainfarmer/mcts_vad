export const meta = {
  name: 'vad-team-map20-v16',
  description: 'VAD agent team v16 — settle the v167 CGR-M question with a CLEAN full-pipeline cgr-on vs cgr-off A/B, then promote or archive-close. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON, full-290 ~0.173); CGR-M (v167, --cgr_relocate) is code-present default-OFF. WHY THIS SESSION: v15 judged CGR with a paired field snapshotted PRE-BUE, so its +0.06105 "paired marginal" actually bundled BUE+CBT and was inflated. An offline reconstruction of the CGR-only step gave +0.01919 (P=0.886, CI90[-0.005,+0.051]) — promising but under the P>=0.90 bar, AND the offline method cannot see how BUE/CBT interact with a RELOCATED window (BUE grows from the new position; CBT clamps along a different curve region), which is exactly where CGR\'s real effect on FINAL mAP lives. v16 measures that interaction directly. KEY STRUCTURAL FACT (verified): CGR (pipeline.py ~L1790), CCR (~L1810), BUE (~L1818), CBT (~L1833) all run DOWNSTREAM of every VLM call, on cached motion-energy + CLIP curves — they are deterministic post-processing. So both A/B arms can be computed inside ONE eval run: arm-OFF = BUE+CBT applied to the pre-CGR detection snapshot; arm-ON = CGR then BUE+CBT. Same videos, same VLM outputs, same detection count -> an EXACT paired A/B with ZERO cross-run VLM noise, isolating CGR\'s true contribution to final mAP including the BUE/CBT interaction. The two arms differ by exactly one function call, so the comparison is sound by construction. CRITICAL IMPLEMENTATION HAZARD (verified): _cgr_motion_relocate / _bue_boundary_union_extension / _cbt_category_boundary_tighten all MUTATE a["t_start"]/a["t_end"] IN PLACE (pipeline.py L1298, L2174, L2378, L2532) — the dual-arm code MUST deep-copy the detection list before each arm or the arms corrupt each other. FALLBACK: if the dual-arm instrument fails its self-test, fall back to two REAL separate 88-video evals (cgr-on vs cgr-off) judged cross-run. Concurrency 2. All evals on the 88-video hard harness with --categories forced.',
  phases: [
    { title: 'Implement', detail: 'D1 implements the --cgr_ab_diag dual-arm instrument (deep-copy discipline, persists arm-OFF boundaries), then runs a HARD self-test proving correctness by construction' },
    { title: 'Verify88', detail: 'one 88-video hard eval with cgr-on + dual-arm diag -> exact paired bootstrap (arm-ON vs arm-OFF), zero VLM noise' },
    { title: 'Confirm290', detail: 'if 88 clears the gate: same dual-arm on full 290 to measure the effect at ship scale (Normal videos must be arm-identical — a free sanity check)' },
    { title: 'Decide', detail: 'promote cgr_enabled default-ON iff paired delta >= +0.008 with P>=0.90 at 88 AND net-neutral-or-positive at 290; else archive-close CGR and record the closure' },
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
// committed defaults, passed explicitly (byte-identical to defaults)
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const AB_FIELD = 'ab_arm_off_boundaries'   // per-detection arm-OFF windows; run's own final = arm-ON
const V_IMPL = 'v169'
const V_290 = 'v170'
// gates
const MIN_PAIRED_DELTA = 0.008
const MIN_P = 0.90
const HARD_REF_MAP = 0.11806        // committed CBT-recal hard-88 estimate
const REF290_MAP = 0.17254          // committed BUE+CBT+recal @290 (v160_full)

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE: v137 + BUE + CBT + CBT-recal, ALL DEFAULT-ON (ScoringConfig().bue_enabled/cbt_enabled/cbt_recal are True). full-290 mAP ~0.173 (v160_full 0.17254); hard-88 ~' + HARD_REF_MAP + '. CGR-M (v167) and CCR-SE (v168) are code-present DEFAULT-OFF (cgr_enabled=False, ccr_enabled=False). Do NOT change BUE/CBT/recal/CCR defaults. !!!',
'',
'!!! THE SINGLE QUESTION v16 ANSWERS: does CGR-M (--cgr_relocate) raise FINAL mAP, measured cleanly? !!!',
'CGR-M = count-neutral motion-energy-peak RELOCATION of the lowest-confidence off-GT detection onto an uncovered motion peak, gated to the 4 hard cats, firing at the FINAL detection list immediately BEFORE the pre-BUE snapshot.',
'v15 got this wrong: it judged CGR with paired field pre_cgr_boundaries, which is snapshotted PRE-BUE, so the measured +0.06105 delta bundled the committed BUE+CBT gains and was INFLATED (it was NOT a CGR-only figure). An offline reconstruction isolating the CGR step alone gave +0.01919, P(delta>0)=0.886, CI90[-0.005,+0.051] — directionally positive, just under the P>=0.90 bar. But that offline figure ALSO cannot capture the BUE/CBT interaction: after CGR moves a window, BUE grows outward from the NEW position and CBT clamps against a DIFFERENT region of the category curve. CGR\'s true effect on final mAP is exactly that end-to-end quantity. v16 measures it directly.',
'',
'!!! THE INSTRUMENT (why one run suffices, and why it beats two runs) !!!',
'VERIFIED STRUCTURE: CGR (pipeline.py ~L1790) -> CCR (~L1810, OFF) -> BUE (~L1818) -> CBT (~L1833). ALL of these run AFTER every VLM call, reading only cached per-frame motion energy and the cached CLIP category curve. They are DETERMINISTIC post-processing. Therefore both A/B arms can be produced inside a SINGLE eval run:',
'  arm-OFF = BUE+CBT applied to a deep copy of the pre-CGR detection list (i.e. what the committed pipeline would have produced)',
'  arm-ON  = CGR, then BUE+CBT (this is the run\'s real final output)',
'Same videos, same VLM sampling, same detection count -> an EXACT PAIRED A/B with ZERO cross-run VLM noise. Two separate runs would re-sample the VLM and reintroduce exactly the noise that made v165 (P=0.780) and the offline CGR estimate (P=0.886) inconclusive. The arms differ by exactly ONE function call, so the contrast is sound by construction.',
'',
'!!! CRITICAL IMPLEMENTATION HAZARD (verified by reading the code — do not skip) !!!',
'_cgr_motion_relocate, _bue_boundary_union_extension and _cbt_category_boundary_tighten all MUTATE a["t_start"] / a["t_end"] IN PLACE on the detection dicts (pipeline.py L1298-1299, L2174-2175, L2378-2379, L2532-2533). If the dual-arm code passes the SAME dict objects to both arms, the arms corrupt each other and the measurement is garbage (silently — the numbers will look plausible). You MUST copy.deepcopy the detection list before running EACH arm. State in your self-test how you proved the arms are independent.',
'',
'!!! MEASUREMENT — reuse the PROVEN tool, add no new bootstrap code !!!',
'Persist the arm-OFF windows as a per-detection field named EXACTLY ' + AB_FIELD + ' (index-aligned with anomalies, same length — copy the working write path from pre_bue_boundaries / pre_cbt_boundaries in models.py to_dict()). The run\'s own final output IS arm-ON. Then the existing tool measures the paired A/B directly, with its length-mismatch HARD-VOID check for free:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-json ' + AB_FIELD + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P,
'point_delta = arm-ON minus arm-OFF = the clean CGR effect on FINAL mAP including the BUE/CBT interaction. new_map = absolute hard mAP with CGR on.',
'',
'!!! PROMOTION GATE: paired delta >= +' + MIN_PAIRED_DELTA + ' AND P(delta>0) >= ' + MIN_P + ' at 88, AND net-neutral-or-positive at 290. Otherwise CGR is CLOSED and archived (leave default-OFF). Report the honest number either way — a negative result that closes a lead cleanly is a successful loop. !!!',
'',
'FM DISCIPLINE (hard-won; violating these has cost this project whole sessions): FM2 count/FP (CGR is count-neutral — assert ndet unchanged). FM5 marker must actually fire. FM6 offline gains do not transfer — that is why this session measures end-to-end. FM17 per-video marker must fire on the EMPTY/0-detection path. FM18 read the ABSOLUTE mAP, not only the delta. FM20 the paired field must be WRITTEN and GREPPED in a produced JSON — a claim is not evidence (v165 falsely claimed this). FM21 the activation marker string must EXACTLY match what the orchestrator greps. HARNESS BUG (cost v14 two loops): --categories must be in the eval command — the launcher below bakes it in; the watcher verifies num_videos.',
'HARD CONSTRAINTS: flag default OFF (flag-off byte-identical to the committed pipeline); CLI in BOTH run.py and batch_evaluate.py; eval FIXED --concurrency ' + CONCURRENCY + '. StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s except the stated self-test smoke; Read with offset/limit; keep progress lines short.',
].join('\n')

const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'PASTED actual evidence: (a) marker grep count, (b) a produced JSON printed showing ' + AB_FIELD + ' with values AND len == len(anomalies), (c) proof the arms are independent (deep-copy) e.g. a video where CGR fired 0 times has arm-OFF byte-identical to final, and a video where CGR fired has them differing, (d) ndet unchanged' }, deepcopy_proof: { type: 'string', description: 'exactly how arm independence was established given the in-place mutation hazard' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string', description: 'FULL flag string incl ' + MECH_BASE + ' + --cgr_relocate + the diag flag; do NOT include --categories (launcher adds it)' }, files_changed: { type: 'string' }, validation: { type: 'string', description: 'py_compile + import MultiAnomalyVAD + flag-off byte-identity: ScoringConfig() unchanged and from_args(Namespace()).scoring == ScoringConfig()' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'instrument_selftest', 'activation_marker', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number' }, paired_jsons: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number', description: 'arm-OFF absolute mAP' }, per_category: { type: 'string' }, count_neutral: { type: 'boolean' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
const decideSchema = { type: 'object', properties: { promoted: { type: 'boolean' }, decision: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['promoted', 'decision', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags, forceHarness) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const harn = forceHarness ? (HARNESS + ' ') : ''
  return [
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + dir + ' ; log=' + dir + '.log ; report=' + dir + '/evaluation_report.json',
'REQUIRED FLAGS (verbatim): ' + harn + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. Already running for this V_tag -> adopt if the flags match, else kill and relaunch. 3. report exists AND the log shows the required flags -> done, do not relaunch. 4. A DIFFERENT version is running -> sleep 140 (up to 3x), then report blocked.',
'5. Otherwise write ' + dir + '/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + dir + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + harn + evalFlags + " > " + dir + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log to catch an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, marker, nTotal) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const rep = dir + '/evaluation_report.json'
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + ' (expect ' + nTotal + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE. marker="' + marker + '".',
'1. If ' + rep + ' exists -> done=true. Read it: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'))" . Put num_videos in num_videos — it MUST be ' + nTotal + '; if it is not, say so loudly in note (harness bug).',
'   markers: grep -cF "' + marker + '" ' + dir + '.log (also ' + dir + '/run.log; trust the LARGER).',
'   FM20: PJ=$(grep -l ' + AB_FIELD + ' ' + dir + '/*.json 2>/dev/null | grep -v evaluation_report | wc -l) -> put PJ in paired_jsons and note "pairedJSONs=<PJ>". PJ==0 means the instrument did not persist -> VOID.',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 (up to 2x), then done=false note="running <n>/' + nTotal + '" where n = count of result JSONs. If STALLED >30min with no new JSON -> kill and relaunch the SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead with no report -> tail -30 the log. TRANSIENT (API/network) -> relaunch. CODE BUG -> crashed=true with the traceback in note.',
'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, marker_count, paired_jsons, note.',
  ].join('\n')
}
async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal, ph, forceHarness) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags, forceHarness), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: ph }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, paired_jsons: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  for (let w = 0; w < (maxWatch || 70); w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, marker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: ph }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, marker_count: -1, paired_jsons: -1, note: 'watcher cap reached' }
}
function bootPrompt(V, tag, nTotal, ph) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  return [
'You are Agent D-adjudicator for the CLEAN CGR A/B on ' + nTotal + ' videos. The run carries BOTH arms: its final output is arm-ON, and per-detection field ' + AB_FIELD + ' holds arm-OFF. This is an EXACT paired comparison — zero VLM noise. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'STEP 1 (FM20 gate): grep -l ' + AB_FIELD + ' ' + dir + '/*.json | grep -v evaluation_report | wc -l . If 0 -> ran=false, note="FM20 VOID: ' + AB_FIELD + ' absent" and STOP (do NOT substitute a cross-run comparison).',
'STEP 2 (the decisive number): ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + AB_FIELD + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P + '  -> point_delta is CGR\'s clean effect on FINAL mAP (arm-ON minus arm-OFF, including the BUE/CBT interaction); new_map is absolute mAP with CGR on; also record arm-OFF absolute mAP as ref_map.',
'STEP 3 per-category AP for Shooting/RoadAccidents/Shoplifting/Explosion on both arms (evaluate.load_ground_truth / load_predictions / compute_temporal_map, filtering gt to the videos present in the dir, and substituting ' + AB_FIELD + ' index-aligned for the arm-OFF pass). Put the 4 per-cat deltas in per_category.',
'STEP 4 count-neutrality: mean detections-per-video must be IDENTICAL across arms (CGR relocates, never adds) -> set count_neutral.',
(nTotal === N_FULL ? 'STEP 5 (290 only, free sanity check): CGR is gated to the 4 hard cats, so on Normal and other non-hard videos arm-OFF MUST be byte-identical to arm-ON. Verify and report how many non-hard videos differ — anything other than 0 means the gating leaks and the result is suspect.' : ''),
'Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, per_category, count_neutral, note (VERDICT line verbatim).',
  ].join('\n')
}

// ============================ MAIN ============================
log('v16 starting. ONE question: does CGR-M raise FINAL mAP under a clean A/B? Instrument = single-run DUAL-ARM (arm-OFF = BUE+CBT on the pre-CGR snapshot; arm-ON = CGR then BUE+CBT) -> exact paired, zero VLM noise, captures the BUE/CBT interaction the offline method missed. Gate: paired delta >= +' + MIN_PAIRED_DELTA + ' with P >= ' + MIN_P + '. concurrency ' + CONCURRENCY + '.')

// ---------- PHASE 1: implement the dual-arm instrument ----------
phase('Implement')
const impl = await retryAgent(() => agent([
'You are Agent D1. Implement the DUAL-ARM A/B instrument for CGR-M, then PROVE it correct. This is an instrumentation task: the mechanism already exists — you are building the measurement. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'IMPLEMENT:',
'1. Back up every file you touch to ' + ROOT + '/.qoder/backups/' + V_IMPL + '/ FIRST, and ls the backup dir to confirm.',
'2. Add ONE new config flag `cgr_ab_diag` (default False) + CLI `--cgr_ab_diag` in BOTH run.py and batch_evaluate.py, threaded through VADConfig.from_args like the existing cgr_enabled. Flag-off must be byte-identical to the committed pipeline.',
'3. In pipeline.py at the CGR site (~L1790, immediately before the pre_bue snapshot), when cgr_ab_diag AND cgr_enabled are both on:',
'   a. import copy; base = copy.deepcopy(final_anomalies)   # the pre-CGR state',
'   b. ARM-OFF: run the SAME downstream chain the committed pipeline runs — BUE then CBT (and CCR only if ccr_enabled, which is OFF) — on copy.deepcopy(base). Capture the resulting [[t_start,t_end],...] list.',
'   c. ARM-ON: the normal path — CGR, then the same downstream chain — producing the run\'s real final_anomalies.',
'   d. Persist the arm-OFF list as a per-detection field named EXACTLY ' + AB_FIELD + ', index-aligned with anomalies and the SAME LENGTH (CGR is count-neutral). Copy the write path used by pre_bue_boundaries / pre_cbt_boundaries in models.py to_dict() so it survives serialization.',
'   THE HAZARD: those operators mutate a["t_start"]/a["t_end"] IN PLACE. Deep-copy before EACH arm or the arms silently corrupt each other.',
'   LOGGING: the arm-OFF pass will re-run BUE/CBT and would double-fire their [BUE]/[CBT-VID] markers, corrupting FM5 counts. Suppress logging during the arm-OFF pass (or tag it distinctly) and say in notes exactly what you did.',
'4. Emit an unconditional per-video marker `[CGR-AB] vid=... ndet=<n> armoff_len=<n> fired=<0|1>` INCLUDING on the empty/0-detection path (FM17), with the string EXACTLY as you report in activation_marker.',
'5. Bump the pipeline.py header to ' + V_IMPL + '. Do NOT touch BUE/CBT/recal/CCR defaults or the v167 CGR method itself — only ADD the diagnostic.',
'',
'HARD SELF-TEST (mandatory — paste ACTUAL command output, do not describe it). A false pass here wastes a multi-hour eval, and v165 already cost this project a loop by claiming a self-test it had not run:',
'  Run a REAL smoke: ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V_IMPL + '_selftest --max_per_category 1 --concurrency 1 ' + HARNESS + ' ' + MECH_BASE + ' --cgr_relocate --cgr_ab_diag',
'  Then CONFIRM ALL FOUR, pasting output for each:',
'  (a) grep -cF "<your marker>" the log > 0;',
'  (b) load a produced *.json and print ' + AB_FIELD + ' with real values, AND assert len(' + AB_FIELD + ') == len(anomalies);',
'  (c) ARM INDEPENDENCE: find a video where the marker shows fired=0 and show arm-OFF is byte-identical to the final boundaries; find one where fired=1 and show they DIFFER. If every smoke video has fired=0, widen the smoke (e.g. --max_per_category 2) until at least one fire is observed — an all-fired=0 smoke proves nothing about the ON arm;',
'  (d) count-neutrality: ndet identical in both arms.',
'  Set selftest_passed=true ONLY if (a)-(d) all literally pass. If any fails, FIX and rerun.',
'VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; both new flags appear in --help; ScoringConfig() BUE/CBT/recal/CGR/CCR defaults UNCHANGED; from_args(Namespace()).scoring == ScoringConfig().',
'Return implemented, selftest_passed, instrument_selftest (pasted evidence), deepcopy_proof, activation_marker, eval_flags, files_changed, validation, notes.',
'The self-test smoke is a few videos at concurrency 1 (~5-10 min) — that is expected and allowed. No OTHER Bash > ~150s.',
].join('\n'), { schema: implSchema, label: 'D1-implement-' + V_IMPL, phase: 'Implement' }))

let boot88 = null, boot290 = null, full88 = null, full290 = null
let dualArmOk = !!(impl && impl.implemented && impl.selftest_passed && impl.activation_marker)
if (!dualArmOk) {
  log('Phase 1: dual-arm instrument NOT verified (' + (impl ? ('selftest=' + impl.selftest_passed) : 'null') + ') -> falling back to two REAL separate 88-video evals (cross-run A/B)')
} else {
  log('Phase 1: dual-arm instrument verified. marker="' + impl.activation_marker + '"')
}

// ---------- PHASE 2: the decisive 88-video paired A/B ----------
phase('Verify88')
if (dualArmOk) {
  full88 = await runEval(V_IMPL, 'ab88', 999, impl.eval_flags, impl.activation_marker, 70, N_HARD, 'Verify88', true)
  log('Verify88: done=' + full88.done + ' crashed=' + full88.crashed + ' mAP=' + full88.mAP + ' n=' + (full88.num_videos != null ? full88.num_videos : '?') + ' markers=' + full88.marker_count + ' pairedJSONs=' + (full88.paired_jsons != null ? full88.paired_jsons : '?') + ' ' + (full88.note || ''))
  if (full88.done && !full88.crashed && full88.marker_count > 0 && !(full88.paired_jsons != null && full88.paired_jsons <= 0)) {
    boot88 = await retryAgent(() => agent(bootPrompt(V_IMPL, 'ab88', N_HARD, 'Verify88'), { schema: bootSchema, label: 'D-boot-ab88', phase: 'Verify88' }))
  } else {
    log('Verify88: eval unusable -> no adjudication')
  }
} else {
  // FALLBACK: two real runs, cross-run judged
  const onFlags = MECH_BASE + ' --cgr_relocate'
  const offFlags = MECH_BASE
  const onRun = await runEval('v169fb', 'on88', 999, onFlags, '[CGR-VID]', 70, N_HARD, 'Verify88', true)
  const offRun = await runEval('v169fb', 'off88', 999, offFlags, '[CBT-VID]', 70, N_HARD, 'Verify88', true)
  log('Fallback A/B: on mAP=' + onRun.mAP + ' off mAP=' + offRun.mAP)
  if (onRun.done && offRun.done && !onRun.crashed && !offRun.crashed) {
    boot88 = await retryAgent(() => agent([
'You are Agent D-adjudicator for a CROSS-RUN CGR A/B at 88 videos (the dual-arm instrument could not be verified, so this arm carries VLM sampling noise — say so plainly in note). Bash < 150s. StructuredOutput EXACTLY ONCE.',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_v169fb_on88 --ref ' + ROOT + '/results/batch_eval_v169fb_off88 --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P,
'Also per-category AP for the 4 hard cats on both dirs, and mean detections-per-video for each (count-neutrality). Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, per_category, count_neutral, note (VERDICT verbatim + an explicit caveat that a cross-run comparison is noisier than the paired design).',
    ].join('\n'), { schema: bootSchema, label: 'D-boot-fallback88', phase: 'Verify88' }))
  }
}
const pass88 = !!(boot88 && boot88.ran && boot88.point_delta >= MIN_PAIRED_DELTA && boot88.p_gt0 >= MIN_P)
log('88 verdict: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3) + ' absON=' + boot88.new_map.toFixed(5) + ' absOFF=' + (boot88.ref_map != null ? boot88.ref_map.toFixed(5) : '?') + ' -> ' + (pass88 ? 'PASSES gate' : 'FAILS gate')) : 'no valid adjudication'))

// ---------- PHASE 3: 290 confirm (only if 88 passed) ----------
phase('Confirm290')
if (pass88 && dualArmOk) {
  log('88 cleared the gate -> measuring the same dual-arm A/B at full 290 (ship scale; non-hard videos must be arm-identical)')
  full290 = await runEval(V_290, 'ab290', 999, impl.eval_flags, impl.activation_marker, 130, N_FULL, 'Confirm290', false)
  log('Confirm290: done=' + full290.done + ' crashed=' + full290.crashed + ' mAP=' + full290.mAP + ' n=' + (full290.num_videos != null ? full290.num_videos : '?') + ' ' + (full290.note || ''))
  if (full290.done && !full290.crashed && !(full290.paired_jsons != null && full290.paired_jsons <= 0)) {
    boot290 = await retryAgent(() => agent(bootPrompt(V_290, 'ab290', N_FULL, 'Confirm290'), { schema: bootSchema, label: 'D-boot-ab290', phase: 'Confirm290' }))
  }
} else {
  log('Skipping the 290 confirm: the 88 A/B did not clear the gate, so there is nothing to promote.')
}

// ---------- PHASE 4: promote or archive-close ----------
phase('Decide')
const ok290 = !boot290 ? null : (boot290.ran && boot290.point_delta >= -0.003 && boot290.new_map >= (REF290_MAP - 0.008))
const promote = !!(pass88 && boot290 && boot290.ran && ok290)
const decision = await retryAgent(() => agent([
'You are Agent C. Execute the promotion decision for CGR-M and write the session record. Be scrupulously honest: a clean negative that CLOSES this lead is a successful outcome, and this project has already been burned once by an over-optimistic CGR verdict. Plain text; no schema fields beyond those requested.',
'', CONTEXT, '',
'EVIDENCE:',
'  instrument: ' + (dualArmOk ? 'single-run DUAL-ARM (exact paired, zero VLM noise)' : 'FALLBACK cross-run two-eval A/B (noisier)'),
'  88 A/B: ' + JSON.stringify(boot88),
'  290 A/B: ' + JSON.stringify(boot290),
'  gate: paired delta >= +' + MIN_PAIRED_DELTA + ' AND P >= ' + MIN_P + ' at 88; at 290 delta >= -0.003 AND abs >= ' + (REF290_MAP - 0.008).toFixed(5),
'  computed: pass88=' + pass88 + ' ok290=' + String(ok290) + ' -> promote=' + promote,
'',
'DO:',
promote
 ? ('1. PROMOTE. Back up config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/cgr_promote/ , then set cgr_enabled default True in ALL of: the ScoringConfig dataclass, the from_args getattr default, and set_defaults in BOTH run.py and batch_evaluate.py (keep --no_cgr_relocate as the opt-out). REPLACE the stale inflated-figure comment on the cgr_enabled line with the real measured numbers from this session. VALIDATE and paste output: py_compile; ScoringConfig().cgr_enabled is True; bue/cbt/cbt_recal still True; ccr_enabled still False; import MultiAnomalyVAD.')
 : ('1. DO NOT PROMOTE — leave cgr_enabled default-OFF and CONFIRM it: paste output of ScoringConfig().cgr_enabled showing False, plus bue/cbt/recal still True. Mark CGR-M as MEASURED-CLOSED in the record so no future session re-litigates it: state the clean measured delta, its CI and P, and that the closure came from an exact paired end-to-end A/B rather than a noisy proxy.'),
'2. Turn the diagnostic OFF by default (cgr_ab_diag stays default False) and confirm it. Leave the instrument in the tree — it is reusable for any future pre-BUE operator, and note that in the record.',
'3. date +%Y%m%d ; write ' + LOGS + '/summary_v16_' + V_IMPL + '-' + V_290 + '_<YYYYMMDD>.md covering: the instrument and WHY it is sound (deterministic post-VLM chain -> exact paired arms) and why v15\'s pre-BUE-snapshot figure was inflated; the 88 and 290 numbers with CI/P and per-category deltas; count-neutrality and (at 290) the non-hard arm-identity check; the decision; and the updated CLOSED-families list.',
'4. APPEND ledger rows to ' + LOGS + '/session_v97_ledger.md for ' + V_IMPL + ' (the 88 A/B) and, if it ran, ' + V_290 + ' (the 290 confirm): loop|version|title+verdict|smoke|mAP|AP@0.5|AP@0.7|lat|kept. Use -1 for absent values. Do NOT rewrite existing rows.',
'5. Reply with a 12-line recap leading with the single headline number (CGR\'s clean effect on final mAP) and the decision.',
'Return promoted, decision, validation (pasted), note.',
].join('\n'), { schema: decideSchema, label: 'C-decide-cgr', phase: 'Decide' }))

log('v16 complete. instrument=' + (dualArmOk ? 'dual-arm(paired)' : 'fallback(cross-run)') +
    ' | 88: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3)) : 'n/a') +
    ' | 290: ' + (boot290 && boot290.ran ? ('delta=' + boot290.point_delta.toFixed(5) + ' abs=' + boot290.new_map.toFixed(5)) : 'n/a') +
    ' | promoted=' + promote)
return {
  instrument: dualArmOk ? 'dual-arm-paired' : 'fallback-cross-run',
  selftest: impl ? impl.selftest_passed : null,
  ab88: boot88, ab290: boot290, pass88, promote,
  decision: decision ? decision.decision : null,
}
