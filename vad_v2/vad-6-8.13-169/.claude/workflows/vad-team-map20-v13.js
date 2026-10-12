export const meta = {
  name: 'vad-team-map20-v13',
  description: 'VAD agent team v13 — FOUR effective loops (v160->v163) that build ON the v155 CBT win to push the 4 hard categories further, then a paired-290 verify to decide CBT default-ON promotion. Committed = v137 + BUE (default-ON) + v155 CBT (code present, default-OFF; --cbt_tighten). CBT (Category-conditioned Boundary Tightening, post-BUE inward clamp) is the current hard-harness best: 88-video mAP 0.09970 (paired +0.05446, P=0.997), driven almost entirely by RoadAccidents (AP 0.067->0.202); dominant hard-cat miss = BOUNDARY-too-wide, not cand-gen. v12 left THREE strong leads blocked by INSTRUMENT bugs, not merit: v158 CBT-RECAL (per-category GT-p75 caps) hit hard mAP 0.12085 cross-run but its paired field pre_recal_boundary was never written -> void; v157 CBT-T (trigger 1.5->1.1) never launched its 88-eval (smoke-gate string/category mismatch). v13 reclaims these with HARD instrument discipline (FM20 paired-field-written self-test, FM21 marker-grep self-test), attacks the one hard-cat miss CBT cannot fix (Shoplifting candidate-generation, cand-miss 11/25), and ends with a paired-290 CBT verify. All evals run on the 88-video hard harness (--categories Shooting,RoadAccidents,Shoplifting,Explosion) WITH --cbt_tighten. Concurrency 2. NONE of the 10 banned families.',
  phases: [
    { title: 'Research', detail: 'A reclaims/strengthens CBT recalibration (per-cat caps + trigger); B attacks Shoplifting candidate-generation (the miss CBT cannot fix); each proposes one' },
    { title: 'Debate', detail: 'synthesize one implementable CBT-building mechanism per loop' },
    { title: 'Review', detail: 'C gates on FM1-FM21 + banned families + INSTRUMENT self-tests (paired field written, marker greps) + >= +0.015 absolute over 0.0997' },
    { title: 'Implement', detail: 'D implements + INSTRUMENT self-test, smoke(4) -> full 88-video eval WITH --cbt_tighten -> paired bootstrap + per-category -> keep/revert' },
    { title: 'Verify290', detail: 'paired-290 verify of the best CBT variant (pre_cbt_boundaries) -> promote cbt_enabled default-ON iff net-neutral-or-positive on 290' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 4
const START_VERSION = 160
const SMOKE_GATE = 0.02
const MAP_TOL = 0.005
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS = '--categories ' + HARD_CATS
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten'   // committed BUE+Qscore + CBT ON; loops add their variant flag
const N_HARD = 88
// current CBT-on hard-harness best (v155). Reused as REF — no baseline re-run.
let   REF_DIR = ROOT + '/results/batch_eval_v155_full'
let   REF_MAP = 0.09970
let   FLOOR = 0.09470                         // REF_MAP - 0.005 (do-not-regress below current CBT)
const MIN_ABS_IMPROVE = 0.015                 // KEEP requires new hard mAP >= REF_MAP + 0.015 (beat current CBT by a margin)
const MIN_P = 0.90
const CBT_PAIRED_FIELD = 'pre_cbt_boundaries' // clean total-CBT effect vs BUE-only (paired, zero VLM noise)
const REF290_DIR = ROOT + '/results/batch_eval_full290_bue'
const REF290_MAP = 0.15114

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test = temporal segment proposals). Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! OBJECTIVE: push the 4 HARD categories (Shooting, RoadAccidents, Shoplifting, Explosion; 88-video hard harness, --categories ' + HARD_CATS + ') ABOVE the current CBT best. Committed = v137 + BUE (default-ON) + v155 CBT (--cbt_tighten, code present default-OFF). CURRENT CBT-ON HARD BASELINE = mAP ' + REF_MAP + ' (AP@0.3 0.2002 / AP@0.5 0.0762 / AP@0.7 0.0227; dir ' + REF_DIR + '). All v13 evals run WITH ' + MECH_BASE + ' (CBT ON) + the loop variant. KEEP requires new hard mAP >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' (beat current CBT by >= ' + MIN_ABS_IMPROVE + ') AND paired P(delta>0) >= ' + MIN_P + ' AND >= floor ' + FLOOR.toFixed(5) + '. !!!',
'',
'!!! WHAT WE KNOW ABOUT CBT + THE HARD CATEGORIES (v155 diagnosis, per cat nGT/hit@.3/hit@.5/boundary-miss/cand-miss) !!!',
'Shooting 25/9/7/11/5, RoadAccidents 23/6/0/13/4, Shoplifting 25/3/2/11/11, Explosion 22/10/7/11/1. Dominant miss = BOUNDARY-too-wide (~46 GT segs adjacent-but-too-wide). CBT (inward clamp, count-neutral, confidence-frozen, post-BUE terminal) fixed much of RoadAccidents (over-wide dets -> IoU hits: AP 0.067->0.202). REMAINING headroom: (1) CBT RECALIBRATION — v158 per-category GT-p75 caps {RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5} + lower over-wide trigger 1.5->1.1 hit hard mAP 0.12085 cross-run (RoadAccidents +0.0197) but was VOID (paired field never written); (2) Shoplifting CANDIDATE-GENERATION — cand-miss 11/25 is the HIGHEST and CBT (boundary-only) CANNOT fix it: the GT is elsewhere in the video, the pipeline never proposes a window there.',
'',
'!!! v12 LOST 3 OF 5 LOOPS TO INSTRUMENTATION, NOT IDEAS — HARD DISCIPLINE THIS SESSION (FM20/FM21) !!!',
'FM20 PAIRED-FIELD-NOT-WRITTEN (v158 killer): the per-fire pre-mechanism field MUST be written into EVERY result JSON, else the paired bootstrap cannot resample and the loop is VOID. D1 MUST self-test: after implementing, run a 1-2 video mechanism smoke and grep a produced result JSON for the field; do NOT return implemented=true until the field is confirmed present. Reuse the WORKING pattern: BUE writes pre_bue_boundaries and CBT writes ' + CBT_PAIRED_FIELD + ' correctly — copy that exact write path.',
'FM21 SMOKE-GATE STRING/CATEGORY MISMATCH (v157 killer): the activation marker string must EXACTLY match what the orchestrator greps, and must appear on the hard-category smoke set (not some other category). D1 MUST self-test the marker grep on the actual smoke output before returning. Every per-video marker fires on the EMPTY/0-detection path (FM17).',
'',
'!!! ADMISSIBLE LEVERS (build on CBT; NONE of the 10 banned families) !!!',
'- CBT RECALIBRATION: expose/adjust per-category caps + the over-wide trigger multiplier (v158/v157 reclaim); MUST write ' + CBT_PAIRED_FIELD + ' (or a new correctly-written paired field) and pass the FM20 self-test.',
'- Shoplifting (and Shooting) CANDIDATE-GENERATION: category-aware scan density / segment proposal in anomaly_discovery.py so the pipeline PROPOSES a window where the GT actually is (CBT cannot rescue a video with zero overlap). Count-effect must be argued (FM2 — additive dets demote precision unless they land on GT).',
'- CATEGORY-SPECIFIC SCORE/QUERY quality for these cats (CLIP prompts/CoOp) — improves the underlying signal (NOT curve re-rank, banned 1).',
'',
'!!! BANNED / CLOSED FAMILIES (auto C-REJECT) !!! 1 curve within-video re-rank; 2 VLM merge/batch; 3 max_scan/scan-tail; 4 per-video normalization; 5 dense proposal append; 6 compute trims; 7 boundary OUTWARD-growth=BUE; 8 scan_segment-ABS-conf gates; 9 Normal-FP suppression for mAP; 10 MIL-relocation on the 32-snippet substrate. NOTE: v159 PMX (max-over-6-prompts category logit) is measured-CLOSED (-0.035, amplifies flat-curve noise) — do not re-propose prompt max-pooling.',
'',
'!!! MEASUREMENT !!! 88-video bootstrap 90% CI half-width ~+/-0.02. Verdict = video-matched bootstrap. For CBT-variant loops PREFER paired-json ' + CBT_PAIRED_FIELD + ' (measures the TOTAL CBT-variant effect vs BUE-only, clean+large, easily clears P>=0.9) and read new_map (ABSOLUTE hard mAP) as the headline — KEEP iff new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '. For candidate-generation loops that change the detection SET, cross-run vs ' + REF_DIR + ' (the CBT baseline). Report per-category AP for the 4 cats. Tool: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> [--paired-json ' + CBT_PAIRED_FIELD + ' | --ref ' + REF_DIR + '].',
'REUSABLE: analysis_v140_livecurve_lib.py; analysis_v138_rankdiag_orch.py; analysis_bootstrap_ab.py. v155 CBT method _cbt_category_boundary_tighten (pipeline.py ~L2279, called post-BUE ~L1723); CBT_CAPS={RoadAccidents:6,Explosion:13,Shoplifting:10,Shooting:14}; over-wide trigger det_dur>1.5*cap; writes ' + CBT_PAIRED_FIELD + '. features/train+test CLIP snippets; model_mil_v6.pth; prompt_weights.json.',
'',
'FM1 curve-shape edits regress. FM2 added/count -> FPs under global AP. FM5 never-activated. FM6 offline gains do not transfer. FM8 one change/loop. FM9 MPS .detach().cpu().numpy(). FM13 bound ceiling first. FM16 noise floor. FM17 empty-video marker. FM18 target ABSOLUTE mAP. FM19 count-neutral position moves not floor-safe. FM20 paired-field-written (self-test). FM21 marker-grep/category self-test.',
'',
'HARD CONSTRAINTS: ONE change/loop; ARCHITECTURE-LEVEL; per-category effect argument (which hard cat gains, expected hard-harness new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '); config flag (default OFF, flag-off byte-identical to committed BUE+CBT-on-when-flagged) + CLI in run.py AND batch_evaluate.py + marker on EMPTY-video path; do NOT disturb BUE (default-ON), CEX (default-OFF), or the committed v155 CBT method (only ADD recal params gated by a new flag). Eval FIXED --concurrency ' + CONCURRENCY + ' (NOT 3). StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s; Read offset/limit; short progress line.',
].join('\n')

const history = [
  { loop: 5, version: 'v142', title: 'BUE — SHIPPED default-ON', full_map: 0.21932, kept: true, lessons: 'Committed. 290-paired +0.02164 P=0.938.' },
  { loop: 17, version: 'v155', title: 'CBT Category-conditioned Boundary Tightening — KEPT (code, default-OFF)', full_map: 0.09970, kept: true, lessons: 'Post-BUE inward clamp; hard mAP 0.0493->0.0997 (paired +0.05446 P=0.997); RoadAccidents AP 0.067->0.202. 290 confirm 0.14836 vs 0.15114 (-0.0028, within noise). Dominant hard miss = boundary-too-wide.' },
  { loop: 20, version: 'v158', title: 'CBT-RECAL per-cat GT-p75 caps + trigger 1.1 — VOID (instrument)', full_map: 0.12085, kept: false, lessons: 'Hard mAP 0.12085 cross-run (RoadAccidents +0.0197) BUT paired field pre_recal_boundary NEVER written -> bootstrap void. REAL LEAD, reclaim with FM20 discipline.' },
  { loop: 19, version: 'v157', title: 'CBT-T trigger 1.5->1.1 — VOID (instrument)', full_map: -1, kept: false, lessons: '88-eval never launched (smoke-gate string/category mismatch, FM21). Lever OPEN.' },
  { loop: 21, version: 'v159', title: 'PMX prompt max-pool — CLOSED (measured -0.035)', full_map: 0.06450, kept: false, lessons: 'Real negative; max-over-6-prompts amplifies flat-curve noise. Do not re-propose.' },
]
let bestFullMap = REF_MAP
let bestVersion = 'v155-CBT'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": hard mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED: v137 + BUE(default-ON) + CBT(code,default-OFF). Current CBT-on hard baseline mAP ' + REF_MAP.toFixed(5) + ' (floor ' + FLOOR.toFixed(5) + '; KEEP target >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '). full-290 committed 0.15114.\nHistory:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string' },
    target_family: { type: 'string', description: 'cbt-recalibration | category-candidate-gen | category-score-quality (NOT any of the 10 banned; NOT prompt max-pool)' },
    target_categories: { type: 'string' },
    expected_hardmap: { type: 'number', description: 'Honest expected ABSOLUTE hard-harness (88-video) mAP; must be >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) },
    per_category_arithmetic: { type: 'string', description: 'which hard cat gains, how many videos gain a hit, why new_map beats ' + REF_MAP },
    instrument_plan: { type: 'string', description: 'FM20: exact paired field name + the EXACT write path copied from BUE/CBT; FM21: exact marker string. The self-test D1 will run to confirm both.' },
    banned_family_check: { type: 'string' },
    count_effect: { type: 'string', description: 'FM2 — count-neutral preferred; if additive (cand-gen), the FP bound' },
    implementation_plan: { type: 'string', description: 'exact files/functions/lines, config field, CLI flags, marker incl EMPTY-video path' },
    paired_measurable: { type: 'boolean' },
    fm_risk_table: { type: 'string' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string' },
  },
  required: ['title', 'mechanism', 'target_family', 'target_categories', 'expected_hardmap', 'per_category_arithmetic', 'instrument_plan', 'banned_family_check', 'count_effect', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}
const verdictSchema = { type: 'object', properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' }, effect_size_ok: { type: 'boolean' }, banned_family_ok: { type: 'boolean' }, instrument_ok: { type: 'boolean', description: 'FM20/FM21: is there a concrete paired-field write path (copied from BUE/CBT) + exact marker string + a self-test?' } }, required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'instrument_ok'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'ACTUAL output proving the paired field is written to a produced JSON AND the marker greps on the mechanism smoke (FM20/FM21)' }, selftest_passed: { type: 'boolean' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string' }, paired_marker: { type: 'string' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'activation_marker', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, latency: { type: 'number' }, marker_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
const resultSchema = { type: 'object', properties: { version: { type: 'string' }, implemented: { type: 'boolean' }, smoke_map: { type: 'number' }, full_map: { type: 'number' }, kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' }, report_path: { type: 'string' } }, required: ['version', 'implemented', 'full_map', 'kept', 'summary', 'lessons'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}
function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
  return [
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' runs WITH THE REQUIRED FLAGS, return IMMEDIATELY. SHORT; Bash < 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=<same>.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (verbatim): ' + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. running for this V_tag -> adopt if flags present else kill&relaunch. 3. report exists+flags in log -> exists. 4. DIFFERENT version running -> sleep 140 up to 3x else blocked.',
'5. Else record <output_dir>/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + ROOT + "/results/batch_eval_" + V + "_" + tag + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + evalFlags + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 log for Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, marker, nTotal, pairedField) {
  const rep = ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json'
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const N = nTotal || N_HARD
  return [
'You are Agent D-watcher for ' + tag + ' eval of ' + V + ' (' + N + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE. report=' + rep + ' log=' + dir + '.log inner=' + dir + '/run.log marker="' + (marker || '') + '".',
'1. report exists -> done=true: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti[\'avg_per_video_seconds\'])" ; markers: grep -cF "' + (marker || '') + '" ' + dir + '.log (also inner run.log; trust LARGER).',
(pairedField ? '   ALSO (FM20) confirm the paired field is written: grep -l ' + pairedField + ' ' + dir + '/*.json | grep -v evaluation_report | wc -l — put the count in note as "pairedJSONs=<n>".' : ''),
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 up to 2x, then done=false note="running <n>/' + N + '". STALL >30min no growth -> kill & relaunch SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead no report -> tail -30. TRANSIENT -> relaunch. CODE BUG -> crashed=true. FM17 (markers<succeeded) -> crashed=true note="FM17".',
'Return done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
  ].join('\n')
}
async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal, pairedField, ph) {
  const phase_ = ph || 'Implement'
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: phase_ }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  const cap = maxWatch || 60
  for (let w = 0; w < cap; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, marker, nTotal, pairedField), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: phase_ }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap' }
}
function bootstrapPrompt(V, pairedMarker) {
  const dir = ROOT + '/results/batch_eval_' + V + '_full'
  const paired = pairedMarker && pairedMarker.length > 0
  return [
'You are Agent D-adjudicator for ' + V + ' on the 88-video hard harness. Run the bootstrap CI AND per-category AP. Bash < 150s. StructuredOutput EXACTLY ONCE.',
paired
  ? ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + pairedMarker + ' --min-delta 0.02 --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5) + '  (new_map = ABSOLUTE hard mAP; this is the KEEP headline vs ' + REF_MAP + ')')
  : ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta 0.02 --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5)),
'Then per-category AP for Shooting/RoadAccidents/Shoplifting/Explosion (evaluate.load_ground_truth/load_predictions/compute_temporal_map filtered by filename prefix, ' + dir + ' vs ' + REF_DIR + '). Put the 4 per-cat deltas in per_category.',
'Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map (ABSOLUTE hard mAP), per_category, note (VERDICT verbatim).',
  ].join('\n')
}
function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + '. GOAL: an IMPLEMENTABLE mechanism (building on the committed BUE+CBT pipeline) that raises the 88-video hard-harness ABSOLUTE mAP to >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' (current CBT-on baseline ' + REF_MAP.toFixed(5) + '). ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK: 1. DIAGNOSE with the CACHED runs (results/batch_eval_v155_full = CBT-on, results/batch_eval_v158_full = CBT-recal 0.121) + analysis_v140_livecurve_lib.py (no API): confirm where the residual hard-cat headroom is per category. 2. >=3 real WebSearch queries relevant to your lever (if unavailable, say so). 3. Derive ONE IMPLEMENTABLE mechanism (NOT one of the 10 banned families, NOT prompt max-pool): config flag (default OFF, flag-off byte-identical) + CLI + marker on EMPTY-video path; a CORRECTLY-WRITTEN paired field copied from the BUE/CBT write path (FM20); per-category arithmetic to new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '; the exact instrument self-test. Prefer count-neutral (FM2). Return via StructuredOutput (EXACTLY ONCE).',
'!!! ANTI-STALL: no Bash > ~60s, Read offset/limit, short progress line, finish in <18 tool calls.',
  ].join('\n')
}
function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). Implement the APPROVED proposal exactly, then RUN THE INSTRUMENT SELF-TEST. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED:\n' + verdict.mandated_changes : '',
'STEPS:',
'1. mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy every file you will edit BEFORE editing.',
'2. Implement. config.py default OFF/prior + from_args threading; CLI in BOTH run.py and batch_evaluate.py; bump pipeline.py header to ' + V + '. Do NOT disturb BUE (default-ON), CEX (default-OFF), or the v155 CBT method (only ADD gated params). Copy the EXACT paired-field write path from CBT (' + CBT_PAIRED_FIELD + ') / BUE (pre_bue_boundaries) so FM20 cannot recur.',
'3. FM5+FM17+FM21: unique INFO marker per fire AND a per-video summary marker on the EMPTY/0-detection path, string EXACTLY as reported in activation_marker.',
'4. INSTRUMENT SELF-TEST (MANDATORY, FM20/FM21 — the 3 v12 losses were here): run a REAL 2-video mechanism smoke: ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V + '_selftest --max_per_category 1 --concurrency 1 ' + HARNESS + ' ' + MECH_BASE + ' <your_flag> ; then CONFIRM (a) grep -cF "<marker>" the log > 0, and (b) grep -l "<paired_field>" the produced *.json > 0. Put the ACTUAL counts in instrument_selftest and set selftest_passed. If either is 0, FIX and rerun; do NOT return selftest_passed=true until both pass.',
'5. VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; flags in --help; ScoringConfig() BUE/CBT/CEX defaults unchanged; from_args(Namespace()).scoring==ScoringConfig().',
'6. Return implemented, selftest_passed, instrument_selftest, activation_marker, eval_flags (the FULL flag string incl ' + MECH_BASE + ' + your flag), paired_marker (the paired field name), files_changed, validation, notes.',
'ANTI-STALL: the self-test smoke is 2 videos at concurrency 1 (~5 min) — that is allowed; no OTHER Bash > ~150s.',
  ].join('\n')
}
function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record honestly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '', 'IMPL:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL(88): ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot) + '\nDECISION: ' + decisionText + (fm5 ? '\nFM5/FM17/FM20/FM21 VIOLATION.' : ''),
'TASKS: 1. If REVERT: restore from ' + ROOT + '/.qoder/backups/' + V + '/, diff -q, confirm 0 mechanism symbols + BUE default-ON/CBT method intact/CEX default-OFF. If KEEP: leave code, flip the new flag default ON if the proposal says so. 2. Archive ' + V + '_hard_eval.json/_run.log/bootstrap_ab.json into ' + LOGS + '/results/. 3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date +%Y%m%d): mechanism, per-cat diagnosis, hard bootstrap CI + ABSOLUTE new_map vs ' + REF_MAP + ' + per-cat deltas as headline, INSTRUMENT self-test result, FM table, WON/OPEN/CLOSED, next-loop guidance. 4. APPEND one ledger row to ' + LOGS + '/session_v97_ledger.md (loop|version|title+verdict|smoke|hard_mAP|AP@0.5|AP@0.7|lat|kept). -1 absent. Do NOT rewrite rows. 5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v13 starting at v' + START_VERSION + '. Building ON committed BUE+CBT. CBT-on hard baseline ' + REF_MAP + ' (REF=v155_full, no re-run). KEEP target >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '. ' + MAX_LOOPS + ' loops + paired-290 verify. concurrency ' + CONCURRENCY + '. HARD instrument discipline (FM20/FM21) after v12 lost 3 loops to it.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' ===')

  phase('Research')
  const focusA = (loop <= 2)
    ? 'YOUR ANGLE: RECLAIM + strengthen CBT RECALIBRATION (v158 hit hard mAP 0.12085 cross-run but was VOID on FM20). Expose per-category caps + the over-wide trigger multiplier as config, set to the v158 GT-p75 values (RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0, Shooting 13.5; trigger 1.1) or better, and WRITE THE PAIRED FIELD CORRECTLY (copy CBT/BUE write path). This is the highest-probability win.'
    : 'YOUR ANGLE: iterate on the best CBT-recal variant so far, or category-specific score/query quality for the still-weak cats (Shooting/Explosion), staying count-neutral + paired.'
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', focusA), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: Shoplifting (and Shooting) CANDIDATE-GENERATION — cand-miss is highest for Shoplifting (11/25) and CBT (boundary-only) CANNOT fix it: the pipeline never proposes a window where the GT is. Add category-aware scan density / segment proposal in anomaly_discovery.py so a window is proposed at the right place. Argue the FM2 count/FP effect carefully (additive dets that miss GT demote precision — v143 died there).'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Synthesize ONE IMPLEMENTABLE mechanism for this loop (FM8). Prefer the higher-probability CBT-recalibration lever early; keep count-neutral where possible; MUST carry a correct paired-field write path (FM20) and exact marker (FM21).',
'', CONTEXT, '', 'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'Kill anything with expected hard new_map < ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' or any of the 10 banned families. StructuredOutput EXACTLY ONCE.',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the quality gate. Each 88-video eval ~2h AND v12 wasted 3 loops on instrument bugs — so the INSTRUMENT gate is as important as the effect gate. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'CHECKLIST: 1. IMPLEMENTABLE + expected hard new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' with per-cat arithmetic -> effect_size_ok. 2. INSTRUMENT (FM20/FM21): concrete paired-field write path copied from BUE/CBT + exact marker string + a stated self-test? -> instrument_ok. If missing -> REJECT (this is what killed v157/v158). 3. BANNED FAMILIES 1-10 + not prompt max-pool (v159 closed) -> banned_family_ok. 4. FM2 count/FP for cand-gen. 5. FM8 one change; BUE default-ON/CBT method/CEX untouched. REJECT with concrete feedback or APPROVE (optionally mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (r' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 180))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. C REJECTED. Revise or pivot (CBT-recal or category cand-gen/score-quality; none of the 10 banned; correct paired field FM20 + exact marker FM21). ONE change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'REJECTED:\n' + JSON.stringify(proposal, null, 1), '', "C FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) { proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED:\n' + verdict.mandated_changes }); log('Loop ' + loop + ': proceeding with C-mandated revision') }
    else { log('Loop ' + loop + ': C rejected 3x - skipping'); history.push({ loop: loop + 21, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: (verdict.feedback || '').slice(0, 300) }); continue }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker || impl.selftest_passed === false) {
    log('Loop ' + loop + ': impl failed / self-test not passed (FM20/FM21) - skipping eval to avoid a void loop')
    history.push({ loop: loop + 21, version: V, title: proposal.title + ' [IMPL/SELFTEST FAILED]', full_map: -1, kept: false, lessons: impl ? ('selftest=' + impl.selftest_passed + ' ' + (impl.instrument_selftest || impl.notes || '').slice(0, 200)) : 'null' }); continue
  }

  const pf = impl.paired_marker || CBT_PAIRED_FIELD
  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker, 25, 4, pf, 'Implement')
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count + ' ' + (smoke.note || ''))
  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (smoke)' }
  let boot = null, decisionText = 'REVERT', keep = false, fm5 = false
  if (smoke.done && !smoke.crashed) {
    if (smoke.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from smoke') }
    else {
      full = await runEval(V, 'full', 999, impl.eval_flags, impl.activation_marker, 70, N_HARD, pf, 'Implement')
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' markers=' + full.marker_count + ' ' + (full.note || ''))
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from full') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, pf), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        if (!boot || !boot.ran) decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'null') + ' — check FM20 paired field written)'
        else {
          const okImprove = boot.new_map >= (REF_MAP + MIN_ABS_IMPROVE), okP = boot.p_gt0 >= MIN_P, okF = boot.new_map >= FLOOR
          if (okImprove && okP && okF) { keep = true; decisionText = 'KEEP (hard mAP ' + boot.new_map.toFixed(5) + ' >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' [beats CBT ' + REF_MAP + '], paired P=' + boot.p_gt0.toFixed(3) + '. per-cat ' + (boot.per_category || '') + '. NEW hard baseline.)' }
          else if (!okF) decisionText = 'REVERT (floor breached: ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          else if (!okImprove) decisionText = 'REVERT (did not beat current CBT by ' + MIN_ABS_IMPROVE + ': hard mAP ' + boot.new_map.toFixed(5) + ' < ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '; per-cat ' + (boot.per_category || '') + ')'
          else decisionText = 'REVERT (paired P=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ',' + boot.ci90_high + '])'
        }
      }
    }
  } else log('Loop ' + loop + ': smoke failed')
  if (fm5) decisionText = 'REVERT (FM5/FM17: marker never appeared)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [hardmAP=' + boot.new_map.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ']') : ''
  history.push({ loop: loop + 21, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) { bestFullMap = result.full_map; bestVersion = V; REF_DIR = ROOT + '/results/batch_eval_' + V + '_full'; REF_MAP = result.full_map; FLOOR = Math.max(0.09470, result.full_map - MAP_TOL); log('Loop ' + loop + ': KEPT ' + V + ' hard mAP ' + result.full_map + ' (new REF, floor ' + FLOOR.toFixed(5) + ')') }
  else log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 150))
}

// ============================ FINAL: paired-290 verify of the best CBT variant -> promotion decision ============================
phase('Verify290')
log('=== FINAL: paired-290 verify of the best CBT variant (' + bestVersion + ') to decide cbt_enabled default-ON ===')
const evalFlags290 = MECH_BASE + (bestVersion !== 'v155-CBT' && bestVersion !== 'v155' ? '' : '')  // best variant flags are already committed in the tree if KEPT; CBT flag drives it
const v290 = await runEval('v13best', 'verify290', 999, MECH_BASE, '[CBT]', 130, 290, CBT_PAIRED_FIELD, 'Verify290')
let boot290 = null
if (v290 && v290.done && !v290.crashed) {
  boot290 = await retryAgent(() => agent([
'You are Agent D-adjudicator for the FINAL 290-video CBT verify. Paired bootstrap on ' + CBT_PAIRED_FIELD + ' isolates CBT\'s marginal effect on the full 290 (vs pre-CBT / BUE-only windows), zero VLM noise. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_v13best_verify290 --paired-json ' + CBT_PAIRED_FIELD + ' --min-delta 0.0 --min-p 0.5 --floor 0.0',
'Report ran, mode, point_delta (CBT marginal on 290), ci90_low, ci90_high, p_gt0, new_map (ABSOLUTE 290 mAP with CBT), note (VERDICT verbatim + the pre-CBT/BUE-only 290 mAP if printed).',
  ].join('\n'), { schema: bootSchema, label: 'D-bootstrap-290-cbt', phase: 'Verify290' }))
}
let cbtPromote = false, promoteDecision = 'leave CBT default-OFF'
if (boot290 && boot290.ran) {
  // promote iff CBT does NOT hurt the full 290 (marginal >= -noise) AND absolute 290 mAP is not below the committed 0.15114 by more than noise
  const notWorse = boot290.point_delta >= -0.005 && boot290.new_map >= (REF290_MAP - 0.008)
  if (notWorse) { cbtPromote = true; promoteDecision = 'PROMOTE CBT default-ON: 290 paired marginal ' + boot290.point_delta.toFixed(5) + ', abs 290 mAP ' + boot290.new_map.toFixed(5) + ' (committed ' + REF290_MAP + ') — net-neutral-or-positive, and it doubles the hard categories.' }
  else promoteDecision = 'leave CBT default-OFF: 290 paired marginal ' + boot290.point_delta.toFixed(5) + ' / abs ' + boot290.new_map.toFixed(5) + ' shows a real full-290 cost.'
} else promoteDecision = 'leave CBT default-OFF (290 verify inconclusive: ' + (v290 ? v290.note : 'eval failed') + ')'
log('CBT 290 verdict: ' + promoteDecision)

phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the v13 END-OF-SESSION SUMMARY (v160-v163 + CBT 290 verify). Plain text; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '', 'CBT 290 VERIFY: ' + JSON.stringify(boot290) + '\nPROMOTION DECISION: ' + promoteDecision, '',
'TASK: 1. If the decision is PROMOTE: back up config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/cbt_promote/, set cbt_enabled default True (config dataclass + from_args getattr default + set_defaults in both runners; keep a --no_cbt_tighten opt-out if present), VALIDATE (py_compile + ScoringConfig().cbt_enabled is True + BUE still True). Else leave default-OFF, confirm ScoringConfig().cbt_enabled is False. 2. Read ledger tail + new v160-v163 idea files. 3. date +%Y%m%d; write ' + LOGS + '/summary_v13_v160-v163_<YYYYMMDD>.md: which loops raised hard mAP above ' + REF_MAP + ' (per-cat), the CBT 290 verify verdict + promotion decision, updated FM1-FM21 taxonomy, next directions. 4. APPEND ledger rows for KEEPs + the CBT@290 verify. 5. Reply with a 12-line recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v13 complete. Best hard-harness ' + bestVersion + ' mAP ' + bestFullMap.toFixed(5) + '. CBT 290: ' + promoteDecision + '. ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, refMap: REF_MAP, cbtPromote, promoteDecision, verify290: boot290 ? boot290.new_map : null }
