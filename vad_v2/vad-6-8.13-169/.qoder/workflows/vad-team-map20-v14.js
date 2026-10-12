export const meta = {
  name: 'vad-team-map20-v14',
  description: 'VAD agent team v14 — settle the CBT-recal @290 promotion FIRST (zero-API offline paired reconstruction on the cached full290_bue boundaries + a hard-vs-nonhard cross-run decomposition), then THREE effective loops on the one lever CBT cannot fix: category-aware CANDIDATE-GENERATION for the cand-miss hard videos (Shoplifting 11/25, Shooting 5/25 — pipeline never proposes a window where the GT is). Committed = v137 + BUE (default-ON) + v155 CBT (code, default-OFF; --cbt_tighten) + v160 CBT-recal (code, default-OFF; --cbt_recal). KEY FINDING v14 opens on: v160_full (--bue_extension --qscore_rerank --cbt_tighten --cbt_recal, 290) = mAP 0.17254 vs committed full290_bue (BUE+qscore only) = 0.15114 = +0.0214 @290 — a POTENTIAL SECOND SHIPPABLE WIN, but confounded (CBT-recal never wrote its paired field, FM20). v14 phase-0 resolves it cleanly offline (CBT/CBT-recal boundary transforms are deterministic given the pre-CBT BUE windows + cached CLIP category curves) -> promote cbt_enabled+cbt_recal default-ON iff paired @290 marginal is net-neutral-or-positive. CBT anchor micro-variants (v161 CBT-TW, v162 CBT-SA/FA, v163 CBT-SP/CA) are ALL measured-CLOSED — do NOT re-propose anchor/seed re-centering. All loop evals on the 88-video hard harness (--categories Shooting,RoadAccidents,Shoplifting,Explosion) WITH --cbt_tighten --cbt_recal. Concurrency 2. NONE of the 10 banned families.',
  phases: [
    { title: 'Adjudicate290', detail: 'zero-API: offline-reconstruct the paired CBT/CBT-recal effect on cached full290_bue boundaries + hard-vs-nonhard cross-run decomposition -> promote CBT-recal default-ON iff net-neutral-or-positive @290' },
    { title: 'Research', detail: 'A + B each derive one category-aware CANDIDATE-GENERATION mechanism (propose a window where the cand-miss GT actually is); CBT anchor variants are CLOSED' },
    { title: 'Debate', detail: 'synthesize one implementable cand-gen mechanism per loop' },
    { title: 'Review', detail: 'C gates on FM1-FM21 + banned families + FM2 count/FP bound (additive dets that miss GT demote precision) + INSTRUMENT self-tests + >= +0.015 absolute over the CBT-recal hard baseline' },
    { title: 'Implement', detail: 'D implements + INSTRUMENT self-test, smoke(4) -> full 88-video eval WITH --cbt_tighten --cbt_recal -> paired/cross-run bootstrap + per-category -> keep/revert' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 3
const START_VERSION = 164
const MAP_TOL = 0.005
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS = '--categories ' + HARD_CATS
// committed BUE+Qscore + CBT + CBT-recal ON; loops add their cand-gen variant flag
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const N_HARD = 88
// hard-harness CBT-on REF (v155_full is the last clean @88 CBT dir; v160 recal ran @290 only, no clean @88).
// Conservative: KEEP must beat this by >= MIN_ABS_IMPROVE.
let   REF_DIR = ROOT + '/results/batch_eval_v155_full'
let   REF_MAP = 0.09970
let   FLOOR = 0.09470                          // REF_MAP - 0.005 (do-not-regress below current CBT)
const MIN_ABS_IMPROVE = 0.015                  // KEEP requires new hard mAP >= REF_MAP + 0.015
const MIN_P = 0.90
const CBT_PAIRED_FIELD = 'pre_cbt_boundaries'  // committed v155 CBT writes this; recal path must too (FM20)
// --- @290 promotion inputs (all CACHED — zero API) ---
const REF290_DIR = ROOT + '/results/batch_eval_full290_bue'   // committed BUE+qscore, 0.15114
const REF290_MAP = 0.15114
const CBT290_DIR = ROOT + '/results/batch_eval_v160_full'     // BUE+qscore+CBT+recal, 0.17254 (confounded)
const CBT290_MAP = 0.17254

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test = temporal segment proposals). Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! OBJECTIVE: raise the 4 HARD categories (Shooting, RoadAccidents, Shoplifting, Explosion; 88-video hard harness, --categories ' + HARD_CATS + ') via the ONE lever CBT cannot fix — category-aware CANDIDATE-GENERATION. Committed = v137 + BUE (default-ON) + v155 CBT (--cbt_tighten) + v160 CBT-recal (--cbt_recal), CBT code default-OFF. CURRENT CBT-ON HARD BASELINE = mAP ' + REF_MAP + ' (dir ' + REF_DIR + '). All v14 loop evals run WITH ' + MECH_BASE + ' (CBT+recal ON) + the loop cand-gen variant. KEEP requires new hard mAP >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' AND paired/cross-run P(delta>0) >= ' + MIN_P + ' AND >= floor ' + FLOOR.toFixed(5) + '. !!!',
'',
'!!! WHY CANDIDATE-GENERATION (not more CBT) — v155 per-cat diagnosis nGT/hit@.3/hit@.5/boundary-miss/cand-miss !!!',
'Shooting 25/9/7/11/5, RoadAccidents 23/6/0/13/4, Shoplifting 25/3/2/11/11, Explosion 22/10/7/11/1. CBT (inward clamp, count-neutral, post-BUE) already reclaimed the BOUNDARY-too-wide miss (RoadAccidents AP 0.067->0.202). What remains and CBT PHYSICALLY CANNOT FIX = CAND-MISS: the pipeline never PROPOSES a window overlapping the GT (Shoplifting 11/25 highest, Shooting 5/25). A boundary operator cannot rescue a video with zero candidate overlap. The lever: category-aware scan density / segment proposal in anomaly_discovery.py so a window is proposed where the GT actually is. THIS IS THE UNEXPLORED FRONTIER.',
'',
'!!! FM2 IS THE CAND-GEN KILLER (v143 died here) — count/FP discipline is MANDATORY !!!',
'Under global-ranking temporal AP, EVERY added detection that does NOT land on GT DEMOTES precision at the recall levels the curve reaches. A cand-gen mechanism that proposes 3 extra windows to catch 1 GT usually NETS NEGATIVE. Admissible cand-gen must be either (a) count-neutral (RELOCATE an existing low-value candidate to the GT-likely region rather than ADD), or (b) additive with a hard, argued FP bound (the added window fires on <= K videos and lands on GT with high precision). D1 MUST instrument the count delta (dets added/relocated per video) and the paired field.',
'',
'!!! BANNED / CLOSED FAMILIES (auto C-REJECT) !!! 1 curve within-video re-rank; 2 VLM merge/batch; 3 max_scan/scan-tail; 4 per-video normalization; 5 dense proposal append (UNCONDITIONAL dense-append is banned — a TARGETED, category-gated, count-bounded relocation/proposal is NOT this, but must carry the FM2 bound); 6 compute trims; 7 boundary OUTWARD-growth=BUE; 8 scan_segment-ABS-conf gates; 9 Normal-FP suppression for mAP; 10 MIL-relocation on the 32-snippet substrate. ALSO CLOSED: v159 PMX prompt max-pool (-0.035); v161 CBT-TW / v162 CBT-SA / v162 CBT-FA / v163 CBT-SP / v163 CBT-CA CBT anchor/seed re-centering micro-variants (all measured net-negative or floor-failing) — do NOT re-propose CBT contrast-window anchor/seed changes.',
'',
'!!! INSTRUMENT DISCIPLINE (FM20/FM21 — v12 lost 3 loops, v160 recal @290 lost its paired field) !!!',
'FM20 PAIRED-FIELD-NOT-WRITTEN: the per-fire pre-mechanism field MUST be written into EVERY result JSON. For a cand-gen mechanism, write a field recording the pre-mechanism candidate/detection set (or a count marker) so the effect is measurable. Copy the WORKING write path: BUE writes pre_bue_boundaries, CBT writes ' + CBT_PAIRED_FIELD + '. D1 MUST self-test: run a 1-2 video mechanism smoke and grep a produced JSON for the field BEFORE returning implemented=true.',
'FM21 SMOKE-GATE STRING/CATEGORY MISMATCH: the activation marker string must EXACTLY match what the orchestrator greps, and must appear on the hard-category smoke set. Every per-video marker must ALSO fire on the EMPTY/0-detection path (FM17). D1 MUST self-test the marker grep on the actual smoke output.',
'',
'!!! MEASUREMENT !!! 88-video bootstrap 90% CI half-width ~+/-0.02. Verdict = video-matched bootstrap. Cand-gen CHANGES the detection SET, so pure paired-json is not always clean — cross-run vs ' + REF_DIR + ' (the CBT baseline) is the primary; if a count-neutral relocation variant, ALSO report paired. Report per-category AP for the 4 cats + the count delta. Tool: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> [--ref ' + REF_DIR + ' | --paired-json <field>].',
'REUSABLE: analysis_v140_livecurve_lib.py; analysis_v138_rankdiag_orch.py; analysis_bootstrap_ab.py. CBT method _cbt_category_boundary_tighten (pipeline.py ~L2279, post-BUE ~L1723); CBT_CAPS={RoadAccidents:6,Explosion:13,Shoplifting:10,Shooting:14}, recal caps {RoadAccidents:4.5,Explosion:13.0,Shoplifting:9.0,Shooting:13.5} trigger 1.1 behind --cbt_recal; writes ' + CBT_PAIRED_FIELD + '. Candidate generation lives in anomaly_discovery.py. features/train+test CLIP snippets; model_mil_v6.pth; prompt_weights.json.',
'',
'FM1 curve-shape edits regress. FM2 added/count -> FPs under global AP (THE cand-gen killer). FM5 never-activated. FM6 offline gains do not transfer. FM8 one change/loop. FM9 MPS .detach().cpu().numpy(). FM13 bound ceiling first. FM16 noise floor. FM17 empty-video marker. FM18 target ABSOLUTE mAP. FM19 count-neutral position moves not floor-safe. FM20 paired-field-written (self-test). FM21 marker-grep/category self-test.',
'',
'HARD CONSTRAINTS: ONE change/loop; ARCHITECTURE-LEVEL; per-category effect argument (which hard cat gains, how many videos gain a hit, expected hard-harness new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '); config flag (default OFF, flag-off byte-identical to committed BUE+CBT+recal-on-when-flagged) + CLI in run.py AND batch_evaluate.py + marker on EMPTY-video path; do NOT disturb BUE (default-ON), CEX (default-OFF), the v155 CBT method, or the v160 recal params (only ADD a new gated cand-gen param). Eval FIXED --concurrency ' + CONCURRENCY + ' (NOT 3). StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s; Read offset/limit; short progress line.',
].join('\n')

const history = [
  { loop: 5, version: 'v142', title: 'BUE — SHIPPED default-ON', full_map: 0.21932, kept: true, lessons: 'Committed. 290-paired +0.02164 P=0.938.' },
  { loop: 17, version: 'v155', title: 'CBT Category-conditioned Boundary Tightening — KEPT (code, default-OFF)', full_map: 0.09970, kept: true, lessons: 'Post-BUE inward clamp; hard mAP 0.0493->0.0997 (paired +0.05446 P=0.997); RoadAccidents AP 0.067->0.202. Dominant remaining hard miss = boundary-too-wide (CBT fixed) + CAND-MISS (CBT cannot fix).' },
  { loop: 22, version: 'v160', title: 'CBT-RECAL per-cat GT-p75 caps + trigger 1.1 — KEPT (code, default-OFF; --cbt_recal)', full_map: 0.11806, kept: true, lessons: 'Hard mAP ~0.118. @290 (v160_full) = 0.17254 vs committed 0.15114 = +0.0214 BUT confounded (paired field pre_cbt not written on recal path, FM20). v14 phase-0 resolves offline.' },
  { loop: 24, version: 'v161', title: 'CBT-TW tight-target width cap — CLOSED', full_map: 0.16852, kept: false, lessons: '@290 0.16852 < v160 0.17254. CBT anchor/width micro-variant net-negative.' },
  { loop: 25, version: 'v162', title: 'CBT-SA/FA smoothed/free anchor — CLOSED', full_map: -1, kept: false, lessons: 'anchor re-centering measured net-negative. Do not re-propose.' },
  { loop: 26, version: 'v163', title: 'CBT-SP/CA smoothed/centroid anchor — CLOSED (hard 0.06626 << floor)', full_map: 0.06626, kept: false, lessons: 'ABSOLUTE hard 0.06626 << KEEP floor. CBT anchor family is CLOSED.' },
]
let bestFullMap = REF_MAP
let bestVersion = 'v160-CBTrecal'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": hard mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED: v137 + BUE(default-ON) + CBT(code,default-OFF) + CBT-recal(code,default-OFF). Current CBT-on hard baseline mAP ' + REF_MAP.toFixed(5) + ' (floor ' + FLOOR.toFixed(5) + '; KEEP target >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '). full-290 committed 0.15114; CBT-recal @290 (confounded) 0.17254.\nHistory:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string' },
    target_family: { type: 'string', description: 'category-candidate-gen (RELOCATE preferred, or additive-with-FP-bound) | category-score-quality (NOT any of the 10 banned; NOT CBT anchor re-centering; NOT prompt max-pool)' },
    target_categories: { type: 'string' },
    expected_hardmap: { type: 'number', description: 'Honest expected ABSOLUTE hard-harness (88-video) mAP; must be >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) },
    per_category_arithmetic: { type: 'string', description: 'which hard cat gains, how many cand-miss videos gain a GT-overlapping window, why new_map beats ' + REF_MAP },
    count_effect: { type: 'string', description: 'FM2 (THE killer) — is it count-neutral (relocate) or additive? if additive, the exact FP bound: on how many videos does it add a window, and the precision argument that it lands on GT' },
    instrument_plan: { type: 'string', description: 'FM20: exact paired/count field name + the EXACT write path copied from BUE/CBT; FM21: exact marker string incl EMPTY-video path. The self-test D1 will run.' },
    banned_family_check: { type: 'string', description: 'explicitly rule out families 1-10, CBT anchor family, and prompt max-pool' },
    implementation_plan: { type: 'string', description: 'exact files/functions/lines (anomaly_discovery.py for cand-gen), config field, CLI flags, marker incl EMPTY-video path' },
    paired_measurable: { type: 'boolean' },
    fm_risk_table: { type: 'string' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string' },
  },
  required: ['title', 'mechanism', 'target_family', 'target_categories', 'expected_hardmap', 'per_category_arithmetic', 'count_effect', 'instrument_plan', 'banned_family_check', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}
const verdictSchema = { type: 'object', properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' }, effect_size_ok: { type: 'boolean' }, banned_family_ok: { type: 'boolean' }, count_fp_ok: { type: 'boolean', description: 'FM2: is the count/FP effect count-neutral OR additive with a concrete argued bound?' }, instrument_ok: { type: 'boolean', description: 'FM20/FM21: concrete paired/count-field write path (copied from BUE/CBT) + exact marker string + a self-test?' } }, required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'count_fp_ok', 'instrument_ok'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'ACTUAL output proving the paired/count field is written to a produced JSON AND the marker greps on the mechanism smoke (FM20/FM21)' }, selftest_passed: { type: 'boolean' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string' }, paired_marker: { type: 'string' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'activation_marker', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, latency: { type: 'number' }, marker_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
const resultSchema = { type: 'object', properties: { version: { type: 'string' }, implemented: { type: 'boolean' }, smoke_map: { type: 'number' }, full_map: { type: 'number' }, kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' }, report_path: { type: 'string' } }, required: ['version', 'implemented', 'full_map', 'kept', 'summary', 'lessons'], additionalProperties: false }
const promoteSchema = { type: 'object', properties: { ran: { type: 'boolean' }, paired_marginal_290: { type: 'number', description: 'clean paired CBT/CBT-recal marginal on 290 from the offline reconstruction (deterministic transform on full290_bue boundaries)' }, paired_p_gt0: { type: 'number' }, crossrun_hard_delta: { type: 'number', description: 'v160_full vs full290_bue mAP delta restricted to the 4 hard cats' }, crossrun_nonhard_delta: { type: 'number', description: 'same delta on NON-hard cats = VLM run-noise proxy (should be ~0)' }, abs_290_with_cbt: { type: 'number' }, promote: { type: 'boolean' }, decision: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'promote', 'decision', 'note'], additionalProperties: false }

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
(pairedField ? '   ALSO (FM20) confirm the paired/count field is written: grep -l ' + pairedField + ' ' + dir + '/*.json | grep -v evaluation_report | wc -l — put the count in note as "pairedJSONs=<n>".' : ''),
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
'PRIMARY (cand-gen changes the detection set): cross-run vs the CBT baseline:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta 0.015 --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5),
paired ? ('SECONDARY (if the variant is count-neutral relocation, also clean paired): ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + pairedMarker + ' --min-delta 0.0 --min-p 0.5 --floor 0.0') : '',
'Then per-category AP for Shooting/RoadAccidents/Shoplifting/Explosion (evaluate.load_ground_truth/load_predictions/compute_temporal_map filtered by filename prefix, ' + dir + ' vs ' + REF_DIR + '). ALSO report the mean detections-per-video for ' + dir + ' vs ' + REF_DIR + ' (the FM2 count delta). Put per-cat deltas + count delta in per_category.',
'Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map (ABSOLUTE hard mAP), per_category, note (VERDICT verbatim).',
  ].join('\n')
}
function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + '. GOAL: an IMPLEMENTABLE category-aware CANDIDATE-GENERATION mechanism (building on the committed BUE+CBT+recal pipeline) that raises the 88-video hard-harness ABSOLUTE mAP to >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' (current CBT-on baseline ' + REF_MAP.toFixed(5) + ') by making the pipeline PROPOSE a window where a cand-miss GT actually is. ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK: 1. DIAGNOSE with the CACHED runs (results/batch_eval_v155_full = CBT-on; results/batch_eval_v160_full = CBT-recal @290) + analysis_v140_livecurve_lib.py + the actual candidate-generation code in anomaly_discovery.py (READ it, offset/limit): for the cand-miss videos (Shoplifting 11/25, Shooting 5/25), WHERE does the current scan propose windows and WHY does it miss the GT region? Is there signal (motion / CLIP category curve) at the GT that the current proposal threshold/density skips? 2. >=3 real WebSearch queries on category-aware temporal action proposal / weakly-supervised anomaly localization. 3. Derive ONE IMPLEMENTABLE mechanism (category-aware proposal density/threshold, or count-neutral RELOCATE of a low-value candidate to a higher-category-signal region; NOT the 10 banned, NOT CBT anchor, NOT prompt max-pool): config flag (default OFF, flag-off byte-identical) + CLI + marker on EMPTY-video path; a CORRECTLY-WRITTEN paired/count field (FM20); per-category arithmetic to new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '; the EXACT FM2 count/FP bound; the exact instrument self-test. Return via StructuredOutput (EXACTLY ONCE).',
'!!! ANTI-STALL: no Bash > ~60s, Read offset/limit, short progress line, finish in <18 tool calls.',
  ].join('\n')
}
function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). Implement the APPROVED cand-gen proposal exactly, then RUN THE INSTRUMENT SELF-TEST. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED:\n' + verdict.mandated_changes : '',
'STEPS:',
'1. mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy every file you will edit BEFORE editing.',
'2. Implement (cand-gen usually in anomaly_discovery.py). config.py default OFF/prior + from_args threading; CLI in BOTH run.py and batch_evaluate.py; bump pipeline.py header to ' + V + '. Do NOT disturb BUE (default-ON), CEX (default-OFF), the v155 CBT method, or the v160 recal params (only ADD a new gated cand-gen param). Copy the EXACT paired/count-field write path from CBT (' + CBT_PAIRED_FIELD + ') / BUE (pre_bue_boundaries) so FM20 cannot recur.',
'3. FM5+FM17+FM21: unique INFO marker per fire AND a per-video summary marker on the EMPTY/0-detection path, string EXACTLY as reported in activation_marker.',
'4. INSTRUMENT SELF-TEST (MANDATORY, FM20/FM21): run a REAL 2-video mechanism smoke: ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V + '_selftest --max_per_category 1 --concurrency 1 ' + HARNESS + ' ' + MECH_BASE + ' <your_flag> ; then CONFIRM (a) grep -cF "<marker>" the log > 0, and (b) grep -l "<paired_or_count_field>" the produced *.json > 0. Put the ACTUAL counts in instrument_selftest and set selftest_passed. If either is 0, FIX and rerun; do NOT return selftest_passed=true until both pass.',
'5. VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; flags in --help; ScoringConfig() BUE/CBT/CEX/recal defaults unchanged; from_args(Namespace()).scoring==ScoringConfig().',
'6. Return implemented, selftest_passed, instrument_selftest, activation_marker, eval_flags (the FULL flag string incl ' + MECH_BASE + ' + your flag), paired_marker (the paired/count field name), files_changed, validation, notes.',
'ANTI-STALL: the self-test smoke is 2 videos at concurrency 1 (~5 min) — that is allowed; no OTHER Bash > ~150s.',
  ].join('\n')
}
function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record honestly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '', 'IMPL:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL(88): ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot) + '\nDECISION: ' + decisionText + (fm5 ? '\nFM5/FM17/FM20/FM21 VIOLATION.' : ''),
'TASKS: 1. If REVERT: restore from ' + ROOT + '/.qoder/backups/' + V + '/, diff -q, confirm 0 mechanism symbols + BUE default-ON/CBT method/recal params intact/CEX default-OFF. If KEEP: leave code, flip the new flag default ON if the proposal says so. 2. Archive ' + V + '_hard_eval.json/_run.log/bootstrap_ab.json into ' + LOGS + '/results/. 3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date +%Y%m%d): mechanism, per-cat cand-miss diagnosis, hard bootstrap CI + ABSOLUTE new_map vs ' + REF_MAP + ' + per-cat deltas + count delta as headline, INSTRUMENT self-test result, FM table, WON/OPEN/CLOSED, next-loop guidance. 4. APPEND one ledger row to ' + LOGS + '/session_v97_ledger.md (loop|version|title+verdict|smoke|hard_mAP|AP@0.5|AP@0.7|lat|kept). -1 absent. Do NOT rewrite rows. 5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v14 starting. Phase-0: settle CBT-recal @290 promotion offline (zero-API). Then ' + MAX_LOOPS + ' cand-gen loops from v' + START_VERSION + '. CBT-on hard baseline ' + REF_MAP + ' (REF=v155_full). KEEP target >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '. concurrency ' + CONCURRENCY + '. CBT anchor family CLOSED — cand-gen only.')

// ---------- PHASE 0: zero-API CBT-recal @290 promotion adjudication ----------
phase('Adjudicate290')
log('=== Phase-0: offline paired CBT/CBT-recal @290 reconstruction (zero API) to decide default-ON promotion ===')
const promo = await retryAgent(() => agent([
'You are Agent P, the @290 promotion adjudicator. DECIDE whether to promote CBT + CBT-recal to default-ON, using ONLY cached data + deterministic offline reconstruction — ZERO new eval/API. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'THE QUESTION: v160_full (' + CBT290_DIR + ', flags --bue_extension --qscore_rerank --cbt_tighten --cbt_recal, 290 videos) has mAP ' + CBT290_MAP + '. Committed full290_bue (' + REF290_DIR + ', flags --bue_extension --qscore_rerank, 290) has mAP ' + REF290_MAP + '. Naive delta +' + (CBT290_MAP - REF290_MAP).toFixed(5) + ' is CONFOUNDED: two different VLM runs, AND CBT-recal never wrote its paired field (FM20) so v160_full JSONs have no pre-CBT boundaries.',
'',
'DO BOTH of these zero-API analyses and REPORT both:',
'(1) CROSS-RUN DECOMPOSITION (robust, easy): compute mAP for BOTH ' + CBT290_DIR + ' and ' + REF290_DIR + ' restricted to the 4 HARD cats (Shooting/RoadAccidents/Shoplifting/Explosion) -> crossrun_hard_delta. Then restricted to the NON-hard cats -> crossrun_nonhard_delta. CBT only fires on hard cats (grep "[CBT-VID]" ... "fired=" in ' + CBT290_DIR + '.log to confirm non-hard fired=0), so crossrun_nonhard_delta is a pure VLM-RUN-NOISE proxy: it SHOULD be ~0. If crossrun_hard_delta >> crossrun_nonhard_delta and hard_delta > 0, CBT-recal genuinely helps at 290. Use evaluate.load_ground_truth/load_predictions/compute_temporal_map filtered by filename prefix.',
'(2) OFFLINE PAIRED RECONSTRUCTION (clean, preferred if feasible): CBT/CBT-recal is a DETERMINISTIC boundary transform given the pre-CBT (BUE) window + the video category curve. Write ' + LOGS + '/analysis_v164_cbt290_offline.py that, for every anomaly detection in ' + REF290_DIR + ' (these ARE the pre-CBT BUE boundaries), reconstructs the per-video category rel-curve the SAME way pipeline._cbt_category_boundary_tighten does (reuse the pipeline method / features/test CLIP snippets + prompt_weights.json), applies the committed CBT-recal transform (recal caps {RoadAccidents:4.5,Explosion:13.0,Shoplifting:9.0,Shooting:13.5}, trigger 1.1), and computes 290 mAP with the transformed boundaries vs the original. That is a PAIRED (same detection set) delta = clean CBT-recal marginal @290 with ZERO VLM noise. Paired-bootstrap it (reuse analysis_bootstrap_ab.py logic). If faithfully reconstructing the curve is infeasible in <~20 tool calls, SAY SO in note and rely on (1).',
'',
'DECISION RULE: promote=true iff (paired_marginal_290 >= -0.003 with paired_p_gt0 >= 0.5) OR (crossrun_hard_delta >= +0.010 AND crossrun_nonhard_delta is within +/-0.010 of 0). If promote=true, BACK UP config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/cbt_promote/ then set cbt_enabled=True AND cbt_recal=True as defaults (config dataclass + from_args getattr defaults + set_defaults in BOTH run.py and batch_evaluate.py; keep --no_cbt_tighten / a recal opt-out), VALIDATE (py_compile + ScoringConfig().cbt_enabled is True + cbt_recal True + BUE still True + import MultiAnomalyVAD). If promote=false, leave defaults OFF and confirm ScoringConfig().cbt_enabled is False.',
'ANTI-STALL: no Bash > ~150s; Read offset/limit; if (2) is too fiddly, deliver (1) and decide. Return ran, paired_marginal_290, paired_p_gt0, crossrun_hard_delta, crossrun_nonhard_delta, abs_290_with_cbt, promote, decision, note.',
].join('\n'), { schema: promoteSchema, label: 'P-adjudicate290', phase: 'Adjudicate290' }))
const promoteDecision = promo ? (promo.decision + ' | crossrun hard=' + promo.crossrun_hard_delta + ' nonhard=' + promo.crossrun_nonhard_delta + ' paired=' + promo.paired_marginal_290 + ' promote=' + promo.promote) : 'Phase-0 adjudicator failed'
log('CBT @290 promotion: ' + promoteDecision)

// ============================ LOOPS: candidate-generation ============================
for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' (cand-gen) ===')

  phase('Research')
  const focusA = 'YOUR ANGLE: SHOPLIFTING candidate-generation (cand-miss 11/25, the HIGHEST). The GT is a diffuse in-store event the current scan skips. Find where anomaly_discovery.py proposes windows and make it propose one at the Shoplifting GT region — PREFER a count-neutral RELOCATE of the lowest-value existing candidate to the highest category-signal region over an additive proposal (FM2).'
  const focusB = 'YOUR ANGLE: SHOOTING candidate-generation (cand-miss 5/25) + a cross-category general cand-gen improvement. Shooting is brief and can fall between scan segments. Propose category-aware scan DENSITY near high category-curve peaks, count-bounded with an explicit FM2 FP argument, OR a count-neutral relocate. Argue precision (lands on GT).'
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', focusA), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', focusB), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Synthesize ONE IMPLEMENTABLE category-aware candidate-generation mechanism for this loop (FM8). PREFER count-neutral RELOCATE over additive (FM2 is the cand-gen killer); MUST carry a correct paired/count-field write path (FM20) and exact marker (FM21).',
'', CONTEXT, '', 'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'Kill anything with expected hard new_map < ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ', any of the 10 banned families, CBT anchor re-centering, or an additive proposal without a concrete FP bound. StructuredOutput EXACTLY ONCE.',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the quality gate. Each 88-video eval ~2h; cand-gen is FM2-dangerous (v143 died adding dets). The COUNT/FP gate and the INSTRUMENT gate are as important as the effect gate. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'CHECKLIST: 1. IMPLEMENTABLE + expected hard new_map >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ' with per-cat arithmetic -> effect_size_ok. 2. FM2 COUNT/FP: count-neutral relocate OR additive with a concrete argued bound (how many videos get a window, precision it lands on GT)? -> count_fp_ok. If additive-unbounded -> REJECT. 3. INSTRUMENT (FM20/FM21): concrete paired/count-field write path copied from BUE/CBT + exact marker string + a stated self-test? -> instrument_ok. 4. BANNED FAMILIES 1-10 + NOT CBT anchor + NOT prompt max-pool -> banned_family_ok. 5. FM8 one change; BUE/CBT/recal/CEX untouched. REJECT with concrete feedback or APPROVE (optionally mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (r' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 180))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. C REJECTED. Revise or pivot (category cand-gen, count-neutral relocate preferred; none of the 10 banned; not CBT anchor; correct paired/count field FM20 + exact marker FM21; concrete FM2 bound). ONE change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'REJECTED:\n' + JSON.stringify(proposal, null, 1), '', "C FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) { proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED:\n' + verdict.mandated_changes }); log('Loop ' + loop + ': proceeding with C-mandated revision') }
    else { log('Loop ' + loop + ': C rejected 3x - skipping'); history.push({ loop: loop + 26, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: (verdict.feedback || '').slice(0, 300) }); continue }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker || impl.selftest_passed === false) {
    log('Loop ' + loop + ': impl failed / self-test not passed (FM20/FM21) - skipping eval to avoid a void loop')
    history.push({ loop: loop + 26, version: V, title: proposal.title + ' [IMPL/SELFTEST FAILED]', full_map: -1, kept: false, lessons: impl ? ('selftest=' + impl.selftest_passed + ' ' + (impl.instrument_selftest || impl.notes || '').slice(0, 200)) : 'null' }); continue
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
        if (!boot || !boot.ran) decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'null') + ')'
        else {
          const okImprove = boot.new_map >= (REF_MAP + MIN_ABS_IMPROVE), okP = boot.p_gt0 >= MIN_P, okF = boot.new_map >= FLOOR
          if (okImprove && okP && okF) { keep = true; decisionText = 'KEEP (hard mAP ' + boot.new_map.toFixed(5) + ' >= ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + ', P=' + boot.p_gt0.toFixed(3) + '. per-cat ' + (boot.per_category || '') + '. NEW hard baseline.)' }
          else if (!okF) decisionText = 'REVERT (floor breached: ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          else if (!okImprove) decisionText = 'REVERT (did not beat CBT by ' + MIN_ABS_IMPROVE + ': ' + boot.new_map.toFixed(5) + ' < ' + (REF_MAP + MIN_ABS_IMPROVE).toFixed(5) + '; per-cat ' + (boot.per_category || '') + ')'
          else decisionText = 'REVERT (P=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ',' + boot.ci90_high + '])'
        }
      }
    }
  } else log('Loop ' + loop + ': smoke failed')
  if (fm5) decisionText = 'REVERT (FM5/FM17: marker never appeared)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [hardmAP=' + boot.new_map.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ']') : ''
  history.push({ loop: loop + 26, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) { bestFullMap = result.full_map; bestVersion = V; REF_DIR = ROOT + '/results/batch_eval_' + V + '_full'; REF_MAP = result.full_map; FLOOR = Math.max(0.09470, result.full_map - MAP_TOL); log('Loop ' + loop + ': KEPT ' + V + ' hard mAP ' + result.full_map + ' (new REF, floor ' + FLOOR.toFixed(5) + ')') }
  else log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 150))
}

// ============================ FINAL SUMMARY ============================
phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the v14 END-OF-SESSION SUMMARY (CBT @290 promotion + v164-v' + (START_VERSION + MAX_LOOPS - 1) + ' cand-gen loops). Plain text; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '', 'CBT @290 PROMOTION: ' + promoteDecision + '\nFULL PROMO OBJ: ' + JSON.stringify(promo), '',
'TASK: 1. Read ledger tail + new v164+ idea files. 2. date +%Y%m%d; write ' + LOGS + '/summary_v14_' + START_VERSION + '-v' + (START_VERSION + MAX_LOOPS - 1) + '_<YYYYMMDD>.md: the CBT @290 promotion verdict + whether cbt_enabled/cbt_recal were flipped default-ON, which cand-gen loops raised hard mAP above ' + REF_MAP + ' (per-cat + count delta), updated FM1-FM21 taxonomy, updated CLOSED-families list (add any cand-gen deadends), next directions. 3. APPEND ledger rows for KEEPs + the CBT@290 promotion decision. 4. Reply with a 12-line recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v14 complete. CBT @290: ' + (promo ? ('promote=' + promo.promote) : 'failed') + '. Best hard-harness ' + bestVersion + ' mAP ' + bestFullMap.toFixed(5) + '. ' + history.length + ' recorded loops.')
return { cbtPromote: promo ? promo.promote : false, promoteDecision, bestVersion, bestFullMap, refMap: REF_MAP }
