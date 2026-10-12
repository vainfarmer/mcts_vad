export const meta = {
  name: 'vad-team-map20-v12',
  description: 'VAD agent team v12 — FIVE EFFECTIVE loops (v155->v159) that IMPLEMENT + EVALUATE real mechanisms on a NEW HARD-CATEGORY harness (the 4 categories that dominate and drag the full-290 mAP: Shooting 23 + RoadAccidents 23 + Shoplifting 21 + Explosion 21 = 88 videos, selected via the new batch_evaluate --categories filter). These 4 categories are 63% of the 290 anomaly videos yet score only cat-mAP 0.065-0.14; the committed BUE-on pipeline scores mAP 0.04928 on the 88-video hard harness (AP@0.3 0.106 / AP@0.5 0.035 / AP@0.7 0.006) — that is the baseline to RAISE. Committed pipeline = v137 + BUE (default-ON). This session is EFFORT-BIASED: unlike v11 (5 analytical closures, 0 evals), every loop here should IMPLEMENT a category-targeting mechanism and spend a real 88-video eval unless a cheap pre-check proves it catastrophic. Also produce a DESIGN deliverable of feasible NORMAL-video recognition schemes (to be tested separately later; must not lower mAP) — design only, not evaluated this session. Ends with a full-290 confirmation of any KEEP. Concurrency 2 (conc 3 thrashes 16 GB swap). Admissible levers: category-specific score/query quality (CLIP prompts/CoOp), category-aware candidate generation / scan density / boundary windows, supervised category priors — NOT the 10 closed/banned families (curve re-rank / VLM-merge / scan-tail / per-video-norm / dense-append / compute-trim / boundary-growth=BUE / scan_segment-abs-conf / Normal-FP-suppression-for-mAP / MIL-relocation-on-32-snippet).',
  phases: [
    { title: 'Baseline', detail: 'fresh BUE-on 88-video hard-category reference (REF + per-category baseline)' },
    { title: 'Research', detail: 'A attacks WHERE/score-quality for the 4 hard categories; B attacks candidate-generation/boundary for them + drafts feasible Normal-recognition schemes (design-only); each proposes one' },
    { title: 'Debate', detail: 'synthesize one implementable category-targeting mechanism per loop' },
    { title: 'Review', detail: 'C gates on FM1-FM19 + banned families + per-category effect argument; EFFORT-BIASED: approve implementable mechanisms with a real >= +0.02 hard-harness argument' },
    { title: 'Implement', detail: 'D implements (flag + empty-video-safe marker), smoke -> full 88-video eval -> bootstrap + per-category AP -> keep/revert; final 290 confirmation' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 5
const START_VERSION = 155
const SMOKE_GATE = 0.02                    // hard-cat baseline is ~0.05, so a low smoke gate
const MAP_TOL = 0.005
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS = '--categories ' + HARD_CATS   // appended to every eval launch (88-video hard harness)
const N_HARD = 88
// Hard-category baseline (88-video slice of the committed BUE-on 290 run). Phase 0 refreshes it same-draw.
let   REF_DIR = ROOT + '/results/batch_eval_v154hard_ref_full'
let   REF_MAP = 0.04928
let   FLOOR = 0.04428                       // REF_MAP - 0.005 (do-not-regress guard); refreshed in Phase 0
const MIN_DELTA_CROSS = 0.02
const MIN_DELTA_PAIRED = 0.015
const MIN_P = 0.90
// full-290 committed references (for the final confirmation only)
const REF290_DIR = ROOT + '/results/batch_eval_full290_bue'
const REF290_MAP = 0.15114

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test = temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py; config.py; anomaly_expert.py; anomaly_discovery.py; frame_scorer.py; mil_model.py; evaluate.py; batch_evaluate.py). INTERPRETER: ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! OBJECTIVE THIS SESSION: RAISE temporal mAP on the 4 HARD CATEGORIES (Shooting, RoadAccidents, Shoplifting, Explosion) — the 88-video hard harness (batch_evaluate --categories ' + HARD_CATS + ', --max_per_category 999 = all 88). These 4 categories are 63% of the 290 anomaly videos and the binding drag on the full-290 mAP. Committed BUE-on baseline on the hard harness = mAP ' + REF_MAP + ' (AP@0.3 0.106 / AP@0.5 0.035 / AP@0.7 0.006). Committed pipeline = v137 + BUE (default-ON, --no_bue_extension to disable). !!!',
'',
'!!! EFFORT-BIASED SESSION. v11 produced 5 analytical closures with 0 evals — VALID but the owner now wants ACTION. Every loop here should IMPLEMENT a category-targeting mechanism and spend a real 88-video eval, UNLESS a cheap pre-check proves it catastrophic (floor-breaching / no-fire). Do NOT default to analytical closure. A modest measured gain that clears the gate is the goal. !!!',
'',
'!!! WHY THESE 4 CATEGORIES ARE HARD (diagnose before proposing) !!!',
'They are the largest AND lowest-AP categories (Shooting cat-mAP 0.076, RoadAccidents 0.074, Shoplifting 0.065, Explosion 0.140). Likely failure mixes per category: Shooting/Explosion/RoadAccidents are BRIEF/sudden (localization + candidate-timing hard); Shoplifting is LONG/SUBTLE (weak CLIP signal, boundary sprawl). Profile with the CACHED 290 run + analysis_v140_livecurve_lib.py: for each hard category, are the misses candidate-generation (GT elsewhere, wrong place) or boundary (adjacent but wrong extent) or ranking (right window, buried confidence)? This per-category diagnosis is the deliverable that justifies your mechanism.',
'',
'!!! ADMISSIBLE LEVERS (category-targeting; NONE of the 10 banned families) !!!',
'- CATEGORY-SPECIFIC SCORE / QUERY QUALITY: better CLIP text queries / prompt_weights / CoOp-lite embeddings for these 4 categories so the category curve localizes their events better (upstream signal quality — NOT re-ranking). prompt_weights.json (v94) is the proven-safe lineage.',
'- CATEGORY-AWARE CANDIDATE GENERATION / SCAN DENSITY: where/how densely anomaly_discovery.py directs the VLM scan for brief vs long events; category-conditioned segment proposal.',
'- CATEGORY-AWARE BOUNDARY WINDOWS: _clip_boundary_detection already adapts smoothing to expected duration — verify the per-category windows are right for these 4 (Shooting/Explosion short, Shoplifting long) and fix if mis-set.',
'- SUPERVISED CATEGORY PRIOR: model_mil_v6 or actionness as a category-conditioned candidate/confidence signal (verdict-independent, pre-calibratable). NOT MIL relocation on the 32-snippet substrate (banned 10).',
'',
'!!! BANNED / CLOSED FAMILIES (auto C-REJECT) !!!',
'1. Curve-derived within-video confidence re-ranking. 2. VLM merge/batch/contact-sheet/propagation. 3. max_scan/scan-tail/Source-2 pruning. 4. Per-video score NORMALIZATION (destroys cross-video level). 5. Dense bottom-ranked proposal append. 6. Adaptive-fps/local-compute trims. 7. Boundary outward-growth/UNION/OIC = committed BUE. 8. Additive/relocation gates on RAW scan_segment ABSOLUTE conf (v143/v144). 9. Normal-video FP suppression for the mAP objective (oracle +0.00253). 10. SUPERVISED-MIL count-neutral RELOCATION on the 32-snippet substrate (v143-v149 CLOSED: 0/17 verdict-independent, signal ceiling).',
'NOTE: category-specific SIGNAL/QUERY quality is NOT banned 1 (that is CURVE-derived within-video re-rank of the SAME signal; improving the underlying CLIP query changes the signal itself). A category-conditioned candidate generator is NOT banned 3 (that is scan-tail pruning). Argue the distinction explicitly.',
'',
'!!! NORMAL-RECOGNITION DESIGN DELIVERABLE (design-only this session; do NOT evaluate on mAP) !!!',
'Separately, draft 2-3 FEASIBLE schemes to raise Normal-video recognition WITHOUT lowering mAP (owner will test them separately later). Measured fact: perfectly deleting all Normal-video FPs is only +0.00253 mAP (banned 9 for mAP), BUT it matters for deployment false-alarm rate / frame-level AUC. Candidate ideas to flesh out: (a) a supervised video-level normalcy gate (model_mil_v6 video score below a calibrated threshold -> suppress low-conf dets, mAP-neutral by construction if it only touches sub-floor confidences); (b) a cheap normalcy VLM verification only on videos with all-low-confidence detections; (c) adding a frame-level/video-level ROC-AUC metric to evaluate.py so the effect is measurable. Deliver as a written design memo with the mAP-neutrality argument for each — NOT a code change this session.',
'',
'!!! MEASUREMENT (noise floor) !!!',
'88-video bootstrap 90% CI half-width ~+/-0.02. VERDICTS = video-matched bootstrap CIs; PAIRED preferred for score/confidence mechanisms (log pre-mechanism field), CROSS-RUN vs the fresh Phase-0 REF for candidate-generation mechanisms that change the detection SET. KEEP requires ALL: point delta >= ' + MIN_DELTA_PAIRED + ' paired (' + MIN_DELTA_CROSS + ' cross) AND P(delta>0) >= ' + MIN_P + ' AND NEW mAP >= floor (do-not-regress). Report per-category AP (the 4 hard cats) alongside overall. Tool: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> [--paired-json <FIELD> | --ref <REF>].',
'REUSABLE: analysis_v140_livecurve_lib.py (faithful live curve), analysis_v138_rankdiag_orch.py (oracle ceilings), analysis_bootstrap_ab.py. features/train (1610) + features/test (290) CLIP snippets; prompt_weights.json; model_mil_v6.pth. mil_scores at pipeline.py:1345; category-aware boundary at _clip_boundary_detection (~pipeline.py:2565).',
'',
'FAILURE MODES FM1-FM19 (abbrev): FM1 curve-shape edits regress. FM2 added/count changes -> FPs under global AP. FM5 never-activated (flag+marker+grep). FM6 offline gains do not transfer. FM8 one change/loop. FM9 MPS .detach().cpu().numpy(). FM13 bound ceiling analytically first. FM16 noise floor. FM17 EMPTY-VIDEO MARKER DEADLOCK: per-video marker MUST fire on 0-detection path. FM18 target ABSOLUTE mAP, not a low within-run arm. FM19 count-neutral position moves are not floor-safe.',
'',
'HARD CONSTRAINTS: ONE change/loop; ARCHITECTURE-LEVEL; per-category effect argument (which hard category improves, how many of its videos gain a hit, expected hard-harness dmAP >= +0.02); verdict-independent/pre-calibratable signals only; config flag (default OFF, flag-off byte-identical to committed BUE-on) + CLI in run.py AND batch_evaluate.py + unique marker on the EMPTY-video path (FM17); do NOT disturb BUE (default-ON) or CEX (default-OFF). Eval FIXED: --concurrency ' + CONCURRENCY + ' (NOT 3). StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s; Read offset/limit; short progress line between tool calls.',
].join('\n')

const history = [
  { loop: 5, version: 'v142', title: 'BUE — SHIPPED default-ON', full_map: 0.21932, kept: true, lessons: '290-paired +0.02164 P=0.938. Committed. The one durable win.' },
  { loop: 14, version: 'v149', title: 'MIL-relocation family CLOSED (32-snippet)', full_map: -1, kept: false, lessons: 'Signal ceiling: 0/17 verdict-independent recovery. Banned 10.' },
  { loop: 15, version: 'v150-154', title: 'v11 optimization-space audit — both leads oracle-closed', full_map: -1, kept: false, lessons: 'Cross-video-level calibration realizable +0.0005 (MIL presence != localization quality); finer-resolution recall achievable +0.00985 (selector ceiling). Practical ceiling on the balanced 65-harness.' },
  { loop: 16, version: 'DIAG', title: '65 vs 290 gap explained + hard-category target set', full_map: -1, kept: false, lessons: '65-harness (0.219) is an EASY balanced subset; full 290 (0.151) is dominated by 4 hard cats (Shooting 0.076/RoadAcc 0.074/Shoplift 0.065/Explosion 0.140, 63% of anomaly videos). Hard-harness baseline mAP 0.04928. v12 attacks these 4.' },
]
let bestFullMap = REF_MAP
let bestVersion = 'v137+BUE'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED: ' + bestVersion + ' (hard-harness baseline mAP ' + REF_MAP.toFixed(5) + ', floor ' + FLOOR.toFixed(5) + '; full-290 committed 0.15114).\nHistory:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string' },
    target_family: { type: 'string', description: 'category-score-quality | category-candidate-gen | category-boundary | category-supervised-prior | other (NOT any of the 10 banned)' },
    target_categories: { type: 'string', description: 'which of Shooting/RoadAccidents/Shoplifting/Explosion this helps and why' },
    expected_hardmap_gain: { type: 'number', description: 'Honest expected hard-harness (88-video) dmAP; must be >= 0.02' },
    per_category_arithmetic: { type: 'string', description: 'Which hard category improves, how many of its videos gain a hit at IoU>=0.3/0.5, why that yields >= +0.02 on the 88-video harness' },
    banned_family_check: { type: 'string', description: 'Explicit: none of the 10 banned families; for score/query levers argue why NOT banned 1' },
    count_effect: { type: 'string', description: 'Does it change detection COUNT? (FM2) count-neutral preferred; if additive, bound FP cost' },
    inv_ivo_sentinels: { type: 'string', description: 'smoke sentinels that run before the full eval' },
    implementation_plan: { type: 'string', description: 'exact files/functions/lines, config field, CLI flags, per-fire pre-mechanism field (if paired), marker incl EMPTY-video path (FM17)' },
    paired_measurable: { type: 'boolean' },
    fm_risk_table: { type: 'string' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string' },
  },
  required: ['title', 'mechanism', 'target_family', 'target_categories', 'expected_hardmap_gain', 'per_category_arithmetic', 'banned_family_check', 'count_effect', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}
const verdictSchema = { type: 'object', properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' }, effect_size_ok: { type: 'boolean' }, banned_family_ok: { type: 'boolean' }, implementable_ok: { type: 'boolean', description: 'Is this a concrete implementable mechanism (not an analysis)?' } }, required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'implementable_ok'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string' }, paired_marker: { type: 'string' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'activation_marker', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
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
function evalLaunchPrompt(V, tag, maxPerCat, evalFlags, harness) {
  return [
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT; Bash < 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (verbatim): ' + (harness || '') + ' ' + (evalFlags || ''),
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. running for this V_tag -> adopt if flags present else kill&relaunch. 3. report exists+flags in log -> launched=true note=exists. 4. DIFFERENT version running -> sleep 140 up to 3x else blocked.',
'5. Else record <output_dir>/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + ROOT + "/results/batch_eval_" + V + "_" + tag + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + (harness || '') + " " + (evalFlags || '') + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 log for Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, marker, nTotal) {
  const rep = ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json'
  const lg = ROOT + '/results/batch_eval_' + V + '_' + tag + '.log'
  const N = nTotal || N_HARD
  return [
'You are Agent D-watcher for ' + tag + ' eval of ' + V + ' (' + N + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE. report=' + rep + ' log=' + lg + ' inner=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/run.log marker="' + (marker || '') + '".',
'1. report exists -> done=true: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti[\'avg_per_video_seconds\'])" ; markers: grep -cF "' + (marker || '') + '" ' + lg + ' (also inner run.log; trust LARGER).',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 up to 2x, then done=false note="running <n>/' + N + '". STALL >30min no growth -> kill & relaunch SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead no report -> tail -30. TRANSIENT -> relaunch. CODE BUG -> crashed=true. FM17 (markers<succeeded) -> crashed=true note="FM17 deadlock".',
'Return done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
  ].join('\n')
}
async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal, harness) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags, harness), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: tag === 'base' ? 'Baseline' : 'Implement' }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  const cap = maxWatch || 60
  for (let w = 0; w < cap; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, marker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: tag === 'base' ? 'Baseline' : 'Implement' }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap' }
}
function bootstrapPrompt(V, pairedMarker) {
  const dir = ROOT + '/results/batch_eval_' + V + '_full'
  const paired = pairedMarker && pairedMarker.length > 0
  return [
'You are Agent D-adjudicator for ' + V + ' on the 88-video hard harness. Run the bootstrap CI AND report per-category AP for the 4 hard cats. Bash < 150s. StructuredOutput EXACTLY ONCE.',
paired
  ? ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + pairedMarker + ' --min-delta ' + MIN_DELTA_PAIRED + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5) + '  (if field absent try --paired-log ' + dir + '/run.log --marker "' + pairedMarker + '")')
  : ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta ' + MIN_DELTA_CROSS + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5)),
'Then per-category AP: for each of Shooting/RoadAccidents/Shoplifting/Explosion compute mAP on that category\'s videos in ' + dir + ' vs ' + REF_DIR + ' (use evaluate.load_ground_truth/load_predictions/compute_temporal_map filtered by filename prefix). Put the 4 per-category deltas in per_category.',
'If >150s re-invoke with --resamples 300. Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, per_category, note (VERDICT line verbatim).',
  ].join('\n')
}
function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. GOAL: an IMPLEMENTABLE mechanism that raises hard-harness (88-video: Shooting/RoadAccidents/Shoplifting/Explosion) temporal mAP by an HONEST >= +0.02 over the committed BUE-on baseline ' + REF_MAP.toFixed(5) + '. ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. DIAGNOSE per hard category with the CACHED 290 run (results/batch_eval_full290_bue) + analysis_v140_livecurve_lib.py (no API): for Shooting/RoadAccidents/Shoplifting/Explosion, split the misses into candidate-generation (GT elsewhere) vs boundary (adjacent) vs ranking (buried). Quantify. This justifies your mechanism.',
'2. SEARCH: >=3 real WebSearch queries relevant to your lever (category-specific CLIP prompts / actionness for brief events / weakly-supervised localization of subtle long events 2024-2026). If unavailable, say so and rely on code+data.',
'3. Derive ONE IMPLEMENTABLE category-targeting mechanism (NOT one of the 10 banned families): config flag (default OFF, flag-off byte-identical) + CLI + marker on the EMPTY-video path (FM17), per-category effect arithmetic (>= +0.02 hard-harness), paired-measurable where possible. Prefer count-neutral (FM2).',
(role === 'B' ? '4. ALSO draft the NORMAL-RECOGNITION DESIGN deliverable (2-3 feasible schemes with an mAP-neutrality argument each; design-only, not evaluated). Put it in weaknesses/notes or a short appendix in your mechanism\'s implementation_plan under "NORMAL-SCHEMES:".' : '4. Focus on the WHERE/score-quality angle for the 4 categories.'),
'5. Return via StructuredOutput (EXACTLY ONCE).',
'!!! ANTI-STALL: no Bash > ~60s, no sleep, Read offset/limit, short progress line, finish in <18 tool calls.',
  ].join('\n')
}
function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). Implement the APPROVED proposal exactly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED:\n' + verdict.mandated_changes : '',
'STEPS:',
'1. mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy every file you will edit BEFORE editing.',
'2. Implement. config.py default OFF/prior + from_args threading; CLI in BOTH run.py and batch_evaluate.py; bump pipeline.py header to ' + V + '. Do NOT disturb BUE (default-ON) or CEX (default-OFF).',
'3. FM5+FM17: unique INFO marker per fire AND a per-video summary marker on the EMPTY/0-detection path. If paired, log the per-fire pre-mechanism field index-aligned for --paired-json.',
'4. VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; flags in --help; ScoringConfig() BUE/CEX defaults unchanged; from_args(Namespace()).scoring==ScoringConfig(); empty-video marker check (FM17). Report outputs.',
'5. Return implemented, activation_marker, eval_flags, paired_marker, files_changed, validation, notes.',
'ANTI-STALL: no Bash > ~150s; Read offset/limit; NO eval here.',
  ].join('\n')
}
function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record honestly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '', 'IMPL:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL(88): ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot) + '\nDECISION: ' + decisionText + (fm5 ? '\nFM5/FM17 VIOLATION.' : ''),
'TASKS:',
'1. If REVERT: restore from ' + ROOT + '/.qoder/backups/' + V + '/, diff -q, confirm 0 mechanism symbols + BUE default-ON/CEX default-OFF. If KEEP: leave code, flip default ON if proposal says so.',
'2. Archive into ' + LOGS + '/results/: ' + V + '_hard_eval.json, ' + V + '_hard_run.log, bootstrap_ab.json (whichever exist).',
'3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date +%Y%m%d): mechanism, per-category diagnosis, hard-harness bootstrap CI + per-category AP deltas as headline, FM table, whether this category lever is WON/OPEN/CLOSED, next-loop guidance. If this loop carried the NORMAL-SCHEMES design, also write ' + LOGS + '/ideas/normal_recognition_schemes_<YYYYMMDD>.md.',
'4. APPEND one row to ' + LOGS + '/session_v97_ledger.md (loop|version|title+verdict+evidence|smoke_mAP|hard_mAP|AP@0.5|AP@0.7|latency_s|kept). -1 for absent. Do NOT rewrite rows.',
'5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ PHASE 0: FRESH 88-VIDEO HARD-CATEGORY BASELINE ============================
log('v12 starting at v' + START_VERSION + '. HARD harness = ' + HARD_CATS + ' (88 videos). Committed BUE-on hard baseline ' + REF_MAP + '. ' + MAX_LOOPS + ' EFFECTIVE loops (implement+eval). concurrency ' + CONCURRENCY + '.')
phase('Baseline')
const base = await runEval('v154hard', 'ref_full', 999, '--bue_extension --qscore_rerank', '[BUE]', 60, N_HARD, HARNESS)
if (base && base.done && !base.crashed && base.mAP >= 0) {
  REF_MAP = base.mAP; REF_DIR = ROOT + '/results/batch_eval_v154hard_ref_full'; FLOOR = Math.max(0.0, REF_MAP - MAP_TOL); bestFullMap = REF_MAP
  log('Phase 0 DONE: fresh hard-harness BUE-on baseline mAP=' + REF_MAP.toFixed(5) + ' (AP@0.3=' + base.ap30 + ' AP@0.5=' + base.ap50 + ' AP@0.7=' + base.ap70 + '). floor=' + FLOOR.toFixed(5) + '.')
} else {
  log('Phase 0 baseline FAILED (' + (base ? base.note : 'null') + '); using the 290-slice prior ' + REF_MAP + ', floor ' + FLOOR.toFixed(5) + '. Loops PREFER paired adjudication.')
}

// ============================ 5 EFFECTIVE LOOPS ============================
for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' ===')

  phase('Research')
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', 'YOUR ANGLE: WHERE the pipeline looks + SCORE/QUERY quality for the 4 hard categories — better CLIP text queries / prompt_weights / category-conditioned candidate generation so Shooting/RoadAccidents/Explosion (brief) and Shoplifting (long/subtle) events are located. NOT curve re-rank (banned 1) — improve the underlying signal.'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: candidate-generation / boundary windows / supervised category prior for the 4 hard categories (category-aware scan density; per-category boundary smoothing for brief vs long events; model_mil_v6 as a category-conditioned prior — NOT 32-snippet relocation, banned 10). ALSO deliver the NORMAL-RECOGNITION design memo.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Synthesize ONE IMPLEMENTABLE category-targeting mechanism for this loop (FM8: one change).',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'RULES: pick the ONE with the larger HONEST hard-harness gain (>= +0.02) that is concretely IMPLEMENTABLE this loop; kill anything < +0.02 or any of the 10 banned families; keep count-neutral where possible, empty-video-safe (FM17). This session is EFFORT-BIASED — prefer a real implementable mechanism over an analytical closure. StructuredOutput EXACTLY ONCE.',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the quality gate. Each 88-video eval ~2h. EFFORT-BIASED: approve concrete implementable category mechanisms with a real >= +0.02 hard-harness argument; only REJECT banned families, sub-+0.02 effects, or non-implementable analyses. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'CHECKLIST: 1. IMPLEMENTABLE concrete mechanism (not an analysis)? set implementable_ok. 2. EFFECT: expected hard-harness dmAP >= +0.02 with per-category arithmetic? set effect_size_ok. 3. BANNED FAMILIES 1-10 (esp. 1 curve re-rank vs genuine signal/query improvement; 10 MIL-relocation)? set banned_family_ok. 4. FM2 count/FP. 5. FM5+FM17 flag+CLI+empty-video marker. 6. FM8 one change, BUE default-ON/CEX default-OFF untouched. REJECT with concrete feedback or APPROVE (optionally mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (r' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 180))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. C REJECTED. Revise or pivot to another IMPLEMENTABLE category-targeting mechanism (>= +0.02 hard-harness, none of the 10 banned families). ONE change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'REJECTED:\n' + JSON.stringify(proposal, null, 1), '', "C FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) { proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED:\n' + verdict.mandated_changes }); log('Loop ' + loop + ': proceeding with C-mandated revision') }
    else { log('Loop ' + loop + ': C rejected 3x - skipping'); history.push({ loop: loop + 16, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: (verdict.feedback || '').slice(0, 300) }); continue }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker) {
    log('Loop ' + loop + ': impl failed / no marker (FM5)')
    history.push({ loop: loop + 16, version: V, title: proposal.title + ' [IMPL FAILED]', full_map: -1, kept: false, lessons: impl ? (impl.notes || '').slice(0, 200) : 'null' }); continue
  }

  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker, 25, 4, HARNESS)  // 1 per hard cat = 4 videos
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count)
  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (smoke)' }
  let boot = null, decisionText = 'REVERT', keep = false, fm5 = false
  if (smoke.done && !smoke.crashed) {
    if (smoke.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from smoke') }
    else {
      full = await runEval(V, 'full', 999, impl.eval_flags, impl.activation_marker, 70, N_HARD, HARNESS)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' markers=' + full.marker_count)
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from full') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, impl.paired_marker || ''), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        const paired = !!(impl.paired_marker && impl.paired_marker.length > 0)
        const bar = paired ? MIN_DELTA_PAIRED : MIN_DELTA_CROSS
        if (!boot || !boot.ran) decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'null') + ')'
        else {
          const okD = boot.point_delta >= bar, okP = boot.p_gt0 >= MIN_P, okF = boot.new_map >= FLOOR
          if (okD && okP && okF) { keep = true; decisionText = 'KEEP (' + (paired ? 'paired' : 'cross') + ' delta ' + boot.point_delta.toFixed(5) + ' >= ' + bar + ', P=' + boot.p_gt0.toFixed(3) + ', hard mAP ' + boot.new_map.toFixed(5) + ' >= floor ' + FLOOR.toFixed(5) + '. per-cat ' + (boot.per_category || '') + '. NEW hard baseline.)' }
          else if (!okF) decisionText = 'REVERT (floor breached: ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          else if (!okD) decisionText = 'REVERT (effect too small: ' + boot.point_delta.toFixed(5) + ' < ' + bar + '; P=' + boot.p_gt0.toFixed(3) + ')'
          else decisionText = 'REVERT (P=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ',' + boot.ci90_high + '] straddles 0)'
        }
      }
    }
  } else log('Loop ' + loop + ': smoke failed')
  if (fm5) decisionText = 'REVERT (FM5/FM17: marker never appeared / deadlocked)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5/FM17' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [d=' + boot.point_delta.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ' hardmAP=' + boot.new_map.toFixed(5) + ']') : ''
  history.push({ loop: loop + 16, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) { bestFullMap = result.full_map; bestVersion = V; REF_DIR = ROOT + '/results/batch_eval_' + V + '_full'; REF_MAP = result.full_map; FLOOR = Math.max(0.0, result.full_map - MAP_TOL); log('Loop ' + loop + ': KEPT ' + V + ' hard mAP ' + result.full_map) }
  else log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 150))
}

// ============================ FINAL: full-290 confirmation of the KEEP(s) ============================
phase('Implement')
let conf290 = null
if (bestVersion !== 'v137+BUE') {
  log('=== FINAL: full-290 confirmation of committed ' + bestVersion + ' ===')
  conf290 = await runEval(bestVersion, 'confirm290', 999, '--bue_extension --qscore_rerank', '[BUE]', 120, 290, '')
  log('290 confirmation: done=' + (conf290 && conf290.done) + ' mAP=' + (conf290 ? conf290.mAP : -1) + ' (committed 290 baseline ' + REF290_MAP + ')')
} else {
  log('No hard-category KEEP -> skipping the 290 confirmation (nothing changed).')
}

// ============================ SUMMARY ============================
phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for v12 (v155-v159 hard-category attack + Normal-recognition design). Plain text; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'290 CONFIRMATION: ' + JSON.stringify(conf290), '',
'TASK: (1) Read the ledger tail + new v155-v159 idea files + the normal_recognition_schemes memo. (2) date +%Y%m%d. (3) Write ' + LOGS + '/summary_v12_v155-v159_<YYYYMMDD>.md: executive summary (did any loop raise the hard-harness mAP >= +0.02 with P>=' + MIN_P + '? which of the 4 categories moved? did the 290 confirmation hold vs ' + REF290_MAP + '?), a per-version + per-category table, a verdict on whether each hard category is improvable or ceilinged, the Normal-recognition design schemes (with mAP-neutrality arguments), updated FM taxonomy, and next directions. (4) APPEND a ledger row only for a KEEP. (5) Reply with a 12-line recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v12 complete. Committed ' + bestVersion + ' hard-harness mAP ' + bestFullMap.toFixed(5) + '. ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, refMap: REF_MAP, floor: FLOOR, conf290: conf290 ? conf290.mAP : null }
