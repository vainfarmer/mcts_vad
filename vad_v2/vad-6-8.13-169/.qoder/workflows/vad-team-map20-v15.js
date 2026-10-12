export const meta = {
  name: 'vad-team-map20-v15',
  description: 'VAD agent team v15 — RE-DO the v165 motion-peak count-neutral candidate-RELOCATION lead that v14 buried on instrument failure, this time with correct instrumentation, then one more count-neutral cand-gen variant. Committed pipeline is NOW = v137 + BUE + CBT + CBT-recal ALL default-ON (v14 promoted CBT+recal: clean paired @290 marginal +0.02283 P=0.918; full-290 mAP ~0.173, up from BUE-only 0.15114). v14 ran 3 cand-gen loops (v164-v166) all REVERTED, but v165 CGR-M (per-frame motion-energy-peak count-neutral relocation of the lowest-conf off-GT candidate, hard cats) hit hard-subset point estimate 0.12113 (> the 0.11470 KEEP target) and was rejected ONLY because it was judged CROSS-RUN (P=0.780<0.9) instead of PAIRED — and its pre_cgr_boundaries paired field was never actually persisted to the JSONs (FM20 self-test was false). v15 fixes BOTH instrument bugs that cost v14: (1) HARNESS BUG — v165/v166 full evals ran on 290 not 88 because --categories was never forced into the eval command; v15 bakes --categories into every eval launch. (2) PAIRED ENFORCEMENT — for count-neutral relocation variants the paired-json bootstrap is MANDATORY; if the paired field is not present in the produced JSONs the loop is VOID (FM20), never silently downgraded to a noisy cross-run verdict. All loop evals on the 88-video hard harness WITH --cbt_tighten --cbt_recal. Concurrency 2. NONE of the 10 banned families; CBT anchor family + prompt max-pool CLOSED.',
  phases: [
    { title: 'Research', detail: 'A re-derives the v165 motion-peak count-neutral relocation done RIGHT (real paired field + FM20 self-test that greps a produced JSON); B derives one distinct count-neutral cand-gen variant (category-curve or dual-signal relocation)' },
    { title: 'Debate', detail: 'synthesize one implementable count-neutral relocation mechanism per loop' },
    { title: 'Review', detail: 'C gates on FM1-FM21 + banned families + FM2 (count-neutral REQUIRED — no additive) + MANDATORY real paired-field write path + harness/paired self-tests' },
    { title: 'Implement', detail: 'D implements + HARD instrument self-test (grep the produced JSON for the paired field, paste the value), smoke(4, --categories forced) -> full 88-video eval -> PAIRED bootstrap (VOID if field absent) -> keep/revert' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 2
const START_VERSION = 167
const MAP_TOL = 0.005
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS = '--categories ' + HARD_CATS
// committed defaults (all default-ON as of v14) passed explicitly = byte-identical; loops add their relocation variant flag
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const N_HARD = 88
// hard-harness baseline = committed BUE+CBT+recal pipeline. v160-recal hard estimate ~0.118 (no clean @88 dir; the
// pre-variant boundaries inside loop-1's PAIRED run give the authoritative committed hard-88 mAP for free).
let   REF_DIR = ROOT + '/results/batch_eval_v155_full'   // fallback cross-run ref only; paired is primary
let   REF_MAP = 0.11806
let   FLOOR = 0.11306                                     // REF_MAP - 0.005 (do-not-regress)
const MIN_PAIRED_DELTA = 0.008                            // KEEP: paired marginal >= +0.008 on the hard cats
const MIN_P = 0.90
// paired field is per-variant; loop-1 (motion relocation) uses pre_cgr_boundaries, loop-2 its own — D1 reports paired_marker

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test = temporal segment proposals). Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE (as of v14): v137 + BUE + CBT + CBT-recal, ALL DEFAULT-ON. full-290 mAP ~0.173 (BUE-only was 0.15114; CBT+recal added clean paired @290 +0.02283, P=0.918). ScoringConfig().bue_enabled/cbt_enabled/cbt_recal are all True. Do NOT disturb these defaults — only ADD a new gated cand-gen param (default OFF). !!!',
'',
'!!! OBJECTIVE: raise the 4 HARD categories (Shooting, RoadAccidents, Shoplifting, Explosion; 88-video hard harness, --categories ' + HARD_CATS + ') via COUNT-NEUTRAL category-aware candidate RELOCATION — the lever CBT cannot fix (cand-miss: pipeline never proposes a window where the GT is). Current hard baseline ~mAP ' + REF_MAP + '. All v15 loop evals run WITH ' + MECH_BASE + ' + the loop variant. KEEP = PAIRED marginal >= +' + MIN_PAIRED_DELTA + ' on the hard cats AND paired P(delta>0) >= ' + MIN_P + ' AND absolute hard new_map >= floor ' + FLOOR.toFixed(5) + '. !!!',
'',
'!!! THE v165 LEAD TO RECLAIM (loop 1) — and WHY v14 buried it !!!',
'v165 CGR-M: per-frame MOTION-ENERGY-peak, COUNT-NEUTRAL relocation of the LOWEST-confidence off-GT candidate onto the motion peak, gated to the 4 hard cats, pre-BUE. Hard-subset point estimate 0.12113 (> the 0.11470 target). REVERTED only because (a) it was judged CROSS-RUN P=0.780<0.9 — but a count-neutral relocation MUST be judged PAIRED (same detection count, boundaries moved) where VLM noise cancels; (b) its pre_cgr_boundaries paired field was NEVER persisted to the JSONs (verified: v165_full JSONs have zero pre_* fields), so the paired bootstrap could not run and it fell back to the noisy cross-run. RE-IMPLEMENT it with a REAL paired field and PAIRED judging. Diagnosis (per cat nGT/hit@.3/cand-miss): Shooting 25/9/5, RoadAccidents 23/6/4, Shoplifting 25/3/11, Explosion 22/10/1.',
'',
'!!! TWO INSTRUMENT BUGS v15 HARD-FIXES (they cost v14 its cand-gen loops) !!!',
'BUG-1 HARNESS: v165/v166 full evals ran on ALL 290 videos, not the 88 hard harness, because --categories was never in the eval command. v15 bakes ' + HARNESS + ' into EVERY smoke/full eval launch. Confirm n=88 in the report timing.num_videos.',
'BUG-2 PAIRED-FIELD / PAIRED-ENFORCEMENT (FM20): the count-neutral variant MUST write a per-detection pre-relocation field into EVERY result JSON, copied EXACTLY from the working BUE (pre_bue_boundaries) / CBT (pre_cbt_boundaries) write path. D1 MUST self-test by GREPPING A PRODUCED JSON and pasting the actual field value — a claim is not enough (v165 returned selftest_passed=true falsely). If the paired field is absent from the full-run JSONs, the loop is VOID (revert), NEVER downgraded to a cross-run verdict.',
'',
'!!! FM2 (THE CAND-GEN KILLER) — v15 REQUIRES COUNT-NEUTRAL !!! No additive proposals this session. The mechanism RELOCATES an existing low-value candidate (total detection count unchanged) so it cannot demote precision by adding FPs; it can only convert a wasted candidate into a GT hit. D1 must instrument that count is unchanged (ndet before == ndet after) per video.',
'',
'!!! BANNED / CLOSED FAMILIES (auto C-REJECT) !!! 1 curve within-video re-rank; 2 VLM merge/batch; 3 max_scan/scan-tail; 4 per-video normalization; 5 dense proposal append (additive — BANNED this session, count-neutral relocation is NOT this); 6 compute trims; 7 boundary OUTWARD-growth=BUE; 8 scan_segment-ABS-conf gates; 9 Normal-FP suppression for mAP; 10 MIL-relocation on the 32-snippet substrate. CLOSED: v159 PMX prompt max-pool; v161-v163 CBT anchor/seed re-centering micro-variants. Do NOT re-propose these.',
'',
'!!! MEASUREMENT !!! 88-video bootstrap 90% CI half-width ~+/-0.02. For a COUNT-NEUTRAL relocation the PAIRED-json bootstrap (variant boundaries vs the pre-relocation boundaries within the SAME run) is the PRIMARY and MANDATORY test — VLM noise cancels, so a true +0.008 marginal clears P>=0.9 easily. The pre-relocation boundaries in the run also give the committed hard-88 baseline mAP for free. Report per-category AP for the 4 cats + confirm count-neutrality (mean ndet unchanged). Tool: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-json <field> --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5) + '.',
'REUSABLE: analysis_v140_livecurve_lib.py; analysis_v138_rankdiag_orch.py; analysis_bootstrap_ab.py (supports --paired-json FIELD). Candidate generation lives in anomaly_discovery.py; the pre-BUE detection list is where a relocation applies. CBT method _cbt_category_boundary_tighten (pipeline.py ~L2279). features/train+test CLIP snippets; model_mil_v6.pth; prompt_weights.json; per-frame motion energy is already computed in the pipeline (reuse it, do not recompute).',
'',
'FM1 curve-shape edits regress. FM2 added/count -> FPs (v15 forbids additive). FM5 never-activated. FM6 offline gains do not transfer. FM8 one change/loop. FM9 MPS .detach().cpu().numpy(). FM13 bound ceiling first. FM16 noise floor. FM17 empty-video marker. FM18 target ABSOLUTE mAP (read off the paired run). FM19 count-neutral position moves not floor-safe -> the paired test + floor guard handle this. FM20 paired-field-WRITTEN-and-GREPPED (hard self-test). FM21 marker-grep/category self-test.',
'',
'HARD CONSTRAINTS: ONE change/loop; ARCHITECTURE-LEVEL; COUNT-NEUTRAL relocation only; per-category effect argument (which hard cat gains, how many cand-miss videos gain a GT-overlapping window); config flag (default OFF, flag-off byte-identical) + CLI in run.py AND batch_evaluate.py + marker on EMPTY-video path + a real per-detection paired field; do NOT disturb BUE/CBT/recal/CEX. Eval FIXED --concurrency ' + CONCURRENCY + ' on the 88 hard harness (--categories forced). StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s; Read offset/limit; short progress line.',
].join('\n')

const history = [
  { loop: 5, version: 'v142', title: 'BUE — SHIPPED default-ON', full_map: 0.21932, kept: true, lessons: 'Committed. 290-paired +0.02164 P=0.938.' },
  { loop: 17, version: 'v155', title: 'CBT — committed', full_map: 0.09970, kept: true, lessons: 'Post-BUE inward clamp; RoadAccidents AP 0.067->0.202.' },
  { loop: 22, version: 'v160', title: 'CBT-recal — committed (v14 promoted default-ON)', full_map: 0.11806, kept: true, lessons: 'Clean paired @290 +0.02283 P=0.918; full-290 ~0.173. Now DEFAULT-ON with BUE+CBT.' },
  { loop: 28, version: 'v165', title: 'CGR-M motion-peak count-neutral relocation — REVERTED on INSTRUMENT (reclaim in v15 loop 1)', full_map: 0.12113, kept: false, lessons: 'Hard point est 0.12113 (> 0.11470 target) but judged CROSS-RUN P=0.780 because pre_cgr_boundaries was never persisted (FM20). Count-neutral -> MUST judge PAIRED. REAL LEAD.' },
  { loop: 29, version: 'v166', title: 'CCR-SE category-curve relocation — REVERTED', full_map: 0.11072, kept: false, lessons: 'Category-curve relocation gated Shooting+Explosion; hard 0.11072, no paired field, cross-run not confident.' },
]
let bestFullMap = REF_MAP
let bestVersion = 'v160-committed'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": hard mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED (default-ON): BUE + CBT + CBT-recal. Hard baseline ~' + REF_MAP.toFixed(5) + ' (floor ' + FLOOR.toFixed(5) + '; KEEP = paired marginal >= +' + MIN_PAIRED_DELTA + ' P>=' + MIN_P + ' + no floor regress). full-290 ~0.173.\nHistory:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string' },
    target_family: { type: 'string', description: 'count-neutral category-aware candidate RELOCATION (NOT additive; NOT the 10 banned; NOT CBT anchor; NOT prompt max-pool)' },
    target_categories: { type: 'string' },
    expected_paired_delta: { type: 'number', description: 'Honest expected PAIRED hard-cat marginal; must be >= +' + MIN_PAIRED_DELTA },
    per_category_arithmetic: { type: 'string', description: 'which hard cat gains, how many cand-miss videos get a relocated GT-overlapping window' },
    count_neutrality_proof: { type: 'string', description: 'FM2: exactly why total detection count is unchanged (relocate not add); how D1 will assert ndet_before==ndet_after' },
    instrument_plan: { type: 'string', description: 'FM20: exact per-detection paired field name + the EXACT write path copied from BUE (pre_bue_boundaries) / CBT (pre_cbt_boundaries); the grep-a-produced-JSON self-test; FM21 exact marker incl EMPTY path' },
    banned_family_check: { type: 'string' },
    implementation_plan: { type: 'string', description: 'exact files/functions/lines (anomaly_discovery.py), config field, CLI flags in BOTH runners, marker, paired-field write' },
    paired_measurable: { type: 'boolean', description: 'MUST be true — count-neutral variants are judged paired' },
    fm_risk_table: { type: 'string' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string' },
  },
  required: ['title', 'mechanism', 'target_family', 'target_categories', 'expected_paired_delta', 'per_category_arithmetic', 'count_neutrality_proof', 'instrument_plan', 'banned_family_check', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}
const verdictSchema = { type: 'object', properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' }, effect_size_ok: { type: 'boolean' }, banned_family_ok: { type: 'boolean' }, count_neutral_ok: { type: 'boolean', description: 'FM2: is it genuinely count-neutral (relocate, not add)?' }, instrument_ok: { type: 'boolean', description: 'FM20/FM21: real per-detection paired-field write path copied from BUE/CBT + grep-a-JSON self-test + exact marker?' } }, required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'count_neutral_ok', 'instrument_ok'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'ACTUAL grep output: the paired field name AND its value from a NAMED produced JSON + marker count + ndet-neutrality check (before==after)' }, selftest_passed: { type: 'boolean' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string' }, paired_marker: { type: 'string', description: 'the per-detection paired field name that IS present in produced JSONs' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'activation_marker', 'eval_flags', 'paired_marker', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, latency: { type: 'number' }, marker_count: { type: 'number' }, num_videos: { type: 'number' }, paired_jsons: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, per_category: { type: 'string' }, count_neutral: { type: 'boolean' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
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
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' runs WITH THE REQUIRED FLAGS on the 88 HARD HARNESS, return IMMEDIATELY. SHORT; Bash < 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=<same>.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (verbatim): ' + HARNESS + ' ' + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. running for this V_tag -> adopt if flags (incl --categories) present else kill&relaunch. 3. report exists AND log shows --categories -> exists. 4. DIFFERENT version running -> sleep 140 up to 3x else blocked.',
'5. Else record <output_dir>/COMMAND.txt and launch EXACTLY (NOTE ' + HARNESS + ' IS INCLUDED — this is the harness-bug fix):',
"bash -lc 'mkdir -p " + ROOT + "/results/batch_eval_" + V + "_" + tag + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + HARNESS + " " + evalFlags + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 log for Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, marker, nTotal, pairedField) {
  const rep = ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json'
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const N = nTotal || N_HARD
  return [
'You are Agent D-watcher for ' + tag + ' eval of ' + V + ' (expect ' + N + ' videos on the hard harness). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE. report=' + rep + ' log=' + dir + '.log inner=' + dir + '/run.log marker="' + (marker || '') + '".',
'1. report exists -> done=true: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti[\'avg_per_video_seconds\'],ti.get(\'num_videos\'))" ; put num_videos in num_videos (MUST be ' + N_HARD + ' — if 290 the harness bug recurred, note it). markers: grep -cF "' + (marker || '') + '" ' + dir + '.log (also inner run.log; trust LARGER).',
(pairedField ? '   (FM20) confirm paired field present: PJ=$(grep -l ' + pairedField + ' ' + dir + '/*.json 2>/dev/null | grep -v evaluation_report | wc -l) ; put PJ in paired_jsons and note "pairedJSONs=<PJ>". PJ==0 means FM20 VOID.' : ''),
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 up to 2x, then done=false note="running <n>/' + N + '". STALL >30min no growth -> kill & relaunch SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead no report -> tail -30. TRANSIENT -> relaunch. CODE BUG -> crashed=true. FM17 (markers<succeeded) -> crashed=true note="FM17".',
'Return done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, num_videos, paired_jsons, note.',
  ].join('\n')
}
async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal, pairedField, ph) {
  const phase_ = ph || 'Implement'
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: phase_ }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, paired_jsons: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  const cap = maxWatch || 60
  for (let w = 0; w < cap; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, marker, nTotal, pairedField), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: phase_ }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, paired_jsons: -1, note: 'watcher cap' }
}
function bootstrapPrompt(V, pairedField) {
  const dir = ROOT + '/results/batch_eval_' + V + '_full'
  return [
'You are Agent D-adjudicator for ' + V + ' on the 88-video hard harness. This is a COUNT-NEUTRAL relocation -> the PAIRED bootstrap is MANDATORY and PRIMARY. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'FIRST confirm the paired field is present: grep -l ' + pairedField + ' ' + dir + '/*.json | grep -v evaluation_report | wc -l . If 0 -> set ran=false, note="FM20 VOID: ' + pairedField + ' absent" and STOP (do NOT run a cross-run fallback).',
'PAIRED: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + pairedField + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5) + ' . new_map = ABSOLUTE hard mAP with relocation; point_delta = the clean paired marginal (VLM noise cancels).',
'Then per-category AP for Shooting/RoadAccidents/Shoplifting/Explosion (' + dir + ' filtered by filename prefix) AND confirm count-neutrality: mean detections-per-video for ' + dir + ' vs the pre-relocation boundaries (should be IDENTICAL) -> set count_neutral. Put per-cat deltas in per_category.',
'Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, per_category, count_neutral, note (VERDICT verbatim).',
  ].join('\n')
}
function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + '. GOAL: an IMPLEMENTABLE COUNT-NEUTRAL category-aware candidate-RELOCATION mechanism (building on the committed BUE+CBT+recal pipeline, all default-ON) whose PAIRED hard-cat marginal is >= +' + MIN_PAIRED_DELTA + '. ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK: 1. READ the candidate-generation code in anomaly_discovery.py (offset/limit) and the per-frame motion-energy + category-curve computation in pipeline.py: for the cand-miss videos (Shoplifting 11/25, Shooting 5/25) confirm WHERE the current scan proposes windows and WHY it misses the GT, and confirm the signal (motion energy / CLIP category curve) IS elevated at the GT region so a relocation target exists. 2. >=3 real WebSearch queries on weakly-supervised temporal action proposal / anomaly localization / candidate refinement. 3. Derive ONE IMPLEMENTABLE COUNT-NEUTRAL relocation (move the lowest-value existing candidate to the highest category/motion signal region; NOT additive; NOT the 10 banned; NOT CBT anchor; NOT prompt max-pool): config flag (default OFF, byte-identical off) + CLI in BOTH runners + marker on EMPTY path + a REAL per-detection paired field copied from BUE/CBT write path; per-category arithmetic; the count-neutrality proof (ndet unchanged); the exact grep-a-JSON instrument self-test. Return via StructuredOutput (EXACTLY ONCE).',
'!!! ANTI-STALL: no Bash > ~60s, Read offset/limit, short progress line, finish in <18 tool calls.',
  ].join('\n')
}
function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). Implement the APPROVED count-neutral relocation exactly, then RUN THE HARD INSTRUMENT SELF-TEST. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED:\n' + verdict.mandated_changes : '',
'STEPS:',
'1. mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy EVERY file you will edit BEFORE editing (config.py, run.py, batch_evaluate.py, anomaly_discovery.py, pipeline.py as needed). Verify the copies exist (ls the backup dir).',
'2. Implement the relocation (usually anomaly_discovery.py, on the pre-BUE candidate list). config.py default OFF/prior + from_args threading; CLI in BOTH run.py and batch_evaluate.py; bump pipeline.py header to ' + V + '. Do NOT disturb BUE/CBT/recal/CEX defaults (all default-ON now). Copy the EXACT per-detection paired-field write path from CBT (pre_cbt_boundaries) / BUE (pre_bue_boundaries) — the field MUST be attached to each detection dict and survive to_dict() into the JSON.',
'3. FM5+FM17+FM21: unique INFO marker per fire AND a per-video summary marker on the EMPTY/0-detection path, string EXACTLY as reported in activation_marker.',
'4. HARD INSTRUMENT SELF-TEST (MANDATORY — a false pass here is what killed v165): run a REAL 2-video mechanism smoke: ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V + '_selftest --max_per_category 1 --concurrency 1 ' + HARNESS + ' ' + MECH_BASE + ' <your_flag> ; then CONFIRM ALL THREE and PASTE the actual command output into instrument_selftest: (a) grep -cF "<marker>" the log > 0; (b) ' + PY + ' -c to load a PRODUCED *.json and print one detection dict that CONTAINS your paired field with its value (if absent, the field is not persisting -> FIX the to_dict/serialization path and rerun); (c) ndet BEFORE == ndet AFTER (count-neutral). Do NOT set selftest_passed=true until (a),(b),(c) all literally pass with pasted evidence.',
'5. VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; flags in --help; ScoringConfig() BUE/CBT/recal/CEX defaults unchanged; from_args(Namespace()).scoring==ScoringConfig().',
'6. Return implemented, selftest_passed, instrument_selftest (with pasted evidence), activation_marker, eval_flags (FULL string incl ' + MECH_BASE + ' + your flag; do NOT include --categories, the launcher adds it), paired_marker (the field CONFIRMED present in the JSON), files_changed, validation, notes.',
'ANTI-STALL: the self-test smoke is 2 videos at concurrency 1 (~5 min) — allowed; no OTHER Bash > ~150s.',
  ].join('\n')
}
function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record honestly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '', 'IMPL:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL(88): ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot) + '\nDECISION: ' + decisionText + (fm5 ? '\nFM5/FM17/FM20/FM21 VIOLATION.' : ''),
'TASKS: 1. If REVERT: restore from ' + ROOT + '/.qoder/backups/' + V + '/, diff -q each file, confirm 0 mechanism symbols in anomaly_discovery.py + BUE/CBT/recal defaults intact + pipeline header restored. If KEEP: leave code, flip the new flag default ON if the proposal says so. 2. Archive ' + V + '_hard_eval.json/_run.log/bootstrap_ab.json into ' + LOGS + '/results/. 3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date +%Y%m%d): mechanism, per-cat cand-miss diagnosis, PAIRED bootstrap CI + ABSOLUTE new_map + per-cat deltas + count-neutrality as headline, INSTRUMENT self-test evidence, n_videos (MUST be 88), FM table, WON/OPEN/CLOSED, next-loop guidance. 4. APPEND one ledger row to ' + LOGS + '/session_v97_ledger.md (loop|version|title+verdict|smoke|hard_mAP|AP@0.5|AP@0.7|lat|kept). -1 absent. Do NOT rewrite rows. 5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v15 starting at v' + START_VERSION + '. Committed = BUE+CBT+recal ALL default-ON (full-290 ~0.173). RE-DO v165 motion-peak count-neutral relocation with correct instrument (loop 1) + 1 more count-neutral variant. Harness bug FIXED (--categories forced). PAIRED judging ENFORCED (FM20 VOID if field absent). concurrency ' + CONCURRENCY + '.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' (count-neutral relocation) ===')

  phase('Research')
  const focusA = (loop === 1)
    ? 'YOUR ANGLE (RECLAIM v165): re-derive the v165 CGR-M mechanism DONE RIGHT — per-frame MOTION-ENERGY-peak, COUNT-NEUTRAL relocation of the lowest-confidence off-GT candidate onto the motion peak, gated to the 4 hard cats, pre-BUE. The mechanism was sound (hard point est 0.12113); it died on instrument (paired field never persisted + judged cross-run). Your job: the SAME mechanism with a REAL per-detection paired field (pre_cgr_boundaries) that survives to_dict() into the JSON, a grep-a-JSON self-test, and paired judging. Reuse the pipeline per-frame motion energy (do not recompute).'
    : 'YOUR ANGLE: a DISTINCT count-neutral relocation — e.g. dual-signal (motion x category-curve) target selection, or relocation gated to the specific cand-miss cats where the loop-1 mechanism under-delivered. Count-neutral, paired, real field.'
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', focusA), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: the category-CLIP-curve relocation target (v166 CCR-SE idea) refined + count-neutral + REAL paired field, OR a Shoplifting-specific relocation (cand-miss 11/25, the highest) using whatever signal is elevated at the diffuse in-store GT. Argue the relocation lands on GT.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Synthesize ONE IMPLEMENTABLE COUNT-NEUTRAL relocation mechanism for this loop (FM8). Loop 1 SHOULD be the reclaimed v165 motion-peak relocation done right. MUST carry a real per-detection paired-field write path (FM20) + exact marker (FM21) + count-neutrality proof.',
'', CONTEXT, '', 'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'Kill anything additive (FM2), any of the 10 banned families, CBT anchor, or with expected paired delta < +' + MIN_PAIRED_DELTA + '. StructuredOutput EXACTLY ONCE.',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the quality gate. v14 lost its cand-gen loops to TWO instrument bugs (harness=290 not 88; paired field never persisted -> noisy cross-run verdict). The INSTRUMENT gate and COUNT-NEUTRAL gate are as important as the effect gate. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'CHECKLIST: 1. IMPLEMENTABLE + expected PAIRED hard-cat marginal >= +' + MIN_PAIRED_DELTA + ' with per-cat arithmetic -> effect_size_ok. 2. FM2 COUNT-NEUTRAL: genuinely relocates (total ndet unchanged), not additive -> count_neutral_ok. If additive -> REJECT. 3. INSTRUMENT (FM20/FM21): a REAL per-detection paired-field write path copied from BUE/CBT that survives to_dict() + a grep-a-PRODUCED-JSON self-test (not a claim) + exact marker string -> instrument_ok. If vague -> REJECT. 4. BANNED 1-10 + NOT CBT anchor + NOT prompt max-pool -> banned_family_ok. 5. FM8 one change; BUE/CBT/recal/CEX untouched. REJECT with concrete feedback or APPROVE (optionally mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (r' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 180))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. C REJECTED. Revise or pivot (count-neutral relocation; none banned; not CBT anchor; REAL paired field FM20 + exact marker FM21; count-neutrality proof). ONE change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'REJECTED:\n' + JSON.stringify(proposal, null, 1), '', "C FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) { proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED:\n' + verdict.mandated_changes }); log('Loop ' + loop + ': proceeding with C-mandated revision') }
    else { log('Loop ' + loop + ': C rejected 3x - skipping'); history.push({ loop: loop + 29, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: (verdict.feedback || '').slice(0, 300) }); continue }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker || !impl.paired_marker || impl.selftest_passed === false) {
    log('Loop ' + loop + ': impl failed / self-test not passed (FM20/FM21) - skipping eval to avoid a void loop')
    history.push({ loop: loop + 29, version: V, title: proposal.title + ' [IMPL/SELFTEST FAILED]', full_map: -1, kept: false, lessons: impl ? ('selftest=' + impl.selftest_passed + ' ' + (impl.instrument_selftest || impl.notes || '').slice(0, 200)) : 'null' }); continue
  }

  const pf = impl.paired_marker
  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker, 25, 4, pf, 'Implement')
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count + ' pairedJSONs=' + (smoke.paired_jsons != null ? smoke.paired_jsons : '?') + ' ' + (smoke.note || ''))
  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, paired_jsons: -1, note: 'skipped (smoke)' }
  let boot = null, decisionText = 'REVERT', keep = false, fm5 = false
  if (smoke.done && !smoke.crashed) {
    if (smoke.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from smoke') }
    else if (smoke.paired_jsons != null && smoke.paired_jsons <= 0) { fm5 = true; log('Loop ' + loop + ': FM20 - paired field absent from smoke JSONs (would recreate the v165 failure) - VOID') }
    else {
      full = await runEval(V, 'full', 999, impl.eval_flags, impl.activation_marker, 70, N_HARD, pf, 'Implement')
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' n=' + (full.num_videos != null ? full.num_videos : '?') + ' markers=' + full.marker_count + ' pairedJSONs=' + (full.paired_jsons != null ? full.paired_jsons : '?') + ' ' + (full.note || ''))
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from full') }
      if (full.done && full.paired_jsons != null && full.paired_jsons <= 0) { fm5 = true; log('Loop ' + loop + ': FM20 - paired field absent from full JSONs - VOID (no cross-run fallback)') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, pf), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        if (!boot || !boot.ran) decisionText = 'REVERT (paired adjudication did not run / FM20 VOID: ' + (boot ? boot.note : 'null') + ')'
        else {
          const okDelta = boot.point_delta >= MIN_PAIRED_DELTA, okP = boot.p_gt0 >= MIN_P, okF = boot.new_map >= FLOOR
          if (okDelta && okP && okF) { keep = true; decisionText = 'KEEP (paired marginal +' + boot.point_delta.toFixed(5) + ' >= +' + MIN_PAIRED_DELTA + ', P=' + boot.p_gt0.toFixed(3) + ', abs hard ' + boot.new_map.toFixed(5) + '. per-cat ' + (boot.per_category || '') + '. count_neutral=' + boot.count_neutral + '. NEW hard baseline.)' }
          else if (!okF) decisionText = 'REVERT (floor breached: ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          else if (!okDelta) decisionText = 'REVERT (paired marginal +' + boot.point_delta.toFixed(5) + ' < +' + MIN_PAIRED_DELTA + '; per-cat ' + (boot.per_category || '') + ')'
          else decisionText = 'REVERT (paired P=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ',' + boot.ci90_high + '])'
        }
      }
    }
  } else log('Loop ' + loop + ': smoke failed')
  if (fm5) decisionText = 'REVERT (FM5/FM17/FM20: marker or paired field never appeared)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5/FM20' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [paired+' + boot.point_delta.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ' abs=' + boot.new_map.toFixed(5) + ']') : ''
  history.push({ loop: loop + 29, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && boot && boot.new_map > 0) { bestFullMap = boot.new_map; bestVersion = V; REF_MAP = boot.new_map; FLOOR = Math.max(FLOOR, boot.new_map - MAP_TOL); log('Loop ' + loop + ': KEPT ' + V + ' abs hard ' + boot.new_map + ' (new baseline, floor ' + FLOOR.toFixed(5) + ')') }
  else log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 150))
}

// ============================ FINAL SUMMARY ============================
phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the v15 END-OF-SESSION SUMMARY (v' + START_VERSION + '-v' + (START_VERSION + MAX_LOOPS - 1) + ' count-neutral relocation loops). Plain text; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'TASK: 1. Read ledger tail + new v' + START_VERSION + '+ idea files. 2. date +%Y%m%d; write ' + LOGS + '/summary_v15_' + START_VERSION + '-v' + (START_VERSION + MAX_LOOPS - 1) + '_<YYYYMMDD>.md: whether the reclaimed v165 motion-peak relocation held up under CORRECT paired judging on the fixed 88 harness (per-cat + paired marginal + count-neutrality), any KEEPs (new hard baseline), the updated instrument-discipline checklist (harness=88 confirmed, paired-field-grepped), updated CLOSED-families list, next directions. 3. APPEND ledger rows for KEEPs. 4. Reply with a 12-line recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v15 complete. Best hard-harness ' + bestVersion + ' mAP ' + bestFullMap.toFixed(5) + '. ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, refMap: REF_MAP }
