export const meta = {
  name: 'vad-team-map20-v11',
  description: 'VAD agent team v11 (ACCURACY continuation on the FIXED 65-video BUE-on harness) — FIVE rounds v150->v154 that FIRST audit whether any >= +0.03 optimization space remains, then iterate on whatever is admissible. Committed pipeline = v137 + BUE (default-ON), BUE-on 65-video baseline mAP 0.21932, absolute floor 0.21432. Most families are now CLOSED: curve re-rank (banned 1), VLM merge/batch (2), scan-tail (3), per-video normalization (4), dense-append (5), local-compute trims (6), boundary-growth=BUE (7), scan_segment-abs-conf gates (8, refuted v143/v144), Normal-video-FP suppression (9, oracle +0.00253), and the SUPERVISED-MIL count-neutral RELOCATION family (v143->v149, CLOSED by a noise-free offline proxy: signal ceiling 0.21493 < baseline, 0/17 verdict-independent recovery on the 32-snippet substrate, FM13 temporal resolution). The TWO genuinely-unrefuted leads v11 must investigate: (A) CROSS-VIDEO CONFIDENCE-LEVEL CALIBRATION — the ranking diagnostic measured a +0.05180 cross-video-level oracle (set each video\'s confidence LEVEL right, keep intra-video order); no loop has attacked it; it is NOT per-video normalization (banned 4 destroys level) and NOT curve-derived within-video re-rank (banned 1) — it must SET cross-video levels using a global-comparable SUPERVISED anomaly-likelihood (e.g. model_mil_v6 video-level score; cf. the existing MIL-sanity 30% confidence cut at pipeline.py:1579). (B) FINER-RESOLUTION localizer substrate for recall (denser than 32 snippets) — the only unrefuted recall pivot, but MANDATORY GT-aware-vs-verdict-independent run-IoU oracle pre-check BEFORE any eval (FM13). An HONEST analytical "no admissible >= +0.03 space remains" is a VALID loop outcome (cf. v135/v140) — do not force a bad eval. Concurrency 2 (conc 3 thrashes 16 GB swap).',
  phases: [
    { title: 'Research', detail: 'A audits + attacks the +0.05180 cross-video-level calibration lever; B audits + attacks the finer-resolution recall substrate or a novel family; each proposes one (or an honest no-space closure)' },
    { title: 'Debate', detail: 'A and B cross-critique on reachable effect + measurability, synthesize one joint proposal' },
    { title: 'Review', detail: 'C gates on FM1-FM19 + the >=+0.03 rule + banned families 1-9 + absolute-floor (FM18) + mandatory analytic/oracle pre-check; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (flag + empty-video-safe marker), smoke + in-vivo sentinels -> full 65-video eval -> PAIRED bootstrap vs absolute floor -> keep/revert. Analysis/oracle-only loops spend no eval.' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 5
const START_VERSION = 150
const SMOKE_GATE = 0.06
const MAP_TOL = 0.005
const REF_V = 'v144c'
const REF_TAG = 'bue65ref'
let   REF_DIR = ROOT + '/results/batch_eval_' + REF_V + '_' + REF_TAG
let   REF_MAP = 0.21932
let   FLOOR = 0.21432
const CONCURRENCY = 2
const MIN_DELTA_CROSS = 0.02
const MIN_DELTA_PAIRED = 0.01
const MIN_P = 0.90

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py; config.py; anomaly_expert.py; anomaly_discovery.py; frame_scorer.py; mil_model.py; mil_inference.py; evaluate.py; batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing. INTERPRETER: ' + PY + '.',
'',
'!!! OBJECTIVE: RAISE full-eval temporal mAP on the FIXED 65-video harness (--max_per_category 5). Committed pipeline = v137 + BUE (BUE default-ON, config bue_enabled=True; --no_bue_extension to disable). BUE-on 65-video baseline mAP = ' + REF_MAP + ' (dir ' + REF_DIR + '; AP@0.3 0.32097 / AP@0.5 0.19014 / AP@0.7 0.14685). ABSOLUTE KEEP FLOOR = ' + FLOOR.toFixed(5) + '. !!!',
'',
'!!! THIS SESSION IS AN OPTIMIZATION-SPACE AUDIT + 5 iteration attempts. Be brutally honest: most levers are CLOSED. Your job is to find a GENUINELY unrefuted >= +0.03 lever with a measurement path, or to prove analytically that none remains (a valid outcome, cf. v135/v140/v149 — do NOT force a doomed eval). !!!',
'',
'!!! CLOSED / BANNED FAMILIES (auto C-REJECT — proposing any is a wasted loop) !!!',
'1. Curve-derived within-video confidence RE-RANKING (v111 Q-score already claims the +0.006 within-video oracle; v138-v140 GPC all refuted).',
'2. VLM merge/batch/contact-sheet/verdict-propagation (v97/v126/v129 all cratered mAP to ~0.12).',
'3. max_scan/scan-tail/Source-2 padding pruning (v125/v130/v131; max_scan binds, mean_dropped=0).',
'4. PER-VIDEO score NORMALIZATION (z-score -0.0898, rank -0.0931 — DESTROYS the cross-video level). NOTE: lead (A) below is NOT this — it SETS the level right, it does not normalize it away.',
'5. Dense bottom-ranked proposal append (DPA) — metric gaming.',
'6. Adaptive-fps / local-compute trims (invisible to the metric).',
'7. Boundary outward-growth / UNION / OIC = committed BUE. Do not touch.',
'8. Additive OR relocation gates on RAW scan_segment ABSOLUTE conf (v143/v144: uncalibrated, pins ~0.9).',
'9. Normal-video FALSE-POSITIVE suppression (oracle +0.00253 dmAP — below the floor; low-conf FPs sit below the AP checkpoints).',
'10. SUPERVISED-MIL count-neutral RELOCATION of far dets (v143->v149 FULL ARC): CLOSED by v149 noise-free offline proxy — best gated variant caps ABSOLUTE mAP at 0.21493 < 0.21932; the 32-snippet MIL curve gives 0/17 verdict-independent @IoU>=0.3 recovery (11/17 only WITH GT). This is a SIGNAL CEILING (FM13 temporal resolution), not measurement power. Do NOT re-propose relocation on the 32-snippet substrate.',
'',
'!!! THE TWO GENUINELY-UNREFUTED LEADS (attack these, or prove they are closed) !!!',
'(A) CROSS-VIDEO CONFIDENCE-LEVEL CALIBRATION. The v138 ranking diagnostic (analysis_v138_rankdiag_orch.py) measured a **cross-video-level oracle of +0.05180** (dAP@0.5 +0.05089): if each video\'s confidence LEVEL were set correctly WHILE KEEPING its intra-video order, mAP would rise +0.052 — the LARGEST unattacked headroom in the whole session. Global AP ranks ALL detections across ALL videos together, so a truly-anomalous video whose detections carry too-low confidence gets buried under a weakly-anomalous video\'s over-confident detections. A verdict-independent, PRE-CALIBRATABLE, global-comparable SUPERVISED anomaly-likelihood per video (e.g. model_mil_v6.pth video-level max/mean score, or a calibrated MIL video prior) could RE-LEVEL each video\'s confidences toward the right cross-video order. Precedent already in the tree: the MIL-sanity gate at pipeline.py:1579-1581 already cuts confidence 30% when MIL max<0.3 — this lead GENERALIZES that into a calibrated continuous cross-video re-level. It is NOT banned 4 (which normalizes the level AWAY) and NOT banned 1 (which is curve-derived WITHIN-video). Risks to argue: FM2 (does re-leveling change the greedy-match set?), FM13 (how much of +0.0518 is reachable with a real MIL prior vs the perfect oracle), and whether the MIL video score is actually monotone with true anomaly severity (measure on cached runs first).',
'(B) FINER-RESOLUTION LOCALIZER SUBSTRATE for recall (denser than the 32-snippet MIL curve). The recall ceiling (50/99 zero-overlap dets, median ~20s from GT) is real, but relocation FAILED because the 32-snippet curve cannot identify the true run without GT (0/17). A denser track (per-second CLIP / dense actionness / a finer MIL) MIGHT carry enough positional info. MANDATORY BEFORE ANY CODE/EVAL (FM13): reproduce the GT-aware-vs-verdict-independent run-IoU oracle on the denser track and show the verdict-independent selector beats ~0/17 @IoU>=0.3. If it cannot, this lead is closed too — report that and stop.',
'',
'!!! MEASUREMENT PROTOCOL (noise floor, FM16) — MANDATORY !!!',
'65-video video-level bootstrap 90% CI half-width ~+/-0.025; effects <~+0.03 are unfalsifiable. VERDICTS = video-matched bootstrap CIs; PAIRED preferred (mechanism logs a per-fire pre-mechanism field so --paired-json removes ~+/-0.019 VLM noise; cross-run vs REF conflates with BUE). KEEP requires ALL THREE: point delta >= ' + MIN_DELTA_PAIRED + ' paired (' + MIN_DELTA_CROSS + ' cross) AND P(delta>0) >= ' + MIN_P + ' AND NEW ABSOLUTE mAP >= ' + FLOOR.toFixed(5) + ' (FM18: a positive paired delta over a LOW within-run arm is NOT a win; target ABSOLUTE mAP > ' + REF_MAP + '). Tool: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-json <FIELD>  (or --ref ' + REF_DIR + ' for cross-run).',
'REUSABLE: analysis_v138_rankdiag_orch.py (oracle ceilings incl. the +0.0518 cross-video-level oracle — USE IT to bound lead A before any eval), analysis_v140_livecurve_lib.py (fidelity-gated live category curve), analysis_bootstrap_ab.py (adjudicator, --paired-json). MIL: model_mil_v6.pth + model_mil.pth; mil_scores at pipeline.py:1345; MIL-sanity at :1579. features/test = correct substrate for MIL/actionness prechecks (FM15 caveat is CURVE-specific).',
'',
'KNOWN FAILURE MODES FM1-FM19:',
'FM1 curve-shape edits regress. FM2 additive/count/match-set changes -> FPs under greedy global AP (v143 confirmed). FM3 VLM context-bleed. FM4 self-consistency on inclusion. FM5 never-activated (flag+unique marker+grep gate). FM6 offline/sim gains do not transfer (v143/v148 materialized). FM7 video-level priors cannot fix intra-video mislocation (BUT they CAN fix cross-video LEVEL — lead A). FM8 one change/loop. FM9 MPS .detach().cpu().numpy(). FM10 wall-clock API-drift. FM11 prefer ONE-eval validation; spend ZERO eval on a predetermined outcome (v135/v140/v149). FM12 drop provably-null calls only.',
'FM13 STRUCTURAL-CAP / temporal-resolution: bound the ceiling analytically (oracle) BEFORE any eval (closed the relocation family). FM14 verdict-visibility via global-max ranking. FM15 substrate infidelity for LIVE CURVE signals (use livecurve_lib; features/test ok for MIL). FM16 noise floor (n=65 CI ~+/-0.025; effect must be genuinely large). FM17 EMPTY-VIDEO MARKER DEADLOCK: every per-video marker fires once per SUCCEEDED video incl. the 0-detection path; smoke includes a known-empty video. FM18 PAIRED-DELTA-vs-ABSOLUTE-FLOOR: target ABSOLUTE mAP > ' + REF_MAP + '. FM19 (v148) COUNT-NEUTRAL IS NOT FLOOR-SAFE: moving a matched detection\'s POSITION onto a wrong-but-plausible run demotes it — position moves need a positional-accuracy guarantee, not just count/conf freeze.',
'',
'HARD CONSTRAINTS: ONE controlled change/loop; ARCHITECTURE-LEVEL; expected effect >= +0.03 ABSOLUTE over ' + REF_MAP + ' with explicit AP-checkpoint arithmetic AND an analytic/oracle pre-check (FM13) BEFORE any eval; verdict-independent + pre-calibratable signals only (never scan_segment abs conf); PAIRED-measurable; new params via config.py (default OFF, flag-off byte-identical to committed BUE-on) + CLI in run.py AND batch_evaluate.py + unique INFO marker on the EMPTY-video path (FM17); do NOT disturb BUE (default-ON) or CEX (default-OFF). Eval FIXED: --max_per_category 5, --concurrency ' + CONCURRENCY + ' (NOT 3). StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s; Read offset/limit; short progress line between tool calls.',
].join('\n')

const history = [
  { loop: 5, version: 'v142', title: 'BUE — SHIPPED default-ON', full_map: 0.21932, kept: true, lessons: '290-paired +0.02164 P=0.938; 65-video baseline 0.15632->0.21932. COMMITTED. The one durable win.' },
  { loop: 9, version: 'v146', title: 'MARL-D supervised-MIL relocation (best of the relocation arc)', full_map: 0.21190, kept: false, lessons: 'Fired 73x, +0.03639 paired RIGHT direction FM2-immune, but P=0.876<0.9 & below floor (FM18).' },
  { loop: 12, version: 'v147', title: 'MRR-RUN v1-MIL wide-run relocation + z-gate', full_map: 0.20803, kept: false, lessons: '0 fires (z-gate capped); reported +0.033 was a BUE artifact. REVERT.' },
  { loop: 13, version: 'v148', title: 'TSAR tri-signal (v1+v6 MIL + CLIP-delta) relocation', full_map: 0.12588, kept: false, lessons: 'Fired 37x but DESTRUCTIVE -0.0855; position moves onto wrong-but-agreed runs demote matched dets. FM19 born. REVERT.' },
  { loop: 14, version: 'v149', title: 'MVR-HG head-freeze relocation — OFFLINE proxy, no eval', full_map: -1, kept: false, lessons: 'Noise-free proxy caps ABS mAP at 0.21493 < baseline. CLOSES the whole MIL-relocation family on the 32-snippet substrate: SIGNAL CEILING (FM13), not measurement power. 0/17 verdict-independent recovery.' },
  { loop: 15, version: 'DIAG', title: 'Two unattacked leads for v11', full_map: -1, kept: false, lessons: 'LEAD A: cross-video-level oracle +0.05180 (largest unattacked headroom) via supervised MIL re-leveling — NOT banned 4/1. LEAD B: finer-resolution localizer substrate (oracle-gated). Normal-FP CLOSED (+0.00253).' },
]
let bestFullMap = REF_MAP
let bestVersion = 'v137+BUE'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": full mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED baseline: ' + bestVersion + ' (BUE-on 65-video reference mAP ' + REF_MAP + ', floor ' + FLOOR.toFixed(5) + ').\nSession history:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'What it does mechanically, 3-6 sentences' },
    target_family: { type: 'string', description: 'cross-video-level-calibration | finer-resolution-recall | novel-<name> | ANALYTICAL-CLOSURE (no admissible space)' },
    expected_map_gain: { type: 'number', description: 'Honest expected ABSOLUTE mAP gain over ' + REF_MAP + '; must be >= 0.03 (or 0 if this is an analytical closure)' },
    precheck_plan: { type: 'string', description: 'The analytic/ORACLE pre-check (FM13) that will run BEFORE any eval — e.g. analysis_v138_rankdiag_orch.py for the +0.0518 oracle, or a GT-aware-vs-verdict-independent run-IoU oracle on the denser substrate — and the threshold that must pass to justify an eval spend' },
    effect_size_arithmetic: { type: 'string', description: 'Which AP checkpoints move and why ABSOLUTE mAP exceeds ' + REF_MAP + ' with a paired CI that can clear zero (P>=0.9)' },
    localizer_calibration: { type: 'string', description: 'The verdict-independent pre-calibratable signal + its calibration. NOT scan_segment abs conf.' },
    count_neutral: { type: 'boolean' },
    inv_ivo_sentinels: { type: 'string', description: 'Pre-registered smoke sentinels that WILL run before the full eval' },
    implementation_plan: { type: 'string', description: 'Exact files/functions/lines, config field, CLI flags, per-fire pre-mechanism field, marker incl EMPTY-video path (FM17). If ANALYTICAL-CLOSURE, the analysis that proves no space remains.' },
    paired_measurable: { type: 'boolean' },
    fm_risk_table: { type: 'string', description: 'FM1-FM19 each: immune / at-risk + why' },
    banned_family_check: { type: 'string', description: 'Explicit: none of the 10 closed/banned families, and why (esp. vs banned 4 for lead A, vs banned 10 for lead B)' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string' },
  },
  required: ['title', 'mechanism', 'target_family', 'expected_map_gain', 'precheck_plan', 'effect_size_arithmetic', 'localizer_calibration', 'count_neutral', 'inv_ivo_sentinels', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'banned_family_check', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}
const verdictSchema = { type: 'object', properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' }, effect_size_ok: { type: 'boolean' }, banned_family_ok: { type: 'boolean' }, precheck_ok: { type: 'boolean', description: 'Is there a real analytic/oracle pre-check (FM13) that gates the eval spend?' }, absolute_floor_argument_ok: { type: 'boolean' } }, required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'precheck_ok', 'absolute_floor_argument_ok'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, is_analysis_only: { type: 'boolean', description: 'true if this loop is an oracle/analysis closure that spends NO eval' }, precheck_result: { type: 'string', description: 'The actual oracle/analytic pre-check output (numbers). If it fails the gate, say so — no eval will be spent.' }, precheck_passed: { type: 'boolean' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string' }, paired_marker: { type: 'string' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'is_analysis_only', 'precheck_passed', 'activation_marker', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, latency: { type: 'number' }, marker_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
const resultSchema = { type: 'object', properties: { version: { type: 'string' }, implemented: { type: 'boolean' }, smoke_map: { type: 'number' }, full_map: { type: 'number' }, kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' }, report_path: { type: 'string' } }, required: ['version', 'implemented', 'full_map', 'kept', 'summary', 'lessons'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent: attempt ' + (t + 1) + '/' + n + ' returned null') }
    catch (e) { const msg = (e && e.message) ? e.message : String(e); log('retryAgent: attempt ' + (t + 1) + '/' + n + ' threw: ' + msg.slice(0, 200)) }
  }
  return null
}
function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
  return [
'You are Agent D-launcher. ONLY job: ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT; every Bash < 150s. StructuredOutput EXACTLY ONCE.',
'Paths: output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (verbatim): ' + (evalFlags || '(none)'),
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. Already running for this V_tag -> adopt if flags present else kill&relaunch. 3. report exists + flags in log -> launched=true note="exists". 4. DIFFERENT version running -> sleep 140 re-check up to 3x, else launched=false note="blocked".',
'5. Else record <output_dir>/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + ROOT + "/results/batch_eval_" + V + "_" + tag + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + (evalFlags || '') + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log for a Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, marker, nTotal) {
  const rep = ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json'
  const lg = ROOT + '/results/batch_eval_' + V + '_' + tag + '.log'
  const N = nTotal || 65
  return [
'You are Agent D-watcher for ' + tag + ' eval of ' + V + ' (' + N + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE. report=' + rep + ' log=' + lg + ' inner=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/run.log marker="' + (marker || '(none)') + '".',
'1. report exists -> extract & done=true: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti[\'avg_per_video_seconds\'])" ; markers: grep -cF "' + (marker || '') + '" ' + lg + ' (also inner run.log; trust LARGER).',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 re-check up to 2x, then done=false note="running: <n>/' + N + '". STALL >30min no growth -> kill & relaunch SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead no report -> tail -30. TRANSIENT -> relaunch. CODE BUG -> crashed=true. FM17 (markers<succeeded) -> crashed=true note="FM17 deadlock".',
'Return done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
  ].join('\n')
}
async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: 'Implement' }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  const cap = maxWatch || 80
  for (let w = 0; w < cap; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, marker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: 'Implement' }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap reached' }
}
function bootstrapPrompt(V, pairedMarker) {
  const dir = ROOT + '/results/batch_eval_' + V + '_full'
  const paired = pairedMarker && pairedMarker.length > 0
  return [
'You are Agent D-adjudicator for ' + V + '. Run the MANDATED video-matched bootstrap CI. Bash < 150s. StructuredOutput EXACTLY ONCE.',
paired
  ? ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + pairedMarker + ' --min-delta ' + MIN_DELTA_PAIRED + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5) + '  (if that field absent, try --paired-log ' + dir + '/run.log --marker "' + pairedMarker + '")')
  : ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta ' + MIN_DELTA_CROSS + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5)),
'If >150s re-invoke with --resamples 300. FM18: new_map is ABSOLUTE mAP; KEEP needs new_map >= ' + FLOOR.toFixed(5) + '. Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, note (VERDICT line verbatim).',
  ].join('\n')
}
function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. GOAL: find a GENUINELY-unrefuted mechanism with HONEST expected effect >= +0.03 ABSOLUTE mAP over the committed BUE-on baseline ' + REF_MAP + ' on the FIXED 65-video harness — or prove analytically that none remains in your area (a valid outcome). ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. AUDIT FIRST (Read offset/limit): ' + LOGS + '/summary_v10_v147-v149_20260801.md (the relocation-family closure + FM13/FM18/FM19), ' + LOGS + '/ideas/loop1_v138_gpc-global-peak-containment_20260727.md (the ORACLE table incl. the +0.05180 cross-video-level oracle and +0.006 within-video oracle), ' + LOGS + '/summary_v9_v145-v146_20260801.md. Enumerate honestly which families are CLOSED and what remains.',
'2. RUN THE ORACLE PRE-CHECK for your lead (FM13, no API cost): for lead A use ' + LOGS + '/analysis_v138_rankdiag_orch.py to reconfirm the +0.0518 cross-video-level oracle on the CURRENT BUE-on cached runs and estimate how much a REAL model_mil_v6 video-level prior recovers (is the MIL video score monotone with true anomaly severity? measure on cached preds + features/test). For lead B, build/run the GT-aware-vs-verdict-independent run-IoU oracle on a DENSER substrate and show it beats ~0/17 @IoU>=0.3. DO NOT propose a mechanism whose oracle pre-check does not clear a path to >= +0.03.',
'3. SEARCH: >=4 real WebSearch queries relevant to your lead (cross-video score calibration / global anomaly ranking / dense actionness localization 2024-2026). If unavailable, say so and rely on code+data.',
'4. Derive ONE architecture-level mechanism (or an ANALYTICAL-CLOSURE if the oracle says no): verdict-independent + pre-calibratable, PAIRED-measurable, count-neutral where possible, empty-video-safe marker (FM17), with the precheck_plan, effect-size arithmetic, ABSOLUTE-floor argument (FM18), config flag + CLI + marker, and FM1-FM19 table. It must NOT be any of the 10 closed/banned families.',
'5. Return via StructuredOutput (EXACTLY ONCE).',
'!!! ANTI-STALL: no Bash > ~60s, no sleep, Read offset/limit, short progress line, finish in <18 tool calls.',
  ].join('\n')
}
function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED:\n' + verdict.mandated_changes : '',
'STEPS:',
'0. RUN THE ORACLE PRE-CHECK FIRST (from precheck_plan). Report precheck_result + precheck_passed. If it FAILS the pre-registered gate, set is_analysis_only=true, implement NOTHING, and return (no eval will be spent — FM11/FM13). This is a valid, valuable outcome.',
'1. If the pre-check passes: mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy every file you will edit BEFORE editing.',
'2. Implement. config.py default OFF/prior + from_args threading; CLI in BOTH run.py and batch_evaluate.py; bump pipeline.py header to ' + V + '. Do NOT disturb BUE (default-ON) or CEX (default-OFF). Read-only use of any MIL model; do not change the fused curve (FM1).',
'3. FM5+FM17: unique INFO marker per fire AND a per-video summary marker on the EMPTY/0-detection path. Log the per-fire PRE-MECHANISM value (e.g. pre_' + V.toLowerCase() + ' field) index-aligned for --paired-json.',
'4. VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; flags in --help; ScoringConfig() BUE/CEX defaults unchanged; from_args(Namespace()).scoring==ScoringConfig(); empty-video marker check (FM17). Report outputs.',
'5. Return implemented, is_analysis_only, precheck_result, precheck_passed, activation_marker, eval_flags, paired_marker, files_changed, validation, notes.',
'ANTI-STALL: no Bash > ~150s; Read offset/limit; do NOT run a full eval here.',
  ].join('\n')
}
function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record honestly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '', 'IMPLEMENTATION:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL: ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot) + '\nDECISION: ' + decisionText + (fm5 ? '\nFM5/FM17 VIOLATION.' : ''),
'TASKS:',
'1. If REVERT and code was changed: restore from ' + ROOT + '/.qoder/backups/' + V + '/, diff -q, confirm 0 mechanism symbols (grep) + BUE default-ON/CEX default-OFF. If analysis-only: nothing to restore. If KEEP: leave code, flip default ON if proposal says so.',
'2. Archive into ' + LOGS + '/results/ (whichever exist): ' + V + '_full_eval.json, ' + V + '_full_run.log, ' + V + '_smoke_eval.json, bootstrap_ab.json, ' + V + '_precheck.txt.',
'3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date +%Y%m%d): mechanism, the ORACLE PRE-CHECK result, effect-size vs actual, BOOTSTRAP CI + ABSOLUTE mAP vs floor ' + FLOOR.toFixed(5) + ' as headline, FM1-FM19 table, whether this lead is now WON/OPEN/CLOSED, what the next loop should NOT repeat.',
'4. APPEND one row to ' + LOGS + '/session_v97_ledger.md (loop|version|title+verdict+evidence|smoke_mAP|full_mAP|AP@0.5|AP@0.7|latency_s|kept). -1 for absent. Do NOT rewrite rows.',
'5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v11 starting at v' + START_VERSION + '. Committed = v137+BUE (baseline ' + REF_MAP + ', floor ' + FLOOR.toFixed(5) + '). ' + MAX_LOOPS + ' loops auditing + attacking the two unrefuted leads (A cross-video-level calibration +0.0518 oracle; B finer-resolution recall). Analytical closure is a valid outcome. concurrency ' + CONCURRENCY + '.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' ===')

  phase('Research')
  const focusA = (loop === 1)
    ? 'YOUR ANGLE (loop 1, the biggest unattacked headroom): LEAD A — CROSS-VIDEO CONFIDENCE-LEVEL CALIBRATION toward the +0.05180 oracle, using a global-comparable SUPERVISED anomaly-likelihood (model_mil_v6 video-level score / calibrated MIL prior) to RE-LEVEL each video\'s detection confidences while keeping intra-video order. Prove with analysis_v138_rankdiag_orch.py how much of +0.0518 a REAL MIL prior recovers before proposing. NOT banned 4 (no normalization) or 1 (not within-video curve).'
    : 'YOUR ANGLE: whichever of the two leads (A cross-video-level calibration, B finer-resolution recall) the previous loop showed the most reachable oracle headroom for — push it, or if both are oracle-capped below +0.03, deliver the ANALYTICAL-CLOSURE that proves no >= +0.03 space remains.'
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', focusA), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: LEAD B — a FINER-RESOLUTION localizer substrate (denser than the 32-snippet MIL curve: per-second CLIP / dense actionness / finer MIL) to break the 0/17 verdict-independent recall wall. MANDATORY: run the GT-aware-vs-verdict-independent run-IoU oracle on the denser track FIRST and show it beats ~0/17 @IoU>=0.3 — if it cannot, deliver the ANALYTICAL-CLOSURE that this lead is also closed. You may also propose a genuinely NOVEL family if you can defend >= +0.03 physical headroom + a measurement path.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Cross-critique and synthesize ONE proposal (or agree on an ANALYTICAL-CLOSURE if both leads are oracle-capped below +0.03).',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'RULES: pick the ONE with the larger ORACLE-VERIFIED reachable ABSOLUTE gain over ' + REF_MAP + ' (FM8: one change). Kill anything whose oracle pre-check caps below +0.03 (FM13/FM16). REJECT any of the 10 closed/banned families. Require: count-neutral where possible, paired-measurable, empty-video-safe (FM17), ABSOLUTE-floor argument (FM18), and a concrete oracle pre-check. If BOTH leads are oracle-capped below +0.03, return an ANALYTICAL-CLOSURE proposal (expected_map_gain 0, target_family ANALYTICAL-CLOSURE) stating so with numbers. StructuredOutput EXACTLY ONCE.',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the strict quality gate. Each full eval ~1-2h. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'CHECKLIST:',
'1. If target_family==ANALYTICAL-CLOSURE: APPROVE it as an analysis-only loop IF the numbers genuinely show no >= +0.03 space (valuable closure, no eval). Otherwise judge as a mechanism:',
'2. PRE-CHECK (FM13, binding): is there a REAL analytic/oracle pre-check that gates the eval and shows a path to >= +0.03 ABSOLUTE? No credible pre-check -> REJECT. Set precheck_ok.',
'3. EFFECT SIZE: expected ABSOLUTE gain over ' + REF_MAP + ' >= +0.03 with honest arithmetic? If ~+0.01, REJECT (FM16).',
'4. ABSOLUTE FLOOR (FM18): argues NEW ABSOLUTE mAP > ' + REF_MAP + ', not a within-run paired delta over a low arm? Set absolute_floor_argument_ok.',
'5. BANNED FAMILIES 1-10 (esp. 4 per-video normalization for lead A; 10 relocation-on-32-snippet for lead B; 1 curve within-video re-rank)? If yes -> REJECT.',
'6. FM2/FM19: count/match-set change or position move that could demote matched dets? Demand the guarantee.',
'7. FM5+FM17: flag (default OFF, flag-off byte-identical) + CLI both runners + empty-video marker + per-fire pre-mechanism field for --paired-json?',
'8. FM8/FM11: ONE change, one eval (or zero for analysis-only), BUE default-ON + CEX default-OFF untouched?',
'Set effect_size_ok, banned_family_ok, precheck_ok, absolute_floor_argument_ok. REJECT with concrete feedback or APPROVE (optionally mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (r' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. C REJECTED your proposal. Revise or pivot within the two unrefuted leads (or deliver an ANALYTICAL-CLOSURE); clear >= +0.03 ABSOLUTE over ' + REF_MAP + ' with a real oracle pre-check, avoid all 10 banned families. ONE change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'REJECTED:\n' + JSON.stringify(proposal, null, 1), '', "C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) { proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED:\n' + verdict.mandated_changes }); log('Loop ' + loop + ': proceeding with C-mandated revision') }
    else { log('Loop ' + loop + ': C rejected 3x - no eval (FM11)'); history.push({ loop: loop + 15, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) }); continue }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl) { log('Loop ' + loop + ': D1 null'); history.push({ loop: loop + 15, version: V, title: proposal.title + ' [IMPL NULL]', full_map: -1, kept: false, lessons: 'agent null' }); continue }

  // analysis-only loop or failed pre-check: no eval, record closure
  if (impl.is_analysis_only || !impl.precheck_passed) {
    const dec = impl.is_analysis_only ? 'ANALYSIS-ONLY closure (no eval spent, FM11/FM13)' : 'REVERT: oracle pre-check FAILED (' + (impl.precheck_result || '').slice(0, 200) + ') — no eval spent (FM13)'
    log('Loop ' + loop + ': ' + dec)
    const finA = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, { note: 'no smoke (analysis-only / precheck-fail)' }, { note: 'no full eval' }, null, dec, false), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
    history.push({ loop: loop + 15, version: V, title: proposal.title + ' [' + (impl.is_analysis_only ? 'ANALYTICAL-CLOSURE' : 'PRECHECK-FAIL') + ']', full_map: -1, kept: false, lessons: ((finA && finA.lessons) || impl.precheck_result || '').slice(0, 400) })
    continue
  }
  if (!impl.implemented || !impl.activation_marker) {
    log('Loop ' + loop + ': implementation failed / no marker (FM5)')
    history.push({ loop: loop + 15, version: V, title: proposal.title + ' [IMPL FAILED]', full_map: -1, kept: false, lessons: (impl.notes || '').slice(0, 200) }); continue
  }

  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker, 30, 13)
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count)
  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (smoke gate)' }
  let boot = null, decisionText = 'REVERT', keep = false, fm5 = false
  if (smoke.done && !smoke.crashed && smoke.mAP >= SMOKE_GATE) {
    if (smoke.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from smoke') }
    else {
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker, 80, 65)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' markers=' + full.marker_count)
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from full') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, impl.paired_marker || ''), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        const paired = !!(impl.paired_marker && impl.paired_marker.length > 0)
        const bar = paired ? MIN_DELTA_PAIRED : MIN_DELTA_CROSS
        if (!boot || !boot.ran) decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'agent null') + ')'
        else {
          const okD = boot.point_delta >= bar, okP = boot.p_gt0 >= MIN_P, okF = boot.new_map >= FLOOR
          if (okD && okP && okF) { keep = true; decisionText = 'KEEP (' + (paired ? 'paired' : 'cross') + ' delta ' + boot.point_delta.toFixed(5) + ' >= ' + bar + ', P=' + boot.p_gt0.toFixed(3) + ' >= ' + MIN_P + ', ABS mAP ' + boot.new_map.toFixed(5) + ' >= floor ' + FLOOR.toFixed(5) + '. CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + ']. NEW BASELINE.)' }
          else if (!okF) decisionText = 'REVERT (FM18 ABSOLUTE FLOOR breached: ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          else if (!okD) decisionText = 'REVERT (effect too small: ' + boot.point_delta.toFixed(5) + ' < ' + bar + '; P=' + boot.p_gt0.toFixed(3) + '. FM16.)'
          else decisionText = 'REVERT (P(d>0)=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + '] straddles zero.)'
        }
      }
    }
  } else log('Loop ' + loop + ': smoke gate failed')
  if (fm5) decisionText = 'REVERT (FM5/FM17: marker never appeared / deadlocked)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5/FM17' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [d=' + boot.point_delta.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ' mAP=' + boot.new_map.toFixed(5) + ']') : ''
  history.push({ loop: loop + 15, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) { bestFullMap = result.full_map; bestVersion = V; REF_DIR = ROOT + '/results/batch_eval_' + V + '_full'; REF_MAP = result.full_map; FLOOR = Math.max(0.21432, result.full_map - MAP_TOL); log('Loop ' + loop + ': KEPT ' + V + ' - new baseline ' + result.full_map + ' (floor now ' + FLOOR.toFixed(5) + ')') }
  else log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 160))
}

// ============================ END-OF-SESSION SUMMARY ============================
phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for the v11 optimization-space audit (v150-v154). Plain text; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'TASK: (1) Read ' + LOGS + '/session_v97_ledger.md (tail) + the new v150-v154 idea files. (2) date +%Y%m%d. (3) Write ' + LOGS + '/summary_v11_v150-v154_<YYYYMMDD>.md: executive summary (did ANY loop clear >= +0.03 ABSOLUTE with P>=' + MIN_P + '? is there ANY admissible optimization space left, per the oracle pre-checks?), a per-lead verdict (A cross-video-level calibration: reachable fraction of +0.0518 with a real MIL prior, WON/OPEN/CLOSED; B finer-resolution recall: verdict-independent run-IoU oracle result, WON/OPEN/CLOSED), a per-version table, the updated FM1-FM19 taxonomy, and a clear FINAL statement on whether the pipeline has reached a practical ceiling on this harness or what the single most promising remaining direction is. (4) APPEND a ledger row only if a loop was KEPT. (5) Reply with a 12-line recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v11 complete. Committed baseline ' + bestVersion + ' mAP ' + bestFullMap.toFixed(5) + ' (BUE-on). ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, refMap: REF_MAP, floor: FLOOR }
