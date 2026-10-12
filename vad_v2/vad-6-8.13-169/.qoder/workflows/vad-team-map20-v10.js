export const meta = {
  name: 'vad-team-map20-v10',
  description: 'VAD agent team v10 (ACCURACY continuation on the FIXED 65-video BUE-on harness). THREE rounds v147->v149 pushing the ONE unrefuted >= +0.03 recall lever the session has found: SUPERVISED-MIL count-neutral relocation of far-mislocated detections onto the right event. Direct continuation of v146 MARL-D, which was the first mechanism to fire at scale (73 relocations) and move mAP the RIGHT way (+0.03639 paired, FM2-immune) but died on measurement power (P=0.876<0.9) + the absolute floor (NEW 0.21190 < floor 0.21432). v10 strengthens the SIGNAL, not the gate: (loop1/v147) swap the relocation target from v1 model_mil.pth to the stronger standalone model_mil_v6.pth and replace the blind NORMAL-train-P90 decisiveness gate with an IN-VIVO-calibrated one, to convert MORE of the 50/99 zero-overlap detections so the effect is large enough to move the paired CI off zero AND lift ABSOLUTE mAP above the committed BUE-on baseline 0.21932; (loops2-3) iterate on whatever v147 shows. Committed pipeline = v137 + BUE (default-ON, DONE). Concurrency 2 (conc 3 thrashes the 16 GB machine). NORMAL-VIDEO FALSE-POSITIVE SUPPRESSION IS CLOSED for the mAP objective: measured oracle ceiling from perfectly deleting every Normal-video FP is only +0.00253 dmAP (analysis_v9_normal_fp.py) — below the noise floor; do NOT propose it.',
  phases: [
    { title: 'Research', detail: 'A strengthens the MARL-D SIGNAL (model_mil_v6.pth target + in-vivo-calibrated gate, convert more zero-overlap dets); B explores an alternative verdict-independent supervised localizer / calibration; each proposes one' },
    { title: 'Debate', detail: 'A and B cross-critique on effect size + measurability, synthesize one joint proposal' },
    { title: 'Review', detail: 'C gates on FM1-FM18 + the >=+0.03 rule + banned families 1-9 + absolute-floor (FM18) + calibration; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (flag + empty-video-safe marker), smoke + in-vivo sentinels -> full 65-video eval -> PAIRED bootstrap -> keep/revert against absolute floor' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 3
const START_VERSION = 147
const SMOKE_GATE = 0.06
const MAP_TOL = 0.005
// Reuse the v9 Phase-0 fresh BUE-ON 65-video reference (committed pipeline). No baseline re-run.
const REF_V = 'v144c'
const REF_TAG = 'bue65ref'
let   REF_DIR = ROOT + '/results/batch_eval_' + REF_V + '_' + REF_TAG
let   REF_MAP = 0.21932                   // measured BUE-on committed 65-video baseline (v9 Phase 0)
let   FLOOR = 0.21432                      // REF_MAP - 0.005 (absolute KEEP floor, FM18)
const CONCURRENCY = 2

const MIN_DELTA_CROSS = 0.02
const MIN_DELTA_PAIRED = 0.01
const MIN_P = 0.90

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline; config.py = params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, mil_model.py, mil_inference.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Edit files directly; back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'INTERPRETER: use ' + PY + ' for everything.',
'',
'!!! OBJECTIVE: RAISE full-eval temporal mAP on the FIXED 65-video harness (--max_per_category 5). Do NOT enlarge it. Committed pipeline = v137 + BUE (BUE default-ON, config bue_enabled=True; --no_bue_extension to disable). Fresh BUE-ON 65-video baseline mAP = ' + REF_MAP + ' (dir ' + REF_DIR + '; AP@0.3 0.32097 / AP@0.5 0.19014 / AP@0.7 0.14685). ABSOLUTE KEEP FLOOR = ' + FLOOR.toFixed(5) + '. !!!',
'',
'!!! THIS SESSION\'S MANDATE (owner-directed): push the SUPERVISED-MIL count-neutral relocation lever — the ONE recall mechanism that fires and moves mAP the right way. Do NOT chase Normal-video false-positive suppression: it is CLOSED for the mAP objective (measured oracle ceiling from perfectly deleting EVERY Normal-video FP = only +0.00253 dmAP on the 290 run, analysis_v9_normal_fp.py; the Normal FPs are low-confidence (median 0.517) and sit below the ~4 AP hit-checkpoints that decide mAP). See banned family 9. !!!',
'',
'!!! THE RECALL LEVER — WHERE WE ARE (v143->v146) !!!',
'The binding, STILL-OPEN headroom is the recall/localization ceiling: max recall 0.379@0.3, 0.212@0.5, 0.121@0.7; **50 of 99 detections have ZERO GT overlap** (median IoU 0.000, median GT distance ~20s). BUE grows EXISTING windows and cannot reach these. Four attacks bracket the family:',
'- v143 CGR (ADDITIVE VLM rescue on raw scan_segment ABS conf): floor-breach -0.018; additive dets = high-ranked FPs demoting precision (FM2). Uncalibrated abs conf pins ~0.9 on isolated clips.',
'- v144 PACR (count-neutral, VLM-comparative gate): CRASHED (FM17 empty-video marker deadlock), never evaluated.',
'- v145 CSR (count-neutral, LIVE CLIP-curve-peak geometry gate): NULL — gate too selective, fired on only 3 dets, bit-identical arms; leverage-free. CLOSED.',
'- v146 MARL-D (count-neutral, SUPERVISED v1 model_mil.pth argmax-run, NORMAL-train-P90 decisiveness gate): fired AT SCALE (73 relocations), point_delta **+0.03639** (largest of the arc, RIGHT direction, FM2 immunity CONFIRMED) — but CI90 [-0.01774,+0.09582] crosses zero, **P(Δ>0)=0.876 < 0.9**, and **NEW mAP 0.21190 < floor 0.21432** (short 0.00242). Two of three gates failed. The loss was MEASUREMENT POWER + the ABSOLUTE FLOOR, NOT the signal.',
'KEY CONCLUSION (v9 summary): a verdict-independent SUPERVISED localizer DOES fire and move mAP the right way. To WIN, make the effect LARGER and more CONSISTENT — convert MORE of the 50 zero-overlap dets — so the paired CI clears zero (P>=0.9) AND absolute mAP exceeds ' + REF_MAP + ' (FM18). Strengthen the SIGNAL, not the gate.',
'',
'!!! THE MIL ASSETS (verdict-independent, pre-calibratable — exactly what this lever needs) !!!',
'- model_mil_v6.pth (5 MB, STRONGER standalone MIL than v1) and model_mil.pth (v1, in production). A prior FUSION of v6 into the score FAILED at FM1 (curve-shape), but a NON-fusion use as a RELOCATION TARGET (read-only argmax/run, windows moved not scores changed) is a different use and is allowed.',
'- mil_scores are computed live at pipeline.py:1345-1346 (self.mil_model(_snippet_tensor)) into video_data[\'mil_scores\']; v1 model loaded at pipeline.py:982; MIL-sanity gate at :1579; MIL curve interpolation at :2656. v146 read video_data[\'mil_scores\'] (v1). v147 should load/read model_mil_v6.pth scores as the relocation target (add a second read-only inference or a v6 curve), keeping the pipeline\'s scoring/confidence untouched.',
'- features/train (1610 videos) + features/test (290) pre-extracted CLIP snippets — the CORRECT substrate for MIL/actionness prechecks (features/test IS what the MIL model consumes; the FM15 infidelity caveat is specific to the LIVE CLIP category CURVE, not MIL scores).',
'',
'!!! THE NOISE FLOOR (v140) + MEASUREMENT PROTOCOL — MANDATORY !!!',
'65-video video-level bootstrap 90% CI half-width ~+/-0.025; effects <~+0.03 are unfalsifiable. VERDICTS = video-matched bootstrap CIs; PAIRED mode is REQUIRED for this lever (the mechanism logs a per-fire pre-mechanism window field so the paired bootstrap removes ~+/-0.019 VLM noise; cross-run would conflate with BUE). Tool:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <full_dir> --paired-json <FIELD>   [preferred]',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <full_dir> --ref ' + REF_DIR + '        [cross-run fallback]',
'KEEP requires ALL THREE: point delta >= ' + MIN_DELTA_PAIRED + ' (paired) / ' + MIN_DELTA_CROSS + ' (cross) AND P(delta>0) >= ' + MIN_P + ' AND NEW ABSOLUTE mAP >= ' + FLOOR.toFixed(5) + ' (FM18: a positive paired delta over a LOW within-run arm does NOT clear the absolute floor — v146 died here).',
'',
'!!! BANNED FAMILIES (auto C-REJECT) !!!',
'1. Curve-derived confidence RE-RANKING. 2. VLM merge/batch/contact-sheet/propagation. 3. max_scan/scan-tail/Source-2 pruning. 4. Per-video score normalization. 5. Dense bottom-ranked proposal append (DPA). 6. Adaptive-fps/local-compute trims. 7. Boundary outward-growth/UNION/OIC (= committed BUE). 8. Additive OR relocation gates on RAW scan_segment ABSOLUTE conf (refuted v143/v144: uncalibrated, pins ~0.9). 9. (v10) NORMAL-VIDEO FALSE-POSITIVE SUPPRESSION for the mAP objective (oracle ceiling +0.00253, below the floor).',
'',
'KNOWN FAILURE MODES (FM1-FM18):',
'FM1 curve-shape edits regress (10+). FM2 additive/count changes -> FPs under greedy global AP (v143 CONFIRMED; count-neutrality is the fix). FM3 VLM context-bleed. FM4 self-consistency on inclusion. FM5 never-activated (require flag + unique INFO marker + grep gate). FM6 offline/sim gains do not transfer (v143 TP-only oracle over-predicted). FM7 video-level priors cannot fix intra-video mislocation. FM8 one controlled change per loop. FM9 MPS tensors need .detach().cpu().numpy(). FM10 wall-clock is API-drift-dominated. FM11 prefer ONE-full-eval validation. FM12 drop provably-null calls only.',
'FM13 structural-cap/magnitude — bound the ceiling analytically first (v145 fire-rate over-predicted ~6x). FM14 verdict-visibility via global-max ranking. FM15 SUBSTRATE INFIDELITY for LIVE CURVE signals (use analysis_v140_livecurve_lib.py; features/test is fine for MIL). FM16 MEASUREMENT BELOW THE NOISE FLOOR (n=65 CI ~+/-0.025; even +0.036 paired gave P=0.876 — the effect must be genuinely large).',
'FM17 EMPTY-VIDEO MARKER DEADLOCK (v144 killer): every per-video marker MUST fire once per SUCCEEDED video INCLUDING the 0-detection / mil_scores==None / flat-curve paths; smoke must include a known-empty video. Reference template: v145/v146 markers.',
'FM18 PAIRED-DELTA-vs-ABSOLUTE-FLOOR MISMATCH (v146 killer): a positive within-run paired delta over a LOW within-run REF arm does NOT clear the ABSOLUTE floor. Target ABSOLUTE mAP > ' + REF_MAP + ', not a within-run low arm.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop. ARCHITECTURE-LEVEL. Expected effect >= +0.03 with explicit AP-checkpoint arithmetic AND an argument for why it lifts ABSOLUTE mAP over ' + REF_MAP + ' and can clear P>=0.9 on n=65 (i.e. genuinely large/consistent).',
'- The localizer MUST be VERDICT-INDEPENDENT and PRE-CALIBRATABLE (MIL/actionness/curve), NEVER raw scan_segment abs conf. COUNT-NEUTRAL required (FM2). PAIRED-MEASURABLE required (log a per-fire pre-mechanism window field).',
'- Pre-register in-vivo SENTINELS (relocation rate, on-GT>=off-GT displacement, no degenerate pin) and RUN THEM IN SMOKE before any full eval.',
'- New params via config.py (default OFF, flag-off byte-identical to the committed BUE-on pipeline) + CLI in BOTH run.py and batch_evaluate.py + unique INFO marker that fires on the EMPTY-video path (FM17). Do NOT disturb BUE (default-ON) or CEX (default-OFF).',
'- Eval harness FIXED: --max_per_category 5 (65 videos), --concurrency ' + CONCURRENCY + ' (NOT 3 — thrashes 16 GB swap; stall >30 min -> kill & relaunch at ' + CONCURRENCY + ', runner skips existing JSONs).',
'- StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s; Read with offset/limit; short progress line between tool calls (runtime kills agents silent >180s).',
].join('\n')

const history = [
  { loop: 5, version: 'v142', title: 'BUE Boundary-Union Extension — SHIPPED default-ON', full_map: 0.21932, kept: true, lessons: '290-paired +0.02164 P=0.938; 65-video BUE-on baseline 0.21932 (vs BUE-off 0.15632). COMMITTED. Grows existing windows; cannot reach the 50/99 zero-overlap misses.' },
  { loop: 6, version: 'v143', title: 'CGR additive VLM recall rescue', full_map: 0.13937, kept: false, lessons: 'REVERT floor breach Δ-0.018; additive dets = FPs demoting precision (FM2); uncalibrated scan_segment abs conf. DO NOT repeat (banned 8).' },
  { loop: 7, version: 'v144', title: 'PACR count-neutral peak-anchored relocation', full_map: -1, kept: false, lessons: 'REVERT no verdict; crashed at FM5/FM17 empty-video marker deadlock; never evaluated.' },
  { loop: 8, version: 'v145', title: 'CSR count-neutral CLIP-curve-peak relocation', full_map: 0.20599, kept: false, lessons: 'REVERT null: gate too selective (3 dets fired), Δ+0.000 bit-identical; below floor. CLOSED — CLIP-curve-peak geometry gate has no leverage.' },
  { loop: 9, version: 'v146', title: 'MARL-D SUPERVISED MIL-v1 argmax-run relocation', full_map: 0.21190, kept: false, lessons: 'REVERT but PROMISING: fired at scale (73), Δ+0.03639 RIGHT direction FM2-immune, but P=0.876<0.9 and NEW 0.21190<floor 0.21432. Loss = measurement power + absolute floor (FM18), NOT signal. NEXT: stronger signal (model_mil_v6.pth target, in-vivo-calibrated gate) to convert MORE zero-overlap dets.' },
  { loop: 10, version: 'v9-diag', title: 'Normal-video FP suppression — CLOSED for mAP', full_map: -1, kept: false, lessons: '52% of Normal videos fire >=1 det, but oracle (delete ALL Normal FPs) = +0.00253 dmAP only (low-conf tail, below the 4 AP checkpoints). Banned family 9. analysis_v9_normal_fp.py.' },
]
let bestFullMap = REF_MAP
let bestVersion = 'v137+BUE'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": full mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED baseline: ' + bestVersion + ' (BUE-on; fresh 65-video reference mAP ' + REF_MAP + ', floor ' + FLOOR.toFixed(5) + ').\nSession history:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'What it does, mechanically, 3-6 sentences' },
    target_family: { type: 'string', description: 'supervised-mil-relocation | candidate-generation | recall-ceiling | other (NOT boundary-growth=BUE; NOT scan_segment-abs-conf; NOT normal-fp-suppression)' },
    expected_map_gain: { type: 'number', description: 'Honest expected absolute mAP gain over ' + REF_MAP + '; must be >= 0.03' },
    effect_size_arithmetic: { type: 'string', description: 'Which AP checkpoints move, how many of the 50 zero-overlap dets convert to IoU>=0.3/0.5, and why absolute mAP exceeds ' + REF_MAP + ' with a paired CI that can clear zero (P>=0.9)' },
    localizer_calibration: { type: 'string', description: 'The verdict-independent pre-calibratable signal (which MIL model / actionness) and its IN-VIVO calibration (NOT a blind train-side threshold like v146 NORMAL-P90). If it touches scan_segment abs conf -> auto-reject.' },
    count_neutral: { type: 'boolean', description: 'must be true (FM2); if false, justify' },
    inv_ivo_sentinels: { type: 'string', description: 'Pre-registered smoke sentinels (relocation rate, on-GT>=off-GT displacement, no degenerate pin) that WILL run before the full eval' },
    implementation_plan: { type: 'string', description: 'Exact files/functions/lines (e.g. pipeline.py:1345 mil_scores, :982 model load), config field, CLI flags, per-fire pre-mechanism field name, marker string incl. EMPTY-video path (FM17)' },
    paired_measurable: { type: 'boolean', description: 'must be true — the per-fire pre-mechanism window field for --paired-json' },
    fm_risk_table: { type: 'string', description: 'FM1-FM18 each: immune / at-risk + why' },
    banned_family_check: { type: 'string', description: 'Explicit statement it is none of the 9 banned families' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string', description: 'The two strongest arguments against it' },
  },
  required: ['title', 'mechanism', 'target_family', 'expected_map_gain', 'effect_size_arithmetic', 'localizer_calibration', 'count_neutral', 'inv_ivo_sentinels', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'banned_family_check', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}

const verdictSchema = { type: 'object', properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' }, effect_size_ok: { type: 'boolean' }, banned_family_ok: { type: 'boolean' }, calibration_ok: { type: 'boolean' }, absolute_floor_argument_ok: { type: 'boolean', description: 'Does it argue absolute mAP > ' + REF_MAP + ', not just a within-run paired delta (FM18)?' } }, required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'calibration_ok', 'absolute_floor_argument_ok'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string' }, paired_marker: { type: 'string', description: 'the --paired-json field name (or marker) for paired mode' }, files_changed: { type: 'string' }, validation: { type: 'string' }, notes: { type: 'string' } }, required: ['implemented', 'activation_marker', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
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
'You are Agent D-launcher. ONLY job: ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT replies; every Bash call < 150s. StructuredOutput EXACTLY ONCE.',
'', 'Paths: output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (verbatim): ' + (evalFlags || '(none)'),
'STEPS:',
'1. ps aux | grep -i batch_evaluate | grep -v grep',
'2. If an eval for batch_eval_' + V + '_' + tag + ' is already running -> ps -o command= -p <pid>; if required flags MISSING kill & relaunch; if present return launched=true note="adopted".',
'3. If the report exists AND the log has the required flags -> launched=true note="exists".',
'4. If a DIFFERENT version eval is running -> Bash "sleep 140", re-check up to 3x; still busy -> launched=false note="blocked".',
'5. Else record the command in <output_dir>/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + ROOT + "/results/batch_eval_" + V + "_" + tag + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + (evalFlags || '') + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. Bash "sleep 30"; tail -5 the log for a Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, marker, nTotal) {
  const rep = ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json'
  const lg = ROOT + '/results/batch_eval_' + V + '_' + tag + '.log'
  const N = nTotal || 65
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + ' (' + N + ' videos). SHORT; NO Bash > 150s. Do NOT edit code. StructuredOutput EXACTLY ONCE.',
'Paths: report=' + rep + ' ; log=' + lg + ' ; inner=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/run.log ; Marker "' + (marker || '(none)') + '"',
'STEPS:',
'1. If the report exists, extract & return done=true crashed=false:',
'   ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti[\'avg_per_video_seconds\'])"',
'   markers: grep -cF "' + (marker || '') + '" ' + lg + ' (also inner run.log; trust the LARGER count).',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". If alive -> Bash "sleep 140", re-check up to 2 sleeps, then done=false note="running: <n>/' + N + ' JSONs".',
'   STALL: inner run.log not grown >30 min while alive -> kill & relaunch the SAME nohup at --concurrency ' + CONCURRENCY + ' (skips existing JSONs). note="restarted after stall".',
'3. If dead no report: tail -30. TRANSIENT (API timeout/network) -> relaunch note="relaunched". CODE BUG (project Traceback) -> crashed=true. FM17 (markers<succeeded, empty video skipped marker) -> crashed=true note="FM17 deadlock".',
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
'You are Agent D-adjudicator for ' + V + '. Run the MANDATED video-matched bootstrap CI. Do NOT edit pipeline code. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'Run (self-reports the pre-registered gate):',
paired
  ? ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + pairedMarker + ' --min-delta ' + MIN_DELTA_PAIRED + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5) + '   (if that field is absent, try --paired-log ' + dir + '/run.log --marker "' + pairedMarker + '")')
  : ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta ' + MIN_DELTA_CROSS + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5)),
'If >150s, re-invoke with --resamples 300 and say so. Tool reads per-video artifacts directly if the report is missing.',
'CRITICAL (FM18): new_map is the ABSOLUTE mAP; KEEP needs new_map >= ' + FLOOR.toFixed(5) + ', not merely a positive paired delta over a low within-run arm.',
'Report ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, note (VERDICT line verbatim).',
  ].join('\n')
}
function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. GOAL: a mechanism whose HONEST expected effect is >= +0.03 ABSOLUTE mAP over the committed BUE-on baseline ' + REF_MAP + ' on the FIXED 65-video harness, in the SUPERVISED-MIL count-neutral relocation family. ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. MANDATORY READING (Read offset/limit): ' + LOGS + '/ideas/loop2_v146_marl-d-mil-actionness-run-localization_20260801.md (the mechanism you are strengthening — its 15/37-recoverable ceiling, the P=0.876 + floor loss), ' + LOGS + '/summary_v9_v145-v146_20260801.md (FM18, next-direction #1), the MIL code at pipeline.py:982/1345/1579/2656.',
'2. PROFILE with the CACHED runs + the MIL scores: how many of the 50 zero-overlap dets does the v1 MIL argmax-run land ON GT, vs model_mil_v6.pth? Quantify on-GT vs off-GT displacement separation for BOTH models BEFORE proposing. Show that your stronger signal converts MORE dets than v146 (which needed a larger effect to clear P>=0.9 and the floor).',
'3. SEARCH: >=4 real WebSearch queries on weakly-supervised TAL / MIL snippet localization / actionness completeness 2024-2026. If unavailable, say so and rely on code+data.',
'4. Derive ONE architecture-level mechanism: COUNT-NEUTRAL, verdict-independent SUPERVISED localizer (NOT scan_segment abs conf), PAIRED-measurable (per-fire pre-mechanism window field), IN-VIVO-calibrated gate (NOT blind train-P90), empty-video-safe marker (FM17). It MUST argue ABSOLUTE mAP > ' + REF_MAP + ' (FM18) and a path to P>=0.9 on n=65. Carry the effect-size arithmetic, config flag + CLI, marker, and FM1-FM18 table.',
'5. Return via StructuredOutput (EXACTLY ONCE).',
'!!! ANTI-STALL: no Bash > ~60s, no sleep, Read offset/limit, short progress line, finish in <18 tool calls.',
  ].join('\n')
}
function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). Implement the APPROVED proposal exactly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED CHANGES:\n' + verdict.mandated_changes : '',
'STEPS:',
'1. mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy every file you will edit BEFORE editing.',
'2. Implement. New params in config.py (default OFF/prior behavior) + config.from_args threading; CLI in BOTH run.py and batch_evaluate.py; bump pipeline.py header to ' + V + '. Do NOT disturb BUE (default-ON) or CEX (default-OFF). If loading model_mil_v6.pth, load it read-only as a relocation target; do NOT change the pipeline scoring/confidence/curve.',
'3. FM5+FM17: unique INFO marker per fire AND a per-video summary marker that fires ONCE PER SUCCEEDED VIDEO INCLUDING the empty/0-detection / mil_scores==None path. Log the per-fire PRE-MECHANISM window (e.g. pre_' + V.toLowerCase() + '_boundaries) index-aligned for --paired-json.',
'4. VALIDATE and report outputs: ' + PY + ' -m py_compile each edited file; import MultiAnomalyVAD; flags in --help; ScoringConfig() BUE/CEX defaults unchanged; from_args(Namespace()).scoring == ScoringConfig(); AND a direct check the marker fires on a synthetic empty-detection path (FM17).',
'5. Return implemented, activation_marker, eval_flags, paired_marker (the --paired-json field name), files_changed, validation, notes.',
'ANTI-STALL: no Bash > ~150s; Read offset/limit; do NOT run an eval here.',
  ].join('\n')
}
function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record honestly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '', 'IMPLEMENTATION:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL: ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot),
'\nORCHESTRATOR DECISION: ' + decisionText + (fm5 ? '\nFM5/FM17 VIOLATION: marker never appeared / deadlocked.' : ''),
'TASKS:',
'1. If REVERT: restore every file from ' + ROOT + '/.qoder/backups/' + V + '/, diff -q per file, confirm 0 mechanism symbols (grep) and BUE default-ON / CEX default-OFF intact. If KEEP, leave code and flip the config default ON if the proposal says so.',
'2. Archive into ' + LOGS + '/results/: ' + V + '_full_eval.json, ' + V + '_full_run.log, ' + V + '_smoke_eval.json, bootstrap_ab.json (whichever exist).',
'3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date: date +%Y%m%d): mechanism, effect-size arithmetic vs actual, the BOOTSTRAP CI + ABSOLUTE mAP vs floor ' + FLOOR.toFixed(5) + ' as the headline, FM1-FM18 table, weaknesses, what the next loop should NOT repeat.',
'4. APPEND one row to ' + LOGS + '/session_v97_ledger.md (loop | version | title+verdict+evidence | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept). -1 for absent. Do NOT rewrite rows.',
'5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v10 starting at v' + START_VERSION + '. Committed = v137+BUE (BUE-on 65-video baseline ' + REF_MAP + ', floor ' + FLOOR.toFixed(5) + '). ' + MAX_LOOPS + ' loops pushing the SUPERVISED-MIL count-neutral relocation lever (v146 MARL-D follow-up). concurrency ' + CONCURRENCY + '. Normal-FP suppression is CLOSED (+0.00253 ceiling).')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' ===')

  phase('Research')
  const focusA = (loop === 1)
    ? 'YOUR ANGLE (loop 1, the owner-mandated lever): STRENGTHEN THE MARL-D SIGNAL. Use model_mil_v6.pth (stronger standalone) as the relocation target instead of v1 model_mil.pth, and replace v146\'s blind NORMAL-train-P90 decisiveness gate with an IN-VIVO-calibrated one, so MORE of the 50 zero-overlap dets are relocated onto GT — enough to lift ABSOLUTE mAP over ' + REF_MAP + ' and move the paired CI off zero (P>=0.9). Count-neutral, paired-measurable, empty-video-safe.'
    : 'YOUR ANGLE: iterate on what the previous loop showed for the SUPERVISED-MIL relocation lever — push the signal / calibration / eligibility to convert more zero-overlap dets while staying count-neutral and clearing the ABSOLUTE floor ' + REF_MAP + '.'
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', focusA), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: an ALTERNATIVE verdict-independent SUPERVISED localizer or a better IN-VIVO calibration of the relocation eligibility (e.g. actionness/completeness from features/test, MIL run-geometry, or a calibrated on-GT-probability gate), still count-neutral + paired + empty-video-safe. Attack v146\'s failure mode: the effect was too small/noisy to clear P>=0.9 and the absolute floor — your job is a LARGER, more CONSISTENT conversion of the 50 zero-overlap dets.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Cross-critique and synthesize ONE proposal for this loop.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'RULES: pick the ONE with the larger HONEST expected ABSOLUTE gain over ' + REF_MAP + ' and cleaner measurability (FM8: one change). Kill any part < +0.03 (FM16). REJECT any boundary-growth (BUE), scan_segment-abs-conf gate (banned 8), or Normal-FP suppression (banned 9). Ensure count-neutral, paired-measurable, in-vivo-calibrated, empty-video-safe (FM17), and an explicit ABSOLUTE-floor argument (FM18). Return the unified proposal via StructuredOutput (EXACTLY ONCE).',
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
'1. EFFECT SIZE (binding): expected ABSOLUTE gain over ' + REF_MAP + ' >= +0.03 with HONEST AP-checkpoint arithmetic (how many of the 50 zero-overlap dets convert)? If ~+0.01, REJECT (FM16).',
'2. ABSOLUTE FLOOR (FM18, v146 killer): does it argue NEW ABSOLUTE mAP > ' + REF_MAP + ', not merely a positive within-run paired delta? Set absolute_floor_argument_ok. If it only promises a paired delta, REJECT.',
'3. BANNED FAMILIES 1-9 (incl. 7 boundary=BUE, 8 scan_segment-abs-conf, 9 Normal-FP)? If yes -> REJECT.',
'4. CALIBRATION: verdict-independent + pre-calibratable (MIL/actionness), NOT scan_segment abs conf, with an IN-VIVO-calibrated gate (not blind train-P90)? Are the in-vivo sentinels pre-registered to run in SMOKE? Set calibration_ok.',
'5. FM2: count-neutral? If additive, REJECT unless the FP-demotion bound is airtight.',
'6. FM5+FM17: flag (default OFF, flag-off byte-identical to BUE-on committed) + CLI both runners + marker on the EMPTY-video path + a per-fire pre-mechanism field for --paired-json?',
'7. FM8/FM11: exactly ONE change, one full eval, BUE default-ON + CEX default-OFF untouched?',
'Set effect_size_ok, banned_family_ok, calibration_ok, absolute_floor_argument_ok. REJECT with concrete feedback or APPROVE (optionally mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (r' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. C REJECTED your proposal. Revise or pivot within the SUPERVISED-MIL count-neutral relocation family; clear >= +0.03 ABSOLUTE over ' + REF_MAP + ', avoid all 9 banned families, keep count-neutral+paired+in-vivo-calibrated+empty-video-safe. ONE change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'REJECTED:\n' + JSON.stringify(proposal, null, 1), '', "C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) {
      proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED:\n' + verdict.mandated_changes })
      log('Loop ' + loop + ': proceeding with C-mandated revision')
    } else {
      log('Loop ' + loop + ': C rejected 3x - no eval spent (FM11)')
      history.push({ loop: loop + 10, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
      continue
    }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker) {
    log('Loop ' + loop + ': implementation failed / no marker (FM5) - skipping eval')
    history.push({ loop: loop + 10, version: V, title: proposal.title + ' [IMPL FAILED]', full_map: -1, kept: false, lessons: impl ? ('failed: ' + (impl.notes || '').slice(0, 200)) : 'agent null' })
    continue
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
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from full log') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, impl.paired_marker || ''), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        const paired = !!(impl.paired_marker && impl.paired_marker.length > 0)
        const bar = paired ? MIN_DELTA_PAIRED : MIN_DELTA_CROSS
        if (!boot || !boot.ran) decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'agent null') + ')'
        else {
          const okD = boot.point_delta >= bar, okP = boot.p_gt0 >= MIN_P, okF = boot.new_map >= FLOOR
          if (okD && okP && okF) { keep = true; decisionText = 'KEEP (' + (paired ? 'paired' : 'cross') + ' delta ' + boot.point_delta.toFixed(5) + ' >= ' + bar + ', P(d>0)=' + boot.p_gt0.toFixed(3) + ' >= ' + MIN_P + ', ABSOLUTE mAP ' + boot.new_map.toFixed(5) + ' >= floor ' + FLOOR.toFixed(5) + '. CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + ']. New baseline.)' }
          else if (!okF) decisionText = 'REVERT (FM18 ABSOLUTE FLOOR breached: mAP ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ' — a positive paired delta over a low within-run arm is NOT a win)'
          else if (!okD) decisionText = 'REVERT (effect too small: delta ' + boot.point_delta.toFixed(5) + ' < ' + bar + '; P=' + boot.p_gt0.toFixed(3) + '. FM16.)'
          else decisionText = 'REVERT (delta ' + boot.point_delta.toFixed(5) + ' but P(d>0)=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + '] straddles zero — v146 repeat)'
        }
      }
    }
  } else log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ')')
  if (fm5) decisionText = 'REVERT (FM5/FM17: marker never appeared / deadlocked)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5/FM17' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [d=' + boot.point_delta.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ' mAP=' + boot.new_map.toFixed(5) + ']') : ''
  history.push({ loop: loop + 10, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) {
    bestFullMap = result.full_map; bestVersion = V; REF_DIR = ROOT + '/results/batch_eval_' + V + '_full'; REF_MAP = result.full_map; FLOOR = Math.max(0.21432, result.full_map - MAP_TOL)
    log('Loop ' + loop + ': KEPT ' + V + ' - new baseline mAP ' + result.full_map + ' (floor now ' + FLOOR.toFixed(5) + ')')
  } else log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 160))
}

// ============================ END-OF-SESSION SUMMARY ============================
phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for the v10 continuation (v147-v149, SUPERVISED-MIL relocation lever). Plain text; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'TASK: (1) Read ' + LOGS + '/session_v97_ledger.md (tail) + the new v147-v149 idea files. (2) date +%Y%m%d. (3) Write ' + LOGS + '/summary_v10_v147-v149_<YYYYMMDD>.md: executive summary (did any loop clear >= +0.03 with P>=' + MIN_P + ' AND ABSOLUTE mAP > ' + REF_MAP + '? did the floor hold?), per-version table (mechanism | MIL model | reloc count | point delta | CI90 | P(d>0) | ABSOLUTE mAP | kept), an explicit verdict on whether the SUPERVISED-MIL relocation lever is now WON, still-open, or CLOSED (and if closed, why — signal ceiling vs measurement power), the updated FM1-FM18 taxonomy, and concrete next directions respecting the noise floor + absolute floor. (4) APPEND a ledger row only if a loop was KEPT (do NOT rewrite rows). (5) Reply with a 10-line recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v10 complete. Committed baseline ' + bestVersion + ' mAP ' + bestFullMap.toFixed(5) + ' (BUE-on). ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, refMap: REF_MAP, floor: FLOOR }
