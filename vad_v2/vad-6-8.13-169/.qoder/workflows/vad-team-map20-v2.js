export const meta = {
  name: 'vad-team-map20-v2',
  description: 'VAD agent team v2 (FM5-hardened): A/B research+debate, C skeptical review, D implement+eval with mandatory activation gates. Continues at v103 after the v97-v102 session (6 loops done, true baseline still v94 mAP=0.1494). Target full-eval temporal mAP>=0.20.',
  phases: [
    { title: 'Research', detail: 'A and B independently research papers + codebase, each proposes one architecture-level idea' },
    { title: 'Debate', detail: 'A and B cross-critique, then synthesize one joint consensus proposal' },
    { title: 'Review', detail: 'C approves or rejects with feedback; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (CLI-activatable + log marker), smoke+full eval with FM5 activation gate, keep/revert' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const TARGET = 0.2
const MAX_LOOPS = 20
const START_VERSION = 103
const BASE_LATENCY = 128

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision: video-level labels for training; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline, versioned docstring header; config.py centralizes ALL params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up before editing.',
'',
'TRUE BASELINE (full 65-video eval): temporal mAP = 0.1494 (v89/v94), AP@0.5 = 0.1335, AP@0.7 = 0.1096, frame AUC ~0.69-0.71, avg latency ~128 s/video at concurrency 3.',
'TARGET: full-eval temporal mAP >= 0.20 while keeping per-video latency near or below ~128s.',
'VLM API stochasticity is about +/-0.03 mAP: deltas within noise are NOT real improvements.',
'',
'!!! FM5 LESSON (the most important lesson of the v97-v102 session) !!!',
'In the previous session, SIX consecutive versions (v97-v102) were evaluated with the new mechanism NEVER ACTUALLY ACTIVATED: the implementation added an opt-in CLI flag, but the eval launcher command never passed it, so the live runs were byte-identical to the baseline and every "result" was pure VLM noise. v97\'s apparent mAP=0.1616 was EXPOSED as flag-off noise (0 FCCS markers, 59 [Directed] lines in its own log) - the true baseline remains v94=0.1494. THEREFORE, in this session:',
'- Every mechanism MUST be activatable via a REAL CLI flag on batch_evaluate.py in the CURRENT tree (verify with: python3 batch_evaluate.py --help) OR be default-ON via config.py.',
'- Every mechanism MUST emit a unique INFO-level log marker per video when it fires (e.g. "[XYZ] ...").',
'- The orchestrator reports eval_flags + activation_marker; the launcher MUST include eval_flags; after each eval the log is greped for the marker. 0 markers = FM5 = automatic REVERT, no exceptions.',
'',
'MANDATORY READING before proposing or reviewing (skim smartly, newest first):',
'1. ' + CODE + '/iteration_logs/session_v97_ledger.md  (v97-v102 outcomes + FM5 gate failures)',
'2. ' + CODE + '/iteration_logs/ideas/ - ALL files, newest first (v97-v102 idea+report files are the most relevant; then loop*_v8x/v9x)',
'3. ' + CODE + '/iteration_logs/summary_loops_v90-94_20260712.md  (failure-mode taxonomy FM1-FM4)',
'4. ' + CODE + '/iteration_logs/summary_loops1-10_v86-89_20260712.md',
'5. ' + ROOT + '/.qoder/specs/VAD_mAP_50%_优化方案_c990fbf7.md  (long-term roadmap + rejected alternatives)',
'6. Research methodology: ' + ROOT + '/academic-research-skills/deep-research/SKILL.md (+ references/ dir) - then run REAL WebSearch/WebFetch queries (arXiv, PapersWithCode, CVF openaccess).',
'',
'KNOWN FAILURE MODES:',
'FM1 curve-shape modification (fusion weights, feature re-centering, confusion-negative scoring, candidate pooling) regresses even when the modifying signal is individually better.',
'FM2 detection-count changes under greedy-match evaluation (e.g. Soft-NMS) turn near-duplicates into FPs. NOTE: v97-style WHOLESALE scan-mechanism replacement was argued to escape FM2 (same VLM, same segments; only coverage + judgment mode change) but remains UNVERIFIED in vivo.',
'FM3 VLM self-consistency via temperature diversity hurts perceptual/extraction tasks (reading timestamps).',
'FM4 self-consistency on segment-inclusion decisions is dangerous; on already-fixed detection labels (category tie-break) it is safe.',
'FM5 (NEW, dominant in v97-v102): mechanism implemented but never activated in the eval run (missing CLI flag / flag-off-equivalent). Six consecutive versions wasted. Guard: mandatory eval_flags + activation marker + log grep gate.',
'Also known: MIL v3/v4 LGT-Adapter collapsed (75-83% of videos); MIL v6 rejected (frame-level localization near chance, archived v80-85b); mAP ~0.14-0.16 is a STRUCTURAL ceiling for incremental tuning - only ARCHITECTURE-LEVEL changes can break it.',
'',
'VALIDATED SAFE/positive PATTERNS: confidence-only modulation computed read-only from existing curves (v95 TSM sharpness); category tie-break on fixed detections (v89); learned prompt-ensemble weights strictly upstream of fusion/boundary/candidate logic (v94).',
'',
'CURRENT CODE STATE (working tree): contains v97 FCCS implementation (full-coverage comparative VLM scanning; config switch full_coverage_comparative_scan default OFF; CLI flag --full_coverage_scan exists) + v100 EUBC code (inert, flag off). FCCS was C-approved with independent verification but NEVER TESTED in vivo - it is the loop-1 seeded experiment of this session.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop (controlled experiment).',
'- ARCHITECTURE-LEVEL ideas only - NO pure parameter/threshold tuning.',
'- All new params via config.py with a switch restoring prior behavior; every mechanism CLI-activatable + INFO log marker (FM5 guard).',
'- Keep the v51 half-max boundary-crossing mechanism unless a strong literature-backed argument says otherwise.',
'- Latency matters: prefer zero-cost (precomputed text embeddings) or training-free signals; avoid extra per-video VLM calls.',
'- When calling StructuredOutput: call it EXACTLY ONCE, at the very end, with the final answer. NEVER call it twice.',
].join('\n')

const history = [
  { loop: 1, version: 'v97', title: 'FCCS full-coverage comparative scan', smoke_map: 0.15035, full_map: 0.16155, avg_latency: 141.47, kept: true, lessons: 'FM5 EXPOSED: --full_coverage_scan never passed (0 [FCCS] markers, 59 [Directed] lines in own log); 0.1616 is flag-off NOISE; mechanism implemented but UNTESTED in vivo' },
  { loop: 2, version: 'v98', title: 'Verified FCCS baseline attempt', smoke_map: 0.15035, full_map: 0.14504, avg_latency: 147.9, kept: false, lessons: 'FM5: flag omitted again; FCCS still unevaluated' },
  { loop: 3, version: 'v99', title: 'ECE event-continuation extension', smoke_map: 0.14666, full_map: 0.15814, avg_latency: 148.18, kept: false, lessons: 'FM5: 0 [ECE] lines; never activated' },
  { loop: 4, version: 'v100', title: 'EUBC event-unit boundary contraction', smoke_map: 0.15035, full_map: 0.15713, avg_latency: 133.27, kept: true, lessons: 'FM5: 0 [EUBC] lines; kept code inert (flag off, byte-identical)' },
  { loop: 5, version: 'v101', title: 'BRP bottom-ranked recovery proposals', smoke_map: 0.15035, full_map: 0.13949, avg_latency: 145.51, kept: false, lessons: 'FM5: 0 [Recovery] lines' },
  { loop: 6, version: 'v102', title: 'HZG-FC hierarchical zoom grounding + forced-choice zoom', smoke_map: 0.1637, full_map: 0.15861, avg_latency: 150.57, kept: false, lessons: 'FM5: 0 [HZG] lines; VAD_REQUIRE_FLAGS never set by launcher' },
]
let bestFullMap = 0.1494
let bestVersion = 'v94'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": smoke=' + h.smoke_map + ' full=' + h.full_map + ' latency=' + h.avg_latency + 's kept=' + h.kept + ' | lessons: ' + h.lessons)
  return 'TRUE current best: ' + bestVersion + ' full mAP=' + bestFullMap + ' (v97 0.1616 was flag-off noise, NOT a real best). Session history (previous run, loops 1-6 = v97-v102):\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  required: ['title', 'mechanism', 'rationale', 'implementation_plan', 'expected_gain', 'risks', 'latency_impact', 'literature'],
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'the architecture-level change, precisely' },
    rationale: { type: 'string', description: 'why it should break the 0.149 ceiling; evidence from papers + this codebase' },
    implementation_plan: { type: 'string', description: 'exact files/functions to change, step by step; MUST state the CLI flag to activate it and the INFO log marker it emits (FM5 guard)' },
    expected_gain: { type: 'string' },
    risks: { type: 'string', description: 'which known failure modes FM1-FM5 it might trigger and why it avoids them' },
    latency_impact: { type: 'string' },
    literature: { type: 'string', description: 'papers cited (name, venue/year, what mechanism was borrowed)' },
  },
}

const verdictSchema = {
  type: 'object',
  required: ['approve', 'feedback'],
  properties: {
    approve: { type: 'boolean' },
    feedback: { type: 'string', description: 'concrete reasons; if rejecting, what specifically must change' },
    mandated_changes: { type: 'string', description: 'optional: minimal changes that would make this approvable' },
  },
}

const resultSchema = {
  type: 'object',
  required: ['version', 'implemented', 'smoke_map', 'full_map', 'avg_latency', 'kept', 'summary', 'lessons'],
  properties: {
    version: { type: 'string' },
    implemented: { type: 'boolean' },
    smoke_map: { type: 'number', description: '-1 if not run' },
    full_map: { type: 'number', description: '-1 if not run' },
    avg_latency: { type: 'number', description: 'full eval avg_per_video_seconds; -1 if not run' },
    kept: { type: 'boolean', description: 'true if the change was kept, false if reverted' },
    summary: { type: 'string' },
    lessons: { type: 'string', description: 'what was learned; which failure modes confirmed/refuted' },
    report_path: { type: 'string' },
  },
}

const implSchema = {
  type: 'object',
  required: ['implemented', 'files_changed', 'notes', 'eval_flags', 'activation_marker'],
  properties: {
    implemented: { type: 'boolean' },
    files_changed: { type: 'string' },
    notes: { type: 'string' },
    eval_flags: { type: 'string', description: 'exact extra CLI args for batch_evaluate.py that ACTIVATE the mechanism (e.g. "--full_coverage_scan"); empty string only if the mechanism is default-ON via config' },
    activation_marker: { type: 'string', description: 'unique literal log substring emitted at INFO level per video when the mechanism fires (e.g. "[FCCS]"); used to grep-verify activation (FM5 guard)' },
  },
}
const launchSchema = {
  type: 'object',
  required: ['launched', 'note'],
  properties: { launched: { type: 'boolean' }, note: { type: 'string' } },
}
const watchSchema = {
  type: 'object',
  required: ['done', 'crashed', 'mAP', 'ap30', 'ap50', 'ap70', 'latency', 'marker_count', 'note'],
  properties: {
    done: { type: 'boolean' },
    crashed: { type: 'boolean' },
    mAP: { type: 'number', description: '-1 if unknown' },
    ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' },
    latency: { type: 'number', description: 'avg_per_video_seconds, -1 if unknown' },
    marker_count: { type: 'number', description: 'occurrences of the activation marker in the eval log; -1 if not checked' },
    note: { type: 'string' },
  },
}

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try {
      const r = await fn()
      if (r !== null && r !== undefined) return r
      log('retryAgent: attempt ' + (t + 1) + '/' + n + ' returned null')
    } catch (e) {
      const msg = (e && e.message) ? e.message : String(e)
      log('retryAgent: attempt ' + (t + 1) + '/' + n + ' threw: ' + msg.slice(0, 200))
    }
  }
  return null
}

function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. ' + focus,
'',
CONTEXT,
'',
'SESSION HISTORY (do NOT repeat already-tried ideas; note how many died from FM5 non-activation, not bad mechanisms):\n' + hist(),
'',
'TASK:',
'1. Do the MANDATORY READING (session ledger + newest idea files are the highest value; then skim pipeline.py docstring + config.py + the 2-3 most relevant source files for your angle).',
'2. SEARCH-STORM: run at least 6 real WebSearch queries on 2023-2026 papers about TEMPORAL LOCALIZATION mechanisms (how exactly do they localize: proposal generation, boundary regression, start/end probability curves, Gaussian modeling, MIL snippet scoring, temporal contrast, motion/optical-flow signals, training-free localization). Use WebFetch on the most promising 2-3 papers (arXiv abs/html pages) to extract their exact localization mechanism.',
'3. From the literature + codebase evidence, derive ONE architecture-level idea that could break the 0.149 structural ceiling. It must respect the HARD CONSTRAINTS and avoid FM1-FM5. Parameter tuning is FORBIDDEN. The idea MUST be CLI-activatable and emit a unique INFO log marker (FM5 guard) - state both in the implementation_plan.',
'4. Return your proposal via StructuredOutput (call it EXACTLY ONCE). Be concrete: name the exact files/functions to change.',
  ].join('\n')
}

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
  return [
'You are Agent D-launcher. Your ONLY job: ensure the ' + tag + ' evaluation for version ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. Do NOT wait for it to finish. Keep replies SHORT; every Bash call < 9 minutes. Call StructuredOutput EXACTLY ONCE at the end.',
'',
'Paths: code=' + CODE + ' ; output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (FM5-critical, must appear verbatim in the command): ' + (evalFlags || '(none - mechanism is default-on)'),
'',
'STEPS:',
'1. ps aux | grep -i batch_evaluate | grep -v grep',
'2. If an eval for batch_eval_' + V + '_' + tag + ' is ALREADY running -> check its command line contains the required flags: ps -o command -p <pid> ; if flags are MISSING, kill it and relaunch with flags; if present -> launched=true, note="adopted existing run".',
'3. If the report file already exists AND the log shows the required flags were used (grep the flag string in the log; if no flags required, any existing report is fine) -> launched=true, note="report already exists". Otherwise delete the stale output_dir and relaunch.',
'4. If a DIFFERENT version\'s eval is running -> sleep 120 and re-check, up to 3 times; if still busy -> launched=false, note="blocked by other eval".',
'5. Otherwise launch exactly (note the extra flags before the redirect):',
"bash -lc 'source " + CODE + "/.env 2>/dev/null; export HF_ENDPOINT=https://hf-mirror.com; export PYTHONUNBUFFERED=1; cd " + CODE + "; nohup python3 -u batch_evaluate.py --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency 3 --detect_model qwen-vl-plus --classify_model qwen3-vl-plus --fps 2 --clip_model ViT-L-14 --clip_pretrained openai --device mps --mcts_simulations 12 --mcts_max_depth 4 " + (evalFlags || '') + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log to confirm it started (no immediate Traceback). Return via StructuredOutput: launched (bool), note.',
  ].join('\n')
}

function evalWatchPrompt(V, tag, maxPerCat, evalFlags, marker) {
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + '. Keep replies SHORT; every Bash call < 9 minutes. Do NOT edit any project code. Call StructuredOutput EXACTLY ONCE at the end.',
'',
'Paths: report=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log',
'Activation marker to verify (FM5 guard): "' + (marker || '(none)') + '" ; required flags: ' + (evalFlags || '(none)'),
'',
'STEPS:',
'1. If the report exists, extract metrics AND count activation markers, then return done=true, crashed=false:',
"   python3 -c \"import json;d=json.load(open('" + ROOT + "/results/batch_eval_" + V + "_" + tag + "/evaluation_report.json'));t=d['evaluation']['temporal_map'];ti=d['timing'];print(t['mAP'],t.get('AP@0.3'),t.get('AP@0.5'),t.get('AP@0.7'),ti['avg_per_video_seconds'])\"",
'   Map the 5 printed numbers to mAP, ap30, ap50, ap70, latency. Marker count: grep -cF "' + (marker || '') + '" ' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log  (use -1 only if no marker was specified).',
"2. Else check the process: pgrep -f 'batch_evaluate.*batch_eval_" + V + "_" + tag + "' -> if alive: Bash 'sleep 360' then re-check the report once more; if it appeared, extract metrics + marker count (done=true); otherwise return done=false, crashed=false, marker_count=-1, note='still running'.",
'3. If the process is dead and no report exists: tail -30 the log. If the failure looks TRANSIENT (API timeout / ConnectTimeout / rate limit / network), relaunch with the exact nohup command (max_per_category ' + maxPerCat + ', WITH the required flags: ' + (evalFlags || '(none)') + ') and return done=false, crashed=false, marker_count=-1, note="relaunched after: <cause>". If it looks like a CODE BUG (Python Traceback referencing project code), return done=false, crashed=true, marker_count=-1, note=<error summary>.',
'Return via StructuredOutput (EXACTLY ONCE): done, crashed, mAP, ap30, ap50, ap70, latency (use -1 when unknown), marker_count, note.',
  ].join('\n')
}

async function runEval(V, tag, maxPerCat, evalFlags, marker) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: 'Implement' }))
  if (!launch || !launch.launched) {
    return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'agent null after retries') }
  }
  for (let w = 0; w < 16; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, maxPerCat, evalFlags, marker), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: 'Implement' }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap reached without completion' }
}

function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1, the implementer of this VAD team. Implement the approved proposal in the codebase. DO NOT run any evaluation - later agents handle evals. Call StructuredOutput EXACTLY ONCE at the end.',
'',
CONTEXT,
'',
'APPROVED PROPOSAL FOR ' + V + ':\n' + JSON.stringify(proposal, null, 1),
'',
"C's review feedback + mandated changes (must follow):\n" + (verdict ? ((verdict.feedback || '') + '\n' + (verdict.mandated_changes || '')) : '(none)'),
'',
'RULES:',
'- Work CONTEXT-EFFICIENTLY: use Grep to locate code; Read with offset/limit; NEVER read whole large files or cat logs.',
'- Backup BEFORE editing: mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and cp every file you will modify into it (skip files already backed up by a previous attempt).',
'- A previous attempt may have PARTIALLY implemented this version. Inspect the target files first; complete what is missing instead of starting over.',
'- Implement the proposal faithfully, including ALL of C\'s mandated changes. All new params via config.py with a switch restoring prior behavior.',
'- FM5-CRITICAL: (a) the mechanism MUST be activatable via a real CLI flag on batch_evaluate.py (add it via argparse + VADConfig.from_args getattr threading, verify with: python3 batch_evaluate.py --help | grep your-flag) OR be default-ON via config; (b) the mechanism MUST emit a unique INFO-level log marker per video when it fires (e.g. logger.info("[XYZ] ...")).',
'- Update the pipeline.py docstring header with a concise ' + V + ' entry (follow the existing version-note convention).',
'- Sanity-check each edited file: python3 -c "import ast; ast.parse(open(\'<f>\').read())"; then cd ' + CODE + ' && python3 -c "import pipeline".',
'Return via StructuredOutput (EXACTLY ONCE): implemented (bool), files_changed (comma-separated), notes, eval_flags (exact CLI args that ACTIVATE the mechanism, e.g. "--full_coverage_scan"; empty string only if default-ON), activation_marker (the literal INFO log substring, e.g. "[FCCS]").',
  ].join('\n')
}

function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, decision, fm5) {
  return [
'You are Agent D4, the finalizer for version ' + V + '. Implementation and evals are DONE. Enforce the keep/revert decision, save artifacts, write the report, return the summary. Keep replies focused; every Bash call < 9 minutes. Call StructuredOutput EXACTLY ONCE at the end.',
'',
'Paths: code=' + CODE + ' ; results=' + ROOT + '/results ; backups=' + ROOT + '/.qoder/backups/' + V + ' ; ideas=' + CODE + '/iteration_logs/ideas/ ; runlogs=' + CODE + '/iteration_logs/results/',
'',
'DECISION FROM ORCHESTRATOR (do not second-guess): ' + decision.toUpperCase() + (fm5 ? '  [FM5: mechanism never activated in the live run - 0 marker lines]' : ''),
'Metrics: smoke mAP=' + smoke.mAP + ' (marker_count=' + smoke.marker_count + ') ; full mAP=' + full.mAP + ' ; AP@0.3=' + full.ap30 + ' AP@0.5=' + full.ap50 + ' AP@0.7=' + full.ap70 + ' ; full avg latency=' + full.latency + 's (marker_count=' + full.marker_count + ')',
'Eval flags used: ' + (impl.eval_flags || '(none)') + ' ; activation marker: "' + (impl.activation_marker || '(none)') + '"',
'Eval notes: smoke: ' + (smoke.note || '-') + ' ; full: ' + (full.note || '-'),
'',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
'',
"C's feedback:\n" + (verdict ? (verdict.feedback || '') : '(none)'),
'',
'Implementation notes from D1 (files: ' + (impl.files_changed || '?') + '):\n' + (impl.notes || '(none)'),
'',
'TASKS:',
'1. If decision=REVERT: restore every file present in ' + ROOT + '/.qoder/backups/' + V + '/ back into ' + CODE + '/ and verify each restore with diff (must be identical). If there is no backup dir for ' + V + ' (e.g. verify-only loop), do nothing. If decision=KEEP: do not touch the code.',
'2. Copy artifacts that exist: cp ' + ROOT + '/results/batch_eval_' + V + '_smoke.log ' + CODE + '/iteration_logs/results/' + V + '_smoke_run.log ; cp ' + ROOT + '/results/batch_eval_' + V + '_smoke/evaluation_report.json ' + CODE + '/iteration_logs/results/' + V + '_smoke_eval.json ; same two copies for _full (if the full eval ran).',
'3. date +%Y%m%d, then write ' + CODE + '/iteration_logs/ideas/loop' + loop + '_' + V + '_<kebab-case-slug>_<YYYYMMDD>.md with: the approved proposal, C\'s feedback, implementation notes, the eval flags + activation marker, marker counts in smoke/full logs, a results table (smoke mAP; full mAP, AP@0.3/0.5/0.7, latency), the keep/revert decision + reason, and lessons learned (which failure modes confirmed/refuted).',
'4. Append one row to ' + CODE + '/iteration_logs/session_v97_ledger.md (create the file with a markdown table header if it is missing): | loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept |',
'5. Return via StructuredOutput (EXACTLY ONCE): version="' + V + '", implemented=true, smoke_map=' + smoke.mAP + ', full_map=' + full.mAP + ', avg_latency=' + full.latency + ', kept=' + (decision === 'keep') + ', summary (5-8 sentences), lessons (2-4 sentences), report_path.',
  ].join('\n')
}

// Loop-1 seeded experiment: FCCS live activation test (v97 mechanism, C-approved, never tested in vivo).
const SEED_PROPOSAL = {
  title: 'FCCS Live Activation Test (v97 mechanism, first in-vivo evaluation)',
  mechanism: 'Full-Coverage Comparative VLM Scanning (already implemented in the tree behind --full_coverage_scan): all 6s segments packed into contiguous batches of 8 (16 for >60-segment videos), comparative 0-1 rating against in-video normal baseline, 0.3 threshold, _merge_adjacent, unchanged downstream (top_k=3, v51 half-max, v95/v96 modulators, hard NMS).',
  rationale: 'C-approved in v97 with independent verification (dead-but-intact v56/v57 comparative scan code resurrected + truncation guard; v56 isolated comparative judging as mAP-positive +0.006; coverage traces prove wrong-location-by-construction). The v97 eval never activated it (FM5) - this loop runs the actual experiment.',
  implementation_plan: 'Verify-only: confirm --full_coverage_scan exists in batch_evaluate.py (already verified by orchestrator: batch_evaluate.py:189, config.py:86+168), confirm [FCCS] INFO markers in anomaly_discovery.scan_full_coverage, then evaluate with the flag ON.',
  expected_gain: '+0.01..0.03 (floor band per v97 analysis) if comparative calibration helps; true coverage recovery is the open question.',
  risks: 'FM2-category risk bounded by unchanged top_k/merge/NMS; FM1 untouched; FM3/FM4 absent by construction; FM5 eliminated by this loop\'s activation gate.',
  latency_impact: 'v97 flag-off run measured 141.47s; live run expected similar (+16 images/call but 3-4x fewer calls).',
  literature: 'TimeChat/VTimeLLM (CVPR24), Holmes-VAU (CVPR25), TSPO (AAAI26), T3AL (CVPR24), LLM-as-a-judge calibration (Zheng 2023); see v97 idea file.',
}
const SEED_VERDICT = { approve: true, feedback: 'Pre-approved as v97 (C verdict, previous run). This loop only activates and measures it.', mandated_changes: '' }
const SEED_IMPL = { implemented: true, files_changed: '(none - verify only)', notes: 'v97 FCCS already implemented and verified present in the tree; evaluating with --full_coverage_scan', eval_flags: '--full_coverage_scan', activation_marker: '[FCCS]' }

log('VAD agent team v2 starting. Target mAP>=' + TARGET + ', versions v' + START_VERSION + '+, max ' + MAX_LOOPS + ' loops. FM5-hardened (activation gates).')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION - 1 + loop)
  log('=== Loop ' + loop + ' (' + V + ') ===')

  let proposal, verdict, impl
  if (loop === 1) {
    proposal = SEED_PROPOSAL
    verdict = SEED_VERDICT
    impl = SEED_IMPL
    log('Loop 1: seeded FCCS live-activation test (mechanism pre-approved as v97; skipping research/review/implement)')
  } else {
    phase('Research')
    const [propA, propB] = await parallel([
      () => retryAgent(() => agent(researchPrompt('A', 'Your specialty: TEMPORAL LOCALIZATION ARCHITECTURES from the TAL/VAD literature (BSN/BMN/DBG/ActionFormer-style proposal+boundary mechanisms, VadCLIP/PE-MIL/MGFN/AnomalyCLIP/GS-MoE/FreeZAD-style VAD mechanisms) and how their localization core can be adapted training-free or light-training into this pipeline.'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
      () => retryAgent(() => agent(researchPrompt('B', 'Your specialty: CODEBASE-FIT + BOTTLENECK analysis. Known bottlenecks: 28.8% of candidates at completely wrong locations (low-CLIP categories Shoplifting/Stealing/Arson/Burglary have flat CLIP curves), 39% of detections have IoU 0.1-0.5 (boundary imprecision), MIL temporal discrimination collapsed. You focus on signals the pipeline underuses (motion energy, temporal gradient structure, MIL v6, snippet resolution, multi-signal boundary evidence) and on recent weakly-supervised VAD SOTA papers for transferable mechanisms.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
    ])
    if (!propA && !propB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }
    const pA = propA ? JSON.stringify(propA, null, 1) : '(A failed to return a proposal)'
    const pB = propB ? JSON.stringify(propB, null, 1) : '(B failed to return a proposal)'

    phase('Debate')
    const debateBase = [
CONTEXT,
'',
'SESSION HISTORY:\n' + hist(),
'',
"Proposal from A:\n" + pA + "\n\nProposal from B:\n" + pB,
    ].join('\n')
    const [critAonB, critBonA] = await parallel([
      () => retryAgent(() => agent('You are Agent A continuing the debate. ' + debateBase + '\n\nTASK: Critique B\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM5 or repeat a tried/failed version? (b) is it truly architecture-level? (c) is the expected gain plausible given +/-0.03 noise? (d) latency risk? Then state: would you accept B\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'A-critiques-B-' + V, phase: 'Debate' })),
      () => retryAgent(() => agent('You are Agent B continuing the debate. ' + debateBase + '\n\nTASK: Critique A\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM5 or repeat a tried/failed version? (b) is it truly architecture-level? (c) is the expected gain plausible given +/-0.03 noise? (d) latency risk? Then state: would you accept A\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'B-critiques-A-' + V, phase: 'Debate' })),
    ])

    const consensus = await retryAgent(() => agent([
'You are the joint A+B consensus. Two researchers proposed ideas and cross-critiqued. Reach the SINGLE joint optimal proposal both would sign.',
'',
debateBase,
'',
"A's critique of B:\n" + (critAonB || '(missing)'),
'',
"B's critique of A:\n" + (critBonA || '(missing)'),
'',
'TASK: Synthesize ONE final proposal: pick the stronger idea, or merge both into one coherent SINGLE controlled change (never two independent changes). It must be architecture-level, respect all HARD CONSTRAINTS, avoid FM1-FM5, and not repeat anything in SESSION HISTORY. It MUST be CLI-activatable with a stated INFO log marker (FM5 guard). Return via StructuredOutput (EXACTLY ONCE, same schema as the proposals).',
    ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
    if (!consensus) { log('Loop ' + loop + ': consensus failed, skipping'); continue }

    phase('Review')
    proposal = consensus
    verdict = null
    for (let r = 0; r < 3; r++) {
      verdict = await retryAgent(() => agent([
'You are Agent C, the skeptical reviewer of this VAD team. You gatekeep what reaches implementation. Call StructuredOutput EXACTLY ONCE at the end.',
'',
CONTEXT,
'',
'SESSION HISTORY:\n' + hist(),
'',
'JOINT PROPOSAL FROM A+B (attempt ' + (r + 1) + '):\n' + JSON.stringify(proposal, null, 1),
'',
'REVIEW CHECKLIST (be strict but fair):',
'1. Architecture-level, not disguised parameter tuning?',
'2. Avoids FM1-FM5 and all previously failed/reverted approaches in the history and idea logs?',
'3. Expected gain plausible vs +/-0.03 VLM noise? Is the mechanism causal, not wishful?',
'4. Latency impact acceptable (target <= ~' + BASE_LATENCY + 's/video, hard limit +15%)?',
'5. Implementation risk: reversible, config-gated, one controlled change?',
'6. Is it sufficiently BETTER than doing nothing - does it attack a real bottleneck (wrong candidate locations / boundary imprecision / MIL collapse)?',
'7. FM5 GUARD: does the implementation_plan state (a) the exact CLI flag that activates the mechanism on batch_evaluate.py (or config default-ON) and (b) the unique INFO log marker it emits per video? If either is missing, REJECT and demand it.',
'If the idea is fundamentally flawed or too weak, REJECT with concrete feedback. If good but imperfect, you may APPROVE and still give feedback, or approve with mandated_changes. Return via StructuredOutput (EXACTLY ONCE).',
      ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
      if (!verdict || verdict.approve) break
      log('Loop ' + loop + ': C rejected (round ' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
      if (r === 2) break
      proposal = await retryAgent(() => agent([
'You are the joint A+B team. Reviewer C REJECTED your proposal. Revise it into a NEW or FIXED proposal addressing every point of C\'s feedback. You may pivot to a different mechanism if C\'s critique is fundamental. Still one controlled architecture-level change, respecting all constraints. Call StructuredOutput EXACTLY ONCE.',
'',
CONTEXT,
'',
'SESSION HISTORY:\n' + hist(),
'',
'REJECTED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
'',
"C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED CHANGES:\n' + verdict.mandated_changes : ''),
'',
'Return the revised proposal via StructuredOutput (EXACTLY ONCE).',
      ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
      if (!proposal) { proposal = consensus; break }
    }
    if (verdict && !verdict.approve) {
      if (verdict.mandated_changes) {
        proposal = Object.assign({}, proposal, {
          title: proposal.title + ' (C-mandated revision)',
          implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED CHANGES (must follow):\n' + verdict.mandated_changes,
        })
        log('Loop ' + loop + ': proceeding with C-mandated revision')
      } else {
        log('Loop ' + loop + ': C rejected 3x without mandate, skipping implementation this loop')
        history.push({ loop: loop + 6, version: V, title: proposal.title + ' [REJECTED - no implementation]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
        continue
      }
    }

    phase('Implement')
    impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
    if (!impl || !impl.implemented) {
      log('Loop ' + loop + ': D1 implementation failed')
      history.push({ loop: loop + 6, version: V, title: proposal.title + ' [D1 FAILED]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'implementation failed: ' + (impl && impl.notes ? impl.notes.slice(0, 200) : 'agent null after retries') })
      continue
    }
    if (!impl.activation_marker) {
      log('Loop ' + loop + ': D1 reported no activation marker - FM5 risk, treating as failure')
      history.push({ loop: loop + 6, version: V, title: proposal.title + ' [NO MARKER - FM5]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'D1 did not provide an activation marker; loop aborted before evals' })
      continue
    }
  }

  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker)
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count + ' | ' + (smoke.note || '').slice(0, 150))

  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (gate)' }
  let decision = 'revert'
  let fm5 = false
  if (smoke.done && !smoke.crashed && smoke.mAP >= 0.11) {
    if (impl.activation_marker && smoke.marker_count <= 0) {
      fm5 = true
      log('Loop ' + loop + ': FM5 - activation marker "' + impl.activation_marker + '" absent from smoke log; skipping full eval, will revert')
    } else {
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' latency=' + full.latency + ' markers=' + full.marker_count + ' | ' + (full.note || '').slice(0, 150))
      if (impl.activation_marker && full.done && full.marker_count <= 0) {
        fm5 = true
        log('Loop ' + loop + ': FM5 - activation marker absent from full log')
      }
      if (!fm5 && full.done && !full.crashed && full.mAP >= bestFullMap - 0.005 && (full.latency <= 0 || full.latency <= Math.round(BASE_LATENCY * 1.15))) {
        decision = 'keep'
      }
    }
  } else {
    log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + '), skipping full eval, will revert')
  }

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, decision, fm5), { schema: resultSchema, label: 'D4-finalize-' + V, phase: 'Implement' }))
  const result = fin || {
    version: V, implemented: true,
    smoke_map: smoke.mAP, full_map: full.mAP, avg_latency: full.latency,
    kept: decision === 'keep',
    summary: '(D4 finalize agent failed after retries; metrics recorded by orchestrator. Artifacts may be incomplete - check results/ and iteration_logs/ manually.)',
    lessons: fm5 ? 'FM5: mechanism never activated in vivo' : 'finalize agent failed; see orchestrator logs',
    report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json',
  }
  history.push({ loop: loop + 6, version: V, title: proposal.title, smoke_map: result.smoke_map, full_map: result.full_map, avg_latency: result.avg_latency, kept: result.kept, lessons: result.lessons })
  if (result.kept && result.full_map > bestFullMap) {
    bestFullMap = result.full_map
    bestVersion = V
    log('Loop ' + loop + ': NEW BEST ' + V + ' mAP=' + bestFullMap)
  } else {
    log('Loop ' + loop + ': ' + V + ' full=' + result.full_map + ' kept=' + result.kept + ' (best still ' + bestVersion + '=' + bestFullMap + ')')
  }
  if (bestFullMap >= TARGET) {
    log('TARGET REACHED at loop ' + loop + ': ' + bestVersion + ' mAP=' + bestFullMap + ' >= ' + TARGET)
    break
  }
}

phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for this VAD agent-team session (versions v' + START_VERSION + '+, continuation of the v97-v102 session). Call StructuredOutput EXACTLY ONCE if a schema is required; otherwise reply in plain text.',
'',
CONTEXT,
'',
'FULL SESSION HISTORY:\n' + hist(),
'',
'TASK:',
'1. Read ' + CODE + '/iteration_logs/session_v97_ledger.md and the new idea files in ' + CODE + '/iteration_logs/ideas/ from this session (versions v' + START_VERSION + '+).',
'2. Get today\'s date via: date +%Y%m%d',
'3. Write ' + CODE + '/iteration_logs/summary_loops_v' + START_VERSION + 'plus_<YYYYMMDD>.md in the SAME style as summary_loops_v90-94_20260712.md: executive summary (final best mAP, whether target 0.20 was reached, the FCCS in-vivo verdict), per-version what-was-tried table with results, updated failure-mode taxonomy (FM1-FM5 + any new modes), latency analysis, and concrete next directions.',
'4. Reply with a 10-line plain-text recap of the session outcome.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('Session complete. Best: ' + bestVersion + ' full mAP=' + bestFullMap + ' over ' + history.length + ' recorded loops.')
