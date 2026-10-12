export const meta = {
  name: 'vad-team-map20-v3',
  description: 'VAD agent team v3 (FM5+FM6-hardened): A/B research+debate, C skeptical review, D implement+eval with mandatory activation gates. Continues at v108 after the v103-v107 session (FCCS in-vivo negative 0.051, ECE sim-to-live gap confirmed, true baseline still v94 mAP=0.1494). Loop 1 seeded: complete the stalled v104 TSG full eval + pre-registered coverage adjudication. Target full-eval temporal mAP>=0.20, min 10 loops.',
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
const MAX_LOOPS = 10
const START_VERSION = 108
const BASE_LATENCY = 128
const SMOKE_GATE = 0.08

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision: video-level labels for training; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline, versioned docstring header; config.py centralizes ALL params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up before editing.',
'',
'TRUE BASELINE (full 65-video eval): temporal mAP = 0.1494 (v89/v94), AP@0.5 = 0.1335, AP@0.7 = 0.1096, frame AUC ~0.69-0.71, avg latency ~128 s/video at concurrency 3. (v107 headline 0.17291 was sanity-only, NOT a new baseline: its deterministic intra-run A/B delta was +0.002.)',
'TARGET: full-eval temporal mAP >= 0.20 while keeping per-video latency near or below ~128s.',
'VLM API stochasticity is about +/-0.03 mAP: deltas within noise are NOT real improvements.',
'',
'!!! FM5 LESSON (the dominant lesson of v97-v102) !!!',
'Six consecutive versions (v97-v102) were evaluated with the new mechanism NEVER ACTUALLY ACTIVATED: opt-in CLI flag implemented but never passed by the launcher, so live runs were byte-identical to baseline and every "result" was pure VLM noise. v97\'s apparent 0.1616 was flag-off noise. THEREFORE:',
'- Every mechanism MUST be activatable via a REAL CLI flag on batch_evaluate.py in the CURRENT tree (verify with: python3 batch_evaluate.py --help) OR be default-ON via config.py.',
'- Every mechanism MUST emit a unique INFO-level log marker per video when it fires (e.g. "[XYZ] ...").',
'- The orchestrator reports eval_flags + activation_marker; the launcher MUST include eval_flags; after each eval the log is greped for the marker. 0 markers = FM5 = automatic REVERT, no exceptions.',
'',
'!!! FM6 LESSON (new, confirmed by v107) — SIMULATION-TO-LIVE GAP !!!',
'v99 ECE showed +0.031 mAP in offline candidate-pool simulation (twice independently reproduced). v107 ran the identical mechanism live: deterministic intra-run A/B delta only +0.002, below the +0.005 revert threshold -> REVERTED. Offline simulations on cached candidates/systematically overestimate live gains. Any proposal whose primary evidence is a simulation must be discounted; only in-vivo full-eval deltas count.',
'',
'CONFIRMED IN-VIVO RESULTS FROM v103-v107 (do NOT re-try):',
'- v103 FCCS (full-coverage comparative VLM scan, --full_coverage_scan): GATE PASSED (177 markers full), full mAP 0.05109 vs 0.1494 baseline (-0.11, catastrophic). Comparative packing against in-video normal baseline DILUTES anomaly saliency. Genuine negative: full-coverage comparative scanning is HARMFUL; directed scanning is far better. Code remains flag-off in tree.',
'- v107 ECE-Live (event-continuation boundary extension, whole-video half-max level-set growth): GATE PASSED, live A/B delta +0.002 -> REVERTED (see FM6). v51 half-max local-window boundaries are near-optimal for this signal; boundary extension does not help live.',
'- v108 TSG (temporal-structure-guided scan targeting for flat-CLIP fallback videos): COMPLETED in loop 1 of this session (gate passed, 69/65 markers, fired only 4/65). Full mAP 0.1583 (within the 0.1494+/-0.03 noise band, NO gain); per-video mean latency 295s vs 155s same-config (broad-based, likely API slowness, not TSG-caused); pre-registered coverage adjudication: selected-vs-baseline GT-overlap NOT improved -> REVERTED per protocol and ALL TSG CODE REMOVED from the tree (pipeline.py/config.py/batch_evaluate.py/run.py; backup at .qoder/backups/v108-revert/). Lesson: a mechanism touching only ~6% of videos cannot move full-set mAP - proposals MUST estimate their target-video fraction first. Do NOT repropose TSG/structure-saliency scan retargeting.',
'',
'MANDATORY READING before proposing or reviewing (skim smartly, newest first):',
'1. ' + CODE + '/iteration_logs/session_v97_ledger.md  (v97-v107 outcomes)',
'2. ' + CODE + '/iteration_logs/ideas/ - ALL files, newest first (v103-v107 files are the most relevant; then loop*_v8x/v9x)',
'3. ' + CODE + '/iteration_logs/summary_loops_v90-94_20260712.md  (failure-mode taxonomy FM1-FM4)',
'4. ' + CODE + '/iteration_logs/summary_loops1-10_v86-89_20260712.md',
'5. ' + ROOT + '/.qoder/specs/VAD_mAP_50%_优化方案_c990fbf7.md  (long-term roadmap + rejected alternatives)',
'6. Research methodology: ' + ROOT + '/academic-research-skills/deep-research/SKILL.md (+ references/ dir) - then run REAL WebSearch/WebFetch queries (arXiv, PapersWithCode, CVF openaccess).',
'',
'KNOWN FAILURE MODES:',
'FM1 curve-shape modification (fusion weights, feature re-centering, confusion-negative scoring, candidate pooling) regresses even when the modifying signal is individually better.',
'FM2 detection-count changes under greedy-match evaluation (e.g. Soft-NMS) turn near-duplicates into FPs. NOTE: wholesale scan-mechanism replacement was argued to escape FM2 but FCCS (v103) proved comparative full-coverage actively harmful for a different reason (saliency dilution).',
'FM3 VLM self-consistency via temperature diversity hurts perceptual/extraction tasks (reading timestamps).',
'FM4 self-consistency on segment-inclusion decisions is dangerous; on already-fixed detection labels (category tie-break) it is safe.',
'FM5 mechanism implemented but never activated in the eval run. Guard: mandatory eval_flags + activation marker + log grep gate.',
'FM6 simulation-to-live gap: offline candidate-pool simulation gains (even +0.03) do not transfer to live runs. Discount simulation evidence; trust only in-vivo full evals.',
'Also known: MIL v3/v4 LGT-Adapter collapsed (75-83% of videos); MIL v6 rejected (frame-level localization near chance, archived v80-85b); mAP ~0.14-0.16 is a STRUCTURAL ceiling for incremental tuning - only ARCHITECTURE-LEVEL changes can break it.',
'',
'VALIDATED SAFE/positive PATTERNS: confidence-only modulation computed read-only from existing curves (v95 TSM sharpness); category tie-break on fixed detections (v89); learned prompt-ensemble weights strictly upstream of fusion/boundary/candidate logic (v94).',
'',
'CURRENT CODE STATE (working tree): v94 baseline + inert flag-off mechanisms: v97 FCCS (--full_coverage_scan, PROVEN HARMFUL in vivo), v100 EUBC (inert), v105 ECE-Live (config scoring.event_continuation_extension, PROVEN +0.002 in vivo). All default OFF = byte-identical v94 behavior. v104 TSG was REVERTED and PHYSICALLY REMOVED from the tree after the v108 in-vivo verdict - do not assume it exists.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop (controlled experiment).',
'- ARCHITECTURE-LEVEL ideas only - NO pure parameter/threshold tuning.',
'- All new params via config.py with a switch restoring prior behavior; every mechanism CLI-activatable + INFO log marker (FM5 guard).',
'- Keep the v51 half-max boundary-crossing mechanism unless a strong literature-backed argument says otherwise (v107 confirmed it live).',
'- Latency matters: prefer zero-cost (precomputed text embeddings) or training-free signals; avoid extra per-video VLM calls.',
'- When calling StructuredOutput: call it EXACTLY ONCE, at the very end, with the final answer. NEVER call it twice.',
'- ANTI-STALL (runtime kills agents silent >180s): no single Bash call may exceed ~150s (never sleep longer); Read with offset/limit; emit a short progress line between tool calls.',
].join('\n')

const history = [
  { loop: 1, version: 'v97', title: 'FCCS full-coverage comparative scan', smoke_map: 0.15035, full_map: 0.16155, avg_latency: 141.47, kept: true, lessons: 'FM5 EXPOSED: --full_coverage_scan never passed; 0.1616 is flag-off NOISE' },
  { loop: 2, version: 'v98', title: 'Verified FCCS baseline attempt', smoke_map: 0.15035, full_map: 0.14504, avg_latency: 147.9, kept: false, lessons: 'FM5: flag omitted again' },
  { loop: 3, version: 'v99', title: 'ECE event-continuation extension', smoke_map: 0.14666, full_map: 0.15814, avg_latency: 148.18, kept: false, lessons: 'FM5: never activated; sim +0.0311 untested in vivo' },
  { loop: 4, version: 'v100', title: 'EUBC event-unit boundary contraction', smoke_map: 0.15035, full_map: 0.15713, avg_latency: 133.27, kept: true, lessons: 'FM5: kept code inert (flag off, byte-identical)' },
  { loop: 5, version: 'v101', title: 'BRP bottom-ranked recovery proposals', smoke_map: 0.15035, full_map: 0.13949, avg_latency: 145.51, kept: false, lessons: 'FM5: never activated' },
  { loop: 6, version: 'v102', title: 'HZG-FC hierarchical zoom grounding', smoke_map: 0.1637, full_map: 0.15861, avg_latency: 150.57, kept: false, lessons: 'FM5: never activated' },
  { loop: 7, version: 'v103', title: 'FCCS live activation test', smoke_map: 0.17821, full_map: 0.05109, avg_latency: 145.54, kept: false, lessons: 'GATE PASSED; genuine NEGATIVE: comparative full-coverage dilutes saliency, -0.11 mAP; FM5 refuted for FCCS' },
  { loop: 8, version: 'v108', title: 'TSG live completion + coverage adjudication', smoke_map: 0.1443, full_map: 0.15834, avg_latency: 295.0, kept: false, lessons: 'GATE PASSED (fired 4/65); mAP within noise (no gain); latency ~1.9x (API drift, not TSG); GT-overlap coverage NOT improved -> REVERTED + code REMOVED; lesson: mechanisms touching ~6% of videos cannot move full-set mAP' },
  { loop: 9, version: 'v105/v107', title: 'ECE-Live deterministic in-vivo A/B', smoke_map: 0.14744, full_map: 0.17291, avg_latency: 166.42, kept: false, lessons: 'GATE PASSED; live A/B delta +0.002 < +0.005 threshold -> REVERT; FM6 sim-to-live gap confirmed (sim +0.031 did not transfer)' },
]
let bestFullMap = 0.1494
let bestVersion = 'v94'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": smoke=' + h.smoke_map + ' full=' + h.full_map + ' latency=' + h.avg_latency + 's kept=' + h.kept + ' | lessons: ' + h.lessons)
  return 'TRUE current best: ' + bestVersion + ' full mAP=' + bestFullMap + '. Session history:\n' + lines.join('\n')
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
    risks: { type: 'string', description: 'which known failure modes FM1-FM6 it might trigger and why it avoids them' },
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
const adjSchema = {
  type: 'object',
  required: ['coverage_improved', 'coverage_detail', 'note'],
  properties: {
    coverage_improved: { type: 'boolean', description: 'true iff TSG-selected segment sets cover GT anomaly segments strictly better than baseline pre-screen sets (per pre-registered adjudication)' },
    coverage_detail: { type: 'string', description: 'per-video aggregate numbers: GT-overlap of selected vs baseline index sets' },
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
'SESSION HISTORY (do NOT repeat already-tried ideas; note how many died from FM5 non-activation, and that FCCS + ECE are now CONFIRMED in-vivo negatives):\n' + hist(),
'',
'TASK:',
'1. Do the MANDATORY READING (session ledger + newest idea files are the highest value; then skim pipeline.py docstring + config.py + the 2-3 most relevant source files for your angle).',
'2. SEARCH-STORM: run at least 6 real WebSearch queries on 2023-2026 papers about TEMPORAL LOCALIZATION mechanisms (how exactly do they localize: proposal generation, boundary regression, start/end probability curves, Gaussian modeling, MIL snippet scoring, temporal contrast, motion/optical-flow signals, training-free localization). Use WebFetch on the most promising 2-3 papers (arXiv abs/html pages) to extract their exact localization mechanism.',
'3. From the literature + codebase evidence, derive ONE architecture-level idea that could break the 0.149 structural ceiling. It must respect the HARD CONSTRAINTS and avoid FM1-FM6. Parameter tuning is FORBIDDEN. The idea MUST be CLI-activatable and emit a unique INFO log marker (FM5 guard) - state both in the implementation_plan. Remember FM6: if your only evidence is an offline simulation, say so and discount it.',
'4. Return your proposal via StructuredOutput (call it EXACTLY ONCE). Be concrete: name the exact files/functions to change.',
'',
'!!! ANTI-STALL RULES (CRITICAL - the runtime KILLS any agent silent for >180s, and every previous research agent in this session died this way):',
'- Keep EVERY single tool call fast: prefer WebSearch over WebFetch; WebFetch at most 2 arXiv ABS pages (small), never big HTML/full-text pages; Read with offset/limit, never whole large files; NEVER run sleep or any Bash command longer than ~60s.',
'- Emit one short progress line (a few words) before and after every tool call so you never go silent.',
'- Budget yourself: if a search or fetch fails or is slow once, move on - do not retry it. Aim to finish the whole task in <15 tool calls.',
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
"2. Else check the process: pgrep -f 'batch_evaluate.*batch_eval_" + V + "_" + tag + "' -> if alive: Bash 'sleep 170', then re-check the report; if still absent, Bash 'sleep 170' once more and re-check; if it appeared, extract metrics + marker count (done=true); otherwise return done=false, crashed=false, marker_count=-1, note='still running'. ANTI-STALL: the runtime kills any agent silent for >180s - no single Bash call may exceed 170s.",
'3. If the process is dead and no report exists: tail -30 the log. If the failure looks TRANSIENT (API timeout / ConnectTimeout / rate limit / network), relaunch with the exact nohup command (max_per_category ' + maxPerCat + ', WITH the required flags: ' + (evalFlags || '(none)') + ') and return done=false, crashed=false, marker_count=-1, note="relaunched after: <cause>". If it looks like a CODE BUG (Python Traceback referencing project code), return done=false, crashed=true, marker_count=-1, note=<error summary>. NOTE: the TSG mechanism has a built-in [TSG-GATE] that exits with code 2 and refuses to write the report if markers<succeeded or fired==0 - if the log shows [TSG-GATE] failure, that is a REAL result (mechanism did not fire), return crashed=true with that note.',
'Return via StructuredOutput (EXACTLY ONCE): done, crashed, mAP, ap30, ap50, ap70, latency (use -1 when unknown), marker_count, note.',
  ].join('\n')
}

async function runEval(V, tag, maxPerCat, evalFlags, marker) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: 'Implement' }))
  if (!launch || !launch.launched) {
    return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'agent null after retries') }
  }
  for (let w = 0; w < 40; w++) {
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

function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, decisionText, fm5) {
  return [
'You are Agent D4, the finalizer for version ' + V + '. Implementation and evals are DONE. Enforce the keep/revert decision, save artifacts, write the report, return the summary. Keep replies focused; every Bash call < 9 minutes. Call StructuredOutput EXACTLY ONCE at the end.',
'',
'Paths: code=' + CODE + ' ; results=' + ROOT + '/results ; backups=' + ROOT + '/.qoder/backups/' + V + ' ; ideas=' + CODE + '/iteration_logs/ideas/ ; runlogs=' + CODE + '/iteration_logs/results/',
'',
'DECISION FROM ORCHESTRATOR: ' + decisionText + (fm5 ? '  [FM5: mechanism never activated in the live run - 0 marker lines]' : ''),
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
'1. If decision=REVERT: restore every file present in ' + ROOT + '/.qoder/backups/' + V + '/ back into ' + CODE + '/ and verify each restore with diff (must be identical). If there is no backup dir for ' + V + ' (e.g. verify-only loop where the mechanism already ships flag-off by default), ensure the mechanism config stays default-OFF (byte-identical baseline behavior) and do nothing else. If decision=KEEP: do not touch the code. If decision=KEEP-FLAG-ON: flip the mechanism config default to True in config.py (this becomes the new baseline) and sanity-import pipeline.',
'2. Copy artifacts that exist: cp ' + ROOT + '/results/batch_eval_' + V + '_smoke.log ' + CODE + '/iteration_logs/results/' + V + '_smoke_run.log ; cp ' + ROOT + '/results/batch_eval_' + V + '_smoke/evaluation_report.json ' + CODE + '/iteration_logs/results/' + V + '_smoke_eval.json ; same two copies for _full (if the full eval ran). For verify-only loops that reused an earlier version\'s smoke, copy that smoke log/report too and note the source.',
'3. date +%Y%m%d, then write ' + CODE + '/iteration_logs/ideas/loop' + loop + '_' + V + '_<kebab-case-slug>_<YYYYMMDD>.md with: the approved proposal, C\'s feedback, implementation notes, the eval flags + activation marker, marker counts in smoke/full logs, a results table (smoke mAP; full mAP, AP@0.3/0.5/0.7, latency), the keep/revert decision + reason, and lessons learned (which failure modes confirmed/refuted).',
'4. Append one row to ' + CODE + '/iteration_logs/session_v97_ledger.md (create the file with a markdown table header if it is missing): | loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept |',
'5. Return via StructuredOutput (EXACTLY ONCE): version="' + V + '", implemented=true, smoke_map=' + smoke.mAP + ', full_map=' + full.mAP + ', avg_latency=' + full.latency + ', kept=' + (decisionText.indexOf('KEEP') === 0) + ', summary (5-8 sentences), lessons (2-4 sentences), report_path.',
  ].join('\n')
}

function tsgAdjudicatePrompt(full) {
  return [
'You are the TSG adjudicator for v108. The v104 TSG mechanism (temporal-structure-guided scan targeting for flat-CLIP fallback videos, --temporal_structure_scan) has a PRE-REGISTERED adjudication protocol that overrides the headline mAP. Execute it and return StructuredOutput EXACTLY ONCE.',
'',
'Full eval log: ' + ROOT + '/results/batch_eval_v108_full.log ; report: ' + ROOT + '/results/batch_eval_v108_full/evaluation_report.json (full mAP=' + full.mAP + ', AP@0.3=' + full.ap30 + ' AP@0.5=' + full.ap50 + ' AP@0.7=' + full.ap70 + ').',
'',
'PROTOCOL (from the pipeline.py v104 header, pre-registered, no post-hoc wiggle room):',
'1. From the full log, extract every "[TSG] summary: fired selected=[...] baseline=[...] peaks=N" line, keyed by the video being processed at that time (the preceding "[i/65] START <video>" line).',
'2. For each fired video, compute GT temporal-overlap coverage of the selected 6s-segment index set vs the baseline pre-screen index set (segment i covers [6i, 6i+6) seconds; fps/timestamps consistent with anomaly_discovery segment construction). GT temporal annotations: find how evaluate.py / batch_evaluate.py loads UCF-Crime temporal GT (annotation files under /Users/zzl/Movies/ucf-crime or the dataset dir) and reuse that mapping. Write a small throwaway analysis script under ' + CODE + '/iteration_logs/ (name it analysis_v108_tsg_coverage.py) and run it.',
'3. coverage_improved = true ONLY if the selected sets cover GT anomaly segments strictly better (higher total GT seconds covered, or more GT segments hit) than the baseline sets across the fired videos. Ties or worse = false.',
'4. Also record how many videos fired vs not-applicable, and quote the [TSG-GATE] line.',
'Return via StructuredOutput (EXACTLY ONCE): coverage_improved (bool), coverage_detail (aggregate numbers), note.',
  ].join('\n')
}

// Loop-1 seeded experiment: TSG live completion + pre-registered adjudication (v104 mechanism, smoke-passed, full stalled at 5/65).
const SEED_PROPOSAL = {
  title: 'TSG Live Completion Test (v104 mechanism, first full in-vivo evaluation + pre-registered coverage adjudication)',
  mechanism: 'Temporal-Structure-Guided scan targeting (already implemented in the tree behind --temporal_structure_scan): replaces ONLY the flat-CLIP fallback branch segment-selection ranking with EventVAD-style temporal-gradient+motion peak picking; exact VLM-call-count parity with incumbent pre-screen; judgment/merge/boundary/fusion/NMS byte-identical; [TSG-GATE] hard gate in batch_evaluate.py.',
  rationale: '28.8% of wrong-location candidates concentrate in flat-CLIP categories (Shoplifting/Stealing/Arson/Burglary, ~20/65 videos take the Stage 4 fallback); incumbent generic CLIP pre-screen std is only 0.001-0.005 there, so scanned segments are near-random. Smoke passed (mAP 0.1443, 14/14 markers, fired 1/14). Full eval stalled at 5/65 in the previous session - this loop completes it and runs the pre-registered adjudication.',
  implementation_plan: 'Verify-only: confirm --temporal_structure_scan in batch_evaluate.py --help, [TSG] markers in anomaly_discovery, then run the full 65-video eval with the flag ON.',
  expected_gain: 'Pre-registered: coverage not improved -> REVERT regardless of headline; coverage improved + mAP within 0.1494+/-0.03 -> keep code flag-OFF (no new baseline); coverage improved + mAP > 0.1794 -> KEEP-FLAG-ON (new baseline).',
  risks: 'FM2 bounded by exact call-count parity + unchanged downstream; FM1/FM3/FM4 absent by construction; FM5 eliminated by [TSG-GATE] hard gate; FM6 n/a (no simulation evidence used).',
  latency_impact: 'Zero extra CLIP/VLM/MIL calls (<20ms numpy); latency-neutral.',
  literature: 'EventVAD (arXiv 2504.13092); LAVAD (CVPR 2024); UBoCo (CVPR 2022)/GEBD; T3AL (CVPR 2024); LEC-VAD (ICML 2025).',
}
const SEED_VERDICT = { approve: true, feedback: 'Pre-approved as v104 (C verdict with 4 mandated items, previous session). This loop only completes the stalled full eval and runs the pre-registered adjudication.', mandated_changes: '' }
const SEED_IMPL = { implemented: true, files_changed: '(none - verify only)', notes: 'v104 TSG already implemented and verified present in the tree; smoke already passed (0.1443, 14/14 markers, fired 1/14); evaluating full with --temporal_structure_scan', eval_flags: '--temporal_structure_scan', activation_marker: '[TSG]' }

log('VAD agent team v3 starting. Target mAP>=' + TARGET + ', versions v' + START_VERSION + '+, max ' + MAX_LOOPS + ' loops. FM5+FM6-hardened. Loop 1 seeded: TSG live completion + adjudication.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION - 1 + loop)
  log('=== Loop ' + loop + ' (' + V + ') ===')

  let proposal, verdict, impl
  if (loop === 1) {
    proposal = SEED_PROPOSAL
    verdict = SEED_VERDICT
    impl = SEED_IMPL
    log('Loop 1: seeded TSG live-completion test (mechanism pre-approved as v104; skipping research/review/implement)')
  } else {
    phase('Research')
    const [propA, propB] = await parallel([
      () => retryAgent(() => agent(researchPrompt('A', 'Your specialty: TEMPORAL LOCALIZATION ARCHITECTURES from the TAL/VAD literature (BSN/BMN/DBG/ActionFormer-style proposal+boundary mechanisms, VadCLIP/PE-MIL/MGFN/AnomalyCLIP/GS-MoE/FreeZAD-style VAD mechanisms) and how their localization core can be adapted training-free or light-training into this pipeline.'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
      () => retryAgent(() => agent(researchPrompt('B', 'Your specialty: CODEBASE-FIT + BOTTLENECK analysis. Known bottlenecks: 28.8% of candidates at completely wrong locations (low-CLIP categories Shoplifting/Stealing/Arson/Burglary have flat CLIP curves), 39% of detections have IoU 0.1-0.5 (boundary imprecision - but v107 proved boundary EXTENSION does not help live), MIL temporal discrimination collapsed. You focus on signals the pipeline underuses (motion energy, temporal gradient structure, MIL v6, snippet resolution, multi-signal boundary evidence) and on recent weakly-supervised VAD SOTA papers for transferable mechanisms.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
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
      () => retryAgent(() => agent('You are Agent A continuing the debate. ' + debateBase + '\n\nTASK: Critique B\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM6 or repeat a tried/failed version? (b) is it truly architecture-level? (c) is the expected gain plausible given +/-0.03 noise and the FM6 sim-to-live gap? (d) latency risk? Then state: would you accept B\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'A-critiques-B-' + V, phase: 'Debate' })),
      () => retryAgent(() => agent('You are Agent B continuing the debate. ' + debateBase + '\n\nTASK: Critique A\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM6 or repeat a tried/failed version? (b) is it truly architecture-level? (c) is the expected gain plausible given +/-0.03 noise and the FM6 sim-to-live gap? (d) latency risk? Then state: would you accept A\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'B-critiques-A-' + V, phase: 'Debate' })),
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
'TASK: Synthesize ONE final proposal: pick the stronger idea, or merge both into one coherent SINGLE controlled change (never two independent changes). It must be architecture-level, respect all HARD CONSTRAINTS, avoid FM1-FM6, and not repeat anything in SESSION HISTORY. It MUST be CLI-activatable with a stated INFO log marker (FM5 guard). Return via StructuredOutput (EXACTLY ONCE, same schema as the proposals).',
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
'2. Avoids FM1-FM6 and all previously failed/reverted approaches in the history and idea logs?',
'3. Expected gain plausible vs +/-0.03 VLM noise? Is the mechanism causal, not wishful? Is the evidence in-vivo or merely simulation (FM6 - discount simulation)?',
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
        history.push({ loop: loop + 9, version: V, title: proposal.title + ' [REJECTED - no implementation]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
        continue
      }
    }

    phase('Implement')
    impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
    if (!impl || !impl.implemented) {
      log('Loop ' + loop + ': D1 implementation failed')
      history.push({ loop: loop + 9, version: V, title: proposal.title + ' [D1 FAILED]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'implementation failed: ' + (impl && impl.notes ? impl.notes.slice(0, 200) : 'agent null after retries') })
      continue
    }
    if (!impl.activation_marker) {
      log('Loop ' + loop + ': D1 reported no activation marker - FM5 risk, treating as failure')
      history.push({ loop: loop + 9, version: V, title: proposal.title + ' [NO MARKER - FM5]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'D1 did not provide an activation marker; loop aborted before evals' })
      continue
    }
  }

  // Loop 1 reuses the already-passed v104 TSG smoke (0.1443, 14/14 markers) and goes straight to the full eval.
  let smoke
  if (loop === 1) {
    smoke = { done: true, crashed: false, mAP: 0.1443, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: 14, note: 'reused v104 smoke (flag-on, TSG-GATE ok, fired 1/14)' }
  } else {
    smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker)
  }
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count + ' | ' + (smoke.note || '').slice(0, 150))

  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (gate)' }
  let decision = 'revert'
  let decisionText = 'REVERT'
  let fm5 = false
  if (smoke.done && !smoke.crashed && smoke.mAP >= SMOKE_GATE) {
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
      if (!fm5 && full.done && !full.crashed) {
        if (loop === 1) {
          // TSG pre-registered adjudication overrides the headline mAP.
          const adj = await retryAgent(() => agent(tsgAdjudicatePrompt(full), { schema: adjSchema, label: 'TSG-adjudicate-' + V, phase: 'Implement' }))
          const covUp = adj ? adj.coverage_improved : false
          log('Loop ' + loop + ': TSG adjudication coverage_improved=' + covUp + ' | ' + (adj ? (adj.coverage_detail || '').slice(0, 200) : 'agent null'))
          if (!adj || !covUp) {
            decision = 'revert'
            decisionText = 'REVERT (pre-registered TSG adjudication: GT-overlap coverage NOT improved by selected vs baseline segment sets - revert regardless of headline mAP; mechanism already flag-off, keep code inert, no baseline change). Adjudication detail: ' + (adj ? adj.coverage_detail : 'adjudication agent failed - defaulting to revert per pre-registration (no evidence of coverage gain)')
          } else if (full.mAP > 0.1794) {
            decision = 'keep-flag-on'
            decisionText = 'KEEP-FLAG-ON (pre-registered TSG adjudication: coverage improved AND full mAP ' + full.mAP + ' > 0.1794 = above the 0.1494+0.03 noise band - flip config default to ON, this becomes the new baseline). Coverage: ' + adj.coverage_detail
          } else {
            decision = 'keep-flag-off'
            decisionText = 'KEEP (pre-registered TSG adjudication: coverage improved but full mAP ' + full.mAP + ' within the 0.1494+/-0.03 noise band - keep code flag-OFF, do NOT claim a new baseline). Coverage: ' + adj.coverage_detail
          }
        } else if (full.mAP >= bestFullMap - 0.005 && (full.latency <= 0 || full.latency <= Math.round(BASE_LATENCY * 1.15))) {
          decision = 'keep'
          decisionText = 'KEEP'
        }
      }
    }
  } else {
    log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + '), skipping full eval, will revert')
  }

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, decisionText, fm5), { schema: resultSchema, label: 'D4-finalize-' + V, phase: 'Implement' }))
  const result = fin || {
    version: V, implemented: true,
    smoke_map: smoke.mAP, full_map: full.mAP, avg_latency: full.latency,
    kept: decision.indexOf('keep') === 0,
    summary: '(D4 finalize agent failed after retries; metrics recorded by orchestrator. Artifacts may be incomplete - check results/ and iteration_logs/ manually.)',
    lessons: fm5 ? 'FM5: mechanism never activated in vivo' : 'finalize agent failed; see orchestrator logs',
    report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json',
  }
  history.push({ loop: loop + 9, version: V, title: proposal.title, smoke_map: result.smoke_map, full_map: result.full_map, avg_latency: result.avg_latency, kept: result.kept, lessons: result.lessons })
  if (decision === 'keep-flag-on' && result.full_map > bestFullMap) {
    bestFullMap = result.full_map
    bestVersion = V
    log('Loop ' + loop + ': NEW BEST ' + V + ' mAP=' + bestFullMap + ' (flag now default-ON)')
  } else if (result.kept && decision === 'keep' && result.full_map > bestFullMap) {
    bestFullMap = result.full_map
    bestVersion = V
    log('Loop ' + loop + ': NEW BEST ' + V + ' mAP=' + bestFullMap)
  } else {
    log('Loop ' + loop + ': ' + V + ' full=' + result.full_map + ' decision=' + decision + ' (best still ' + bestVersion + '=' + bestFullMap + ')')
  }
  if (bestFullMap >= TARGET) {
    log('TARGET REACHED at loop ' + loop + ': ' + bestVersion + ' mAP=' + bestFullMap + ' >= ' + TARGET)
    break
  }
}

phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for this VAD agent-team session (versions v' + START_VERSION + '+, continuation of the v97-v107 sessions). Call StructuredOutput EXACTLY ONCE if a schema is required; otherwise reply in plain text.',
'',
CONTEXT,
'',
'FULL SESSION HISTORY:\n' + hist(),
'',
'TASK:',
'1. Read ' + CODE + '/iteration_logs/session_v97_ledger.md and the new idea files in ' + CODE + '/iteration_logs/ideas/ from this session (versions v' + START_VERSION + '+).',
'2. Get today\'s date via: date +%Y%m%d',
'3. Write ' + CODE + '/iteration_logs/summary_loops_v' + START_VERSION + 'plus_<YYYYMMDD>.md in the SAME style as summary_loops_v90-94_20260712.md: executive summary (final best mAP, whether target 0.20 was reached, the TSG in-vivo adjudication verdict), per-version what-was-tried table with results, updated failure-mode taxonomy (FM1-FM6 + any new modes), latency analysis, and concrete next directions.',
'4. Reply with a 10-line plain-text recap of the session outcome.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('Session complete. Best: ' + bestVersion + ' full mAP=' + bestFullMap + ' over ' + history.length + ' recorded loops.')
