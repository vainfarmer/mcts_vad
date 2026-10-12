export const meta = {
  name: 'vad-team-map20-v4',
  description: 'VAD agent team v4 (FM5+FM6+FM7+FM8-hardened): A/B research+debate, C skeptical review, D implement+eval with mandatory activation gates. Continues at v114 after the v108-v113 session (TSG reverted; MIL video-level prior -0.004/FM7; VideoMAE window saturated; Q-score v111 offline +0.0087 KEPT flag-off; duration-fit non-discriminative; v113 FCCS+Q-score confounded catastrophe 0.052). True in-vivo baseline still v94 mAP=0.1494. Loop 1 seeded: clean IN-VIVO Q-score verdict via a deterministic on/off A/B on FRESH detections (resolves the v113 confound + FM6). Target full-eval temporal mAP>=0.20, min 10 loops.',
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
const START_VERSION = 114
const BASE_LATENCY = 128
const SMOKE_GATE = 0.08

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision: video-level labels for training; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline, versioned docstring header; config.py centralizes ALL params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up before editing.',
'',
'TRUE BASELINE (full 65-video eval, IN VIVO): temporal mAP = 0.1494 (v89/v94), AP@0.5 = 0.1335, AP@0.7 = 0.1096, frame AUC ~0.69-0.71, avg latency ~128 s/video at concurrency 3. NOTE: v111 Q-score reported 0.16703 but that is an OFFLINE re-score of v108 CACHED detections, NOT a live baseline - loop 1 tests whether it transfers to fresh detections.',
'TARGET: full-eval temporal mAP >= 0.20 while keeping per-video latency near or below ~128s.',
'VLM API stochasticity is about +/-0.03 mAP: deltas within noise are NOT real improvements. This is why the PREFERRED validation for pure re-rank changes is a DETERMINISTIC offline A/B on a prior run\'s REAL detections (noise-free), not a two-run headline comparison.',
'',
'!!! FM5 LESSON (v97-v102) !!! Six consecutive versions were evaluated with the new mechanism NEVER ACTUALLY ACTIVATED: the opt-in CLI flag was implemented but never passed by the launcher, so live runs were byte-identical to baseline and every "result" was pure VLM noise. THEREFORE:',
'- Every mechanism MUST be activatable via a REAL CLI flag on batch_evaluate.py in the CURRENT tree (verify with: python3 batch_evaluate.py --help) OR be default-ON via config.py.',
'- Every mechanism MUST emit a unique INFO-level log marker per video when it fires (e.g. "[XYZ] ...").',
'- The orchestrator reports eval_flags + activation_marker; the launcher MUST include eval_flags; after each eval the log is greped for the marker. 0 markers = FM5 = automatic REVERT. (Exception: a pure post-hoc re-rank verified by a DETERMINISTIC offline A/B on the run\'s real detections does not need the live flag to fire, because the offline re-score IS the in-vivo measurement - see loop 1.)',
'',
'!!! FM6 LESSON (confirmed by v107) - SIMULATION-TO-LIVE GAP !!! v99 ECE showed +0.031 mAP in offline candidate-pool SIMULATION; v107 ran it live and the deterministic A/B delta was only +0.002 -> REVERTED. Offline simulations on cached candidates systematically overestimate live gains. Discount any proposal whose primary evidence is a simulation; only in-vivo deltas count. IMPORTANT DISTINCTION: a deterministic re-score of a run\'s OWN real final detections (re-ranking that cannot change boundaries/count) is NOT a "simulation" - it is faithful in-vivo, because mAP depends only on the global confidence ordering. That protocol is trusted; candidate-pool simulations are not.',
'',
'!!! FM7 LESSON (new, v109) - VIDEO-LEVEL PRIORS DO NOT FIX INTRA-VIDEO MISLOCATION !!! A cross-video/video-level confidence prior (v109 MIL top-k prior, conf*(0.7+0.3*prior)) REGRESSED -0.004 offline: every wrong-location detection INSIDE an anomaly video shares the same high video-level prior, so it cannot demote them; meanwhile it hurts the few true anomalies with low MIL. The dominant error is intra-video mislocalization, so a re-rank signal MUST be PER-DETECTION (score WHERE each detection sits), not per-video.',
'',
'!!! FM8 LESSON (new, v113) - OFFLINE PRECHECK IS CONFIG-SPECIFIC !!! An offline precheck validated on one pipeline configuration\'s detections does NOT generalize to a different scan mode\'s detections. Worse, v113 turned on TWO flags at once (--full_coverage_scan AND --qscore_rerank) - a CONFOUNDED experiment: FCCS alone is a PROVEN -0.11 catastrophe (v103), so v113\'s 0.052 says nothing about Q-score. NEVER combine a proven-harmful mechanism with the mechanism under test; change ONE thing.',
'',
'CONFIRMED IN-VIVO / OFFLINE RESULTS (do NOT re-try):',
'- v103 FCCS (--full_coverage_scan): GATE PASSED live, full mAP 0.051 (-0.11, catastrophic). Comparative full-coverage packing DILUTES anomaly saliency. FCCS is PROVEN HARMFUL - do not enable it, do not combine it with anything. Code stays flag-off in tree.',
'- v107 ECE-Live (boundary extension): GATE PASSED, live A/B delta +0.002 -> REVERTED. v51 half-max boundaries are near-optimal live.',
'- v108 TSG (structure-guided scan retargeting): REVERTED; fired 4/65, coverage NOT improved; code REMOVED from tree. Lesson: a mechanism touching ~6% of videos cannot move full-set mAP. Do NOT repropose TSG/structure-saliency scan retargeting.',
'- v109 MIL cross-video prior: REJECTED at offline gate (-0.004). See FM7. Do NOT repropose video-level/cross-video confidence priors.',
'- v110 TVR VideoMAE per-detection re-rank: REJECTED at offline kill-gate. The UCF-finetuned VideoMAE sliding-window score SATURATES to ~1.0 on every window of every test video (incl. normals) - zero within-video variance, no discriminative content. Do NOT build any signal on the raw VideoMAE window score.',
'- v111 Q-score localization-quality re-rank (--qscore_rerank, ScoringConfig.qscore_rerank): OFFLINE +0.0087 on v108 cached detections, broad-based (AP@0.3/0.5/0.7 all rise), robust across 6 configs; KEPT flag-OFF in tree. NEVER cleanly tested live (v113 confounded it with FCCS). Loop 1 of THIS session resolves that. The winning signal = fused-curve GEOMETRY (boundary sharpness + peak contrast + center fit), within-video min-max normalized. Do NOT repropose the same 3-signal Q-score; NEW per-detection signals must be ORTHOGONAL to it.',
'- v112 duration-fit stacking onto Q-score: REJECTED offline (neutral or -0.01). CATEGORY_DURATION_PRIORS are ALREADY enforced as boundary constraints, so dur_fit ~= 1.0 for every final detection -> zero variance. Lesson: a re-rank signal already baked into the pipeline carries no fresh content. Stack only signals ORTHOGONAL to what the pipeline already uses.',
'',
'MANDATORY READING before proposing or reviewing (skim smartly, newest first):',
'1. ' + CODE + '/iteration_logs/session_v97_ledger.md  (v97-v108 outcomes)',
'2. ' + CODE + '/iteration_logs/ideas/ - newest first: loop6_v113 / loop4_v112 / loop3_v111 / loop2_v110 / loop1_v109 are the most relevant; then v103-v108; then loop*_v8x/v9x',
'3. ' + CODE + '/iteration_logs/summary_loops_v90-94_20260712.md  (failure-mode taxonomy FM1-FM4)',
'4. ' + CODE + '/iteration_logs/summary_loops1-10_v86-89_20260712.md',
'5. ' + ROOT + '/.qoder/specs/VAD_mAP_50%_优化方案_c990fbf7.md  (long-term roadmap + rejected alternatives)',
'6. Research methodology: ' + ROOT + '/academic-research-skills/deep-research/SKILL.md (+ references/ dir) - then run REAL WebSearch/WebFetch queries (arXiv, PapersWithCode, CVF openaccess).',
'',
'KNOWN FAILURE MODES:',
'FM1 curve-shape modification (fusion weights, feature re-centering, confusion-negative scoring, candidate pooling) regresses even when the modifying signal is individually better.',
'FM2 detection-count changes under greedy-match evaluation (e.g. Soft-NMS) turn near-duplicates into FPs.',
'FM3 VLM self-consistency via temperature diversity hurts perceptual/extraction tasks (reading timestamps).',
'FM4 self-consistency on segment-inclusion decisions is dangerous; on already-fixed detection labels (category tie-break) it is safe.',
'FM5 mechanism implemented but never activated in the eval run. Guard: eval_flags + activation marker + log grep gate.',
'FM6 simulation-to-live gap: offline candidate-pool simulation gains do not transfer. Trust only in-vivo deltas and deterministic re-scores of real final detections.',
'FM7 video-level/cross-video confidence priors cannot fix the dominant intra-video mislocation error. Re-rank signals must be PER-DETECTION.',
'FM8 offline precheck is config-specific and does not transfer across scan modes; never confound the mechanism under test with a proven-harmful flag. One controlled change per loop.',
'Also known: MIL v3/v4 LGT-Adapter collapsed; MIL v6 rejected (frame-level localization near chance); mAP ~0.14-0.16 is a STRUCTURAL ceiling for incremental tuning - only ARCHITECTURE-LEVEL changes can break it.',
'',
'VALIDATED SAFE/positive PATTERNS: PURE POST-HOC CONFIDENCE RE-RANK of final detections computed read-only from existing curves, validated by a DETERMINISTIC offline A/B on a prior run\'s real detections (this is the most productive family right now - v95 TSM, v111 Q-score). Category tie-break on fixed detections (v89). Learned prompt-ensemble weights strictly upstream of fusion/boundary/candidate logic (v94). PROMISING UNEXPLORED per-detection signals ORTHOGONAL to Q-score geometry and dur-prior: motion-energy / optical-flow boundary evidence, CLIP-text alignment measured AT the detection window, multi-scale/temporal-context curve shape. These must be per-detection, intra-video-discriminative, and not already baked into the pipeline.',
'',
'CURRENT CODE STATE (working tree): v94 baseline + inert flag-off mechanisms: v97 FCCS (--full_coverage_scan, PROVEN HARMFUL - do not enable), v100 EUBC (inert), v105/v107 ECE-Live (config scoring.event_continuation_extension, PROVEN +0.002 in vivo, inert), v111 Q-score (--qscore_rerank / ScoringConfig.qscore_rerank, offline +0.0087 UNVERIFIED live, inert - fully wired in config.py/pipeline.py/batch_evaluate.py/run.py). All default OFF = byte-identical v94 behavior. v104 TSG was REVERTED and PHYSICALLY REMOVED. The offline harness ' + CODE + '/iteration_logs/analysis_v111_qscore_precheck.py reproduces v108 exactly and is the reusable validation tool for re-rank loops.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop (controlled experiment). Never enable a proven-harmful flag (FCCS) alongside the mechanism under test.',
'- ARCHITECTURE-LEVEL ideas only - NO pure parameter/threshold tuning.',
'- All new params via config.py with a switch restoring prior behavior; every mechanism CLI-activatable + INFO log marker (FM5 guard).',
'- Keep the v51 half-max boundary-crossing mechanism unless a strong literature-backed argument says otherwise (v107 confirmed it live).',
'- Latency matters: prefer zero-cost (precomputed text embeddings) or training-free signals; avoid extra per-video VLM calls.',
'- When calling StructuredOutput: call it EXACTLY ONCE, at the very end, with the final answer. NEVER call it twice.',
'- ANTI-STALL (runtime kills agents silent >180s): no single Bash call may exceed ~150s (never sleep longer); Read with offset/limit; emit a short progress line between tool calls.',
].join('\n')

const history = [
  { loop: 1, version: 'v97', title: 'FCCS full-coverage comparative scan', smoke_map: 0.15035, full_map: 0.16155, avg_latency: 141.47, kept: true, lessons: 'FM5 EXPOSED: --full_coverage_scan never passed; 0.1616 is flag-off NOISE' },
  { loop: 7, version: 'v103', title: 'FCCS live activation test', smoke_map: 0.17821, full_map: 0.05109, avg_latency: 145.54, kept: false, lessons: 'GATE PASSED; genuine NEGATIVE: comparative full-coverage dilutes saliency, -0.11 mAP; FCCS PROVEN HARMFUL' },
  { loop: 8, version: 'v108', title: 'TSG live completion + coverage adjudication', smoke_map: 0.1443, full_map: 0.15834, avg_latency: 295.0, kept: false, lessons: 'GATE PASSED (fired 4/65); coverage NOT improved -> REVERTED + code REMOVED; mechanisms touching ~6% of videos cannot move full-set mAP' },
  { loop: 9, version: 'v107', title: 'ECE-Live deterministic in-vivo A/B', smoke_map: 0.14744, full_map: 0.17291, avg_latency: 166.42, kept: false, lessons: 'GATE PASSED; live A/B delta +0.002 < +0.005 -> REVERT; FM6 sim-to-live gap confirmed' },
  { loop: 10, version: 'v109', title: 'MIL cross-video confidence prior', smoke_map: -1, full_map: 0.15412, avg_latency: -1, kept: false, lessons: 'REJECTED at offline gate (-0.004). FM7: video-level prior cannot fix intra-video mislocation; re-rank must be PER-DETECTION' },
  { loop: 11, version: 'v110', title: 'TVR VideoMAE per-detection re-rank', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'REJECTED at offline kill-gate: UCF-finetuned VideoMAE window score SATURATES ~1.0 everywhere (incl normals), zero variance, dead signal' },
  { loop: 12, version: 'v111', title: 'Q-score localization-quality re-rank', smoke_map: -1, full_map: 0.16703, avg_latency: -1, kept: true, lessons: 'OFFLINE +0.0087 on v108 cached detections, broad-based & robust; KEPT flag-OFF; NEVER cleanly tested live (loop 1 of v4 resolves this)' },
  { loop: 13, version: 'v112', title: 'duration-fit stacking onto Q-score', smoke_map: -1, full_map: 0.16703, avg_latency: -1, kept: false, lessons: 'REJECTED offline (neutral/-0.01): dur-prior already baked into boundary constraints -> zero variance. Stack only ORTHOGONAL signals' },
  { loop: 14, version: 'v113', title: 'FCCS + Q-score stacking', smoke_map: 0.0524, full_map: 0.0524, avg_latency: -1, kept: false, lessons: 'CATASTROPHE 0.052 but CONFOUNDED: enabled proven-harmful FCCS alongside Q-score. Says nothing about Q-score. FM8: never combine a proven-harmful flag with the mechanism under test' },
]
let bestFullMap = 0.1494
let bestVersion = 'v94'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": smoke=' + h.smoke_map + ' full=' + h.full_map + ' latency=' + h.avg_latency + 's kept=' + h.kept + ' | lessons: ' + h.lessons)
  return 'TRUE current in-vivo best: ' + bestVersion + ' full mAP=' + bestFullMap + '. Session history:\n' + lines.join('\n')
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
    risks: { type: 'string', description: 'which known failure modes FM1-FM8 it might trigger and why it avoids them' },
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
    eval_flags: { type: 'string', description: 'exact extra CLI args for batch_evaluate.py that ACTIVATE the mechanism (e.g. "--qscore_rerank"); empty string only if the mechanism is default-ON via config' },
    activation_marker: { type: 'string', description: 'unique literal log substring emitted at INFO level per video when the mechanism fires (e.g. "[QSCORE-GATE]"); used to grep-verify activation (FM5 guard)' },
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
const qAdjSchema = {
  type: 'object',
  required: ['ran', 'map_off', 'map_on', 'delta', 'note'],
  properties: {
    ran: { type: 'boolean', description: 'true iff the deterministic on/off A/B on the FRESH v114 detections completed' },
    map_off: { type: 'number', description: 'mAP of the fresh flag-off detections (harness baseline; must ~match the live full mAP as a sanity anchor)' },
    map_on: { type: 'number', description: 'mAP after applying the v111 all-equal base-0.5 Q-score re-rank to the SAME fresh detections' },
    delta: { type: 'number', description: 'map_on - map_off (noise-free, same detections)' },
    ap_detail: { type: 'string', description: 'AP@0.3/0.5/0.7 for off and on' },
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
'SESSION HISTORY (do NOT repeat already-tried ideas; note how many died from FM5 non-activation; FCCS + ECE + TSG + MIL-prior + VideoMAE + dur-fit are CONFIRMED negatives; Q-score is the one positive-so-far offline signal being verified live in loop 1):\n' + hist(),
'',
'TASK:',
'1. Do the MANDATORY READING (session ledger + newest idea files are the highest value; then skim pipeline.py docstring + config.py + the 2-3 most relevant source files for your angle).',
'2. SEARCH-STORM: run at least 6 real WebSearch queries on 2023-2026 papers about TEMPORAL LOCALIZATION mechanisms (how exactly do they localize: proposal generation, boundary regression, start/end probability curves, Gaussian modeling, MIL snippet scoring, temporal contrast, motion/optical-flow signals, training-free localization, confidence/IoU-aware re-ranking). Use WebFetch on the most promising 2-3 papers (arXiv abs/html pages) to extract their exact localization mechanism.',
'3. From the literature + codebase evidence, derive ONE architecture-level idea that could break the 0.149 structural ceiling. STRONGLY PREFER the validated-safe family: a PURE PER-DETECTION post-hoc confidence re-rank signal that is (a) intra-video-discriminative, (b) ORTHOGONAL to the existing v111 Q-score geometry and the dur-prior, (c) not already baked into the pipeline, (d) validatable by a DETERMINISTIC offline A/B on a prior run\'s real detections. It must respect the HARD CONSTRAINTS and avoid FM1-FM8. Parameter tuning is FORBIDDEN. State the CLI flag + unique INFO log marker (FM5 guard) in the implementation_plan.',
'4. Return your proposal via StructuredOutput (call it EXACTLY ONCE). Be concrete: name the exact files/functions to change.',
'',
'!!! ANTI-STALL RULES (CRITICAL - the runtime KILLS any agent silent for >180s):',
'- Keep EVERY single tool call fast: prefer WebSearch over WebFetch; WebFetch at most 2 arXiv ABS pages (small), never big HTML/full-text pages; Read with offset/limit, never whole large files; NEVER run sleep or any Bash command longer than ~60s.',
'- Emit one short progress line (a few words) before and after every tool call so you never go silent.',
'- Budget yourself: if a search or fetch fails or is slow once, move on. Aim to finish the whole task in <15 tool calls.',
  ].join('\n')
}

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
  return [
'You are Agent D-launcher. Your ONLY job: ensure the ' + tag + ' evaluation for version ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. Do NOT wait for it to finish. Keep replies SHORT; every Bash call < 9 minutes. Call StructuredOutput EXACTLY ONCE at the end.',
'',
'Paths: code=' + CODE + ' ; output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (FM5-critical, must appear verbatim in the command): ' + (evalFlags || '(none - baseline/default-on run)'),
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
'3. If the process is dead and no report exists: tail -30 the log. If the failure looks TRANSIENT (API timeout / ConnectTimeout / rate limit / network), relaunch with the exact nohup command (max_per_category ' + maxPerCat + ', WITH the required flags: ' + (evalFlags || '(none)') + ') and return done=false, crashed=false, marker_count=-1, note="relaunched after: <cause>". If it looks like a CODE BUG (Python Traceback referencing project code), return done=false, crashed=true, marker_count=-1, note=<error summary>.',
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
'Return via StructuredOutput (EXACTLY ONCE): implemented (bool), files_changed (comma-separated), notes, eval_flags (exact CLI args that ACTIVATE the mechanism, e.g. "--qscore_rerank"; empty string only if default-ON), activation_marker (the literal INFO log substring, e.g. "[QSCORE-GATE]").',
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
'1. If decision=REVERT: restore every file present in ' + ROOT + '/.qoder/backups/' + V + '/ back into ' + CODE + '/ and verify each restore with diff (must be identical). If there is no backup dir for ' + V + ' (e.g. verify-only loop where the mechanism already ships flag-off by default), ensure the mechanism config stays default-OFF (byte-identical baseline behavior) and do nothing else. If decision=KEEP: do not touch the code. If decision starts with KEEP-FLAG-ON: flip the exact config field named in the decision text default to True in config.py (this becomes the new baseline) and sanity-import pipeline.',
'2. Copy artifacts that exist: cp ' + ROOT + '/results/batch_eval_' + V + '_smoke.log ' + CODE + '/iteration_logs/results/' + V + '_smoke_run.log ; cp ' + ROOT + '/results/batch_eval_' + V + '_smoke/evaluation_report.json ' + CODE + '/iteration_logs/results/' + V + '_smoke_eval.json ; same two copies for _full (if the full eval ran). For verify-only loops that reused an earlier version\'s smoke, copy that smoke log/report too and note the source.',
'3. date +%Y%m%d, then write ' + CODE + '/iteration_logs/ideas/loop' + loop + '_' + V + '_<kebab-case-slug>_<YYYYMMDD>.md with: the approved proposal, C\'s feedback, implementation notes, the eval flags + activation marker, marker counts in smoke/full logs, a results table (smoke mAP; full mAP, AP@0.3/0.5/0.7, latency; for loop 1 also the deterministic map_off/map_on/delta A/B), the keep/revert decision + reason, and lessons learned (which failure modes confirmed/refuted).',
'4. Append one row to ' + CODE + '/iteration_logs/session_v97_ledger.md (create the file with a markdown table header if it is missing): | loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept |',
'5. Return via StructuredOutput (EXACTLY ONCE): version="' + V + '", implemented=true, smoke_map=' + smoke.mAP + ', full_map=' + full.mAP + ', avg_latency=' + full.latency + ', kept=' + (decisionText.indexOf('KEEP') === 0) + ', summary (5-8 sentences), lessons (2-4 sentences), report_path.',
  ].join('\n')
}

function qscoreAdjudicatePrompt(V, full) {
  return [
'You are the Q-score IN-VIVO adjudicator for ' + V + '. The v111 localization-quality Q-score re-rank (--qscore_rerank / ScoringConfig.qscore_rerank) was KEPT on the strength of a +0.0087 OFFLINE gain measured against v108 CACHED detections, but it was NEVER cleanly tested live (v113 confounded it with the proven-harmful FCCS flag). This loop delivers the clean verdict. Call StructuredOutput EXACTLY ONCE. Keep every Bash call < 150s.',
'',
'The ' + V + ' full eval just ran FLAG-OFF, producing FRESH real detections at ' + ROOT + '/results/batch_eval_' + V + '_full/ (per-video JSONs with frame_scores, timestamps, anomalies). Its headline full mAP = ' + full.mAP + ' (this is map_off; use it as a sanity anchor).',
'',
'PROTOCOL (deterministic on/off A/B on the SAME fresh detections - noise-free, faithful in-vivo per the FM6 distinction; a pure re-rank cannot change boundaries/count):',
'1. Copy ' + CODE + '/iteration_logs/analysis_v111_qscore_precheck.py to ' + CODE + '/iteration_logs/analysis_' + V + '_qscore_invivo.py.',
'2. Edit ONLY the V108_DIR constant to point at ' + ROOT + '/results/batch_eval_' + V + '_full ; keep everything else byte-identical (the harness reproduces the raw dir mAP as its baseline, then re-ranks). To save time you MAY reduce the configs dict to just the chosen v111 config {\'all-equal b0.5\': ((0.34,0.33,0.33),0.5)} plus the implicit baseline.',
'3. Run it: cd ' + CODE + ' && python3 iteration_logs/analysis_' + V + '_qscore_invivo.py  (numpy only, no VLM; should finish in <90s). Read the printed [SANITY] baseline mAP and the all-equal b0.5 row.',
'4. map_off = harness baseline mAP (MUST be within 0.0005 of the headline ' + full.mAP + ' - if not, something is wrong, report ran=false with the mismatch). map_on = all-equal b0.5 mAP. delta = map_on - map_off.',
'Return via StructuredOutput (EXACTLY ONCE): ran (bool), map_off, map_on, delta, ap_detail (AP@0.3/0.5/0.7 for off and on), note.',
  ].join('\n')
}

// Loop-1 seeded experiment: clean IN-VIVO Q-score verdict (v111 mechanism; deterministic on/off A/B on fresh detections).
const SEED_PROPOSAL = {
  title: 'Q-score In-Vivo Verdict (v111 mechanism, clean fresh-detection deterministic A/B, FM6 resolution + v113 confound cleanup)',
  mechanism: 'v111 localization-quality Q-score post-hoc confidence re-rank (already wired behind --qscore_rerank / ScoringConfig.qscore_rerank): re-ranks final-detection confidences from fused-curve GEOMETRY (boundary sharpness + peak contrast + center fit, within-video min-max), boundaries/counts byte-identical. This loop changes NO code: it runs a fresh flag-OFF full eval, then applies the deterministic Q-score re-rank (v111 harness, all-equal base 0.5) to THOSE fresh detections to measure the noise-free on/off delta.',
  rationale: 'Q-score was KEPT flag-off on a +0.0087 OFFLINE gain vs v108 CACHED detections; v113 then enabled it TOGETHER with the proven -0.11 FCCS flag (confounded) and wrongly declared it dead. It has never had a clean live verdict. Because a pure post-hoc re-rank cannot change boundaries/counts (mAP depends only on the global confidence ordering), a deterministic on/off A/B on FRESH detections is a faithful, VLM-noise-free in-vivo measurement that directly answers FM6: does the v108-cached +0.0087 transfer to fresh detections?',
  implementation_plan: 'Verify-only, no code change. 1) run fresh full 65-video eval FLAG-OFF (baseline fresh detections; batch_evaluate saves per-video JSONs with frame_scores/timestamps/anomalies). 2) copy analysis_v111_qscore_precheck.py -> analysis_v114_qscore_invivo.py, repoint V108_DIR to batch_eval_v114_full, run baseline + all-equal b0.5. 3) delta = mAP(all-equal b0.5) - mAP(baseline) on the SAME fresh detections. Activation is via the deterministic harness, not the live flag (the pipeline helper byte-matches the harness per v111), so no live marker is required for this loop.',
  expected_gain: 'Pre-registered: delta>=+0.005 -> KEEP-FLAG-ON (flip ScoringConfig.qscore_rerank default False->True; new baseline = map_on). 0<=delta<+0.005 -> KEEP code flag-OFF (offline gain did not clear the live threshold; no baseline change). delta<0 -> REVERT (FM6 confirmed for qscore; keep code inert).',
  risks: 'FM1/FM2 immune by construction (re-rank only). FM6 is the exact question under test. FM8 addressed by changing ONE thing (qscore only; FCCS stays OFF). FM5 N/A (deterministic offline re-rank of the run\'s own fresh detections).',
  latency_impact: 'One baseline full eval (~128s/video) + <90s numpy re-score. No extra VLM calls.',
  literature: 'v111 (this project); IoU-aware confidence re-rank / Q-score (.qoder spec Step 3.2); BSN/BMN boundary-evidence localization; T3AL (CVPR 2024).',
}
const SEED_VERDICT = { approve: true, feedback: 'Pre-approved: this loop only delivers the clean in-vivo verdict on the already-implemented v111 Q-score, resolving the v113 confound and the FM6 transfer question. No code change; pre-registered decision rule on the deterministic fresh-detection A/B delta.', mandated_changes: '' }
const SEED_IMPL = { implemented: true, files_changed: '(none - verify only)', notes: 'v111 Q-score already implemented and wired (config.py ScoringConfig.qscore_rerank + pipeline._qscore_rerank + batch_evaluate/run --qscore_rerank). This loop runs a fresh FLAG-OFF full eval then applies the deterministic v111 harness on the fresh detections.', eval_flags: '', activation_marker: '' }

log('VAD agent team v4 starting. Target mAP>=' + TARGET + ', versions v' + START_VERSION + '+, max ' + MAX_LOOPS + ' loops. FM5+FM6+FM7+FM8-hardened. Loop 1 seeded: clean in-vivo Q-score verdict (deterministic fresh-detection A/B).')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION - 1 + loop)
  log('=== Loop ' + loop + ' (' + V + ') ===')

  let proposal, verdict, impl
  if (loop === 1) {
    proposal = SEED_PROPOSAL
    verdict = SEED_VERDICT
    impl = SEED_IMPL
    log('Loop 1: seeded clean in-vivo Q-score verdict (v111 mechanism; skipping research/review/implement; fresh flag-off eval + deterministic on/off A/B)')
  } else {
    phase('Research')
    const [propA, propB] = await parallel([
      () => retryAgent(() => agent(researchPrompt('A', 'Your specialty: TEMPORAL LOCALIZATION ARCHITECTURES from the TAL/VAD literature (BSN/BMN/DBG/ActionFormer-style proposal+boundary mechanisms, VadCLIP/PE-MIL/MGFN/AnomalyCLIP/GS-MoE/FreeZAD-style VAD mechanisms) and how their localization/confidence core can be adapted into a training-free PER-DETECTION post-hoc re-rank signal for this pipeline.'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
      () => retryAgent(() => agent(researchPrompt('B', 'Your specialty: CODEBASE-FIT + BOTTLENECK analysis. Known bottlenecks: 28.8% of candidates at completely wrong locations (low-CLIP categories Shoplifting/Stealing/Arson/Burglary have flat CLIP curves), 39% of detections have IoU 0.1-0.5 (boundary imprecision - but v107 proved boundary EXTENSION does not help live), MIL temporal discrimination collapsed, VideoMAE window score saturates ~1.0. You focus on PER-DETECTION signals the pipeline underuses (motion energy, optical-flow boundary evidence, temporal-context/multi-scale curve shape, CLIP-text alignment measured AT the detection window) that are ORTHOGONAL to the v111 Q-score geometry, and on recent weakly-supervised VAD SOTA for transferable re-rank mechanisms.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
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
      () => retryAgent(() => agent('You are Agent A continuing the debate. ' + debateBase + '\n\nTASK: Critique B\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM8 or repeat a tried/failed version? (b) is it truly architecture-level and PER-DETECTION / orthogonal to v111 Q-score? (c) is the expected gain plausible given +/-0.03 noise and the FM6 sim-to-live gap? (d) latency risk? Then state: would you accept B\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'A-critiques-B-' + V, phase: 'Debate' })),
      () => retryAgent(() => agent('You are Agent B continuing the debate. ' + debateBase + '\n\nTASK: Critique A\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM8 or repeat a tried/failed version? (b) is it truly architecture-level and PER-DETECTION / orthogonal to v111 Q-score? (c) is the expected gain plausible given +/-0.03 noise and the FM6 sim-to-live gap? (d) latency risk? Then state: would you accept A\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'B-critiques-A-' + V, phase: 'Debate' })),
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
'TASK: Synthesize ONE final proposal: pick the stronger idea, or merge both into one coherent SINGLE controlled change (never two independent changes). It must be architecture-level, respect all HARD CONSTRAINTS, avoid FM1-FM8, not repeat anything in SESSION HISTORY, and (strongly preferred) be a PER-DETECTION post-hoc re-rank signal orthogonal to the v111 Q-score. It MUST be CLI-activatable with a stated INFO log marker (FM5 guard). Return via StructuredOutput (EXACTLY ONCE, same schema as the proposals).',
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
'2. Avoids FM1-FM8 and all previously failed/reverted approaches in the history and idea logs? (No video-level priors=FM7; no signal already baked in like dur-prior; no VideoMAE window score; no FCCS; no scan-retargeting.)',
'3. Expected gain plausible vs +/-0.03 VLM noise? Is the mechanism causal, not wishful? Is the evidence in-vivo or a deterministic re-score of real detections (trusted) vs a candidate-pool simulation (FM6 - discount)?',
'4. Latency impact acceptable (target <= ~' + BASE_LATENCY + 's/video, hard limit +15%)?',
'5. Implementation risk: reversible, config-gated, ONE controlled change (no confounding flags)?',
'6. Is it a genuine PER-DETECTION, intra-video-discriminative signal ORTHOGONAL to the v111 Q-score, attacking a real bottleneck (wrong candidate locations / boundary imprecision)?',
'7. FM5 GUARD: does the implementation_plan state (a) the exact CLI flag that activates the mechanism on batch_evaluate.py (or config default-ON) and (b) the unique INFO log marker it emits per video? If either is missing, REJECT and demand it. (Verify-only re-rank loops that use a deterministic offline A/B on real detections may substitute that for the live marker.)',
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
        history.push({ loop: loop + 13, version: V, title: proposal.title + ' [REJECTED - no implementation]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
        continue
      }
    }

    phase('Implement')
    impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
    if (!impl || !impl.implemented) {
      log('Loop ' + loop + ': D1 implementation failed')
      history.push({ loop: loop + 13, version: V, title: proposal.title + ' [D1 FAILED]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'implementation failed: ' + (impl && impl.notes ? impl.notes.slice(0, 200) : 'agent null after retries') })
      continue
    }
    if (!impl.activation_marker) {
      log('Loop ' + loop + ': D1 reported no activation marker - FM5 risk, treating as failure')
      history.push({ loop: loop + 13, version: V, title: proposal.title + ' [NO MARKER - FM5]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'D1 did not provide an activation marker; loop aborted before evals' })
      continue
    }
  }

  // Loop 1 (seeded) skips the smoke gate: it is a flag-off baseline run whose fresh detections feed a deterministic on/off Q-score A/B.
  let smoke
  if (loop === 1) {
    smoke = { done: true, crashed: false, mAP: 0.14, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: 1, note: 'loop1 seed: baseline flag-off run; Q-score applied offline-deterministically on fresh detections (FM6 protocol); smoke gate synthetic-pass' }
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
      // Loop 1 runs the full eval FLAG-OFF (empty eval_flags, no marker) to get fresh baseline detections.
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' latency=' + full.latency + ' markers=' + full.marker_count + ' | ' + (full.note || '').slice(0, 150))
      if (impl.activation_marker && full.done && full.marker_count <= 0) {
        fm5 = true
        log('Loop ' + loop + ': FM5 - activation marker absent from full log')
      }
      if (!fm5 && full.done && !full.crashed) {
        if (loop === 1) {
          // Clean in-vivo Q-score verdict: deterministic on/off A/B on the fresh v114 detections overrides the headline.
          const adj = await retryAgent(() => agent(qscoreAdjudicatePrompt(V, full), { schema: qAdjSchema, label: 'QSCORE-invivo-' + V, phase: 'Implement' }))
          const delta = (adj && adj.ran) ? adj.delta : null
          log('Loop ' + loop + ': Q-score in-vivo A/B ran=' + (adj ? adj.ran : 'null') + ' map_off=' + (adj ? adj.map_off : '?') + ' map_on=' + (adj ? adj.map_on : '?') + ' delta=' + (delta === null ? '?' : delta) + ' | ' + (adj ? (adj.note || '').slice(0, 160) : 'agent null'))
          if (delta === null) {
            decision = 'revert'
            decisionText = 'REVERT (Q-score in-vivo adjudicator failed to produce a clean delta - defaulting to keep-flag-off inert per pre-registration; no baseline change)'
          } else if (delta >= 0.005) {
            decision = 'keep-flag-on'
            decisionText = 'KEEP-FLAG-ON (deterministic fresh-detection A/B delta ' + delta.toFixed(5) + ' >= +0.005: v111 Q-score transfers live; flip config.py ScoringConfig.qscore_rerank default False->True; new baseline map_on=' + adj.map_on + ', map_off=' + adj.map_off + '). ' + (adj.ap_detail || '')
            full = Object.assign({}, full, { mAP: adj.map_on, note: 'FRESH-DETECTION deterministic A/B: map_off=' + adj.map_off + ' map_on=' + adj.map_on + ' delta=' + delta.toFixed(5) + '. ' + (full.note || '') })
          } else if (delta >= 0) {
            decision = 'keep-flag-off'
            decisionText = 'KEEP (deterministic fresh-detection A/B delta ' + delta.toFixed(5) + ' in [0,+0.005): offline v108 gain did NOT clear the live threshold - keep Q-score code flag-OFF inert, NO baseline change). map_off=' + adj.map_off + ' map_on=' + adj.map_on + '. ' + (adj.ap_detail || '')
            full = Object.assign({}, full, { note: 'FRESH-DETECTION deterministic A/B: map_off=' + adj.map_off + ' map_on=' + adj.map_on + ' delta=' + delta.toFixed(5) + '. ' + (full.note || '') })
          } else {
            decision = 'revert'
            decisionText = 'REVERT (deterministic fresh-detection A/B delta ' + delta.toFixed(5) + ' < 0: FM6 CONFIRMED for Q-score - the offline v108-cached +0.0087 did NOT transfer to fresh detections; keep code inert flag-OFF). map_off=' + adj.map_off + ' map_on=' + adj.map_on + '. ' + (adj.ap_detail || '')
            full = Object.assign({}, full, { note: 'FRESH-DETECTION deterministic A/B: map_off=' + adj.map_off + ' map_on=' + adj.map_on + ' delta=' + delta.toFixed(5) + '. ' + (full.note || '') })
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
  history.push({ loop: loop + 13, version: V, title: proposal.title, smoke_map: result.smoke_map, full_map: result.full_map, avg_latency: result.avg_latency, kept: result.kept, lessons: result.lessons })
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
'You are Agent C writing the END-OF-SESSION SUMMARY for this VAD agent-team session (versions v' + START_VERSION + '+, continuation of the v97-v113 sessions). Call StructuredOutput EXACTLY ONCE if a schema is required; otherwise reply in plain text.',
'',
CONTEXT,
'',
'FULL SESSION HISTORY:\n' + hist(),
'',
'TASK:',
'1. Read ' + CODE + '/iteration_logs/session_v97_ledger.md and the new idea files in ' + CODE + '/iteration_logs/ideas/ from this session (versions v' + START_VERSION + '+).',
'2. Get today\'s date via: date +%Y%m%d',
'3. Write ' + CODE + '/iteration_logs/summary_loops_v' + START_VERSION + 'plus_<YYYYMMDD>.md in the SAME style as summary_loops_v90-94_20260712.md: executive summary (final best in-vivo mAP, whether target 0.20 was reached, the Q-score in-vivo verdict from loop 1), per-version what-was-tried table with results, updated failure-mode taxonomy (FM1-FM8 + any new modes), latency analysis, and concrete next directions.',
'4. Reply with a 10-line plain-text recap of the session outcome.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('Session complete. Best: ' + bestVersion + ' full mAP=' + bestFullMap + ' over ' + history.length + ' recorded loops.')
