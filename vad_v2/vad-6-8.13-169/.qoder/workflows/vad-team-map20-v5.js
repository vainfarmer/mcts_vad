export const meta = {
  name: 'vad-team-map20-v5',
  description: 'VAD agent team v5 (LATENCY-REDUCTION session, FM5-FM11-hardened): A/B research+debate, C skeptical review, D implement+eval. DUAL OBJECTIVE - reduce per-video latency while HOLDING mAP within -0.005 of the committed v114 baseline (0.15632). Continues at v124 after v114-v123 (v114 Q-score KEPT default-on = baseline 0.15632; v115-v123 all reverted: confidence-re-rank family TAPPED, boundary-placement un-adjudicated/refuted, recurring harness-skip process failure). Latency judged by DRIFT-FREE VLM/CLIP call-count reduction (primary) + same-session wall-clock (secondary); accuracy by full-eval mAP. Loop 1 seeded: complete the v114 CLI-default flip + add always-on [COST] per-video instrumentation to set the same-session reference. Min 5 new versions.',
  phases: [
    { title: 'Research', detail: 'A and B independently research latency-reduction mechanisms + profile the codebase cost, each proposes one architecture-level idea' },
    { title: 'Debate', detail: 'A and B cross-critique, then synthesize one joint consensus proposal' },
    { title: 'Review', detail: 'C approves or rejects with feedback; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (CLI-activatable + log marker), smoke+full eval with FM5 activation gate + [COST] cost adjudication, keep/revert on the latency+accuracy rule' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const MAX_LOOPS = 7
const START_VERSION = 124
const BASE_LATENCY = 128
const SMOKE_GATE = 0.06
// Dual-objective keep rule (user: "严格保精度" = protect accuracy hard):
const BASE_MAP = 0.15632      // committed v114 in-vivo baseline; accuracy FLOOR anchor (constant, no ratchet-down)
const MAP_TOL = 0.005         // a latency change may not drop full mAP below BASE_MAP - MAP_TOL = 0.15132
const LAT_WIN = 0.15          // must cut VLM/CLIP calls OR same-session wall-clock by >= 15% to be worth keeping
let REF_CALLS = -1            // same-session reference mean VLM calls/video (set in loop 1, updated on each kept latency win)
let REF_LAT = -1              // same-session reference mean wall-clock s/video (set in loop 1, updated on each kept latency win)

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision: video-level labels for training; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline, versioned docstring header; config.py centralizes ALL params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, llm_client.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up before editing.',
'',
'!!! THIS SESSION\'S OBJECTIVE IS LATENCY REDUCTION !!!',
'PRIMARY GOAL: reduce per-video execution time. HARD ACCURACY GUARD: full-eval temporal mAP must stay >= ' + BASE_MAP + ' - ' + MAP_TOL + ' = ' + (BASE_MAP - MAP_TOL).toFixed(5) + ' (the committed v114 baseline is ' + BASE_MAP + '). A latency change that drops mAP below that floor is REVERTED no matter how fast it is. "尽量不影响精度" - accuracy is near-sacred; latency is the maximize target.',
'KEEP RULE (latency loops): KEEP-FLAG-ON iff (a) full mAP >= ' + (BASE_MAP - MAP_TOL).toFixed(5) + ' AND (b) per-video VLM/CLIP call count OR same-session wall-clock dropped by >= ' + (LAT_WIN * 100) + '% vs the current reference. Otherwise REVERT.',
'',
'!!! FM10 - WALL-CLOCK LATENCY IS DOMINATED BY API DRIFT, NOT THE MECHANISM !!! The v114-v123 ledger shows the SAME pipeline config swinging 133s -> 165s -> 232s -> 295s -> 672s per video purely from qwen-VL API load. THEREFORE: (1) NEVER compare a run\'s wall-clock to a historical number - only to a SAME-SESSION baseline. (2) The PRIMARY, drift-free latency signal is the per-video VLM+CLIP CALL COUNT (deterministic given the mechanism), logged via the [COST] marker this session adds in loop 1. A mechanism that provably cuts calls has cut latency even if that run\'s wall-clock looks high. Report call-count reduction as the headline; wall-clock as corroboration.',
'',
'TRUE BASELINE: committed v114 full-eval temporal mAP = ' + BASE_MAP + ' (Q-score localization-quality re-rank default-ON: fused-curve geometry sharpness+contrast+center, within-video min-max; validated by deterministic fresh-detection A/B +0.00619). Prior v94/v89 baseline was 0.1494. Typical latency ~128-165 s/video at concurrency 3 (drift-noisy).',
'',
'LATENCY DRIVERS (per video, biggest first, to target): (1) VLM API calls - detect (qwen-vl-plus) + classify (qwen3-vl-plus) are the DOMINANT cost; cutting redundant/low-value VLM calls is the highest-leverage lever. (2) fps=2 frame sampling volume. (3) CLIP scoring (ViT-L-14) over segments. (4) MCTS (mcts_simulations=12, max_depth=4). (5) MIL + VideoMAE (VideoMAE window score is SATURATED ~1.0 = adds cost with near-zero signal, a prime prune candidate). CANDIDATE LEVERS (must be architecture-level, config-gated, mAP-neutral within the floor): early-exit / decisive-confidence short-circuit to skip further VLM calls; skip VLM scan of clearly-normal regions (gate on CLIP/MIL pre-screen); adaptive segment/keyframe count instead of fixed grid; reduce MCTS simulations if final detections are unchanged; drop or down-weight the saturated VideoMAE stage; cache/reuse embeddings across stages. Each must emit an INFO marker and be validatable by a single full eval (mAP + [COST]).',
'',
'!!! FM5 LESSON (v97-v102, and v116/v117/v120/v121/v123 in the last session) !!! A mechanism that is not actually ACTIVATED in the eval run produces pure noise. Guard: every mechanism MUST be activatable via a REAL CLI flag on batch_evaluate.py (verify: python3 batch_evaluate.py --help) OR default-ON via config, AND emit a unique INFO log marker per video when it fires; after each eval the log is greped for the marker; 0 markers = automatic REVERT.',
'',
'!!! FM9 - DEVICE-PLACEMENT BUG (new, v116) !!! v116 CRASHED 12/14 smoke videos with np.asarray() on an MPS tensor. Any code touching clip_features / text_features / torch tensors on device=mps MUST call .detach().cpu().numpy() before numpy ops, and MUST be smoke-tested on the ACTUAL mps device before a full eval.',
'',
'!!! FM11 - OFFLINE HARNESS MUST COMPLETE AS AN UPSTREAM, POLLABLE STEP (new, v117/v120/v121/v123) !!! Four of the last 10 loops died because a STEP-0 offline adjudication harness was skipped or exceeded the ~150s per-Bash-call anti-stall limit (v123 precheck was >8.5min and got killed). LESSON FOR THIS SESSION: PREFER latency mechanisms that need NO offline harness - a latency change is validated directly by ONE full eval (mAP + [COST] call count), so there is no bake-off harness to stall on. If a proposal genuinely needs a >150s precompute, it MUST be launched via nohup and POLLED like the eval (never a single blocking Bash call).',
'',
'CONFIRMED RESULTS FROM v114-v123 (do NOT re-try; full detail in iteration_logs/ideas/loop*_v11*_*.md and session_v97_ledger.md):',
'- v114 Q-score in-vivo verdict: KEPT, default-ON. Deterministic fresh-detection A/B +0.00619 (map_off 0.15013 -> map_on 0.15632). This is the committed baseline. FM6 refuted for pure post-hoc re-ranks.',
'- v115 PROM peak-prominence re-rank: REVERT (real +0.0026 but < +0.005 accuracy gate).',
'- v116 SLM CLIP text-margin: REVERT (FM9 MPS crash, untested).',
'- v117 FDR CLIP-feature bake-off: REVERT (FM11 harness never ran).',
'- v118 FSL raw-CLIP structural: REVERT (null +0.0003).',
'- v119 MELC motion-energy contrast: REVERT (null, inside noise).',
'- v120 BREF boundary-refine: REVERT (FM11 harness skipped, ran placeholder).',
'- v121 HYBG hysteresis boundary: REVERT (FM11, launch gate refused).',
'- v122 BOR boundary-offset recenter: REVERT (harness RAN, both arms refuted offline - clean true-negative).',
'- v123 SLM-CV cross-video margin: REVERT (FM11 harness >8.5min, killed).',
'SESSION-LEVEL LESSON: the cheap post-hoc CONFIDENCE-RE-RANK family is TAPPED (everything lands +0.002-0.003, below the +0.005 accuracy gate) and boundary-placement is unproductive so far. That is why THIS session pivots to LATENCY, where the win metric (call count) is different and drift-free.',
'',
'MANDATORY READING before proposing or reviewing (skim smartly, newest first):',
'1. ' + CODE + '/iteration_logs/summary_loops_v114plus_20260725.md  (last session executive summary + FM taxonomy)',
'2. ' + CODE + '/iteration_logs/session_v97_ledger.md  (all outcomes incl v114-v123 latency column)',
'3. ' + CODE + '/iteration_logs/ideas/ - newest first (loop1_v114 .. loop10_v123)',
'4. pipeline.py docstring header + config.py + llm_client.py + anomaly_discovery.py + frame_scorer.py (to profile where VLM/CLIP calls happen).',
'5. Research methodology: ' + ROOT + '/academic-research-skills/deep-research/SKILL.md - then run REAL WebSearch/WebFetch (arXiv, PapersWithCode, CVF) on EFFICIENT video understanding / fast TAL / VLM-call reduction / keyframe selection / cascade early-exit.',
'',
'KNOWN FAILURE MODES:',
'FM1 curve-shape modification regresses even when the modifying signal is individually better.',
'FM2 detection-count changes under greedy-match evaluation turn near-duplicates into FPs.',
'FM3 VLM self-consistency via temperature diversity hurts perceptual/extraction tasks.',
'FM4 self-consistency on segment-inclusion decisions is dangerous; on fixed detection labels (tie-break) it is safe.',
'FM5 mechanism implemented but never activated. Guard: eval_flags + activation marker + log grep gate.',
'FM6 candidate-pool simulation gains do not transfer; trust in-vivo deltas and deterministic re-scores of real detections.',
'FM7 video-level priors cannot fix intra-video mislocation; re-rank signals must be PER-DETECTION.',
'FM8 offline precheck is config-specific; never confound the mechanism under test with a proven-harmful flag. ONE controlled change per loop.',
'FM9 device-placement: MPS tensors need .detach().cpu().numpy(); smoke on the real device.',
'FM10 wall-clock latency is API-drift-dominated; use drift-free per-video CALL COUNT and same-session comparison.',
'FM11 an offline harness must complete as an upstream pollable step; prefer mechanisms needing none.',
'Also known: MIL v3/v4/v6 rejected; VideoMAE window score saturates ~1.0 (prune candidate); mAP ~0.15-0.16 is a structural accuracy ceiling for incremental post-hoc work.',
'',
'VALIDATED SAFE PATTERNS: pure post-hoc confidence re-rank validated by deterministic offline A/B on real detections (v95/v111); category tie-break on fixed detections (v89); learned prompt-ensemble weights upstream of fusion (v94). FOR LATENCY: computation-pruning that provably does not change (or negligibly changes) the final detection set is the safe archetype - it is mAP-neutral by construction, so the accuracy floor is easy to hold while calls drop.',
'',
'CURRENT CODE STATE (working tree): committed v114 baseline = v94 pipeline + Q-score re-rank (config.py ScoringConfig.qscore_rerank default True; note: from_args currently still defaults it False via getattr, so plain CLI runs need --qscore_rerank until loop 1 completes the flip). All v115-v123 mechanisms were REVERTED and removed. Loop 1 of THIS session completes the flip and adds always-on [COST] instrumentation.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop. ARCHITECTURE-LEVEL only - NO pure parameter/threshold tuning (reducing a magic number like mcts_simulations counts as tuning UNLESS accompanied by an architectural gate/short-circuit rationale and proven mAP-neutral).',
'- All new params via config.py with a switch restoring prior behavior; every mechanism CLI-activatable + INFO log marker (FM5 guard).',
'- Keep Q-score default-ON (it is the baseline) and the v51 half-max boundary mechanism.',
'- Latency is validated by a single full eval (mAP + [COST]) - do NOT design mechanisms that need a stalling offline harness (FM11).',
'- When calling StructuredOutput: call it EXACTLY ONCE, at the very end. NEVER twice.',
'- ANTI-STALL (runtime kills agents silent >180s): no single Bash call may exceed ~150s (never sleep longer); Read with offset/limit; emit a short progress line between tool calls.',
].join('\n')

const history = [
  { loop: 1, version: 'v114', title: 'Q-score in-vivo verdict', smoke_map: 0.14, full_map: 0.15632, avg_latency: 165.87, kept: true, lessons: 'KEPT default-ON; deterministic fresh-detection A/B +0.00619; committed baseline; FM6 refuted for post-hoc re-rank' },
  { loop: 2, version: 'v115', title: 'PROM peak-prominence re-rank', smoke_map: 0.18344, full_map: 0.17559, avg_latency: 177.62, kept: false, lessons: 'REVERT: real +0.0026 but < +0.005 gate; confidence-re-rank family tapped' },
  { loop: 3, version: 'v116', title: 'SLM CLIP text-margin', smoke_map: 0.0, full_map: -1, avg_latency: -1, kept: false, lessons: 'REVERT: FM9 MPS->numpy device crash 12/14 smoke; untested' },
  { loop: 4, version: 'v117', title: 'FDR CLIP-feature bake-off', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'REVERT: FM11 STEP-0 harness never ran; unlaunchable' },
  { loop: 5, version: 'v118', title: 'FSL raw-CLIP structural', smoke_map: 0.08953, full_map: 0.15664, avg_latency: 672.13, kept: false, lessons: 'REVERT: effect null (+0.0003); latency 672s = API drift not mechanism (FM10)' },
  { loop: 6, version: 'v119', title: 'MELC motion-energy contrast', smoke_map: 0.08171, full_map: 0.16883, avg_latency: 232.13, kept: false, lessons: 'REVERT: new modality still null, inside noise' },
  { loop: 7, version: 'v120', title: 'BREF boundary-refine', smoke_map: 0.13889, full_map: 0.17488, avg_latency: 158.35, kept: false, lessons: 'REVERT: FM11 harness skipped, ran placeholder; headline in noise' },
  { loop: 8, version: 'v121', title: 'HYBG hysteresis boundary', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'REVERT: FM11 launch gate refused (no promoted_arm)' },
  { loop: 9, version: 'v122', title: 'BOR boundary-offset recenter', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'REVERT: harness RAN, both arms refuted offline; clean true-negative at zero cost' },
  { loop: 10, version: 'v123', title: 'SLM-CV cross-video margin', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'REVERT: FM11 harness >8.5min killed; unlaunchable' },
]
let bestFullMap = BASE_MAP
let bestVersion = 'v114'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": smoke=' + h.smoke_map + ' full=' + h.full_map + ' latency=' + h.avg_latency + 's kept=' + h.kept + ' | lessons: ' + h.lessons)
  return 'COMMITTED baseline: ' + bestVersion + ' full mAP=' + bestFullMap + ' (accuracy floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '). Same-session latency reference: REF_vlm_calls=' + REF_CALLS + ', REF_wallclock=' + REF_LAT + 's/video.\nSession history:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  required: ['title', 'mechanism', 'rationale', 'implementation_plan', 'expected_gain', 'risks', 'latency_impact', 'literature'],
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'the architecture-level LATENCY-reduction change, precisely' },
    rationale: { type: 'string', description: 'why it cuts per-video VLM/CLIP calls (or clearly cuts compute) while staying mAP-neutral within the accuracy floor; evidence from papers + this codebase profile' },
    implementation_plan: { type: 'string', description: 'exact files/functions to change, step by step; MUST state the CLI flag to activate it and the INFO log marker it emits (FM5 guard); MUST NOT require a stalling offline harness (FM11)' },
    expected_gain: { type: 'string', description: 'expected per-video VLM/CLIP call-count reduction (the drift-free headline) and expected mAP impact (must hold >= ' + (BASE_MAP - MAP_TOL).toFixed(5) + ')' },
    risks: { type: 'string', description: 'which failure modes FM1-FM11 it might trigger and why it avoids them; especially the accuracy-floor risk' },
    latency_impact: { type: 'string' },
    literature: { type: 'string', description: 'papers cited (name, venue/year, what mechanism was borrowed)' },
  },
}

const verdictSchema = {
  type: 'object',
  required: ['approve', 'feedback'],
  properties: {
    approve: { type: 'boolean' },
    feedback: { type: 'string' },
    mandated_changes: { type: 'string' },
  },
}

const resultSchema = {
  type: 'object',
  required: ['version', 'implemented', 'smoke_map', 'full_map', 'avg_latency', 'kept', 'summary', 'lessons'],
  properties: {
    version: { type: 'string' },
    implemented: { type: 'boolean' },
    smoke_map: { type: 'number' },
    full_map: { type: 'number' },
    avg_latency: { type: 'number' },
    kept: { type: 'boolean' },
    summary: { type: 'string' },
    lessons: { type: 'string' },
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
    eval_flags: { type: 'string', description: 'exact extra CLI args that ACTIVATE the mechanism; include --qscore_rerank so the committed baseline is on; empty only if fully default-on' },
    activation_marker: { type: 'string', description: 'unique literal INFO log substring emitted per video when the mechanism fires (FM5 guard)' },
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
    mAP: { type: 'number' },
    ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' },
    latency: { type: 'number', description: 'avg_per_video_seconds (wall-clock, API-drift-noisy)' },
    marker_count: { type: 'number' },
    note: { type: 'string' },
  },
}
const costSchema = {
  type: 'object',
  required: ['ran', 'mean_vlm_calls', 'mean_clip_calls', 'mean_elapsed_s', 'videos_counted', 'note'],
  properties: {
    ran: { type: 'boolean', description: 'true iff [COST] lines were parsed' },
    mean_vlm_calls: { type: 'number', description: 'mean per-video VLM (detect+classify) calls; -1 if unknown' },
    mean_clip_calls: { type: 'number', description: 'mean per-video CLIP scoring calls; -1 if unknown' },
    mean_elapsed_s: { type: 'number', description: 'mean per-video elapsed seconds from [COST]; -1 if unknown' },
    videos_counted: { type: 'number' },
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
'You are Agent ' + role + ' on a VAD research team. THIS SESSION\'S GOAL IS LATENCY REDUCTION (cut per-video VLM/CLIP calls) WITHOUT dropping mAP below the accuracy floor. ' + focus,
'',
CONTEXT,
'',
'SESSION HISTORY (do NOT repeat tried ideas; the confidence-re-rank accuracy family is TAPPED - this session is about SPEED):\n' + hist(),
'',
'TASK:',
'1. Do the MANDATORY READING (last-session summary + ledger latency column; then PROFILE the codebase: grep where detect/classify VLM calls and CLIP scoring happen in pipeline.py / anomaly_discovery.py / anomaly_expert.py / frame_scorer.py / llm_client.py; count how many VLM calls a typical video makes and where).',
'2. SEARCH-STORM: run at least 5 real WebSearch queries on 2023-2026 papers about EFFICIENT video understanding / fast temporal action localization / reducing VLM or LLM calls / keyframe & frame selection / cascade & early-exit inference / cheap normal-vs-anomaly pre-gating. WebFetch at most 2 small arXiv abs pages.',
'3. Derive ONE architecture-level LATENCY-reduction idea: it must (a) cut per-video VLM/CLIP call count meaningfully (>= ~15%), (b) be mAP-neutral within the floor (' + (BASE_MAP - MAP_TOL).toFixed(5) + '), ideally by not changing the final detection set, (c) be config-gated + CLI-activatable + emit an INFO marker (FM5), (d) need NO stalling offline harness - validatable by ONE full eval (FM11). Avoid FM1-FM11. Pure magic-number tuning is FORBIDDEN unless it is an architectural gate proven mAP-neutral.',
'4. Return your proposal via StructuredOutput (call it EXACTLY ONCE). Name exact files/functions and the expected call-count reduction.',
'',
'!!! ANTI-STALL (runtime KILLS any agent silent >180s): prefer WebSearch over WebFetch; Read with offset/limit; NEVER sleep or run any Bash > ~60s; emit a short progress line before/after every tool call; finish in <15 tool calls.',
  ].join('\n')
}

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
  return [
'You are Agent D-launcher. Your ONLY job: ensure the ' + tag + ' evaluation for version ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. Do NOT wait for it to finish. Keep replies SHORT; every Bash call < 9 minutes. Call StructuredOutput EXACTLY ONCE at the end.',
'',
'Paths: code=' + CODE + ' ; output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (FM5-critical, must appear verbatim in the command): ' + (evalFlags || '(none - default-on run)'),
'',
'STEPS:',
'1. ps aux | grep -i batch_evaluate | grep -v grep',
'2. If an eval for batch_eval_' + V + '_' + tag + ' is ALREADY running -> check its command line contains the required flags: ps -o command -p <pid> ; if flags MISSING, kill it and relaunch with flags; if present -> launched=true, note="adopted existing run".',
'3. If the report file already exists AND the log shows the required flags were used (grep the flag string; if no flags required, any existing report is fine) -> launched=true, note="report already exists". Otherwise delete the stale output_dir and relaunch.',
'4. If a DIFFERENT version\'s eval is running -> sleep 120 and re-check, up to 3 times; if still busy -> launched=false, note="blocked by other eval".',
'5. Otherwise launch exactly (extra flags before the redirect):',
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
"2. Else check the process: pgrep -f 'batch_evaluate.*batch_eval_" + V + "_" + tag + "' -> if alive: Bash 'sleep 170', then re-check the report; if still absent, Bash 'sleep 170' once more and re-check; if it appeared, extract metrics + marker count (done=true); otherwise return done=false, crashed=false, marker_count=-1, note='still running'. ANTI-STALL: no single Bash call may exceed 170s.",
'3. If the process is dead and no report exists: tail -30 the log. If TRANSIENT (API timeout / ConnectTimeout / rate limit / network), relaunch with the exact nohup command (max_per_category ' + maxPerCat + ', WITH the required flags: ' + (evalFlags || '(none)') + ') and return done=false, crashed=false, marker_count=-1, note="relaunched after: <cause>". If a CODE BUG (Python Traceback referencing project code), return done=false, crashed=true, marker_count=-1, note=<error summary>.',
'Return via StructuredOutput (EXACTLY ONCE): done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
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

function costAdjudicatePrompt(V, tag) {
  return [
'You are Agent D-cost for the ' + tag + ' eval of ' + V + '. Parse the drift-free per-video cost from the [COST] log lines. Do NOT edit code. Every Bash call < 150s. Call StructuredOutput EXACTLY ONCE.',
'',
'Log: ' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log',
'Each processed video emits a line like: [COST] video=<name> vlm_calls=<int> clip_calls=<int> elapsed_s=<float>',
'',
'STEP (one fast python one-liner, read the log, aggregate):',
"python3 -c \"import re;xs=[re.search(r'vlm_calls=(\\\\d+).*clip_calls=(\\\\d+).*elapsed_s=([0-9.]+)',l) for l in open('" + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log')];xs=[m for m in xs if m];import statistics as s;print(len(xs), s.mean([int(m.group(1)) for m in xs]) if xs else -1, s.mean([int(m.group(2)) for m in xs]) if xs else -1, s.mean([float(m.group(3)) for m in xs]) if xs else -1)\"",
'The 4 printed numbers = videos_counted, mean_vlm_calls, mean_clip_calls, mean_elapsed_s. If the regex field names differ, grep one raw [COST] line first and adapt the pattern. If there are ZERO [COST] lines, return ran=false with a note (the instrumentation did not fire).',
'Return via StructuredOutput (EXACTLY ONCE): ran, mean_vlm_calls, mean_clip_calls, mean_elapsed_s, videos_counted, note.',
  ].join('\n')
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
'- Work CONTEXT-EFFICIENTLY: Grep to locate code; Read with offset/limit; NEVER read whole large files or cat logs.',
'- Backup BEFORE editing: mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and cp every file you will modify into it (skip files already backed up).',
'- A previous attempt may have PARTIALLY implemented this version. Inspect target files first; complete what is missing.',
'- Implement faithfully incl. ALL of C\'s mandated changes. All new params via config.py with a switch restoring prior behavior.',
'- FM5-CRITICAL: mechanism activatable via a real CLI flag on batch_evaluate.py (argparse + VADConfig.from_args getattr threading; verify: python3 batch_evaluate.py --help | grep your-flag) OR default-ON via config; AND emit a unique INFO log marker per video when it fires.',
'- FM9: any numpy op on an mps/torch tensor MUST use .detach().cpu().numpy() first.',
'- Update the pipeline.py docstring header with a concise ' + V + ' entry.',
'- Sanity-check each edited file: python3 -c "import ast; ast.parse(open(\'<f>\').read())"; then cd ' + CODE + ' && python3 -c "import pipeline".',
'Return via StructuredOutput (EXACTLY ONCE): implemented (bool), files_changed, notes, eval_flags (exact CLI args that ACTIVATE the mechanism AND include --qscore_rerank so the committed baseline is on), activation_marker (the literal INFO log substring).',
  ].join('\n')
}

function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, cost, decisionText, fm5) {
  return [
'You are Agent D4, the finalizer for version ' + V + '. Implementation and evals are DONE. Enforce the keep/revert decision, save artifacts, write the report, return the summary. Every Bash call < 9 minutes. Call StructuredOutput EXACTLY ONCE.',
'',
'Paths: code=' + CODE + ' ; results=' + ROOT + '/results ; backups=' + ROOT + '/.qoder/backups/' + V + ' ; ideas=' + CODE + '/iteration_logs/ideas/ ; runlogs=' + CODE + '/iteration_logs/results/',
'',
'DECISION FROM ORCHESTRATOR: ' + decisionText + (fm5 ? '  [FM5: mechanism never activated - 0 marker lines]' : ''),
'Metrics: smoke mAP=' + smoke.mAP + ' (marker_count=' + smoke.marker_count + ') ; full mAP=' + full.mAP + ' ; AP@0.3=' + full.ap30 + ' AP@0.5=' + full.ap50 + ' AP@0.7=' + full.ap70 + ' ; full wall-clock=' + full.latency + 's',
'COST (drift-free): mean_vlm_calls=' + (cost ? cost.mean_vlm_calls : '?') + ' mean_clip_calls=' + (cost ? cost.mean_clip_calls : '?') + ' mean_elapsed_s=' + (cost ? cost.mean_elapsed_s : '?') + ' over ' + (cost ? cost.videos_counted : '?') + ' videos ; SESSION REFERENCE: REF_vlm_calls=' + REF_CALLS + ' REF_wallclock=' + REF_LAT + 's ; accuracy floor=' + (BASE_MAP - MAP_TOL).toFixed(5),
'Eval flags used: ' + (impl.eval_flags || '(none)') + ' ; activation marker: "' + (impl.activation_marker || '(none)') + '"',
'',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
'',
"C's feedback:\n" + (verdict ? (verdict.feedback || '') : '(none)'),
'',
'Implementation notes from D1 (files: ' + (impl.files_changed || '?') + '):\n' + (impl.notes || '(none)'),
'',
'TASKS:',
'1. If decision=REVERT: restore every file in ' + ROOT + '/.qoder/backups/' + V + '/ back into ' + CODE + '/ and verify each with diff (must be identical). If no backup dir exists, ensure the mechanism config stays default-OFF. If decision starts with KEEP-FLAG-ON: flip the exact config field named in the decision text to True in config.py (new baseline) and sanity-import pipeline. If decision=KEEP: do not touch code.',
'2. Copy artifacts that exist: cp ' + ROOT + '/results/batch_eval_' + V + '_smoke.log ' + CODE + '/iteration_logs/results/' + V + '_smoke_run.log ; cp ' + ROOT + '/results/batch_eval_' + V + '_smoke/evaluation_report.json ' + CODE + '/iteration_logs/results/' + V + '_smoke_eval.json ; same two for _full if it ran.',
'3. date +%Y%m%d, then write ' + CODE + '/iteration_logs/ideas/loop' + loop + '_' + V + '_<kebab-case-slug>_<YYYYMMDD>.md with: the proposal, C\'s feedback, implementation notes, eval flags + activation marker + marker counts, a results table (smoke mAP; full mAP, AP@0.3/0.5/0.7; wall-clock latency; AND the drift-free [COST] mean_vlm_calls / mean_clip_calls / mean_elapsed_s vs the session reference), the keep/revert decision + reason (latency win % on calls AND wall-clock, plus whether the accuracy floor held), and lessons (which failure modes confirmed/refuted).',
'4. Append one row to ' + CODE + '/iteration_logs/session_v97_ledger.md: | loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept | (put the [COST] mean_vlm_calls and call-reduction % in the title cell so the latency story is legible).',
'5. Return via StructuredOutput (EXACTLY ONCE): version="' + V + '", implemented=true, smoke_map=' + smoke.mAP + ', full_map=' + full.mAP + ', avg_latency=' + full.latency + ', kept=' + (decisionText.indexOf('KEEP') === 0) + ', summary (5-8 sentences incl. the call-count delta), lessons (2-4 sentences), report_path.',
  ].join('\n')
}

// Loop-1 seed: baseline calibration - complete the v114 CLI-default flip + add always-on [COST] instrumentation.
const SEED_PROPOSAL = {
  title: 'Baseline Calibration: complete v114 Q-score CLI-default flip + add always-on [COST] per-video cost instrumentation',
  mechanism: 'Two mAP-neutral edits that make this session measurable. (1) Complete the v114 flip so plain runs use the committed Q-score baseline. (2) Add always-on per-video cost logging: count VLM (detect+classify) calls and CLIP scoring calls and wall-clock per video, emit one [COST] INFO line per video. Pure logging + a default change - the final detection set is UNCHANGED, so mAP is byte-neutral.',
  rationale: 'Every subsequent loop is judged on drift-free per-video VLM/CLIP call reduction (FM10), which requires an instrumented same-session reference. And the v114 keep-flag-on was only half-applied (dataclass default True but from_args still defaults False), so plain runs silently use the OLD baseline - this loop fixes that so the reference eval truly is the 0.15632 baseline.',
  implementation_plan: 'Edit A (complete flip): config.py from_args - change qscore_rerank=getattr(args,"qscore_rerank",False) to default True; add a --no_qscore_rerank store_false opt-out (dest=qscore_rerank) on batch_evaluate.py AND run.py so the committed baseline is the default and can still be disabled. Verify: python3 batch_evaluate.py --help | grep qscore. Edit B (instrumentation, DEFAULT-ON, no flag): in pipeline.py maintain per-video counters - increment on each VLM detect call, each VLM classify call (find the llm_client detect/classify invocations), and each CLIP scoring call; record a per-video start time; at the END of each video emit logger.info(f"[COST] video={name} vlm_calls={n_detect+n_classify} clip_calls={n_clip} elapsed_s={elapsed:.1f}"). The counters MUST NOT alter any detection. Keep --qscore_rerank passed explicitly in eval_flags as a belt-and-suspenders. activation_marker="[COST]".',
  expected_gain: 'Zero mAP change (pure logging + honest-default flip). Establishes REF_vlm_calls, REF_clip_calls, REF_wallclock for the session. Full eval mAP should reproduce ~0.15632 within VLM noise.',
  risks: 'FM9 avoided (no tensor math added). No accuracy risk (final detections unchanged). Only risk is a mis-placed counter double-counting - verify counts are plausible (a few detect + a few classify per video).',
  latency_impact: 'Negligible (<1ms logging per video).',
  literature: 'n/a (instrumentation loop).',
}
const SEED_VERDICT = { approve: true, feedback: 'Pre-approved: instrumentation + honest-baseline flip, mAP-neutral by construction, required to measure the rest of the session.', mandated_changes: 'The [COST] line MUST be machine-parseable with the exact tokens vlm_calls= clip_calls= elapsed_s=. Counters must be verified not to change detections (diff a 1-video run\'s anomalies vs a pre-edit run if in doubt).' }

log('VAD agent team v5 (LATENCY) starting. Baseline v114 mAP=' + BASE_MAP + ', accuracy floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + ', latency-win gate ' + (LAT_WIN * 100) + '%. Versions v' + START_VERSION + '+, max ' + MAX_LOOPS + ' loops. Loop 1 seeded: baseline calibration + [COST] instrumentation.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION - 1 + loop)
  log('=== Loop ' + loop + ' (' + V + ') ===')

  let proposal, verdict
  if (loop === 1) {
    proposal = SEED_PROPOSAL
    verdict = SEED_VERDICT
    log('Loop 1: seeded baseline calibration (flip + [COST] instrumentation); skipping research/debate/review')
  } else {
    phase('Research')
    const [propA, propB] = await parallel([
      () => retryAgent(() => agent(researchPrompt('A', 'Your specialty: EFFICIENT video-understanding / fast-TAL LITERATURE - keyframe & frame selection, cascade/early-exit inference, token/segment pruning, cheap pre-gating to skip expensive VLM calls. Adapt one such mechanism to cut this pipeline\'s per-video VLM/CLIP calls training-free.'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
      () => retryAgent(() => agent(researchPrompt('B', 'Your specialty: CODEBASE COST PROFILING - map exactly how many VLM detect/classify calls and CLIP scoring calls a typical video makes and where (pipeline.py stages, anomaly_discovery scan, anomaly_expert). Find the biggest call-count sink and a mAP-neutral way to short-circuit it (e.g. skip VLM on clearly-normal pre-gated regions, adaptive segment count, prune the saturated VideoMAE stage, reduce redundant re-scans).'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
    ])
    if (!propA && !propB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }
    const pA = propA ? JSON.stringify(propA, null, 1) : '(A failed)'
    const pB = propB ? JSON.stringify(propB, null, 1) : '(B failed)'

    phase('Debate')
    const debateBase = [CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '', "Proposal from A:\n" + pA + "\n\nProposal from B:\n" + pB].join('\n')
    const [critAonB, critBonA] = await parallel([
      () => retryAgent(() => agent('You are Agent A continuing the debate. ' + debateBase + '\n\nTASK: Critique B\'s LATENCY proposal ruthlessly: (a) does it really cut per-video VLM/CLIP calls, and by roughly how much? (b) is it mAP-neutral within the floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + ', or does it risk dropping detections? (c) any FM1-FM11 violation, esp. FM11 (needs a stalling harness)? (d) is it truly architecture-level not magic-number tuning? Then: accept B over yours, reject, or merge - exactly how. <400 words.', { label: 'A-critiques-B-' + V, phase: 'Debate' })),
      () => retryAgent(() => agent('You are Agent B continuing the debate. ' + debateBase + '\n\nTASK: Critique A\'s LATENCY proposal ruthlessly: (a) real per-video call reduction and magnitude? (b) mAP-neutral within the floor, or detection-loss risk? (c) FM1-FM11 violations, esp. FM11? (d) architecture-level not tuning? Then: accept A over yours, reject, or merge - exactly how. <400 words.', { label: 'B-critiques-A-' + V, phase: 'Debate' })),
    ])

    const consensus = await retryAgent(() => agent([
'You are the joint A+B consensus. Reach the SINGLE joint optimal LATENCY-reduction proposal both would sign.',
'', debateBase, '',
"A's critique of B:\n" + (critAonB || '(missing)'), '',
"B's critique of A:\n" + (critBonA || '(missing)'), '',
'TASK: Synthesize ONE final proposal (pick stronger or merge into ONE controlled change). Must cut per-video VLM/CLIP calls >= ~15%, be mAP-neutral within the floor, architecture-level, avoid FM1-FM11 (especially need NO stalling offline harness), be CLI-activatable with an INFO marker. Return via StructuredOutput (EXACTLY ONCE, proposal schema).',
    ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
    if (!consensus) { log('Loop ' + loop + ': consensus failed, skipping'); continue }

    phase('Review')
    proposal = consensus
    verdict = null
    for (let r = 0; r < 3; r++) {
      verdict = await retryAgent(() => agent([
'You are Agent C, the skeptical reviewer. You gatekeep what reaches implementation. Call StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'JOINT LATENCY PROPOSAL (attempt ' + (r + 1) + '):\n' + JSON.stringify(proposal, null, 1),
'',
'REVIEW CHECKLIST (strict but fair):',
'1. Does it genuinely cut per-video VLM/CLIP call count (the drift-free win metric), and by a plausible >= 15%?',
'2. Is it mAP-neutral within the floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '? Prefer changes that do NOT alter the final detection set. If it drops detections/regions, is the accuracy risk argued down?',
'3. Architecture-level, NOT pure magic-number tuning (reducing mcts_simulations alone = tuning unless gated by an architectural short-circuit proven neutral)?',
'4. Avoids FM1-FM11? Especially FM11: does it need a stalling offline harness? If so, REJECT or demand it be validatable by a single full eval.',
'5. Reversible, config-gated, ONE controlled change?',
'6. FM5 GUARD: implementation_plan states (a) the CLI flag (or default-on) and (b) the unique INFO marker? If missing, REJECT and demand it. FM9: any tensor->numpy uses .detach().cpu().numpy()?',
'If fundamentally flawed or accuracy-risky, REJECT with concrete feedback. If good but imperfect, APPROVE with mandated_changes. Return via StructuredOutput (EXACTLY ONCE).',
      ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
      if (!verdict || verdict.approve) break
      log('Loop ' + loop + ': C rejected (round ' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
      if (r === 2) break
      proposal = await retryAgent(() => agent([
'You are the joint A+B team. Reviewer C REJECTED your LATENCY proposal. Revise into a NEW or FIXED proposal addressing every point. You may pivot mechanisms. Still one controlled architecture-level change cutting VLM/CLIP calls while holding mAP. Call StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'REJECTED PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
"C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
'', 'Return the revised proposal via StructuredOutput (EXACTLY ONCE).',
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
        log('Loop ' + loop + ': C rejected 3x without mandate, skipping implementation')
        history.push({ loop: loop + 20, version: V, title: proposal.title + ' [REJECTED - no implementation]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
        continue
      }
    }
  }

  phase('Implement')
  let impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented) {
    log('Loop ' + loop + ': D1 implementation failed')
    history.push({ loop: loop + 20, version: V, title: proposal.title + ' [D1 FAILED]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'implementation failed: ' + (impl && impl.notes ? impl.notes.slice(0, 200) : 'agent null after retries') })
    continue
  }
  // Loop 1's instrumentation is default-on with the fixed [COST] marker; force it so the FM5 gate + cost adjudication key off it.
  if (loop === 1) {
    impl = Object.assign({}, impl, { activation_marker: '[COST]', eval_flags: (impl.eval_flags && impl.eval_flags.indexOf('--qscore_rerank') >= 0) ? impl.eval_flags : ('--qscore_rerank ' + (impl.eval_flags || '')).trim() })
  }
  if (!impl.activation_marker) {
    log('Loop ' + loop + ': D1 reported no activation marker - FM5 risk, treating as failure')
    history.push({ loop: loop + 20, version: V, title: proposal.title + ' [NO MARKER - FM5]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'D1 did not provide an activation marker' })
    continue
  }

  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker)
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count + ' | ' + (smoke.note || '').slice(0, 150))

  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (gate)' }
  let cost = null
  let decision = 'revert'
  let decisionText = 'REVERT'
  let fm5 = false
  if (smoke.done && !smoke.crashed && smoke.mAP >= SMOKE_GATE) {
    if (impl.activation_marker && smoke.marker_count <= 0) {
      fm5 = true
      log('Loop ' + loop + ': FM5 - marker "' + impl.activation_marker + '" absent from smoke log; skipping full, will revert')
    } else {
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' wallclock=' + full.latency + ' markers=' + full.marker_count + ' | ' + (full.note || '').slice(0, 150))
      if (impl.activation_marker && full.done && full.marker_count <= 0) {
        fm5 = true
        log('Loop ' + loop + ': FM5 - marker absent from full log')
      }
      if (!fm5 && full.done && !full.crashed) {
        cost = await retryAgent(() => agent(costAdjudicatePrompt(V, 'full'), { schema: costSchema, label: 'D-cost-' + V, phase: 'Implement' }))
        log('Loop ' + loop + ': cost ran=' + (cost ? cost.ran : 'null') + ' vlm=' + (cost ? cost.mean_vlm_calls : '?') + ' clip=' + (cost ? cost.mean_clip_calls : '?') + ' elapsed=' + (cost ? cost.mean_elapsed_s : '?') + ' n=' + (cost ? cost.videos_counted : '?'))
        if (loop === 1) {
          // Baseline calibration: set the same-session reference and keep the instrumentation.
          if (cost && cost.ran) {
            if (cost.mean_vlm_calls > 0) REF_CALLS = cost.mean_vlm_calls
            if (cost.mean_elapsed_s > 0) REF_LAT = cost.mean_elapsed_s
          }
          decision = 'keep-flag-on'
          decisionText = 'KEEP-FLAG-ON (loop-1 baseline calibration: completed the v114 Q-score CLI-default flip [config.py from_args qscore_rerank default True + --no_qscore_rerank opt-out] and added always-on [COST] instrumentation. Session reference set: REF_vlm_calls=' + REF_CALLS + ', REF_wallclock=' + REF_LAT + 's, fresh mAP=' + full.mAP + '. mAP-neutral by construction. Config field to keep True: qscore_rerank [already default True in dataclass; the flip is in from_args, no further config flip needed - just KEEP code, do NOT revert].)'
          // Loop 1 is a KEEP (code already committed by D1); no config re-flip beyond what D1 did.
          decision = 'keep'
        } else {
          const mapOK = full.mAP >= (BASE_MAP - MAP_TOL)
          const callWin = (REF_CALLS > 0 && cost && cost.ran && cost.mean_vlm_calls > 0) ? (1 - cost.mean_vlm_calls / REF_CALLS) : -999
          const latWin = (REF_LAT > 0 && cost && cost.ran && cost.mean_elapsed_s > 0) ? (1 - cost.mean_elapsed_s / REF_LAT) : ((REF_LAT > 0 && full.latency > 0) ? (1 - full.latency / REF_LAT) : -999)
          const win = Math.max(callWin, latWin)
          const winPct = (x) => (x <= -999 ? 'n/a' : (x * 100).toFixed(1) + '%')
          if (mapOK && win >= LAT_WIN) {
            decision = 'keep-flag-on'
            decisionText = 'KEEP-FLAG-ON (latency win: VLM-call reduction ' + winPct(callWin) + ' / wall-clock ' + winPct(latWin) + ' >= ' + (LAT_WIN * 100) + '%, AND full mAP ' + full.mAP + ' >= accuracy floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '. Flip the config field named in the proposal implementation_plan to its speed-on default; this becomes the new latency baseline.)'
            if (cost && cost.ran) {
              if (cost.mean_vlm_calls > 0) REF_CALLS = cost.mean_vlm_calls
              if (cost.mean_elapsed_s > 0) REF_LAT = cost.mean_elapsed_s
            }
          } else if (!mapOK) {
            decisionText = 'REVERT (ACCURACY GUARD failed: full mAP ' + full.mAP + ' < floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '; a latency change must not drop accuracy. call_win=' + winPct(callWin) + ' wall_win=' + winPct(latWin) + ')'
          } else {
            decisionText = 'REVERT (latency win insufficient: max(VLM-call ' + winPct(callWin) + ', wall-clock ' + winPct(latWin) + ') < ' + (LAT_WIN * 100) + '% required; mAP held at ' + full.mAP + ' - not worth the added complexity)'
          }
        }
      }
    }
  } else {
    log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + '), skipping full, will revert')
  }

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, cost, decisionText, fm5), { schema: resultSchema, label: 'D4-finalize-' + V, phase: 'Implement' }))
  const result = fin || {
    version: V, implemented: true,
    smoke_map: smoke.mAP, full_map: full.mAP, avg_latency: full.latency,
    kept: decision.indexOf('keep') === 0,
    summary: '(D4 finalize agent failed after retries; metrics recorded by orchestrator. Check results/ and iteration_logs/ manually.)',
    lessons: fm5 ? 'FM5: mechanism never activated in vivo' : 'finalize agent failed; see orchestrator logs',
    report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json',
  }
  const costTag = cost && cost.ran ? (' [COST vlm=' + cost.mean_vlm_calls + ' clip=' + cost.mean_clip_calls + ' elapsed=' + cost.mean_elapsed_s + 's]') : ''
  history.push({ loop: loop + 20, version: V, title: proposal.title + costTag, smoke_map: result.smoke_map, full_map: result.full_map, avg_latency: result.avg_latency, kept: result.kept, lessons: result.lessons })
  if (decision === 'keep-flag-on' && result.full_map > 0) {
    log('Loop ' + loop + ': KEPT ' + V + ' (latency win) mAP=' + result.full_map + ' new REF_vlm_calls=' + REF_CALLS + ' REF_wallclock=' + REF_LAT + 's')
  } else if (decision === 'keep') {
    log('Loop ' + loop + ': KEPT ' + V + ' (baseline calibration) REF_vlm_calls=' + REF_CALLS + ' REF_wallclock=' + REF_LAT + 's fresh_mAP=' + result.full_map)
  } else {
    log('Loop ' + loop + ': ' + V + ' decision=' + decision + ' (' + decisionText.slice(0, 120) + ')')
  }
}

phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for this LATENCY-reduction VAD session (v' + START_VERSION + '+). Call StructuredOutput EXACTLY ONCE if a schema is required; else plain text.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. Read ' + CODE + '/iteration_logs/session_v97_ledger.md and the new v' + START_VERSION + '+ idea files.',
'2. date +%Y%m%d.',
'3. Write ' + CODE + '/iteration_logs/summary_loops_v' + START_VERSION + 'plus_latency_<YYYYMMDD>.md: executive summary (baseline vlm_calls/latency vs final kept vlm_calls/latency = total per-video speedup achieved, and whether the mAP floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + ' held throughout), per-version table (mechanism | call reduction % | wall-clock | full mAP | kept), which latency levers worked vs failed and why, updated FM taxonomy (FM1-FM11), and concrete next latency directions.',
'4. Reply with a 10-line plain-text recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('Session complete. Committed baseline stays v114 mAP=' + bestFullMap + '. Latency reference now REF_vlm_calls=' + REF_CALLS + ', REF_wallclock=' + REF_LAT + 's. ' + history.length + ' recorded loops.')
