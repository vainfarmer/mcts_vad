export const meta = {
  name: 'vad-team-map20-v6',
  description: 'VAD agent team v6 (LATENCY-REDUCTION continuation, FM5-FM12-hardened). Continues at v131 after v124-v130. Committed baseline v124: Q-score default-on, full mAP~0.17 (committed 0.15632), REF 31.969 VLM calls / 183s per video (drift-free [COST] instrumentation now always-on). v5 finding: every call-MERGING/BATCHING lever (v97/v126/v129) broke the accuracy floor; every accuracy-safe lever saved <15% (max_scan already binds / gate never fires / local-compute-only). v6 CENTRAL PRINCIPLE (FM12): cut latency ONLY by DROPPING provably-null/redundant WHOLE calls, never by merging/repackaging calls - surviving calls stay byte-identical, so mAP is neutral by construction. Targets the two UNTOUCHED levers: adaptive max_scan (drop provably-below-threshold scan segments) + classify-path pruning (skip qwen3-vl-plus classify when the detect verdict is unambiguous). HARD floor: full mAP >= 0.15132. Win: >=15% drift-free VLM-call reduction. Min 5 new versions.',
  phases: [
    { title: 'Research', detail: 'A and B research drop-not-merge latency levers on the detect-scan and classify paths, each proposes one' },
    { title: 'Debate', detail: 'A and B cross-critique, then synthesize one joint consensus proposal' },
    { title: 'Review', detail: 'C approves or rejects with feedback; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (CLI-activatable + marker), smoke+full eval + [COST] adjudication, keep/revert on the latency+accuracy rule' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const MAX_LOOPS = 6
const START_VERSION = 131
const BASE_LATENCY = 183
const SMOKE_GATE = 0.06
const BASE_MAP = 0.15632           // committed v114/v124 in-vivo baseline
const MAP_TOL = 0.005              // accuracy floor = BASE_MAP - MAP_TOL = 0.15132; a latency change may not drop mAP below it
const LAT_WIN = 0.15               // must cut per-video VLM calls by >= 15% (the drift-free deciding metric)
// Same-session reference is deterministic in call-count, so it is SEEDED from v124 (call count is a property of the
// pipeline config, not API load; wall-clock REF is cross-session and used only as weak corroboration, never to decide).
let REF_CALLS = 31.969             // v124 mean per-video VLM (detect+classify) calls
let REF_LAT = 183.11               // v124 mean per-video wall-clock s (cross-session; corroboration only)

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline; config.py = all params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, llm_client.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up before editing.',
'',
'!!! OBJECTIVE: LATENCY REDUCTION, ACCURACY PROTECTED !!!',
'PRIMARY GOAL: reduce per-video execution time by cutting the DOMINANT cost = VLM API calls (detect qwen-vl-plus + classify qwen3-vl-plus). HARD ACCURACY GUARD: full-eval temporal mAP must stay >= ' + (BASE_MAP - MAP_TOL).toFixed(5) + ' (committed baseline ' + BASE_MAP + '). A latency change that drops mAP below that floor is REVERTED no matter how fast.',
'KEEP RULE: KEEP-FLAG-ON iff (a) full mAP >= ' + (BASE_MAP - MAP_TOL).toFixed(5) + ' AND (b) per-video VLM call count dropped by >= ' + (LAT_WIN * 100) + '% vs REF ' + REF_CALLS + ' (drift-free [COST] metric - THE decider). Wall-clock is corroboration only (FM10). Otherwise REVERT.',
'',
'!!! FM12 - THE CENTRAL PRINCIPLE OF THIS SESSION (learned the hard way in v5) !!!',
'Cut latency ONLY by DROPPING provably-null or redundant WHOLE calls. NEVER by merging, batching, or repackaging calls into a shared payload. RATIONALE: v5 proved that every merge/batch lever (v97 FCCS, v126 run-length merge, v129 4-segment contact-sheet) achieved huge call cuts (-65%, -69%) but CRATERED mAP to ~0.12 because packing/propagating flips per-segment VLM verdicts (FM2/FM3) - per-call independence is load-bearing for mAP. Conversely, a lever that simply does not ISSUE a call it can prove is null/redundant leaves every surviving call byte-identical => mAP-neutral BY CONSTRUCTION. Your proposal MUST keep every surviving VLM call\'s payload and prompt byte-identical to baseline; it may only DECIDE NOT TO MAKE some calls. C must reject any merge/batch/propagation design outright.',
'',
'!!! FM10 - WALL-CLOCK IS API-DRIFT-DOMINATED, NOT THE MECHANISM !!! v124-v130 wall-clock swung 135s-275s on near-identical call counts purely from qwen-VL API load. The DECIDING latency metric is the drift-free per-video VLM CALL COUNT from the [COST] log line (already instrumented, always-on). Report call reduction as the headline; wall-clock only corroborates.',
'',
'THE TWO UNTOUCHED, HIGH-VALUE LEVERS (v5 summary next-directions; both fit FM12):',
'1. ADAPTIVE max_scan (attack the binding ceiling, not the padding): v5 v125/v130 died because max_scan / candidate_top_k already caps common-path VLM cost, so pruning padding saved nothing. The real lever is to SHRINK the scan budget on LOW-PRIOR videos - drop scan segments whose CLIP pre-screen score is PROVABLY below the detection threshold (they could never yield a detection), so the final detection set is unchanged. This drops whole detect calls (FM12-clean) and directly cuts VLM calls.',
'2. CLASSIFY-PATH PRUNING (the banked NOCLS lever): ALL SIX v5 levers targeted the DETECT/scan path; the classify call (qwen3-vl-plus) is a SECOND dominant VLM cost NEVER touched. Skip / short-circuit the classify call when the detect verdict already fixes the category unambiguously (single dominant category, high margin), keeping the detection set and payloads independent (no FM2/FM3). This drops whole classify calls (FM12-clean) and cuts VLM calls.',
'(A calibrated whole-video early-exit gate - the v127 remedy with a less-conservative single-signal threshold that actually fires on the addressable hard-case population - is an acceptable third angle. AVOID adaptive-fps as a headline lever: it cuts frames-per-call not call-count, so it is invisible to the drift-free metric and only shows in noisy wall-clock.)',
'',
'CONFIRMED RESULTS v124-v130 (do NOT re-try; detail in iteration_logs/ideas/loop*_v12*_*.md, loop7_v130, and summary_loops_v124plus_latency_20260726.md):',
'- v124 baseline calibration: KEPT. Completed the Q-score CLI-default flip; added always-on [COST] instrumentation (llm_client.call_count + anomaly_expert.clip_call_count + pipeline emits "[COST] video=<n> vlm_calls=<i> clip_calls=<i> elapsed_s=<f>"). REF: 31.969 VLM / 22.308 CLIP / 183.11s per video; full mAP 0.17076.',
'- v125 PREGATE adaptive CLIP gate on scan padding: REVERT (near-no-op, +0.2% calls; threshold below mean AND max_scan binds).',
'- v126 TRLM run-length merge: REVERT (-65% calls but mAP 0.125 < floor; FM2/FM12 - MERGE breaks accuracy).',
'- v127 CNG-DUAL whole-video normal early-exit (clip_peak<0.5 AND mil_max<0.3): REVERT (0 fires, dual-AND too conservative; accuracy held).',
'- v128 VMW-SKIP prune saturated VideoMAE stage: REVERT (mAP-neutral but local-compute only, -0.6% calls / -12.7% wall < 15%). Local-compute trims are capped below the call metric - do NOT repropose.',
'- v129 BDS 4-segment contact-sheet batch: REVERT (-69% calls but mAP 0.120 < floor; FM2/FM3/FM12 - BATCH breaks accuracy).',
'- v130 CASCADE skip Source-2 padding on high-prior videos: REVERT (high fire-rate but -3.1% calls; max_scan already binds the firing videos; accuracy held).',
'SESSION LESSON: call-MERGING/BATCHING family REFUTED for accuracy; padding-pruning family capped by max_scan binding; local-compute trims below the metric. v6 must DROP whole provably-null calls on the two untouched paths.',
'',
'MANDATORY READING before proposing/reviewing (newest first, skim smart):',
'1. ' + CODE + '/iteration_logs/summary_loops_v124plus_latency_20260726.md  (v5 exec summary + next-directions + FM taxonomy)',
'2. ' + CODE + '/iteration_logs/session_v97_ledger.md  (v124-v130 rows with the [COST] call column)',
'3. ' + CODE + '/iteration_logs/ideas/loop7_v130_*, loop4_v127_*, loop2_v125_*  (the padding/gate near-no-ops you must not repeat)',
'4. PROFILE the code: grep where detect (qwen-vl-plus) and classify (qwen3-vl-plus) VLM calls are issued - pipeline.py, anomaly_discovery.py (_directed_llm_scan, scan budget / max_scan / candidate_top_k), anomaly_expert.py (classify path). Count calls per video and find the classify-call site (never touched).',
'5. Research: ' + ROOT + '/academic-research-skills/deep-research/SKILL.md; then REAL WebSearch on cascade/early-exit inference, confidence-thresholded LLM-call skipping (FrugalGPT/LayerSkip), cheap-signal pre-gating, adaptive budget allocation.',
'',
'KNOWN FAILURE MODES:',
'FM1 curve-shape modification regresses. FM2 detection-count/verdict changes under greedy match make FPs. FM3 VLM self-consistency/context-bleed hurts perceptual verdicts. FM4 self-consistency on inclusion is dangerous. FM5 mechanism never activated (guard: flag + marker + grep gate). FM6 candidate-pool simulation gains do not transfer. FM7 video-level priors cannot fix intra-video mislocation. FM8 offline precheck is config-specific; one controlled change per loop. FM9 MPS tensors need .detach().cpu().numpy(). FM10 wall-clock is API-drift-dominated; decide on drift-free call count. FM11 an offline harness must complete as a pollable step; prefer mechanisms needing none (a drop-call latency lever needs only ONE full eval). FM12 cut calls by DROPPING provably-null whole calls, NEVER by merging/batching/repackaging (breaks verdicts).',
'',
'VALIDATED SAFE ARCHETYPE for THIS session: a config-gated decision to SKIP issuing a VLM call whose result is provably null (scan segment below detection threshold) or redundant (classify when category already unambiguous), leaving every surviving call byte-identical. mAP-neutral by construction => the accuracy floor holds trivially and the win is measured purely by the drift-free VLM-call drop.',
'',
'CURRENT CODE STATE (working tree): committed v124 baseline = v94 pipeline + Q-score re-rank (config.py ScoringConfig.qscore_rerank default True in BOTH dataclass and from_args; --no_qscore_rerank opt-out) + always-on [COST] instrumentation. All v125-v130 mechanisms reverted/removed. Every eval already logs [COST]; --qscore_rerank is now the default (still pass it explicitly for clarity).',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop. ARCHITECTURE-LEVEL only. Reducing a raw budget number counts ONLY if gated by an architectural provably-null test (e.g. "skip segments below the detection threshold"), not blind tuning.',
'- FM12 is non-negotiable: NO merging/batching/verdict-propagation. Surviving calls byte-identical.',
'- All new params via config.py with a switch restoring prior behavior; CLI-activatable + unique INFO marker per fire (FM5).',
'- Keep Q-score default-ON and the v51 half-max boundary mechanism.',
'- Validatable by ONE full eval (mAP + [COST]); no stalling offline harness (FM11).',
'- StructuredOutput EXACTLY ONCE at the end. ANTI-STALL: no Bash call > ~150s; Read with offset/limit; emit a short progress line between tool calls (runtime kills agents silent >180s).',
].join('\n')

const history = [
  { loop: 1, version: 'v124', title: 'Baseline calibration + [COST] instrumentation + Q-score flip', full_map: 0.17076, vlm_calls: 31.969, wall: 183.11, kept: true, lessons: 'KEPT; REF 31.969 VLM / 183s; mAP-neutral; committed baseline' },
  { loop: 2, version: 'v125', title: 'PREGATE adaptive CLIP gate on scan padding', full_map: 0.16802, vlm_calls: 32.046, wall: 275.5, kept: false, lessons: 'REVERT near-no-op (+0.2% calls); max_scan binds; FM10 killed spurious -48% wall win' },
  { loop: 3, version: 'v126', title: 'TRLM run-length MERGE near-dup segments', full_map: 0.12547, vlm_calls: 11.2, wall: 134.7, kept: false, lessons: 'REVERT -65% calls but mAP<floor; FM2/FM12 MERGE breaks accuracy' },
  { loop: 4, version: 'v127', title: 'CNG-DUAL whole-video normal early-exit', full_map: 0.16152, vlm_calls: 31.877, wall: 251.83, kept: false, lessons: 'REVERT 0 fires (dual-AND too conservative); accuracy held' },
  { loop: 5, version: 'v128', title: 'VMW-SKIP prune saturated VideoMAE stage', full_map: 0.17590, vlm_calls: 31.785, wall: 164.9, kept: false, lessons: 'REVERT mAP-neutral but local-compute only (-0.6% call/-12.7% wall < 15%)' },
  { loop: 6, version: 'v129', title: 'BDS 4-segment contact-sheet BATCH', full_map: 0.11980, vlm_calls: 9.769, wall: 189.09, kept: false, lessons: 'REVERT -69% calls but mAP<floor; FM2/FM3/FM12 BATCH breaks accuracy' },
  { loop: 7, version: 'v130', title: 'CASCADE skip Source-2 padding (high-prior)', full_map: 0.17708, vlm_calls: 30.985, wall: 208.6, kept: false, lessons: 'REVERT high fire-rate but -3.1% calls (max_scan binds firing videos); accuracy held' },
]
let bestFullMap = BASE_MAP
let bestVersion = 'v124'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": full mAP=' + h.full_map + ' vlm_calls=' + h.vlm_calls + ' wall=' + h.wall + 's kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED baseline: ' + bestVersion + ' full mAP~0.17 (committed ' + bestFullMap + ', floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '). Latency reference: REF_vlm_calls=' + REF_CALLS + ', REF_wallclock=' + REF_LAT + 's.\nSession history:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  required: ['title', 'mechanism', 'rationale', 'implementation_plan', 'expected_gain', 'risks', 'latency_impact', 'literature'],
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'the architecture-level DROP-CALL latency change, precisely (which whole VLM calls it skips and the provably-null/redundant test)' },
    rationale: { type: 'string', description: 'why it cuts per-video VLM calls while leaving surviving calls byte-identical (mAP-neutral by construction, FM12)' },
    implementation_plan: { type: 'string', description: 'exact files/functions; the CLI flag + unique INFO marker (FM5); must NOT merge/batch calls (FM12) and must NOT need a stalling harness (FM11)' },
    expected_gain: { type: 'string', description: 'expected per-video VLM-call reduction % vs REF ' + REF_CALLS + ' (the drift-free headline) and why mAP holds >= ' + (BASE_MAP - MAP_TOL).toFixed(5) },
    risks: { type: 'string', description: 'which FM1-FM12 it might trigger and why it avoids them; especially FM12 and the accuracy floor' },
    latency_impact: { type: 'string' },
    literature: { type: 'string' },
  },
}
const verdictSchema = { type: 'object', required: ['approve', 'feedback'], properties: { approve: { type: 'boolean' }, feedback: { type: 'string' }, mandated_changes: { type: 'string' } } }
const resultSchema = {
  type: 'object',
  required: ['version', 'implemented', 'smoke_map', 'full_map', 'avg_latency', 'kept', 'summary', 'lessons'],
  properties: {
    version: { type: 'string' }, implemented: { type: 'boolean' }, smoke_map: { type: 'number' }, full_map: { type: 'number' },
    avg_latency: { type: 'number' }, kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' }, report_path: { type: 'string' },
  },
}
const implSchema = {
  type: 'object',
  required: ['implemented', 'files_changed', 'notes', 'eval_flags', 'activation_marker'],
  properties: {
    implemented: { type: 'boolean' }, files_changed: { type: 'string' }, notes: { type: 'string' },
    eval_flags: { type: 'string', description: 'exact CLI args that ACTIVATE the mechanism AND include --qscore_rerank' },
    activation_marker: { type: 'string', description: 'unique literal INFO log substring per fire (FM5)' },
  },
}
const launchSchema = { type: 'object', required: ['launched', 'note'], properties: { launched: { type: 'boolean' }, note: { type: 'string' } } }
const watchSchema = {
  type: 'object',
  required: ['done', 'crashed', 'mAP', 'ap30', 'ap50', 'ap70', 'latency', 'marker_count', 'note'],
  properties: {
    done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' },
    latency: { type: 'number', description: 'avg_per_video_seconds (wall-clock, drift-noisy)' }, marker_count: { type: 'number' }, note: { type: 'string' },
  },
}
const costSchema = {
  type: 'object',
  required: ['ran', 'mean_vlm_calls', 'mean_clip_calls', 'mean_elapsed_s', 'videos_counted', 'note'],
  properties: {
    ran: { type: 'boolean' }, mean_vlm_calls: { type: 'number' }, mean_clip_calls: { type: 'number' }, mean_elapsed_s: { type: 'number' }, videos_counted: { type: 'number' }, note: { type: 'string' },
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
'You are Agent ' + role + ' on a VAD research team. THIS SESSION\'S GOAL: cut per-video VLM API calls by DROPPING provably-null/redundant WHOLE calls (FM12 - never merge/batch), holding mAP >= the floor. ' + focus,
'',
CONTEXT,
'',
'SESSION HISTORY (do NOT repeat; padding-pruning and merge/batch families are refuted):\n' + hist(),
'',
'TASK:',
'1. MANDATORY READING (v5 summary next-directions + the padding/gate near-no-op idea files); then PROFILE the code: grep the detect (qwen-vl-plus) call site in _directed_llm_scan / scan budget (max_scan, candidate_top_k) AND the classify (qwen3-vl-plus) call site (never touched). Count calls/video and identify which whole calls are provably-null or redundant.',
'2. SEARCH-STORM: >=5 real WebSearch queries on cascade/early-exit LLM inference, confidence-thresholded call skipping (FrugalGPT/LayerSkip), cheap-signal pre-gating, adaptive compute budget. WebFetch <=2 small arXiv abs pages.',
'3. Derive ONE architecture-level DROP-CALL idea on the adaptive-max_scan OR classify-pruning path (or a calibrated whole-video early-exit that actually fires): it MUST (a) drop >= ~15% of per-video VLM calls vs REF ' + REF_CALLS + ', (b) leave every surviving call byte-identical => mAP-neutral by construction (FM12), (c) be config-gated + CLI-activatable + emit an INFO marker (FM5), (d) need NO offline harness (FM11). NO merging/batching/propagation. NO blind number tuning.',
'4. Return via StructuredOutput (EXACTLY ONCE). Name exact files/functions, the provably-null test, and the expected call reduction.',
'',
'!!! ANTI-STALL (runtime kills agents silent >180s): prefer WebSearch over WebFetch; Read offset/limit; NEVER sleep or run Bash > ~60s; short progress line before/after each tool call; finish in <15 tool calls.',
  ].join('\n')
}

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
  return [
'You are Agent D-launcher. ONLY job: ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. Do NOT wait. SHORT replies; every Bash call < 9 min. StructuredOutput EXACTLY ONCE.',
'',
'Paths: code=' + CODE + ' ; output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (must appear verbatim): ' + (evalFlags || '(none)'),
'',
'STEPS:',
'1. ps aux | grep -i batch_evaluate | grep -v grep',
'2. If an eval for batch_eval_' + V + '_' + tag + ' is running -> ps -o command -p <pid>; if flags MISSING kill+relaunch; if present launched=true note="adopted".',
'3. If the report exists AND the log shows the required flags (grep the flag; if none required any report is fine) -> launched=true note="exists". Else delete the stale output_dir and relaunch.',
'4. If a DIFFERENT version eval is running -> sleep 120 and re-check up to 3x; still busy -> launched=false note="blocked".',
'5. Else launch exactly (extra flags before the redirect):',
"bash -lc 'source " + CODE + "/.env 2>/dev/null; export HF_ENDPOINT=https://hf-mirror.com; export PYTHONUNBUFFERED=1; cd " + CODE + "; nohup python3 -u batch_evaluate.py --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency 3 --detect_model qwen-vl-plus --classify_model qwen3-vl-plus --fps 2 --clip_model ViT-L-14 --clip_pretrained openai --device mps --mcts_simulations 12 --mcts_max_depth 4 " + (evalFlags || '') + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log (no immediate Traceback). Return: launched, note.',
  ].join('\n')
}

function evalWatchPrompt(V, tag, maxPerCat, evalFlags, marker) {
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + '. SHORT replies; every Bash call < 9 min. Do NOT edit code. StructuredOutput EXACTLY ONCE.',
'',
'Paths: report=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log',
'Marker (FM5): "' + (marker || '(none)') + '" ; required flags: ' + (evalFlags || '(none)'),
'',
'STEPS:',
'1. If the report exists, extract metrics + count markers, return done=true crashed=false:',
"   python3 -c \"import json;d=json.load(open('" + ROOT + "/results/batch_eval_" + V + "_" + tag + "/evaluation_report.json'));t=d['evaluation']['temporal_map'];ti=d['timing'];print(t['mAP'],t.get('AP@0.3'),t.get('AP@0.5'),t.get('AP@0.7'),ti['avg_per_video_seconds'])\"",
'   Map -> mAP, ap30, ap50, ap70, latency. Markers: grep -cF "' + (marker || '') + '" ' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log.',
"2. Else pgrep -f 'batch_evaluate.*batch_eval_" + V + "_" + tag + "' -> if alive: Bash 'sleep 170', re-check; if still absent Bash 'sleep 170' once more; if it appeared extract (done=true); else done=false crashed=false marker_count=-1 note='still running'. No single Bash call > 170s.",
'3. If dead and no report: tail -30 the log. If TRANSIENT (API timeout/ConnectTimeout/rate limit/network) relaunch the exact nohup cmd (max_per_category ' + maxPerCat + ', WITH flags ' + (evalFlags || '(none)') + ') return done=false crashed=false note="relaunched: <cause>". If CODE BUG (Traceback in project code) return done=false crashed=true note=<summary>.',
'Return: done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
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
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap reached' }
}

function costAdjudicatePrompt(V, tag) {
  return [
'You are Agent D-cost for the ' + tag + ' eval of ' + V + '. Parse the drift-free per-video cost from [COST] log lines. Do NOT edit code. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'',
'Log: ' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log',
'Each processed video emits: [COST] video=<name> vlm_calls=<int> clip_calls=<int> elapsed_s=<float>',
'',
'Run one fast python one-liner to aggregate:',
"python3 -c \"import re;xs=[re.search(r'vlm_calls=(\\\\d+).*clip_calls=(\\\\d+).*elapsed_s=([0-9.]+)',l) for l in open('" + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log')];xs=[m for m in xs if m];import statistics as s;print(len(xs), s.mean([int(m.group(1)) for m in xs]) if xs else -1, s.mean([int(m.group(2)) for m in xs]) if xs else -1, s.mean([float(m.group(3)) for m in xs]) if xs else -1)\"",
'The 4 numbers = videos_counted, mean_vlm_calls, mean_clip_calls, mean_elapsed_s. If field names differ, grep one raw [COST] line and adapt. ZERO [COST] lines => ran=false with a note.',
'Return: ran, mean_vlm_calls, mean_clip_calls, mean_elapsed_s, videos_counted, note.',
  ].join('\n')
}

function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1, the implementer. Implement the approved DROP-CALL latency proposal. DO NOT run evals. StructuredOutput EXACTLY ONCE.',
'',
CONTEXT,
'',
'APPROVED PROPOSAL FOR ' + V + ':\n' + JSON.stringify(proposal, null, 1),
'',
"C's feedback + mandated changes:\n" + (verdict ? ((verdict.feedback || '') + '\n' + (verdict.mandated_changes || '')) : '(none)'),
'',
'RULES:',
'- CONTEXT-EFFICIENT: Grep to locate code; Read offset/limit; never cat logs / whole large files.',
'- Backup BEFORE editing: mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and cp every file you will modify.',
'- A prior attempt may have partially implemented this; inspect first, complete what is missing.',
'- FM12: the mechanism may only DECIDE NOT TO ISSUE some VLM calls; every surviving call\'s payload/prompt MUST be byte-identical to baseline. NO merging/batching/propagation.',
'- FM5: activatable via a real CLI flag on batch_evaluate.py (argparse + VADConfig.from_args getattr; verify python3 batch_evaluate.py --help | grep your-flag) OR default-ON; emit a unique INFO marker per fire (log which calls it skipped, e.g. "[NOCLS] video=.. skipped_classify=N" or "[ASCAN] video=.. dropped_segments=N").',
'- FM9: any mps tensor -> .detach().cpu().numpy() first.',
'- All new params via config.py with a switch restoring prior behavior. Update the pipeline.py docstring header with a ' + V + ' entry.',
'- Sanity: python3 -c "import ast; ast.parse(open(\'<f>\').read())" per edited file; then cd ' + CODE + ' && python3 -c "import pipeline".',
'Return: implemented, files_changed, notes, eval_flags (activate the mechanism AND include --qscore_rerank), activation_marker (literal INFO substring).',
  ].join('\n')
}

function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, cost, decisionText, fm5) {
  return [
'You are Agent D4, the finalizer for ' + V + '. Implementation + evals DONE. Enforce keep/revert, save artifacts, write report, return summary. Bash < 9 min. StructuredOutput EXACTLY ONCE.',
'',
'Paths: code=' + CODE + ' ; results=' + ROOT + '/results ; backups=' + ROOT + '/.qoder/backups/' + V + ' ; ideas=' + CODE + '/iteration_logs/ideas/ ; runlogs=' + CODE + '/iteration_logs/results/',
'',
'DECISION: ' + decisionText + (fm5 ? '  [FM5: mechanism never activated - 0 markers]' : ''),
'Metrics: smoke mAP=' + smoke.mAP + ' (markers=' + smoke.marker_count + ') ; full mAP=' + full.mAP + ' ; AP@0.3=' + full.ap30 + ' AP@0.5=' + full.ap50 + ' AP@0.7=' + full.ap70 + ' ; wall-clock=' + full.latency + 's',
'COST (drift-free): mean_vlm_calls=' + (cost ? cost.mean_vlm_calls : '?') + ' mean_clip_calls=' + (cost ? cost.mean_clip_calls : '?') + ' mean_elapsed_s=' + (cost ? cost.mean_elapsed_s : '?') + ' over ' + (cost ? cost.videos_counted : '?') + ' videos ; REF_vlm_calls=' + REF_CALLS + ' ; floor=' + (BASE_MAP - MAP_TOL).toFixed(5),
'Eval flags: ' + (impl.eval_flags || '(none)') + ' ; marker: "' + (impl.activation_marker || '(none)') + '"',
'',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
'',
"C's feedback:\n" + (verdict ? (verdict.feedback || '') : '(none)'),
'',
'D1 notes (files: ' + (impl.files_changed || '?') + '):\n' + (impl.notes || '(none)'),
'',
'TASKS:',
'1. If decision=REVERT: restore every file in ' + ROOT + '/.qoder/backups/' + V + '/ into ' + CODE + '/ and verify each with diff (identical). If no backup dir, ensure the mechanism config stays default-OFF. If decision starts KEEP-FLAG-ON: flip the config field named in the decision text to its speed-on default (new baseline) and sanity-import pipeline.',
'2. Copy artifacts: cp ' + ROOT + '/results/batch_eval_' + V + '_smoke.log ' + CODE + '/iteration_logs/results/' + V + '_smoke_run.log ; cp ' + ROOT + '/results/batch_eval_' + V + '_smoke/evaluation_report.json ' + CODE + '/iteration_logs/results/' + V + '_smoke_eval.json ; same two for _full if it ran.',
'3. date +%Y%m%d, then write ' + CODE + '/iteration_logs/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md: proposal, C feedback, impl notes, flags+marker+marker counts, results table (smoke mAP; full mAP + AP@0.3/0.5/0.7; wall-clock; and drift-free mean_vlm_calls / mean_clip_calls / mean_elapsed_s vs REF ' + REF_CALLS + '), keep/revert decision + reason (VLM-call reduction % AND whether the accuracy floor held), lessons (FMs confirmed/refuted).',
'4. Append one row to ' + CODE + '/iteration_logs/session_v97_ledger.md: | loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept | (put mean_vlm_calls and call-reduction % in the title cell).',
'5. Return: version="' + V + '", implemented=true, smoke_map=' + smoke.mAP + ', full_map=' + full.mAP + ', avg_latency=' + full.latency + ', kept=' + (decisionText.indexOf('KEEP') === 0) + ', summary (5-8 sentences incl. call-count delta), lessons, report_path.',
  ].join('\n')
}

log('VAD agent team v6 (LATENCY, drop-not-merge) starting. Baseline v124, floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + ', win gate ' + (LAT_WIN * 100) + '% VLM-call reduction vs REF ' + REF_CALLS + '. Versions v' + START_VERSION + '+, max ' + MAX_LOOPS + ' loops. Levers: adaptive max_scan + classify-path pruning.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION - 1 + loop)
  log('=== Loop ' + loop + ' (' + V + ') ===')

  phase('Research')
  const [propA, propB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', 'Your lever: ADAPTIVE max_scan / scan-budget on the DETECT path. Shrink the number of scan-segment detect calls on low-prior videos by DROPPING scan segments whose CLIP pre-screen score is PROVABLY below the detection threshold (they can never become detections), leaving the surviving detect calls byte-identical. Attack the binding max_scan ceiling itself, not the padding (v125/v130 failed on padding).'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'Your lever: CLASSIFY-PATH pruning (the untouched qwen3-vl-plus call). Skip / short-circuit the classify VLM call when the detect verdict already fixes the category unambiguously (single dominant category, high margin, or category recoverable from the detect output), keeping every surviving call independent (no FM2/FM3). Profile anomaly_expert / pipeline to find the classify call site and count classify calls per video.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!propA && !propB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }
  const pA = propA ? JSON.stringify(propA, null, 1) : '(A failed)'
  const pB = propB ? JSON.stringify(propB, null, 1) : '(B failed)'

  phase('Debate')
  const debateBase = [CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '', "Proposal from A (adaptive max_scan):\n" + pA + "\n\nProposal from B (classify pruning):\n" + pB].join('\n')
  const [critAonB, critBonA] = await parallel([
    () => retryAgent(() => agent('You are Agent A continuing the debate. ' + debateBase + '\n\nTASK: Critique B\'s classify-pruning proposal: (a) real per-video VLM-call reduction magnitude? (b) is it FM12-clean (drops whole calls, no merge) and mAP-neutral by construction? (c) does skipping classify ever change the final category/detection? (d) FM5/FM9? Then accept/reject/merge - exactly how. <400 words.', { label: 'A-critiques-B-' + V, phase: 'Debate' })),
    () => retryAgent(() => agent('You are Agent B continuing the debate. ' + debateBase + '\n\nTASK: Critique A\'s adaptive-max_scan proposal: (a) does it drop enough detect calls given max_scan already binds? (b) is the "provably below detection threshold" test SOUND (dropped segments truly could never become detections => mAP-neutral)? (c) FM12-clean, no merge? (d) FM5? Then accept/reject/merge - exactly how. <400 words.', { label: 'B-critiques-A-' + V, phase: 'Debate' })),
  ])

  const consensus = await retryAgent(() => agent([
'You are the joint A+B consensus. Reach the SINGLE joint optimal DROP-CALL latency proposal both would sign.',
'', debateBase, '',
"A's critique of B:\n" + (critAonB || '(missing)'), '',
"B's critique of A:\n" + (critBonA || '(missing)'), '',
'TASK: Synthesize ONE final proposal (pick the stronger of the two levers, or ONE controlled change - never two). Must cut per-video VLM calls >= ~15% vs REF ' + REF_CALLS + ', be mAP-neutral by construction (FM12: drop whole provably-null/redundant calls, surviving calls byte-identical), architecture-level, CLI-activatable with an INFO marker, no offline harness. Return via StructuredOutput (EXACTLY ONCE, proposal schema).',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  if (!consensus) { log('Loop ' + loop + ': consensus failed, skipping'); continue }

  phase('Review')
  let proposal = consensus
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the skeptical reviewer. Gatekeep what reaches implementation. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'JOINT LATENCY PROPOSAL (attempt ' + (r + 1) + '):\n' + JSON.stringify(proposal, null, 1),
'',
'REVIEW CHECKLIST (strict):',
'1. Does it genuinely cut per-video VLM calls by a plausible >= 15% vs REF ' + REF_CALLS + '? Not capped by max_scan already binding (v125/v130 lesson)?',
'2. FM12 GUARD: does it ONLY drop whole provably-null/redundant calls, leaving surviving calls byte-identical? REJECT outright any merge/batch/verdict-propagation (v126/v129 breached the floor).',
'3. Is it mAP-neutral BY CONSTRUCTION (dropped calls provably could not have changed the detection set)? If it can change detections, REJECT or demand the provably-null argument.',
'4. Architecture-level, not blind budget tuning?',
'5. Avoids FM1-FM11? FM11: validatable by ONE full eval, no stalling harness?',
'6. FM5: states the CLI flag (or default-on) + unique INFO marker logging which calls were skipped? FM9: mps->numpy safe? If missing, REJECT and demand it.',
'If flawed/accuracy-risky/merge-based, REJECT with concrete feedback. If good but imperfect, APPROVE with mandated_changes. Return via StructuredOutput (EXACTLY ONCE).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (round ' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. Reviewer C REJECTED your proposal. Revise into a NEW or FIXED DROP-CALL proposal addressing every point (you may pivot between the max_scan and classify levers). Still ONE controlled change, FM12-clean, cutting VLM calls while holding mAP. StructuredOutput EXACTLY ONCE.',
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
        implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED CHANGES:\n' + verdict.mandated_changes,
      })
      log('Loop ' + loop + ': proceeding with C-mandated revision')
    } else {
      log('Loop ' + loop + ': C rejected 3x without mandate, skipping implementation')
      history.push({ loop: loop + 20, version: V, title: proposal.title + ' [REJECTED - no implementation]', full_map: -1, vlm_calls: -1, wall: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
      continue
    }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented) {
    log('Loop ' + loop + ': D1 implementation failed')
    history.push({ loop: loop + 20, version: V, title: proposal.title + ' [D1 FAILED]', full_map: -1, vlm_calls: -1, wall: -1, kept: false, lessons: 'implementation failed: ' + (impl && impl.notes ? impl.notes.slice(0, 200) : 'agent null') })
    continue
  }
  if (!impl.activation_marker) {
    log('Loop ' + loop + ': D1 no activation marker - FM5 risk, treating as failure')
    history.push({ loop: loop + 20, version: V, title: proposal.title + ' [NO MARKER - FM5]', full_map: -1, vlm_calls: -1, wall: -1, kept: false, lessons: 'D1 did not provide an activation marker' })
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
      log('Loop ' + loop + ': FM5 - marker "' + impl.activation_marker + '" absent from smoke; skipping full, will revert')
    } else {
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' wall=' + full.latency + ' markers=' + full.marker_count + ' | ' + (full.note || '').slice(0, 150))
      if (impl.activation_marker && full.done && full.marker_count <= 0) {
        fm5 = true
        log('Loop ' + loop + ': FM5 - marker absent from full log')
      }
      if (!fm5 && full.done && !full.crashed) {
        cost = await retryAgent(() => agent(costAdjudicatePrompt(V, 'full'), { schema: costSchema, label: 'D-cost-' + V, phase: 'Implement' }))
        log('Loop ' + loop + ': cost ran=' + (cost ? cost.ran : 'null') + ' vlm=' + (cost ? cost.mean_vlm_calls : '?') + ' clip=' + (cost ? cost.mean_clip_calls : '?') + ' elapsed=' + (cost ? cost.mean_elapsed_s : '?') + ' n=' + (cost ? cost.videos_counted : '?'))
        const mapOK = full.mAP >= (BASE_MAP - MAP_TOL)
        const callWin = (REF_CALLS > 0 && cost && cost.ran && cost.mean_vlm_calls > 0) ? (1 - cost.mean_vlm_calls / REF_CALLS) : -999
        const latWin = (REF_LAT > 0 && cost && cost.ran && cost.mean_elapsed_s > 0) ? (1 - cost.mean_elapsed_s / REF_LAT) : -999
        const winPct = (x) => (x <= -999 ? 'n/a' : (x * 100).toFixed(1) + '%')
        // Decision is on the DRIFT-FREE VLM-call reduction only (FM10); wall-clock is logged as corroboration.
        if (mapOK && callWin >= LAT_WIN) {
          decision = 'keep-flag-on'
          decisionText = 'KEEP-FLAG-ON (drift-free VLM-call reduction ' + winPct(callWin) + ' >= ' + (LAT_WIN * 100) + '% [wall-clock corroboration ' + winPct(latWin) + '], AND full mAP ' + full.mAP + ' >= floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '. Flip the config field named in the proposal implementation_plan to its speed-on default; this becomes the new latency baseline.)'
          if (cost && cost.ran) { if (cost.mean_vlm_calls > 0) REF_CALLS = cost.mean_vlm_calls; if (cost.mean_elapsed_s > 0) REF_LAT = cost.mean_elapsed_s }
        } else if (!mapOK) {
          decisionText = 'REVERT (ACCURACY GUARD failed: full mAP ' + full.mAP + ' < floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + '. VLM-call reduction was ' + winPct(callWin) + ' - a latency change must not drop accuracy.)'
        } else {
          decisionText = 'REVERT (insufficient VLM-call reduction: ' + winPct(callWin) + ' < ' + (LAT_WIN * 100) + '% required [wall-clock ' + winPct(latWin) + ' is drift-noise, not decisive]; mAP held at ' + full.mAP + '.)'
        }
      }
    }
  } else {
    log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + '), skipping full, will revert')
  }

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, cost, decisionText, fm5), { schema: resultSchema, label: 'D4-finalize-' + V, phase: 'Implement' }))
  const result = fin || {
    version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, avg_latency: full.latency,
    kept: decision.indexOf('keep') === 0,
    summary: '(D4 finalize agent failed after retries; metrics recorded by orchestrator. Check results/ and iteration_logs/.)',
    lessons: fm5 ? 'FM5: mechanism never activated' : 'finalize agent failed; see orchestrator logs',
    report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json',
  }
  const cv = cost && cost.ran ? cost.mean_vlm_calls : -1
  const costTag = cost && cost.ran ? (' [COST vlm=' + cost.mean_vlm_calls + ' clip=' + cost.mean_clip_calls + ' elapsed=' + cost.mean_elapsed_s + 's]') : ''
  history.push({ loop: loop + 20, version: V, title: proposal.title + costTag, full_map: result.full_map, vlm_calls: cv, wall: result.avg_latency, kept: result.kept, lessons: result.lessons })
  if (decision === 'keep-flag-on' && result.full_map > 0) {
    log('Loop ' + loop + ': KEPT ' + V + ' (latency win) mAP=' + result.full_map + ' new REF_vlm_calls=' + REF_CALLS + ' (was 31.969)')
  } else {
    log('Loop ' + loop + ': ' + V + ' decision=' + decision + ' (' + decisionText.slice(0, 120) + ')')
  }
}

phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for this LATENCY continuation (v' + START_VERSION + '+). StructuredOutput EXACTLY ONCE if a schema is required; else plain text.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. Read ' + CODE + '/iteration_logs/session_v97_ledger.md and the new v' + START_VERSION + '+ idea files.',
'2. date +%Y%m%d.',
'3. Write ' + CODE + '/iteration_logs/summary_loops_v' + START_VERSION + 'plus_latency_<YYYYMMDD>.md: executive summary (starting REF 31.969 VLM calls / 183s vs final kept VLM calls = total per-video VLM-call reduction achieved, and whether the mAP floor ' + (BASE_MAP - MAP_TOL).toFixed(5) + ' held), per-version table (mechanism | VLM-call reduction % | full mAP | kept), whether the adaptive-max_scan and classify-pruning levers worked and why, updated FM taxonomy (FM1-FM12), and concrete next directions.',
'4. Reply with a 10-line plain-text recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('Session complete. Committed baseline v124 mAP~0.17. Latency reference now REF_vlm_calls=' + REF_CALLS + ' (started 31.969), REF_wallclock=' + REF_LAT + 's. ' + history.length + ' recorded loops.')
