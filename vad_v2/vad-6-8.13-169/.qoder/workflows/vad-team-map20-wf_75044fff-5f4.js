export const meta = {
  name: 'vad-team-map20',
  description: 'VAD agent team: A/B research+debate, C skeptical review, D implement+eval. Iterates the UCF-Crime VAD pipeline until full-eval temporal mAP>=0.20 (min 15 loops, versions v97+), minimizing per-video latency.',
  phases: [
    { title: 'Research', detail: 'A and B independently research papers + codebase, each proposes one architecture-level idea' },
    { title: 'Debate', detail: 'A and B cross-critique, then synthesize one joint consensus proposal' },
    { title: 'Review', detail: 'C approves or rejects with feedback; up to 2 revision cycles' },
    { title: 'Implement', detail: 'D implements, runs smoke + full eval, saves versioned logs and report' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const TARGET = 0.2
const MAX_LOOPS = 25
const START_VERSION = 97
const BASE_LATENCY = 128

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision: video-level labels for training; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline, versioned docstring header; config.py centralizes ALL params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code (post-v96). NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up before editing.',
'',
'CURRENT BASELINE (full 65-video eval): best temporal mAP = 0.1494 (v89/v94), AP@0.5 = 0.1335, AP@0.7 = 0.1096 (best ever), frame AUC ~0.69-0.71, avg latency ~128 s/video at concurrency 3.',
'TARGET: full-eval temporal mAP >= 0.20 while keeping per-video latency near or below ~128s.',
'VLM API stochasticity is about +/-0.03 mAP: deltas within noise are NOT real improvements.',
'',
'MANDATORY READING before proposing or reviewing (skim smartly, newest first):',
'1. ' + CODE + '/iteration_logs/summary_loops_v90-94_20260712.md  (failure-mode taxonomy, 4 modes)',
'2. ' + CODE + '/iteration_logs/summary_loops1-10_v86-89_20260712.md',
'3. ' + CODE + '/iteration_logs/summary_loops8-17_20260710.md',
'4. ' + CODE + '/iteration_logs/summary_loops4-7_20260709.md',
'5. ' + ROOT + '/.qoder/specs/VAD_mAP_50%_优化方案_c990fbf7.md  (long-term roadmap + rejected alternatives)',
'6. ' + CODE + '/iteration_logs/ideas/*.md  (per-loop idea files; read the newest ~8 fully, skim older)',
'7. Research methodology: ' + ROOT + '/academic-research-skills/deep-research/SKILL.md (+ references/ dir) - then run REAL WebSearch/WebFetch queries (arXiv, PapersWithCode, CVF openaccess).',
'',
'KNOWN FAILURE MODES (validated across v50-v96, 20+ versions):',
'FM1 curve-shape modification (fusion weights, feature re-centering, confusion-negative scoring, candidate pooling) regresses even when the modifying signal is individually better.',
'FM2 detection-count changes under greedy-match evaluation (e.g. Soft-NMS) turn near-duplicates into FPs.',
'FM3 VLM self-consistency via temperature diversity hurts perceptual/extraction tasks (reading timestamps).',
'FM4 self-consistency on segment-inclusion decisions is dangerous; on already-fixed detection labels (category tie-break) it is safe.',
'Also known: MIL v3/v4 LGT-Adapter collapsed (75-83% of videos); mAP ~0.14 is a STRUCTURAL ceiling for incremental tuning - only ARCHITECTURE-LEVEL changes can break it.',
'',
'VALIDATED SAFE PATTERNS: confidence-only modulation computed read-only from existing curves (v95 TSM sharpness); category tie-break on fixed detections (v89); learned prompt-ensemble weights strictly upstream of fusion/boundary/candidate logic (v94).',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop (controlled experiment).',
'- ARCHITECTURE-LEVEL ideas only - NO pure parameter/threshold tuning.',
'- All new params go through config.py with a switch restoring prior behavior.',
'- Keep the v51 half-max boundary-crossing mechanism unless a strong literature-backed argument says otherwise.',
'- Latency matters: prefer zero-cost (precomputed text embeddings) or training-free signals; avoid extra per-video VLM calls.',
].join('\n')

const EVAL_PROTOCOL = [
'EVAL PROTOCOL (follow exactly):',
'1) Check no other eval is running: ps aux | grep -i batch_evaluate | grep -v grep  (must be empty; never run two evals concurrently).',
'2) Launch evals with nohup in background (Bash tool calls themselves must NEVER exceed 9 minutes):',
'SMOKE (13-14 videos, ~15-25 min):',
"bash -lc 'source " + CODE + "/.env 2>/dev/null; export HF_ENDPOINT=https://hf-mirror.com; export PYTHONUNBUFFERED=1; cd " + CODE + "; nohup python3 -u batch_evaluate.py --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_{V}_smoke --max_per_category 1 --concurrency 3 --detect_model qwen-vl-plus --classify_model qwen3-vl-plus --fps 2 --clip_model ViT-L-14 --clip_pretrained openai --device mps --mcts_simulations 12 --mcts_max_depth 4 > " + ROOT + "/results/batch_eval_{V}_smoke.log 2>&1 & echo PID $!'",
'FULL (65 videos, ~50-70 min): identical but --max_per_category 5 and output_dir/log named {V}_full.',
"3) Poll in a loop, each Bash call < 9 min: sleep 480; tail -3 <logfile>; ls <output_dir>/evaluation_report.json  -> repeat until evaluation_report.json exists (or process died: then read the log tail, diagnose, retry once).",
"4) Extract metrics: python3 -c \"import json;d=json.load(open('<output_dir>/evaluation_report.json'));print(d['evaluation']['temporal_map']);print(d['timing'])\"",
'   mAP = evaluation.temporal_map.mAP ; latency = timing.avg_per_video_seconds.',
].join('\n')

const proposalSchema = {
  type: 'object',
  required: ['title', 'mechanism', 'rationale', 'implementation_plan', 'expected_gain', 'risks', 'latency_impact', 'literature'],
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'the architecture-level change, precisely' },
    rationale: { type: 'string', description: 'why it should break the 0.14 ceiling; evidence from papers + this codebase' },
    implementation_plan: { type: 'string', description: 'exact files/functions to change, step by step' },
    expected_gain: { type: 'string' },
    risks: { type: 'string', description: 'which known failure modes it might trigger and why it avoids them' },
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

const history = []
let bestFullMap = 0.1494
let bestVersion = 'v94'

function hist() {
  if (!history.length) return 'No loops completed yet this session. Current best: ' + bestVersion + ' full mAP=' + bestFullMap + '.'
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": smoke=' + h.smoke_map + ' full=' + h.full_map + ' latency=' + h.avg_latency + 's kept=' + h.kept + ' | lessons: ' + h.lessons)
  return 'Current best this session + history: best=' + bestVersion + ' full mAP=' + bestFullMap + '\n' + lines.join('\n')
}

function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. ' + focus,
'',
CONTEXT,
'',
'SESSION HISTORY (do NOT repeat already-tried ideas):\n' + hist(),
'',
'TASK:',
'1. Do the MANDATORY READING (item 6 ideas/ and summaries are the highest value; then skim pipeline.py docstring + config.py + the 2-3 most relevant source files for your angle).',
'2. SEARCH-STORM: run at least 6 real WebSearch queries on 2023-2026 papers about TEMPORAL LOCALIZATION mechanisms (how exactly do they localize: proposal generation, boundary regression, start/end probability curves, Gaussian modeling, MIL snippet scoring, temporal contrast, motion/optical-flow signals, training-free localization). Use WebFetch on the most promising 2-3 papers (arXiv abs/html pages) to extract their exact localization mechanism.',
'3. From the literature + codebase evidence, derive ONE architecture-level idea that could break the 0.149 structural ceiling. It must respect the HARD CONSTRAINTS and avoid all 4 failure modes. Parameter tuning is FORBIDDEN.',
'4. Return your proposal via StructuredOutput. Be concrete: name the exact files/functions to change.',
].join('\n')
}

log('VAD agent team starting. Target mAP>=' + TARGET + ', versions v' + START_VERSION + '+, max ' + MAX_LOOPS + ' loops.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION - 1 + loop)
  log('=== Loop ' + loop + ' (' + V + ') ===')

  phase('Research')
  const [propA, propB] = await parallel([
    () => agent(researchPrompt('A', 'Your specialty: TEMPORAL LOCALIZATION ARCHITECTURES from the TAL/VAD literature (BSN/BMN/DBG/ActionFormer-style proposal+boundary mechanisms, VadCLIP/PE-MIL/MGFN/AnomalyCLIP/GS-MoE/FreeZAD-style VAD mechanisms) and how their localization core can be adapted training-free or light-training into this pipeline.'), { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' }),
    () => agent(researchPrompt('B', 'Your specialty: CODEBASE-FIT + BOTTLENECK analysis. Known bottlenecks: 28.8% of candidates at completely wrong locations (low-CLIP categories Shoplifting/Stealing/Arson/Burglary have flat CLIP curves), 39% of detections have IoU 0.1-0.5 (boundary imprecision), MIL temporal discrimination collapsed. You focus on signals the pipeline underuses (motion energy, temporal gradient structure, MIL v6, snippet resolution, multi-signal boundary evidence) and on recent weakly-supervised VAD SOTA papers for transferable mechanisms.'), { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' }),
  ])
  if (!propA && !propB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue
  }
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
    () => agent('You are Agent A continuing the debate. ' + debateBase + '\n\nTASK: Critique B\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM4 or repeat a tried/failed version? (b) is it truly architecture-level? (c) is the expected gain plausible given +/-0.03 noise? (d) latency risk? Then state: would you accept B\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'A-critiques-B-' + V, phase: 'Debate' }),
    () => agent('You are Agent B continuing the debate. ' + debateBase + '\n\nTASK: Critique A\'s proposal ruthlessly but constructively: (a) does it violate any known failure mode FM1-FM4 or repeat a tried/failed version? (b) is it truly architecture-level? (c) is the expected gain plausible given +/-0.03 noise? (d) latency risk? Then state: would you accept A\'s proposal over yours, reject it, or merge - and exactly how. Reply in <400 words.', { label: 'B-critiques-A-' + V, phase: 'Debate' }),
  ])

  const consensus = await agent([
'You are the joint A+B consensus. Two researchers proposed ideas and cross-critiqued. Reach the SINGLE joint optimal proposal both would sign.',
'',
debateBase,
'',
"A's critique of B:\n" + (critAonB || '(missing)'),
'',
"B's critique of A:\n" + (critBonA || '(missing)'),
'',
'TASK: Synthesize ONE final proposal: pick the stronger idea, or merge both into one coherent SINGLE controlled change (never two independent changes). It must be architecture-level, respect all HARD CONSTRAINTS, avoid FM1-FM4, and not repeat anything in SESSION HISTORY. Return via StructuredOutput (same schema as the proposals).',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' })
  if (!consensus) { log('Loop ' + loop + ': consensus failed, skipping'); continue }

  phase('Review')
  let proposal = consensus
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await agent([
'You are Agent C, the skeptical reviewer of this VAD team. You gatekeep what reaches implementation.',
'',
CONTEXT,
'',
'SESSION HISTORY:\n' + hist(),
'',
'JOINT PROPOSAL FROM A+B (attempt ' + (r + 1) + '):\n' + JSON.stringify(proposal, null, 1),
'',
'REVIEW CHECKLIST (be strict but fair):',
'1. Architecture-level, not disguised parameter tuning?',
'2. Avoids FM1-FM4 and all previously failed/reverted approaches in the history and idea logs?',
'3. Expected gain plausible vs +/-0.03 VLM noise? Is the mechanism causal, not wishful?',
'4. Latency impact acceptable (target <= ~' + BASE_LATENCY + 's/video, hard limit +15%)?',
'5. Implementation risk: reversible, config-gated, one controlled change?',
'6. Is it sufficiently BETTER than doing nothing - does it attack a real bottleneck (wrong candidate locations / boundary imprecision / MIL collapse)?',
'If the idea is fundamentally flawed or too weak, REJECT with concrete feedback. If good but imperfect, you may APPROVE and still give feedback, or approve with mandated_changes. Return via StructuredOutput.',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' })
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (round ' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
    if (r === 2) break
    proposal = await agent([
'You are the joint A+B team. Reviewer C REJECTED your proposal. Revise it into a NEW or FIXED proposal addressing every point of C\'s feedback. You may pivot to a different mechanism if C\'s critique is fundamental. Still one controlled architecture-level change, respecting all constraints.',
'',
CONTEXT,
'',
'SESSION HISTORY:\n' + hist(),
'',
'REJECTED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
'',
"C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED CHANGES:\n' + verdict.mandated_changes : ''),
'',
'Return the revised proposal via StructuredOutput.',
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' })
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
      history.push({ loop: loop, version: V, title: proposal.title + ' [REJECTED - no implementation]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
      continue
    }
  }

  phase('Implement')
  const result = await agent([
'You are Agent D, the implementer + evaluator of this VAD team. You receive an approved proposal and must implement it, evaluate it, and produce the version report.',
'',
CONTEXT,
'',
EVAL_PROTOCOL,
'',
'SESSION HISTORY:\n' + hist(),
'',
'APPROVED PROPOSAL FOR ' + V + ':\n' + JSON.stringify(proposal, null, 1),
'',
"C's review feedback (respect it):\n" + (verdict ? verdict.feedback : '(none)'),
'',
'WORKFLOW (follow exactly):',
'1. Get today\'s date: date +%Y%m%d (use it in file names).',
'2. BACKUP: mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy EVERY file you will modify there BEFORE editing (preserve relative names).',
'3. IMPLEMENT the proposal faithfully. All new params via config.py with a switch to restore prior behavior. Update the pipeline.py docstring header with a concise ' + V + ' entry (follow the existing version-note convention).',
'4. Sanity-check: python3 -c "import ast; ast.parse(open(\'<each edited file>\').read())" and a quick import check: cd ' + CODE + ' && python3 -c "import pipeline".',
'5. Run SMOKE eval per EVAL PROTOCOL with {V}=' + V + '.',
'6. SMOKE GATE: if smoke mAP < 0.11 (baseline smoke ~0.15, noise +/-0.03) or the run crashed -> REVERT: restore all files from the backup dir, verify with diff, skip full eval, and report full_map=-1, kept=false.',
'7. Otherwise run FULL eval per EVAL PROTOCOL.',
'8. KEEP/REVERT DECISION: KEEP only if full mAP >= ' + (bestFullMap - 0.005).toFixed(4) + ' AND avg_per_video_seconds <= ' + Math.round(BASE_LATENCY * 1.15) + 's. Otherwise REVERT from backup and verify with diff. (Current best kept version: ' + bestVersion + ' mAP=' + bestFullMap + '.)',
'9. SAVE ARTIFACTS (regardless of keep/revert):',
'   - cp ' + ROOT + '/results/batch_eval_' + V + '_smoke.log ' + CODE + '/iteration_logs/results/' + V + '_smoke_run.log ; cp the smoke evaluation_report.json to ' + CODE + '/iteration_logs/results/' + V + '_smoke_eval.json',
'   - same for full: ' + V + '_full_run.log and ' + V + '_full_eval.json (if full was run)',
'   - Write the idea+report file: ' + CODE + '/iteration_logs/ideas/loop' + loop + '_' + V + '_<kebab-case-slug>_<YYYYMMDD>.md containing: the approved proposal, C\'s feedback, implementation notes, smoke+full results table (mAP, AP@0.3/0.5/0.7, latency), keep/revert decision + reason, and lessons learned (which failure modes confirmed/refuted).',
'   - Append one line to ' + CODE + '/iteration_logs/session_v97_ledger.md (create with a markdown table header if missing): | loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept |',
'10. Return the final result via StructuredOutput.',
].join('\n'), { schema: resultSchema, label: 'D-implement-' + V, phase: 'Implement' })

  if (!result) {
    log('Loop ' + loop + ': D failed/returned nothing')
    history.push({ loop: loop, version: V, title: proposal.title + ' [D FAILED]', smoke_map: -1, full_map: -1, avg_latency: -1, kept: false, lessons: 'implementation agent failed' })
    continue
  }
  history.push({ loop: loop, version: V, title: proposal.title, smoke_map: result.smoke_map, full_map: result.full_map, avg_latency: result.avg_latency, kept: result.kept, lessons: result.lessons })
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
await agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for this VAD agent-team session (versions v' + START_VERSION + '+).',
'',
CONTEXT,
'',
'FULL SESSION HISTORY:\n' + hist(),
'',
'TASK:',
'1. Read ' + CODE + '/iteration_logs/session_v97_ledger.md and the new idea files in ' + CODE + '/iteration_logs/ideas/ from this session (loop*_v' + START_VERSION + '* and later versions).',
'2. Get today\'s date via: date +%Y%m%d',
'3. Write ' + CODE + '/iteration_logs/summary_loops_v' + START_VERSION + 'plus_<YYYYMMDD>.md in the SAME style as summary_loops_v90-94_20260712.md: executive summary (final best mAP, whether target 0.20 was reached), per-version what-was-tried table with results, updated failure-mode taxonomy (any new modes discovered), latency analysis, and concrete next directions.',
'4. Reply with a 10-line plain-text recap of the session outcome.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' })

log('Session complete. Best: ' + bestVersion + ' full mAP=' + bestFullMap + ' over ' + history.length + ' loops.')