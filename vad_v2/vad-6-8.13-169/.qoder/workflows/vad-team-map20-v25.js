export const meta = {
  name: 'vad-team-map20-v25',
  description: 'Unified Claude+Qoder VAD iteration workflow. Runs exactly five serial versions v191-v195. Every version must produce a real results/batch_eval_vNN_smoke.log, COMMAND.txt, per-video JSONs, and evaluation_report.json even when its proposed mechanism is rejected or fails an offline gate; approved mechanisms additionally run hard-88 and video-level bootstrap adjudication. Combines multi-agent research/debate/review, one-change discipline, train-only prechecks, flag/marker instrumentation, serial queueing, artifact grep verification, cost accounting, and keep/revert records.',
  phases: [
    { title: 'Research', detail: 'parallel code/data diagnosis, literature research, proposal synthesis' },
    { title: 'Review', detail: 'closed-family, leakage, effect-size, instrumentation, and one-change gate' },
    { title: 'Implement', detail: 'backup, config-gated implementation, tests, marker self-test' },
    { title: 'BatchEval', detail: 'mandatory version smoke log; eligible mechanisms continue to hard-88' },
    { title: 'Adjudicate', detail: 'artifact greps, bootstrap, keep/revert, summary and ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const PKG = ROOT + '/video_anomaly_detection'
const ITER = PKG + '/iteration_logs'
const PY = 'python3.14'
const START_VERSION = 191
const LOOPS = 5
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARD_FLAGS = '--categories ' + HARD_CATS
const HARD_N = 88
const MIN_HARD_DELTA = 0.008
const MIN_P = 0.90

const RULES = [
  'The dirty working tree is the current source of truth. Never checkout/reset/stash/clean and never modify unrelated user changes.',
  'Before editing, copy every touched file to ' + ROOT + '/.qoder/backups/<version>/. Never git commit.',
  'One controlled mechanism per version. New behavior must be config-gated, default OFF, threaded through from_args and both CLIs, and flag-OFF must preserve committed behavior.',
  'Never edit evaluate.py, GT data, test annotations, or analysis_bootstrap_ab.py. Never tune on test GT. Training-only prechecks are allowed; test GT is verdict-only.',
  'Never run two batch_evaluate processes concurrently. This is a 16 GB machine; concurrency is fixed at 2.',
  'Long work uses nohup plus short polls. No foreground command may run longer than 150 seconds.',
  'Every factual marker/count/artifact claim must be grepped/read from produced files, never asserted from intent.',
  'Every implemented mechanism emits one unconditional per-video marker on all paths, including early Normal return and zero detections, plus a FIRE marker when material.',
  'Every version v191-v195 MUST complete a real batch_evaluate smoke and produce the top-level batch_eval log. A rejected/no-go proposal runs the committed baseline under that version tag; it is not allowed to skip the log.',
].join('\n')

const CLOSED = [
  'curve-derived within-video confidence reranking; per-video normalization; synthetic/dense proposal append',
  'VLM call merge/batch/contact sheets; scan-tail/max_scan pruning; scan_segment absolute-confidence rescue gates',
  'boundary outward growth (BUE owns it); CGR/CCR/MIL relocation; whole-video CNV/SNH veto or MIL/CLIP protection variants',
  'KFD frame spread, TCP prompt wording, FSD 6s-to-3s split, uniform PTE, fixed curve smoothing',
  'CLIP dense/native/motion-zoom/SigLIP backbone recall curves already closed by v185/v188/v189',
].join('; ')

const CONTEXT = [
  'PROJECT: UCF-Crime temporal anomaly localization. Committed pipeline is v137 + Q-score + BUE + CBT + CBT-recal, default ON; most experimental flags default OFF.',
  'Current authoritative history is ' + ITER + '/session_v97_ledger.md. Read rows v179-v189 and any v190 artifacts before proposing.',
  'v190 VMC exists as an unfinished offline VideoMAE sliding-window curve precheck (' + ITER + '/analysis_v190_vmc_precheck.py); inspect and finish/adjudicate it before deciding whether that representation family is open.',
  'Full-290 committed evidence is approximately mAP 0.173; hard-88 is approximately 0.11-0.12 and noisy. Prefer paired-json/log designs. Cross-run changes require a fresh same-session reference and carry about +/-0.019 VLM noise.',
  'Closed families: ' + CLOSED + '.',
  'Promising qualitatively different directions must be train-label-valid: learned temporal context using train video labels, VideoMAE curve if v190 passes, supervised score quality, or a genuinely different presentation/localizer. Do not invent frame labels.',
].join('\n')

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string' },
    target_failure: { type: 'string' },
    evidence: { type: 'string' },
    train_only_precheck: { type: 'string' },
    expected_effect: { type: 'string' },
    implementation: { type: 'string' },
    flag: { type: 'string' },
    marker: { type: 'string' },
    paired_field: { type: 'string' },
    closed_family_check: { type: 'string' },
    leakage_check: { type: 'string' },
    risks: { type: 'string' },
  },
  required: ['title', 'mechanism', 'target_failure', 'evidence', 'train_only_precheck', 'expected_effect', 'implementation', 'flag', 'marker', 'paired_field', 'closed_family_check', 'leakage_check', 'risks'],
  additionalProperties: false,
}

const reviewSchema = {
  type: 'object',
  properties: {
    approve: { type: 'boolean' },
    feedback: { type: 'string' },
    one_change: { type: 'boolean' },
    closed_family_ok: { type: 'boolean' },
    leakage_ok: { type: 'boolean' },
    instrumentation_ok: { type: 'boolean' },
    effect_ok: { type: 'boolean' },
  },
  required: ['approve', 'feedback', 'one_change', 'closed_family_ok', 'leakage_ok', 'instrumentation_ok', 'effect_ok'],
  additionalProperties: false,
}

const implSchema = {
  type: 'object',
  properties: {
    ready: { type: 'boolean' },
    precheck_go: { type: 'boolean' },
    eval_flags: { type: 'string' },
    marker: { type: 'string' },
    paired_field: { type: 'string' },
    changed_files: { type: 'string' },
    validation: { type: 'string' },
    note: { type: 'string' },
  },
  required: ['ready', 'precheck_go', 'eval_flags', 'marker', 'paired_field', 'changed_files', 'validation', 'note'],
  additionalProperties: false,
}

const evalSchema = {
  type: 'object',
  properties: {
    done: { type: 'boolean' }, crashed: { type: 'boolean' },
    mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' },
    succeeded: { type: 'integer' }, failed: { type: 'integer' }, marker_count: { type: 'integer' },
    log_exists: { type: 'boolean' }, command_exists: { type: 'boolean' }, report_exists: { type: 'boolean' },
    note: { type: 'string' },
  },
  required: ['done', 'crashed', 'mAP', 'succeeded', 'failed', 'marker_count', 'log_exists', 'command_exists', 'report_exists', 'note'],
  additionalProperties: false,
}

const bootSchema = {
  type: 'object',
  properties: {
    ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' },
    ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' },
    new_map: { type: 'number' }, ref_map: { type: 'number' }, per_category: { type: 'string' }, note: { type: 'string' },
  },
  required: ['ran', 'mode', 'point_delta', 'p_gt0', 'new_map', 'ref_map', 'note'],
  additionalProperties: false,
}

async function retry(fn, n) {
  for (let i = 0; i < (n || 3); i++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r }
    catch (e) { log('retry ' + (i + 1) + ': ' + String(e).slice(0, 180)) }
  }
  return null
}

function researchPrompt(V, role, focus, history) {
  return [
    'You are VAD researcher ' + role + ' for ' + V + '. ' + focus,
    RULES, CONTEXT, 'SESSION v191+ HISTORY:\n' + (history.length ? history.join('\n') : '(none)'),
    'Read the authoritative ledger tail, v183-v189 summaries, v190 script/artifacts, current config/pipeline/discovery, and relevant cached reports. Diagnose with numbers. Do at least 3 real literature/web queries if available; never fabricate citations.',
    'Propose ONE architecture-level mechanism outside every closed family. It needs a train-only zero-VLM precheck with a pre-registered numerical GO gate, exact files/functions, one default-OFF flag wired to both CLIs, an unconditional per-video marker, FIRE marker, and paired snapshot where technically possible.',
    'If no defensible mechanism clears the gate, return the strongest honest NO-GO proposal; this version will still run a mandatory committed-baseline smoke log.',
    'StructuredOutput exactly once.',
  ].join('\n\n')
}

function implementPrompt(V, proposal, review) {
  return [
    'You are implementer D for ' + V + '.', RULES, CONTEXT,
    'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 2),
    'REVIEW:\n' + JSON.stringify(review, null, 2),
    'First run/finish the proposal train-only precheck. If NO-GO, do not edit production code: return ready=false, precheck_go=false, eval_flags="", marker="[COST]". The mandatory baseline smoke will still run.',
    'If GO: back up every touched file, implement exactly one default-OFF mechanism, wire config/from_args/run.py/batch_evaluate.py, add unconditional [<MECH>-VID] on every return path and [<MECH>-FIRE] on material action, and write an index-stable paired field if feasible.',
    'Validate py_compile, focused pytest, both --help outputs, default equality, flag-OFF behavior, and a cheap synthetic marker/path test. Do not run batch_evaluate here.',
    'Return actual outputs, not plans. StructuredOutput exactly once.',
  ].join('\n\n')
}

function evalPrompt(V, tag, maxPerCat, flags, marker, expectedN) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const logF = dir + '.log'
  const report = dir + '/evaluation_report.json'
  return [
    'You are the serial batch evaluator for ' + V + ' ' + tag + '. ' + RULES,
    'Artifacts: dir=' + dir + ' top_log=' + logF + ' report=' + report + ' expected=' + expectedN + '.',
    'Wait until pgrep -f batch_evaluate is empty. Never overlap evaluations. If this exact run is alive, adopt it.',
    'Before launch create dir and write the exact command to ' + dir + '/COMMAND.txt. Launch with nohup from ROOT using:',
    PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + dir + ' --max_per_category ' + maxPerCat + ' --concurrency ' + CONCURRENCY + ' ' + flags,
    'Redirect stdout+stderr to ' + logF + '. The top-level .log is mandatory and must never be replaced only by inner run.log. Poll in <=140s calls. Resume the same command after transient failure; JSON skipping is expected. Diagnose a >40min no-growth stall before restart.',
    'Completion gate: report exists; top log exists and non-empty; COMMAND.txt exists; succeeded+failed matches expectedN (or explain exact harness count); marker count from max(top log, inner run.log). For baseline/no-go smoke marker is [COST]. Read temporal mAP and AP@0.3/0.5/0.7 from report.',
    'Do not return done=true unless all three mandatory artifacts exist. StructuredOutput exactly once.',
  ].join('\n\n')
}

function bootstrapPrompt(V, field) {
  const newDir = ROOT + '/results/batch_eval_' + V + '_hard88'
  const refDir = ROOT + '/results/batch_eval_' + V + '_ref88'
  return [
    'You are adjudicator D for ' + V + '. ' + RULES,
    field
      ? ('Run ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --paired-json ' + field + ' --min-delta ' + MIN_HARD_DELTA + ' --min-p ' + MIN_P)
      : ('Run ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --ref ' + refDir + ' --min-delta ' + MIN_HARD_DELTA + ' --min-p ' + MIN_P),
    'Also compute evaluator-faithful per-category deltas for all four hard categories. Preserve bootstrap_ab.json in the NEW directory. Quote the cross-run +/-0.019 caveat when not paired.',
    'StructuredOutput exactly once.',
  ].join('\n\n')
}

const history = []
log('v25 unified workflow starting: exactly five serial versions v191-v195; every version requires batch_eval log + report + COMMAND.txt.')

for (let loop = 0; loop < LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop)
  log('=== ' + V + ' (' + (loop + 1) + '/' + LOOPS + ') ===')
  phase('Research')
  const [a, b] = await parallel([
    () => retry(() => agent(researchPrompt(V, 'A', 'Focus on train-valid temporal representation/localizer and the unfinished v190 VideoMAE curve evidence.', history), { schema: proposalSchema, label: 'A-' + V, phase: 'Research' })),
    () => retry(() => agent(researchPrompt(V, 'B', 'Focus on score quality, candidate recall, and whether a lightweight learned temporal model can use existing train video labels without invented frame labels.', history), { schema: proposalSchema, label: 'B-' + V, phase: 'Research' })),
  ])
  const proposal = await retry(() => agent([
    'You are A+B consensus for ' + V + '. Select exactly ONE proposal. Reject closed-family repetition, test-label leakage, weak effect arithmetic, and mechanisms that cannot prove activation. Do not merge independent ideas.',
    RULES, CONTEXT, 'A:\n' + JSON.stringify(a), 'B:\n' + JSON.stringify(b), 'StructuredOutput exactly once.',
  ].join('\n\n'), { schema: proposalSchema, label: 'AB-' + V, phase: 'Research' })) || a || b

  phase('Review')
  const review = proposal ? await retry(() => agent([
    'You are skeptical reviewer C for ' + V + '.', RULES, CONTEXT,
    'PROPOSAL:\n' + JSON.stringify(proposal, null, 2),
    'Approve only if one change, outside every closed family, no test-GT tuning, train-only precheck has a fixed GO gate, expected effect is measurable, and marker/paired instrumentation is concrete. Otherwise reject honestly; the version still gets a baseline smoke log.',
    'StructuredOutput exactly once.',
  ].join('\n\n'), { schema: reviewSchema, label: 'C-' + V, phase: 'Review' })) : null

  phase('Implement')
  let impl = { ready: false, precheck_go: false, eval_flags: '', marker: '[COST]', paired_field: '', changed_files: '', validation: '', note: 'proposal/review unavailable or rejected' }
  if (proposal && review && review.approve) {
    impl = await retry(() => agent(implementPrompt(V, proposal, review), { schema: implSchema, label: 'D-impl-' + V, phase: 'Implement' })) || impl
  }

  phase('BatchEval')
  const active = !!(impl.ready && impl.precheck_go)
  const smokeFlags = active ? impl.eval_flags : ''
  const smokeMarker = active ? impl.marker : '[COST]'
  const smoke = await retry(() => agent(evalPrompt(V, 'smoke', 1, smokeFlags, smokeMarker, 14), { schema: evalSchema, label: 'D-smoke-' + V, phase: 'BatchEval' }))
  if (!smoke || !smoke.done || !smoke.log_exists || !smoke.command_exists || !smoke.report_exists) {
    history.push(V + ' INCOMPLETE: mandatory batch_eval smoke artifacts missing')
    log(V + ' blocked: mandatory batch_eval log/report/COMMAND gate failed')
    continue
  }

  let hard = null, ref = null, boot = null
  if (active && smoke.marker_count > 0 && !smoke.crashed) {
    if (!impl.paired_field) {
      ref = await retry(() => agent(evalPrompt(V, 'ref88', 999, HARD_FLAGS, '[COST]', HARD_N), { schema: evalSchema, label: 'D-ref-' + V, phase: 'BatchEval' }))
    }
    hard = await retry(() => agent(evalPrompt(V, 'hard88', 999, HARD_FLAGS + ' ' + impl.eval_flags, impl.marker, HARD_N), { schema: evalSchema, label: 'D-hard-' + V, phase: 'BatchEval' }))
    if (hard && hard.done && hard.marker_count > 0 && (impl.paired_field || (ref && ref.done))) {
      phase('Adjudicate')
      boot = await retry(() => agent(bootstrapPrompt(V, impl.paired_field), { schema: bootSchema, label: 'D-boot-' + V, phase: 'Adjudicate' }))
    }
  }

  phase('Adjudicate')
  const keep = !!(boot && boot.ran && boot.point_delta >= MIN_HARD_DELTA && boot.p_gt0 >= MIN_P)
  await retry(() => agent([
    'You are recorder C for ' + V + '. ' + RULES,
    'PROPOSAL=' + JSON.stringify(proposal), 'REVIEW=' + JSON.stringify(review), 'IMPL=' + JSON.stringify(impl),
    'SMOKE=' + JSON.stringify(smoke), 'REF88=' + JSON.stringify(ref), 'HARD88=' + JSON.stringify(hard), 'BOOT=' + JSON.stringify(boot), 'KEEP=' + keep,
    'Re-read and grep all artifacts. If production code was implemented and KEEP=false, restore only files backed up for this version and verify the mechanism symbols/defaults are gone or inert; never overwrite unrelated concurrent changes. If KEEP=true, leave the mechanism in-tree but do not silently flip default ON without a full-290 confirm.',
    'Write ' + ITER + '/summary_' + V + '_<slug>_20260820.md and append one ledger row. The summary must link the mandatory smoke log ' + ROOT + '/results/batch_eval_' + V + '_smoke.log and any ref/hard/bootstrap artifacts, include marker counts, COST lines, metrics, decision, and next-version exclusions.',
    'Return a concise artifact list and decision.',
  ].join('\n\n'), { label: 'C-record-' + V, phase: 'Adjudicate' }))
  history.push(V + ' smoke=' + smoke.mAP + ' active=' + active + ' hard=' + (hard ? hard.mAP : -1) + ' delta=' + (boot ? boot.point_delta : 0) + ' keep=' + keep + ' log=' + ROOT + '/results/batch_eval_' + V + '_smoke.log')
  log(history[history.length - 1])
}

phase('Adjudicate')
await retry(() => agent([
  'You are final reviewer C. Verify v191-v195 each has a non-empty top-level results/batch_eval_vNN_smoke.log, COMMAND.txt, and evaluation_report.json. Do not claim success for a missing artifact.',
  RULES, 'HISTORY:\n' + history.join('\n'),
  'Write ' + ITER + '/summary_v25_v191-v195_20260820.md with a five-row artifact/metric/decision table, kept baseline, closed/open families, and exact follow-up. Return missing artifacts first, otherwise state all five complete.',
].join('\n\n'), { label: 'C-final-v25', phase: 'Adjudicate' }))

log('v25 complete: ' + history.length + '/5 version records; mandatory batch_eval artifact verification delegated to final reviewer.')
return { versions: history, expected: ['v191', 'v192', 'v193', 'v194', 'v195'] }
