export const meta = {
  name: 'vad-team-map20-v24',
  description: 'VAD agent team v24 — v182 FSD (Fine-Scan Directed, category-gated 3s sub-segment scan): a candidate-GRANULARITY lever on the committed directed scan, aimed at the brief-event residual that TCP (v177, prompt-side, CLOSED: clip-averaging hypothesis wrong for this VLM — whole-segment prompts cannot be steered by text) and KFD (v173, frame placement, CLOSED) could not touch. If the prompt cannot make the VLM find a brief event inside a 6s window, shrink the WINDOW: for videos whose pipeline-resolved category is Shoplifting or RoadAccidents (the two cand-miss/recognition-bound brief-event hard cats: Shoplifting cand-miss 11/25 — the biggest single residual in the v16 record; RoadAccidents GT 1-3s events), the directed scan runs on 3s HALF-segments (each 6s segment split into its 2 chronological halves, ~6 frames each at 2fps) instead of the 6s grid — same scan_segment prompt, same JSON schema, same directed path (FCCS stays off, pre-registered VOID check), exactly 2x the scan calls on GATED videos only. All downstream (candidate build, merge/NMS, classify, BUE, CBT) UNCHANGED. This is NOT banned-3 (max_scan/scan-tail = dropping segments) and NOT banned-5 (dense-append = appending to the candidate pool) — it REPLACES the segment grid on a category-gated subset, the v12-admissible "category-aware candidate generation / scan density" lever. Flag fsd_cats default OFF, CLI run.py+batch_evaluate.py. Markers: [FSD-VID] unconditional per video (gated=0/1, n_segments) on EVERY path + [FSD-FIRE] once per gated video. Self-test: OFF byte-identical; ON smoke shows >= 1 gated video with doubled segment count and [FSD-VID]==#videos. CROSS-RUN A/B vs FRESH same-session committed REF (VLM inputs change — no paired instrument; +/-0.019 caveat pre-registered). GATES loop1 hard-88: point_delta >= +0.008 AND P(delta>0) >= 0.9 AND new hard mAP >= 0.11306; FCCS==0 both logs; det/video increase flagged > +1.0; cost report ([COST] VLM call delta on gated videos). Loop2 (only if loop1 passes) full-290 net-neutrality: delta >= -0.003 AND abs >= 0.16454 AND Normal-FA rel increase <= +2%. Purpose-population check: RA009 (GT 7-8s) / RA021 (GT 1-3s) conversion. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON); every other flag default-OFF and stays so. Concurrency 2, never two evals. Gating source: the pipeline\'s OWN resolved video category (the same category string that drives category-curve selection); if Shoplifting/RoadAccidents never occur in that label space the implementer must say so loudly and gate on the CLIP top-1 curve category instead (documented in the record).',
  phases: [
    { title: 'Implement', detail: 'v182 --fsd_cats + self-test' },
    { title: 'AB88', detail: 'fresh REF88 + FSD88 cross-run bootstrap + purpose check' },
    { title: 'Confirm290', detail: 'if 88 passes: REF290 + FSD290 net-neutrality' },
    { title: 'Decide', detail: 'promote-or-close + summary + ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const PKG = ROOT + '/video_anomaly_detection'
const ITER = PKG + '/iteration_logs'
const PY = 'python3.14'
const V = 'v182'
const V290 = 'v184'
const HARNESS = '--categories Shooting,RoadAccidents,Shoplifting,Explosion'
const N_HARD = 88
const N_FULL = 290

const RULES = [
  'NEVER edit the evaluator, GT labels, or ' + ITER + '/analysis_bootstrap_ab.py.',
  'NEVER run two batch_evaluate processes concurrently (16 GB swap-thrash).',
  'NO Bash command may run > 150s in the foreground — the runtime kills stalled agents at 180s. Long work = nohup background + short polls.',
  'grep claims must be GREPPED in produced artifacts, never asserted (FM20/FM21).',
  'Do NOT git-commit anything. StructuredOutput EXACTLY ONCE per agent (only where a schema is given).',
].join('\n')

const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number' }, fccs_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'fccs_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, det_per_video_new: { type: 'number' }, det_per_video_ref: { type: 'number' }, fa_rel_inc: { type: 'number' }, purpose_converted: { type: 'string' }, per_category: { type: 'string' }, cost_delta: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'ref_map', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

function launchPrompt(dir, logF, rep, extraFlags) {
  return [
    'You are Agent D-launcher. Ensure the eval for ' + dir + ' is running, then return IMMEDIATELY. SHORT. ' + RULES,
    '0. QUEUE: pgrep -f "batch_evaluate"; if another run is active, sleep 140 and re-check (up to 20x, separate short Bash calls); launch only when clear.',
    '1. If ' + rep + ' exists -> launched=true note="already complete".',
    '2. Else write ' + dir + '/COMMAND.txt and launch EXACTLY:',
    "bash -lc 'mkdir -p " + dir + "; source " + PKG + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category 999 --concurrency 2 " + extraFlags + " > " + logF + " 2>&1 & echo PID $!'",
    '3. sleep 30; tail -5 ' + logF + ' for an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function watchPrompt(dir, logF, rep, nTotal, marker) {
  return [
    'You are Agent D-watcher for ' + dir + ' (expect ' + nTotal + ' videos). SHORT. ' + RULES,
    '1. If ' + rep + ' exists -> done=true. Metrics: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'),ti.get(\'succeeded\'),ti.get(\'failed\'))" . num_videos MUST be ' + nTotal + ' (else scream harness bug).',
    '   marker_count=$(grep -cF "' + marker + '" ' + logF + ' 2>/dev/null); fccs_count=$(grep -cF "FCCS" ' + logF + ' 2>/dev/null) (MUST be 0 — scan path constant; nonzero = VOID confound, scream it).',
    '2. Else pgrep -f "batch_evaluate.*' + dir.split('/').pop() + '". Alive -> ONE sleep 140, n=$(ls ' + dir + '/*.json 2>/dev/null | wc -l), done=false note="running <n>/' + nTotal + '". >40 min no progress -> tail -30 diagnose, no kill without evidence.',
    '3. Dead with no report -> tail -30 ' + logF + '. TRANSIENT -> relaunch the SAME nohup (runner skips completed). CODE BUG -> crashed=true + traceback.',
    'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, marker_count, fccs_count, note.',
  ].join('\n')
}
async function runEval(tag, extraFlags, marker, nTotal, ph) {
  const dir = ROOT + '/results/batch_eval_' + tag
  const logF = dir + '.log'
  const rep = dir + '/evaluation_report.json'
  const launch = await retryAgent(() => agent(launchPrompt(dir, logF, rep, extraFlags), { schema: launchSchema, label: 'D-launch-' + tag, phase: ph }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, fccs_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  let st = null
  if (launch.note && launch.note.indexOf('already complete') >= 0) {
    st = await retryAgent(() => agent(watchPrompt(dir, logF, rep, nTotal, marker), { schema: watchSchema, label: 'D-watch-adopt-' + tag, phase: ph }))
  }
  for (let w = 0; w < 160 && !(st && (st.done || st.crashed)); w++) {
    st = await retryAgent(() => agent(watchPrompt(dir, logF, rep, nTotal, marker), { schema: watchSchema, label: 'D-watch-' + tag + '-' + w, phase: ph }))
  }
  return st || { done: false, crashed: true, mAP: -1, marker_count: -1, fccs_count: -1, note: 'watcher cap/null' }
}

// ============================ PHASE 1 — IMPLEMENT ============================
phase('Implement')
log('v24: implement v182 FSD (category-gated 3s fine scan)')

const IMPL_PROMPT = [
  'You are Agent D-implementer for v182 FSD. ' + RULES,
  'TASK — flag fsd_cats (default OFF), version tag v182:',
  '1. BACKUP: cp pipeline.py anomaly_discovery.py config.py run.py batch_evaluate.py into ' + ROOT + '/.qoder/backups/' + V + '/ .',
  '2. FIND the directed-scan segment construction (the committed path: full_coverage_comparative_scan=False; scan_segment in anomaly_discovery.py ~L358 receives 6s segments; find where the segment list is built in pipeline.py / anomaly_discovery.py — grep for the segment builder) and the pipeline-resolved video category string (the one driving category-curve selection; verify whether Shoplifting/RoadAccidents occur in that label space — check the VideoMAE mapping and any CLIP fallback; document your gating source).',
  '3. Implement: when fsd_cats is on AND the video\'s resolved category is Shoplifting or RoadAccidents, the DIRECTED scan iterates 3s HALF-segments (each 6s segment split into its 2 chronological halves, ~6 frames at 2fps; last half may be shorter — keep it, min 3 frames) instead of the 6s grid. Same scan_segment call/prompt/schema per sub-segment; NOTHING else changes (candidate build/merge/NMS/classify/BUE/CBT untouched; the >60-segment long-video rule is computed on the ORIGINAL 6s grid BEFORE splitting — document the resulting segment count). FCCS stays OFF (the _fccs switch must NOT be affected).',
  '4. Markers: [FSD-FIRE] once per GATED video (vid, category, n_segments_6s -> n_segments_3s); [FSD-VID] UNCONDITIONAL per video on EVERY path incl. early-return-normal/0-detection (vid, gated=0|1, n_segments) — grep -cF == #videos.',
  '5. config: fsd_cats: bool = False (correct dataclass) + CLI --fsd_cats in run.py AND batch_evaluate.py (set_defaults False).',
  '6. VALIDATE: py_compile all; from_args defaults show fsd_cats False and every other flag unchanged; both --help show it.',
  '7. SELF-TEST (queue-disciplined, cwd=' + ROOT + '): (a) OFF smoke (--categories Normal,Shoplifting --max_per_category 1): grep -c FSD log == 0. (b) ON smoke (same videos --fsd_cats): [FSD-VID]==2; if the Shoplifting video is gated, [FSD-FIRE] shows its segment count doubling and its scan call count ([COST] lines) doubling vs OFF; if it is NOT gated (label-space miss), say so LOUDLY and switch the gating source to the CLIP top-1 curve category, re-run, and document. Paste everything. Verdict line: READY or NOT-READY (+reasons).',
].join('\n')

const impl = await retryAgent(() => agent(IMPL_PROMPT, { label: 'D1-implement-' + V, phase: 'Implement' }))
if (!impl || /NOT-READY/.test(impl)) {
  log('v24 ABORT: implement NOT-READY — no eval spent. ' + (impl || 'null').slice(0, 400))
} else {

// ============================ PHASE 2 — AB88 ============================
phase('AB88')
log('v24: fresh REF88 + FSD88 (cross-run)')

const ref88 = await runEval(V + '_ref88', HARNESS, '[COST]', N_HARD, 'AB88')
if (!ref88.done || ref88.fccs_count !== 0) {
  log('v24 ABORT: REF88 ' + (ref88.done ? 'FCCS confound' : 'incomplete') + ': ' + ref88.note)
} else {
const fsd88 = await runEval(V + '_fsd88', HARNESS + ' --fsd_cats', 'FSD-VID', N_HARD, 'AB88')
if (!fsd88.done || fsd88.fccs_count !== 0) {
  log('v24 ABORT: FSD88 ' + (fsd88.done ? 'FCCS confound' : 'incomplete') + ': ' + fsd88.note)
} else if (fsd88.marker_count < 80) {
  log('v24 ABORT: FSD-VID markers ' + fsd88.marker_count + ' << 88 — FM17 fail')
} else {
log('v24 AB88 runs done: REF mAP=' + ref88.mAP + ' FSD mAP=' + fsd88.mAP)

const BOOT88_PROMPT = [
  'You are Agent D-adjudicator for the FSD CROSS-RUN hard-88 A/B. Bash < 150s. ' + RULES,
  'STEP 1 (confound gates): FCCS==0 in BOTH logs; FSD-VID ~88 in the FSD log; FSD-FIRE > 0 (grep it — the mechanism actually gated some videos, FM5; if 0, ran=false and scream).',
  'STEP 2: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_' + V + '_fsd88 --ref ' + ROOT + '/results/batch_eval_' + V + '_ref88 --min-delta 0.008 --min-p 0.9 -> point_delta, ci90, p_gt0, new_map, ref_map. Paste the cross-run CAVEAT verbatim.',
  'STEP 3 (FM2 watch): mean detections/video both dirs; flag increase > +1.0.',
  'STEP 4 (purpose population): RoadAccidents009 (GT 7-8s) and RoadAccidents021 (GT 1-3s): does FSD gain an IoU>=0.1 detection where REF has none? (GT via evaluate.load_ground_truth, frames/30).',
  'STEP 5: per-category AP deltas (4 hard cats) into per_category. cost_delta: VLM scan call counts from the [COST] lines, gated videos only, FSD vs REF (expect ~2x on gated). fa_rel_inc: no Normal videos here -> -1.',
  'GATES loop1: point_delta >= +0.008 AND p_gt0 >= 0.9 AND new_map >= 0.11306. Report all schema fields; note = VERDICT verbatim + caveat.',
].join('\n')

const b88 = await retryAgent(() => agent(BOOT88_PROMPT, { schema: bootSchema, label: 'D-boot88-' + V, phase: 'AB88' }))
const pass88 = !!(b88 && b88.ran && b88.point_delta >= 0.008 && b88.p_gt0 >= 0.9 && b88.new_map >= 0.11306)
log('v24 loop1 gates: pass=' + pass88 + ' delta=' + (b88 && b88.point_delta) + ' P=' + (b88 && b88.p_gt0))

// ============================ PHASE 3 — CONFIRM 290 (conditional) ============================
let b290 = null
if (pass88) {
  phase('Confirm290')
  const ref290 = await runEval(V290 + '_ref290', '', '[COST]', N_FULL, 'Confirm290')
  if (ref290.done) {
    const fsd290 = await runEval(V290 + '_fsd290', '--fsd_cats', 'FSD-VID', N_FULL, 'Confirm290')
    if (fsd290.done) {
      const BOOT290_PROMPT = [
        'You are Agent D-adjudicator for the FSD full-290 net-neutrality confirm. Bash < 150s. ' + RULES,
        'STEP 1: FCCS==0 both logs; FSD-VID ~290 in the FSD log; FSD-FIRE > 0.',
        'STEP 2: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_' + V290 + '_fsd290 --ref ' + ROOT + '/results/batch_eval_' + V290 + '_ref290 --min-delta -0.003 --min-p 0.5 -> full output. CAVEAT verbatim.',
        'STEP 3: Normal-video FA rate both arms (category==Normal videos with >=1 det conf>0); fa_rel_inc = (new-ref)/ref.',
        'GATES loop2: point_delta >= -0.003 AND new_map >= 0.16454 AND fa_rel_inc <= +0.02. Same schema fields as loop1 (purpose_converted -> "n/a at 290"; cost_delta = gated-video [COST] delta).',
      ].join('\n')
      b290 = await retryAgent(() => agent(BOOT290_PROMPT, { schema: bootSchema, label: 'D-boot290-' + V290, phase: 'Confirm290' }))
    }
  }
}

// ============================ PHASE 4 — DECIDE + RECORDS ============================
phase('Decide')
const promote = !!(pass88 && b290 && b290.ran && b290.point_delta >= -0.003 && b290.new_map >= 0.16454 && b290.fa_rel_inc <= 0.02)
log('v24 decision: promote=' + promote)

const DECIDE_PROMPT = [
  'You are Agent C (records) for v182 FSD. Write the session records from the data below; re-grep artifacts to verify numbers (FM20). Do NOT re-run evals. ' + RULES,
  'LOOP1 (hard-88 cross-run): ' + JSON.stringify(b88) + ' REF mAP=' + ref88.mAP + ' FSD mAP=' + fsd88.mAP,
  'LOOP2 (290, null if not armed): ' + JSON.stringify(b290),
  'WORKFLOW DECISION: promote=' + promote,
  '1. Write ' + ITER + '/summary_v24_fsd_20260817.md: verdict; motivation (TCP v177 closed the prompt side — if text cannot steer the VLM to brief events inside a 6s window, shrink the window; Shoplifting cand-miss 11/25 + RA009/RA021 recognition-bound); the gating source actually used (and whether the label-space fallback fired); loop1 numbers incl. caveat verbatim + purpose-population outcome; loop2 numbers or not-armed; per-cat; cost delta; gate tables; FM evidence; decision; family update (is scan GRANULARITY the answer to the brief-event residual, or is the recognition ceiling confirmed a third time?).',
  '2. Append ONE ledger row to ' + ITER + '/session_v97_ledger.md: loop | ' + V + ' | FSD ... | kept=<true|false> with loop1 (and loop2 if armed) numbers, FM greps, per-cat, purpose-population, cost delta, revert/promote state.',
  '3. DECISION EXECUTION:',
  '   - NOT promote: fsd_cats stays default False (verify + paste); code stays in-tree inert; one-line config closure comment.',
  '   - promote: backup ' + ROOT + '/.qoder/backups/' + V + '_promote/ FIRST; flip fsd_cats default True (dataclass + from_args + set_defaults both CLIs) + --no_fsd_cats opt-out; validate py_compile + defaults.',
  '4. Report files written + final flag defaults.',
].join('\n')

await retryAgent(() => agent(DECIDE_PROMPT, { label: 'C-records-' + V, phase: 'Decide' }))
log('v24 complete: promote=' + promote + ' loop1_delta=' + (b88 && b88.point_delta) + ' loop2=' + (b290 ? JSON.stringify({ d: b290.point_delta, m: b290.new_map }) : 'not-armed'))
}
}
}
