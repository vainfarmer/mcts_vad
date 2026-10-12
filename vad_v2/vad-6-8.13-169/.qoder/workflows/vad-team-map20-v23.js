export const meta = {
  name: 'vad-team-map20-v23',
  description: 'VAD agent team v23 — v181 PTE (Per-category prompt-Text Ensemble for the CLIP category curves): a SIGNAL-QUALITY lever on the curve substrate that BUE/CBT/frame_scores consume. The flat-in-span RoadAccidents/Shoplifting curves are the measured residual (v16 record: "next levers must attack SIGNAL quality"); the current single-text-prompt-per-category embedding is the weakest link. MECHANISM: for EACH of the 13 anomaly categories replace the ONE text prompt with K=4 pre-registered paraphrases (fixed in code, written down in the summary BEFORE any eval — no test-side tuning), text-encode all K, L2-normalize each, average, renormalize -> the category embedding. Frame features UNCHANGED (cached clip_features reused as-is); zero new VLM calls; only the text side of the CLIP similarity changes. Affects ONLY deterministic post-VLM consumers: get_category_scores curves -> BUE/CBT boundary ops + frame_scores + qscore. PHASE 0 (offline, ZERO eval spend): on TRAIN-split videos (4 per hard cat + 4 normals, local CLIP re-encode of frames; featextract machinery from analysis_v19_snh_featextract.py is the model), build each video\'s category curve with CURRENT prompts vs PTE prompts and measure GT-window prominence (max-z of the curve inside the GT interval vs the video median/MAD, the v18 recipe). PRE-REGISTERED GO GATE: mean GT-window prominence improves by >= +0.3 z on >= 3 of the 4 hard cats (Shooting/RoadAccidents/Shoplifting/Explosion) AND no hard cat regresses by > 0.1 AND normal-video curve max-z inflates by < +0.5 on average (curve-level FP guard). NO-GO -> record + STOP. GO -> implement flag pte_curves default OFF (CLI run.py+batch_evaluate.py, marker [PTE-VID] unconditional per video incl. early-return paths), self-test (OFF byte-identical curves — grep the curve hash; ON smoke shows PTE curves differ), then CROSS-RUN A/B on the hard-88 vs a FRESH same-session committed REF (VLM inputs unchanged but the boundary operators change, so no paired-log restore — pre-registered cross-run with the +/-0.019 VLM-variance caveat). GATES loop1 hard-88: point_delta >= +0.008 AND P(delta>0) >= 0.9 AND new hard mAP >= 0.11306, FCCS==0 in both logs (scan path constant), det/video increase flagged > +1.0. Loop2 (only if loop1 passes) full-290 net-neutrality: delta >= -0.003 AND abs >= 0.16454 AND Normal-FA rel increase <= +2%. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON; full-290 ~0.173); cgr/ccr/cnv/kfd/tcp/snh/snh_protect all default-OFF and stay so. Concurrency 2, never two evals. NONE of the 10 banned families (PTE changes the curve SIGNAL, not a within-video re-rank; not per-video normalization; not an abs-conf gate).',
  phases: [
    { title: 'Precheck', detail: 'offline train-split curve prominence A/B -> GO/NO-GO' },
    { title: 'Implement', detail: 'v181 --pte_curves + self-test' },
    { title: 'AB88', detail: 'fresh REF88 + PTE88 cross-run bootstrap' },
    { title: 'Confirm290', detail: 'if 88 passes: REF290 + PTE290 net-neutrality' },
    { title: 'Decide', detail: 'promote-or-close + summary + ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const PKG = ROOT + '/video_anomaly_detection'
const ITER = PKG + '/iteration_logs'
const PY = 'python3.14'
const V = 'v181'
const V290 = 'v183'
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

const preSchema = { type: 'object', properties: {
  go: { type: 'boolean' },
  prompts: { type: 'string', description: 'the FINAL 13x4 paraphrase table used (must be fixed before any eval)' },
  per_cat: { type: 'string', description: 'one line per hard cat: n videos, mean GT-window max-z current vs PTE, delta' },
  normal_inflation: { type: 'number', description: 'mean normal-video curve max-z delta (PTE - current)' },
  n_train_videos: { type: 'integer' },
  note: { type: 'string' },
}, required: ['go', 'prompts', 'per_cat', 'normal_inflation', 'n_train_videos', 'note'], additionalProperties: false }

const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number' }, fccs_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'fccs_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, det_per_video_new: { type: 'number' }, det_per_video_ref: { type: 'number' }, fa_rel_inc: { type: 'number' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'ref_map', 'note'], additionalProperties: false }

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

// ============================ PHASE 0 — OFFLINE PRECHECK ============================
phase('Precheck')
log('v23 phase 0: offline train-split PTE curve prominence check')

const PRE_PROMPT = [
  'You are Agent D-precheck for v181 PTE. OFFLINE ONLY — zero eval spend, NO VLM API, NO batch_evaluate. ' + RULES,
  'GOAL: fix the 13x4 paraphrase table, then measure on TRAIN videos whether PTE prompts improve category-curve GT-window prominence vs the CURRENT single prompts.',
  '1. FIND the current prompt construction: read the AnomalyExpert class (models.py or anomaly_expert.py — grep for text_features / prompt_weights.json / the 13 category prompts) and document exactly how the category text embeddings are built today (incl. any learned prompt_weights.json interplay — PTE must compose with or cleanly bypass it; your call, documented).',
  '2. WRITE the paraphrase table: K=4 diverse surveillance-style phrasings per anomaly category (e.g. RoadAccidents: "a photo of a road accident, vehicle collision" / "a car crash on the street" / "vehicles collided with debris on the road" / "a traffic accident just happened"). Keep the SAME prompt template family as the current code. This table is FROZEN once written — it is the pre-registered mechanism. Save it to ' + ITER + '/v181_pte_prompts.json.',
  '3. SAMPLE: from the TRAIN split (/Users/zzl/Movies/ucf-crime, train parquet), pick 4 videos per hard cat (Shooting/RoadAccidents/Shoplifting/Explosion) + 4 Normal, videos short enough to encode locally (~1-2 min each).',
  '4. COMPUTE: per-frame CLIP ViT-L features ONCE per video (reuse the featextract machinery of ' + ITER + '/analysis_v19_snh_featextract.py as a library; write a NEW ' + ITER + '/analysis_v181_pte_precheck.py), then each video\'s category curve cat/(cat+normal) with CURRENT prompts vs PTE prompts (text side only). Metric per video+cat: GT-window prominence = max-z of the curve inside the GT interval vs the video\'s median/MAD (v18 recipe).',
  '5. GO GATE (pre-registered): mean prominence delta >= +0.3 z on >= 3/4 hard cats AND no hard cat delta < -0.1 AND normal-video curve max-z inflation < +0.5 mean.',
  'Report go, prompts (the full table), per_cat (4 lines), normal_inflation, n_train_videos, note.',
].join('\n')

const pre = await retryAgent(() => agent(PRE_PROMPT, { schema: preSchema, label: 'D-precheck-' + V, phase: 'Precheck' }))
log('v23 precheck: go=' + (pre && pre.go) + ' normal_inflation=' + (pre && pre.normal_inflation))

if (!pre || !pre.go) {
  log('v23 NO-GO: PTE fails its offline curve-prominence gate — closed without eval spend. precheck=' + JSON.stringify(pre))
} else {

// ============================ PHASE 1 — IMPLEMENT ============================
phase('Implement')

const IMPL_PROMPT = [
  'You are Agent D-implementer for v181 PTE. Offline precheck GO. ' + RULES,
  'The FROZEN paraphrase table is ' + ITER + '/v181_pte_prompts.json — use it VERBATIM (load it from that path at runtime; do not paraphrase it yourself).',
  'TASK — flag pte_curves (default OFF), version tag v181:',
  '1. BACKUP: cp the files you will edit (models.py / anomaly_expert.py wherever text_features are built + config.py run.py batch_evaluate.py pipeline.py if touched) into ' + ROOT + '/.qoder/backups/' + V + '/ .',
  '2. Implement: when pte_curves is on, the category text embedding for EACH of the 13 categories = L2-normalized average of the L2-normalized embeddings of its 4 paraphrases (encode once at load; the Normal prompt is UNCHANGED). Flag-off: the current single-prompt path bit-identical. Composition with prompt_weights.json: document your choice (cleanest = PTE replaces the base prompt text but keeps any learned weights applied as today — verify in code what prompt_weights actually modulates).',
  '3. config: pte_curves: bool = False (correct dataclass for wherever text_features are configured) + CLI --pte_curves in run.py AND batch_evaluate.py (set_defaults False).',
  '4. Markers: one unconditional [PTE-VID] per video on EVERY path incl. early-return (grep -cF == #videos), logging pte=on/off + the video\'s top-1 curve category.',
  '5. SELF-TEST (queue-disciplined, cwd=' + ROOT + '): (a) OFF smoke (1 Normal + 1 Arson, --categories Normal,Arson --max_per_category 1): grep -c PTE log == 0; (b) ON smoke: [PTE-VID]==2, and PROVE the curves differ — log or print a small curve checksum (e.g. first-8 values hash of the video\'s top-cat curve) on/off and show they differ ON vs OFF while the OFF hash matches a committed-only run (byte-identical off-path). Paste everything. Verdict line: READY or NOT-READY (+reasons).',
].join('\n')

const impl = await retryAgent(() => agent(IMPL_PROMPT, { label: 'D1-implement-' + V, phase: 'Implement' }))
if (!impl || /NOT-READY/.test(impl)) {
  log('v23 ABORT: implement NOT-READY — no eval spent. ' + (impl || 'null').slice(0, 400))
} else {

// ============================ PHASE 2 — AB88 ============================
phase('AB88')
log('v23: fresh REF88 + PTE88 (cross-run)')

const ref88 = await runEval(V + '_ref88', HARNESS, '[COST]', N_HARD, 'AB88')
if (!ref88.done || ref88.fccs_count !== 0) {
  log('v23 ABORT: REF88 ' + (ref88.done ? 'FCCS confound' : 'incomplete') + ': ' + ref88.note)
} else {
const pte88 = await runEval(V + '_pte88', HARNESS + ' --pte_curves', 'PTE-VID', N_HARD, 'AB88')
if (!pte88.done || pte88.fccs_count !== 0) {
  log('v23 ABORT: PTE88 ' + (pte88.done ? 'FCCS confound' : 'incomplete') + ': ' + pte88.note)
} else if (pte88.marker_count < 80) {
  log('v23 ABORT: PTE-VID markers ' + pte88.marker_count + ' << 88 — FM17 fail')
} else {
log('v23 AB88 runs done: REF mAP=' + ref88.mAP + ' PTE mAP=' + pte88.mAP)

const BOOT88_PROMPT = [
  'You are Agent D-adjudicator for the PTE CROSS-RUN hard-88 A/B. Bash < 150s. ' + RULES,
  'STEP 1 (confound gates): FCCS grep == 0 in BOTH logs; PTE-VID ~88 in the PTE log; [COST] lines present in both.',
  'STEP 2: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_' + V + '_pte88 --ref ' + ROOT + '/results/batch_eval_' + V + '_ref88 --min-delta 0.008 --min-p 0.9 -> point_delta, ci90, p_gt0, new_map, ref_map. Paste the cross-run CAVEAT verbatim.',
  'STEP 3 (FM2 watch): mean detections/video both dirs; flag > +1.0.',
  'STEP 4: per-category AP deltas (4 hard cats) into per_category. fa_rel_inc: Normal-video FA-rate relative increase (there are no Normal videos in the hard-88 — report -1 and say so).',
  'GATES loop1: point_delta >= +0.008 AND p_gt0 >= 0.9 AND new_map >= 0.11306. Report ran, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, det_per_video_new, det_per_video_ref, fa_rel_inc, per_category, note (VERDICT verbatim + caveat).',
].join('\n')

const b88 = await retryAgent(() => agent(BOOT88_PROMPT, { schema: bootSchema, label: 'D-boot88-' + V, phase: 'AB88' }))
const pass88 = !!(b88 && b88.ran && b88.point_delta >= 0.008 && b88.p_gt0 >= 0.9 && b88.new_map >= 0.11306)
log('v23 loop1 gates: pass=' + pass88 + ' delta=' + (b88 && b88.point_delta) + ' P=' + (b88 && b88.p_gt0))

// ============================ PHASE 3 — CONFIRM 290 (conditional) ============================
let b290 = null
if (pass88) {
  phase('Confirm290')
  const ref290 = await runEval(V290 + '_ref290', '', '[COST]', N_FULL, 'Confirm290')
  if (ref290.done) {
    const pte290 = await runEval(V290 + '_pte290', '--pte_curves', 'PTE-VID', N_FULL, 'Confirm290')
    if (pte290.done) {
      const BOOT290_PROMPT = [
        'You are Agent D-adjudicator for the PTE full-290 net-neutrality confirm. Bash < 150s. ' + RULES,
        'STEP 1: FCCS==0 both logs; PTE-VID ~290 in the PTE log.',
        'STEP 2: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_' + V290 + '_pte290 --ref ' + ROOT + '/results/batch_eval_' + V290 + '_ref290 --min-delta -0.003 --min-p 0.5 -> full output. CAVEAT verbatim.',
        'STEP 3: Normal-video FA rate both arms (category==Normal videos with >=1 det conf>0); fa_rel_inc = (new-ref)/ref.',
        'GATES loop2: point_delta >= -0.003 AND new_map >= 0.16454 AND fa_rel_inc <= +0.02. Same schema fields as loop1.',
      ].join('\n')
      b290 = await retryAgent(() => agent(BOOT290_PROMPT, { schema: bootSchema, label: 'D-boot290-' + V290, phase: 'Confirm290' }))
    }
  }
}

// ============================ PHASE 4 — DECIDE + RECORDS ============================
phase('Decide')
const promote = !!(pass88 && b290 && b290.ran && b290.point_delta >= -0.003 && b290.new_map >= 0.16454 && b290.fa_rel_inc <= 0.02)
log('v23 decision: promote=' + promote)

const DECIDE_PROMPT = [
  'You are Agent C (records) for v181 PTE. Write the session records from the data below; re-grep artifacts to verify numbers (FM20). Do NOT re-run evals. ' + RULES,
  'PRECHECK: ' + JSON.stringify(pre),
  'LOOP1 (hard-88 cross-run): ' + JSON.stringify(b88) + ' REF mAP=' + ref88.mAP + ' PTE mAP=' + pte88.mAP,
  'LOOP2 (290, null if not armed): ' + JSON.stringify(b290),
  'WORKFLOW DECISION: promote=' + promote,
  '1. Write ' + ITER + '/summary_v23_pte_20260817.md: verdict; the flat-curve signal-quality motivation (v16 record); the FROZEN 13x4 prompt table (inline it); precheck numbers; loop1 numbers incl. the cross-run caveat verbatim; loop2 numbers or "not armed"; per-cat table; gate tables; FM evidence; decision; family update (precisely what is refuted/validated — e.g. whether text-side ensembling of the CLIP category prompts moves the curve substrate that BUE/CBT consume).',
  '2. Append ONE ledger row to ' + ITER + '/session_v97_ledger.md: loop | ' + V + ' | PTE ... | kept=<true|false> with loop1 (and loop2 if armed) numbers, FM greps, per-cat, and revert/promote state. If loop2 ran, note it under ' + V290 + ' in the same row text (one row only).',
  '3. DECISION EXECUTION:',
  '   - NOT promote: pte_curves stays default False (verify + paste checks); code stays in-tree inert; one-line config closure comment.',
  '   - promote: backup ' + ROOT + '/.qoder/backups/' + V + '_promote/ FIRST; flip pte_curves default True (dataclass + from_args + set_defaults both CLIs) + --no_pte_curves opt-out; validate py_compile + defaults.',
  '4. Report files written + final flag defaults.',
].join('\n')

await retryAgent(() => agent(DECIDE_PROMPT, { label: 'C-records-' + V, phase: 'Decide' }))
log('v23 complete: promote=' + promote + ' loop1_delta=' + (b88 && b88.point_delta) + ' loop2=' + (b290 ? JSON.stringify({ d: b290.point_delta, m: b290.new_map }) : 'not-armed'))
}
}
}
}
