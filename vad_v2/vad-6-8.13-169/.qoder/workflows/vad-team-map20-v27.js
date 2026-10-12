export const meta = {
  name: 'vad-team-map20-v27',
  description: 'VAD agent team v27 — FIVE real full evals (no smoke-only loops) that RE-ADJUDICATE existing default-OFF terminal-slot operators with the EXACT dual-arm paired instrument, plus one minimal principled extension of the proven CBT win. RATIONALE: v183-v200 is 18 consecutive versions with zero promotion, so producing a 19th new idea has low expected yield. Meanwhile v16 measured the same-config cross-run spread at 0.02503 on hard-88 (v167_full 0.12246 vs v169_ab88 0.09743, byte-identical flag strings) — 3x the +0.008 detection target. CONSEQUENCE: any mechanism REVERTED on a cross-run delta smaller than ~0.02 was never actually tested. Several unshipped operators are in exactly that position, their code already in the tree. Re-adjudicating them with the zero-noise dual-arm rig costs one eval each, needs no new mechanism design, and can recover a real win from code that already exists. Baseline just re-confirmed: v26 REF290 = 0.16682 at 290/290 vs authoritative v160_full 0.17254 (delta -0.0057, inside noise). Committed = v137 + qscore + BUE + CBT + cbt_recal. Per-mechanism HARNESS SELECTION IS MANDATORY: CBT fires only on 4 hard cats and SNH/CNV only on Normal videos, so running those on hard-88 would measure a FALSE NULL. Concurrency 2 always.',
  phases: [
    { title: 'Oracle', detail: 'zero-API ceiling decomposition on cached runs: how much headroom remains in boundary vs candidate-set vs confidence-ranking layers, and re-rank the 5-loop roster by measured ceiling' },
    { title: 'Rig', detail: 'generalize the v169 cgr_ab_diag dual-arm rig to an operator-agnostic --ab_ablate <MECH>, self-tested against the known CGR case' },
    { title: 'Loops', detail: 'five loops, each: wire one operator into the rig -> ONE real full eval on the harness where that operator actually fires -> exact paired bootstrap -> keep/close' },
    { title: 'Confirm', detail: 'any KEEP gets a full-290 dual-arm confirm, then promote or archive-close' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const NONHARD_CATS = 'Abuse,Arrest,Arson,Assault,Burglary,Fighting,Robbery,Stealing,Vandalism'
const AB_FIELD = 'ab_arm_off_boundaries'
const AB_CONF_FIELD = 'ab_arm_off_confidences'
const MIN_PAIRED_DELTA = 0.008
const MIN_P = 0.90
const REF290_MAP = 0.17254          // authoritative 290 baseline (v160_full, 290/0)
const REF290_REPRO = 0.16682        // v26 independent repro, 290/0
const NOISE_HARD88 = 0.02503        // measured same-config cross-run spread

// ---- the five loops. harness is chosen by WHERE THE OPERATOR ACTUALLY FIRES ----
const ROSTER = [
  { v: 'v201', mech: 'eubc', flag: '--event_unit_boundary_contraction', marker: '[EUBC',
    harness: '--categories ' + HARD_CATS, n: 88, newCode: false,
    why: 'EUBC = inward contraction of over-wide detections by frame-to-frame visual coherence. Closest cousin of the PROVEN CBT inward clamp, but keyed on a DIFFERENT signal (visual coherence, not the CLIP category curve) and ungated by category. Its original verdict predates the dual-arm rig.' },
  { v: 'v202', mech: 'ece', flag: '--event_continuation_extension', marker: '[ECE]',
    harness: '--categories ' + HARD_CATS, n: 88, newCode: false,
    why: 'ECE = whole-video half-max LEVEL region growing of under-extended detections, with CATEGORY_DURATION_PRIORS reject-cap. Cousin of the PROVEN BUE outward union but grows by level-set rather than outer-inner contrast. Original verdict also predates the rig.' },
  { v: 'v203', mech: 'cbt_all', flag: '--cbt_all_cats', marker: '[CBT-VID]',
    harness: '', n: 290, newCode: true,
    why: 'THE MOST OBVIOUS UNTRIED LEVER: CBT+cbt_recal is the projects second durable win, yet CBT_CAPS covers only 4 of 13 anomaly categories — 52 anomaly videos never see it. Extend the SAME method to the remaining 9 categories. CAPS MUST come from CATEGORY_DURATION_PRIORS (already in pipeline.py, all 13 cats) or train-split durations — NEVER from test GT, which would be test-fitting. Fires only on non-hard cats, so it MUST be judged at 290 (or the non-hard harness); hard-88 would show an exact zero.' },
  { v: 'v204', mech: 'bue_recal', flag: '--bue_recal', marker: '[BUE]',
    harness: '--categories ' + HARD_CATS, n: 88, newCode: true,
    why: 'CBT gained +0.0228 purely from recalibrating its two scalar inputs (caps + trigger). BUE has never had the equivalent treatment: bue_lmax_sec/bue_alpha are single global scalars applied to all categories despite 13x duration-prior spread. Per-category lmax from CATEGORY_DURATION_PRIORS, same method otherwise. Exactly the change that already paid once.' },
  { v: 'v205', mech: 'snh', flag: '--snh_veto', marker: '[SNH',
    harness: '', n: 290, newCode: false,
    why: 'SNH = supervised video-level P(normal) veto (train CV AUC 0.996, transfer confirmed at v21: Normal 0.993 vs Arson 0.002), closed at v21 on GT collateral damage. It zeroes confidence while KEEPING the detection, so under GLOBAL-ranking AP it can lift every anomaly videos rank without deleting anything — a different quantity from the +0.0025 deletion ceiling of banned family 9. Fires on Normal videos, so 290 is mandatory. NOTE: it changes CONFIDENCE not boundaries -> the boundary-only paired field would be an exact-zero FALSE NULL; ' + AB_CONF_FIELD + ' is required.' },
]

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '. Read ' + ROOT + '/PROJECT_DOCUMENTATION.md sections 5, 8.3, 8.4, 8.5 before proposing anything — they encode the operator chain, the noise floor, the coverage trap and the citable baselines.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE (pipeline.py v182): v137 + qscore_rerank + bue_enabled + bue_closure_gate + cbt_enabled + cbt_recal + classify_skip_class_agnostic, all DEFAULT-ON. Authoritative full-290 = ' + REF290_MAP + ' (batch_eval_v160_full, 290/0 failures); independently reproduced ' + REF290_REPRO + ' (batch_eval_v26_full290, 290/0, delta -0.0057 = inside noise). Unshipped default-OFF: ccr, cex_extend, cgr, cnv_veto, event_continuation_extension, event_unit_boundary_contraction, snh_veto. DO NOT change any committed default. !!!',
'',
'!!! WHY THIS SESSION RE-TESTS OLD CODE INSTEAD OF INVENTING NEW MECHANISMS !!!',
'v183-v200 = 18 consecutive versions, zero promotions. Most died at research/precheck for lack of an auditable localized-supervision validator. So a 19th new idea is low-yield. Meanwhile v16 measured the SAME-CONFIG cross-run spread at ' + NOISE_HARD88 + ' on hard-88 (v167_full 0.12246 vs v169_ab88 0.09743, byte-identical flags) — 3x the +' + MIN_PAIRED_DELTA + ' target. Therefore ANY mechanism reverted on a cross-run delta below ~0.02 was NEVER ACTUALLY TESTED. Several such operators sit in the tree already implemented. Re-adjudicating them with the exact dual-arm rig is cheap, decisive, and can recover a real win with no new design.',
'',
'!!! THE INSTRUMENT — exact paired, zero VLM noise (PROJECT_DOCUMENTATION 8.3.2) !!!',
'Every terminal-slot operator runs DOWNSTREAM of all VLM calls on cached motion-energy / CLIP curves, i.e. deterministic post-processing. So both A/B arms are computable inside ONE eval: arm-OFF = replay the remaining chain on a snapshot taken before the operator; arm-ON = the runs real output. Same videos, same VLM sampling, same detection count; the arms differ by exactly ONE function call. Persist arm-OFF into ' + AB_FIELD + ' (and ' + AB_CONF_FIELD + ' whenever the operator touches confidence), index-aligned and same length, then adjudicate with the existing tool:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-json ' + AB_FIELD + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P,
'DUAL-ARM APPLICABILITY: verified compatible = ECE, EUBC, CBT, BUE, qscore, CNV, SNH (all 0 VLM refs). NOT compatible = CEX (binary VLM continuation probe) and FSD (rewrites the scan grid upstream of the VLM). Never attempt a dual-arm on those two.',
'IMPLEMENTATION HAZARD (verified, fails SILENTLY with plausible-looking numbers): _cgr_motion_relocate / _bue_boundary_union_extension / _cbt_category_boundary_tighten all MUTATE a["t_start"]/a["t_end"] IN PLACE. copy.deepcopy before EACH arm, and prove arm independence in the self-test (a video where the operator fired 0 times must have arm-OFF byte-identical to final; one where it fired must differ).',
'CONFIDENCE-VETO TRAP: SNH/CNV change confidence and leave boundaries untouched, so comparing only ' + AB_FIELD + ' yields an EXACT-ZERO FALSE NULL. Those loops MUST compare ' + AB_CONF_FIELD + '. Validate by construction: a video where the veto did not fire must give delta exactly 0.',
'',
'!!! HARNESS SELECTION IS PART OF CORRECTNESS (this has silently voided loops before) !!!',
'Run each operator on a harness where it actually fires, and verify num_videos in the report:',
'  hard-88  (--categories ' + HARD_CATS + ') = 88 videos: EUBC, ECE, BUE-recal (fire on any anomaly detection)',
'  full-290 (no --categories) = 290 videos: CBT-ALL (fires only on the 9 NON-hard cats, absent from hard-88) and SNH (fires on the 150 Normal videos, absent from hard-88)',
'A mechanism evaluated on a harness where it cannot fire produces an exact zero and looks like a clean null. That is a VOID loop, not a result.',
'',
'!!! COVERAGE GATE (PROJECT_DOCUMENTATION 8.4 — cost this project a whole round) !!!',
'evaluation_report.json carries no completeness field, and evaluate() drops videos that have no prediction file TOGETHER WITH THEIR GT, which INFLATES mAP. v26 measured this exactly: 24/290 failures (concentrated in 19 Shoplifting + 5 Shooting) reported 0.17673 while the true complete value was 0.16682 (+0.00991 inflation, enough to flip a -0.0057 regression into a +0.0042 "gain"). EVERY adjudication in this session MUST first assert timing.failed == 0 and num_videos == the expected harness size. If failed > 0, RESUME the same --output_dir (existing per-video JSONs are SKIPped; v26 refilled 24 videos in 17.9min and all 24 passed first try) and only then adjudicate.',
'',
'!!! GATES: KEEP requires paired delta >= +' + MIN_PAIRED_DELTA + ' AND P(delta>0) >= ' + MIN_P + ' on the mechanisms own harness. A KEEP then needs a full-290 dual-arm confirm that is net-neutral-or-positive before promotion. Anything else is CLOSED and archived, and a clean negative that closes a lever is a SUCCESSFUL loop — report the honest number either way. !!!',
'',
'FM DISCIPLINE: FM2 count-frozen (assert ndet identical across arms). FM5 marker must actually fire. FM6 offline gains do not transfer — that is why every loop here ends in a real eval. FM13 bound the ceiling first (that is Phase Oracle). FM17 per-video marker on EVERY path incl. empty/0-detection and early-return-normal. FM18 report the ABSOLUTE mAP, not only the delta. FM20 paired field must be WRITTEN and GREPPED in a produced JSON — a claim is not evidence. FM21 marker string must match the orchestrator grep EXACTLY.',
'HARD CONSTRAINTS: one mechanism per loop; new flags default OFF and flag-off byte-identical (from_args(Namespace()).scoring == ScoringConfig()); CLI in BOTH run.py and batch_evaluate.py; NEVER edit evaluate.py or analysis_bootstrap_ab.py; NEVER run two evals concurrently; eval always --concurrency ' + CONCURRENCY + '. StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s except a stated self-test smoke; Read with offset/limit; keep progress lines short.',
].join('\n')

const oracleSchema = { type: 'object', properties: { ran: { type: 'boolean' }, boundary_ceiling: { type: 'number', description: 'oracle dmAP from perfect boundaries, detection set + confidence order frozen' }, candidate_ceiling: { type: 'number', description: 'oracle dmAP from optimal add/remove of detections' }, ranking_ceiling: { type: 'number', description: 'oracle dmAP from perfect global confidence ordering, windows frozen' }, recommended_order: { type: 'string', description: 'the five roster mechs re-ranked by measured ceiling, most promising first, with one clause of justification each' }, drop: { type: 'string', description: 'any roster entry whose ceiling is provably below the +' + MIN_PAIRED_DELTA + ' gate, with the number' }, note: { type: 'string' } }, required: ['ran', 'recommended_order', 'note'], additionalProperties: false }
const rigSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, selftest_evidence: { type: 'string', description: 'PASTED output: field present in a produced JSON with values and len == len(anomalies); arm independence (fired=0 identical, fired=1 differs); ndet equal across arms' }, usage: { type: 'string', description: 'the exact CLI form later loops must pass' }, files_changed: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'selftest_evidence', 'usage', 'validation'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string', description: 'FULL flag string; do NOT include --categories (the launcher adds the harness)' }, paired_field: { type: 'string', description: AB_FIELD + ' or ' + AB_CONF_FIELD + ' — whichever the operator actually changes' }, files_changed: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'activation_marker', 'eval_flags', 'paired_field', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, num_videos: { type: 'number' }, failed: { type: 'number' }, marker_count: { type: 'number' }, paired_jsons: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, coverage_ok: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, per_category: { type: 'string' }, count_neutral: { type: 'boolean' }, fired_videos: { type: 'number' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
const recSchema = { type: 'object', properties: { version: { type: 'string' }, kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' } }, required: ['version', 'kept', 'summary', 'lessons'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}
function launchPrompt(V, tag, harness, evalFlags) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const h = harness ? (harness + ' ') : ''
  return [
'You are Agent D-launcher for ' + V + ' (' + tag + '). Ensure the eval is running with the REQUIRED flags, then return IMMEDIATELY. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + dir + ' log=' + dir + '.log report=' + dir + '/evaluation_report.json',
'REQUIRED FLAGS (verbatim): ' + h + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep — NEVER run two evals at once. 2. Already running for this V_tag -> adopt if flags match else kill and relaunch. 3. report exists AND log shows these flags -> done. 4. A DIFFERENT eval is running -> sleep 140 (up to 3x) then report blocked.',
'5. Otherwise write ' + dir + '/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + dir + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category 999 --concurrency " + CONCURRENCY + " " + h + evalFlags + " > " + dir + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log for an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function watchPrompt(V, tag, marker, n, pairedField) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const rep = dir + '/evaluation_report.json'
  return [
'You are Agent D-watcher for ' + V + ' (' + tag + '), expecting ' + n + ' videos. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE. marker="' + marker + '".',
'1. If ' + rep + ' exists: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'timing\'];print(d[\'evaluation\'][\'temporal_map\'][\'mAP\'],t.get(\'num_videos\'),t.get(\'failed\'))" -> put these in mAP, num_videos, failed.',
'   COVERAGE GATE: num_videos MUST be ' + n + ' and failed MUST be 0. If failed > 0, RESUME (relaunch the SAME nohup on the SAME --output_dir; completed per-video JSONs are SKIPped) and keep watching — do NOT return done=true with failed > 0.',
'   markers: grep -cF "' + marker + '" ' + dir + '.log and ' + dir + '/run.log (trust the LARGER).',
'   FM20: PJ=$(grep -l ' + pairedField + ' ' + dir + '/*.json 2>/dev/null | grep -v evaluation_report | wc -l) -> paired_jsons=PJ, note "pairedJSONs=<PJ>". PJ==0 means VOID.',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 (up to 2x) then done=false note="running <n>/' + n + '" using the count of result JSONs. Stalled >30min with no new JSON -> kill and relaunch the same nohup.',
'3. Dead with no report -> tail -30. TRANSIENT (API/network) -> relaunch. CODE BUG -> crashed=true with the traceback.',
'Return done, crashed, mAP, num_videos, failed, marker_count, paired_jsons, note.',
  ].join('\n')
}
async function runEval(V, tag, harness, evalFlags, marker, n, pairedField, cap, ph) {
  const l = await retryAgent(() => agent(launchPrompt(V, tag, harness, evalFlags), { schema: launchSchema, label: 'D-launch-' + V, phase: ph }))
  if (!l || !l.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, paired_jsons: -1, note: 'launch failed: ' + (l ? l.note : 'null') }
  for (let w = 0; w < (cap || 90); w++) {
    const st = await retryAgent(() => agent(watchPrompt(V, tag, marker, n, pairedField), { schema: watchSchema, label: 'D-watch-' + V + '-' + w, phase: ph }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, marker_count: -1, paired_jsons: -1, note: 'watcher cap' }
}

// ============================ MAIN ============================
log('v27 starting. FIVE real full evals re-adjudicating existing terminal-slot operators with the exact dual-arm rig. Baseline just re-confirmed at 290/290: v26 ' + REF290_REPRO + ' vs authoritative ' + REF290_MAP + '. Noise floor (same-config, hard-88) = ' + NOISE_HARD88 + ' -> cross-run verdicts under ~0.02 are meaningless, which is why these levers get a second, honest test.')

// ---------- Phase Oracle: bound the ceilings before spending eval time ----------
phase('Oracle')
const oracle = await retryAgent(() => agent([
'You are Agent O. ZERO new evals, ZERO API calls — work only on cached runs. Bound the remaining headroom per layer so the five loops target measured ceilings instead of hope (FM13). StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'CACHES: results/batch_eval_v26_full290 (290/290, 0 failures, the fresh committed baseline — USE THIS as the primary substrate), results/batch_eval_v160_full (290/0), results/batch_eval_v169_ab88 (hard-88 with both dual-arm fields), results/batch_eval_v155_full, features/test.',
'COMPUTE three oracle ceilings on the committed 290 detections, each holding the other two layers FIXED (use evaluate.load_ground_truth / load_predictions / compute_temporal_map; never modify evaluate.py):',
'  (a) BOUNDARY ceiling: keep the detection set and confidence order; replace each windows [t_start,t_end] with the best achievable window for that video (e.g. snap to the GT segment maximizing IoU). dmAP vs committed = the ceiling for ANY boundary operator (EUBC, ECE, BUE-recal, CBT-ALL all live under this).',
'  (b) CANDIDATE-SET ceiling: keep boundaries and confidences; optimally add missing GT-covering detections and/or drop non-matching ones. dmAP = ceiling for candidate-generation levers.',
'  (c) RANKING ceiling: keep every window; reorder confidence globally to the optimal ranking. dmAP = ceiling for confidence operators (SNH lives here).',
'Also report, for the 52 NON-hard-category anomaly videos specifically, how many carry a detection whose duration exceeds its CATEGORY_DURATION_PRIORS-implied cap — that is the direct addressable population for loop v203 CBT-ALL, and if it is ~0 the loop is pointless.',
'THEN re-rank the five roster mechanisms by measured ceiling and flag any whose ceiling is provably below the +' + MIN_PAIRED_DELTA + ' gate:',
JSON.stringify(ROSTER.map(r => ({ v: r.v, mech: r.mech, harness: r.harness || 'full-290', n: r.n, why: r.why })), null, 1),
'Be explicit when an oracle is loose vs tight. A ceiling is an upper bound: if (a) is below the gate, every boundary loop in the roster is dead and you must say so plainly — that is a valuable finding, not a failure.',
'Return ran, boundary_ceiling, candidate_ceiling, ranking_ceiling, recommended_order, drop, note.',
].join('\n'), { schema: oracleSchema, label: 'O-ceilings', phase: 'Oracle' }))
log('Oracle: boundary=' + (oracle ? oracle.boundary_ceiling : '?') + ' candidate=' + (oracle ? oracle.candidate_ceiling : '?') + ' ranking=' + (oracle ? oracle.ranking_ceiling : '?') + ' | order: ' + (oracle ? String(oracle.recommended_order).slice(0, 300) : 'n/a') + (oracle && oracle.drop ? (' | DROP: ' + String(oracle.drop).slice(0, 200)) : ''))

// ---------- Phase Rig: generalize the dual-arm instrument once ----------
phase('Rig')
const rig = await retryAgent(() => agent([
'You are Agent R. Generalize the existing v169 dual-arm rig from CGR-only to OPERATOR-AGNOSTIC, so the five loops below reuse one proven instrument instead of each re-implementing it. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'TODAY: pipeline.py has cgr_ab_diag, which snapshots the pre-CGR detection list and replays the remaining terminal chain to produce arm-OFF into ' + AB_FIELD + '. models.py already serializes that field.',
'BUILD: one new config field ab_ablate (default "" = OFF, byte-identical) + CLI --ab_ablate <MECH> in BOTH run.py and batch_evaluate.py, accepting eubc | ece | cbt_all | bue_recal | snh. When set, at the terminal slot: base = copy.deepcopy(final_anomalies) taken BEFORE the named operator; arm-OFF = replay the remaining chain with ONLY that operator skipped; arm-ON = the normal path (the runs real output). Persist arm-OFF boundaries into ' + AB_FIELD + ' and arm-OFF confidences into ' + AB_CONF_FIELD + ', BOTH index-aligned and the same length as anomalies. Writing both makes the rig correct for boundary operators AND confidence-veto operators (SNH/CNV), where a boundary-only field would be an exact-zero false null.',
'DEEP-COPY DISCIPLINE: the operators mutate a["t_start"]/a["t_end"] in place — deepcopy before EACH arm or the arms silently corrupt each other with plausible-looking numbers.',
'LOGGING: the arm-OFF replay re-runs BUE/CBT and would double-fire their markers, corrupting FM5 counts. Suppress or distinctly tag logging during the arm-OFF pass and state exactly what you did.',
'SELF-TEST (paste ACTUAL output; a false pass here wastes a multi-hour eval): run --ab_ablate cgr against the KNOWN-GOOD case with the smallest smoke that produces at least one fire —',
'  ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_v27rig_selftest --max_per_category 2 --concurrency 1 --categories ' + HARD_CATS + ' --cgr_relocate --ab_ablate cgr',
'  CONFIRM and paste: (a) both fields present in a produced JSON with real values and len == len(anomalies); (b) ARM INDEPENDENCE — a video whose marker shows the operator fired 0 times has arm-OFF byte-identical to final, and one where it fired differs (widen max_per_category until at least one fire is observed; an all-zero-fire smoke proves nothing); (c) ndet identical across arms.',
'  Structural agreement with v169 is the bar (bit-exact cross-run reproduction is impossible — different VLM sampling — so do NOT claim it).',
'VALIDATE: py_compile; import MultiAnomalyVAD; --ab_ablate in --help of both runners; ScoringConfig() defaults unchanged; from_args(Namespace()).scoring == ScoringConfig().',
'Return implemented, selftest_passed, selftest_evidence, usage, files_changed, validation, note.',
].join('\n'), { schema: rigSchema, label: 'R-generalize-rig', phase: 'Rig' }))
const rigOk = !!(rig && rig.implemented && rig.selftest_passed)
log('Rig: ' + (rigOk ? ('OK — ' + String(rig.usage).slice(0, 160)) : ('NOT verified (' + (rig ? rig.selftest_passed : 'null') + ') -> each loop must build its own dual-arm following the v169 pattern')))

// ---------- Phase Loops: five real evals ----------
phase('Loops')
const results = []
for (let i = 0; i < ROSTER.length; i++) {
  const R = ROSTER[i]
  log('=== Loop ' + (i + 1) + '/' + ROSTER.length + ' -> ' + R.v + ' ' + R.mech.toUpperCase() + ' on ' + (R.harness ? ('hard-' + R.n) : ('full-' + R.n)) + ' ===')

  const impl = await retryAgent(() => agent([
'You are Agent D1 for ' + R.v + ' (' + R.mech + '). ' + (R.newCode ? 'Implement a NEW minimal mechanism' : 'The mechanism ALREADY EXISTS in the tree (' + R.flag + ') — your job is only to wire it into the dual-arm rig and prove the instrument') + ', then run the self-test. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'MECHANISM: ' + R.mech + '  |  flag: ' + R.flag + '  |  harness: ' + (R.harness || 'full-290 (no --categories)') + ' (' + R.n + ' videos)  |  marker: ' + R.marker,
'WHY THIS LOOP: ' + R.why,
'ORACLE FINDINGS (use them; if they say this lever is capped below the gate, say so and return implemented=false rather than burn an eval): ' + (oracle ? JSON.stringify({ boundary: oracle.boundary_ceiling, candidate: oracle.candidate_ceiling, ranking: oracle.ranking_ceiling, order: oracle.recommended_order, drop: oracle.drop }) : 'oracle unavailable'),
'RIG: ' + (rigOk ? ('generalized rig available — ' + rig.usage) : 'rig NOT available — implement a per-mechanism dual-arm following the v169 cgr_ab_diag pattern (deepcopy per arm; persist ' + AB_FIELD + ' and, if the operator touches confidence, ' + AB_CONF_FIELD + ')'),
R.newCode
  ? 'NEW CODE RULES: one new flag, default OFF, flag-off byte-identical; CLI in BOTH runners; reuse the EXISTING method and change only its scalar inputs / category gate — do not rewrite the algorithm. Any per-category constant MUST come from CATEGORY_DURATION_PRIORS (pipeline.py, all 13 categories) or official TRAIN-split durations. NEVER derive a constant from test GT — that is test-fitting and voids the result.'
  : 'EXISTING CODE: do NOT modify the mechanism itself. Only add the dual-arm wiring and (if missing) an unconditional per-video marker.',
'FM17: ensure an unconditional per-video marker on EVERY return path including empty/0-detection and early-return-normal, string EXACTLY as you report in activation_marker.',
'SELF-TEST (paste ACTUAL output): a real smoke on ' + (R.harness || 'a small full-290 slice') + ' at --concurrency 1 with ' + R.flag + ' + the dual-arm, confirming (a) marker grep > 0, (b) the paired field present in a produced JSON with len == len(anomalies), (c) arm independence (0-fire video identical, firing video differs), (d) ndet identical across arms. Set selftest_passed only if all pass.',
'PAIRED FIELD CHOICE: report ' + AB_CONF_FIELD + ' if this operator changes CONFIDENCE (SNH/CNV) and ' + AB_FIELD + ' if it changes BOUNDARIES. Getting this wrong yields an exact-zero false null.',
'VALIDATE: py_compile; import MultiAnomalyVAD; flag in --help of both runners; committed defaults unchanged; from_args(Namespace()).scoring == ScoringConfig().',
'Return implemented, selftest_passed, instrument_selftest, activation_marker, eval_flags, paired_field, files_changed, validation, note.',
  ].join('\n'), { schema: implSchema, label: 'D1-' + R.v, phase: 'Loops' }))

  if (!impl || !impl.implemented || impl.selftest_passed === false || !impl.activation_marker) {
    log('Loop ' + (i + 1) + ' ' + R.v + ': impl/self-test not verified -> skipping the eval to avoid a void loop. ' + (impl ? String(impl.note || impl.instrument_selftest || '').slice(0, 200) : 'null'))
    results.push({ v: R.v, mech: R.mech, kept: false, reason: 'impl/selftest not verified', boot: null })
    continue
  }

  const ev = await runEval(R.v, 'ab', R.harness, impl.eval_flags, impl.activation_marker, R.n, impl.paired_field, R.n > 100 ? 140 : 90, 'Loops')
  log('Loop ' + (i + 1) + ' ' + R.v + ': done=' + ev.done + ' crashed=' + ev.crashed + ' mAP=' + ev.mAP + ' n=' + (ev.num_videos != null ? ev.num_videos : '?') + ' failed=' + (ev.failed != null ? ev.failed : '?') + ' markers=' + ev.marker_count + ' pairedJSONs=' + (ev.paired_jsons != null ? ev.paired_jsons : '?'))

  let boot = null
  const usable = ev.done && !ev.crashed && ev.marker_count > 0 &&
                 !(ev.paired_jsons != null && ev.paired_jsons <= 0) &&
                 !(ev.failed != null && ev.failed > 0) &&
                 !(ev.num_videos != null && ev.num_videos !== R.n)
  if (usable) {
    boot = await retryAgent(() => agent([
'You are Agent D-adjudicator for ' + R.v + ' (' + R.mech + ') on ' + R.n + ' videos. The run carries BOTH arms: its output is arm-ON and field ' + impl.paired_field + ' holds arm-OFF. Exact paired, zero VLM noise. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'STEP 1 COVERAGE GATE: read evaluation_report.json and assert timing.failed == 0 and timing.num_videos == ' + R.n + '. If either fails set coverage_ok=false, ran=false and STOP — a report with failures has an INFLATED mAP (PROJECT_DOCUMENTATION 8.4). Do not adjudicate it.',
'STEP 2 FM20 GATE: grep -l ' + impl.paired_field + ' <dir>/*.json | grep -v evaluation_report | wc -l . If 0 -> ran=false note="FM20 VOID". NEVER substitute a cross-run comparison — the noise floor is ' + NOISE_HARD88 + ' and would swamp the effect.',
'STEP 3 the decisive number: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + ROOT + '/results/batch_eval_' + R.v + '_ab --paired-json ' + impl.paired_field + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P + ' -> point_delta = this operators clean effect on FINAL mAP including downstream interactions; new_map = absolute with it ON; ref_map = arm-OFF absolute.',
'STEP 4 report how many videos the operator actually FIRED on (from the marker lines) -> fired_videos. If that is 0 the loop is VOID, not a null result: say so — it means the harness or the gate is wrong.',
'STEP 5 per-category AP deltas across the arms, and count-neutrality (mean detections-per-video identical) -> count_neutral.',
'Return ran, coverage_ok, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, per_category, count_neutral, fired_videos, note (VERDICT verbatim).',
    ].join('\n'), { schema: bootSchema, label: 'D-boot-' + R.v, phase: 'Loops' }))
  } else {
    log('Loop ' + (i + 1) + ' ' + R.v + ': eval unusable (coverage/marker/paired gate) -> no adjudication. ' + String(ev.note || '').slice(0, 200))
  }

  const pass = !!(boot && boot.ran && boot.point_delta >= MIN_PAIRED_DELTA && boot.p_gt0 >= MIN_P && (boot.fired_videos == null || boot.fired_videos > 0))
  const decision = boot && boot.ran
    ? (pass ? ('KEEP — paired +' + boot.point_delta.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ' abs=' + boot.new_map.toFixed(5))
            : ('CLOSE — paired ' + boot.point_delta.toFixed(5) + ' P=' + boot.p_gt0.toFixed(3) + ' (gate: >=+' + MIN_PAIRED_DELTA + ' and P>=' + MIN_P + ')'))
    : 'CLOSE — no valid adjudication'
  log('Loop ' + (i + 1) + ' ' + R.v + ': ' + decision)

  const rec = await retryAgent(() => agent([
'You are Agent D4 (recorder) for ' + R.v + ' (' + R.mech + '). Record honestly — a clean negative that closes a lever is a successful loop, and this project has been burned by over-optimistic verdicts. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'IMPL: ' + JSON.stringify(impl) + '\nEVAL: ' + JSON.stringify(ev) + '\nADJUDICATION: ' + JSON.stringify(boot) + '\nDECISION: ' + decision,
pass ? '1. KEEP: leave the code, keep the flag DEFAULT-OFF for now (promotion happens only after the full-290 confirm in the final phase). Record the exact numbers.'
     : '1. CLOSE: restore from ' + ROOT + '/.qoder/backups/' + R.v + '/ if you added code, diff -q to confirm, and verify the committed defaults are untouched (ScoringConfig() bue/cbt/cbt_recal/qscore True; the mechanism flag False). Mark the lever MEASURED-CLOSED so no future session re-litigates it, stating the instrument was an exact paired end-to-end A/B rather than a noisy proxy.',
'2. date +%Y%m%d; write ' + LOGS + '/summary_' + R.v + '_' + R.mech + '_<YYYYMMDD>.md: mechanism and why it was re-tested now (the ' + NOISE_HARD88 + ' noise floor invalidated its original cross-run verdict), the harness and WHY that harness (where the operator fires), the coverage gate result (failed==0, num_videos==' + R.n + '), fired_videos, the paired delta with CI and P, absolute arm-ON/arm-OFF mAP, per-category deltas, count-neutrality, and the decision.',
'3. APPEND one ledger row to ' + LOGS + '/session_v97_ledger.md matching the existing 9-column format (| loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept |), using -1 for absent values. Do NOT rewrite existing rows.',
'Return version, kept, summary, lessons.',
  ].join('\n'), { schema: recSchema, label: 'D4-' + R.v, phase: 'Loops' }))
  results.push({ v: R.v, mech: R.mech, kept: pass, boot, rec: rec ? rec.summary : null })
}

// ---------- Phase Confirm ----------
phase('Confirm')
const kept = results.filter(r => r.kept)
log('Loops complete. KEEP count = ' + kept.length + ' / ' + ROSTER.length + (kept.length ? (' -> ' + kept.map(k => k.v + ':' + k.mech).join(', ')) : ''))
let confirm = null
if (kept.length) {
  const best = kept.slice().sort((a, b) => (b.boot ? b.boot.point_delta : -9) - (a.boot ? a.boot.point_delta : -9))[0]
  log('Confirming the strongest KEEP (' + best.v + ' ' + best.mech + ', paired +' + (best.boot ? best.boot.point_delta.toFixed(5) : '?') + ') at full-290')
  const c = await runEval(best.v + 'c', 'confirm290', '', (best.boot && best.boot.mode ? '' : '') + ROSTER.find(r => r.v === best.v).flag + ' --ab_ablate ' + best.mech, ROSTER.find(r => r.v === best.v).marker, 290, AB_FIELD, 160, 'Confirm')
  log('Confirm290: done=' + c.done + ' mAP=' + c.mAP + ' n=' + (c.num_videos != null ? c.num_videos : '?') + ' failed=' + (c.failed != null ? c.failed : '?'))
  confirm = await retryAgent(() => agent([
'You are Agent C. Decide promotion for ' + best.v + ' (' + best.mech + ') and write the v27 session summary. Plain text.',
'', CONTEXT, '',
'HARD-HARNESS RESULT: ' + JSON.stringify(best.boot) + '\nFULL-290 CONFIRM: ' + JSON.stringify(c) + '\nALL LOOPS: ' + JSON.stringify(results.map(r => ({ v: r.v, mech: r.mech, kept: r.kept, delta: r.boot ? r.boot.point_delta : null, p: r.boot ? r.boot.p_gt0 : null }))) + '\nORACLE: ' + JSON.stringify(oracle),
'1. COVERAGE GATE the 290 confirm first (failed==0, num_videos==290). If it failed, RESUME the same output_dir before judging — never judge an incomplete 290 (PROJECT_DOCUMENTATION 8.4).',
'2. Promote to default-ON only if the 290 paired effect is net-neutral-or-positive AND the absolute 290 mAP does not fall below ' + (REF290_MAP - 0.008).toFixed(5) + '. If promoting: back up config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/' + best.mech + '_promote/, flip the default in the dataclass AND the from_args getattr AND set_defaults in both runners, keep an opt-out flag, and PASTE validation (py_compile; the flag True; bue/cbt/cbt_recal/qscore still True; import MultiAnomalyVAD). Otherwise leave default-OFF and confirm it.',
'3. Turn --ab_ablate back off by default and confirm; leave the rig in the tree as reusable.',
'4. date +%Y%m%d; write ' + LOGS + '/summary_v27_' + ROSTER[0].v + '-' + ROSTER[ROSTER.length - 1].v + '_<YYYYMMDD>.md: the oracle ceilings and how they redirected the roster; each loops harness/fired_videos/paired delta/CI/P; the promotion decision; and an updated CLOSED-families list. Be explicit that these verdicts supersede the original cross-run ones because the noise floor is ' + NOISE_HARD88 + '.',
'5. APPEND ledger rows for the confirm and any promotion. Do NOT rewrite rows.',
'6. Reply with a 12-line recap leading with the single strongest number.',
  ].join('\n'), { label: 'C-confirm-promote', phase: 'Confirm' }))
} else {
  log('No KEEP -> nothing to confirm. Writing the closure summary.')
  confirm = await retryAgent(() => agent([
'You are Agent C. All five loops closed. Write the v27 closure summary — this is a legitimate, valuable outcome: five levers were tested with an instrument that cannot be fooled by VLM noise, and their original verdicts are now either confirmed or corrected on solid evidence. Plain text.',
'', CONTEXT, '',
'ALL LOOPS: ' + JSON.stringify(results.map(r => ({ v: r.v, mech: r.mech, kept: r.kept, delta: r.boot ? r.boot.point_delta : null, p: r.boot ? r.boot.p_gt0 : null, fired: r.boot ? r.boot.fired_videos : null, note: r.boot ? String(r.boot.note || '').slice(0, 300) : r.reason }))) + '\nORACLE: ' + JSON.stringify(oracle),
'1. Confirm the committed defaults are untouched and every mechanism flag is still default-OFF — PASTE the output.',
'2. Turn --ab_ablate off by default and confirm; leave the rig in the tree.',
'3. date +%Y%m%d; write ' + LOGS + '/summary_v27_' + ROSTER[0].v + '-' + ROSTER[ROSTER.length - 1].v + '_<YYYYMMDD>.md with the oracle ceilings, each loops harness / fired_videos / paired delta / CI / P, and an updated CLOSED-families list. State plainly which layer the oracle showed still has headroom (if any) and what instrument a future session would need — especially if the binding constraint is the absence of auditable localized train supervision rather than a missing mechanism.',
'4. APPEND ledger rows for any loop D4 did not already record. Do NOT rewrite rows.',
'5. Reply with a 12-line recap.',
  ].join('\n'), { label: 'C-closure', phase: 'Confirm' }))
}

log('v27 complete. ' + results.map(r => r.v + ':' + r.mech + '=' + (r.kept ? 'KEEP' : 'close') + (r.boot && r.boot.ran ? ('(' + r.boot.point_delta.toFixed(5) + ',P=' + r.boot.p_gt0.toFixed(3) + ')') : '')).join(' | '))
return { oracle, rigOk, results: results.map(r => ({ v: r.v, mech: r.mech, kept: r.kept, delta: r.boot ? r.boot.point_delta : null, p: r.boot ? r.boot.p_gt0 : null, fired: r.boot ? r.boot.fired_videos : null })), keptCount: kept.length }
