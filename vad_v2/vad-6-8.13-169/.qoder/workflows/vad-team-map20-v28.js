export const meta = {
  name: 'vad-team-map20-v28',
  description: 'VAD agent team v28 — ONE mechanism: v206 ECE-TERMINAL. Move the existing ECE operator from the FRONT of the terminal chain (pipeline.py L2474) to the TERMINAL-MOST slot (after CBT, L2658), then adjudicate with the exact dual-arm paired rig on hard-88, and add a full-290 confirm only if the gate is cleared. WHY: v202 re-adjudicated ECE at its current front slot and measured an EXACT ZERO — 31 [ECE] lines with 17 material extensions, yet arm-OFF was byte-identical to the final output on 12/12 videos and 19/19 detections (independently re-verified). The cause is structural, not mechanical: BUE runs after ECE, grows outward to the contrast-optimal window and takes the UNION, so it already occupies ECE move set and erases every ECE extension. That also exposed a blind spot in the mechanism-scoped oracle, which scored ECE #1 of the whole v27 roster (+0.09829 ceiling, 106 dets / 97 videos, expected +0.0135-0.0161) by treating the slot as EMPTY. At the terminal slot the slot genuinely IS empty (CEX/CNV/SNH are all default-OFF), so the ceiling becomes reachable in principle — but ECE would then grow outward immediately AFTER CBT clamped inward, partially undoing a committed win worth +0.0599 on hard-88 together with BUE. Net effect is therefore genuinely unknown and worth one real eval. Committed = v137 + qscore + BUE + CBT + cbt_recal; authoritative full-290 0.17254 (v160_full), independently reproduced 0.16682 (v26, 290/0). Concurrency 2.',
  phases: [
    { title: 'Precheck', detail: 'zero-API deterministic offline replay of ECE-at-terminal on cached final boundaries — because nothing runs downstream of the terminal slot this replay is EXACT, not a ceiling; GO/NO-GO before spending any eval time' },
    { title: 'Implement', detail: 'new flag --ece_terminal (default OFF, byte-identical off) relocating the ECE call, mirrored in the arm-OFF replay, with the ECE hard-gate marker preserved on every path' },
    { title: 'Verify88', detail: 'one real hard-88 dual-arm eval -> exact paired bootstrap' },
    { title: 'Confirm290', detail: 'only if hard-88 clears the gate: full-290 dual-arm confirm, then promote or close' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const CONCURRENCY = 2
const V = 'v206'
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS_HARD = '--categories ' + HARD_CATS
const N_HARD = 88
const N_FULL = 290
const MECH_BASE = '--event_continuation_extension --ece_terminal'
const AB_FIELD = 'ab_arm_off_boundaries'
const AB_MECH = 'ece_terminal'
const MIN_PAIRED_DELTA = 0.008
const MIN_P = 0.90
const REF290_MAP = 0.17254
const REF290_REPRO = 0.16682
const NOISE_HARD88 = 0.02503

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '. READ ' + ROOT + '/PROJECT_DOCUMENTATION.md sections 5, 8.3.1, 8.3.2, 8.4, 8.5 FIRST — they encode the operator chain, the measured noise floor, the dual-arm rig, the coverage trap and the citable baselines.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up every file you touch to ' + ROOT + '/.qoder/backups/' + V + '/ BEFORE editing and ls the backup dir to confirm.',
'',
'!!! COMMITTED PIPELINE (pipeline.py header v182+): v137 + qscore_rerank + bue_enabled + bue_closure_gate + cbt_enabled + cbt_recal + classify_skip_class_agnostic, all DEFAULT-ON. Authoritative full-290 = ' + REF290_MAP + ' (batch_eval_v160_full, 290/0); independently reproduced ' + REF290_REPRO + ' (batch_eval_v26_full290, 290/0). Everything else is default-OFF. DO NOT change any committed default. !!!',
'',
'!!! THE EXACT CHAIN (verified line numbers; confirm before editing, they drift) !!!',
'MAIN PATH: L2474 ECE -> L2482 EUBC -> L2517 qscore -> L2575 CGR -> L2604 CCR -> L2642 BUE -> L2658 CBT -> L2876 ab_ablate arm-OFF replay -> (CEX/CNV/SNH, all default-OFF).',
'ARM-OFF REPLAY: _ab_ablate_arm_off_replay is defined at L2997 and MIRRORS the same order at L3094 ECE / L3096 EUBC / L3117 qscore / L3125 CGR / L3127 CCR / L3129 BUE / L3132 CBT.',
'ECE method itself: _event_continuation_extension at L3481.',
'',
'!!! WHAT v202 ESTABLISHED (do not re-litigate it; build on it) !!!',
'ECE at its CURRENT front slot has an EXACT-ZERO effect on final mAP. Real 12-video hard-cat smoke: 31 [ECE] lines, 17 MATERIAL extensions, yet arm-OFF byte-identical to final on 12/12 videos and 19/19 detections (independently re-verified from the JSONs). The rig was PROVEN not to be the cause: [c-diff] arms differ on a firing detection (arm-ON [[40.0,60.0]] vs arm-OFF [[46.0,48.0]]), [c-zero] byte-identical on a 0-fire detection, [c-alias] caller snapshot unmutated, [c-cfg] arms differ by exactly the one flag, [c-len] count frozen; plus a 400-case deterministic sweep through qscore+BUE+closure_gate+CBT+cbt_recal measured the ABSORPTION over a population. CAUSE: BUE runs AFTER ECE, grows outward to the contrast-optimal window and takes the UNION -> it already occupies ECE move set.',
'CONSEQUENCE FOR THIS SESSION: the fix under test is purely POSITIONAL. Move ECE to the terminal-most slot, after CBT, where nothing downstream can overwrite it.',
'',
'!!! THE HONEST PRIOR — this can plausibly come out NEGATIVE !!!',
'At the terminal slot ECE grows outward IMMEDIATELY AFTER CBT clamped inward. BUE+CBT together are worth +0.0599 on hard-88 (measured by rolling windows back to pre_bue / pre_cbt on the committed cache). So ECE-at-terminal may simply undo part of CBT. Report the number you measure, in whichever direction. A clean negative that closes the positional hypothesis is a SUCCESSFUL loop.',
'',
'!!! WHY THE OFFLINE PRECHECK IS EXACT HERE (not merely a ceiling) !!!',
'ECE is deterministic given (boundaries, cached read-only category curve, CATEGORY_DURATION_PRIORS) and makes ZERO VLM/CLIP calls. At the TERMINAL slot there is no operator downstream. Therefore replaying ECE offline on a cached runs FINAL boundaries reproduces exactly what an in-run terminal ECE would have produced — the same argument that makes the dual-arm rig exact (PROJECT_DOCUMENTATION 8.3.2). This is NOT the usual FM6 situation where offline gains fail to transfer; FM6 bites when downstream operators react to the change, and here there is no downstream. State this reasoning explicitly and flag any way it could fail.',
'',
'!!! LANDMINE — the ECE hard gate will silently destroy a 3-hour eval !!!',
'batch_evaluate.py check_ece_activation_gate counts lines containing the EXACT string "[ECE] summary:" in run.log whenever --event_continuation_extension is passed, and if count < succeeded it logs [ECE-GATE] ... ok=False and sys.exit(2) WITHOUT WRITING evaluation_report.json. The pipeline currently emits that marker once per video including two early-return paths (L1946 "0 detections (early return: video normal)" and L2229 "0 detections (early return: no candidates)"). Relocating the ECE call MUST keep exactly one such marker per video on EVERY return path. Verify in the self-test that [ECE-GATE] reports ok=True and that a report was actually written.',
'',
'!!! MEASUREMENT — exact paired, zero VLM noise !!!',
'Persist arm-OFF into ' + AB_FIELD + ', index-aligned and same length as anomalies (ECE changes boundaries only, so the boundary field is the correct one; count is frozen). Adjudicate with the existing tool, never a cross-run comparison (the same-config cross-run spread is ' + NOISE_HARD88 + ' on hard-88, 3x the gate):',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-json ' + AB_FIELD + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P,
'HARNESS: hard-88 (' + HARNESS_HARD + ') — ECE fires on anomaly detections, so this is where it is measurable and it is the cheaper harness. The 290 confirm runs only if the gate is cleared.',
'COVERAGE GATE (PROJECT_DOCUMENTATION 8.4, cost this project a whole round): evaluation_report.json carries no completeness field and evaluate() drops videos with no prediction TOGETHER WITH THEIR GT, inflating mAP — v26 measured 0.17673 reported vs 0.16682 true with 24/290 missing. Before ANY adjudication assert timing.failed == 0 and timing.num_videos == the expected harness size; if failed > 0, RESUME the same --output_dir (completed per-video JSONs are SKIPped) and only then adjudicate.',
'',
'!!! GATE: KEEP requires paired delta >= +' + MIN_PAIRED_DELTA + ' AND P(delta>0) >= ' + MIN_P + ' on hard-88, then a full-290 dual-arm confirm that is net-neutral-or-positive before any promotion. Otherwise CLOSE and archive. !!!',
'',
'FM DISCIPLINE: FM2 count frozen (ECE must not change ndet — assert across arms). FM5 marker must actually fire. FM6 offline gains normally do not transfer (see the terminal-slot exemption above). FM13 bound the ceiling first (that is the Precheck phase). FM17 per-video marker on EVERY path incl. both early returns. FM18 report the ABSOLUTE mAP, not only the delta. FM20 the paired field must be WRITTEN and GREPPED in a produced JSON — a claim is not evidence. FM21 the marker string must match the orchestrator grep EXACTLY.',
'HARD CONSTRAINTS: ONE change (the positional move) behind ONE new flag, default OFF, flag-off byte-identical (from_args(Namespace()).scoring == ScoringConfig()); CLI in BOTH run.py and batch_evaluate.py; the arm-OFF replay MUST mirror the new order or the arm-OFF is wrong; NEVER edit evaluate.py or analysis_bootstrap_ab.py; NEVER run two evals at once; eval always --concurrency ' + CONCURRENCY + '. StructuredOutput EXACTLY ONCE. ANTI-STALL: no Bash > ~150s except the stated self-test smoke; Read with offset/limit; short progress lines.',
].join('\n')

const preSchema = { type: 'object', properties: { ran: { type: 'boolean' }, go: { type: 'boolean' }, offline_delta_hard88: { type: 'number', description: 'deterministic offline ECE-at-terminal delta on the hard-88 subset of the cached committed run' }, offline_delta_290: { type: 'number' }, n_fire: { type: 'number', description: 'detections ECE would materially extend at the terminal slot' }, cbt_undo: { type: 'string', description: 'how much of CBT inward clamp ECE would reverse, measured not asserted' }, exactness: { type: 'string', description: 'why the offline replay is exact at the terminal slot, and any way it could fail' }, note: { type: 'string' } }, required: ['ran', 'go', 'note'], additionalProperties: false }
const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'PASTED evidence: [ECE-GATE] ok=True AND a report written; marker count == videos; ' + AB_FIELD + ' present with len == len(anomalies); arm independence (0-fire identical, firing differs); ndet equal across arms; and that the arm-OFF replay was updated to the new order' }, activation_marker: { type: 'string' }, eval_flags: { type: 'string', description: 'FULL flag string; do NOT include --categories (the launcher adds the harness)' }, files_changed: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'instrument_selftest', 'activation_marker', 'eval_flags', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, num_videos: { type: 'number' }, failed: { type: 'number' }, marker_count: { type: 'number' }, paired_jsons: { type: 'number' }, ece_gate_ok: { type: 'boolean' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, coverage_ok: { type: 'boolean' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, per_category: { type: 'string' }, count_neutral: { type: 'boolean' }, fired_videos: { type: 'number' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}
function launchPrompt(tag, harness, evalFlags) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const h = harness ? (harness + ' ') : ''
  return [
'You are Agent D-launcher for ' + V + ' (' + tag + '). Ensure the eval runs with the REQUIRED flags, then return IMMEDIATELY. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + dir + ' log=' + dir + '.log report=' + dir + '/evaluation_report.json',
'REQUIRED FLAGS (verbatim): ' + h + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep — NEVER two evals at once. 2. Already running for this tag -> adopt if flags match else kill and relaunch. 3. report exists AND log shows these flags -> done. 4. A DIFFERENT eval is running -> sleep 140 (up to 3x) then report blocked.',
'5. Otherwise write ' + dir + '/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + dir + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category 999 --concurrency " + CONCURRENCY + " " + h + evalFlags + " > " + dir + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log for an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function watchPrompt(tag, marker, n) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const rep = dir + '/evaluation_report.json'
  return [
'You are Agent D-watcher for ' + V + ' (' + tag + '), expecting ' + n + ' videos. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE. marker="' + marker + '".',
'1. If ' + rep + ' exists: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'timing\'];print(d[\'evaluation\'][\'temporal_map\'][\'mAP\'],t.get(\'num_videos\'),t.get(\'failed\'))" -> mAP, num_videos, failed.',
'   COVERAGE GATE: num_videos MUST be ' + n + ' and failed MUST be 0. If failed > 0, RESUME (relaunch the SAME nohup on the SAME --output_dir; completed JSONs are SKIPped) and keep watching — do NOT return done=true with failed > 0.',
'   ECE GATE: grep -F "[ECE-GATE]" ' + dir + '/run.log | tail -1 -> set ece_gate_ok from ok=True/False. If the report is MISSING but the log shows [ECE-GATE] ok=False, that is the hard gate refusing to write it: crashed=true, note="ECE-GATE FM5 failure — marker missing on some path".',
'   markers: grep -cF "' + marker + '" ' + dir + '.log and ' + dir + '/run.log (trust the LARGER).',
'   FM20: PJ=$(grep -l ' + AB_FIELD + ' ' + dir + '/*.json 2>/dev/null | grep -v evaluation_report | wc -l) -> paired_jsons=PJ, note "pairedJSONs=<PJ>". PJ==0 means VOID.',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 (up to 2x) then done=false note="running <n>/' + n + '" from the count of result JSONs. Stalled >30min with no new JSON -> kill and relaunch the same nohup.',
'3. Dead with no report -> tail -30. TRANSIENT (API/network) -> relaunch. CODE BUG -> crashed=true with the traceback.',
'Return done, crashed, mAP, num_videos, failed, marker_count, paired_jsons, ece_gate_ok, note.',
  ].join('\n')
}
async function runEval(tag, harness, evalFlags, marker, n, cap, ph) {
  const l = await retryAgent(() => agent(launchPrompt(tag, harness, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag, phase: ph }))
  if (!l || !l.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, paired_jsons: -1, note: 'launch failed: ' + (l ? l.note : 'null') }
  for (let w = 0; w < (cap || 90); w++) {
    const st = await retryAgent(() => agent(watchPrompt(tag, marker, n), { schema: watchSchema, label: 'D-watch-' + tag + '-' + w, phase: ph }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, marker_count: -1, paired_jsons: -1, note: 'watcher cap' }
}
function bootPrompt(tag, n, ph) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  return [
'You are Agent D-adjudicator for ' + V + ' ECE-TERMINAL on ' + n + ' videos. The run carries BOTH arms: its output is arm-ON, field ' + AB_FIELD + ' holds arm-OFF. Exact paired, zero VLM noise. Bash < 150s. StructuredOutput EXACTLY ONCE.',
'STEP 1 COVERAGE GATE: assert timing.failed == 0 and timing.num_videos == ' + n + '. If either fails -> coverage_ok=false, ran=false, STOP (a report with failures has an INFLATED mAP; PROJECT_DOCUMENTATION 8.4).',
'STEP 2 FM20 GATE: grep -l ' + AB_FIELD + ' ' + dir + '/*.json | grep -v evaluation_report | wc -l . If 0 -> ran=false note="FM20 VOID". NEVER fall back to a cross-run comparison.',
'STEP 3 the decisive number: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-json ' + AB_FIELD + ' --min-delta ' + MIN_PAIRED_DELTA + ' --min-p ' + MIN_P + ' -> point_delta, ci90, p_gt0, new_map (absolute, ECE-terminal ON), ref_map (arm-OFF absolute).',
'STEP 4 fired_videos: count videos whose arm-OFF actually DIFFERS from the final output (not merely videos with an [ECE] line — v202 had 17 material extensions and still zero output change). If 0, the positional move did not take effect: say so plainly, that is VOID not a null.',
'STEP 5 per-category AP deltas across arms; count-neutrality (mean detections-per-video identical) -> count_neutral.',
'STEP 6 diagnose the DIRECTION: quantify how much of CBT inward clamp ECE reversed (compare arm-OFF vs arm-ON durations against pre_cbt_boundaries where present). Put it in note. This is what tells the lead whether a variant is worth trying.',
'Return ran, coverage_ok, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, per_category, count_neutral, fired_videos, note (VERDICT verbatim).',
  ].join('\n')
}

// ============================ MAIN ============================
log('v28 starting. ONE question: does moving ECE to the TERMINAL slot (after CBT) make its effect nonzero and positive? v202 proved ECE at the FRONT slot is an exact zero because BUE absorbs it. Gate: paired >= +' + MIN_PAIRED_DELTA + ' with P >= ' + MIN_P + ' on hard-88, then a 290 confirm. Honest prior: this may be NEGATIVE because ECE would grow outward right after CBT clamped inward.')

// ---------- Precheck ----------
phase('Precheck')
const pre = await retryAgent(() => agent([
'You are Agent P. ZERO API calls, ZERO evals. Determine the DETERMINISTIC offline effect of ECE-at-terminal on cached runs, and decide GO/NO-GO before any eval time is spent (FM13). StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'SUBSTRATE: results/batch_eval_v26_full290 (290/290, failed=0, the fresh committed baseline, mAP ' + REF290_REPRO + ' — verify the coverage gate on it first), plus results/batch_eval_v202_ece_selftest (12 videos, already carries pre_ece_boundaries / ' + AB_FIELD + ' / ab_arm_off_confidences), features/test, and the reusable oracle scripts iteration_logs/analysis_v200_oracle_ceilings.py, analysis_v200b_mech_scoped_ceilings.py, analysis_v200c_transfer_calibration.py (read them and reuse their helpers rather than rewriting).',
'TASK: replay the ACTUAL ECE rule (_event_continuation_extension, pipeline.py L3481 — read it; do not reimplement from the docstring) on the cached runs FINAL post-CBT boundaries, using the same read-only category curve substrate and the same CATEGORY_DURATION_PRIORS reject-cap the live code uses. Score mAP before vs after, on the hard-88 subset AND on the full 290.',
'  Because the terminal slot has NOTHING downstream, this replay is EXACT — it is what an in-run terminal ECE would produce, not an upper bound. Explain in `exactness` why that holds and name any way it could fail (e.g. if reconstructing the curve offline is not bit-faithful to the in-run call, say so and treat the number as approximate).',
'ALSO MEASURE, do not assert: (a) n_fire = how many detections ECE materially extends at the terminal slot; (b) cbt_undo = how much of CBT inward clamp gets reversed — compare the post-ECE windows against pre_cbt_boundaries on videos where CBT fired. This is the mechanism of any negative result.',
'GO/NO-GO: go=true only if the offline hard-88 delta is >= +' + (MIN_PAIRED_DELTA * 0.6).toFixed(4) + ' (allowing headroom for offline-vs-in-run slack) OR the replay could not be made faithful and the question genuinely requires the eval. go=false if the offline delta is clearly negative or clearly below the gate — that closes the lever for free and is a successful outcome, not a failure.',
'Return ran, go, offline_delta_hard88, offline_delta_290, n_fire, cbt_undo, exactness, note.',
].join('\n'), { schema: preSchema, label: 'P-precheck-ece-terminal', phase: 'Precheck' }))
log('Precheck: go=' + (pre ? pre.go : '?') + ' offline_hard88=' + (pre ? pre.offline_delta_hard88 : '?') + ' offline_290=' + (pre ? pre.offline_delta_290 : '?') + ' n_fire=' + (pre ? pre.n_fire : '?') + ' | ' + (pre ? String(pre.note).slice(0, 260) : 'n/a'))

let impl = null, ev88 = null, boot88 = null, ev290 = null, boot290 = null
if (!pre || !pre.ran || pre.go === false) {
  log('Precheck says NO-GO -> closing the positional hypothesis without spending an eval. Recording the closure.')
} else {
  // ---------- Implement ----------
  phase('Implement')
  impl = await retryAgent(() => agent([
'You are Agent D1 for ' + V + ' ECE-TERMINAL. Implement the POSITIONAL move behind one new flag, wire it into the dual-arm rig, then run the self-test. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'PRECHECK RESULT (use it; if it says the effect is tiny, still proceed since the lead asked for the eval, but say so in note): ' + JSON.stringify(pre),
'IMPLEMENT:',
'1. Back up pipeline.py / config.py / run.py / batch_evaluate.py to ' + ROOT + '/.qoder/backups/' + V + '/ and ls to confirm.',
'2. New config field ece_terminal (default False) + CLI --ece_terminal in BOTH run.py and batch_evaluate.py, threaded via VADConfig.from_args exactly like the existing flags. flag-off MUST be byte-identical.',
'3. In the MAIN path: when ece_terminal AND event_continuation_extension are both on, do NOT call ECE at its current front slot (L2474) — call it instead at the TERMINAL-most slot, AFTER CBT (L2658) and before the ab_ablate replay hook. ECE runs exactly ONCE either way; this is a MOVE, not an addition.',
'4. MIRROR the same conditional ordering inside _ab_ablate_arm_off_replay (L2997; its ECE call is at L3094 and its CBT call at L3132). If the replay keeps the old order the arm-OFF is WRONG and every number this session produces is garbage. State explicitly in instrument_selftest that you updated it.',
'5. Extend the ab_ablate dispatch (see _abx_mech at ~L2456) to accept "' + AB_MECH + '", ablating the terminal ECE call in arm-OFF. Persist arm-OFF into ' + AB_FIELD + ' index-aligned, same length as anomalies.',
'6. ECE HARD GATE: keep EXACTLY ONE "[ECE] summary:" INFO line per video on EVERY return path, including both early returns (L1946 video-normal, L2229 no-candidates). If the relocation drops one, batch_evaluate.py sys.exit(2)s and writes NO report — a 3-hour eval lost.',
'SELF-TEST (paste ACTUAL output; a false pass here wastes the eval):',
'  ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V + '_selftest --max_per_category 2 --concurrency 1 ' + HARNESS_HARD + ' ' + MECH_BASE + ' --ab_ablate ' + AB_MECH,
'  CONFIRM ALL SIX and paste each: (a) evaluation_report.json WAS written and [ECE-GATE] ... ok=True; (b) "[ECE] summary:" count == videos processed; (c) ' + AB_FIELD + ' present in a produced JSON with len == len(anomalies); (d) ARM INDEPENDENCE — a video where ECE did not materially fire has arm-OFF byte-identical to final, and one where it did fire DIFFERS (widen max_per_category until at least one differing video exists; note that v202 had material fires with ZERO output change, so you must verify a difference at the OUTPUT, not just an [ECE] line); (e) ndet identical across arms; (f) with --ece_terminal OFF the run is byte-identical to the committed pipeline.',
'  Set selftest_passed only if (a)-(f) all literally pass.',
'VALIDATE: py_compile; import MultiAnomalyVAD; both new flags in --help of both runners; ScoringConfig() committed defaults unchanged; from_args(Namespace()).scoring == ScoringConfig().',
'Return implemented, selftest_passed, instrument_selftest, activation_marker, eval_flags (FULL string incl ' + MECH_BASE + ' --ab_ablate ' + AB_MECH + ', WITHOUT --categories), files_changed, validation, note.',
  ].join('\n'), { schema: implSchema, label: 'D1-' + V, phase: 'Implement' }))

  if (!impl || !impl.implemented || impl.selftest_passed === false || !impl.activation_marker) {
    log('Implement/self-test not verified -> refusing to launch the eval (a void loop is worse than no loop). ' + (impl ? String(impl.note || '').slice(0, 220) : 'null'))
  } else {
    // ---------- Verify88 ----------
    phase('Verify88')
    ev88 = await runEval('ab88', HARNESS_HARD, impl.eval_flags, impl.activation_marker, N_HARD, 90, 'Verify88')
    log('Verify88: done=' + ev88.done + ' crashed=' + ev88.crashed + ' mAP=' + ev88.mAP + ' n=' + (ev88.num_videos != null ? ev88.num_videos : '?') + ' failed=' + (ev88.failed != null ? ev88.failed : '?') + ' eceGate=' + (ev88.ece_gate_ok != null ? ev88.ece_gate_ok : '?') + ' markers=' + ev88.marker_count + ' pairedJSONs=' + (ev88.paired_jsons != null ? ev88.paired_jsons : '?'))
    const ok88 = ev88.done && !ev88.crashed && ev88.marker_count > 0 &&
                 !(ev88.paired_jsons != null && ev88.paired_jsons <= 0) &&
                 !(ev88.failed != null && ev88.failed > 0) &&
                 !(ev88.num_videos != null && ev88.num_videos !== N_HARD)
    if (ok88) boot88 = await retryAgent(() => agent(bootPrompt('ab88', N_HARD, 'Verify88'), { schema: bootSchema, label: 'D-boot-88', phase: 'Verify88' }))
    else log('Verify88 unusable (coverage / marker / paired / ECE-gate) -> no adjudication.')

    // ---------- Confirm290 ----------
    const pass88 = !!(boot88 && boot88.ran && boot88.point_delta >= MIN_PAIRED_DELTA && boot88.p_gt0 >= MIN_P && (boot88.fired_videos == null || boot88.fired_videos > 0))
    log('hard-88 verdict: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3) + ' absON=' + boot88.new_map.toFixed(5) + ' absOFF=' + (boot88.ref_map != null ? boot88.ref_map.toFixed(5) : '?') + ' fired=' + boot88.fired_videos + ' -> ' + (pass88 ? 'PASSES gate' : 'FAILS gate')) : 'no valid adjudication'))
    phase('Confirm290')
    if (pass88) {
      log('Gate cleared -> running the full-290 dual-arm confirm (~8.5h).')
      ev290 = await runEval('ab290', '', impl.eval_flags, impl.activation_marker, N_FULL, 160, 'Confirm290')
      log('Confirm290: done=' + ev290.done + ' mAP=' + ev290.mAP + ' n=' + (ev290.num_videos != null ? ev290.num_videos : '?') + ' failed=' + (ev290.failed != null ? ev290.failed : '?'))
      const ok290 = ev290.done && !ev290.crashed && !(ev290.failed != null && ev290.failed > 0) && !(ev290.num_videos != null && ev290.num_videos !== N_FULL)
      if (ok290) boot290 = await retryAgent(() => agent(bootPrompt('ab290', N_FULL, 'Confirm290'), { schema: bootSchema, label: 'D-boot-290', phase: 'Confirm290' }))
    } else {
      log('hard-88 did not clear the gate -> skipping the 290 confirm (nothing to promote).')
    }
  }
}

// ---------- Record ----------
phase('Confirm290')
const promote = !!(boot88 && boot88.ran && boot88.point_delta >= MIN_PAIRED_DELTA && boot88.p_gt0 >= MIN_P &&
                   boot290 && boot290.ran && boot290.point_delta >= -0.003 && boot290.new_map >= (REF290_MAP - 0.008))
await retryAgent(() => agent([
'You are Agent C. Record ' + V + ' ECE-TERMINAL honestly and completely. A clean negative that closes the positional hypothesis is a SUCCESSFUL loop — this project has been burned by over-optimistic verdicts, and v202 already showed how easily an exact zero can be misread. Plain text.',
'', CONTEXT, '',
'PRECHECK: ' + JSON.stringify(pre) + '\nIMPL: ' + JSON.stringify(impl) + '\nHARD-88 EVAL: ' + JSON.stringify(ev88) + '\nHARD-88 ADJUDICATION: ' + JSON.stringify(boot88) + '\n290 EVAL: ' + JSON.stringify(ev290) + '\n290 ADJUDICATION: ' + JSON.stringify(boot290) + '\nCOMPUTED promote=' + promote,
promote
 ? ('1. PROMOTE: back up config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/ece_terminal_promote/, then set BOTH event_continuation_extension AND ece_terminal default True (dataclass + from_args getattr + set_defaults in both runners), keeping opt-outs. PASTE validation: py_compile; both True; bue/cbt/cbt_recal/qscore still True; import MultiAnomalyVAD; from_args(Namespace()).scoring == ScoringConfig().')
 : ('1. DO NOT PROMOTE. Restore from ' + ROOT + '/.qoder/backups/' + V + '/ if code was added, diff -q to confirm, and PASTE proof the committed defaults are intact (qscore/bue/bue_closure_gate/cbt/cbt_recal True; event_continuation_extension and ece_terminal False; ab_ablate empty; from_args(Namespace()).scoring == ScoringConfig()). Mark the POSITIONAL hypothesis MEASURED-CLOSED with its number so no future session re-litigates it.'),
'2. Turn --ab_ablate back off by default and confirm; leave the rig in the tree as reusable.',
'3. date +%Y%m%d; write ' + LOGS + '/summary_' + V + '_ece-terminal_<YYYYMMDD>.md covering: WHY the positional move was tested (v202 measured an exact zero at the front slot because BUE occupies ECE move set, and the mechanism-scoped oracle had scored ECE #1 at +0.09829 by treating the slot as empty — record that oracle blind spot explicitly, it generalizes); the precheck offline number and why the terminal slot makes an offline replay exact; the coverage gate result; fired_videos measured at the OUTPUT not from [ECE] lines; the paired delta with CI and P; absolute arm-ON/arm-OFF mAP; per-category deltas; how much of CBT clamp was reversed; and the decision.',
'4. APPEND ledger rows to ' + LOGS + '/session_v97_ledger.md in the existing 9-column format (| loop | version | title | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept |), -1 for absent. Do NOT rewrite rows.',
'5. Reply with a 12-line recap leading with the single headline number.',
].join('\n'), { label: 'C-record-' + V, phase: 'Confirm290' }))

log('v28 complete. precheck_go=' + (pre ? pre.go : '?') + ' | hard-88 ' + (boot88 && boot88.ran ? (boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3)) : 'n/a') + ' | 290 ' + (boot290 && boot290.ran ? (boot290.point_delta.toFixed(5) + ' abs=' + boot290.new_map.toFixed(5)) : 'n/a') + ' | promote=' + promote)
return { precheck: pre, hard88: boot88, full290: boot290, promote }
