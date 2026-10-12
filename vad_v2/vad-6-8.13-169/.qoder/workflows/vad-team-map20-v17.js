export const meta = {
  name: 'vad-team-map20-v17',
  description: 'VAD agent team v17 — implement + adjudicate CNV (Cascade Null-Hypothesis Veto), the owner-directed ORIGINAL mechanism for NORMAL-video false positives, as a confidence-veto at the terminal slot, measured by the PROVEN paired-log conf-mode bootstrap (exact paired, zero VLM noise), with the mAP floor as a NON-INFERIORITY constraint and the normal-video false-alarm rate as the PRIMARY objective. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON, full-290 ~0.173, v160_full 0.17254; hard-88 ~0.11806). v16 CLOSED the entire candidate-RELOCATION family (CGR-M exact-paired -0.00487, P=0.224; CLIP-curve relocation died 3x before) and left the dual-arm rig in the tree. CNV rationale (owner session 2026-08-14): the pipeline has ONLY segment-level decisions and no video-level null hypothesis; VLM absolute confidence on isolated clips is uncalibrated (v143 root cause). CNV = ONE-SIDED veto (only removes confidence, never adds/moves): STAGE 1 video-level global-null gate — veto ALL of a video\'s detections iff THREE independent cached signal families are ALL negative (MIL top-k snippet mean low AND fused CLIP category curve in-video max-z low AND motion-energy outlier fraction low); STAGE 2 segment-level in-video surprisal veto — on surviving videos, veto any detection whose window curve mass is NOT an outlier vs its own video baseline. Thresholds are PRINCIPLED STRUCTURAL DEFAULTS verified (NOT fitted) on a few TRAIN videos — no test-set tuning. Veto = set confidence to 0.0 and keep the detection (count-frozen, index-aligned), logged as [CNV-FIRE] marker lines conf=old->new so analysis_bootstrap_ab.py --paired-log --paired-mode conf restores the arm-OFF arm EXACTLY (the v138/v139 proven path — NO new bootstrap code, NO models.py change). Oracle context (v9): deleting every Normal-video FP is worth only +0.00253 mAP, so mAP is the CONSTRAINT (paired delta >= -0.003, abs >= REF290-0.008), the OBJECTIVE is the false-alarm rate (>= 25% relative reduction in Normal-video FA rate, <= 1 GT-overlapping detection vetoed). ONE conditional structural variant (v172): tighten (add VideoMAE==Normal as 4th conjunct) if harm, loosen (2-of-3 conjunct) if the gate under-fires — never numeric test-fitting. Ends with a C-reviewed DESIGN-ONLY deliverable (no eval) for the next signal-quality lever (Shoplifting cand-miss / flat curves) as v173 input. Concurrency 2. NONE of the 10 banned families; relocation family CLOSED; Normal-FP-for-mAP stays banned — CNV is judged on the FA-rate objective with mAP only as constraint.',
  phases: [
    { title: 'Implement', detail: 'D1 implements --cnv_veto (default OFF, terminal slot, confidence-veto + [CNV-FIRE]/[CNV-VID] markers), self-tests incl. a forced-fire path proving the paired-log restore works' },
    { title: 'Eval290', detail: 'one full-290 eval with cnv-on; adjudicator runs the paired-log conf bootstrap + Normal-video FA rates + GT-veto harm check' },
    { title: 'Variant', detail: 'conditional v172: tighten on harm, loosen on under-fire, else skip; same 290 adjudication' },
    { title: 'Decide', detail: 'promote cnv_veto default-ON iff mAP non-inferior AND FA-rate cut >= 25% AND <=1 GT veto; else archive-close; write summary + ledger' },
    { title: 'NextDesign', detail: 'A/B/C design-only loop for the next signal-quality lever (Shoplifting cand-miss), with an offline go/no-go pre-check — deliverable for v18, no eval spend' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const CONCURRENCY = 2
const N_FULL = 290
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const CNV_FLAG = '--cnv_veto'
const FIRE_MARKER = 'CNV-FIRE'   // per-detection: consumed by --paired-log restore (conf=old->new)
const VID_MARKER = 'CNV-VID'     // per-video unconditional summary (FM5/FM17)
const V_CNV = 'v171'
const V_VAR = 'v172'
// gates (pre-registered)
const MAP_TOL = -0.003           // non-inferiority: paired mAP delta must be >= this
const REF290_MAP = 0.17254       // committed @290 (v160_full)
const ABS_TOL = 0.008
const MIN_FA_REL_RED = 0.25      // primary objective: >=25% relative reduction of Normal-video FA rate
const MAX_GT_VETO = 1            // harm bound: GT-overlapping (IoU>=0.1) detections vetoed

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE: v137 + BUE + CBT + CBT-recal, ALL DEFAULT-ON. full-290 mAP ~0.173 (v160_full ' + REF290_MAP + '); hard-88 ~0.11806. cgr_enabled=False (v16 MEASURED-CLOSED the whole relocation family), cgr_ab_diag=False, ccr_enabled=False. Do NOT change any committed default. !!!',
'',
'!!! THE MECHANISM — CNV (Cascade Null-Hypothesis Veto), owner-directed ORIGINAL (2026-08-14) !!!',
'ROOT CAUSE IT ATTACKS: the pipeline makes only SEGMENT-level decisions; nothing asks "does this video, as a whole, contain an event that breaks its own baseline?" VLM absolute confidence on isolated clips is uncalibrated (v143 proved scan_segment pins ~0.9 on isolated clips), so Normal videos carry hallucinated detections nothing downstream can remove.',
'CNV is a ONE-SIDED CONFIDENCE VETO at the TERMINAL slot (after CBT, on the final list). It NEVER moves a window, NEVER adds a detection, NEVER re-ranks. Veto = confidence := 0.0, detection KEPT in the list (count-frozen, index-aligned).',
'  STAGE 1 (video-level global-null gate): veto ALL detections of a video iff ALL THREE independent cached signal families are negative:',
'    (a) MIL video-level evidence: top-k mean of the snippet scores (_mil_scores, already loaded model_mil.pth) < tau_mil (default 0.20);',
'    (b) fused CLIP category curve in-video outlierness: max over frames of (x - median)/MAD of the video\'s OWN category curve < tau_z (default 1.5);',
'    (c) motion-energy outlier fraction: fraction of frames with motion energy above (median + 3*MAD) of the video < tau_mot (default 0.01).',
'  STAGE 2 (segment-level in-video surprisal veto): on videos that PASS stage 1, veto any detection whose window-aggregated curve mass is NOT an in-video outlier: z_window = (win_mean - median)/MAD < tau_seg (default 0.5) AND the window does not contain the video\'s global curve argmax.',
'  Thresholds are PRINCIPLED STRUCTURAL DEFAULTS. Verify (do NOT fit) them on a handful of TRAIN-split videos (a few anomaly + a few normal): the stage-1 conjunct should essentially never fire on anomaly videos. Report what you observed. NEVER measure thresholds on the test split.',
'',
'!!! WHY THE PAIRED-LOG INSTRUMENT (proven path — reuse it, add no new bootstrap code) !!!',
'Vetoed detections keep their slot with confidence 0.0. For EVERY vetoed detection log ONE line tagged ' + FIRE_MARKER + ' in EXACTLY the marker format analysis_bootstrap_ab.py restore_pre_marker expects for --paired-mode conf (vid + t_start + t_end keys + conf=old->new — READ the tool ~L40-70 and copy a working producer such as the qscore marker before writing yours). Then:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-log <dir>.log --marker ' + FIRE_MARKER + ' --paired-mode conf --min-delta ' + MAP_TOL + ' --min-p 0.9',
'rebuilds the arm-OFF (committed) arm EXACTLY: same videos, same VLM sampling, same run -> EXACT paired contrast, zero cross-run VLM noise. point_delta = CNV\'s effect on FINAL mAP (must be >= ' + MAP_TOL + ').',
'',
'!!! THE OBJECTIVE (read this twice — it is NOT mAP) !!!',
'v9 oracle: deleting every Normal-video FP is worth only +0.00253 mAP, so mAP CANNOT be the objective. The objective is the NORMAL-VIDEO FALSE-ALARM RATE: fraction of Normal-category test videos carrying >=1 surviving (conf>0) detection, arm-ON vs arm-OFF, plus mean surviving FP detections and FP-seconds per Normal video. Harm check: how many vetoed detections overlapped GT at IoU>=0.1 (must be ~0). PROMOTION GATE: paired mAP delta >= ' + MAP_TOL + ' AND abs mAP >= ' + (REF290_MAP - ABS_TOL).toFixed(5) + ' AND FA-rate relative reduction >= ' + (MIN_FA_REL_RED * 100) + '% AND GT-overlap vetoes <= ' + MAX_GT_VETO + '.',
'',
'FM DISCIPLINE: FM2 count frozen (veto keeps the detection — assert len unchanged). FM5 ' + FIRE_MARKER + ' must actually fire when vetoes happen. FM17: ' + VID_MARKER + ' summary fires UNCONDITIONALLY per video INCLUDING the 0-detection path AND the early-return-normal path (~pipeline.py L1543, which returns before the terminal slot). FM20: the marker lines must be GREPPED in the produced log — a claim is not evidence. FM21: marker strings must EXACTLY match what the orchestrator greps (' + FIRE_MARKER + ' / ' + VID_MARKER + '). FM18: read the ABSOLUTE mAP too. HARNESS: full 290 (NO --categories filter — Normal videos ARE the subject), --concurrency ' + CONCURRENCY + '.',
'HARD CONSTRAINTS: flag default OFF (flag-off byte-identical to committed pipeline); CLI in BOTH run.py and batch_evaluate.py; StructuredOutput EXACTLY ONCE per agent. ANTI-STALL: no Bash > ~150s except the stated self-test smoke; Read with offset/limit.',
].join('\n')

const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'PASTED actual evidence: (a) ' + VID_MARKER + ' grep count covering every smoke video incl. empty/early-normal paths; (b) FORCED-FIRE proof: a run where the veto path executed, showing ' + FIRE_MARKER + ' conf=old->new lines AND analysis_bootstrap_ab.py --paired-log --paired-mode conf restoring the old confidences (n_fires>0, matched==n_fires); (c) a natural (unforced) smoke observation: which videos vetoed, stage distribution; (d) count-frozen check (len anomalies identical pre/post CNV)' }, marker_fire: { type: 'string', description: 'exact per-detection fire line format' }, marker_vid: { type: 'string', description: 'exact per-video summary line format' }, thresholds: { type: 'string', description: 'final default thresholds + the TRAIN-split verification observations (no test-set numbers)' }, eval_flags: { type: 'string', description: 'FULL flag string = ' + MECH_BASE + ' + ' + CNV_FLAG + ' (no --categories; 290 harness)' }, files_changed: { type: 'string' }, validation: { type: 'string', description: 'py_compile + import MultiAnomalyVAD + flag-off byte-identity: ScoringConfig() defaults unchanged and from_args(Namespace()).scoring == ScoringConfig()' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'instrument_selftest', 'marker_fire', 'marker_vid', 'thresholds', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number', description: 'grep count of ' + VID_MARKER }, fires: { type: 'number', description: 'grep count of ' + FIRE_MARKER }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'fires', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, n_fires: { type: 'number' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number', description: 'arm-OFF (committed) absolute mAP from the restore' }, fa_video_rate_on: { type: 'number', description: 'fraction of Normal-category videos with >=1 surviving (conf>0) detection, arm-ON' }, fa_video_rate_off: { type: 'number', description: 'same, arm-OFF' }, fa_rel_reduction: { type: 'number', description: '(off-on)/off; 0 if off==0' }, fp_det_on: { type: 'number' }, fp_det_off: { type: 'number', description: 'mean surviving detections per Normal video, each arm' }, gt_vetoed: { type: 'number', description: 'vetoed detections with GT IoU>=0.1' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'n_fires', 'point_delta', 'new_map', 'fa_video_rate_on', 'fa_video_rate_off', 'fa_rel_reduction', 'gt_vetoed', 'note'], additionalProperties: false }
const decideSchema = { type: 'object', properties: { promoted: { type: 'boolean' }, decision: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['promoted', 'decision', 'note'], additionalProperties: false }
const designSchema = { type: 'object', properties: { proposals: { type: 'string' }, winner: { type: 'string' }, precheck: { type: 'string', description: 'offline go/no-go evidence (no eval spend)' }, note: { type: 'string' } }, required: ['proposals', 'winner', 'precheck', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

function evalLaunchPrompt(V, tag, evalFlags) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  return [
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + dir + ' ; log=' + dir + '.log ; report=' + dir + '/evaluation_report.json',
'REQUIRED FLAGS (verbatim, NO --categories — this is the FULL 290 harness): ' + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. Already running for this V_tag -> adopt if the flags match, else kill and relaunch. 3. report exists AND the log shows the required flags AND shows ' + VID_MARKER + ' lines -> launched=true, note="already complete", do not relaunch. 4. A DIFFERENT version is running -> sleep 140 (up to 3x), then report blocked.',
'5. Otherwise write ' + dir + '/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + dir + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category 999 --concurrency " + CONCURRENCY + " " + evalFlags + " > " + dir + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log to catch an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, nTotal) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const rep = dir + '/evaluation_report.json'
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + ' (expect ' + nTotal + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE.',
'1. If ' + rep + ' exists -> done=true. Read it: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'))" . Put num_videos in num_videos — it MUST be ' + nTotal + '; if not, say so loudly in note (harness bug).',
'   markers: VC=$(grep -cF "' + VID_MARKER + '" ' + dir + '.log 2>/dev/null); FC=$(grep -cF "' + FIRE_MARKER + '" ' + dir + '.log 2>/dev/null) -> marker_count=VC, fires=FC. VC should be ~' + nTotal + ' (one per video, FM17); FC>0 means the veto actually fired (FM5). Put both in note as "vidMarker=<VC> fires=<FC>".',
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 (up to 2x), then done=false note="running <n>/' + nTotal + '" where n = count of result JSONs. If STALLED >30min with no new JSON -> kill and relaunch the SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead with no report -> tail -30 the log. TRANSIENT (API/network) -> relaunch. CODE BUG -> crashed=true with the traceback in note.',
'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, marker_count, fires, note.',
  ].join('\n')
}
async function runEval(V, tag, evalFlags, maxWatch, nTotal, ph) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: ph }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, fires: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  if (launch.note && launch.note.indexOf('already complete') >= 0) {
    const st0 = await retryAgent(() => agent(evalWatchPrompt(V, tag, nTotal), { schema: watchSchema, label: 'D-watch-adopt-' + V, phase: ph }))
    if (st0 && st0.done) return st0
  }
  for (let w = 0; w < (maxWatch || 130); w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: ph }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, marker_count: -1, fires: -1, note: 'watcher cap reached' }
}
function bootPrompt(V, tag, ph) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  return [
'You are Agent D-adjudicator for the CNV paired A/B on the FULL 290 (Normal-category videos included — they are the subject). arm-ON = the run as-is (vetoed detections carry confidence 0.0); arm-OFF = the committed pipeline, reconstructed EXACTLY by restoring ' + FIRE_MARKER + ' conf=old->new lines. Bash < 150s per command. StructuredOutput EXACTLY ONCE.',
'STEP 1 (FM5/FM20 gate): FC=$(grep -cF "' + FIRE_MARKER + '" ' + dir + '.log); VC=$(grep -cF "' + VID_MARKER + '" ' + dir + '.log). Put FC in n_fires. VC should be ~290 (FM17). If VC==0 -> ran=false, note="FM20 VOID: ' + VID_MARKER + ' absent" and STOP.',
'STEP 2 (mAP constraint, the proven paired tool — no new bootstrap code): ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-log ' + dir + '.log --marker ' + FIRE_MARKER + ' --paired-mode conf --min-delta ' + MAP_TOL + ' --min-p 0.9 -> point_delta (arm-ON minus arm-OFF), ci90_low/high, p_gt0, new_map (abs mAP with CNV on), ref_map (arm-OFF abs mAP). If FC==0 the tool will report 0 fires — that is a TRUE zero-effect read (delta 0), still set ran=true and continue to STEP 3.',
'STEP 3 (THE OBJECTIVE — Normal-video false-alarm rate, both arms): using evaluate.load_ground_truth / load_predictions, iterate the Normal-category videos in ' + dir + ' (videos whose GT has no anomalies / category Normal). arm-ON: count a detection as SURVIVING iff confidence > 0. arm-OFF: same detections with the ' + FIRE_MARKER + ' old confidences restored. Report fa_video_rate_on/off (fraction of Normal videos with >=1 surviving detection), fa_rel_reduction=(off-on)/off (0 if off==0), and fp_det_on/off (mean surviving detections per Normal video). Paste the code path and numbers.',
'STEP 4 (harm check): for every vetoed detection (from the ' + FIRE_MARKER + ' lines), compute max temporal IoU vs its video\'s GT; gt_vetoed = count with IoU>=0.1. Also list the vetoed-on-anomaly-videos count by category.',
'STEP 5 per-category AP deltas (arm-ON minus arm-OFF) for the 4 hard cats + any category with |delta|>0.002, into per_category.',
'Report ran, n_fires, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, fa_video_rate_on, fa_video_rate_off, fa_rel_reduction, fp_det_on, fp_det_off, gt_vetoed, per_category, note (VERDICT line verbatim).',
  ].join('\n')
}

// ============================ MAIN ============================
log('v17 starting. CNV (Cascade Null-Hypothesis Veto): the owner-directed ORIGINAL normal-FP mechanism. Paired-log conf-mode instrument (zero VLM noise). Objective = Normal-video FA rate (-' + (MIN_FA_REL_RED * 100) + '% or more); constraint = paired mAP delta >= ' + MAP_TOL + ', abs >= ' + (REF290_MAP - ABS_TOL).toFixed(5) + '; harm bound = <= ' + MAX_GT_VETO + ' GT-overlap veto. concurrency ' + CONCURRENCY + '.')

// ---------- PHASE 1: implement CNV ----------
phase('Implement')
const impl = await retryAgent(() => agent([
'You are Agent D1. Implement CNV (Cascade Null-Hypothesis Veto) as a default-OFF flag, then PROVE it and its paired-log instrument correct. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'IMPLEMENT:',
'1. Back up every file you touch to ' + ROOT + '/.qoder/backups/' + V_CNV + '/ FIRST, and ls the backup dir to confirm.',
'2. New config `cnv_veto` (default False) + CLI `--cnv_veto` in BOTH run.py and batch_evaluate.py, threaded through VADConfig.from_args like cgr_enabled. Flag-off byte-identical to the committed pipeline.',
'3. pipeline.py, TERMINAL slot (AFTER the BUE/CBT blocks, on the genuinely-final final_anomalies, before DetectionResult is built): implement CNV exactly as specified in CONTEXT. Veto = confidence := 0.0, detection KEPT (FM2: assert len(final_anomalies) unchanged). Never touch t_start/t_end/category/order. Zero new VLM/CLIP/MIL calls — reuse _mil_scores if already computed (else compute from the already-loaded mil_model on cached clip_features), the video\'s fused category curve (the same cached curve BUE/CBT read), and video_data[\'motion_energy\'] (computed ~L1405).',
'4. MARKERS (FM21, exact): for EVERY vetoed detection, ONE log line tagged ' + FIRE_MARKER + ' in the EXACT format analysis_bootstrap_ab.py restore_pre_marker (--paired-mode conf) parses — READ that function (~L40-70 of the tool) and a working producer (e.g. the qscore marker) FIRST; the line must carry vid + t_start + t_end keys + conf=old->new. For EVERY video, ONE unconditional summary line tagged ' + VID_MARKER + ' (vid, ndet, vetoed, stage distribution, and the three stage-1 signal values) INCLUDING the 0-detection path AND the early-return-normal path (~L1543).',
'5. Dev-only forced-fire path for the self-test: an env var CNV_FORCE_FIRE=1 that forces every gate to fire (clearly dev-only, excluded from default behaviour; say so in notes). This exists because a natural smoke may produce zero vetoes, and an all-zero smoke proves nothing about the fire/restore path.',
'6. Bump the pipeline.py header to ' + V_CNV + '. Do NOT touch BUE/CBT/recal/CCR/CGR defaults. Do NOT touch models.py (no new JSON fields — the paired restore is LOG-based).',
'',
'HARD SELF-TEST (mandatory — paste ACTUAL command output; v165 cost this project a loop by claiming a self-test it had not run):',
'  SMOKE A (natural): ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V_CNV + '_selftest --max_per_category 1 --concurrency 1 ' + MECH_BASE + ' ' + CNV_FLAG + '   (NO --categories: this yields ~13-14 videos INCLUDING Normal-category ones — exactly the subject). Confirm: (a) grep -cF ' + VID_MARKER + ' == number of produced JSONs (EVERY video, FM17); (b) report which videos vetoed and the stage distribution; (c) len(anomalies) identical with the flag off on the same videos (count-frozen).',
'  SMOKE B (forced): rerun 2-3 of those videos with CNV_FORCE_FIRE=1 -> confirm ' + FIRE_MARKER + ' lines exist AND ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <dir> --paired-log <log> --marker ' + FIRE_MARKER + ' --paired-mode conf reports n_fires>0 and matched==n_fires with a sensible negative delta. If the tool parses zero fires, your marker format is wrong — FIX and rerun.',
'  TRAIN-split threshold verification (sanity, NOT fitting): on 2-3 TRAIN anomaly + 2-3 TRAIN normal videos, compute the three stage-1 signals and report them; the conjunct should essentially never fire on the anomaly ones. Do NOT touch test-split numbers for this.',
'  Set selftest_passed=true ONLY if every check literally passes.',
'VALIDATE: ' + PY + ' -m py_compile; import MultiAnomalyVAD; --cnv_veto appears in --help of BOTH entrypoints; ScoringConfig() BUE/CBT/recal/CGR/CCR defaults UNCHANGED and cnv_veto False; from_args(Namespace()).scoring == ScoringConfig().',
'Return implemented, selftest_passed, instrument_selftest (pasted evidence), marker_fire, marker_vid, thresholds, eval_flags, files_changed, validation, notes.',
'The self-test smoke is ~13-14 videos at concurrency 1 (~15-20 min) — expected and allowed. No OTHER Bash > ~150s.',
].join('\n'), { schema: implSchema, label: 'D1-implement-' + V_CNV, phase: 'Implement' }))

const cnvOk = !!(impl && impl.implemented && impl.selftest_passed && impl.eval_flags)
let boot = null, run290 = null, varBoot = null, varRun = null, variantDir = null
if (!cnvOk) {
  log('Phase 1: CNV instrument NOT verified (selftest=' + (impl ? impl.selftest_passed : 'null') + ') -> cannot adjudicate; stopping before any eval spend')
} else {
  log('Phase 1: CNV verified. thresholds=' + (impl.thresholds || '?').slice(0, 120))

  // ---------- PHASE 2: full-290 paired A/B ----------
  phase('Eval290')
  run290 = await runEval(V_CNV, 'cnv290', impl.eval_flags, 130, N_FULL, 'Eval290')
  log('Eval290: done=' + run290.done + ' crashed=' + run290.crashed + ' mAP=' + run290.mAP + ' n=' + (run290.num_videos != null ? run290.num_videos : '?') + ' vidMarker=' + run290.marker_count + ' fires=' + run290.fires + ' ' + (run290.note || ''))
  if (run290.done && !run290.crashed && run290.marker_count > 0) {
    boot = await retryAgent(() => agent(bootPrompt(V_CNV, 'cnv290', 'Eval290'), { schema: bootSchema, label: 'D-boot-cnv290', phase: 'Eval290' }))
  } else {
    log('Eval290: eval unusable -> no adjudication')
  }
}

// ---------- gate evaluation + conditional variant ----------
function evaluateGate(b) {
  if (!b || !b.ran) return { promote: false, harm: false, fa_ok: false, ran: false }
  const harm = (b.gt_vetoed > MAX_GT_VETO) || (b.point_delta < MAP_TOL)
  const fa_ok = b.fa_rel_reduction >= MIN_FA_REL_RED
  const promote = !harm && fa_ok && b.new_map >= (REF290_MAP - ABS_TOL)
  return { promote, harm, fa_ok, ran: true }
}
let g1 = evaluateGate(boot)
log('CNV gate: ran=' + g1.ran + ' promote=' + g1.promote + ' harm=' + g1.harm + ' fa_ok=' + g1.fa_ok +
    (boot && boot.ran ? (' | delta=' + boot.point_delta.toFixed(5) + ' abs=' + boot.new_map.toFixed(5) + ' FA ' + boot.fa_video_rate_off.toFixed(3) + '->' + boot.fa_video_rate_on.toFixed(3) + ' (rel -' + (boot.fa_rel_reduction * 100).toFixed(1) + '%) gt_vetoed=' + boot.gt_vetoed) : ''))

phase('Variant')
if (g1.ran && !g1.promote && impl) {
  variantDir = g1.harm ? 'tighten' : 'loosen'
  log('Variant v172 direction: ' + variantDir.toUpperCase() + ' (one structural step, pre-registered, no numeric test-fitting)')
  const varImpl = await retryAgent(() => agent([
'You are Agent D2. Implement the pre-registered CNV structural VARIANT "' + variantDir.toUpperCase() + '" as ' + V_VAR + ', then re-prove the instrument. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'THE v171 RESULT DICTATES THE DIRECTION:',
'  gate: ' + JSON.stringify({ delta: boot && boot.point_delta, abs: boot && boot.new_map, fa_off: boot && boot.fa_video_rate_off, fa_on: boot && boot.fa_video_rate_on, fa_rel_red: boot && boot.fa_rel_reduction, gt_vetoed: boot && boot.gt_vetoed }),
 variantDir === 'tighten'
   ? 'TIGHTEN (v171 showed harm): stage 1 gains a 4th necessary conjunct — the video\'s videomae_category must be "Normal" (or the early VideoMAE path judged it normal) — AND stage 2 may only fire on detections whose window z < 0 (below the video median). Everything else unchanged. This makes the veto strictly more conservative by CONSTRUCTION (fewer fires), the only legitimate response to harm without test-fitting numbers.'
   : 'LOOSEN (v171 under-fired: FA reduction ' + (boot ? (boot.fa_rel_reduction * 100).toFixed(1) : '?') + '% < ' + (MIN_FA_REL_RED * 100) + '% with no harm): stage 1 becomes a 2-of-3 conjunct (veto when ANY TWO of the three signal families are negative) AND stage 2 additionally vetoes detections whose window z < 0 when the video-level MIL evidence is also below tau_mil. Everything else unchanged.',
'DO: back up to ' + ROOT + '/.qoder/backups/' + V_VAR + '/ first; same flag ' + CNV_FLAG + ' (the variant REPLACES the v171 gate logic — record the variant name in the header comment, bump header to ' + V_VAR + '); markers UNCHANGED (' + FIRE_MARKER + ' / ' + VID_MARKER + '); flag-off byte-identity; py_compile + import + ScoringConfig() defaults unchanged.',
'SELF-TEST (paste output): CNV_FORCE_FIRE=1 restore path still works (n_fires>0, matched==n_fires) on 2 videos; a natural 2-video smoke showing the new gate logic firing/not-firing as designed.',
'Return implemented, selftest_passed, instrument_selftest, marker_fire, marker_vid, thresholds, eval_flags, files_changed, validation, notes.',
  ].join('\n'), { schema: implSchema, label: 'D2-variant-' + V_VAR, phase: 'Variant' }))
  if (varImpl && varImpl.implemented && varImpl.selftest_passed) {
    varRun = await runEval(V_VAR, 'cnv290', varImpl.eval_flags, 130, N_FULL, 'Variant')
    log('Variant eval: done=' + varRun.done + ' crashed=' + varRun.crashed + ' mAP=' + varRun.mAP + ' fires=' + varRun.fires)
    if (varRun.done && !varRun.crashed && varRun.marker_count > 0) {
      varBoot = await retryAgent(() => agent(bootPrompt(V_VAR, 'cnv290', 'Variant'), { schema: bootSchema, label: 'D-boot-var290', phase: 'Variant' }))
    }
  } else {
    log('Variant implementation failed self-test -> variant arm skipped')
  }
} else {
  log('Variant: skipped (' + (!g1.ran ? 'no valid v171 adjudication' : 'v171 already satisfies the gate') + ')')
}
const g2 = evaluateGate(varBoot)
const best = g2.promote ? { v: V_VAR, b: varBoot } : (g1.promote ? { v: V_CNV, b: boot } : null)
log('Final gate: v171 promote=' + g1.promote + ' | v172 promote=' + g2.promote + ' -> ' + (best ? ('PROMOTE ' + best.v) : 'CLOSE CNV'))

// ---------- PHASE 4: decide ----------
phase('Decide')
const decision = await retryAgent(() => agent([
'You are Agent C. Execute the CNV promotion decision and write the session record. Be scrupulously honest: CNV\'s objective is the NORMAL-VIDEO FALSE-ALARM RATE with mAP only as a non-inferiority constraint — do NOT re-judge it as an mAP mechanism, and do NOT let enthusiasm about the FA rate override a real mAP/GT harm signal. Plain text.',
'', CONTEXT, '',
'EVIDENCE:',
'  v171: ' + JSON.stringify(boot),
'  v172 (' + String(variantDir) + '): ' + JSON.stringify(varBoot),
'  gates: promote iff paired delta >= ' + MAP_TOL + ' AND abs >= ' + (REF290_MAP - ABS_TOL).toFixed(5) + ' AND FA relative reduction >= ' + (MIN_FA_REL_RED * 100) + '% AND gt_vetoed <= ' + MAX_GT_VETO,
'  computed: g1=' + JSON.stringify(g1) + ' g2=' + JSON.stringify(g2) + ' -> best=' + (best ? best.v : 'null'),
'',
'DO:',
 best
 ? ('1. PROMOTE ' + best.v + '. Back up config.py/run.py/batch_evaluate.py to ' + ROOT + '/.qoder/backups/cnv_promote/ , then set cnv_veto default True in the ScoringConfig dataclass, the from_args getattr default, and set_defaults in BOTH run.py and batch_evaluate.py (add --no_cnv_veto as opt-out). If ' + V_VAR + ' won, ensure the gate logic in the tree IS the ' + V_VAR + ' variant. VALIDATE and paste: py_compile; ScoringConfig() cnv_veto True; bue/cbt/cbt_recal still True; cgr/ccr still False; import MultiAnomalyVAD.')
 : ('1. DO NOT PROMOTE — confirm cnv_veto stays default False (paste ScoringConfig() output: cnv_veto False; bue/cbt/recal True; cgr/ccr False). Mark CNV MEASURED-CLOSED in the record with the real numbers (mAP delta+CI, FA rates both arms, gt_vetoed) so no future session re-litigates it blindly — and state plainly whether it closed on HARM (the idea is wrong for this pipeline) or on UNDER-FIRE with harm-free readings (the gate logic deserves one future retry with a stronger signal family, e.g. model_mil_v6 or a trained normality head).'),
'2. date +%Y%m%d ; write ' + LOGS + '/summary_v17_' + V_CNV + '-' + V_VAR + '_<YYYYMMDD>.md covering: the mechanism and WHY the objective is FA-rate not mAP (v9 oracle +0.00253); the paired-log instrument and why it is exact; v171 and (if ran) v172 numbers (mAP delta/CI/P, FA rates, FP dets/video, gt_vetoed, per-category); the TRAIN-split threshold verification; the decision; updated CLOSED-families list.',
'3. APPEND ledger rows to ' + LOGS + '/session_v97_ledger.md for ' + V_CNV + ' and, if it ran, ' + V_VAR + ': loop|version|title+verdict|smoke|mAP|AP@0.5|AP@0.7|lat|kept. Use -1 for absent values. Do NOT rewrite existing rows.',
'4. Reply with a 12-line recap leading with the headline: FA-rate change (off->on) and the paired mAP delta.',
'Return promoted, decision, validation (pasted), note.',
].join('\n'), { schema: decideSchema, label: 'C-decide-cnv', phase: 'Decide' }))

// ---------- PHASE 5: next-lever design (design-only, no eval) ----------
phase('NextDesign')
const design = await retryAgent(() => agent([
'You are the A/B/C research council producing the DESIGN-ONLY deliverable for the next session (v18). NO eval spend, NO implementation. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'THE OPEN PROBLEM (v16 session record): candidate RELOCATION is closed on every signal; the biggest single residual is Shoplifting CAND-MISS (the pipeline never proposes a window where the GT is — 11/25) plus flat-in-span CLIP category curves on RoadAccidents/Shoplifting. The lever family is SIGNAL QUALITY: category-specific CLIP prompt/query engineering (incl. CoOp-style or prompt-ensemble approaches), category-aware candidate generation/scan density, supervised category priors. HARD BANS: the 10 banned families (curve within-video re-rank / VLM merge-batch / max_scan scan-tail / per-video normalization / dense proposal APPEND / compute trims / boundary outward-growth=BUE / scan_segment-abs-conf gates / Normal-FP suppression FOR mAP / MIL-relocation on 32-snippet) AND the whole relocation family (v16).',
'DO (role-play all three passes yourself, in order, and label them):',
'  PASS A: propose TWO concrete mechanisms, each with the exact pipeline touch-points, expected effect channel (recall via new candidates vs better curve localization), and latency cost.',
'  PASS B: attack both proposals — why each might be flat on this harness (cite the harness noise floor ~+/-0.012 @290 and the v141-v149 closures); refine or replace the weaker one.',
'  PASS C (skeptical review): check both survivors against the HARD BANS and the closure ledger; reject or scope anything that smells like a banned family; pick ONE winner and define its pre-registered gate.',
'  OFFLINE PRE-CHECK (cheap, local, no eval): take 3-5 cached hard-cat result JSONs (or recompute CLIP category curves locally on 2-3 Shoplifting/RoadAccidents test videos — CLIP is local compute, allowed) and measure the curve-quality diagnostic the winner depends on (e.g. for a prompt-ensemble: in-GT-window curve z-score, prompt-mean vs best single prompt). Report go/no-go with pasted numbers.',
'Return proposals, winner, precheck (pasted evidence), note.',
].join('\n'), { schema: designSchema, label: 'Council-next-design', phase: 'NextDesign' }))

log('v17 complete. CNV: ' + (best ? ('PROMOTED as ' + best.v) : 'CLOSED') +
    ' | v171: ' + (boot && boot.ran ? ('delta=' + boot.point_delta.toFixed(5) + ' FA ' + boot.fa_video_rate_off.toFixed(3) + '->' + boot.fa_video_rate_on.toFixed(3) + ' gt_vetoed=' + boot.gt_vetoed) : 'n/a') +
    ' | v172: ' + (varBoot && varBoot.ran ? ('delta=' + varBoot.point_delta.toFixed(5) + ' FA->' + varBoot.fa_video_rate_on.toFixed(3)) : 'n/a') +
    ' | next-design: ' + (design ? design.winner.slice(0, 80) : 'n/a'))
return {
  cnv: { v171: boot, v172: varBoot, g1, g2, promoted: !!best, version: best ? best.v : null },
  decision: decision ? decision.decision : null,
  nextDesign: design,
}
