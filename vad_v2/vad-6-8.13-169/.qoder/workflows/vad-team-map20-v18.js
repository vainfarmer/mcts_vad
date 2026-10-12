export const meta = {
  name: 'vad-team-map20-v18',
  description: 'VAD agent team v18 — implement + adjudicate KFD-U (Uniform-spread keyframes in the FCCS comparative scan), the v17 council\'s winning design with a CODE-EXACT root cause and a GO offline pre-check. ROOT CAUSE (verified by the v17 council): for every 6s segment, _sample_keyframes returns all ~12 ordered frames (target K=18 >= 12), then _comparative_call (anomaly_discovery.py L110-113) TRUNCATES to keyframes[:keyframes_per_segment] = the FIRST 2 frames (t+0.0s, t+0.5s), or the FIRST 1 under the >60-segment long-video rule — so the VLM\'s evidence for each 6s segment is at best its leading ~0.5s, and brief mid-segment events (RoadAccidents GT 2-3s, Shoplifting GT 2-7s) are literally never shown. Offline coverage arithmetic: committed first-k shows GT intervals in only 36/44 hard-cat videos; KFD-U k=2 {t_start, t_start+3.0s} + midpoint long-video rule covers 43/44 at ZERO cost (same call count, same 16 frames/call, same batch geometry). P1 (PW-TC prompt-weight refit) was REJECTED by its own pre-registered offline gate (one-hot collapse, mixed-sign transfer, offline->online inversion reproduced) — do NOT revive it this session. MECHANISM: replace the first-k truncation with a deterministic uniform spread, flag kfd_uniform_keyframes default OFF, CLI in run.py AND batch_evaluate.py, markers KFD-FIRE per affected batch + KFD-VID unconditionally per video on every path. WHY CROSS-RUN (pre-registered by the council): KFD changes the VLM INPUTS, so the exact-paired instruments (paired-json/paired-log) CANNOT reconstruct the arm-OFF arm — Loop 1 is a cross-run video-level bootstrap vs a FRESH committed REF run in the same session. GATES (pre-registered): Loop 1 hard-88 — PROMOTE-TO-290 iff point_delta >= +0.008 AND P(delta>0) >= 0.9 AND new hard mAP >= 0.11306 (= committed 0.11806 - 0.005); count discipline — flag if mean detections/video > +1.0 vs REF. Loop 2 (only if Loop 1 passes) full-290 net-neutrality — delta >= -0.003 AND abs >= 0.16454 AND Normal-video FA-rate relative increase <= +2%. Fallback (exactly once, only if Loop 1 is positive-but-sub-gate with KFD-FIRE confirmed): k=3 spread ({t, t+2.5, t+5}, 44/44 coverage offline, +50% scan tokens), same gate. Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON); cgr/ccr/cnv all default-OFF and stay so. Concurrency 2.',
  phases: [
    { title: 'Implement', detail: 'D1 implements kfd_uniform_keyframes (default OFF) + kfd_k knob, self-tests spread-vs-truncation indices, markers, byte-identity' },
    { title: 'AB88', detail: 'fresh committed REF88 + KFD88 (hard-88, concurrency 2) -> cross-run bootstrap, per-cat, detection-count check' },
    { title: 'FallbackK3', detail: 'conditional: k=3 arm on hard-88 iff Loop 1 positive-but-sub-gate with fires confirmed' },
    { title: 'Confirm290', detail: 'if gate passed: fresh REF290 + KFD290, net-neutrality + Normal-FA bound' },
    { title: 'Decide', detail: 'promote kfd_uniform_keyframes default-ON iff both loops pass; else archive-close; summary + ledger' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const CONCURRENCY = 2
const HARD_CATS = 'Shooting,RoadAccidents,Shoplifting,Explosion'
const HARNESS = '--categories ' + HARD_CATS
const N_HARD = 88
const N_FULL = 290
const MECH_BASE = '--bue_extension --qscore_rerank --cbt_tighten --cbt_recal'
const KFD_FLAG = '--kfd_uniform_keyframes'
const FIRE_MARKER = 'KFD-FIRE'
const VID_MARKER = 'KFD-VID'
const V_IMPL = 'v173'
const V_290 = 'v174'
const V_K3 = 'v175'
// pre-registered gates (v17 council)
const MIN_DELTA_88 = 0.008
const MIN_P = 0.90
const HARD_FLOOR = 0.11306        // committed 0.11806 - 0.005
const MAP_TOL_290 = -0.003
const ABS_FLOOR_290 = 0.16454     // REF290 0.17254 - 0.008
const MAX_FA_REL_INC = 0.02       // Normal-video FA-rate relative increase bound at 290
const MAX_DET_INC = 1.0           // mean detections/video increase flag (FM2 watch)

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime. Repo ' + ROOT + '; code ' + CODE + '. INTERPRETER ' + PY + '.',
'GIT RULE: the uncommitted working tree IS the latest code. NEVER git checkout/reset/stash/clean. Back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'',
'!!! COMMITTED PIPELINE: v137 + BUE + CBT + CBT-recal ALL DEFAULT-ON. full-290 ~0.173 (v160_full 0.17254); hard-88 ~0.11806. cgr_enabled=False, cgr_ab_diag=False, ccr_enabled=False, cnv_veto=False (v16 closed relocation; v17 closed CNV on HARM). Do NOT change any committed default. !!!',
'',
'!!! THE MECHANISM — KFD-U (v17 council winner, design + offline pre-check COMPLETE) !!!',
'ROOT CAUSE (code-exact): _sample_keyframes (anomaly_discovery.py L45-56) returns ALL ~12 ordered frames of a 6s @2fps segment (SemanticFrameSampler target K=18 >= 12), then _comparative_call (anomaly_discovery.py L110-113) TRUNCATES to keyframes[:keyframes_per_segment] — the FIRST 2 frames (t+0.0s, t+0.5s), or the FIRST 1 under the >60-segment long-video rule. Semantic dedup only shrinks the pre-truncation pool, so the VLM sees at best the leading ~0.5s of every 6s segment. OFFLINE VERIFIED: committed shows >=1 keyframe inside the GT interval in only 36/44 hard-cat videos (invisible GTs: RoadAccidents009/017/021/121/122/131 + Shoplifting017/037); KFD-U k=2 covers 43/44.',
'FIX (pre-registered): when kfd_uniform_keyframes is ON, replace the first-k truncation with a deterministic uniform spread: k=2 -> frame indices nearest {t_start, t_start+3.0s}; the >60-segment 1-keyframe rule -> segment MIDPOINT. Same segment grid, same batch sizes (8x2 / 16x1), same 16 frames/call, same call count, same prompts, same thresholds/merge/top-k — ZERO latency cost. Placement ablation was pre-registered offline: {t,t+3.0s} 43/44 > {t+2,t+4} 41/44 > {t+1.5,t+4.5} 40/44 > {t+1,t+5} 38/44 > linspace-to-edges 37/44 — do NOT substitute another placement.',
'FALLBACK ARM (exactly once, only if triggered): k=3 spread {t, t+2.5, t+5} (44/44 offline, +50% scan tokens). Expose it as a kfd_k config (default 2) with CLI --kfd_k; k=3 changes batch geometry to 8x3 — that is expected for the fallback only.',
'',
'!!! WHY CROSS-RUN (no paired instrument this time) !!!',
'KFD changes the VLM INPUTS — the arm-OFF arm cannot be replayed deterministically from one run. Loop 1 therefore uses TWO fresh same-session runs (committed REF + KFD-on) judged by analysis_bootstrap_ab.py --new <kfd> --ref <ref> --min-delta ' + MIN_DELTA_88 + ' --min-p ' + MIN_P + ' (video-level bootstrap, 1000 resamples). The tool prints the cross-run variance caveat — include it verbatim in the record.',
'',
'!!! GATES (pre-registered by the v17 council — apply exactly) !!!',
'Loop 1 (hard-88): PROMOTE-TO-290 iff point_delta >= +' + MIN_DELTA_88 + ' AND P(delta>0) >= ' + MIN_P + ' AND new hard mAP >= ' + HARD_FLOOR + '. FM2 watch: flag if mean detections/video rises > +' + MAX_DET_INC + ' vs REF (a recall mechanism MAY add detections, but not explode).',
'Loop 2 (full-290, only if Loop 1 passes): net-neutrality — delta >= ' + MAP_TOL_290 + ' AND abs >= ' + ABS_FLOOR_290 + ' AND Normal-video FA-rate relative increase <= +' + (MAX_FA_REL_INC * 100) + '% (harm bound for the recall direction: KFD also shows Normal videos mid-segment frames, which could create NEW hallucinations there — measure it).',
'',
'FM DISCIPLINE: FM5 ' + FIRE_MARKER + ' grep > 0 in the PRODUCED log. FM17 ' + VID_MARKER + ' unconditional per video on EVERY path incl. 0-detection and early-return-normal (mirror the CNV call sites D1 added: pipeline.py ~L1658/L1787/L2129). FM20 grep, not claim. FM21 exact marker strings (' + FIRE_MARKER + ' / ' + VID_MARKER + '). FM18 read ABSOLUTE mAP. FM6: the offline coverage gain (36->43/44) DOES NOT imply online gain — v159 PMX inverted offline->online; the gate is the online numbers. HARNESS BUG (cost v14 two loops): --categories must be in every hard-88 eval command — the launcher bakes it in; watcher verifies num_videos=' + N_HARD + '.',
'HARD CONSTRAINTS: flag default OFF (flag-off byte-identical, including the exact first-k truncation path); CLI in BOTH run.py and batch_evaluate.py; evals FIXED --concurrency ' + CONCURRENCY + '; StructuredOutput EXACTLY ONCE per agent. ANTI-STALL: no Bash > ~150s except the stated self-test smoke; Read with offset/limit.',
].join('\n')

const implSchema = { type: 'object', properties: { implemented: { type: 'boolean' }, selftest_passed: { type: 'boolean' }, instrument_selftest: { type: 'string', description: 'PASTED actual evidence: (a) on a real smoke, flag-ON selected frame indices match the {t_start, t_start+3.0s} / midpoint pattern and flag-OFF indices are byte-identical to the committed first-k truncation; (b) call count and frames/call identical on/off (latency-neutral proof from the [COST] lines or logs); (c) ' + FIRE_MARKER + ' grep > 0 and ' + VID_MARKER + ' grep == number of smoke videos incl. empty/early-normal paths; (d) --kfd_k 3 produces the {t,t+2.5,t+5} pattern' }, marker_fire: { type: 'string' }, marker_vid: { type: 'string' }, eval_flags: { type: 'string', description: 'KFD-on flag string = ' + MECH_BASE + ' + ' + KFD_FLAG + ' (no --categories; the launcher adds it for 88 runs)' }, files_changed: { type: 'string' }, validation: { type: 'string', description: 'py_compile + import MultiAnomalyVAD + flag-off byte-identity: ScoringConfig()/DiscoveryConfig() defaults unchanged and from_args(Namespace()) == defaults' }, notes: { type: 'string' } }, required: ['implemented', 'selftest_passed', 'instrument_selftest', 'marker_fire', 'marker_vid', 'eval_flags', 'files_changed', 'validation'], additionalProperties: false }
const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, num_videos: { type: 'number' }, marker_count: { type: 'number' }, fires: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'fires', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, ref_map: { type: 'number' }, det_per_video_new: { type: 'number' }, det_per_video_ref: { type: 'number' }, fa_rel_inc: { type: 'number', description: '290 only: Normal-video FA-rate relative increase (on-off)/off; omit/0 at 88' }, per_category: { type: 'string' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'ref_map', 'note'], additionalProperties: false }
const decideSchema = { type: 'object', properties: { promoted: { type: 'boolean' }, decision: { type: 'string' }, validation: { type: 'string' }, note: { type: 'string' } }, required: ['promoted', 'decision', 'note'], additionalProperties: false }

async function retryAgent(fn, tries) {
  const n = tries || 3
  for (let t = 0; t < n; t++) {
    try { const r = await fn(); if (r !== null && r !== undefined) return r; log('retryAgent ' + (t + 1) + '/' + n + ' null') }
    catch (e) { log('retryAgent ' + (t + 1) + '/' + n + ' threw: ' + ((e && e.message) ? e.message : String(e)).slice(0, 160)) }
  }
  return null
}

function evalLaunchPrompt(V, tag, evalFlags, forceHarness) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const harn = forceHarness ? (HARNESS + ' ') : ''
  return [
'You are Agent D-launcher. Ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. SHORT; no Bash > 150s. StructuredOutput EXACTLY ONCE.',
'output_dir=' + dir + ' ; log=' + dir + '.log ; report=' + dir + '/evaluation_report.json',
'REQUIRED FLAGS (verbatim): ' + harn + evalFlags,
'1. ps aux|grep -i batch_evaluate|grep -v grep. 2. Already running for this V_tag -> adopt if the flags match, else kill and relaunch. 3. report exists AND the log shows the required flags -> launched=true, note="already complete", do not relaunch. 4. A DIFFERENT version is running -> sleep 140 (up to 3x), then report blocked.',
'5. Otherwise write ' + dir + '/COMMAND.txt and launch EXACTLY:',
"bash -lc 'mkdir -p " + dir + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + dir + " --max_per_category 999 --concurrency " + CONCURRENCY + " " + harn + evalFlags + " > " + dir + ".log 2>&1 & echo PID $!'",
'6. sleep 30; tail -5 the log to catch an immediate Traceback. Return launched, note.',
  ].join('\n')
}
function evalWatchPrompt(V, tag, vidMarker, fireMarker, nTotal) {
  const dir = ROOT + '/results/batch_eval_' + V + '_' + tag
  const rep = dir + '/evaluation_report.json'
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + ' (expect ' + nTotal + ' videos). SHORT; NO Bash > 150s. StructuredOutput EXACTLY ONCE.',
'1. If ' + rep + ' exists -> done=true. Read it: ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti.get(\'num_videos\'))" . num_videos MUST be ' + nTotal + '; if not, say so loudly (harness bug).',
'   markers: VC=$(grep -cF "' + vidMarker + '" ' + dir + '.log 2>/dev/null); FC=$(grep -cF "' + fireMarker + '" ' + dir + '.log 2>/dev/null) -> marker_count=VC, fires=FC, note carries "vidMarker=<VC> fires=<FC>".' + (vidMarker === VID_MARKER ? ' VC must be ~' + nTotal + ' (FM17); FC>0 proves the spread actually changed selected frames (FM5).' : ' (REF arm: no KFD markers expected; VC/FC may be 0.)'),
'2. Else pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". Alive -> sleep 140 (up to 2x), then done=false note="running <n>/' + nTotal + '" (n = count of result JSONs). STALLED >30min -> kill and relaunch the SAME nohup at --concurrency ' + CONCURRENCY + '.',
'3. Dead with no report -> tail -30 the log. TRANSIENT -> relaunch. CODE BUG -> crashed=true with the traceback.',
'Return done, crashed, mAP, ap30, ap50, ap70, num_videos, marker_count, fires, note.',
  ].join('\n')
}
async function runEval(V, tag, evalFlags, vidMarker, fireMarker, maxWatch, nTotal, ph, forceHarness) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, evalFlags, forceHarness), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: ph }))
  if (!launch || !launch.launched) return { done: false, crashed: true, mAP: -1, marker_count: -1, fires: -1, note: 'launch failed: ' + (launch ? launch.note : 'null') }
  if (launch.note && launch.note.indexOf('already complete') >= 0) {
    const st0 = await retryAgent(() => agent(evalWatchPrompt(V, tag, vidMarker, fireMarker, nTotal), { schema: watchSchema, label: 'D-watch-adopt-' + tag + '-' + V, phase: ph }))
    if (st0 && st0.done) return st0
  }
  for (let w = 0; w < (maxWatch || 130); w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, vidMarker, fireMarker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: ph }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, marker_count: -1, fires: -1, note: 'watcher cap reached' }
}
function boot88Prompt(kfdDir, refDir, label, ph) {
  return [
'You are Agent D-adjudicator for the KFD-U CROSS-RUN A/B on the hard-88 (' + label + '). Bash < 150s per command. StructuredOutput EXACTLY ONCE.',
'STEP 1 (FM5/FM20): FC=$(grep -cF "' + FIRE_MARKER + '" ' + kfdDir + '.log); VC=$(grep -cF "' + VID_MARKER + '" ' + kfdDir + '.log). VC must be ~88 and FC>0, else ran=false with the counts and STOP.',
'STEP 2 (the number): ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + kfdDir + ' --ref ' + refDir + ' --min-delta ' + MIN_DELTA_88 + ' --min-p ' + MIN_P + ' -> point_delta, ci90_low/high, p_gt0, new_map (KFD abs), ref_map (fresh committed abs). Paste the cross-run variance CAVEAT line verbatim into note.',
'STEP 3 (FM2 watch): mean detections-per-video in both dirs (det_per_video_new/ref). Flag loudly if the increase exceeds +' + MAX_DET_INC + '.',
'STEP 4: per-category AP deltas for Shooting/RoadAccidents/Shoplifting/Explosion into per_category. ALSO: did KFD produce ANY detection overlapping the previously-invisible GTs (RoadAccidents009/017/021/121/122/131, Shoplifting017/037) at IoU>=0.1 where REF had none? Name the videos — this is the mechanism\'s stated purpose.',
'Report ran, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, det_per_video_new, det_per_video_ref, per_category, note (VERDICT verbatim + caveat).',
  ].join('\n')
}
function boot290Prompt(kfdDir, refDir, ph) {
  return [
'You are Agent D-adjudicator for the KFD-U net-neutrality confirm on the FULL 290. Cross-run A/B (KFD changes VLM inputs — no paired instrument exists). Bash < 150s. StructuredOutput EXACTLY ONCE.',
'STEP 1 (FM5/FM20): grep -cF "' + FIRE_MARKER + '" and "' + VID_MARKER + '" in ' + kfdDir + '.log — VC ~290, FC>0, else ran=false and STOP.',
'STEP 2: ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + kfdDir + ' --ref ' + refDir + ' --min-delta ' + MAP_TOL_290 + ' --min-p ' + MIN_P + ' -> point_delta, ci90, p_gt0, new_map, ref_map. Paste the cross-run CAVEAT verbatim.',
'STEP 3 (Normal-FA harm bound — the recall direction\'s risk): on Normal-category videos, fraction with >=1 detection and mean detections/video, both dirs; fa_rel_inc=(on-off)/off (0 if off==0). Bound is +' + (MAX_FA_REL_INC * 100) + '% relative. Note for context: two same-pipeline committed runs measured FA rates 0.5133 vs 0.52667 (v171/v172 arm-OFF restores) — cross-run drift on this metric is ~+/-2.6% relative, so report the raw rates AND the drift band honestly.',
'STEP 4: per-category AP deltas (all 13 cats, flag |delta|>0.005). Detection-count means both dirs.',
'Report ran, point_delta, ci90_low, ci90_high, p_gt0, new_map, ref_map, det_per_video_new, det_per_video_ref, fa_rel_inc, per_category, note.',
  ].join('\n')
}

// ============================ MAIN ============================
log('v18 starting. KFD-U: the v17 council winner — uniform-spread keyframes fix a CODE-EXACT structural blindness (VLM sees only the leading ~0.5s of each 6s segment; 8/44 hard-cat GTs never shown). Zero-cost recall lever. Gates: 88 delta >= +' + MIN_DELTA_88 + ' & P >= ' + MIN_P + ' & abs >= ' + HARD_FLOOR + '; 290 net-neutral & FA inc <= +' + (MAX_FA_REL_INC * 100) + '%. concurrency ' + CONCURRENCY + '.')

// ---------- PHASE 1: implement ----------
phase('Implement')
const impl = await retryAgent(() => agent([
'You are Agent D1. Implement KFD-U exactly as the v17 council pre-registered it, then PROVE it. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '',
'IMPLEMENT:',
'1. Back up every file you touch to ' + ROOT + '/.qoder/backups/' + V_IMPL + '/ FIRST; ls the backup dir.',
'2. DiscoveryConfig: kfd_uniform_keyframes (default False) + kfd_k (default 2). CLI --kfd_uniform_keyframes and --kfd_k in BOTH run.py and batch_evaluate.py, threaded through VADConfig.from_args. Flag-off byte-identical (the exact first-k truncation path untouched when off).',
'3. anomaly_discovery.py ~L110-113 (_comparative_call): when the flag is on, replace keyframes[:keyframes_per_segment] with a deterministic uniform spread — k=2: the two frames whose timestamps are nearest {seg_start, seg_start+3.0s}; the >60-segment 1-keyframe rule: the frame nearest the segment MIDPOINT; k=3 (fallback knob): nearest {seg_start, seg_start+2.5s, seg_start+5.0s}. Deterministic (no randomness), index-safe (clamp/dedup within the segment\'s frame list), order preserved by timestamp. Do NOT change the segment grid, batch sizes, call count, prompts, thresholds, merge, or top-k. Verify by reading the code that _comparative_call is the ONLY truncation site on the FCCS comparative path (scan_full_coverage -> _scan_comparative_batch_guarded -> _comparative_call).',
'4. MARKERS (FM21 exact): ' + FIRE_MARKER + ' once per batch where the spread changed the selected indices vs the first-k truncation (carry vid + segment indices + old->new index lists, one line). ' + VID_MARKER + ' unconditionally per video on EVERY return path (segments scanned, batches, fires) INCLUDING 0-detection and early-return-normal — mirror the CNV call sites D1 added (pipeline.py ~L1658/L1787/L2129).',
'5. Bump the pipeline.py header to ' + V_IMPL + '. Do NOT touch any committed default (BUE/CBT/recal on; cgr/ccr/cnv off).',
'',
'HARD SELF-TEST (mandatory, paste ACTUAL output):',
'  SMOKE: ' + PY + ' -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir ' + ROOT + '/results/batch_eval_' + V_IMPL + '_selftest --max_per_category 1 --concurrency 1 ' + HARNESS + ' ' + MECH_BASE + ' ' + KFD_FLAG,
'  Confirm and paste: (a) grep -cF ' + FIRE_MARKER + ' > 0 AND grep -cF ' + VID_MARKER + ' == #videos; (b) a debug print proving flag-ON indices follow {t, t+3.0s}/midpoint and a SECOND run with the flag OFF produces the committed first-k indices on the same segments (byte-identical selection); (c) VLM call count and frames/call identical on/off (from the [COST] lines / logs — latency-neutral proof); (d) --kfd_k 3 smoke on 1 video shows the {t, t+2.5, t+5} pattern.',
'VALIDATE: py_compile; import MultiAnomalyVAD; both flags in BOTH --help; DiscoveryConfig() defaults unchanged; from_args(Namespace()) == defaults.',
'Return implemented, selftest_passed, instrument_selftest (pasted), marker_fire, marker_vid, eval_flags, files_changed, validation, notes.',
'Self-test smoke ~4 videos at concurrency 1 (~5-10 min) is expected. No OTHER Bash > ~150s.',
].join('\n'), { schema: implSchema, label: 'D1-implement-' + V_IMPL, phase: 'Implement' }))

const kfdOk = !!(impl && impl.implemented && impl.selftest_passed && impl.eval_flags)
let ref88 = null, kfd88 = null, boot88 = null, k3Run = null, bootK3 = null, ref290 = null, kfd290 = null, boot290 = null, bestArm = null
if (!kfdOk) {
  log('Phase 1: KFD-U NOT verified (selftest=' + (impl ? impl.selftest_passed : 'null') + ') -> stopping before eval spend')
} else {
  log('Phase 1: KFD-U verified. flags=' + impl.eval_flags)

  // ---------- PHASE 2: hard-88 cross-run A/B (fresh REF first, then KFD) ----------
  phase('AB88')
  ref88 = await runEval(V_IMPL, 'ref88', MECH_BASE, 'CBT-VID', 'CBT-VID', 70, N_HARD, 'AB88', true)
  log('REF88: done=' + ref88.done + ' mAP=' + ref88.mAP + ' n=' + (ref88.num_videos != null ? ref88.num_videos : '?') + ' ' + (ref88.note || ''))
  if (ref88.done && !ref88.crashed) {
    kfd88 = await runEval(V_IMPL, 'kfd88', impl.eval_flags, VID_MARKER, FIRE_MARKER, 70, N_HARD, 'AB88', true)
    log('KFD88: done=' + kfd88.done + ' mAP=' + kfd88.mAP + ' n=' + (kfd88.num_videos != null ? kfd88.num_videos : '?') + ' vidMarker=' + kfd88.marker_count + ' fires=' + kfd88.fires)
  }
  if (kfd88 && kfd88.done && !kfd88.crashed && kfd88.marker_count > 0 && kfd88.fires > 0) {
    boot88 = await retryAgent(() => agent(boot88Prompt(ROOT + '/results/batch_eval_' + V_IMPL + '_kfd88', ROOT + '/results/batch_eval_' + V_IMPL + '_ref88', 'k=2', 'AB88'), { schema: bootSchema, label: 'D-boot-88', phase: 'AB88' }))
  } else {
    log('AB88: eval unusable (or zero KFD fires -> mechanism never activated) -> no adjudication')
  }
}
const pass88 = !!(boot88 && boot88.ran && boot88.point_delta >= MIN_DELTA_88 && boot88.p_gt0 >= MIN_P && boot88.new_map >= HARD_FLOOR)
const positiveSubgate = !!(boot88 && boot88.ran && !pass88 && boot88.point_delta > 0)
log('88 verdict: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3) + ' absKFD=' + boot88.new_map.toFixed(5) + ' absREF=' + boot88.ref_map.toFixed(5) + ' det/v ' + (boot88.det_per_video_ref != null ? boot88.det_per_video_ref.toFixed(2) : '?') + '->' + (boot88.det_per_video_new != null ? boot88.det_per_video_new.toFixed(2) : '?') + ' -> ' + (pass88 ? 'PASSES gate' : positiveSubgate ? 'POSITIVE-SUBGATE (fallback eligible)' : 'FAILS')) : 'no valid adjudication'))

// ---------- PHASE 3: conditional k=3 fallback (exactly once) ----------
phase('FallbackK3')
if (positiveSubgate && impl) {
  log('Fallback: launching the pre-registered k=3 arm ({t,t+2.5,t+5}, +50% scan tokens) on hard-88')
  k3Run = await runEval(V_K3, 'kfd88', impl.eval_flags + ' --kfd_k 3', VID_MARKER, FIRE_MARKER, 70, N_HARD, 'FallbackK3', true)
  log('K3-88: done=' + k3Run.done + ' mAP=' + k3Run.mAP + ' fires=' + k3Run.fires)
  if (k3Run.done && !k3Run.crashed && k3Run.fires > 0 && ref88 && ref88.done) {
    bootK3 = await retryAgent(() => agent(boot88Prompt(ROOT + '/results/batch_eval_' + V_K3 + '_kfd88', ROOT + '/results/batch_eval_' + V_IMPL + '_ref88', 'k=3 fallback', 'FallbackK3'), { schema: bootSchema, label: 'D-boot-k3', phase: 'FallbackK3' }))
  }
} else {
  log('Fallback: skipped (' + (pass88 ? 'k=2 already passed' : 'no positive sub-gate signal') + ')')
}
const passK3 = !!(bootK3 && bootK3.ran && bootK3.point_delta >= MIN_DELTA_88 && bootK3.p_gt0 >= MIN_P && bootK3.new_map >= HARD_FLOOR)
if (pass88) bestArm = { k: 2, flags: impl ? impl.eval_flags : null }
else if (passK3) bestArm = { k: 3, flags: impl ? impl.eval_flags + ' --kfd_k 3' : null }
log('Best arm: ' + (bestArm ? ('k=' + bestArm.k) : 'none'))

// ---------- PHASE 4: 290 confirm ----------
phase('Confirm290')
if (bestArm) {
  ref290 = await runEval(V_290, 'ref290', MECH_BASE, 'CBT-VID', 'CBT-VID', 130, N_FULL, 'Confirm290', false)
  log('REF290: done=' + ref290.done + ' mAP=' + ref290.mAP + ' n=' + (ref290.num_videos != null ? ref290.num_videos : '?'))
  if (ref290.done && !ref290.crashed) {
    kfd290 = await runEval(V_290, 'kfd290', bestArm.flags, VID_MARKER, FIRE_MARKER, 130, N_FULL, 'Confirm290', false)
    log('KFD290: done=' + kfd290.done + ' mAP=' + kfd290.mAP + ' n=' + (kfd290.num_videos != null ? kfd290.num_videos : '?') + ' fires=' + kfd290.fires)
  }
  if (kfd290 && kfd290.done && !kfd290.crashed && kfd290.fires > 0 && ref290.done) {
    boot290 = await retryAgent(() => agent(boot290Prompt(ROOT + '/results/batch_eval_' + V_290 + '_kfd290', ROOT + '/results/batch_eval_' + V_290 + '_ref290', 'Confirm290'), { schema: bootSchema, label: 'D-boot-290', phase: 'Confirm290' }))
  }
} else {
  log('Confirm290: skipped — no arm passed the 88 gate')
}
const ok290 = !!(boot290 && boot290.ran && boot290.point_delta >= MAP_TOL_290 && boot290.new_map >= ABS_FLOOR_290 && (boot290.fa_rel_inc == null || boot290.fa_rel_inc <= MAX_FA_REL_INC))
const promote = !!(bestArm && ok290)
log('290 verdict: ' + (boot290 && boot290.ran ? ('delta=' + boot290.point_delta.toFixed(5) + ' abs=' + boot290.new_map.toFixed(5) + ' FA rel inc=' + (boot290.fa_rel_inc != null ? (boot290.fa_rel_inc * 100).toFixed(1) + '%' : '?') + ' -> ' + (ok290 ? 'NET-NEUTRAL OK' : 'FAILS')) : 'n/a') + ' -> promote=' + promote)

// ---------- PHASE 5: decide ----------
phase('Decide')
const decision = await retryAgent(() => agent([
'You are Agent C. Execute the KFD-U promotion decision and write the session record. Be scrupulously honest, including about cross-run noise: this A/B cannot use the paired instrument (VLM inputs change), so say plainly what the cross-run caveat means for the strength of the conclusion. Plain text.',
'', CONTEXT, '',
'EVIDENCE:',
'  88 k=2: ' + JSON.stringify(boot88),
'  88 k=3 fallback: ' + JSON.stringify(bootK3),
'  290 (' + (bestArm ? 'k=' + bestArm.k : 'none') + '): ' + JSON.stringify(boot290),
'  gates: 88 delta >= +' + MIN_DELTA_88 + ' AND P >= ' + MIN_P + ' AND abs >= ' + HARD_FLOOR + '; 290 delta >= ' + MAP_TOL_290 + ' AND abs >= ' + ABS_FLOOR_290 + ' AND FA rel inc <= +' + (MAX_FA_REL_INC * 100) + '%',
'  computed: pass88=' + pass88 + ' passK3=' + passK3 + ' ok290=' + ok290 + ' -> promote=' + promote + (bestArm ? ' (k=' + bestArm.k + ')' : ''),
'',
'DO:',
 promote
 ? ('1. PROMOTE kfd_uniform_keyframes (k=' + bestArm.k + '). Back up config.py/run.py/batch_evaluate.py/anomaly_discovery.py to ' + ROOT + '/.qoder/backups/kfd_promote/ , then set kfd_uniform_keyframes default True' + (bestArm.k === 3 ? ' AND kfd_k default 3' : '') + ' in the DiscoveryConfig dataclass, the from_args getattr default, and set_defaults in BOTH run.py and batch_evaluate.py (add --no_kfd_uniform_keyframes as opt-out). VALIDATE and paste: py_compile; defaults show kfd on (k=' + bestArm.k + '); bue/cbt/recal True; cgr/ccr/cnv False; import MultiAnomalyVAD.')
 : ('1. DO NOT PROMOTE — confirm kfd_uniform_keyframes stays default False (paste defaults; bue/cbt/recal True; cgr/ccr/cnv False). Record the closure honestly: distinguish "coverage was necessary but not sufficient — the VLM did not convert newly-visible GTs into detections" (read the per-video overlap evidence) from "newly-visible GTs converted but symmetric-loss on other segments cancelled it". The structural fix stays in the tree default-OFF; state what evidence would justify revisiting (e.g. a scan-prompt change that makes the VLM act on mid-segment frames).'),
'2. date +%Y%m%d ; write ' + LOGS + '/summary_v18_' + V_IMPL + '-' + V_290 + '_<YYYYMMDD>.md covering: the code-exact root cause and the offline coverage arithmetic (36->43/44); why cross-run not paired; the 88 (and if ran, k=3 and 290) numbers with CI/P/caveat verbatim; per-category; the previously-invisible-GT conversion list (which of RoadAccidents009/017/021/121/122/131, Shoplifting017/037 gained an overlapping detection); Normal-FA read at 290; the decision; updated CLOSED list.',
'3. APPEND ledger rows to ' + LOGS + '/session_v97_ledger.md for ' + V_IMPL + ' (88), and if they ran ' + V_K3 + ' (k=3) and ' + V_290 + ' (290): loop|version|title+verdict|smoke|mAP|AP@0.5|AP@0.7|lat|kept. -1 for absent. Do NOT rewrite existing rows.',
'4. Reply with a 12-line recap leading with the headline number (88 cross-run delta, P) and the decision.',
'Return promoted, decision, validation (pasted), note.',
].join('\n'), { schema: decideSchema, label: 'C-decide-kfd', phase: 'Decide' }))

log('v18 complete. KFD-U: ' + (promote ? ('PROMOTED k=' + bestArm.k) : 'NOT promoted') +
    ' | 88: ' + (boot88 && boot88.ran ? ('delta=' + boot88.point_delta.toFixed(5) + ' P=' + boot88.p_gt0.toFixed(3)) : 'n/a') +
    ' | k3: ' + (bootK3 && bootK3.ran ? ('delta=' + bootK3.point_delta.toFixed(5)) : 'n/a') +
    ' | 290: ' + (boot290 && boot290.ran ? ('delta=' + boot290.point_delta.toFixed(5) + ' abs=' + boot290.new_map.toFixed(5)) : 'n/a'))
return {
  arm88_k2: boot88, arm88_k3: bootK3, arm290: boot290,
  pass88, passK3, ok290, promote, bestK: bestArm ? bestArm.k : null,
  decision: decision ? decision.decision : null,
}
