export const meta = {
  name: 'vad-team-map20-v26',
  description: 'VAD agent team v26 — FIVE REAL-EVAL rounds (v196-v200), each ending in a genuine hard-88 A/B adjudication (NO smoke-only rounds). Committed pipeline = v137 + BUE + CBT + CBT-recal (all default-ON; hard-88 REF ~0.1105-0.1181, full-290 ~0.173). Every cheap recall direction is closed (frames v18, prompt v20/v177, granularity v182, text-embed v181, smoothing v184, dense substrate v185, spatial zoom v188, backbone v189, VideoMAE curves v190, VideoMAE learner v191, learned localizers v192-v195 research-closed on missing localized train supervision); Normal-FP video-level veto exhausted (v17/v21/v180/v186/v187); MIL level-signal family closed (v183). The five rounds attack what remains admissible: R1 v196 VLC — VLM video-level calibration (one extra whole-video VLM read v in [0,1] from 16 uniform frames; conf*=0.5+0.5v; the cross-video confidence-LEVEL oracle was +0.0518 and the MIL-signal variant v183 is closed but a VLM-READ signal is untested) — paired-log conf restore. R2 v197 DSV — dual-model second-opinion verification (each final detection window re-scored by the CLASSIFY client qwen3-vl-plus, a DIFFERENT model from the scan qwen-vl-plus; conf*=0.6+0.4a; <=8 calls/video cap) — paired-log conf restore. R3 v198 MCS — max-over-13-categories pointwise CLIP curve for the BUE/CBT boundary operators instead of the resolved-category curve (deterministic post-VLM, count+conf frozen) — paired-json pre_mcs_boundaries. R4 v199 FDS — frame-density scan: hard-cat-gated videos scan at 4fps (24 frames/6s segment, same placement, double density; v18 KFD varied placement at FIXED count, v182 varied window granularity — density is untested) — cross-run vs a FRESH in-session REF88. R5 v200 SVA — scan-verdict aggregate video level (zero new VLM calls: level = mean of top-3 scan-segment verdict confidences already in the pipeline; conf*=0.5+0.5*level) — paired-log conf restore. GATES loop1 hard-88 (all rounds): point_delta >= +0.008 AND P(delta>0) >= 0.9 AND new mAP >= 0.11306; FCCS==0 VOID check; FM2 det/video increase flagged > +1.0; M4 matched==n_fires / bad_videos empty for paired modes. Loop2 full-290 (only if loop1 passes): delta >= -0.003 AND abs >= 0.16454 AND Normal-FA rel increase <= +2%. Every round: implement flag default-OFF + 2-video self-test with an exact RESTORE-VERIFY (paired modes must reproduce the OFF arm byte-exactly before any 88-eval spend) + real 88-eval + records (summary + one ledger row). Round failure isolation: a failed round records and the next proceeds. Concurrency 2, never two evals (16 GB swap-thrash). NONE of the 10 banned families. Args: none.',
  phases: [
    { title: 'R1-v196-VLC', detail: 'implement + paired-log hard-88' },
    { title: 'R2-v197-DSV', detail: 'implement + paired-log hard-88' },
    { title: 'R3-v198-MCS', detail: 'implement + paired-json hard-88' },
    { title: 'R4-v199-FDS', detail: 'implement + fresh REF88 + cross-run hard-88' },
    { title: 'R5-v200-SVA', detail: 'implement + paired-log hard-88' },
    { title: 'SessionRec', detail: 'session summary across the 5 rounds' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const PKG = ROOT + '/video_anomaly_detection'
const ITER = PKG + '/iteration_logs'
const PY = 'python3.14'
const HARNESS = '--categories Shooting,RoadAccidents,Shoplifting,Explosion'
const N_HARD = 88
const N_FULL = 290
const DATE = '20260822'

const RULES = [
  'NEVER edit the evaluator (evaluate.py), GT labels, or ' + ITER + '/analysis_bootstrap_ab.py.',
  'NEVER run two batch_evaluate processes concurrently (16 GB swap-thrash); pgrep -f batch_evaluate before every launch.',
  'NO Bash command may run > 150s in the foreground — the runtime kills stalled agents at 180s. Long work = nohup background + short polls (sleep <= 140 per call).',
  'grep claims must be GREPPED in produced artifacts, never asserted (FM20/FM21).',
  'All python invocations use ' + PY + '; all evals run with cwd=' + ROOT + ' and source ' + PKG + '/.env.',
  'Do NOT git-commit anything. StructuredOutput EXACTLY ONCE per agent (only where a schema is given).',
  'New flags default OFF in the dataclass AND from_args AND set_defaults in run.py + batch_evaluate.py.',
].join('\n')

const PAIREDLOG_FMT = [
  'PAIRED-LOG CONF MARKER FORMAT (FM21 — READ ' + ITER + '/analysis_bootstrap_ab.py restore_pre_marker ~L44-58 and MATCH EXACTLY):',
  'one line per changed detection: [<TAG>-FIRE] vid=<video_path> idx=<i> <cat> t=<t_start>-<t_end> conf=<old>-><new>',
  'regex keyed on: vid=(\\S+) ... t=(<ts>)-(<te>) ... conf=(<old>)->(<new>) with [\\d.]+ for confs — print plain decimals with >= 6 fraction digits, NEVER scientific notation.',
  'The OFF-arm restore rewrites confidence keyed on (vid, round(ts,2), round(te,2)); t values in the marker MUST be the FINAL post-mechanism window (the mechanism is conf-only so windows are unchanged).',
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

// ---------------- round specs ----------------
const ROUNDS = [
  {
    v: 'v196', tag: 'VLC', ph: 'R1-v196-VLC', cli: '--vlc_calib', flag: 'vlc_calib', mode: 'pairedlog',
    marker: 'VLC-FIRE', vidMarker: '[VLC-VID]', pairedField: '',
    impl: [
      'You are Agent D-implementer for v196 VLC (VLM video-Level Calibration). ' + RULES,
      'MOTIVATION: the cross-video confidence-LEVEL oracle is +0.0518 (v11 diagnostic); the MIL-signal variant is CLOSED (v183) but a VLM-READ video-level signal is untested. mAP ranks detections GLOBALLY by confidence, so calibrating each video\'s LEVEL moves mAP.',
      '1. BACKUP: cp pipeline.py anomaly_discovery.py config.py run.py batch_evaluate.py llm_client.py into ' + ROOT + '/.qoder/backups/v196/ .',
      '2. FIND the terminal post-VLM operator chain in pipeline.py (grep for the CNV/SNH slots ~L2417-2440; also the per-video JSON writer). FIND how the scan issues VLM calls (llm_client, detect model qwen-vl-plus) and how frames are extracted (VideoPreprocessor / _sample_keyframes) so you can reuse them.',
      '3. IMPLEMENT _vlc_video_calibrate: when config.scoring.vlc_calib is ON and the video has >= 1 final detection (post-CBT terminal slot, after the SNH slot, before JSON serialization): sample 16 frames UNIFORMLY across the WHOLE video duration (reuse existing extraction; timestamps logged), issue ONE extra detect-client (qwen-vl-plus) call with a video-level prompt: frames are time-ordered and span the entire video; "Does this video contain any abnormal activity (violence, accident, theft, vandalism, explosion, etc.)? Answer ONLY JSON {\\"anomaly_score\\": x} with x in [0,1]." Parse robustly (regex the first float in [0,1]); on ANY failure v=0.5 and log VLC-PARSE-FALLBACK. Then for EVERY detection: conf_new = conf_old * (0.5 + 0.5*v). Count-frozen, windows-frozen, order-frozen. Videos with 0 detections: SKIP the VLM call (cost) but still log the VID marker.',
      '4. MARKERS: ' + PAIREDLOG_FMT.replace(/<TAG>/g, 'VLC'),
      '   [VLC-VID] vid=<video_path> v=<v> ndet=<n> UNCONDITIONAL on EVERY path incl. early-return-normal / 0-detection (v=NA when no call was made) — grep -cF "[VLC-VID]" == #videos (FM17). [COST] instrumentation must count the +1 call/video.',
      '5. CONFIG: scoring.vlc_calib: bool = False (correct dataclass) + CLI --vlc_calib in run.py AND batch_evaluate.py (set_defaults False). Code comment tag v196.',
      '6. VALIDATE: py_compile all touched files; from_args defaults show vlc_calib False and every other flag unchanged; both --help show it; OFF code path adds ZERO lines/markers (byte-identical JSONs).',
      '7. SELF-TEST (queue-disciplined, nohup + short polls, cwd=' + ROOT + '): (a) OFF smoke: --categories Shoplifting,Explosion --max_per_category 1 -> results/batch_eval_v196_offsmoke (+ .log). (b) ON smoke: same 2 videos + --vlc_calib -> results/batch_eval_v196_onsmoke (+ .log). (c) greps: [VLC-VID]==2 in ON log; 0 VLC lines in OFF log; [VLC-FIRE] count == total detections across the 2 ON videos (every detection reweighted). (d) RESTORE-VERIFY (MANDATORY GATE): ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new results/batch_eval_v196_onsmoke --paired-log results/batch_eval_v196_onsmoke.log --marker VLC-FIRE --paired-mode conf --min-delta -99 --min-p 0 -> matched MUST == n_fires (M4); then verify the reconstructed OFF arm == the OFF-smoke JSONs EXACTLY (per (vid, t_start, t_end): confidence equal to 1e-9; write a 20-line compare script, paste output). If ANY mismatch -> fix the marker format and re-run until exact.',
      '8. Report: gating source details, marker greps, restore-verify output, [COST] delta. Verdict line: READY or NOT-READY (+reasons).',
    ].join('\n'),
    bootExtra: [
      'M4 CHECK: matched == n_fires from the tool output (paste it); if mismatch -> ran=false, VOID.',
      'SANITY: every [VLC-FIRE] has conf_new = conf_old * (0.5+0.5*v) with the video\'s v from its [VLC-VID] line — spot-check 3 videos arithmetically.',
    ],
    summaryWhat: 'VLC = VLM-read video-level calibration of detection confidence (the untested signal variant of the +0.0518 cross-video level oracle after v183 closed the MIL signal).',
  },
  {
    v: 'v197', tag: 'DSV', ph: 'R2-v197-DSV', cli: '--dsv_verify', flag: 'dsv_verify', mode: 'pairedlog',
    marker: 'DSV-FIRE', vidMarker: '[DSV-VID]', pairedField: '',
    impl: [
      'You are Agent D-implementer for v197 DSV (Dual-model Second-opinion Verification). ' + RULES,
      'MOTIVATION: the scan VLM (qwen-vl-plus) reads isolated 6s segments with uncalibrated absolute confidence (v143 root cause). An INDEPENDENT second model (the classify client qwen3-vl-plus — a different model) re-scoring each FINAL detection window adds an independent evidence read. Detection-level (not video-level like v196).',
      '1. BACKUP: cp pipeline.py anomaly_discovery.py config.py run.py batch_evaluate.py llm_client.py into ' + ROOT + '/.qoder/backups/v197/ .',
      '2. FIND the classify-stage client (qwen3-vl-plus; grep pipeline.py for the Stage-7 classify call ~L5076 region comment / anomaly_expert.py) and the terminal operator chain (~L2417-2440).',
      '3. IMPLEMENT _dsv_verify: when config.scoring.dsv_verify is ON, terminal slot (after SNH slot, before JSON serialization): for each final detection, sample up to 8 frames uniformly inside [t_start, t_end] (reuse existing extraction), call the CLASSIFY client (qwen3-vl-plus) with: frames come from a clip flagged as <category>; "Rate whether this clip genuinely shows <category>-type abnormal activity. Answer ONLY JSON {\\"score\\": a}, a in [0,1]." Parse robustly; failure -> a=0.5 + log DSV-PARSE-FALLBACK. conf_new = conf_old * (0.6 + 0.4*a). COST CAP: if a video has > 8 detections, apply only to the top-8 by confidence; others unchanged (no FIRE line for them). Count-frozen, windows-frozen. Videos with 0 detections: no calls, VID marker only.',
      '4. MARKERS: ' + PAIREDLOG_FMT.replace(/<TAG>/g, 'DSV'),
      '   [DSV-VID] vid=<video_path> ncalls=<k> ndet=<n> UNCONDITIONAL on EVERY path (FM17). [COST] counts the extra calls.',
      '5. CONFIG: scoring.dsv_verify: bool = False + CLI --dsv_verify in run.py AND batch_evaluate.py. Code comment tag v197.',
      '6. VALIDATE: py_compile; defaults all OFF/unchanged; both --help; OFF path byte-identical.',
      '7. SELF-TEST (queue-disciplined): (a) OFF smoke --categories Shoplifting,Explosion --max_per_category 1 -> results/batch_eval_v197_offsmoke. (b) ON smoke + --dsv_verify -> results/batch_eval_v197_onsmoke. (c) greps: [DSV-VID]==2; 0 DSV lines OFF; [DSV-FIRE] == min(ndet,8) per video. (d) RESTORE-VERIFY: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new results/batch_eval_v197_onsmoke --paired-log results/batch_eval_v197_onsmoke.log --marker DSV-FIRE --paired-mode conf --min-delta -99 --min-p 0 -> matched == n_fires; reconstructed OFF arm == OFF-smoke JSONs EXACTLY (1e-9; 20-line compare, paste). Fix until exact.',
      '8. Report: greps, restore-verify, [COST] delta. Verdict line: READY or NOT-READY (+reasons).',
    ].join('\n'),
    bootExtra: [
      'M4 CHECK: matched == n_fires (paste). Mismatch -> ran=false VOID.',
      'SANITY: spot-check 2 videos: conf_new = conf_old*(0.6+0.4*a) with a from the DSV lines.',
    ],
    summaryWhat: 'DSV = independent second-model (qwen3-vl-plus vs scan qwen-vl-plus) detection-level verification multiplier.',
  },
  {
    v: 'v198', tag: 'MCS', ph: 'R3-v198-MCS', cli: '--mcs_curves', flag: 'mcs_curves', mode: 'pairedjson',
    marker: 'MCS-FIRE', vidMarker: '[MCS-VID]', pairedField: 'pre_mcs_boundaries',
    impl: [
      'You are Agent D-implementer for v198 MCS (MaxCat curve Selection for BUE/CBT). ' + RULES,
      'MOTIVATION: BUE grows and CBT clamps detection boundaries along the RESOLVED-category CLIP curve. If the resolved category is wrong (or the event reads on another category\'s curve), boundary ops follow the wrong signal. MCS feeds both operators the POINTWISE MAX over all 13 category curves (maxcat(t) = max_c curve_c(t)). Deterministic post-VLM; zero new VLM calls; count-frozen; conf-frozen (BUE/CBT never touch confidence — VERIFY this and keep it so).',
      '1. BACKUP: cp pipeline.py anomaly_discovery.py config.py run.py batch_evaluate.py frame_scorer.py into ' + ROOT + '/.qoder/backups/v198/ .',
      '2. FIND: frame_scorer.py get_category_scores (~L41) returns the per-category curves; pipeline.py BUE call site (~L2340) and CBT/CBT-recal call site (~L2355) and which curve object each receives; the per-video result-JSON writer (grep for json.dump / \'anomalies\').',
      '3. IMPLEMENT: when config.scoring.mcs_curves is ON: (a) compute curve_mc(t) = pointwise max over the 13 category curves ONCE per video; (b) SNAPSHOT: deep-copy the detection list just before BUE; run the COMMITTED BUE+CBT(+recal) with the resolved-category curve on the copy -> these are the ARM-OFF windows; (c) run BUE+CBT(+recal) with curve_mc on the real list -> ARM-ON windows; (d) write the arm-OFF windows as a TOP-LEVEL per-video JSON field pre_mcs_boundaries: list of [t_start, t_end] INDEX-ALIGNED with the anomalies list (the v141 pre_cex_boundaries pattern — grep the pipeline for pre_cex_boundaries and COPY that plumbing exactly, incl. where it is injected into the JSON dict). CBT-recal per-category caps stay keyed on the RESOLVED category in BOTH arms (only the curve changes). Detection COUNT and confidence MUST be identical across arms (BUE/CBT are count/conf-neutral — if you find any path where they are not, STOP and report NOT-READY with the path).',
      '   PAIRED-JSON CONTRACT (READ ' + ITER + '/analysis_bootstrap_ab.py restore_pre_json ~L110-150): top-level field, index-aligned with anomalies, restores t_start/t_end only; length mismatch = VOID (M4).',
      '4. MARKERS: [MCS-VID] vid=<video_path> ndet=<n> n_changed=<k> UNCONDITIONAL on EVERY path incl. early-return / 0-detection (FM17, field present as [] when 0 detections so the field ALWAYS exists top-level). [MCS-FIRE] one per detection whose window changed (vid, idx, old->new window) — observability only.',
      '5. CONFIG: scoring.mcs_curves: bool = False + CLI --mcs_curves in run.py AND batch_evaluate.py. Code comment tag v198.',
      '6. VALIDATE: py_compile; defaults all OFF/unchanged; both --help; OFF path: NO pre_mcs_boundaries field, byte-identical JSONs, zero MCS lines.',
      '7. SELF-TEST (queue-disciplined): (a) OFF smoke --categories Shoplifting,Explosion --max_per_category 1 -> results/batch_eval_v198_offsmoke. (b) ON smoke + --mcs_curves -> results/batch_eval_v198_onsmoke. (c) greps: [MCS-VID]==2; 0 MCS lines OFF; every ON JSON has top-level pre_mcs_boundaries index-aligned (length == len(anomalies) — paste the check). (d) RESTORE-VERIFY: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new results/batch_eval_v198_onsmoke --paired-json pre_mcs_boundaries --min-delta -99 --min-p 0 -> bad_videos MUST be empty (paste); reconstructed OFF windows == OFF-smoke windows EXACTLY per (vid, idx) to 1e-9 AND confidences identical (20-line compare, paste output). Fix until exact.',
      '8. Report: which curve object each operator received before/after, greps, restore-verify output. Verdict line: READY or NOT-READY (+reasons).',
    ].join('\n'),
    bootExtra: [
      'M4 CHECK: bad_videos MUST be empty and n_fires == total detections (paste the tool output line). Any bad video -> ran=false VOID.',
      'SANITY: confirm 0 [COST] VLM-call delta vs a committed run (MCS is deterministic post-VLM): compare the [COST] totals in the ON log vs results/batch_eval_v198_offsmoke.log scaled — or vs the fresh REF88 if present.',
    ],
    summaryWhat: 'MCS = pointwise max-over-13-categories CLIP curve driving the BUE/CBT boundary operators (resolved-category curve is the committed choice; wrong-category curves corrupt boundary placement).',
  },
  {
    v: 'v199', tag: 'FDS', ph: 'R4-v199-FDS', cli: '--fds_scan', flag: 'fds_scan', mode: 'crossrun',
    marker: 'FDS-FIRE', vidMarker: '[FDS-VID]', pairedField: '',
    impl: [
      'You are Agent D-implementer for v199 FDS (Frame-Density Scan, hard-cat gated). ' + RULES,
      'MOTIVATION: the directed scan shows ~12 frames per 6s segment (2fps). Brief 1-3s events (RoadAccidents GT 1-3s, Shoplifting GT 2-7s) occupy ~2-6 of those frames. v18 KFD varied frame PLACEMENT at fixed count (CLOSED); v182 FSD varied window GRANULARITY (CLOSED); frame DENSITY at fixed window+placement is untested. FDS doubles the scan sampling to 4fps (24 frames/6s segment) on hard-category videos only.',
      '1. BACKUP: cp pipeline.py anomaly_discovery.py config.py run.py batch_evaluate.py into ' + ROOT + '/.qoder/backups/v199/ .',
      '2. FIND the directed-scan frame sampling: anomaly_discovery.py scan_segment (~L418) receives keyframes; _sample_keyframes / keyframes_per_segment and the >60-segment long-video rule; and the pipeline-resolved video category string (the one driving category-curve selection — grep where scan segments get built in pipeline.py).',
      '3. IMPLEMENT: when config.scoring.fds_scan is ON AND the video\'s resolved category is in {Shooting, RoadAccidents, Shoplifting, Explosion}: the DIRECTED scan samples 24 frames per 6s segment (4fps, same uniform placement, double density) instead of ~12. Same scan_segment prompt/JSON schema; same 6s segment grid; the >60-segment long-video rule keeps its STRUCTURE but on the doubled base (document the resulting counts). NOTHING else changes (candidate build/merge/NMS/classify/BUE/CBT untouched; FCCS switch unaffected). If the resolved-category label space never contains those 4 strings, say so LOUDLY and gate on the CLIP top-1 curve category instead (document).',
      '4. MARKERS: [FDS-FIRE] once per GATED video (vid, category, frames_per_segment 12 -> 24); [FDS-VID] vid=<video_path> gated=0|1 n_segments=<k> UNCONDITIONAL on EVERY path (FM17).',
      '5. CONFIG: scoring.fds_scan: bool = False + CLI --fds_scan in run.py AND batch_evaluate.py. Code comment tag v199.',
      '6. VALIDATE: py_compile; defaults all OFF/unchanged; both --help; OFF path byte-identical.',
      '7. SELF-TEST (queue-disciplined): (a) OFF smoke --categories Normal,Shoplifting --max_per_category 1 -> results/batch_eval_v199_offsmoke: grep -c FDS log == 0. (b) ON smoke + --fds_scan -> results/batch_eval_v199_onsmoke: [FDS-VID]==2; if the Shoplifting video is gated, [FDS-FIRE] shows 12->24 and its scan-call [COST] frame-token count ~doubles vs OFF (paste the numbers); if NOT gated (label-space miss), switch gating source per step 3 and re-run. Paste everything. Verdict line: READY or NOT-READY (+reasons).',
    ].join('\n'),
    bootExtra: [
      'CONFOUND GATES: FCCS==0 in BOTH logs (paste greps); [FDS-VID] ~88 in the FDS log; [FDS-FIRE] > 0 (mechanism actually gated videos, FM5; if 0 -> ran=false scream).',
      'Paste the cross-run CAVEAT from the tool verbatim (±0.019 VLM run-to-run variance).',
      'COST: [COST] VLM call count + frame-token totals on gated videos, FDS vs REF (expect ~2x tokens on gated).',
    ],
    summaryWhat: 'FDS = 4fps (24-frame) directed-scan sampling on hard-category videos — frame DENSITY at fixed window+placement, the untested axis after KFD (placement) and FSD (granularity) closed.',
  },
  {
    v: 'v200', tag: 'SVA', ph: 'R5-v200-SVA', cli: '--sva_calib', flag: 'sva_calib', mode: 'pairedlog',
    marker: 'SVA-FIRE', vidMarker: '[SVA-VID]', pairedField: '',
    impl: [
      'You are Agent D-implementer for v200 SVA (Scan-Verdict Aggregate video-level calibration). ' + RULES,
      'MOTIVATION: same cross-video LEVEL oracle (+0.0518) as v196, but ZERO new VLM calls: the directed scan already produces a per-segment verdict confidence for every 6s segment. A video where the top scan segments all read high is likelier truly anomalous than a video with one isolated high segment. level = mean of the top-3 scan-segment verdict confidences in the video; conf_new = conf_old * (0.5 + 0.5*level). Distinct signal from v196 (dedicated video-level read) and from the CLOSED MIL-signal variant (v183) — this is the VLM\'s OWN aggregate.',
      '1. BACKUP: cp pipeline.py anomaly_discovery.py config.py run.py batch_evaluate.py into ' + ROOT + '/.qoder/backups/v200/ .',
      '2. FIND where the directed scan\'s per-segment verdicts (with confidences) are collected in pipeline.py / anomaly_discovery.py (the scan results structure feeding candidate build). If they are not retained through to the terminal slot, retain them when the flag is ON (default-OFF path must not change memory/behavior).',
      '3. IMPLEMENT _sva_calibrate: terminal slot (after SNH slot, before JSON serialization), when config.scoring.sva_calib is ON and ndet >= 1: level = mean(top-3 scan-segment verdict confidences) (if < 3 segments, mean of all; confidences already in [0,1] — clamp). conf_new = conf_old * (0.5 + 0.5*level) for every detection. Count-frozen, windows-frozen.',
      '4. MARKERS: ' + PAIREDLOG_FMT.replace(/<TAG>/g, 'SVA'),
      '   [SVA-VID] vid=<video_path> level=<l> nseg=<k> ndet=<n> UNCONDITIONAL on EVERY path (level=NA when no segments) (FM17). [COST] delta MUST be 0 — verify in self-test.',
      '5. CONFIG: scoring.sva_calib: bool = False + CLI --sva_calib in run.py AND batch_evaluate.py. Code comment tag v200.',
      '6. VALIDATE: py_compile; defaults all OFF/unchanged; both --help; OFF path byte-identical.',
      '7. SELF-TEST (queue-disciplined): (a) OFF smoke --categories Shoplifting,Explosion --max_per_category 1 -> results/batch_eval_v200_offsmoke. (b) ON smoke + --sva_calib -> results/batch_eval_v200_onsmoke. (c) greps: [SVA-VID]==2; 0 SVA lines OFF; [SVA-FIRE] == total detections; [COST] totals IDENTICAL to OFF smoke (paste both). (d) RESTORE-VERIFY: ' + PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new results/batch_eval_v200_onsmoke --paired-log results/batch_eval_v200_onsmoke.log --marker SVA-FIRE --paired-mode conf --min-delta -99 --min-p 0 -> matched == n_fires; reconstructed OFF arm == OFF-smoke JSONs EXACTLY (1e-9; compare script, paste). Fix until exact.',
      '8. Report: greps, restore-verify, [COST]-zero proof. Verdict line: READY or NOT-READY (+reasons).',
    ].join('\n'),
    bootExtra: [
      'M4 CHECK: matched == n_fires (paste). Mismatch -> ran=false VOID.',
      'COST-ZERO PROOF: [COST] VLM call totals in the ON log == a committed run\'s (paste comparison).',
    ],
    summaryWhat: 'SVA = zero-cost cross-video level calibration from the scan\'s own top-3 segment verdict aggregate (third signal variant of the level oracle after MIL v183-closed and VLM-read v196).',
  },
]

// ---------------- round runner ----------------
const roundResults = []

for (const spec of ROUNDS) {
  phase(spec.ph)
  log(spec.v + ' ' + spec.tag + ': implement (' + spec.mode + ')')
  const rec = { v: spec.v, tag: spec.tag, mode: spec.mode, ready: false, pass88: false, promote: false, b88: null, b290: null, newMap: -1, refMap: -1 }

  const impl = await retryAgent(() => agent(spec.impl, { label: 'D-impl-' + spec.v, phase: spec.ph }))
  if (!impl || /NOT-READY/.test(impl)) {
    log(spec.v + ' NOT-READY — records-lite, no eval spend. ' + (impl || 'null').slice(0, 300))
    await retryAgent(() => agent([
      'You are Agent C (records) for ' + spec.v + ' ' + spec.tag + '. The mechanism FAILED its implementation/self-test gate — NO eval was spent. ' + RULES,
      'IMPLEMENTER OUTPUT: ' + (impl || 'null').slice(0, 3000),
      '1. Write ' + ITER + '/summary_' + spec.v + '_' + spec.tag.toLowerCase() + '-notready_' + DATE + '.md: what ' + spec.summaryWhat + ' ; why NOT-READY (verbatim reasons); what was tried; whether the direction stays open or closes.',
      '2. Append ONE ledger row to ' + ITER + '/session_v97_ledger.md: loop | ' + spec.v + ' | ' + spec.tag + ' NOT-READY ... | kept=false.',
      '3. Ensure flag ' + spec.flag + ' stays default False (grep + paste). Report files written.',
    ].join('\n'), { label: 'C-rec-notready-' + spec.v, phase: spec.ph }))
    roundResults.push(rec)
    continue
  }
  rec.ready = true
  log(spec.v + ' READY — launching hard-88 eval(s)')

  // ---- loop 1 eval(s) ----
  let ref88 = null
  if (spec.mode === 'crossrun') {
    ref88 = await runEval(spec.v + '_ref88', HARNESS, '[COST]', N_HARD, spec.ph)
    if (!ref88.done || ref88.fccs_count !== 0) {
      log(spec.v + ' ABORT: REF88 ' + (ref88.done ? 'FCCS confound' : 'incomplete') + ': ' + ref88.note)
      ref88 = null
    }
  }
  let new88 = { done: false, crashed: true, mAP: -1, marker_count: -1, fccs_count: -1, note: 'skipped: REF88 failed' }
  if (spec.mode !== 'crossrun' || ref88) {
    new88 = await runEval(spec.v + '_' + spec.tag.toLowerCase() + '88', HARNESS + ' ' + spec.cli, spec.vidMarker, N_HARD, spec.ph)
  }
  rec.newMap = new88.mAP
  rec.refMap = ref88 ? ref88.mAP : -2

  const doBoot = new88.done && new88.fccs_count === 0 && new88.marker_count >= 80 && (spec.mode !== 'crossrun' || (ref88 && ref88.done))
  if (!doBoot) {
    log(spec.v + ' eval problem: done=' + new88.done + ' fccs=' + new88.fccs_count + ' markers=' + new88.marker_count + ' — records-lite')
  }

  // ---- loop 1 adjudication ----
  let b88 = null
  if (doBoot) {
    const newDir = ROOT + '/results/batch_eval_' + spec.v + '_' + spec.tag.toLowerCase() + '88'
    const newLog = newDir + '.log'
    let cmd = ''
    if (spec.mode === 'pairedlog') cmd = PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --paired-log ' + newLog + ' --marker ' + spec.marker + ' --paired-mode conf --min-delta 0.008 --min-p 0.9'
    else if (spec.mode === 'pairedjson') cmd = PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --paired-json ' + spec.pairedField + ' --min-delta 0.008 --min-p 0.9'
    else cmd = PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --ref ' + ROOT + '/results/batch_eval_' + spec.v + '_ref88 --min-delta 0.008 --min-p 0.9'
    const BOOT88 = [
      'You are Agent D-adjudicator for the ' + spec.v + ' ' + spec.tag + ' hard-88 A/B (' + spec.mode + '). Bash < 150s per call. ' + RULES,
      'STEP 1: ' + cmd + ' -> paste the FULL output (VERDICT line, point_delta, ci90, p_gt0, new/ref mAP; cross-run also prints a CAVEAT — paste it verbatim).',
    ].concat(spec.bootExtra).concat([
      'STEP 2 (FM2 watch): mean detections/video in the new dir' + (spec.mode === 'crossrun' ? ' vs the ref dir' : ' (and confirm it equals the committed arm — paired modes are count-frozen)') + '; flag increase > +1.0.',
      'STEP 3: per-category AP deltas (4 hard cats) into per_category (compute per-category mAP both arms with evaluate.py helpers; short scripts OK). fa_rel_inc: no Normal videos in hard-88 -> -1.',
      'GATES loop1: point_delta >= +0.008 AND p_gt0 >= 0.9 AND new_map >= 0.11306. Report all schema fields; note = VERDICT verbatim + key numbers.',
      'new_map for paired modes = the OBSERVED (arm-ON) mAP of ' + newDir + '; ref_map = reconstructed arm-OFF mAP from the tool output.',
    ]).join('\n')
    b88 = await retryAgent(() => agent(BOOT88, { schema: bootSchema, label: 'D-boot88-' + spec.v, phase: spec.ph }))
    rec.b88 = b88
    rec.pass88 = !!(b88 && b88.ran && b88.point_delta >= 0.008 && b88.p_gt0 >= 0.9 && b88.new_map >= 0.11306)
    log(spec.v + ' loop1: pass=' + rec.pass88 + ' delta=' + (b88 && b88.point_delta) + ' P=' + (b88 && b88.p_gt0) + ' new=' + (b88 && b88.new_map))
  }

  // ---- loop 2 (conditional) ----
  if (rec.pass88) {
    log(spec.v + ' loop1 PASSED — arming full-290 net-neutrality')
    let ref290 = null
    if (spec.mode === 'crossrun') {
      ref290 = await runEval(spec.v + '_ref290', '', '[COST]', N_FULL, spec.ph)
      if (!ref290.done) ref290 = null
    }
    let new290 = { done: false }
    if (spec.mode !== 'crossrun' || ref290) {
      new290 = await runEval(spec.v + '_' + spec.tag.toLowerCase() + '290', spec.cli, spec.vidMarker, N_FULL, spec.ph)
    }
    if (new290.done && new290.marker_count >= 280 && (spec.mode !== 'crossrun' || ref290)) {
      const newDir = ROOT + '/results/batch_eval_' + spec.v + '_' + spec.tag.toLowerCase() + '290'
      const newLog = newDir + '.log'
      let cmd = ''
      if (spec.mode === 'pairedlog') cmd = PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --paired-log ' + newLog + ' --marker ' + spec.marker + ' --paired-mode conf --min-delta -0.003 --min-p 0.5'
      else if (spec.mode === 'pairedjson') cmd = PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --paired-json ' + spec.pairedField + ' --min-delta -0.003 --min-p 0.5'
      else cmd = PY + ' ' + ITER + '/analysis_bootstrap_ab.py --new ' + newDir + ' --ref ' + ROOT + '/results/batch_eval_' + spec.v + '_ref290 --min-delta -0.003 --min-p 0.5'
      const BOOT290 = [
        'You are Agent D-adjudicator for the ' + spec.v + ' ' + spec.tag + ' full-290 NET-NEUTRALITY confirm (' + spec.mode + '). Bash < 150s. ' + RULES,
        'STEP 1: ' + cmd + ' -> full output (CAVEAT verbatim if cross-run). M4 checks as in loop1.',
        'STEP 2: Normal-video FA rate both arms (category==Normal videos with >= 1 detection conf > 0); fa_rel_inc = (new-ref)/ref. det/video both arms.',
        'GATES loop2: point_delta >= -0.003 AND new_map >= 0.16454 AND fa_rel_inc <= +0.02. Same schema fields (purpose_converted -> "n/a at 290").',
      ].join('\n')
      rec.b290 = await retryAgent(() => agent(BOOT290, { schema: bootSchema, label: 'D-boot290-' + spec.v, phase: spec.ph }))
    } else {
      log(spec.v + ' loop2 eval problem: done=' + new290.done + ' markers=' + new290.marker_count)
    }
    rec.promote = !!(rec.b290 && rec.b290.ran && rec.b290.point_delta >= -0.003 && rec.b290.new_map >= 0.16454 && rec.b290.fa_rel_inc <= 0.02)
  }

  // ---- records ----
  await retryAgent(() => agent([
    'You are Agent C (records) for ' + spec.v + ' ' + spec.tag + '. Write the records from the data below; re-grep artifacts to verify (FM20). Do NOT re-run evals. ' + RULES,
    'MECHANISM: ' + spec.summaryWhat,
    'MODE: ' + spec.mode + (spec.mode === 'pairedjson' ? ' field=' + spec.pairedField : '') + (spec.mode === 'pairedlog' ? ' marker=' + spec.marker : ''),
    'LOOP1: ' + JSON.stringify(rec.b88) + ' new88_mAP=' + new88.mAP + (ref88 ? ' ref88_mAP=' + ref88.mAP : ''),
    'LOOP2 (null if not armed): ' + JSON.stringify(rec.b290),
    'GATES: loop1 pass=' + rec.pass88 + ' promote=' + rec.promote,
    '1. Write ' + ITER + '/summary_' + spec.v + '_' + spec.tag.toLowerCase() + '_' + DATE + '.md: verdict; motivation (incl. which closures it respects); implementation facts (gating source, marker counts, restore-verify evidence); loop1 numbers (caveat verbatim if cross-run); loop2 numbers or not-armed; per-category table; det/video + cost; gate table; FM evidence (greps); decision; family update (precisely what is now refuted/validated).',
    '2. Append ONE ledger row to ' + ITER + '/session_v97_ledger.md: loop | ' + spec.v + ' | ' + spec.tag + ' ... | kept=<true|false> with numbers, FM greps, per-cat, revert/promote state.',
    '3. DECISION EXECUTION: NOT promote -> ' + spec.flag + ' stays default False (grep + paste), code in-tree inert, one-line config closure comment. promote -> backup ' + ROOT + '/.qoder/backups/' + spec.v + '_promote/ FIRST; flip ' + spec.flag + ' default True (dataclass + from_args + set_defaults both CLIs) + --no_' + spec.flag + ' opt-out; py_compile + defaults check.',
    '4. Report files written + final flag defaults.',
  ].join('\n'), { label: 'C-records-' + spec.v, phase: spec.ph }))

  roundResults.push(rec)
  log(spec.v + ' complete: pass88=' + rec.pass88 + ' promote=' + rec.promote)
}

// ============================ SESSION RECORD ============================
phase('SessionRec')
await retryAgent(() => agent([
  'You are Agent C (session records) for the v26 five-round session. Do NOT re-run evals; verify by grep (FM20). ' + RULES,
  'ROUND RESULTS: ' + JSON.stringify(roundResults.map(r => ({ v: r.v, tag: r.tag, mode: r.mode, ready: r.ready, pass88: r.pass88, promote: r.promote, newMap: r.newMap, refMap: r.refMap, d88: r.b88 && r.b88.point_delta, p88: r.b88 && r.b88.p_gt0, m88: r.b88 && r.b88.new_map, d290: r.b290 && r.b290.point_delta, m290: r.b290 && r.b290.new_map }))),
  '1. Write ' + ITER + '/summary_v26_v196-v200_' + DATE + '.md: one row per round (mechanism, mode, loop1 delta/P/new_map, loop2 if armed, verdict); the cross-round picture (which signals of the cross-video confidence-LEVEL oracle — MIL v183 / VLM-read v196 / second-model v197 / scan-aggregate v200 — are now tested and what the answers were; whether boundary-op curve selection v198 and frame density v199 moved anything); the remaining OPEN admissible space after this session, stated honestly (if none >= +0.008-point scale remains, say so); artifacts index (eval dirs, summaries, ledger rows).',
  '2. Verify every round wrote its summary + ledger row (grep -l); fix any missing row by appending it from the data above.',
  '3. Final flag-default table: EVERY experimental flag (cgr/ccr/cnv/kfd/tcp/snh/snh_protect/pte/fsd/vlc_calib/dsv_verify/mcs_curves/fds_scan/sva_calib) with its default after this session (grep config.py + paste).',
  '4. Report files written.',
].join('\n'), { label: 'C-session-rec-v26', phase: 'SessionRec' }))

log('v26 complete: ' + JSON.stringify(roundResults.map(r => ({ v: r.v, ready: r.ready, pass88: r.pass88, promote: r.promote, d88: r.b88 && r.b88.point_delta }))))
