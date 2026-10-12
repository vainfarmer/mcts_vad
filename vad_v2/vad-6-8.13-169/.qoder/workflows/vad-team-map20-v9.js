export const meta = {
  name: 'vad-team-map20-v9',
  description: 'VAD agent team v9 (ACCURACY continuation on the FIXED 65-video harness). CONTINUES after the v8 arc: v143 CGR (additive VLM recall rescue — REVERT, floor breached, Δ-0.018 P=0.174; additive dets became high-ranked FPs that DEMOTED precision under global AP, FM2 materialized; root cause = UNCALIBRATED scan_segment absolute conf pins ~0.9 on isolated clips) and v144 PACR (count-neutral peak-anchored relocation — REVERT, NO VERDICT: crashed at the FM5 gate on an empty-video marker bug, mechanism never actually evaluated). DECISIVE 290-video result: BUE (Boundary-Union Extension) was PAIRED-adjudicated on all 290 test videos and CONFIRMED: +0.02164 mAP (REF 0.12949 -> NEW 0.15114), 90% CI [-0.0010,+0.0502], P(Δ>0)=0.938 — the FIRST mechanism of the whole session to clear both the effect-size (>=0.02) and confidence (>=0.85) gates on a properly-powered harness. BY OWNER DECISION BUE NOW SHIPS DEFAULT-ON (config bue_enabled=True; --no_bue_extension to disable), so the committed pipeline = v137 + BUE. v9 runs TWO more iteration versions v145->v146 on the SAME 65-video harness (--max_per_category 5, concurrency 2), each attacking the STILL-OPEN candidate-generation / recall-ceiling headroom (50/99 detections have ZERO GT overlap, median GT distance ~20s — BUE grows existing windows but CANNOT reach these) or supervised score quality, verdict = video-matched BOOTSTRAP CI. Concurrency is 2 (owner-diagnosed 2026-07-31: concurrency 3 exhausted the 16 GB machine\'s unified memory and thrashed swap, making 6/141 CLIP-precompute ops run 7-44x slow; concurrency 2 loads the CLIP/VideoMAE models once and holds only 2 per-video frame buffers, which fits).',
  phases: [
    { title: 'Baseline', detail: 'establish the fresh BUE-ON committed 65-video reference (REF for cross-run + the current baseline mAP)' },
    { title: 'Research', detail: 'A and B independently research >=+0.03-effect candidate-generation/recall or score-quality mechanisms (NOT boundary-growth = committed BUE; NOT uncalibrated VLM abs-conf gates = refuted v143/v144), each proposes one' },
    { title: 'Debate', detail: 'A and B cross-critique on effect size and measurability, then synthesize one joint proposal' },
    { title: 'Review', detail: 'C gates on FM1-FM17 + the >=+0.03 rule + banned families + the v143/v144 calibration lessons; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (flag + marker incl. empty-video path), smoke + in-vivo sentinels -> full 65-video eval -> bootstrap-CI adjudication, keep/revert' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'
const MAX_LOOPS = 2
const START_VERSION = 145
const SMOKE_GATE = 0.06
const BASE_MAP = 0.15632                 // committed v137 (65-video, historical, BUE-off scale)
const MAP_TOL = 0.005
let   FLOOR = BASE_MAP - MAP_TOL         // 0.15132; re-derived from the fresh BUE-ON baseline in Phase 0
const CONCURRENCY = 2                     // owner-diagnosed 2026-07-31: conc 3 thrashed 16 GB swap; 2 fits

// Fresh BUE-ON committed 65-video reference produced in Phase 0 (also the cross-run REF).
const REF_V = 'v144c'
const REF_TAG = 'bue65ref'
let   REF_DIR = ROOT + '/results/batch_eval_' + REF_V + '_' + REF_TAG
let   REF_MAP = -1                        // filled by Phase 0
const REF_FLAGS = '--bue_extension --qscore_rerank'   // committed pipeline (BUE now default-on; explicit for clarity)

const MIN_DELTA_CROSS = 0.02             // cross-run KEEP bar
const MIN_DELTA_PAIRED = 0.01            // paired KEEP bar
const MIN_P = 0.90                       // bootstrap P(delta>0) required on the 65-harness

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline; config.py = all params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'INTERPRETER: use ' + PY + ' for everything (plain python3 lacks numpy/torch/open_clip).',
'',
'!!! OBJECTIVE: RAISE full-eval temporal mAP on the FIXED 65-video harness (--max_per_category 5). Do NOT propose enlarging it. !!!',
'',
'!!! CURRENT COMMITTED STATE INHERITED BY v9 (read carefully) !!!',
'- Committed pipeline = v137 + BUE. **BUE (Boundary-Union Extension) NOW SHIPS DEFAULT-ON** (config.py bue_enabled=True; run.py/batch_evaluate set_defaults(bue_enabled=True); opt-out --no_bue_extension). It was paired-adjudicated on ALL 290 test videos: +0.02164 mAP (0.12949->0.15114), 90% CI [-0.0010,+0.0502], P(Δ>0)=0.938 — the first mechanism this session to clear the effect-size AND confidence gates on a properly-powered harness. BUE grows each FINAL detection outward on the read-only category curve (UNION of old window + AutoLoc OIC-contrast argmax window). It is DONE. Do NOT re-propose or modify it.',
'- v141 CEX (--cex_extend) remains default-OFF, superseded, untouched. Leave it.',
'- Phase 0 of THIS workflow re-measures the committed pipeline (BUE-on) on the 65-video harness to give the current baseline mAP and the cross-run REF. The historical committed number 0.15632 was on the BUE-OFF 65 scale; the fresh Phase-0 number is the real current baseline.',
'',
'!!! THE TWO IMMEDIATELY-PRECEDING FAILURES — DO NOT REPEAT (v143, v144) !!!',
'BOTH attacked the candidate-generation/recall family (the 50/99 zero-overlap detections) and BOTH failed on the SAME underlying issue:',
'- v143 CGR (additive per-video 2nd-pass VLM rescue, accept iff conf>firstmax+0.05 AND conf>=0.55): REVERT, ACCURACY FLOOR BREACHED. Cross-run bootstrap Δ=-0.01800, P(Δ>0)=0.174. The additive detections behaved as HIGH-RANKED FALSE POSITIVES that, under GLOBAL AP ranking, DEMOTED precision at the very 14th/20th hit checkpoints it targeted (AP@0.5 0.1464->0.1098, AP@0.7 0.1044->0.0909). FM2 over-fire MATERIALIZED. Root cause = scan_segment\'s "even subtle signs count" prompt is an UNCALIBRATED ABSOLUTE threshold that pins ~0.9 on isolated gap-stratum clips, so the abs/relative gate admitted off-GT FPs. The pre-registered smoke over-fire + on-GT sentinels were listed PENDING and SKIPPED before spending the eval.',
'- v144 PACR (count-NEUTRAL width-preserving TRANSLATION of far-mislocated dets onto the in-vivo category-curve peak): REVERT, NO VERDICT — the run CRASHED at the FM5 [PACR-GATE] because an empty/0-detection video (Robbery050) SKIPPED the per-video summary marker, so markers(13) < succeeded(14) and no report was written; relaunch could not recover (JSONs already existed). The mechanism was NEVER evaluated. Its degeneracy sentinel (accept_rate<=0.5, no uniform-0.9 conf-pin, on-GT>=off-GT) was never run in vivo.',
'LESSONS NOW MANDATORY: (a) NO mechanism may gate on RAW scan_segment ABSOLUTE confidence — it is uncalibrated and pins ~0.9 on isolated clips (this bit v143 AND v144). Use a VERDICT-INDEPENDENT, PRE-CALIBRATABLE localizer instead (MIL snippet scores / model_mil_v6.pth / actionness / the read-only category curve). (b) ADDITIVE detections are dangerous under global AP (FM2): prefer COUNT-NEUTRAL mechanisms, and if additive, bound the FP-demotion cost analytically first. (c) Any per-video activation marker MUST also fire on the EMPTY/0-detection path (FM17, below) or the [GATE] deadlocks. (d) Pre-registered in-vivo SENTINELS (accept-rate, conf-pin, on-GT>=off-GT) MUST actually run in the SMOKE pass before any full eval is spent.',
'',
'!!! THE NOISE FLOOR (v140, measured) !!!',
'A video-level bootstrap on the 65 videos gives a 90% CI half-width ~+/-0.025 for a mAP delta. The +0.005 gate used pre-v7 is 5x below resolution; effects <~+0.03 are unfalsifiable on this harness. "N cached runs agree" is near-worthless (81% shared detections). VERDICTS MUST BE VIDEO-MATCHED BOOTSTRAP CIs, and PAIRED mode (within-run, mechanism logs conf=<old>-><new> or pre-mechanism windows) is STRONGLY PREFERRED because it removes VLM run-to-run noise (~+/-0.019) — and because BUE is now committed, a cross-run comparison vs an old BUE-off reference would conflate BUE with the new mechanism, whereas paired isolates the new mechanism cleanly. n_gt=66, 11-point interpolated AP, max recall 0.379@0.3 => mAP is decided by precision at ~4 hit checkpoints, so ONE detection crossing ONE checkpoint moves mAP 0.005-0.010.',
'THEREFORE v9 admits ONLY proposals whose HONEST expected effect is >= +0.03. C MUST REJECT anything smaller as unmeasurable.',
'',
'!!! BANNED FAMILIES (proposing one is an automatic C REJECT) !!!',
'1. CURVE-DERIVED CONFIDENCE RE-RANKING in any form (v111 Q-score, v112, v115-v123, v138-v140 GPC/bins). Within-video oracle only +0.006; committed Q-score already claims it.',
'2. VLM call MERGING / BATCHING / contact-sheets / verdict propagation (v97 FCCS, v126 TRLM, v129 BDS) — all cratered mAP to ~0.12; per-call independence is load-bearing.',
'3. max_scan / scan-tail / Source-2 padding pruning (v125, v130, v131) — max_scan already binds; mean_dropped=0.00.',
'4. PER-VIDEO score normalization / re-leveling (z-score -0.0898, rank -0.0931) — global AP depends on the cross-video LEVEL.',
'5. DENSE BOTTOM-RANKED PROPOSAL APPEND / "DPA" — metric gaming (signal-free RNG reproduces most of it).',
'6. Adaptive-fps / local-compute trims (v128) — invisible to the deciding metrics.',
'7. BOUNDARY OUTWARD-GROWTH / UNION-EXTENSION / OIC RELOCATION / per-side growth — this is COMMITTED BUE. Do not duplicate or modify it.',
'8. (v9) ADDITIVE OR RELOCATION GATES ON RAW scan_segment ABSOLUTE CONFIDENCE — refuted twice (v143 CGR, v144 PACR): uncalibrated, pins ~0.9 on isolated clips. A localizer must be verdict-independent and pre-calibratable.',
'',
'!!! WHERE THE >= +0.03 HEADROOM ACTUALLY IS (measured) !!!',
'- CANDIDATE-GENERATION / RECALL CEILING is the binding, STILL-OPEN constraint: max recall 0.379@0.3, 0.212@0.5, 0.121@0.7; **50 of 99 detections have ZERO GT overlap** (median IoU 0.000, median GT distance ~20s) — the pipeline looks in the WRONG PLACE for half its detections. BUE cannot reach these (it only grows EXISTING windows). This is the family with >= +0.03 of physical headroom, but it is DANGEROUS under global AP (v143 proved additive dets demote precision) — the winning design must put detections on the RIGHT events without adding uncalibrated FPs (count-neutral relocation via a verdict-independent localizer is the leading unrefuted idea; PACR had the right shape but a fatal impl bug).',
'- Untapped SUPERVISED assets (verdict-independent, pre-calibratable — exactly what v143/v144 lacked): features/train (1610 videos) + features/test (290) pre-extracted CLIP snippets; model_mil.pth (v1, in production at fusion weight 0.25); model_mil_v6.pth (better standalone, prior fusion attempt FAILED at FM1 — a NON-fusion use may differ); prompt_weights.json (v94 CoOp-lite, proven-safe upstream). The deep-research report concludes the gap to SOTA is SCORE QUALITY.',
'- Pure re-rank ORACLE ceiling 0.27273; raising RECALL raises the ceiling itself.',
'',
'!!! MEASUREMENT PROTOCOL - MANDATORY !!!',
'Verdict statistic = VIDEO-MATCHED BOOTSTRAP CI. Tool:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <new_full_dir> --ref <REF_DIR>                 [cross-run vs the fresh BUE-ON baseline]',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <new_full_dir> --paired-log <run.log> --marker <MARKER>   [paired from log]',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <new_full_dir> --paired-json <FIELD>            [paired from per-JSON pre-mechanism field]',
'KEEP requires: point delta >= ' + MIN_DELTA_CROSS + ' (cross-run) or >= ' + MIN_DELTA_PAIRED + ' (paired) AND P(delta>0) >= ' + MIN_P + ' AND full mAP >= the floor. PREFER PAIRED.',
'',
'!!! REUSABLE ASSETS (use them; do not rebuild) !!!',
'- ' + LOGS + '/analysis_v140_livecurve_lib.py: reproduces the IN-VIVO verdict-independent category curve EXACTLY (fidelity-gated 58/58 live peaks at 0.000s). MANDATORY substrate for any offline precheck reading a CLIP curve.',
'- ' + LOGS + '/analysis_bootstrap_ab.py: the adjudicator; supports --paired-json <field> (v143 built cgr_added/pre_cgr_ndet drop modes; a --paired-drop-flag mode existed and was reverted — re-add if needed).',
'- v143/v144 idea memos: ' + LOGS + '/ideas/loop1_v143_cgr-coverage-guided-recall-rescue_20260730.md, ' + LOGS + '/ideas/loop2_v144_pacr-peak-anchored-candidate-relocation_20260731.md (read BOTH — they contain the calibration + empty-video-marker lessons and a working 15/37-recoverable analytic ceiling for count-neutral relocation).',
'- DO NOT build offline prechecks on features/test/*.npy for CURVE signals (FM15: opposite verdicts vs live). For SUPERVISED MIL/actionness scores, features/test IS the correct substrate (that is what the model consumes) — the FM15 caveat is specific to the live category CURVE.',
'',
'KNOWN FAILURE MODES (FM1-FM17):',
'FM1 curve-shape modification regresses (10+ fails). FM2 additive/detection-count changes make FPs under greedy single-match GLOBAL AP (v143 CONFIRMED). FM3 VLM context-bleed hurts verdicts. FM4 self-consistency on inclusion. FM5 mechanism never activated — REQUIRE flag + unique INFO marker + grep gate. FM6 offline/sim gains do not transfer (v143 TP-only oracle over-predicted). FM7 video-level priors cannot fix intra-video mislocation. FM8 one controlled change per loop. FM9 MPS tensors need .detach().cpu().numpy(). FM10 wall-clock is API-drift-dominated. FM11 prefer mechanisms validatable by ONE full eval. FM12 cut calls only by DROPPING provably-null calls.',
'FM13 structural-cap / magnitude-dilution — bound the ceiling analytically first. FM14 verdict-visibility via global-max ranking. FM15 SIGNAL-SUBSTRATE INFIDELITY for live CURVE signals (use analysis_v140_livecurve_lib.py). FM16 MEASUREMENT BELOW THE NOISE FLOOR (65-harness CI ~+/-0.025; use bootstrap CIs, prefer paired). ',
'FM17 (v144, NEW) EMPTY-VIDEO MARKER DEADLOCK: a per-video activation/summary marker that only fires when ndet>0 makes markers < succeeded on normal/empty videos, so a strict [GATE] refuses to write the report AND relaunch cannot recover (JSONs already exist). Every per-video marker MUST fire once per SUCCEEDED video including the empty/0-detection path; smoke must include at least one known-empty video.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop. ARCHITECTURE-LEVEL. No blind number tuning. Expected effect >= +0.03 with explicit AP-checkpoint arithmetic.',
'- Any localizer/relocation/rescue gate MUST be VERDICT-INDEPENDENT and PRE-CALIBRATABLE (MIL/actionness/curve), NEVER raw scan_segment absolute conf. Prefer COUNT-NEUTRAL. If additive, bound the FP-demotion cost first (FM2).',
'- Pre-register in-vivo SENTINELS (accept-rate<=some bound, no uniform conf-pin, on-GT>=off-GT) and RUN THEM IN THE SMOKE PASS before any full eval.',
'- All new params via config.py with a switch restoring prior behavior EXACTLY (flag-off byte-identical to the committed BUE-on pipeline); CLI in BOTH run.py and batch_evaluate.py; unique INFO marker per fire that ALSO fires on the empty-video path (FM17). Do NOT disturb BUE (default-on) or CEX (default-off).',
'- Eval harness FIXED: --max_per_category 5 (65 videos), --concurrency ' + CONCURRENCY + ' (NOT 3 — it thrashes 16 GB swap; if a run stalls >30 min with no log growth, kill and relaunch at ' + CONCURRENCY + ', the runner skips existing JSONs).',
'- StructuredOutput EXACTLY ONCE at the end. ANTI-STALL: no Bash call > ~150s; Read with offset/limit; short progress line between tool calls (runtime kills agents silent >180s).',
].join('\n')

const history = [
  { loop: 3, version: 'v140', title: 'Noise-floor closure (no code)', full_map: -1, kept: false, lessons: '65-harness bootstrap 90% CI ~+/-0.025; only >= +0.03 effects admissible; verdicts must be bootstrap CIs; prefer paired. FM16.' },
  { loop: 5, version: 'v142', title: 'BUE Boundary-Union Extension — SHIPPED default-ON', full_map: 0.15114, kept: true, lessons: '290-video PAIRED verdict +0.02164 (0.12949->0.15114), CI90 [-0.0010,+0.0502], P(Δ>0)=0.938. First mechanism to clear effect-size+confidence gates on a properly-powered harness. NOW COMMITTED (bue_enabled=True). Grows existing windows outward; cannot reach the 50/99 zero-overlap misses.' },
  { loop: 6, version: 'v143', title: 'CGR additive VLM recall rescue', full_map: 0.13937, kept: false, lessons: 'REVERT floor breached; cross-run Δ-0.018 P=0.174. Additive dets = high-ranked FPs that DEMOTED precision under global AP (FM2). Root cause: uncalibrated scan_segment abs conf pins ~0.9 on isolated clips; on-GT sentinels skipped before eval. DO NOT re-propose additive VLM rescue on raw abs conf.' },
  { loop: 7, version: 'v144', title: 'PACR count-neutral peak-anchored relocation', full_map: -1, kept: false, lessons: 'REVERT no verdict; crashed at FM5 gate — empty video skipped the summary marker (markers<succeeded), report never written, relaunch could not recover (FM17). Mechanism NEVER evaluated. Right shape (count-neutral relocation, analytic ceiling 15/37 recoverable) but needs: bug-free empty-video marker, a verdict-independent/pre-calibratable relocation gate (NOT scan_segment abs conf), and the degeneracy/on-GT sentinel actually run in smoke.' },
]
let bestFullMap = BASE_MAP
let bestVersion = 'v137+BUE'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": full mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED baseline: ' + bestVersion + ' (BUE-on; fresh 65-video reference mAP set in Phase 0; floor ' + FLOOR.toFixed(5) + ').\nSession history:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'What it does, mechanically, in 3-6 sentences' },
    target_family: { type: 'string', description: 'candidate-generation | recall-ceiling | score-quality-supervised | other (NOT boundary-growth=BUE; NOT scan_segment-abs-conf-gate)' },
    expected_map_gain: { type: 'number', description: 'Honest expected absolute mAP gain; must be >= 0.03' },
    effect_size_arithmetic: { type: 'string', description: 'Which of the ~4 AP checkpoints moves, how many detections must change, and why that yields >= +0.03; if additive, the FP-demotion cost bound (FM2)' },
    localizer_calibration: { type: 'string', description: 'What verdict-independent, pre-calibratable signal drives it (MIL/actionness/curve) and how it is calibrated. If it touches scan_segment absolute conf -> auto-reject.' },
    count_neutral: { type: 'boolean', description: 'true if detection COUNT is frozen (preferred). If false, justify the FP-demotion bound.' },
    inv_ivo_sentinels: { type: 'string', description: 'The pre-registered smoke sentinels (accept-rate bound, no uniform conf-pin, on-GT>=off-GT) that WILL run before the full eval.' },
    implementation_plan: { type: 'string', description: 'Exact files/functions/lines, config field, CLI flags, marker string (incl. the EMPTY-video path, FM17)' },
    paired_measurable: { type: 'boolean', description: 'true if it can log a per-fire pre-mechanism field / conf=<old>-><new> so paired mode applies (strongly preferred)' },
    fm_risk_table: { type: 'string', description: 'FM1-FM17 each: immune / at-risk + why' },
    banned_family_check: { type: 'string', description: 'Explicit statement that it is none of the 8 banned families, and why' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string', description: 'The two strongest arguments against it' },
  },
  required: ['title', 'mechanism', 'target_family', 'expected_map_gain', 'effect_size_arithmetic', 'localizer_calibration', 'count_neutral', 'inv_ivo_sentinels', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'banned_family_check', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}

const verdictSchema = {
  type: 'object',
  properties: {
    approve: { type: 'boolean' },
    feedback: { type: 'string' },
    mandated_changes: { type: 'string' },
    effect_size_ok: { type: 'boolean' },
    banned_family_ok: { type: 'boolean' },
    calibration_ok: { type: 'boolean', description: 'Is the localizer verdict-independent+pre-calibratable (NOT scan_segment abs conf) and are the in-vivo sentinels pre-registered?' },
  },
  required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok', 'calibration_ok'],
  additionalProperties: false,
}

const implSchema = {
  type: 'object',
  properties: {
    implemented: { type: 'boolean' },
    activation_marker: { type: 'string', description: 'Unique INFO substring emitted per fire AND once per succeeded video incl. empty path (FM17)' },
    eval_flags: { type: 'string', description: 'Exact CLI flags to activate the mechanism (on top of the committed BUE-on pipeline)' },
    paired_marker: { type: 'string', description: 'Marker/field for paired mode if applicable, else empty' },
    files_changed: { type: 'string' },
    validation: { type: 'string', description: 'AST/import/--help/default checks + the FM17 empty-video marker check, actually run, with results' },
    notes: { type: 'string' },
  },
  required: ['implemented', 'activation_marker', 'eval_flags', 'files_changed', 'validation'],
  additionalProperties: false,
}

const launchSchema = { type: 'object', properties: { launched: { type: 'boolean' }, note: { type: 'string' } }, required: ['launched', 'note'], additionalProperties: false }
const watchSchema = { type: 'object', properties: { done: { type: 'boolean' }, crashed: { type: 'boolean' }, mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' }, latency: { type: 'number' }, marker_count: { type: 'number' }, note: { type: 'string' } }, required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false }
const bootSchema = { type: 'object', properties: { ran: { type: 'boolean' }, mode: { type: 'string' }, point_delta: { type: 'number' }, ci90_low: { type: 'number' }, ci90_high: { type: 'number' }, p_gt0: { type: 'number' }, new_map: { type: 'number' }, note: { type: 'string' } }, required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false }
const resultSchema = { type: 'object', properties: { version: { type: 'string' }, implemented: { type: 'boolean' }, smoke_map: { type: 'number' }, full_map: { type: 'number' }, kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' }, report_path: { type: 'string' } }, required: ['version', 'implemented', 'full_map', 'kept', 'summary', 'lessons'], additionalProperties: false }

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

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags, ph) {
  return [
'You are Agent D-launcher. ONLY job: ensure the ' + tag + ' eval for ' + V + ' is running WITH THE REQUIRED FLAGS, then return IMMEDIATELY. Do NOT wait for completion. SHORT replies; every Bash call < 150s. StructuredOutput EXACTLY ONCE.',
'',
'Paths: output_dir=' + ROOT + '/results/batch_eval_' + V + '_' + tag + ' ; log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '.log ; report=<output_dir>/evaluation_report.json',
'REQUIRED EXTRA FLAGS (must appear verbatim): ' + (evalFlags || '(none)'),
'',
'STEPS:',
'1. ps aux | grep -i batch_evaluate | grep -v grep',
'2. If an eval for batch_eval_' + V + '_' + tag + ' is already running -> ps -o command= -p <pid>; if the required flags are MISSING kill it and relaunch; if present return launched=true note="adopted".',
'3. If the report exists AND the log contains the required flags -> launched=true note="exists".',
'4. If a DIFFERENT version eval is running -> Bash "sleep 140" and re-check, up to 3 times; still busy -> launched=false note="blocked".',
'5. Else record the command in <output_dir>/COMMAND.txt and launch EXACTLY (extra flags before the redirect):',
"bash -lc 'mkdir -p " + ROOT + "/results/batch_eval_" + V + "_" + tag + "; source " + CODE + "/.env 2>/dev/null; export PYTHONUNBUFFERED=1; cd " + ROOT + "; nohup " + PY + " -u -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir " + ROOT + "/results/batch_eval_" + V + "_" + tag + " --max_per_category " + maxPerCat + " --concurrency " + CONCURRENCY + " " + (evalFlags || '') + " > " + ROOT + "/results/batch_eval_" + V + "_" + tag + ".log 2>&1 & echo PID $!'",
'6. Bash "sleep 30"; tail -5 the log to confirm no immediate Traceback. Return launched, note.',
  ].join('\n')
}

function evalWatchPrompt(V, tag, marker, nTotal) {
  const rep = ROOT + '/results/batch_eval_' + V + '_' + tag + '/evaluation_report.json'
  const lg = ROOT + '/results/batch_eval_' + V + '_' + tag + '.log'
  const N = nTotal || 65
  return [
'You are Agent D-watcher for the ' + tag + ' eval of ' + V + ' (' + N + ' videos). SHORT replies; NO Bash call > 150s. Do NOT edit code. StructuredOutput EXACTLY ONCE.',
'',
'Paths: report=' + rep + ' ; log=' + lg + ' ; inner log=' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/run.log',
'Marker (FM5): "' + (marker || '(none)') + '"',
'',
'STEPS:',
'1. If the report exists, extract metrics and markers, return done=true crashed=false:',
'   ' + PY + ' -c "import json;d=json.load(open(\'' + rep + '\'));t=d[\'evaluation\'][\'temporal_map\'];ti=d[\'timing\'];print(t[\'mAP\'],t.get(\'AP@0.3\'),t.get(\'AP@0.5\'),t.get(\'AP@0.7\'),ti[\'avg_per_video_seconds\'])"',
'   markers: grep -cF "' + (marker || '') + '" ' + lg + ' (also try the inner run.log; trust the LARGER count).',
'2. Else check liveness: pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". If alive -> Bash "sleep 140", re-check, up to 2 sleeps per invocation, then return done=false crashed=false note="running: <n>/' + N + ' JSONs" (ls ' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/*.json | grep -v evaluation_report | wc -l).',
'   STALL CHECK: if the inner run.log has not grown in >30 min while alive, the 16 GB machine likely exhausted swap — kill the process and relaunch the SAME nohup command at --concurrency ' + CONCURRENCY + ' (runner skips existing JSONs). note="restarted after stall".',
'3. If dead with no report: tail -30 the log. TRANSIENT (API timeout/network) -> relaunch, note="relaunched: <cause>". CODE BUG (Traceback in project code) -> done=false crashed=true note=<summary>. FM17 (markers<succeeded because an empty video skipped its marker) -> done=false crashed=true note="FM17 empty-video marker deadlock".',
'4. If all ' + N + ' JSONs exist but no report, metrics can be computed offline — note="' + N + '/' + N + ' artifacts, report missing".',
'Return: done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
  ].join('\n')
}

async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal, ph) {
  const phase_ = ph || 'Implement'
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags, phase_), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: phase_ }))
  if (!launch || !launch.launched) {
    return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'agent null after retries') }
  }
  const cap = maxWatch || 80
  for (let w = 0; w < cap; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, marker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: phase_ }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap reached' }
}

function bootstrapPrompt(V, pairedMarker) {
  const dir = ROOT + '/results/batch_eval_' + V + '_full'
  const paired = pairedMarker && pairedMarker.length > 0
  return [
'You are Agent D-adjudicator for ' + V + '. Run the MANDATED verdict statistic (video-matched bootstrap CI). Do NOT edit pipeline code. Bash < 150s per call. StructuredOutput EXACTLY ONCE.',
'',
'Run this (self-reports PASS/FAIL of the pre-registered gate):',
paired
  ? ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-log ' + dir + '/run.log --marker "' + pairedMarker + '" --min-delta ' + MIN_DELTA_PAIRED + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5))
  : ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta ' + MIN_DELTA_CROSS + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5)),
'If a per-JSON pre-mechanism field was written instead, use --paired-json <FIELD>. If the run exceeds ~150s, re-invoke with --resamples 300 and say so.',
'The tool reads per-video artifacts directly if the report is missing.',
'',
'Report: ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, note (include the tool\'s VERDICT line verbatim).',
  ].join('\n')
}

function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. GOAL: raise full-eval temporal mAP on the FIXED 65-video harness with a mechanism whose HONEST expected effect is >= +0.03. ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. MANDATORY READING (Read with offset/limit): BOTH v143 + v144 memos in ' + LOGS + '/ideas/ (the calibration + empty-video-marker + count-neutral-ceiling lessons), the v142 BUE header at the top of ' + CODE + '/pipeline.py (so you do NOT touch it), ' + LOGS + '/ideas/loop3_v140_faithful-substrate-and-noise-floor_20260728.md.',
'2. PROFILE the 50/99 zero-overlap detections with the CACHED runs + ' + LOGS + '/analysis_v140_livecurve_lib.py (no API cost) AND the supervised MIL scores (features/test + model_mil*.pth): is a verdict-independent, pre-calibratable signal (MIL snippet score / actionness / curve peak) able to point at the RIGHT event where scan_segment abs-conf failed? Quantify on-GT vs off-GT separation of your chosen signal BEFORE proposing. This diagnosis is the deliverable that justifies your proposal.',
'3. SEARCH: >=4 real WebSearch queries on weakly-supervised TAL / VAD candidate & actionness generation, snippet score quality, MIL localization 2024-2026. If WebSearch is unavailable, say so and rely on code+data measurement.',
'4. Derive ONE architecture-level mechanism, expected >= +0.03, that is NOT boundary-growth (=BUE) and does NOT gate on raw scan_segment absolute conf (refuted v143/v144). PREFER count-neutral + paired-measurable. It MUST carry: effect-size arithmetic (which checkpoint moves; if additive, the FP-demotion bound), the verdict-independent localizer + its calibration, the pre-registered in-vivo sentinels, a config flag + CLI wiring + unique INFO marker that fires on the EMPTY-video path (FM17), and an FM1-FM17 risk table.',
'5. Return via StructuredOutput (EXACTLY ONCE).',
'',
'!!! ANTI-STALL: no Bash > ~60s, no sleep, Read with offset/limit, short progress line between tool calls, finish in <18 tool calls.',
  ].join('\n')
}

function d1Prompt(V, loop, proposal, verdict) {
  return [
'You are Agent D1 (implementer) for ' + V + ' (loop ' + loop + '). Implement the APPROVED proposal exactly. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'APPROVED PROPOSAL:\n' + JSON.stringify(proposal, null, 1),
verdict && verdict.mandated_changes ? '\nC-MANDATED CHANGES (must be honored):\n' + verdict.mandated_changes : '',
'',
'STEPS:',
'1. mkdir -p ' + ROOT + '/.qoder/backups/' + V + ' and copy every file you will edit into it BEFORE editing.',
'2. Implement the mechanism. New params in config.py (dataclass default = OFF/prior behavior) AND config.from_args with explicit threading; CLI flags in BOTH run.py and batch_evaluate.py; bump the pipeline.py header to ' + V + '. Do NOT disturb the committed BUE (default-ON) or CEX (default-OFF) code paths — both must stay byte-identical when their flags/defaults are unchanged.',
'3. FM5 + FM17: emit a UNIQUE INFO marker per fire, and a per-video summary marker that fires ONCE PER SUCCEEDED VIDEO INCLUDING THE EMPTY/0-DETECTION PATH (v144 deadlocked because an empty video skipped it). If post-hoc, the marker carries the pre-mechanism field / conf=<old>-><new> for paired adjudication.',
'4. VALIDATE and report actual outputs: ' + PY + ' -m py_compile on each edited file; ' + PY + ' -c "import sys;sys.path.insert(0,\'' + CODE + '\');from pipeline import MultiAnomalyVAD;print(1)"; flags in --help; ScoringConfig() defaults unchanged for BUE/CEX; from_args(Namespace()).scoring == ScoringConfig(); AND a direct check that the new marker fires on a synthetic empty-detection video path (FM17).',
'5. Return implemented, activation_marker, eval_flags (exact), paired_marker (or empty), files_changed, validation, notes.',
'',
'ANTI-STALL: no Bash > ~150s; Read with offset/limit; do NOT run an eval here.',
  ].join('\n')
}

function d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5) {
  return [
'You are Agent D4 (recorder) for ' + V + '. Record this loop honestly and completely. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'IMPLEMENTATION:\n' + JSON.stringify(impl, null, 1), '',
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL: ' + JSON.stringify(full) + '\nBOOTSTRAP: ' + JSON.stringify(boot),
'\nORCHESTRATOR DECISION: ' + decisionText + (fm5 ? '\nFM5/FM17 VIOLATION: the activation marker never appeared / deadlocked - the mechanism did not run correctly.' : ''),
'',
'TASKS:',
'1. If REVERT: restore every file from ' + ROOT + '/.qoder/backups/' + V + '/, then VERIFY with diff -q per file and confirm 0 remaining symbols of the mechanism (grep), and that BUE stays default-ON and CEX default-OFF. If KEEP, leave the code and flip the config default to ON if the proposal says so.',
'2. Archive artifacts into ' + LOGS + '/results/: ' + V + '_full_eval.json, ' + V + '_full_run.log, ' + V + '_smoke_eval.json, bootstrap_ab.json (whichever exist).',
'3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (date: date +%Y%m%d): mechanism, effect-size arithmetic vs what happened, the BOOTSTRAP CI as the headline verdict, FM1-FM17 table, honest weaknesses, what the next loop should NOT repeat.',
'4. APPEND one row to ' + LOGS + '/session_v97_ledger.md in the existing pipe-table format (loop | version | title+verdict+evidence | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept). Use -1 for absent. Do NOT rewrite existing rows.',
'5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ PHASE 0: FRESH BUE-ON 65-VIDEO REFERENCE ============================
log('v9 continuation starting at v' + START_VERSION + '. BUE is COMMITTED default-ON (290 verdict +0.02164, P=0.938). ' + MAX_LOOPS + ' loops on the FIXED 65-video harness, concurrency ' + CONCURRENCY + '.')
phase('Baseline')
log('Phase 0: establishing the fresh BUE-ON committed 65-video reference (also the cross-run REF).')
const ref = await runEval(REF_V, REF_TAG, 5, REF_FLAGS, '[BUE]', 80, 65, 'Baseline')
if (ref && ref.done && !ref.crashed && ref.mAP > 0) {
  REF_MAP = ref.mAP
  REF_DIR = ROOT + '/results/batch_eval_' + REF_V + '_' + REF_TAG
  FLOOR = Math.max(0.15132, REF_MAP - MAP_TOL)
  bestFullMap = REF_MAP
  log('Phase 0 DONE: BUE-on committed 65-video baseline mAP=' + REF_MAP.toFixed(5) + ' (AP@0.3=' + ref.ap30 + ' AP@0.5=' + ref.ap50 + ' AP@0.7=' + ref.ap70 + '). REF=' + REF_DIR + ', floor=' + FLOOR.toFixed(5) + '.')
} else {
  log('Phase 0 reference eval FAILED (' + (ref ? ref.note : 'null') + '). Falling back to the historical committed baseline ' + BASE_MAP + ' and REF at ' + REF_DIR + ' if present; floor ' + FLOOR.toFixed(5) + '. Loops will PREFER paired adjudication (which does not need a cross-run REF).')
}

// ============================ MAIN LOOP ============================
for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' ===')

  phase('Research')
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', 'YOUR ANGLE: CANDIDATE GENERATION / the recall ceiling — put detections on the RIGHT events for the 50/99 zero-overlap misses, using a VERDICT-INDEPENDENT, PRE-CALIBRATABLE localizer (MIL snippet scores / model_mil_v6.pth / actionness / the faithful category curve), NOT raw scan_segment abs conf (refuted v143/v144). Prefer COUNT-NEUTRAL relocation (the v144 PACR shape, but bug-free and with a calibrated gate). You own the FM2 and FM17 risks and must argue them.'),
      { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: snippet SCORE QUALITY from the untapped SUPERVISED assets (features/train 1610 videos, model_mil_v6.pth, prompt_weights.json) to improve the score detections are selected/ranked from — a NON-fusion use of the MIL model (prior fusion failed FM1) or a calibrated actionness head. Do NOT touch BUE boundaries or the half-max curve. You own the FM1 risk and must argue why this differs from the 10+ curve-shape failures.'),
      { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Cross-critique both ideas and synthesize ONE proposal to implement this loop.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'RULES:',
'- Pick the ONE with the larger HONEST expected effect and cleaner measurability; do not merge two mechanisms (FM8).',
'- Kill any part whose expected effect is < +0.03 (unmeasurable, FM16).',
'- REJECT any boundary-growth part (BUE owns it) and any raw scan_segment-abs-conf gate (refuted v143/v144, banned family 8).',
'- Ensure the winner is count-neutral if possible, has a verdict-independent pre-calibratable localizer, pre-registered in-vivo sentinels, and an empty-video-safe marker (FM17).',
'Return the unified proposal via StructuredOutput (EXACTLY ONCE).',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the strict quality gate. Each full eval costs ~1-2h. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'MANDATORY CHECKLIST:',
'1. EFFECT SIZE (binding): expected gain >= +0.03 with HONEST AP-checkpoint arithmetic? If additive, is the FP-demotion cost bounded (v143 died here)? If ~+0.01, REJECT (FM16).',
'2. BANNED FAMILIES: any of the 8 (incl. #7 boundary-growth=BUE, #8 scan_segment-abs-conf gate)? If yes -> REJECT.',
'3. CALIBRATION (v143/v144 killer): is the localizer VERDICT-INDEPENDENT and PRE-CALIBRATABLE (MIL/actionness/curve), NOT raw scan_segment abs conf? Are the in-vivo sentinels (accept-rate bound, no uniform conf-pin, on-GT>=off-GT) pre-registered to RUN IN SMOKE before the full eval? If not -> REJECT. Set calibration_ok.',
'4. FM2: does it change detection COUNT? Prefer count-neutral. If additive, demand the FP-demotion analysis.',
'5. FM5+FM17: config flag (default OFF, flag-off byte-identical to the committed BUE-on pipeline) + CLI in both runners + unique marker that ALSO fires on the empty-video path? If missing -> REJECT.',
'6. FM1: does it change the fused/half-max curve or BUE? If yes -> demand a specific argument or REJECT.',
'7. FM15: any offline CURVE precheck must use analysis_v140_livecurve_lib.py (features/test is fine for MIL scores, NOT for the live curve).',
'8. FM8/FM11: exactly ONE controlled change, validatable by ONE full eval, leaving BUE default-ON and CEX default-OFF?',
'Set effect_size_ok, banned_family_ok, calibration_ok explicitly. REJECT with concrete feedback, or APPROVE (optionally with mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (round ' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. Reviewer C REJECTED your proposal. Revise it, or pivot to a mechanism that clears the >= +0.03 bar and avoids all 8 banned families and the v143/v144 calibration traps. ONE controlled change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'REJECTED PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
"C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) {
      proposal = Object.assign({}, proposal, { title: proposal.title + ' (C-mandated revision)', implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED CHANGES:\n' + verdict.mandated_changes })
      log('Loop ' + loop + ': proceeding with C-mandated revision')
    } else {
      log('Loop ' + loop + ': C rejected 3x without mandate - no eval spent (FM11 economy)')
      history.push({ loop: loop + 7, version: V, title: proposal.title + ' [REJECTED at gate]', full_map: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
      continue
    }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker) {
    log('Loop ' + loop + ': implementation failed or no marker (FM5) - skipping eval')
    history.push({ loop: loop + 7, version: V, title: proposal.title + ' [IMPL FAILED]', full_map: -1, kept: false, lessons: impl ? ('no marker / failed: ' + (impl.notes || '').slice(0, 200)) : 'agent null after retries' })
    continue
  }

  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker, 30, 13, 'Implement')
  log('Loop ' + loop + ': smoke done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ' markers=' + smoke.marker_count)

  let full = { done: false, crashed: false, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'skipped (smoke gate)' }
  let boot = null
  let decisionText = 'REVERT'
  let keep = false
  let fm5 = false

  if (smoke.done && !smoke.crashed && smoke.mAP >= SMOKE_GATE) {
    if (smoke.marker_count <= 0) {
      fm5 = true
      log('Loop ' + loop + ': FM5 - marker "' + impl.activation_marker + '" absent from smoke; not spending the full eval')
    } else {
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker, 80, 65, 'Implement')
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' markers=' + full.marker_count)
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from the full log') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, impl.paired_marker || ''), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        const paired = !!(impl.paired_marker && impl.paired_marker.length > 0)
        const bar = paired ? MIN_DELTA_PAIRED : MIN_DELTA_CROSS
        if (!boot || !boot.ran) {
          decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'agent null') + ')'
        } else {
          const okD = boot.point_delta >= bar
          const okP = boot.p_gt0 >= MIN_P
          const okF = boot.new_map >= FLOOR
          if (okD && okP && okF) {
            keep = true
            decisionText = 'KEEP (' + (paired ? 'paired' : 'cross-run') + ' delta ' + boot.point_delta.toFixed(5) + ' >= ' + bar + ', P(d>0)=' + boot.p_gt0.toFixed(3) + ' >= ' + MIN_P + ', mAP ' + boot.new_map.toFixed(5) + ' >= floor ' + FLOOR.toFixed(5) + '. CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + ']. New baseline.)'
          } else if (!okF) {
            decisionText = 'REVERT (ACCURACY FLOOR breached: mAP ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          } else if (!okD) {
            decisionText = 'REVERT (effect too small: delta ' + boot.point_delta.toFixed(5) + ' < ' + bar + '; P(d>0)=' + boot.p_gt0.toFixed(3) + '. FM16.)'
          } else {
            decisionText = 'REVERT (delta ' + boot.point_delta.toFixed(5) + ' but P(d>0)=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + '; CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + '] straddles zero.)'
          }
        }
      }
    }
  } else {
    log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ')')
  }
  if (fm5) decisionText = 'REVERT (FM5/FM17: the activation marker never appeared / deadlocked - the mechanism did not run correctly)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder failed)', lessons: fm5 ? 'FM5/FM17' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [bootstrap d=' + boot.point_delta.toFixed(5) + ' P(d>0)=' + boot.p_gt0.toFixed(3) + ']') : ''
  history.push({ loop: loop + 7, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) {
    bestFullMap = result.full_map; bestVersion = V
    REF_DIR = ROOT + '/results/batch_eval_' + V + '_full'   // kept mechanism becomes the new cross-run REF
    FLOOR = Math.max(0.15132, result.full_map - MAP_TOL)
    log('Loop ' + loop + ': KEPT ' + V + ' - new baseline mAP ' + result.full_map)
  } else {
    log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 160))
  }
}

// ============================ END-OF-SESSION SUMMARY ============================
phase('Review')
await retryAgent(() => agent([
'You are Agent C writing the END-OF-SESSION SUMMARY for the v9 continuation (v145-v146 + the committed BUE ship). Plain text reply; no schema.',
'', CONTEXT, '', 'FULL SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. Read ' + LOGS + '/session_v97_ledger.md (tail) and the new v145/v146 idea files.',
'2. Get the date: date +%Y%m%d.',
'3. Write ' + LOGS + '/summary_v9_v145-v146_' + '<YYYYMMDD>.md: executive summary (did v145 or v146 clear >= +0.03 with P>=' + MIN_P + '? did the BUE-on floor hold?), the fresh BUE-ON 65-video baseline mAP (REF ' + REF_DIR + '), a per-version table (mechanism | family | point delta | CI90 | P(d>0) | kept), an explicit verdict on whether the candidate-generation/recall family has >= +0.03 REACHABLE headroom or is closing, the updated FM1-FM17 taxonomy, and concrete next directions respecting the noise floor.',
'4. APPEND to ' + LOGS + '/session_v97_ledger.md one row for the committed BUE@290 ship if not already present (version="BUE@290-SHIP", full_mAP=0.15114, note the +0.02164 paired verdict + P=0.938 + default-ON). Do NOT rewrite existing rows.',
'5. Reply with a 10-line plain-text recap.',
].join('\n'), { label: 'C-final-summary', phase: 'Review' }))

log('v9 session complete. Committed baseline ' + bestVersion + ' mAP ' + bestFullMap.toFixed(5) + ' (BUE-on). ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, refMap: REF_MAP, floor: FLOOR }
