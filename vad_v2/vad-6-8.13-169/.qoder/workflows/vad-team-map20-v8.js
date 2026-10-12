export const meta = {
  name: 'vad-team-map20-v8',
  description: 'VAD agent team v8 (ACCURACY continuation, owner-directed). CONTINUES the v7 arc after v141 (CEX, default-OFF, superseded, never eval\'d) and v142 (BUE Boundary-Union Extension, default-OFF, the SESSION\'S STRONGEST candidate: offline UNION +0.0378, 9/9 out-of-sample caches positive mean +0.0370, smoke 65 mAP 0.2318 vs v137 baseline 0.15632 — but its 65-video FULL eval was INTERRUPTED at ~6/65 and never adjudicated). v8 does exactly what the owner asked on 2026-07-30: (1) run TWO more iteration versions v143->v144 on the SAME FIXED 65-video harness (--max_per_category 5), each attacking the >= +0.03 recall/localization or supervised-score-quality headroom (NEVER ranking refinement, NEVER a banned family), verdict = video-matched BOOTSTRAP CI; then (2) run ONE decisive FULL 290-video eval (--max_per_category 999) WITH --bue_extension enabled, at concurrency 3, and PAIRED-adjudicate BUE via pre_bue_boundaries. The 290 split is the FIRST harness whose CI (~+/-0.012) can actually measure BUE\'s +0.037 (the 65 harness resolves only +/-0.025, per v140). CONCURRENCY IS 3 by explicit owner override (it hung once on 2026-07-28 by exhausting swap on this 16 GB machine; the watchers auto-detect a >30 min stall and relaunch, and the runner skips videos whose result JSON exists, so resumption is cheap).',
  phases: [
    { title: 'Research', detail: 'A and B independently research >=+0.03-effect recall/localization or score-quality mechanisms (NOT boundary outward-growth — BUE owns that), each proposes one' },
    { title: 'Debate', detail: 'A and B cross-critique on effect size and measurability, then synthesize one joint proposal' },
    { title: 'Review', detail: 'C gates on FM1-FM16 + the >=+0.03 effect-size rule + banned families; up to 3 cycles' },
    { title: 'Implement', detail: 'D implements (flag + marker), smoke -> full 65-video eval -> bootstrap-CI adjudication, keep/revert' },
    { title: 'Full-290', detail: '290-video eval WITH --bue_extension at concurrency 3 -> paired bootstrap on pre_bue_boundaries -> decisive BUE verdict -> keep/revert BUE, session summary' },
  ],
}

const ROOT = '/Users/zzl/PycharmProjects/vad'
const CODE = ROOT + '/video_anomaly_detection'
const LOGS = CODE + '/iteration_logs'
const PY = 'python3.14'                  // plain python3 lacks numpy/torch/open_clip in this env
const MAX_LOOPS = 2                       // owner: iterate TWO versions (v143, v144)
const START_VERSION = 143
const SMOKE_GATE = 0.06
const BASE_MAP = 0.15632                 // committed v114/v124/v137 baseline
const MAP_TOL = 0.005
const FLOOR = BASE_MAP - MAP_TOL         // 0.15132 hard accuracy floor
const REF_DIR = ROOT + '/results/batch_eval_v124_full'   // fixed reference RUN for cross-run A/B
const REF_MAP = 0.17076                  // that run's full mAP
const MIN_DELTA_CROSS = 0.02             // cross-run KEEP bar (mechanism changes detections)
const MIN_DELTA_PAIRED = 0.01            // paired KEEP bar (post-hoc mechanism; VLM noise removed)
const MIN_P = 0.90                       // bootstrap P(delta>0) required for the 65-harness loops
const CONCURRENCY = 3                    // OWNER OVERRIDE 2026-07-30 (was 2 after the 2026-07-28 swap hang)

// ---- FINAL 290-video paired BUE adjudication constants ----
const FULL290_TAG = 'bue'                                       // dir: results/batch_eval_full290_bue
const FULL290_V = 'full290'
const FULL290_MAXCAT = 999                                      // selects all 290 test videos
const FULL290_FLAGS = '--bue_extension --qscore_rerank'         // BUE ON + committed Q-score ON
const FULL290_MARKER = '[BUE]'
const BUE_PAIRED_FIELD = 'pre_bue_boundaries'                   // per-JSON pre-mechanism windows
const BUE_MIN_DELTA = 0.02                                      // v142 pre-registered rule
const BUE_MIN_P = 0.85                                          // v142 pre-registered rule (concentrated boundary op, wider CI)

const CONTEXT = [
'PROJECT: video anomaly detection + TEMPORAL LOCALIZATION on UCF-Crime (weak supervision; test requires temporal segment proposals).',
'Repo root: ' + ROOT + '. Main code: ' + CODE + ' (pipeline.py = main pipeline; config.py = all params; anomaly_expert.py, anomaly_discovery.py, frame_scorer.py, post_processor.py, temporal_mcts.py, mil_model.py, evaluate.py, batch_evaluate.py).',
'CRITICAL GIT RULE: the uncommitted working tree IS the latest code. NEVER run git checkout/reset/stash/clean or any destructive git command. Edit files directly; back up to ' + ROOT + '/.qoder/backups/<version>/ before editing.',
'INTERPRETER: use ' + PY + ' for everything (plain python3 lacks numpy/torch/open_clip).',
'',
'!!! OBJECTIVE: RAISE full-eval temporal mAP. The 65-video iteration harness is FIXED by owner instruction (--max_per_category 5); do NOT propose enlarging it for the iteration loops. !!!',
'Committed baseline mAP ' + BASE_MAP + ' (reference full run v124 = ' + REF_MAP + '). HARD FLOOR: full mAP >= ' + FLOOR.toFixed(5) + '.',
'',
'!!! CURRENT STATE INHERITED BY v8 (read carefully) !!!',
'- Pipeline header is at v142. TWO mechanisms exist in the tree, BOTH default-OFF (flag-off is byte-identical to the committed v137 pipeline):',
'  * v141 CEX (Continuation-Evidence Extension, --cex_extend): per-side VLM-confirmed outward growth. Implemented but SUPERSEDED by BUE inside the same loop; NEVER given a full eval; not adjudicated. Leave default-OFF.',
'  * v142 BUE (Boundary-Union Extension, --bue_extension): the STRONGEST candidate of the entire session. Grows each FINAL detection outward on the read-only category CLIP curve and emits the UNION of the old window and the AutoLoc-style outer-inner-contrast (OIC) argmax window (UNION-ONLY; measured 3-cache ablation: closure-gated REPLACE = -0.0065, UNION = +0.0378). Offline out-of-sample: 9/9 held-out caches positive, mean +0.0370; smoke-65 mAP 0.2318 (vs 0.15632 committed). PAIRED-MEASURABLE: with the flag on, each result JSON carries index-aligned pre_bue_boundaries, so a within-run paired bootstrap removes ALL VLM run-to-run noise. Its 65-video FULL eval was INTERRUPTED at ~6/65 and never adjudicated. Default stays OFF until the v8 FINAL PHASE adjudicates it on 290 videos.',
'- BUE ALREADY OWNS boundary outward-growth. The v143/v144 loops must NOT re-propose boundary UNION/outward-extension, CEX-style per-side growth, or OIC relocation — that is BUE\'s territory and is pending its own verdict. Find a DIFFERENT >= +0.03 lever.',
'',
'!!! THE NOISE FLOOR - THE CENTRAL FACT OF THIS ARC (v140, measured) !!!',
'A video-level bootstrap (resampling the 65 videos, the true sampling unit) gives a 90% CI half-width of ~+/-0.025 for a mAP delta. Consequences, all measured:',
'  - The +0.005 promotion gate used since v114 sits FIVE TIMES BELOW the measurement resolution. Any lever whose true effect is ~+0.01 is UNFALSIFIABLE on the 65-video harness.',
'  - "N cached runs all agree" is nearly worthless: the cached full runs share ~81% of their detection windows (mean pairwise Jaccard 0.813), so a deterministic rule agrees across them almost automatically.',
'  - The COMMITTED Q-score win (v111/v114, +0.00619) is itself inside the noise band and must be treated as unfalsified, not demonstrated.',
'  - Root cause is structural: n_gt=66, 11-point interpolated AP, max recall 0.379@IoU0.3 => mAP is decided by precision at only FOUR hit checkpoints (1st/7th/14th/20th hit), so ONE detection crossing ONE checkpoint moves mAP by 0.005-0.010.',
'  - The FULL 290-video test split gives CI ~+/-0.012 and restores the ~51.7% Normal population; it is the FIRST harness that can measure a +0.03 effect. This is exactly why the v8 FINAL PHASE adjudicates BUE there.',
'THEREFORE: v8 admits ONLY proposals whose HONEST expected effect is >= +0.03. C MUST REJECT anything smaller as unmeasurable, however elegant. Do not argue with this rule; it is a property of the harness, not a preference.',
'',
'!!! BANNED FAMILIES (refuted or unmeasurable - proposing one is an automatic C REJECT) !!!',
'1. CURVE-DERIVED CONFIDENCE RE-RANKING in any form (v111 Q-score, v112, v115-v123 prominence/SLM/FDR/FSL/MELC/BREF/hysteresis/BOR/SLM-CV, v138 GPC dense peak, v139 GPC pooled peak, v140 bins ladder). Effect sizes are all ~+/-0.01 = unmeasurable. Within-video re-ranking is additionally EXHAUSTED by measurement: the within-video oracle is only +0.006 and the committed Q-score already claims +0.0062.',
'2. VLM call MERGING / BATCHING / contact-sheets / verdict propagation (v97 FCCS, v126 TRLM, v129 BDS): all cratered mAP to ~0.12 - per-call independence is load-bearing.',
'3. max_scan / scan-tail / Source-2 padding pruning (v125, v130, v131): max_scan already binds real content; mean_dropped=0.00.',
'4. PER-VIDEO score normalization / re-leveling (measured on v124: z-score -0.0898, rank/winner-take-all -0.0931). Global AP depends on the cross-video LEVEL; destroying it is catastrophic.',
'5. DENSE BOTTOM-RANKED PROPOSAL APPEND / "DPA" (measured +0.031 across 6 caches but REJECTED as metric gaming: frame AUC is bit-identical, a signal-FREE random generator reproduces ~+0.024 of it, and it would break comparability with the v51-v142 ledger). Do NOT re-propose padding the prediction list with synthetic proposals.',
'6. Adaptive-fps or other local-compute trims (v128): invisible to the deciding metrics.',
'7. (v8) BOUNDARY OUTWARD-GROWTH / UNION-EXTENSION / OIC RELOCATION / per-side VLM-confirmed growth: OWNED by v142 BUE / v141 CEX and pending BUE\'s 290 verdict. Do not duplicate it in a v143/v144 loop.',
'',
'!!! WHERE THE >= +0.03 HEADROOM ACTUALLY IS (measured, v138-v142) !!!',
'- RECALL / LOCALIZATION CEILING is the binding constraint: max recall 0.379 @IoU0.3, 0.212 @0.5, 0.121 @0.7. **50 of 99 detections have ZERO overlap with GT** (median IoU = 0.000) - the pipeline frequently looks in the WRONG PLACE. Fixing WHERE it looks, or how candidates are GENERATED, is the family with >= +0.03 of physical headroom that BUE does NOT already occupy (BUE only grows EXISTING windows outward; it cannot rescue a detection whose GT is elsewhere in the video).',
'- The pure-re-rank ORACLE ceiling is 0.27273 (perfect ordering of the CURRENT detections), so even perfect ranking cannot reach 0.30; but raising recall raises the ceiling ITSELF.',
'- Detections are systematically SHORT relative to GT (hits: det/GT duration ratio ~1.31, zero-overlap misses ~0.42; mean GT duration ~30s). BUE addresses the SHORTNESS of hits; the ZERO-OVERLAP misses (wrong place entirely) are a CANDIDATE-GENERATION problem BUE cannot fix.',
'- Untapped supervised assets: features/train (1610 videos) + features/test (290) pre-extracted CLIP snippets; model_mil.pth (v1, in production at fusion weight 0.25); model_mil_v6.pth (better standalone, fusion integration FAILED at FM1); prompt_weights.json (v94 CoOp-lite, a PROVEN-SAFE upstream win). The deep-research report concludes the gap to SOTA is SCORE QUALITY, not the boundary algorithm.',
'',
'!!! MEASUREMENT PROTOCOL - MANDATORY, NON-NEGOTIABLE !!!',
'The verdict statistic is a VIDEO-MATCHED BOOTSTRAP CI, never a point delta. Tool (already written and self-tested):',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <new_full_dir> --ref ' + REF_DIR + '   [cross-run]',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <new_full_dir> --paired-log <run.log> --marker <MARKER>   [paired, conf/window from log]',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new <new_full_dir> --paired-json <FIELD>   [paired, from per-JSON pre-mechanism windows, e.g. ' + BUE_PAIRED_FIELD + ']',
'KEEP (65-harness loop) requires: point delta >= ' + MIN_DELTA_CROSS + ' (cross-run) or >= ' + MIN_DELTA_PAIRED + ' (paired) AND bootstrap P(delta>0) >= ' + MIN_P + ' AND full mAP >= ' + FLOOR.toFixed(5) + '.',
'If a mechanism CAN be expressed as a post-hoc transform that logs "conf=<old>-><new>" per fire, PREFER IT: paired mode removes VLM run-to-run noise (~+/-0.019) entirely and is the only design that ever produced a correct verdict in v138/v139.',
'',
'!!! OTHER REUSABLE ASSETS (use them; do not rebuild) !!!',
'- ' + LOGS + '/analysis_v140_livecurve_lib.py: reproduces the IN-VIVO verdict-independent category curve exactly (fidelity-gated: 58/58 recorded live peak times reproduced at 0.000s). MANDATORY substrate for any offline precheck that reads a CLIP curve. Curves cached in ' + LOGS + '/analysis_v140_livecurves/.',
'- ' + LOGS + '/analysis_v138_paired_invivo.py: paired in-vivo A/B from marker logs.',
'- ' + LOGS + '/analysis_v138_rankdiag_orch.py: oracle ceilings / checkpoint decomposition.',
'- ' + LOGS + '/analysis_v142_bue_verify.py, analysis_v142_C_bootstrap.py, analysis_v142_C_gate.py: the offline BUE battery (apply_bue on cached predictions + official bootstrap). Curve cache: analysis_v142_bue_curvecache.pkl.',
'- DO NOT build offline prechecks on features/test/*.npy for curve signals: that substrate produced +0.0191 for a mechanism that measured -0.00287 in vivo (FM15).',
'',
'KNOWN FAILURE MODES (FM1-FM16):',
'FM1 curve-shape modification regresses (10+ fails). FM2 detection-count/verdict changes make FPs under greedy single-match mAP. FM3 VLM self-consistency/context-bleed hurts perceptual verdicts. FM4 self-consistency on segment INCLUSION is as risky as FM1. FM5 mechanism never activated - REQUIRE flag + unique INFO marker + grep gate (bit 5 times). FM6 offline/simulation gains do not transfer. FM7 video-level priors cannot fix intra-video mislocation. FM8 one controlled change per loop. FM9 MPS tensors need .detach().cpu().numpy(). FM10 wall-clock is API-drift-dominated. FM11 prefer mechanisms validatable by ONE full eval. FM12 cut calls only by DROPPING whole provably-null calls, never merge/batch.',
'FM13 structural-cap / magnitude-dilution: a clean, neutral mechanism can still fail purely on MAGNITUDE - bound the ceiling analytically BEFORE spending an eval.',
'FM14 verdict-visibility via global-max ranking: dropping a call is visible in mAP when the surviving score is a MAX over verdicts and AP ranks globally.',
'FM15 (v138/v139) SIGNAL-SUBSTRATE INFIDELITY: an offline precheck must reproduce the in-vivo signal EXACTLY - same features, same temporal resolution. features/test snippets vs live per-frame CLIP encoding produced OPPOSITE verdicts. Remedy: analysis_v140_livecurve_lib.py, fidelity-gated.',
'FM16 (v140) MEASUREMENT BELOW THE NOISE FLOOR: the 65-video harness resolves ~+/-0.025 (90% CI). Effects below ~+0.03 cannot be established there; cross-cache agreement is not independence (81% shared detections). Verdicts must be bootstrap CIs, and gates must exceed the floor. The 290-video split (CI ~+/-0.012) is where a +0.03 effect becomes measurable.',
'',
'HARD CONSTRAINTS:',
'- ONE controlled change per loop. ARCHITECTURE-LEVEL. No blind number tuning.',
'- Expected effect >= +0.03 with an explicit arithmetic argument (which of the 4 AP checkpoints moves, and how many detections must change to move it).',
'- All new params via config.py with a switch restoring prior behavior EXACTLY (flag-off must be byte-identical); CLI-activatable in BOTH run.py and batch_evaluate.py; unique INFO marker per fire (FM5). Verify from_args(empty).scoring == ScoringConfig().',
'- Keep the v51 half-max boundary mechanism and Q-score default-ON unless the proposal IS a boundary redesign, in which case say so explicitly and accept FM1 scrutiny.',
'- Eval harness FIXED for the loops: --max_per_category 5 (65 videos), --concurrency ' + CONCURRENCY + '. Concurrency 3 is an OWNER OVERRIDE for this session; it exhausted swap and hung a run for 70 min on 2026-07-28 on this 16 GB machine, so if a run stalls >30 min with no log growth, kill it and relaunch - the runner SKIPS videos whose result JSON already exists, so resumption is cheap.',
'- StructuredOutput EXACTLY ONCE at the end. ANTI-STALL: no Bash call > ~150s; Read with offset/limit; emit a short progress line between tool calls (runtime kills agents silent >180s).',
].join('\n')

const history = [
  { loop: 1, version: 'v138', title: 'GPC global-peak-containment (dense argmax) post-hoc re-rank', full_map: 0.15372, kept: false, lessons: 'REVERT: paired in-vivo -0.00287 despite offline +0.0191 on 8/8 caches. FM15 born: offline substrate was features/test snippets, in vivo is per-frame CLIP.' },
  { loop: 2, version: 'v139', title: 'GPC pooled-peak fix (gpc_pool_bins=32)', full_map: 0.16450, kept: false, lessons: 'REVERT: paired -0.00149. Signal quality DID improve but AP checkpoints did not move. Resolution matching was necessary, not sufficient - the FEATURE SUBSTRATE differs too.' },
  { loop: 3, version: 'v140', title: 'Live-faithful curve substrate + harness noise-floor closure (no code, no eval)', full_map: -1, kept: false, lessons: 'Built + fidelity-gated the faithful substrate (58/58 live peaks at 0.000s). Proved the harness noise floor: 65-video bootstrap 90% CI ~+/-0.025 = 5x the +0.005 gate; 8 caches share 81% of detections. 290-video split gives ~+/-0.012. => only >= +0.03 effects admissible; verdicts must be bootstrap CIs. FM16 born.' },
  { loop: 4, version: 'v141', title: 'CEX Continuation-Evidence Extension (per-side VLM-confirmed outward growth, --cex_extend)', full_map: -1, kept: false, lessons: 'Implemented default-OFF but SUPERSEDED by BUE within the loop; no full eval, not adjudicated. Boundary outward-growth family is now covered by BUE.' },
  { loop: 5, version: 'v142', title: 'BUE Boundary-Union Extension (--bue_extension) — STRONGEST candidate, PENDING 290 verdict', full_map: -1, kept: false, lessons: 'UNION of old window + AutoLoc OIC-contrast argmax window on the read-only category curve; UNION-only (REPLACE=-0.0065, UNION=+0.0378). Offline 9/9 held-out caches positive mean +0.0370; smoke-65 mAP 0.2318. Full-65 eval INTERRUPTED at ~6/65. Paired-measurable (pre_bue_boundaries). 65-harness CANNOT measure +0.037 (CI +/-0.025); v8 FINAL PHASE adjudicates it on the 290-video split (CI +/-0.012). Default stays OFF until that verdict.' },
]
let bestFullMap = BASE_MAP
let bestVersion = 'v137'

function hist() {
  const lines = history.map(h => 'Loop ' + h.loop + ' ' + h.version + ' "' + h.title + '": full mAP=' + h.full_map + ' kept=' + h.kept + ' | ' + h.lessons)
  return 'COMMITTED baseline: ' + bestVersion + ' (committed mAP ' + bestFullMap + ', reference full run v124 ' + REF_MAP + ', floor ' + FLOOR.toFixed(5) + ').\nSession history:\n' + lines.join('\n')
}

const proposalSchema = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    mechanism: { type: 'string', description: 'What it does, mechanically, in 3-6 sentences' },
    target_family: { type: 'string', description: 'recall-ceiling | candidate-generation | score-quality-supervised | other (NOT boundary-outward-growth — BUE owns it)' },
    expected_map_gain: { type: 'number', description: 'Honest expected absolute mAP gain; must be >= 0.03' },
    effect_size_arithmetic: { type: 'string', description: 'Which of the 4 AP checkpoints (1st/7th/14th/20th hit) moves, how many detections must change, and why that yields >= +0.03' },
    implementation_plan: { type: 'string', description: 'Exact files/functions/lines, config field name, CLI flag names, marker string' },
    paired_measurable: { type: 'boolean', description: 'true if it can log conf=<old>-><new> per fire so paired mode applies' },
    fm_risk_table: { type: 'string', description: 'FM1-FM16 each: immune / at-risk + why' },
    banned_family_check: { type: 'string', description: 'Explicit statement that it is none of the 7 banned families, and why' },
    rollback_plan: { type: 'string' },
    weaknesses: { type: 'string', description: 'The two strongest arguments against it' },
  },
  required: ['title', 'mechanism', 'target_family', 'expected_map_gain', 'effect_size_arithmetic', 'implementation_plan', 'paired_measurable', 'fm_risk_table', 'banned_family_check', 'rollback_plan', 'weaknesses'],
  additionalProperties: false,
}

const verdictSchema = {
  type: 'object',
  properties: {
    approve: { type: 'boolean' },
    feedback: { type: 'string' },
    mandated_changes: { type: 'string' },
    effect_size_ok: { type: 'boolean', description: 'Is the >= +0.03 expectation honestly argued (not inflated)?' },
    banned_family_ok: { type: 'boolean', description: 'Is it genuinely outside all 7 banned families?' },
  },
  required: ['approve', 'feedback', 'effect_size_ok', 'banned_family_ok'],
  additionalProperties: false,
}

const implSchema = {
  type: 'object',
  properties: {
    implemented: { type: 'boolean' },
    activation_marker: { type: 'string', description: 'Unique INFO substring emitted per fire, greppable' },
    eval_flags: { type: 'string', description: 'Exact CLI flags to activate the mechanism' },
    paired_marker: { type: 'string', description: 'Marker for paired mode if the mechanism logs conf=<old>-><new>, else empty' },
    files_changed: { type: 'string' },
    validation: { type: 'string', description: 'AST/import/--help/default checks actually run, with results' },
    notes: { type: 'string' },
  },
  required: ['implemented', 'activation_marker', 'eval_flags', 'files_changed', 'validation'],
  additionalProperties: false,
}

const launchSchema = {
  type: 'object',
  properties: { launched: { type: 'boolean' }, note: { type: 'string' } },
  required: ['launched', 'note'], additionalProperties: false,
}

const watchSchema = {
  type: 'object',
  properties: {
    done: { type: 'boolean' }, crashed: { type: 'boolean' },
    mAP: { type: 'number' }, ap30: { type: 'number' }, ap50: { type: 'number' }, ap70: { type: 'number' },
    latency: { type: 'number' }, marker_count: { type: 'number' }, note: { type: 'string' },
  },
  required: ['done', 'crashed', 'mAP', 'marker_count', 'note'], additionalProperties: false,
}

const bootSchema = {
  type: 'object',
  properties: {
    ran: { type: 'boolean' },
    mode: { type: 'string' },
    point_delta: { type: 'number' },
    ci90_low: { type: 'number' }, ci90_high: { type: 'number' },
    p_gt0: { type: 'number' },
    new_map: { type: 'number' },
    note: { type: 'string' },
  },
  required: ['ran', 'point_delta', 'p_gt0', 'new_map', 'note'], additionalProperties: false,
}

const resultSchema = {
  type: 'object',
  properties: {
    version: { type: 'string' }, implemented: { type: 'boolean' },
    smoke_map: { type: 'number' }, full_map: { type: 'number' },
    kept: { type: 'boolean' }, summary: { type: 'string' }, lessons: { type: 'string' },
    report_path: { type: 'string' },
  },
  required: ['version', 'implemented', 'full_map', 'kept', 'summary', 'lessons'], additionalProperties: false,
}

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

function researchPrompt(role, focus) {
  return [
'You are Agent ' + role + ' on a VAD research team. GOAL THIS SESSION: raise full-eval temporal mAP with a mechanism whose HONEST expected effect is >= +0.03. ' + focus,
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'TASK:',
'1. MANDATORY READING (skim smart, Read with offset/limit): ' + LOGS + '/ideas/loop3_v140_faithful-substrate-and-noise-floor_20260728.md (the noise floor - read this FIRST), ' + LOGS + '/ideas/loop1_v138_gpc-global-peak-containment_20260727.md (oracle ceilings + checkpoint decomposition), the v142 BUE header block at the TOP of ' + CODE + '/pipeline.py (so you do NOT duplicate BUE), ' + LOGS + '/session_v97_ledger.md (tail rows).',
'2. PROFILE THE ACTUAL FAILURE: 50 of 99 detections have ZERO GT overlap. BUE already grows EXISTING windows outward, so it CANNOT help the zero-overlap misses whose GT is elsewhere in the video. Find out WHY the pipeline looks in the wrong place - which stage decides WHERE to look (anomaly_discovery.py directed scan / candidate selection). Quantify with the CACHED runs + ' + LOGS + '/analysis_v140_livecurve_lib.py (faithful curve, no API cost): on the zero-overlap detections, is the GT elsewhere in the video entirely (candidate-generation failure) or adjacent-but-missed? This diagnosis is the deliverable that justifies your proposal.',
'3. SEARCH: >=4 real WebSearch queries on weakly-supervised temporal action localization / VAD proposal generation, candidate/actionness generation, snippet-level score quality 2024-2026. If WebSearch is unavailable, say so explicitly and rely on code+data measurement instead of inventing citations.',
'4. Derive ONE architecture-level mechanism with expected effect >= +0.03 that is NOT boundary outward-growth (BUE owns it). It MUST come with the effect-size arithmetic (which AP checkpoint moves, how many detections must change), a config flag + CLI wiring + unique INFO marker (FM5), and an FM1-FM16 risk table. Prefer PAIRED-MEASURABLE designs where possible.',
'5. Return via StructuredOutput (EXACTLY ONCE).',
'',
'!!! ANTI-STALL: no Bash > ~60s, no sleep, Read with offset/limit, short progress line between tool calls, finish in <18 tool calls.',
  ].join('\n')
}

function evalLaunchPrompt(V, tag, maxPerCat, evalFlags) {
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

function evalWatchPrompt(V, tag, maxPerCat, evalFlags, marker, nTotal) {
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
'   markers: grep -cF "' + (marker || '') + '" ' + lg + ' (also try the inner run.log; stdout can be block-buffered while run.log is line-buffered - trust the LARGER count).',
'2. Else check liveness: pgrep -f "batch_evaluate.*batch_eval_' + V + '_' + tag + '". If alive -> Bash "sleep 140", re-check, up to 2 sleeps per invocation, then return done=false crashed=false note="running: <n>/' + N + ' result JSONs" (count with: ls ' + ROOT + '/results/batch_eval_' + V + '_' + tag + '/*.json | grep -v evaluation_report | wc -l).',
'   STALL CHECK (learned 2026-07-28, 16 GB machine + concurrency 3): if the inner run.log has not grown in >30 min while the process is alive, the machine has likely exhausted swap - kill the process, then relaunch the SAME nohup command with --concurrency ' + CONCURRENCY + '; the runner SKIPS videos whose JSON exists, so nothing is lost. Report note="restarted after stall".',
'3. If dead with no report: tail -30 the log. TRANSIENT (API timeout/ConnectTimeout/rate limit/network) -> relaunch the same nohup command, return done=false crashed=false note="relaunched: <cause>". CODE BUG (Traceback in project code) -> done=false crashed=true note=<summary>.',
'4. If all ' + N + ' result JSONs exist but the process died before writing the report, the metrics can still be computed offline - say so in note and return done=false crashed=false note="' + N + '/' + N + ' artifacts, report missing".',
'Return: done, crashed, mAP, ap30, ap50, ap70, latency, marker_count, note.',
  ].join('\n')
}

async function runEval(V, tag, maxPerCat, evalFlags, marker, maxWatch, nTotal) {
  const launch = await retryAgent(() => agent(evalLaunchPrompt(V, tag, maxPerCat, evalFlags), { schema: launchSchema, label: 'D-launch-' + tag + '-' + V, phase: (tag === FULL290_TAG ? 'Full-290' : 'Implement') }))
  if (!launch || !launch.launched) {
    return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'launch failed: ' + (launch ? launch.note : 'agent null after retries') }
  }
  const cap = maxWatch || 60
  for (let w = 0; w < cap; w++) {
    const st = await retryAgent(() => agent(evalWatchPrompt(V, tag, maxPerCat, evalFlags, marker, nTotal), { schema: watchSchema, label: 'D-watch-' + tag + '-' + V + '-' + w, phase: (tag === FULL290_TAG ? 'Full-290' : 'Implement') }))
    if (!st) continue
    if (st.done || st.crashed) return st
  }
  return { done: false, crashed: true, mAP: -1, ap30: -1, ap50: -1, ap70: -1, latency: -1, marker_count: -1, note: 'watcher cap reached' }
}

function bootstrapPrompt(V, pairedMarker) {
  const dir = ROOT + '/results/batch_eval_' + V + '_full'
  const paired = pairedMarker && pairedMarker.length > 0
  return [
'You are Agent D-adjudicator for ' + V + '. Run the MANDATED verdict statistic (video-matched bootstrap CI) and report the numbers. Do NOT edit pipeline code. Bash < 150s per call. StructuredOutput EXACTLY ONCE.',
'',
'Run this (it self-reports PASS/FAIL of the pre-registered gate):',
paired
  ? ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --paired-log ' + dir + '/run.log --marker "' + pairedMarker + '" --min-delta ' + MIN_DELTA_PAIRED + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5))
  : ('  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir + ' --ref ' + REF_DIR + ' --min-delta ' + MIN_DELTA_CROSS + ' --min-p ' + MIN_P + ' --floor ' + FLOOR.toFixed(5)),
'If the run exceeds ~150s, re-invoke it with --resamples 300 (still valid, slightly wider CI) and say so in the note.',
'If the report file is missing but all 65 per-video JSONs exist, the tool still works - it reads the per-video artifacts directly.',
'',
'Report: ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map, note (include the tool\'s VERDICT line verbatim in note).',
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
'2. Implement the mechanism. New params go in config.py (dataclass default = OFF/prior behavior) AND config.from_args with explicit threading; CLI flags in BOTH run.py and batch_evaluate.py; bump the pipeline.py header to ' + V + ' with a short rationale block. Do NOT disturb the v141 CEX or v142 BUE code paths — both must stay default-OFF and byte-identical when their flags are absent.',
'3. FM5: emit a UNIQUE INFO marker per fire. If the mechanism is a post-hoc confidence transform, the marker MUST include "vid=<video_id> t=<t_start>-<t_end> ... conf=<old>-><new>" so paired-mode adjudication works.',
'4. VALIDATE and report actual outputs: ' + PY + ' -m py_compile on each edited file; ' + PY + ' -c "import sys;sys.path.insert(0,\'' + CODE + '\');from pipeline import MultiAnomalyVAD;print(1)"; the flags appear in --help; ScoringConfig() defaults are OFF; from_args(Namespace()).scoring == ScoringConfig().',
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
'SMOKE: ' + JSON.stringify(smoke) + '\nFULL: ' + JSON.stringify(full) + '\nBOOTSTRAP ADJUDICATION: ' + JSON.stringify(boot),
'\nORCHESTRATOR DECISION: ' + decisionText + (fm5 ? '\nFM5 VIOLATION: the activation marker never appeared - the mechanism did not run.' : ''),
'',
'TASKS:',
'1. If the decision is REVERT: restore every file from ' + ROOT + '/.qoder/backups/' + V + '/, then VERIFY with diff -q per file and confirm 0 remaining symbols of the mechanism (grep). Report exactly what you verified. If the decision is KEEP, leave the code in place and flip the config default to ON if the proposal says so.',
'2. Archive artifacts into ' + LOGS + '/results/: ' + V + '_full_eval.json, ' + V + '_full_run.log, ' + V + '_smoke_eval.json (whichever exist), plus bootstrap_ab.json.',
'3. Write ' + LOGS + '/ideas/loop' + loop + '_' + V + '_<slug>_<YYYYMMDD>.md (get the date with: date +%Y%m%d): mechanism, effect-size arithmetic vs what actually happened, the BOOTSTRAP CI (point delta, CI90, P(delta>0)) as the headline verdict, FM1-FM16 table, honest weaknesses, and what the next loop should NOT repeat.',
'4. APPEND one row to ' + LOGS + '/session_v97_ledger.md in the existing pipe-table format (loop | version | title+verdict+evidence | smoke_mAP | full_mAP | AP@0.5 | AP@0.7 | latency_s | kept). Use -1 for absent numbers. Do NOT rewrite existing rows.',
'5. Return version, implemented, smoke_map, full_map, kept, summary, lessons, report_path.',
  ].join('\n')
}

// ============================ MAIN LOOP ============================
log('v8 continuation starting at v' + START_VERSION + '. ' + MAX_LOOPS + ' iteration loops on the FIXED 65-video harness (--max_per_category 5, --concurrency ' + CONCURRENCY + '), then a decisive 290-video paired BUE adjudication.')
log('Inherited: v141 CEX (default-OFF, superseded), v142 BUE (default-OFF, strongest candidate, full-65 eval interrupted at ~6/65). Noise floor: 65-harness CI ~+/-0.025 so only >= +0.03 effects admissible.')

for (let loop = 1; loop <= MAX_LOOPS; loop++) {
  const V = 'v' + (START_VERSION + loop - 1)
  log('=== Loop ' + loop + '/' + MAX_LOOPS + ' -> ' + V + ' ===')

  phase('Research')
  const [ideaA, ideaB] = await parallel([
    () => retryAgent(() => agent(researchPrompt('A', 'YOUR ANGLE: WHERE the pipeline looks - CANDIDATE GENERATION and the recall ceiling. Diagnose the 50/99 zero-overlap detections as a candidate-selection problem (GT elsewhere in the video, not adjacent) and propose a mechanism that puts detections on the RIGHT events. This is the headroom BUE cannot touch. You may change which segments are scanned/selected, but you own the FM2 risk and must argue it.'),
      { schema: proposalSchema, label: 'A-research-' + V, phase: 'Research' })),
    () => retryAgent(() => agent(researchPrompt('B', 'YOUR ANGLE: snippet SCORE QUALITY from the untapped SUPERVISED assets (features/train 1610 videos, model_mil_v6.pth, prompt_weights.json lineage) to improve the score the detections are ranked/selected from. Do NOT touch boundary formation (BUE owns outward-growth; the v51 half-max owns crossing). You own the FM1 risk and must argue why this differs from the 10+ curve-shape failures.'),
      { schema: proposalSchema, label: 'B-research-' + V, phase: 'Research' })),
  ])
  if (!ideaA && !ideaB) { log('Loop ' + loop + ': both researchers failed, skipping'); continue }

  phase('Debate')
  const consensus = await retryAgent(() => agent([
'You are the joint A+B team. Cross-critique both ideas and synthesize ONE proposal to implement this loop.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'IDEA A:\n' + JSON.stringify(ideaA, null, 1), '', 'IDEA B:\n' + JSON.stringify(ideaB, null, 1), '',
'RULES FOR THE SYNTHESIS:',
'- Pick the ONE with the larger HONEST expected effect and the cleaner measurability; do not merge two mechanisms into one loop (FM8: one controlled change).',
'- Kill any part whose expected effect is < +0.03 - it is unmeasurable on the 65-video harness and will waste the loop.',
'- REJECT any boundary outward-growth / union-extension / OIC relocation part outright — BUE (v142) owns it and is pending its own 290 verdict (banned family 7).',
'- Attack each other on effect size FIRST (is the arithmetic real?), then on FM risk. Say plainly if BOTH ideas are below the bar, and in that case propose the strongest single alternative you can defend with numbers.',
'Return the unified proposal via StructuredOutput (EXACTLY ONCE).',
  ].join('\n'), { schema: proposalSchema, label: 'AB-consensus-' + V, phase: 'Debate' }))
  let proposal = consensus || ideaA || ideaB

  phase('Review')
  let verdict = null
  for (let r = 0; r < 3; r++) {
    verdict = await retryAgent(() => agent([
'You are Agent C, the strict quality gate. Judge this proposal. Your job is to prevent wasted evals - each full eval costs ~2-3 hours of API time. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
'MANDATORY CHECKLIST:',
'1. EFFECT SIZE (the binding gate): is the expected gain >= +0.03 AND is the arithmetic HONEST (which of the 4 AP checkpoints moves, how many detections must change)? An inflated number is worse than a small one - REJECT hand-waving. If the true expected effect is ~+0.01, REJECT: it is unfalsifiable on the 65-video harness (FM16).',
'2. BANNED FAMILIES: is it any of the 7 (curve-derived confidence re-rank / merge-batch / max_scan-tail / per-video normalization / dense-proposal-append / local-compute trim / boundary-outward-growth-owned-by-BUE)? If yes -> REJECT outright.',
'3. FM1: does it change the shape of the fused curve used for half-max boundary detection? If yes, this is the 11th such attempt - demand a specific argument why it differs, or REJECT.',
'4. FM2: does it change detection COUNT or which segments are included? Under greedy single-match mAP with ~1 GT per video, extra detections become FPs. Demand the count analysis.',
'5. FM5: config flag (default OFF, flag-off byte-identical) + CLI in both runners + unique INFO marker per fire? If the mechanism is post-hoc, does the marker carry conf=<old>-><new> for paired adjudication? If missing -> REJECT and demand it.',
'6. FM15: if any offline precheck is proposed on a CLIP curve, does it use analysis_v140_livecurve_lib.py (the fidelity-gated substrate)? A features/test-based precheck is an automatic REJECT.',
'7. FM8/FM11: exactly ONE controlled change, validatable by ONE full eval? Does it leave v141 CEX and v142 BUE default-OFF and untouched?',
'8. Realism: our 0.17 already exceeds most published weakly-supervised numbers. A claim of +0.10 needs extraordinary justification.',
'Set effect_size_ok and banned_family_ok explicitly. REJECT with concrete, actionable feedback, or APPROVE (optionally with mandated_changes).',
    ].join('\n'), { schema: verdictSchema, label: 'C-review-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!verdict || verdict.approve) break
    log('Loop ' + loop + ': C rejected (round ' + (r + 1) + '): ' + (verdict.feedback || '').slice(0, 200))
    if (r === 2) break
    proposal = await retryAgent(() => agent([
'You are the joint A+B team. Reviewer C REJECTED your proposal. Revise it, or pivot to a different mechanism that clears the >= +0.03 effect-size bar and avoids all 7 banned families. ONE controlled change. StructuredOutput EXACTLY ONCE.',
'', CONTEXT, '', 'SESSION HISTORY:\n' + hist(), '',
'REJECTED PROPOSAL:\n' + JSON.stringify(proposal, null, 1), '',
"C's FEEDBACK:\n" + verdict.feedback + (verdict.mandated_changes ? '\nMANDATED:\n' + verdict.mandated_changes : ''),
    ].join('\n'), { schema: proposalSchema, label: 'AB-revise-' + V + '-r' + (r + 1), phase: 'Review' }))
    if (!proposal) { proposal = consensus; break }
  }
  if (verdict && !verdict.approve) {
    if (verdict.mandated_changes) {
      proposal = Object.assign({}, proposal, {
        title: proposal.title + ' (C-mandated revision)',
        implementation_plan: proposal.implementation_plan + '\n\nC-MANDATED CHANGES:\n' + verdict.mandated_changes,
      })
      log('Loop ' + loop + ': proceeding with C-mandated revision')
    } else {
      log('Loop ' + loop + ': C rejected 3x without mandate - no eval spent (FM11 economy)')
      history.push({ loop: loop + 5, version: V, title: proposal.title + ' [REJECTED at gate - no implementation]', full_map: -1, kept: false, lessons: 'C rejected 3x: ' + (verdict.feedback || '').slice(0, 300) })
      continue
    }
  }

  phase('Implement')
  const impl = await retryAgent(() => agent(d1Prompt(V, loop, proposal, verdict), { schema: implSchema, label: 'D1-implement-' + V, phase: 'Implement' }))
  if (!impl || !impl.implemented || !impl.activation_marker) {
    log('Loop ' + loop + ': implementation failed or no marker (FM5) - skipping eval')
    history.push({ loop: loop + 5, version: V, title: proposal.title + ' [IMPL FAILED]', full_map: -1, kept: false, lessons: impl ? ('no marker / failed: ' + (impl.notes || '').slice(0, 200)) : 'agent null after retries' })
    continue
  }

  const smoke = await runEval(V, 'smoke', 1, impl.eval_flags, impl.activation_marker, 30, 13)
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
      full = await runEval(V, 'full', 5, impl.eval_flags, impl.activation_marker, 80, 65)
      log('Loop ' + loop + ': full done=' + full.done + ' crashed=' + full.crashed + ' mAP=' + full.mAP + ' markers=' + full.marker_count)
      if (full.done && full.marker_count <= 0) { fm5 = true; log('Loop ' + loop + ': FM5 - marker absent from the full log') }
      if (!fm5 && full.done && !full.crashed) {
        boot = await retryAgent(() => agent(bootstrapPrompt(V, impl.paired_marker || ''), { schema: bootSchema, label: 'D-bootstrap-' + V, phase: 'Implement' }))
        const paired = !!(impl.paired_marker && impl.paired_marker.length > 0)
        const bar = paired ? MIN_DELTA_PAIRED : MIN_DELTA_CROSS
        if (!boot || !boot.ran) {
          decisionText = 'REVERT (adjudication did not run: ' + (boot ? boot.note : 'agent null') + ' - no verdict statistic, so no KEEP)'
        } else {
          const okD = boot.point_delta >= bar
          const okP = boot.p_gt0 >= MIN_P
          const okF = boot.new_map >= FLOOR
          if (okD && okP && okF) {
            keep = true
            decisionText = 'KEEP (' + (paired ? 'paired' : 'cross-run') + ' point delta ' + boot.point_delta.toFixed(5) + ' >= ' + bar + ', bootstrap P(delta>0)=' + boot.p_gt0.toFixed(3) + ' >= ' + MIN_P + ', mAP ' + boot.new_map.toFixed(5) + ' >= floor ' + FLOOR.toFixed(5) + '. CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + ']. This becomes the new baseline.)'
          } else if (!okF) {
            decisionText = 'REVERT (ACCURACY FLOOR breached: mAP ' + boot.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
          } else if (!okD) {
            decisionText = 'REVERT (effect too small to be real on this harness: point delta ' + boot.point_delta.toFixed(5) + ' < ' + bar + '; P(delta>0)=' + boot.p_gt0.toFixed(3) + '. FM16 - do not KEEP inside the noise floor.)'
          } else {
            decisionText = 'REVERT (delta ' + boot.point_delta.toFixed(5) + ' but bootstrap P(delta>0)=' + boot.p_gt0.toFixed(3) + ' < ' + MIN_P + ' - CI90 [' + boot.ci90_low + ', ' + boot.ci90_high + '] straddles zero, so the gain is not established.)'
          }
        }
      }
    }
  } else {
    log('Loop ' + loop + ': smoke gate failed (done=' + smoke.done + ' crashed=' + smoke.crashed + ' mAP=' + smoke.mAP + ')')
  }
  if (fm5) decisionText = 'REVERT (FM5: the activation marker never appeared - the mechanism never ran, so the measurement is meaningless)'

  const fin = await retryAgent(() => agent(d4Prompt(V, loop, proposal, verdict, impl, smoke, full, boot, decisionText, fm5), { schema: resultSchema, label: 'D4-record-' + V, phase: 'Implement' }))
  const result = fin || { version: V, implemented: true, smoke_map: smoke.mAP, full_map: full.mAP, kept: keep, summary: '(recorder agent failed; orchestrator metrics only)', lessons: fm5 ? 'FM5: never activated' : 'recorder failed', report_path: ROOT + '/results/batch_eval_' + V + '_full/evaluation_report.json' }
  const bootTag = boot && boot.ran ? (' [bootstrap d=' + boot.point_delta.toFixed(5) + ' P(d>0)=' + boot.p_gt0.toFixed(3) + ']') : ''
  history.push({ loop: loop + 5, version: V, title: proposal.title + bootTag, full_map: result.full_map, kept: keep, lessons: (result.lessons || '').slice(0, 400) })
  if (keep && result.full_map > 0) {
    bestFullMap = result.full_map
    bestVersion = V
    log('Loop ' + loop + ': KEPT ' + V + ' - new baseline mAP ' + result.full_map)
  } else {
    log('Loop ' + loop + ': ' + V + ' REVERTED - ' + decisionText.slice(0, 160))
  }
}

// ==================== FINAL PHASE: 290-VIDEO PAIRED BUE ADJUDICATION ====================
phase('Full-290')
log('=== FINAL PHASE: 290-video eval WITH --bue_extension (paired), concurrency ' + CONCURRENCY + ' ===')
log('This is the decisive BUE verdict: the 65-harness cannot measure +0.037 (CI +/-0.025); the 290 split resolves ~+/-0.012.')

const full290 = await runEval(FULL290_V, FULL290_TAG, FULL290_MAXCAT, FULL290_FLAGS, FULL290_MARKER, 200, 290)
log('290-video eval: done=' + full290.done + ' crashed=' + full290.crashed + ' mAP=' + full290.mAP + ' BUE markers=' + full290.marker_count + ' note=' + (full290.note || ''))

let boot290 = null
if (full290.done && !full290.crashed) {
  const dir290 = ROOT + '/results/batch_eval_' + FULL290_V + '_' + FULL290_TAG
  boot290 = await retryAgent(() => agent([
'You are Agent D-adjudicator for the FINAL 290-video BUE verdict. Run the MANDATED paired bootstrap on pre_bue_boundaries and report the numbers verbatim. Do NOT edit pipeline code. Bash < 150s per call; if a run exceeds ~150s re-invoke with --resamples 300 and say so. StructuredOutput EXACTLY ONCE.',
'',
'This is a PAIRED within-run A/B: the treatment arm is the final (BUE-extended) windows in the result JSONs; the baseline arm is the index-aligned pre_bue_boundaries field each JSON carries when --bue_extension is on. This removes ALL VLM run-to-run noise, so the delta is BUE\'s marginal localization effect at 290 videos.',
'Run:',
'  ' + PY + ' ' + LOGS + '/analysis_bootstrap_ab.py --new ' + dir290 + ' --paired-json ' + BUE_PAIRED_FIELD + ' --min-delta ' + BUE_MIN_DELTA + ' --min-p ' + BUE_MIN_P + ' --floor ' + FLOOR.toFixed(5),
'Pre-registered BUE KEEP rule (v142 header): paired point delta >= ' + BUE_MIN_DELTA + ' AND P(delta>0) >= ' + BUE_MIN_P + ' AND mAP >= ' + FLOOR.toFixed(5) + '. A P in [0.85,0.90) is a clean pass under BUE\'s pre-registered rule (concentrated boundary operator, wider CI ~+/-0.055) but MUST be reported with the half-width, never silently.',
'If the tool reports "FM5 GATE FAILED: no ' + BUE_PAIRED_FIELD + ' rows", then --bue_extension did not actually run - say so plainly (ran=false).',
'',
'Report: ran, mode, point_delta, ci90_low, ci90_high, p_gt0, new_map (=the BUE/treatment 290 mAP), note (include the tool\'s VERDICT line verbatim, plus the baseline/pre-BUE 290 mAP if the tool prints it).',
  ].join('\n'), { schema: bootSchema, label: 'D-bootstrap-290-bue', phase: 'Full-290' }))
}

let bueKeep = false
let bue290Decision = 'REVERT / leave BUE default-OFF'
if (boot290 && boot290.ran) {
  const okD = boot290.point_delta >= BUE_MIN_DELTA
  const okP = boot290.p_gt0 >= BUE_MIN_P
  const okF = boot290.new_map >= FLOOR
  if (okD && okP && okF) {
    bueKeep = true
    bue290Decision = 'KEEP BUE (flip bue_enabled default ON): 290-video paired delta ' + boot290.point_delta.toFixed(5) + ' >= ' + BUE_MIN_DELTA + ', P(d>0)=' + boot290.p_gt0.toFixed(3) + ' >= ' + BUE_MIN_P + ', mAP ' + boot290.new_map.toFixed(5) + ' >= floor ' + FLOOR.toFixed(5) + '. CI90 [' + boot290.ci90_low + ', ' + boot290.ci90_high + '].'
  } else if (!okF) {
    bue290Decision = 'REVERT BUE (floor breached: 290 mAP ' + boot290.new_map.toFixed(5) + ' < ' + FLOOR.toFixed(5) + ')'
  } else if (!okD) {
    bue290Decision = 'REVERT BUE (290 paired delta ' + boot290.point_delta.toFixed(5) + ' < ' + BUE_MIN_DELTA + ' - the offline +0.037 did NOT transfer in vivo)'
  } else {
    bue290Decision = 'PROVISIONAL/REVERT BUE (delta ' + boot290.point_delta.toFixed(5) + ' but P(d>0)=' + boot290.p_gt0.toFixed(3) + ' < ' + BUE_MIN_P + '; CI90 [' + boot290.ci90_low + ', ' + boot290.ci90_high + '] not decisive)'
  }
} else {
  bue290Decision = 'INCONCLUSIVE (290 adjudication did not run: ' + (full290.crashed ? '290 eval crashed - ' + (full290.note || '') : (boot290 ? (boot290.note || 'bootstrap null') : 'eval not done')) + '). Leave BUE default-OFF.'
}
log('BUE 290 verdict: ' + bue290Decision)

// Final recorder: apply the BUE decision to config defaults (with backup) and write the session summary.
await retryAgent(() => agent([
'You are Agent D5, the FINAL recorder for the v8 continuation session. Apply the BUE 290-video decision to the code, then write the session summary. StructuredOutput is NOT required; reply with a 12-line plain-text recap at the end.',
'', CONTEXT, '',
'FULL SESSION HISTORY:\n' + hist(), '',
'290-VIDEO BUE ADJUDICATION: ' + JSON.stringify(boot290) + '\nDECISION: ' + bue290Decision, '',
'TASKS:',
'1. BUE DEFAULT: ' + (bueKeep
  ? 'The decision is KEEP. Back up config.py, run.py, batch_evaluate.py to ' + ROOT + '/.qoder/backups/full290_bue_keep/ FIRST, then flip the default so BUE ships ON: config.py ScoringConfig.bue_enabled default False->True, config.from_args getattr default for bue_enabled True, and set_defaults(bue_enabled=True) in BOTH run.py and batch_evaluate.py (keep --no_bue_extension working). VALIDATE: ' + PY + ' -m py_compile on all three; ' + PY + ' -c "import sys;sys.path.insert(0,\'' + CODE + '\');from config import ScoringConfig;print(ScoringConfig().bue_enabled)" prints True; confirm --no_bue_extension still parses. Report the diff.'
  : 'The decision is NOT a clean KEEP. Leave bue_enabled default OFF (byte-identical to committed v137). Do NOT edit config defaults. Confirm ' + PY + ' -c "import sys;sys.path.insert(0,\'' + CODE + '\');from config import ScoringConfig;print(ScoringConfig().bue_enabled)" prints False.') ,
'2. Archive: copy ' + ROOT + '/results/batch_eval_' + FULL290_V + '_' + FULL290_TAG + '/evaluation_report.json (if present) and the bootstrap output to ' + LOGS + '/results/full290_bue_verdict.json / .txt.',
'3. Get the date: date +%Y%m%d. Write ' + LOGS + '/summary_v8_continuation_v143-v144_plus_bue290_<YYYYMMDD>.md containing: (a) executive summary — did v143 or v144 clear >= +0.03 with P>=' + MIN_P + '? did the floor hold? what is the BUE 290 verdict (paired delta, CI90, P(d>0), baseline vs BUE 290 mAP)?; (b) a per-version table (v143, v144: mechanism | target family | point delta | CI90 | P(d>0) | kept); (c) the BUE 290 row as the headline result; (d) whether the recall/candidate-generation family still has >= +0.03 reachable headroom; (e) updated FM1-FM16 taxonomy; (f) concrete next directions that respect the noise floor.',
'4. APPEND to ' + LOGS + '/session_v97_ledger.md one row for the BUE 290 verdict in the existing pipe-table format (loop | version="BUE@290" | title with the paired delta+CI90+P(d>0)+decision | smoke_mAP=-1 | full_mAP=<290 BUE mAP> | AP@0.5=-1 | AP@0.7=-1 | latency_s=-1 | kept=<true/false>). Do NOT rewrite existing rows; the v143/v144 rows were already appended by their loop recorders.',
'5. Reply with the 12-line recap: the two loop verdicts, the BUE 290 verdict, and the current committed baseline.',
].join('\n'), { label: 'D5-final-record', phase: 'Full-290' }))

log('v8 session complete. Committed baseline ' + bestVersion + ' mAP ' + bestFullMap + '. BUE 290 verdict: ' + bue290Decision + '. ' + history.length + ' recorded loops.')
return { bestVersion, bestFullMap, bueKeep, bue290Decision, full290_map: full290.mAP, boot290 }
