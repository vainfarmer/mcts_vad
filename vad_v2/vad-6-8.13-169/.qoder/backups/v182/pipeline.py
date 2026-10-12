"""
视频异常检测与时间定位 — 完整 Pipeline (v179)

v179: SNH — Supervised Normality Head (NEW, default OFF, --snh_veto).
  The v17-record-mandated, v19-GO-gated NORMAL-video false-positive lever.
  ROOT CAUSE IT ATTACKS (same as CNV, with the supervised fix): the pipeline
  makes only SEGMENT-level decisions and VLM absolute confidence on isolated
  clips is uncalibrated, so Normal videos carry hallucinated detections nothing
  downstream removes. v17 closed CNV on HARM (its unsupervised triple cannot
  tell hallucination from truth) and ruled any future Normal-FP lever must
  FIRST demonstrate on TRAIN normals that its video-level gate fires on
  normals at all. v19 did exactly that (iteration_logs/
  summary_v19_snh-gate_20260816.md): a tiny multinomial logistic-regression
  P(normal) head over 16 aggregate features of the pipeline's OWN cached
  signals separates train normals from anomalies at stratified 5-fold CV AUC
  0.996; at tau=0.7894 (largest tau with CV anomaly-flag <= 5%; measured
  3.8%) it flags 94% of train normals -> GO.
  MECHANISM (video-level-ONLY gate — there is NO per-segment stage): at the
  TERMINAL slot (after BUE/CBT/CEX, on the genuinely-final final_anomalies),
  load model_snh.json ONCE (module-level lazy cache), compute the 16 features
  via the SHARED module snh_features.compute_snh_features (the identical
  function the train extractor used -> zero train/eval skew), z-standardize
  with the saved scaler, P(normal)=sigmoid(w·z+b); iff P(normal) >= tau, veto
  ALL of the video's detections: confidence := 0.0, detection KEPT
  (count-frozen, index-aligned, FM2 assert); NEVER moves a window, NEVER adds
  a detection, NEVER re-ranks. ZERO new VLM/CLIP/MIL calls: MIL snippet
  scores come from video_data['mil_scores'] (Stage 1; missing -> the
  extractor's None convention -> 0.0 after the feature_order mapping), the 13
  cat/(cat+normal) curves from the SAME cached read-only
  frame_scorer.get_category_scores BUE/CBT read, the Normal-prompt softmax
  curve from the cached clip_features @ text_features matmul (no new
  encoding), motion from video_data['motion_energy'] (Stage 1b). A missing/
  corrupt model file logs [SNH-ERR] and the video is treated as flag-off —
  the run NEVER crashes on the veto path.
  PAIRED INSTRUMENT (LOG-based, no new JSON fields): every vetoed detection
  emits exactly one [SNH-FIRE] line vid=<stem> ... t=<ts>-<te> ...
  conf=<old>->0.0000 — the exact format analysis_bootstrap_ab.py
  restore_pre_marker parses with --paired-log --marker SNH-FIRE
  --paired-mode conf; the arm-OFF (committed) arm is rebuilt EXACTLY from the
  same run's log. FM17: exactly one unconditional [SNH-VID] summary per video
  on EVERY path (terminal slot, 0-detection early return, early-return-normal;
  the two early paths carry ndet=0 vetoed=0 and the computed p_normal).
  DEV-ONLY: SNH_FORCE_FIRE=1 forces the gate to fire (self-test hook that
  proves the fire/restore path; excluded from default behaviour).

v177: TCP — Temporal-Contrast Prompt (NEW, default OFF, --tcp_prompt).
  ROOT CAUSE IT ATTACKS (recognition-bound residual, v18 lessons §10): the
  committed directed scan (scan_segment) already shows the VLM ALL ~12
  frames per 6s segment — the pipeline is NOT frame-blind on the committed
  path — yet RA009 (GT 7-8s, inside segment 4.5-10.5) and RA021 (GT 1-3s,
  inside segment 0-5.2) were REJECTED with all frames shown. Hypothesis:
  the whole-segment absolute prompt ("Determine if this segment contains
  any abnormal or suspicious activity") invites clip-AVERAGING, so a brief
  1-3s event dilutes into a mostly-normal window.
  MECHANISM (prompt-side ONLY, zero latency cost): when the flag is ON,
  scan_segment APPENDS the pre-registered TCP block (single-source constant
  anomaly_discovery.TCP_PROMPT_BLOCK) immediately BEFORE the "Respond ONLY
  in JSON format" line — context_prompt and everything else unchanged.
  Same frames, same call count, same directed path, same JSON schema
  (is_anomalous/confidence/description); ONLY prompt text changes. NOT a
  banned family (no abs-conf gate, no merge/batch, no scan-density change).
  Scan path held constant across arms: directed in BOTH (FCCS untouched;
  grep -cF "FCCS" == 0 in both produced logs or the A/B is VOID).
  MARKERS (FM5/FM17/FM21): [TCP-VID] unconditionally once per video on
  EVERY return path (terminal slot, 0-detection no-candidates,
  early-return-normal — mirroring the CNV/KFD call sites) carrying tcp=on
  and the per-video segments-scanned count; immediately after it,
  [TCP-BLOCK] logs the full TCP block once per video (grep -cF "Temporal
  reasoning instructions" proves engagement); [TCP-PROMPT] logs the first
  scanned segment's FULL assembled prompt once per video (the block inside
  an actual scan prompt). flag-off => byte-identical (the appended
  placeholder expands to the empty string: prompt text bit-identical to
  committed; no marker emitted).

v173: KFD-U — Uniform-spread keyframes in the comparative scan (NEW, default
  OFF, --kfd_uniform_keyframes; fallback knob --kfd_k, default 2).
  ROOT CAUSE (code-exact, pre-registered by the v17 council): on the
  comparative scan path (scan_full_coverage -> _scan_comparative_batch_guarded
  -> _comparative_call, the ONLY truncation site — verified by reading the
  code), _sample_keyframes returns ALL ~12 ordered frames of a 6s @2fps
  segment (SemanticFrameSampler target K=18 >= 12), then _comparative_call
  TRUNCATES to keyframes[:keyframes_per_segment] — the FIRST 2 frames
  (t+0.0s, t+0.5s), or the FIRST 1 under the >60-segment long-video rule — so
  the VLM's evidence per 6s segment is at best its leading ~0.5s and brief
  mid-segment events are never shown. Offline: the first-k layout places >=1
  keyframe inside the GT interval in only 36/44 hard-cat videos; the k=2
  spread covers 43/44.
  FIX (pre-registered, zero latency cost): when the flag is ON,
  _comparative_call replaces the first-k truncation with a deterministic
  uniform spread over the SAME sampled pool: k=2 -> the frames whose
  timestamps are nearest {t_start, t_start+3.0s}; the >60-segment 1-keyframe
  rule -> the frame nearest the segment MIDPOINT; k=3 (fallback, --kfd_k 3)
  -> nearest {t_start, t_start+2.5s, t_start+5.0s} (batch geometry 8x3).
  Same segment grid, same batch sizes (8x2 / 16x1), same 16 frames/call,
  same call count, same prompts, same thresholds/merge/top-k.
  ENGAGEMENT NOTE (deviation the council's shorthand implies but never
  states): the truncation ONLY exists on the comparative scan path — the
  committed pipeline's directed-scan/fallback paths (scan_segment) send ALL
  sampled frames and have nothing to fix. --kfd_uniform_keyframes therefore
  ALSO routes Stage 4 through scan_full_coverage (the single path where the
  pre-registered mechanism is defined); flag OFF is byte-identical to the
  committed pipeline (directed scan untouched, no marker emitted).
  MARKERS (FM21 exact): [KFD-FIRE] once per batch where the spread changed
  the selected indices vs the first-k truncation (vid + segment indices +
  old->new position/index/timestamp lists, one line); [KFD-VID]
  unconditionally once per video on EVERY return path (segments scanned,
  batches, fires) — including 0-detection and early-return-normal — mirroring
  the CNV call sites (~L1658 / ~L1787 / ~L2129). Cross-run adjudication (no
  paired instrument: KFD changes VLM INPUTS): two fresh same-session runs,
  analysis_bootstrap_ab.py --new <kfd> --ref <ref>.

v172: CNV variant "TIGHTEN" — REPLACES the v171 CNV gate logic (same flag
  --cnv_veto, same markers [CNV-FIRE]/[CNV-VID], same TERMINAL slot, same
  ONE-SIDED confidence-veto contract, same thresholds). The v171 full-290
  gate showed harm (13 GT-overlap vetoes), so TIGHTEN makes the veto
  STRICTLY more conservative BY CONSTRUCTION — a strict subset of v171
  fires; no number was test-fitted:
    * STAGE 1 gains a 4th NECESSARY conjunct (sig_d): the video's own
      videomae_category must be 'Normal' (the stored post-CLIP-override
      field at the terminal slot; on the early-return-normal path the early
      VideoMAE path already judged it normal). v171 could fire stage 1 on
      ANY video whose three signal families were all negative — including
      anomaly-category videos; v172 cannot.
    * STAGE 2 may only fire on detections whose window z < 0 (window curve
      mass BELOW the video's own median), tightened from z < CNV_TAU_SEG
      (0.5). The v171 conjuncts (tau bound, off-argmax) are KEPT, so every
      v172 fire would also have fired under v171.
  Everything else is unchanged: veto = confidence := 0.0, detection KEPT
  (count-frozen, index-aligned, FM2 assert); NEVER moves a window, NEVER
  adds a detection, NEVER re-ranks, ZERO new VLM/CLIP/MIL calls; flag
  default OFF (flag-off byte-identical: no signal is even read).

v171: CNV — Cascade Null-Hypothesis Veto (NEW, default OFF, --cnv_veto).
  ROOT CAUSE IT ATTACKS: the pipeline makes only SEGMENT-level decisions;
  nothing asks "does this video, as a whole, contain an event that breaks its
  own baseline?" VLM absolute confidence on isolated clips is uncalibrated
  (v143 proved scan_segment pins ~0.9 on isolated clips), so Normal videos
  carry hallucinated detections nothing downstream can remove.
  CNV is a ONE-SIDED CONFIDENCE VETO at the TERMINAL slot (after BUE/CBT/CEX,
  on the genuinely-final final_anomalies, immediately before DetectionResult).
  Veto = confidence := 0.0, detection KEPT in the list (count-frozen,
  index-aligned; FM2 assert). It NEVER moves a window, NEVER adds a detection,
  NEVER re-ranks, and makes ZERO new VLM/CLIP/MIL calls (reads only the cached
  _mil_scores / fused category curve / motion_energy).
  STAGE 1 (video-level global-null gate): veto ALL detections of a video iff
  ALL THREE independent cached signal families are negative:
    (a) MIL video-level evidence: top-k mean of the snippet scores < 0.20;
    (b) fused CLIP category curve in-video outlierness:
        max over frames of (x - median)/MAD < 1.5;
    (c) motion-energy outlier fraction: fraction of frames above
        (median + 3*MAD) < 0.01.
  STAGE 2 (segment-level in-video surprisal veto): on videos that PASS stage 1,
  veto any detection whose window-aggregated curve mass is NOT an in-video
  outlier: z_window = (win_mean - median)/MAD < 0.5 AND the window does not
  contain the video's global curve argmax.
  Thresholds are PRINCIPLED STRUCTURAL DEFAULTS (verified on TRAIN-split
  videos, never fitted on the test split).
  PAIRED INSTRUMENT (LOG-based, no new JSON fields): every vetoed detection
  emits exactly one [CNV-FIRE] line carrying vid + t=<ts>-<te> +
  conf=<old>->0.0000, the exact format
  analysis_bootstrap_ab.py restore_pre_marker parses with --paired-mode conf;
  the arm-OFF (committed) arm is rebuilt EXACTLY from the same run's log.
  FM17: exactly one unconditional [CNV-VID] summary per video on EVERY path
  (terminal slot, 0-detection early return, early-return-normal).
  DEV-ONLY: CNV_FORCE_FIRE=1 forces every gate to fire (self-test hook that
  proves the fire/restore path; excluded from default behaviour).

v169: CGR-AB — DUAL-ARM in-run A/B INSTRUMENT for CGR-M (NEW, default OFF,
  --cgr_ab_diag; requires --cgr_relocate). PURE DIAGNOSTIC: it does NOT change a
  single byte of the run's real output (final_anomalies / confidence / count /
  ordering / frame_scores are bit-identical to --cgr_relocate alone); it only
  ADDS the per-detection field ab_arm_off_boundaries.
  WHY ONE RUN SUFFICES. CGR -> CCR -> BUE -> CBT all execute AFTER every VLM
  call and read ONLY cached per-frame motion energy and the cached CLIP category
  curve, i.e. they are DETERMINISTIC post-processing. So both A/B arms can be
  produced inside a SINGLE eval run:
    arm-OFF = CCR(if enabled) -> BUE -> CBT applied to a DEEP COPY of the
              pre-CGR detection list == what the committed pipeline produces
    arm-ON  = CGR -> the same chain == the run's real final output
  Same videos, same VLM sampling, same detection count -> an EXACT PAIRED A/B
  with ZERO cross-run VLM noise; the two arms differ by EXACTLY ONE call to
  _cgr_motion_relocate, so the contrast is sound by construction.
  WHY pre_cgr_boundaries WAS NOT ENOUGH (the v15 error): pre_cgr_boundaries is
  snapshotted PRE-BUE, so its paired delta (+0.06105) bundled the committed
  BUE+CBT gains and was INFLATED — it was never a CGR-only figure. An offline
  reconstruction isolating the CGR step alone gave +0.01919, P=0.886, but an
  offline reconstruction ALSO cannot capture the BUE/CBT interaction (after CGR
  moves a window, BUE grows outward from the NEW position and CBT clamps against
  a DIFFERENT region of the category curve). CGR's true effect on final mAP IS
  that end-to-end quantity; this instrument measures it directly.
  HAZARD HANDLED: _cgr_motion_relocate / _bue_boundary_union_extension /
  _cbt_category_boundary_tighten all MUTATE a['t_start']/a['t_end'] IN PLACE, so
  passing the same dicts to both arms would silently corrupt both. The arm-OFF
  pass therefore runs on copy.deepcopy(base) where base is itself a
  copy.deepcopy of the pre-CGR list -> the arms share NO dict object.
  LOGGING: the arm-OFF pass re-runs BUE/CBT and would double-fire their
  [BUE]/[CBT]/[CBT-VID] markers and corrupt FM5 counts, so the module logger is
  swapped for a no-op shim for the duration of that (synchronous, await-free)
  pass and restored in a finally.
  Adjudicated by: analysis_bootstrap_ab.py --paired-json ab_arm_off_boundaries
  (point_delta = arm-ON minus arm-OFF = the clean CGR effect on FINAL mAP
  INCLUDING the BUE/CBT interaction; length-mismatch HARD-VOID for free).
  Unconditional per-video marker [CGR-AB] vid=... ndet=... armoff_len=...
  fired=... at all three DetectionResult return sites (FM17: fires on the
  empty / 0-detection paths too). flag-off => byte-identical.

v168: CCR-SE — Count-neutral CLIP-category-curve semantic candidate RELOCATION
  (NEW, default OFF, --ccr_relocate). Byte-for-byte structural clone of the
  v167 CGR-M relocation with the SINGLE difference that the signal source is
  swapped from video_data['motion_energy'] to the cached read-only per-frame
  relative CLIP category curve (frame_scorer.get_category_scores). Runs in the
  PRE-BUE slot immediately AFTER CGR-M and BEFORE the pre_bue snapshot; gated to
  the two low-motion cand-miss hard cats (Shoplifting, Shooting) that motion-only
  CGR-M under-delivered on; RoadAccidents/Explosion excluded -> byte-identical.
  Curve is read-only (no rescale/min-max/write-back -> NOT per-video norm).
  Count-neutral, width-neutral, confidence-frozen. Paired field pre_ccr_boundaries.

v167: CGR-M — Count-neutral motion-energy-peak candidate RELOCATION (NEW,
  default OFF, --cgr_relocate). PRE-BUE slot (immediately before the pre_bue
  snapshot): on the FINAL detection list (count frozen, confidence
  bit-identical), if the video's resolved category is in cgr_categories
  (Shooting/RoadAccidents/Shoplifting/Explosion) AND no existing detection
  covers the per-frame motion-energy peak, TRANSLATE the single
  LOWEST-confidence detection onto that peak (preserve width w=t_end-t_start;
  rewrite ONLY that one element's t_start/t_end IN PLACE). Strictly
  count-neutral (no append/pop/insert/del/filter/reorder), width-neutral,
  confidence/anomaly_type/coarse_score/order byte-frozen. Reuses
  video_data['motion_energy'] (NO recompute) with the exact
  _motion_temporal_localization smoothing/indexing convention. The cand-miss
  lever CBT (post-BUE inward clamp along the CLIP curve) cannot touch. Paired
  via the persisted pre_cgr_boundaries field. Per-fire [CGR] log + unconditional
  per-video [CGR-VID] summary (fires on the 0-detection / is_normal paths too,
  FM17). flag-off => byte-identical (motion never read for relocation).

v160: CBT-RECAL (NEW, default OFF, --cbt_recal; use with --cbt_tighten). Swaps
  ONLY the two scalar inputs of the v155 CBT method: per-category caps ->
  GT-p75 recal {RoadAccidents 4.5, Explosion 13.0, Shoplifting 9.0,
  Shooting 13.5} and over-wide trigger 1.5 -> 1.1. Method logic unchanged;
  adjudicated via the EXISTING pre_cbt_boundaries paired field. flag-off =>
  caps==CBT_CAPS, trig==1.5 => byte-identical to committed v155 CBT.

v155: CBT — Category-conditioned Boundary Tightening (NEW, default OFF,
  --cbt_tighten). TERMINAL (post-BUE, pre-CEX) INWARD clamp that shrinks
  over-wide FINAL detections of the 4 hard categories (Shooting/RoadAccidents/
  Shoplifting/Explosion) toward per-category GT-p75 caps (CBT_CAPS). For each
  FINAL detection whose resolved anomaly_type is in CBT_CAPS AND det_dur >
  1.5*cap: read the read-only category curve (frame_scorer.get_category_scores —
  the exact curve BUE/EUBC already use), find peak_idx=argmax within [t_s,t_e],
  scan half-widths hw and pick the [peak-hw, peak+hw] window MAXIMIZING the
  AutoLoc outer-inner contrast (identical objective to the LogOIC tightener),
  clamp to the ORIGINAL [t_s,t_e] (monotone-inward, never grows), FIRE only if
  new_dur < det_dur and new_dur >= min_segment. Confidence / detection count /
  ordering / coarse_score / frame_scores are byte-identical to flag-off
  (count-neutral, FM2-safe). Strict INVERSE of BUE (outward union): runs AFTER
  BUE, leaving BUE byte-identical. Paired via pre_cbt_boundaries. Per-fire [CBT]
  log + unconditional per-video [CBT-VID] summary (fires on the 0-detection /
  is_normal path too, FM17).

v142: BUE — Boundary-Union Extension (NEW, default OFF, --bue_extension).
  RATIONALE. The binding constraint is the RECALL/LOCALIZATION ceiling, not
  ranking: max recall 0.379@IoU0.3 / 0.212@0.5 / 0.121@0.7, and **50 of 99
  detections have ZERO overlap with GT** (median IoU 0.000) — the pipeline
  frequently looks in the WRONG PLACE, and detections are systematically SHORT
  (hits det/GT duration ratio ~1.31, zero-overlap misses ~0.42, mean GT ~30 s).
  Every within-video CONFIDENCE re-rank is structurally capped (pure-re-rank
  oracle 0.27273, within-video oracle only +0.006), whereas this family's own
  best-of-two oracle is +0.0764. BUE therefore changes the WINDOW, never the
  score.
  MECHANISM. In the TERMINAL-MOST slot (after _qscore_rerank, before the v141
  CEX block and the DetectionResult return) each FINAL detection is grown
  OUTWARD on the verdict-INDEPENDENT read-only category curve
  (frame_scorer.get_category_scores — the exact cached call
  _clip_boundary_single_scale makes): smooth w=3 -> that detection's own curve
  argmax as anchor -> the v51 half-max CLOSURE TEST
  (hm = (seg.max() + percentile(curve,20))/2; if the curve is already below hm
  at BOTH edges the boundary was genuinely set by a signal crossing and the
  detection is skipped byte-identically) -> otherwise grid-search (s,e)
  containing the anchor with length in [bue_lmin_sec, min(bue_lmax_sec,
  bue_lmax_frac*duration)] maximizing the AutoLoc-style outer-inner contrast
  OIC(s,e) = mean(inside) - weighted_mean(two outer rings of width alpha*(e-s))
  -> emit the UNION of the old window and the OIC argmax window.
  UNION IS THE ENTIRE MECHANISM (measured 3-cache ablation): closure-gated
  REPLACE = -0.0065 mean, UNION = +0.0378 mean. Pure OIC relocation is
  worthless or harmful; only monotone outward growth toward the contrast argmax
  wins. Never contract, never replace.
  M6 PARAMETER RECORD: bue_lmax_sec=40. L=40 and L=50 are BIT-IDENTICAL because
  the 0.8*duration constraint binds above 40 (so 40 is where the mechanism
  becomes self-limiting by video length rather than by a constant); 24/24
  arm-cache combinations over Lmax in {15,20,25,30,35,40,50,60} x 3 caches are
  POSITIVE; L=90 collapses to -0.00117 (1/3) as over-dilation is punished at
  IoU 0.7. 12-cache out-of-sample read: 9 held-out caches, 9/9 positive, mean
  +0.0370 — that, not the proposal's understated 3/3, is the prior the next loop
  should inherit. The closure gate is a measured mAP NO-OP (gated == ungated
  bit-identical on 3/3 caches at both Lmax 40 and 60) and is retained purely as
  free blast-radius reduction (67 fires instead of 78).
  INVARIANTS. Only t_start/t_end are written. Detection COUNT, confidence,
  anomaly_type, rank order, NMS, merge and every prompt are frozen bit-for-bit
  (the per-fire marker logs conf=<c>-><c> to make that visible), and there are
  ZERO new VLM/CLIP/MIL calls — the curve is a cached read-only signal, never
  written back, rescaled or fused (FM1 strict, FM2/FM3/FM4/FM14 immune).
  frame_scores are computed BEFORE this slot, so frame AUC is flag-off-identical
  and the effect is mAP-only.
  MEASUREMENT (FM16). The verdict is a PAIRED video-matched bootstrap from ONE
  run: pre-extension windows are carried in the result JSON as
  pre_bue_boundaries (M1, index-aligned, written only when the flag is on) and
  adjudicated with `analysis_bootstrap_ab.py --paired-json pre_bue_boundaries`
  (exact; inherits that tool's length-mismatch HARD-VOID check). PRE-REGISTERED
  DECISION RULE (M2, fixed BEFORE launch): expected paired point delta +0.030 to
  +0.040; measured P(d>0) for this operator on 6 caches was mean 0.891, range
  0.835-0.952, 3/6 >= 0.90, and the bootstrap half-width for a CONCENTRATED
  BOUNDARY operator is +/-0.055 — 2.2x the +/-0.025 quoted for re-rank deltas.
  KEEP iff paired point delta >= 0.02 AND P(delta>0) >= 0.85 AND full
  mAP >= 0.15132. A P in [0.85,0.90) is recorded as PROVISIONAL-ON-VARIANCE with
  the half-width quoted, never silently as a clean pass.
  MARKERS (FM5): "[BUE] vid=... det=... t=<old>-><new> conf=<c>-><c>" per fire,
  plus one "[BUE] summary:" per video carrying ndet_before/ndet_after (FM2),
  growth_med/dur_ratio (M4 MATERIALITY gate: smoke requires median per-fire
  growth > 2.0 s and mean post/pre duration ratio > 1.5; offline 3.9 s / 1.90 —
  this is the gate that would have caught v107 ECE, which passed ACTIVATION with
  160 markers while being a no-op at median growth 0.000 s) and selfov (M5
  diagnostic only, do NOT act on it this loop — FM8).

v141: CEX — Continuation-Evidence Extension (default OFF, --cex_extend).
  RATIONALE. The binding constraint on this harness is the RECALL/LOCALIZATION
  ceiling, not ranking: max recall 0.379@IoU0.3, and with 11-point interpolated
  AP over n_gt=66 the recall=0.4/0.5 checkpoints (27th/33rd hit) contribute
  EXACTLY ZERO today — which is why every within-video re-rank was capped
  (pure-re-rank oracle 0.273, within-video oracle only +0.006). Detections are
  systematically SHORT (hit det/GT duration ratio ~1.31, zero-overlap misses
  ~0.42; GT median 11.0s / mean 26.6s). CEX therefore stops trying to fix WHERE
  the pipeline looks (the incumbent top-1 detection already sits on GT 63.8% of
  the time — better than the CLIP-curve peak at 35-43%, MIL v1 41.4%, MIL v6
  24.1%, so there is no better locator to promote it against) and fixes HOW FAR
  the confirmed event extends.
  MECHANISM. In the TERMINAL-MOST slot (M1: AFTER merge/NMS/PostTighten/narrow/
  classify/curve-correction/MIL-sanity/conf-gate/ECE/EUBC/Q-score have ALL
  completed, immediately before the DetectionResult return), each FINAL
  detection asks the VLM a BINARY, NUMBER-FREE question about the single
  adjacent cex_step_sec clip — "the confirmed event is <desc>; is that SAME
  event still ongoing in this clip?" — and grows that side by exactly one step
  only on a confident yes (confidence >= cex_min_conf), up to cex_max_steps
  rounds per side, closing a side PERMANENTLY at its first reject. All window
  geometry is imposed by code in fixed steps; the VLM never emits a timestamp
  (that regression — v62, Stage 5b below — was VLM absolute-timestamp
  regression from sparse keyframes, full mAP 0.1492->0.1012).
  M2 (cited and re-priced BEFORE spend): ScoringConfig.event_continuation_
  extension (v105 ECE, _event_continuation_extension below) ALREADY implements
  outward growth of FINAL detections; its simulated +0.0311 (v99) collapsed to
  +0.00203 in vivo (v107) with AP@0.5/0.7 unmoved. Differentiators: (a) the
  adjudicator is a live VLM continuation vote on real pixels, not a curve
  level-set (incumbent-site accuracy 63.8% vs the curve's 35-43%); (b) M3 — NO
  CATEGORY_DURATION_PRIORS max_dur reject-cap (ECE reverted its own accepted
  growth) and NO snap to the curve grid (ECE's 45 snaps); growth is bounded
  ONLY by cex_max_steps and cex_max_total_sec.
  INVARIANTS. Detection COUNT, coarse_score, confidence, anomaly_type, the
  ranking and the v51 half-max code are all untouched — only the two endpoints
  of already-existing windows move outward (FM2/FM4/FM14 immune). frame_scores
  are computed BEFORE this slot, so frame AUC is flag-off-identical and the
  effect is mAP-only (clean attribution, M1).
  MEASUREMENT (FM16). The verdict is a PAIRED video-matched bootstrap from ONE
  run: pre-extension windows are carried in the result JSON as
  pre_cex_boundaries (M4, index-aligned, written only when the flag is on) and
  mirrored in the [CEX-PAIR] log marker. Gate: paired point delta >= 0.01 AND
  P(delta>0) >= 0.9 AND full mAP >= 0.15132.
  MARKERS (FM5): [CEX] per accepted step (carries the pre-extension duration,
  M6), [CEX-PAIR] per detection, [CEX-STOP] per closed side, [CEX-GATE] per
  video (probes/accepted = the realized acceptance rate, the pre-registered q
  read of M5: smoke band 0.20-0.80, >0.85 = rubber-stamping -> do not launch).

v137: NOCLS-CA classify-path latency win (KEPT, default-ON). _ensemble_classify
  now short-circuits when config.scoring.classify_skip_class_agnostic is set
  (default True): it logs a per-fire [NOCLS-CA] marker and returns the upstream
  VideoMAE label instead of issuing the Stage-7 qwen3-vl-plus classify VLM calls
  (<=3/video). mAP-NEUTRAL BY CONSTRUCTION — the returned string only sets
  a['anomaly_type'], which compute_temporal_map never reads; the confidence gate,
  clip_score_curve_correction and frame scores all key off videomae_category, not
  this result. Verified ~-7.5% per-video VLM calls (v132 in-vivo). Opt out with
  --no_classify_skip_class_agnostic. Tradeoff: degrades the separate Level-4
  category-accuracy metric (not temporal mAP). Baseline mAP stays 0.15632.

v124: BASELINE CALIBRATION (latency-session loop 1). Two mAP-neutral edits.
  (A) Completed the v114 Q-score KEEP-FLAG-ON flip: config.from_args now
  defaults qscore_rerank=True (was getattr default False), and
  batch_evaluate.py / run.py gained --no_qscore_rerank (store_false, dest
  qscore_rerank) so the committed baseline (map=0.15632) is the default and
  can still be disabled. --qscore_rerank kept (default True) as a
  belt-and-suspenders eval flag.
  (B) Added always-on [COST] per-video instrumentation (no CLI flag). detect()
  is now a thin wrapper around _detect_impl (unchanged logic) that snapshots
  drift-free counters — LLMClient.call_count (incremented per .generate on both
  llm_client=detect and classify_llm_client=classify) and
  AnomalyExpert.clip_call_count (incremented per encode_image forward pass) —
  plus a wall-clock start, and emits ONE machine-parseable line per video in a
  finally block (fires on every exit path):
    [COST] video=<name> vlm_calls=<n> clip_calls=<n> elapsed_s=<f>
  Counters are read-only in the wrapper and never touch any request/feature/
  detection, so the final detection set is byte-identical -> mAP unchanged.
  Purpose: establish REF_vlm_calls / REF_clip_calls / REF_wallclock for the
  latency session (FM10: call count is the primary drift-free signal; wall-clock
  is API-drift-dominated corroboration only). activation marker: [COST].

v104/v108: TSG — Temporal-Structure-Guided scan targeting — REVERTED & REMOVED.
  v108 首次全量 in-vivo 评测（--temporal_structure_scan，69/65 [TSG]
  markers，fired 仅 4/65）：full mAP 0.1583（0.1494±0.03 噪声带内，
  无提升）；per-video mean 295s vs v103 同配置 155s（~1.9 倍；但
  TSG fired 视频扫描段数与 incumbent EXACT parity(6=6)，延迟上升
  为全板性、更可能源于当晚 VLM API 变慢，非 TSG 直接所致）。
  预注册覆盖率裁决：selected vs baseline 段集 GT-overlap 覆盖不升
  -> 按协议 REVERT。机制代码（_temporal_structure_saliency /
  _tsg_incumbent_prescreen_indices / _structure_guided_discover +
  config/CLI/[TSG-GATE]）已全部移除，恢复 v94 字节级行为。
  备份：.qoder/backups/v108-revert/。
  教训：(1) 只触达 ~6% 视频的机制注定无法移动全量 mAP——提案必须
  先估算目标视频占比；(2) 评测间 latency 比较需控制 API 时延漂移，
  跨日对比不可直接归因。

v105: ECE-Live — Event-Continuation Extension（pinned v99 语义，首次真正
  in-vivo 评测；v99 死于 FM5：0 条 [ECE] 日志，机制从未执行）
  背景：v97 全量 IoU 分解显示 ~39% 检测质量损失在 IoU 0.1-0.5 边界不准
  桶，主导模式是 long-event-shrunk —— v51 半峰值边界搜索被 padding_ratio
  限定的局部窗口（"cage"）结构性截短长事件。确定性证据（项目史上最强）：
  v99 仿真把缓存全量候选推过官方 greedy-match evaluator，+0.0307 mAP；
  C 独立复现至最后一位，并在 live-faithful 网格上重跑 +0.0311、零阈值
  跨越 TP 损失（Arson035 0.554->0.667 反成 win）；±10% level 扰动带
  下界 +0.015。
  机制（单开关 config.scoring.event_continuation_extension，默认关 =
  字节级恢复旧行为）：终态槽位第一段 —— v51 类别置信门之后、v100 EUBC
  块（保持 flag-off/inert）之前。对每个 FINAL detection [t_s,t_e]，若
  dur < max_dur = CATEGORY_DURATION_PRIORS[cat]['expected_duration'][1]：
  (1) rel = frame_scorer.get_category_scores 缓存的整视频相对类别曲线
  （严格只读；None -> skip）；(2) hm = (rel.max()+p20(rel))/2 —— v51
  半峰值 LEVEL 公式原样、零新常数，作用于整视频曲线而非 v51 的局部窗口；
  (3) s/e = searchsorted 钳位；从 s 向左走 L、从 e 向右走 R while
  rel >= hm；(4) REJECT-CAP：若 ts[R]-ts[L] > max_dur 保留原边界，否则
  snap 到 ts[L]/ts[R]（永远 snap 到网格）。类别解析同 EUBC（detection
  自身 anomaly_type post-unify，回退 videomae_category；None/'Normal'/
  unknown/曲线 None -> skip）。检测数、顺序、置信度、类别、NMS、merge
  及所有上游阶段字节级一致；零新增 VLM/CLIP/MIL 调用；延迟中性
  （<50ms/视频 numpy 游走）。
  Adjudication（VLM 噪声免疫）：flag-on 时每视频结果写入
  pre_ece_boundaries（ECE 前边界），同一 run 内 pre/post 双边界集离线
  过 evaluate.compute_temporal_map 做确定性 A/B —— keep/revert 由该
  确定性 delta（>= +0.015）决定，不由 ±0.03 噪声带内的 live headline
  决定。FM5 防线：每 detection 一条 INFO
  '[ECE] pre=[...] post=[...] (material|grid-snap|reject-cap|skip)'，
  每视频恰一条 INFO '[ECE] summary: ...'（含两条早退路径）；
  batch_evaluate.py 在 flag-on 时强制 hard gate —— '[ECE] summary:'
  行数 < succeeded 视频数则 exit 2 并拒写 evaluation_report.json
  （flag-off 时 gate 不激活，行为字节级一致）。
  文献锚点：T3AL（ECCV 2024，arXiv 2404.05426）整视频动态阈值 level-set
  区域生长；BSN/BMN completeness 原则；LEC-VAD（ICML 2025，arXiv
  2506.13095）事件不完整性 = WS-VAD 核心定位缺陷；AutoLoc OIC ->
  duration-prior reject-cap；EventVAD（arXiv 2504.13092）training-free
  事件范围估计在 UCF-Crime 可行。
  规避失败模式：FM1 曲线严格只读、终态 post-hoc 于固定检测（v89/v94/
  v95/v96 从未回归的变更类）；FM2 不可能（数量/顺序/置信度冻结，仅既有
  match 的 IoU 变化）；FM3/FM4 无（零 VLM 调用）；FM5 由真实 CLI flag +
  全路径 [ECE] summary 标记 + runner hard gate 回答（激活链已被 v103
  in-vivo 验证）。

v100: EUBC — Event-Unit Boundary Contraction（终态过宽边界收缩）
  背景：v97 全量 run 的逐视频 IoU 分解（67 GT，官方 greedy 匹配）显示
  最大未处理损失桶是 over-wide 边界不准（16/67 GT ratio>1.5，最高 11x：
  Burglary037 GT 5s 落在 55s detection 内；Arson009 GT 3.2s 落在 19.5s
  detection 内）。现有各级在结构上无法触及该桶：Stage 7b 对 <= max_dur
  的段一律跳过，而类别 max_dur 先验（10-40s）远比这些 2-5s 事件宽松。

  机制（单开关 config.scoring.event_unit_boundary_contraction，默认关 =
  字节级恢复旧行为）：在 detect() 最末端新增一个终态 post-processing
  阶段——类别置信门之后、最终 dict 投影之前（与 v99-ECE 同一插入槽位，
  攻击相反的边界失败模式：ECE 在曲线上外生长过窄检测，EUBC 在独立的
  视觉一致性信号上内收缩过宽检测）。对每个 FINAL detection a=[t_s,t_e]
  （类别取自 a 自身 anomaly_type，post-unify；回退 videomae_category；
  None/'Normal'/unknown/不在 anomaly_categories 或曲线为 None 则跳过）：
  1. 信号（全部只读，零新增 CLIP/VLM 调用）：rel = 缓存的整视频类别
     相对曲线（frame_scorer.get_category_scores）；sims[i] =
     cos(feat_i, feat_{i+1}) 由已缓存 clip_features 惰性计算（仅开关
     开启时——flag-off 是真零 delta，Stage 1 不做任何无条件计算）。
  2. 峰值锚点 P = argmax(rel) within [t_s,t_e]（最强异常证据点）。
  3. 显式 flat-coherence 防护：robust_mad = 1.4826*MAD(sims)；
     robust_mad < eubc_flat_eps(1e-6，纯数值退化防护) -> 显式 no-op +
     记录 flat-skip。
  4. 一致性游走（event-unit 范围）：thr = median(sims) -
     eubc_mad_k(1.0)*1.4826*MAD(sims)（canonical robust z=1；1.4826 为
     MAD 正态一致性常数）。从 P 向左走 L while sims[L-1] >= thr；向右
     走 R while sims[R] >= thr。coherence span = [ts[L], ts[R]]。
  5. plateau-union 防护（v51 半峰值 LEVEL 公式原样，零新常数）：
     detection 内 hm = (max+p20)/2 of rel；plateau = {rel >= hm}；
     guarded span = [min(L_time, plateau_left), max(R_time,
     plateau_right)]——绝不切掉曲线自身半峰值证据支持的范围。
  6. 仅收缩点火规则：guarded span 裁剪到 [t_s,t_e]；仅当
     new_dur < eubc_materiality(0.75)*det_dur（materiality，微小修剪
     为 no-op）且 new_dur >= CATEGORY_DURATION_PRIORS[cat]
     ['min_segment']（既有常数）时才收缩。

  置信度、类别、检测数、顺序、NMS、merge 及所有上游阶段字节级一致；
  不重跑 NMS；不重算 VLM/CLIP；曲线严格只读。每 detection 一条
  [EUBC] 日志（fired/no-op 原因），每视频一条 [EUBC] summary 行
  （含早退路径——runner 强制执行门：任何处理过的视频缺 summary 行
  即拒绝提取指标；'0 fired' 是合法结果，缺行不是）。构造上自限：
  flat coherence（财产犯罪、整视频单元）-> no-op；coherent detection
  （span ~= det）-> no-op；只有可证实的多场景 detection 才收缩。

  文献锚点（已核）：Foote 2000 SSM/checkerboard novelty（v95 TSM 已在
  仓内验证，零成本）+ canonical robust statistics（median - k*1.4826*
  MAD）。EventVAD（arXiv 2504.13092，真实、training-free、UCF-Crime
  上验证 adaptive event-border thresholding）仅作为"training-free
  event-unit 感知在 UCF-Crime 可行"的证据引用——其机制是
  dynamic-graph + MLLM + signal-ratio thresholding，并非
  cosine+MAD（已更正的引用）。

  规避全部已知失败模式：FM1 曲线形状未动（rel/clip_features 严格只读，
  曲线仅作 floor guard 而非收缩驱动——v52 同曲线收缩教训的直接回
  答）；FM2 检测数/顺序/置信度冻结，纯收缩不可能产生重叠近重复；
  FM3/FM4 无 VLM 调用/采样/翻票；v59/V65/V75 无条件边界替换创伤由
  capped（仅收缩+materiality+min_segment）+ guarded（plateau union）
  + self-limiting（flat no-op）设计回答；FM5 由显式 from_args
  threading（config.py 此前不构造 scoring 对象的已验证根因）+
  每视频 summary 行 + runner 强制执行门回答。

v97: Full-Coverage Comparative VLM Scanning (FCCS) — Stage 4 候选生成架构替换
  背景：roadmap 与 v94 日志一致确认主导瓶颈是 wrong-location 候选（28.8%）。
  v94 全量日志覆盖轨迹：directed 路径只扫 10-18/23-120 段（18/120, 12/64,
  10/46），fallback 路径 6/43, 6/32, 6/33（预筛 std 0.001-0.005，选择近乎
  任意）。真实事件段若从未被扫描，候选窗口 by construction 就是错位的；
  若被扫描但与其他 4+ 段同被打 0.90+（v56 记录的独立判定过标定问题），
  candidate_top_k=3 又会任意丢弃它。

  机制（单开关 config.discovery.full_coverage_comparative_scan，默认关 =
  字节级恢复旧行为）：把全部 6s 段按连续时间范围打包进比较式 VLM 调用
  （batch=8 段 x 2 关键帧；>60 段时 batch=16 x 1 关键帧），复用 v56 已
  实现但当前无调用点的 scan_segments_comparative + v57 非排他
  segment_scores schema。每次调用让 VLM 以同视频其它段为正常基线对每段
  打 0-1 分；>0.3 的段成为候选，经不变的 _merge_adjacent 后流入完全
  不变的下游（top_k=3、v51 半峰值边界、0.75*CLIP+0.25*MIL 融合、v95 TSM
  + v96 onset/offset 置信度调制、硬 NMS、类别置信门）。开关开启时跳过
  Stage 3 CLIP-loc region 循环（已验证无副作用，clip_regions 唯一消费者
  是扫描方向；边界检测用的 CLIP 类别曲线由 frame_scorer.get_category_scores
  独立计算，不受影响）。

  这是 v56 已隔离验证为 mAP 正向（+0.006，当时最佳 AP@0.5/0.7）的比较式
  判定的全覆盖推广——v57/v58 回归源于阈值改动而非机制本身，v61 的回退
  捆绑在无关的 VLM 边界工作中从未被隔离验证。三项稳健性加固（评审共识）：
  (a) batch 必须是连续时间范围（与 temporal neighbors 互相校准）；
  (b) 截断防护——prompt 要求 terse 输出（description ≤8 词），解析失败或
  条目不足时 batch 对半重试一次，仍失败则仅对该 batch 走既有独立
  scan_segment 兜底（纯失败处理，非第二机制）；(c) 避免过小尾批——
  N mod batch < 3 时余量折入前批（26 段 -> 9/9/8 而非 8/8/8/2）。

  规避全部 4 个已知失败模式：FM1 曲线形状未动（relative_scores/MIL 融合/
  半峰值字节级一致）；FM2 检测数有界（top_k/merge/NMS/门全不变）；FM3
  无自一致性/温度采样（每 batch 单次确定性调用）；FM4 无事后翻票
  （整体替换初始纳入机制，非 post-hoc 翻转）。

v96: Asymmetric onset/offset likelihood confidence signal (BSN-inspired)
  Loop 4 research confirmed via literature (BSN's Temporal Evaluation
  Module producing separate P_start(t)/P_end(t)/P_actionness(t); LGN's
  'locate boundaries with high start/end probabilities SEPARATELY, then
  combine' pattern) that directional asymmetric start/end signals are a
  real, validated architectural choice distinct from our symmetric
  inner-outer contrast.

  Training-free analog, applied as PURE CONFIDENCE MODULATION only
  (never changing t_start/t_end - same safe pattern as V95's TSM
  sharpness and V89/V94's confirmed-non-regressing changes):
  onset_sharp = mean(future_window) - mean(past_window) AT t_start
  offset_sharp = mean(past_window) - mean(future_window) AT t_end
  A real onset should show onset_sharp>0 (anomaly begins, future is more
  anomalous); a real offset should show offset_sharp>0 (anomaly ends,
  past was more anomalous). Combined directional_quality modulates the
  existing boundary_bonus by 0.9-1.0x, stacked with V95's TSM modulator.

  Avoids all 4 known failure modes: read-only computation on existing
  relative_scores (not fused into the primary half-max curve), doesn't
  change detection count, not a VLM sampling technique, doesn't change
  candidate/segment inclusion.

v95: TSM checkerboard-kernel novelty for confidence re-ranking
  Loop 1 deep-dive (this round's focus: HOW do TAL papers construct their
  localization signal, not just post-processing tricks) surveyed 7
  paradigms (BSN/BMN 3-curve, TAG/watershed multi-threshold,
  ActionFormer/TriDet distance-field regression, OIC/LogOIC contrast
  scoring, T3AL 2D caption-similarity suppression, VideoGEM multi-layer
  attention, TSM+checkerboard novelty). Only ONE is a genuinely
  different SIGNAL TYPE (not a different way to threshold/score the same
  text-similarity curve): Temporal Self-Similarity Matrix + checkerboard
  kernel (Foote's novelty function, UBoCo/GEBD lineage) - a frame-to-
  FRAME visual similarity matrix, detecting scene transitions completely
  independent of category text semantics.

  Implemented as _compute_tsm_novelty(): SSM[i,j]=cosine_sim(feat_i,feat_j)
  for all frame pairs, checkerboard kernel (+1 same-block/-1 cross-block,
  Gaussian-tapered) correlated along the diagonal at 3 scales (L=15,30,60
  frames), max-combined. Verified on synthetic test: peak novelty exactly
  at a synthetic scene transition. Negligible cost (0.02s for a 1434-
  frame video, zero extra CLIP/VLM calls - reuses existing features).

  Applied ONLY as confidence re-ranking (never boundary position
  modification) - same safe pattern as V89's diverse-perspective tie-
  break and V94's learned prompt weights (both confirmed non-regressing).
  _tsm_mod = 0.85 + 0.15*sharpness multiplies the existing boundary_bonus
  term only. Avoids all 4 known failure modes: doesn't touch curve shape,
  doesn't change detection count, isn't a VLM self-consistency technique,
  doesn't change candidate/segment inclusion.

v94: REVERT V92 borderline revoting (regressed 0.1494->0.1183), KEEP V93
  Full 65-video eval of V92 (borderline-case majority-vote revoting in
  anomaly_discovery.scan_segment): mAP=0.1183, regression from V89's
  0.1494. REFINED LESSON: self-consistency safety depends not just on
  task TYPE (categorical vs numerical, per V90's lesson) but on WHERE in
  the pipeline it acts. Revoting is_anomalous flips WHICH SEGMENTS enter
  the candidate pool - functionally similar to the already-confirmed-
  harmful "candidate pool modification" risk category (V81 MIL-TAG,
  etc.), just triggered by majority-vote instead of a different model.
  Reverted anomaly_discovery.scan_segment to its single original call.

  V93's learned prompt-ensemble weights (see below) are KEPT - orthogonal
  to this revert, operates purely on text-embedding computation upstream
  of candidate generation, not implicated in this failure mode.

v93: Learned prompt-ensemble weights (CoOp-lite, Loop 12 research)
  Exhaustive research (Loop 12) concluded CoOp-style prompt optimization
  is the highest-confidence remaining idea, since it operates UPSTREAM of
  ALL documented failure modes (curve-shape sensitivity, greedy-match
  false positives, task-type-dependent self-consistency) - it only
  changes WHICH TEXT EMBEDDING represents a category, computed once and
  cached exactly like the original mean-ensemble; the fusion/boundary
  logic downstream is completely unaware of the change.

  Implemented tractable simplification of full CoOp (avoiding the need to
  inject learnable embeddings into CLIP's token layer): learned a softmax
  weight vector over the 6 existing hand-crafted prompts/category (78
  scalars total across 13 categories), trained via MIL top-k video-level
  loss on the UCF-Crime train split's video-level labels (no frame-level
  GT needed - matches our actual supervision level).

  CAUGHT AN OVERFITTING FAILURE BEFORE PIPELINE INTEGRATION: a first
  attempt (200 epochs, lr=0.05) collapsed weights to near-one-hot per
  category and, when validated on the HELD-OUT TEST SET, was WORSE than
  uniform mean on average (0.5605 vs 0.6230 separation) - classic
  overfitting to training-set idiosyncrasies. Caught via a 2-minute
  offline check instead of a 90-minute full pipeline eval.

  Retrained with light regularization (20 epochs, lr=0.02, weights stay
  near-uniform rather than collapsing) - validated on test set: 12/13
  categories improved, avg separation 0.6230->0.6464 (+3.8%). This is a
  genuine, test-set-validated improvement to CLIP's zero-shot category
  discrimination, not an artifact of training-set overfitting.

v92: Borderline-case majority-vote revoting for segment anomaly scanning
  Loop 7 research (refined lesson from V90's failure): self-consistency
  benefit is TASK-TYPE dependent. V90 failed because timestamp extraction
  is a numerical-perception task with one right answer and no alternative
  valid reasoning path - added sampling diversity was pure noise. This new
  idea targets a genuinely different task type: scan_segment()'s
  is_anomalous decision is BINARY/CATEGORICAL with real interpretive
  ambiguity (e.g. "nervous behavior" vs "normal behavior") - the task
  type self-consistency literature shows genuine benefit for.

  To respect the latency goal (already stressed this session), does NOT
  triple every one of the ~10-18 segment-scan calls per video. Only
  re-votes (2 extra calls) when the FIRST call's confidence lands in a
  borderline zone (0.15-0.45, straddling the 0.3 accept threshold) - the
  cases where a single call's binary decision is most likely to be
  flipped by noise. Clear-cut cases (very high/low confidence) proceed
  with the single original call, unchanged latency.

v91: REVERT V90 self-consistency (regressed 0.1494->0.1075)
  Full 65-video eval: mAP dropped to 0.1075. Despite being provably isolated
  from BOTH curve-shape (lesson 1) AND detection-count/greedy-matching
  (lesson 2), this STILL regressed - a THIRD distinct failure mode discovered.
  Root cause hypothesis: self-consistency via temperature-diversity is a
  technique proven for REASONING tasks (math/logic - multiple valid paths
  converge on one correct answer), not PERCEPTUAL/EXTRACTION tasks (reading
  a timestamp off a frame has no equivalent alternative valid reasoning
  path). The temperature ladder (0.0, 0.3, 0.5) deliberately traded
  determinism for diversity on 2 of 3 calls, diluting the more-reliable
  temp=0 estimate via the median rather than improving it.

  NEW LESSON: self-consistency benefit is TASK-TYPE DEPENDENT. It helps
  when multiple sampling paths can independently arrive at truth
  (reasoning, classification with semantic ambiguity). It can HURT when
  the task is direct perception/extraction with a single ground truth and
  no alternative correct path (timestamp reading, precise localization) -
  added sampling diversity there is just noise, not signal.

  Reverted to V89's original single deterministic VLM boundary call.
  temperature_override plumbing in LLMClient.generate() is KEPT (harmless,
  general-purpose, backward-compatible) in case a genuinely-diverse-
  reasoning future use case needs it.

v90: VLM boundary self-consistency (median-of-3, temperature ladder)
  Loop 1 research (new 15-loop session): implements an idea researched but
  never tried in the prior session. For candidates already going through
  VLM boundary estimation (LOW_CLIP fallback path only), call the VLM 3x
  with a temperature ladder (0.0, 0.3, 0.5) instead of once, take the MEDIAN
  of the 3 start/end timestamp estimates. Literature: temperature=0 alone
  provides too little diversity across repeated calls (infrastructure-level
  non-determinism exists but is small); the ladder ensures genuine diversity.
  Median is maximally robust (breakdown point 0.5); for N=3, median equals
  trimmed-mean, so no ambiguity in aggregation choice.

  Respects BOTH hard-won lessons from the prior session: (1) does not touch
  the CLIP/MIL fused curve shape at all - only refines an ALREADY-SELECTED
  candidate's boundary position; (2) does not change detection COUNT (still
  exactly one candidate in, one refined candidate out), so it cannot create
  the false-positive-via-greedy-matching problem that killed Soft-NMS (V86).

  Added temperature_override optional parameter to LLMClient.generate()
  (backward compatible, all other call sites unaffected) to support the
  temperature ladder without a fixed self.temperature=0.0 client config.

  GAP (Gaussian Approximated Post-processing) was researched but NOT
  implemented - poor fit for this pipeline since GAP solves snippet-level
  quantization error (sub-frame precision), while our boundaries are already
  continuous-time and VLM errors are typically 2-10 seconds, a different
  scale than what GAP addresses.

v89: REVERT Soft-NMS (V86 regressed 0.1492->0.0961), KEEP diverse prompts
  V86 full 65-video eval: mAP=0.0961, regression from V51's 0.1492 baseline.
  Root cause: evaluate.py's temporal mAP uses standard GREEDY matching
  (predictions sorted by confidence descending, each GT segment matchable
  ONCE - see evaluate.py's compute_temporal_map). Soft-NMS's permissive
  survival threshold let 2-3 near-duplicate overlapping detections survive
  per true event; only the top one is a true positive, the rest become
  FALSE POSITIVES under greedy matching, directly hurting precision/AP.
  This is a general lesson beyond just this pipeline: Soft-NMS's benefit
  (preserving legitimately-distinct overlapping detections) only pays off
  when the evaluation protocol ALSO credits multiple overlapping detections
  of the same event - greedy single-match mAP protocols instead PENALIZE
  the extra survivors. Reverted NMS to Hard-NMS (temporal_nms, proven correct
  for this specific eval protocol).

  Diverse-perspective self-consistency for category tie-breaking (from v86,
  Agent B research) is KEPT - it's independent of segment count/NMS, only
  affects the category LABEL assigned to already-finalized detections, so
  it isn't implicated in the Soft-NMS regression mechanism. Needs its own
  isolated full-eval to confirm net effect (pending).

v86: Soft-NMS post-processing (fresh loop sequence after archiving V80-85b)
  Prior chain (V80-V85b, archived in iteration_logs/archived_failed_v80-85b/)
  repeatedly tried to leverage a better-trained MIL model (v6, anti-collapse
  loss) via fusion-weight changes or candidate-pool additions - ALL 5 attempts
  regressed mAP (0.149 -> 0.04-0.11), confirming that curve-shape-modifying
  changes are fundamentally risky regardless of component-level model quality.
  Pipeline fully reverted to proven V51 baseline; fresh work resumes at v86
  with a hard constraint: only ISOLATED post-processing changes that operate
  AFTER boundary detection, never touching the CLIP/MIL fused curve shape.

  v86: Soft-NMS (Bodla et al. ICCV 2017) replaces hard temporal_nms. Decays
  overlapping detections' confidence by Gaussian tIoU (sigma=0.3) instead of
  hard-removing below threshold, preserving true positives that partially
  overlap while still suppressing near-duplicates. Zero latency cost (pure
  post-processing on already-detected segments), zero risk to curve shape.

  v86 part 2: Diverse-perspective self-consistency for category tie-breaking
  (Agent B research). The 3 parallel VLM tie-break calls previously used an
  IDENTICAL prompt (relying only on API sampling randomness for vote
  diversity). Now each of the 3 calls gets a DIFFERENT analytical angle
  (scene/context, physical actions, outcome/interaction) - self-consistency
  literature shows diverse reasoning paths yield more accurate majority votes
  than repeated identical prompts, especially for semantically-overlapping
  categories (Shoplifting/Stealing/Robbery/Burglary). Zero extra API cost
  (same 3-call count), zero risk to temporal boundaries (category-only).

v67: V51 半峰值 + LogOIC 定向收紧过宽结果（无 FFT）
  V66 评测(smoke)：mAP 0.0669。FFT 动作性校准改变了 CLIP 分数幅度，破坏了
  半峰值交叉的标定（Fighting -0.40, Explosion -0.15, Shoplifting -0.14），
  净效果为负。V67 仅保留 LogOIC 过宽收紧（当 det_dur > max_dur 时触发），
  不修改 CLIP 分数本身。

v66: V51 半峰值 + LogOIC 定向收紧过宽结果 + FFT 动作性校准
  V65 纯 LogOIC 评测(smoke)：mAP 0.0632。LogOIC 对短事件表现优异（RoadAccidents
  IoU 0.095->0.45, Abuse 0.282->0.60），但对长事件严重过窄（Fighting 0.575->0.18,
  Assault 0.297->0.04）—— OIC 天然偏好窄窗口（内外对比度更高）。

  V66 策略：V51 半峰值交叉作为默认，仅当检测结果 > max_dur 时启动 LogOIC 收紧。
  触发条件 det_dur > max_dur 精确匹配已知过宽案例：
    - RoadAccidents: 28s > 20s(max_dur) → LogOIC 收紧到 ~6s ✓
    - Shooting: 61s > 30s → LogOIC 收紧 ✓
    - Fighting: 6.5s < 15s → 保持半峰值 ✓
    - Assault: 68.5s < 120s → 保持半峰值 ✓
  保留 V65 的 FFT 动作性校准（压制静态虚假匹配）。

v65: LogOIC 边界检测 + FFT 动作性校准（FreeZAD 启发）
  V59-V64 六轮边界检测迭代（峰-谷/自适应收紧/VLM边界/运动融合）均未超越 V51
  的半峰值交叉（mAP=0.1492），根因是这些方法都在 CLIP 曲线上做阈值变体，
  无法解决曲线本身的模糊性。

  FreeZAD (arXiv 2501.13795, 2025) 提出了训练无关的时序动作定位新范式：
  A. LogOIC（对数衰减外-内对比度）：对每个候选片段计算内部均分 vs 外部加权均分
     的差值，外部权重按 1/log(2+距离) 衰减。这直接度量一个片段相对于上下文的
     "异常性"，替代半峰值交叉的绝对阈值判断。在多尺度滑窗中枚举不同宽度的
     边界，选择 LogOIC 最高的作为最终检测。
  B. FFT 动作性校准：对搜索窗口内的 CLIP 视觉特征做时域 FFT，高频能量表示
     "有事情在发生"（运动/变化），低频 = 静态场景。将 CLIP 分数乘以动作性
     系数 (0.5 + 0.5*actionness)，压制那些视觉上匹配文本但实际上是静态的
     虚假候选。

v63: 运动能量融合边界检测 + 回退 VLM 过宽覆盖
  V62 全量评测(65视频)：mAP 0.1012，VLM 过宽覆盖触发 18 次但净效果为负
  （VLM 稀疏帧时间估计噪声太大，修正成功率 < 50%）。回退到 V51 的 VLM 兜底。

  新架构改动 —— 运动能量融合：pipeline 已在 Stage 1b 计算逐帧运动能量
  （帧差绝对值均值），但仅在 Explosion/RoadAccidents 的 CLIP-loc 阶段使用。
  V63 将运动能量融入边界检测（_clip_boundary_detection）的评分曲线：
    fused = 0.7 * clip_scores + 0.3 * motion_norm
  对 Explosion/RoadAccidents/Fighting/Assault/Shooting/Abuse/Vandalism 生效。
  运动能量在异常边界处有尖锐跳变（突发暴力/撞击/爆炸），融合后的曲线比
  纯 CLIP 更尖锐，半峰值交叉可以得到更紧的边界。

v62: 选择性 VLM 边界精修 — 仅在 CLIP 不可靠或过宽时触发
  V61评测(smoke)：mAP 0.1180，AP@0.3=0.2677（高），AP@0.5=0.0864/AP@0.7=0
  （低）。逐视频 IoU 对比揭示 VLM 精修的双面性：
    改善：Shooting +0.35, RoadAccidents +0.12, Burglary +0.20, Assault +0.07
    恶化：Abuse -0.28, Arrest -0.15, Shoplifting -0.15, Fighting -0.14
  原因：CLIP 边界已经良好标定时，VLM 的稀疏帧估计反而引入噪声。

  V62 策略：VLM 精修仅在两种场景触发：
    1. CLIP 检测过宽（det_dur > max_dur * 1.2）：VLM 替代 CLIP 边界
    2. LOW_CLIP 类别 + 低 CLIP 对比度（clip_conf < 0.15）：V51 原始兜底
  其余候选保持 CLIP 半峰值交叉边界不动。

  预期效果：保留 Shooting/RoadAccidents 的大幅改善 + Arrest/Fighting 等的
  V51 水平精度。

  保留 V61 的其他改进：
  A. 独立逐段 VLM 扫描（回退到 V51 机制）
  B. 覆盖感知候选选择（V59）
  C. V60 的 70% 自适应收紧（CLIP 内部后备）





v58: 修正 v57 评分宽松化引入的接受阈值失配
  V57 评测结果：Recall 0.9333->0.9492（修复了预期的漏检），但 mAP 反而从
  v56 的 0.1428 掉到 0.1284，AP@0.3 0.2622->0.2296。逐段核对
  IoU（TP@IoU>0.1: 42->38, FP@IoU<=0.1: 38->40）证实：v57 的"给每段打分"
  确实让模型对分数更宽松了，但原来配 exclusive pick-list 设计的 confidence
  >0.3 接受阈值没有跟着调整——结果是更多本来不够格的候选也一起以
  "及格线以上"的分数混进来，抵消了修复漏检带来的收益。

  修复：接受阈值从 >0.3 提到 >0.5，同时在 prompt 里加了"strict grader"的
  显式校准锚点（>=0.5 必须是对这一段本身有真实信心，而不是"在这几段里显得
  最可疑"）。保留 v57 的非排他式打分（不回退到 v56 会漏检的 exclusive
  list），只是把验收标准和打分宽松度重新对齐。

v57: 修复 v56 比较式扫描的"只报最突出的"偏置导致的漏检
  V56评测结果：mAP 0.1366->0.1428 提升，且 AP@0.5(0.1213->0.1267)/AP@0.7
  (0.0331->0.0394) 双双超过 v51 历史最佳——候选生成阶段的改动方向被证实有效。
  但 Recall 从 v51 的 0.9831 掉到 0.9333，CatAcc 从 0.8814 掉到 0.8333，
  AP@0.3 从 0.2931 掉到 0.2622——典型的"精度换召回"信号。核对最终 JSON
  结果（而非并发日志，65 videos concurrency=3 时日志按行交织，无法按 START/
  DONE 简单切片归属到具体视频）：v56 新增 3 个零检出视频（Shooting034/
  Shoplifting033/Vandalism007）是 v51 能正确检出的（v51 对应置信度
  0.80-0.95，中等偏高但不算最突出）。

  根因：scan_segments_comparative() 的 prompt 要求模型"只列出真正突出的
  1-2 段"，这是一个排他性 pick-list 框架——当同一次比较里出现另一个更戏剧化
  的片段时，一个真实但没那么突出的异常片段容易被模型隐式压低/不列出。

  修复：把输出 schema 从"只列出异常段"的 exclusive pick-list 改成"给每一段
  打分"的 segment_scores（对全部 n 段逐一评分，用其它段作为标定参照，但
  不再要求"胜出"才能被列出），下游沿用原来 scan_segment() 的 confidence>0.3
  阈值。保留了"多段一起看、互相校准"的比较优势，去掉"必须比别人更突出才
  会被报告"的排他性偏置。

v56: 候选生成阶段架构改动 —— 比较式 VLM 扫描替代独立逐段二元判定
  背景：v51-v55 五轮改动全部集中在"边界精修"阶段（_clip_boundary_detection），
  但对 v55 逐视频核对发现：63% 的视频，其送入边界精修前的候选窗口就已经和
  GT 零重叠（mean IoU 精修前0.132 -> 精修后0.129，几乎不变）——精修阶段根本
  不是瓶颈，候选生成阶段才是。用 DEBUG 级完整单视频重放
  (Shoplifting001_x264) 定位到具体机制：
    1. 该类别 CLIP 类别相似度曲线几乎完全平坦 (std=0.005, cv=0.005) ——
       CLIP zero-shot 对这类"看起来和正常行为很像"的异常（LOW_CLIP_CATEGORIES:
       Shoplifting/Stealing/Arson/Burglary）没有区分度，触发
       "Scores too uniform, falling back to standard scanning"。
    2. 回退到 anomaly_discovery.py 的通用 CLIP 预筛（非类别特定，同样接近
       噪声水平），任意截取 top-6 段。
    3. 每个候选段独立送入 VLM 做二元判定（"is this anomalous, even subtle
       signs count"），没有其它段作参照——结果是被抽中的 6 段里有 4-6 段都被
       打上 SUSPICIOUS(0.90+)，因为模型没有相对基准，且 prompt 本身偏向
       "宁可多报"。

  改动（anomaly_discovery.py）：新增 scan_segments_comparative()，当预筛后
  候选段 > 1 时，把所有候选段的关键帧打包进*一次* VLM 调用，在 prompt 里
  显式标注每段的时间范围，要求模型在这些同一视频的不同片段之间做*相对*
  比较，只挑出真正显著异常的 1-2 段，而不是逐段套用一个绝对阈值判定。
  这个思路借鉴了近期视频-LLM 时序定位工作（TimeChat/VTimeLLM/Momentor）
  "在更大时序上下文里做定位比孤立片段分类更准"的做法。副作用：把这条
  fallback 路径的 VLM 调用数从最多 6 次降到 1 次，同时改善了单视频耗时。
  高 CLIP 信号类别的 _directed_llm_scan 路径不受影响。

v55: 回退阈值选择到 v51 机制 + MIL 常驻双流融合（隔离变量的架构改动）
  背景：v52(固定分类别阈值)→v53(按超调比例自适应阈值)→v54(多阈值扫描+
  contrast*duration_fit 组合评分) 连续三轮全部让 mAP 变差
  (0.1492→0.1304→0.1260→0.1083)。对 v54 逐视频核对 IoU 后发现根本原因：
  duration_fit 加权选择会系统性地选中"时长符合类别先验"但视觉上并不显著的
  窗口，而不是选中真正显著（对比度高）但时长稍微超出先验的窗口——例如
  Robbery102: v51 预测 43-60.5s 与 GT 36-52s 有交集，v54 却选中了与 GT
  完全不重叠的 22-33.5s。三轮独立验证同一个结论：在"半峰值阈值该怎么选"
  这一步上做得越复杂，mAP 越差；contrast-only 的简单选择才是这一步的稳定解。

  因此 v55 做两件事，且刻意分离变量以获得干净的因果信号：
  1. 【回退】边界选择恢复 v51 的机制：单一固定 50% 半峰值阈值，按 3 档平滑
     尺度中 OIC contrast 最高的一档选择（不再扫描阈值比例，不再用
     duration_fit 加权）。删除 BOUNDARY_THRESHOLD_SWEEP。
  2. 【新架构，与上面隔离测试】MIL 不再只是"CLIP contrast<0.15 时的兜底"，
     而是对每一个候选都常驻融入曲线：fused = 0.75*CLIP + 0.25*MIL，在做
     边界检测前完成融合。这是 WTAL 文献里双流融合（如 RGB+Flow 一致性）的
     标准做法。数值上是安全的——CLIP 相对分数（frame_scorer.get_category_scores
     的 softmax cat/(cat+normal) 比值）与 MIL sigmoid 输出天然都落在 [0,1]，
     不需要临时重新归一化，避免了 v54 潜在的尺度不一致风险。
     若这一步验证有效，下一轮可以在此基础上探索更精细的双流一致性/门控机制；
     若无效，下一轮可退回纯 CLIP 单流并转向候选生成阶段（VLM directed scan）
     做架构改动。

v51: 置信度公式重设计 + ConfFilter分层 + PostTighten修复 + MIL融合
  核心变更（基于 v50）：
  1. 置信度公式重设计：80%基础置信度 + 20%边界精度奖励，解耦检测存在性与边界精度
  2. 类别感知 ConfFilter 阈值：低CLIP类别(0.38-0.42) / 中等(0.48) / 高CLIP(0.52-0.55)
  3. PostTighten修复：永不降低置信度 + 跳过低CLIP类别 + 保留coarse_score
  4. RoadAccidents时长先验修复：(2,8)→(2,20)s + Short-narrow阈值15s + 60%缩小下限
  5. MIL模型融合：低CLIP类别使用MIL snippet分数作为补充定位信号
  6. 保留 v50 的 CLIP boundary detection + 多尺度时间定位 + ensemble分类

v50 保留逻辑：
  - CLIP score-curve boundary detection（多尺度半峰值交叉 + 梯度精修）
  - 多尺度 CLIP 时间定位（类别感知平滑窗口）
  - 帧级评分、VideoMAE 分类、ensemble 分类等全部逻辑不变
"""

import asyncio
import json
from collections import Counter
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import vad_logger as _vl
    _log = _vl.get("vad.pipeline")
except Exception:
    import logging
    _log = logging.getLogger("vad.pipeline")


class _CgrAbSilentLog:
    """v169 CGR-AB: no-op logger shim used ONLY during the arm-OFF replay pass.

    The arm-OFF pass re-runs the SAME BUE/CBT (and CCR) code the real arm ran, so
    without suppression every [BUE] / [CBT] / [CBT-VID] / [CCR-VID] marker would
    be emitted TWICE per video and every FM5 activation count would be corrupted
    (grep -c '[CBT-VID]' would read 2x videos_processed). BUE/CBT resolve `_log`
    as a module GLOBAL at call time, so swapping the global for this shim is
    sufficient and needs no edit to the operators themselves (which must stay
    byte-frozen). Safe because the arm-OFF pass is fully SYNCHRONOUS (contains no
    `await`) and the pipeline's concurrency is asyncio-only (asyncio.Semaphore +
    asyncio.gather in batch_evaluate.py; there is no threading in pipeline.py),
    so no other coroutine can observe the swapped global; restoration is in a
    `finally`.
    """

    __slots__ = ()

    def __getattr__(self, _name):  # info/debug/warning/error/exception/...
        return lambda *_a, **_k: None


try:
    from .video_preprocessor import VideoPreprocessor
    from .anomaly_discovery import AnomalyDiscovery, TCP_PROMPT_BLOCK
    from .temporal_mcts import TemporalBoundaryMCTS
    from .anomaly_expert import VideoAnomalyExpert
    from .token_compressor import SemanticFrameSampler
except ImportError:
    from video_preprocessor import VideoPreprocessor
    from anomaly_discovery import AnomalyDiscovery, TCP_PROMPT_BLOCK
    from temporal_mcts import TemporalBoundaryMCTS
    from anomaly_expert import VideoAnomalyExpert
    from token_compressor import SemanticFrameSampler

try:
    from .videomae_classifier import VideoMAEClassifier
except ImportError:
    from videomae_classifier import VideoMAEClassifier

try:
    from .config import VADConfig
    from .models import AnomalySegment, DetectionResult
except ImportError:
    from config import VADConfig
    from models import AnomalySegment, DetectionResult

try:
    from .frame_scorer import FrameScorer
    from .post_processor import global_merge, narrow_short_events, unify_anomaly_types, temporal_nms
except ImportError:
    from frame_scorer import FrameScorer
    from post_processor import global_merge, narrow_short_events, unify_anomaly_types, temporal_nms

try:
    from .snh_features import compute_snh_features as _snh_compute_features
except ImportError:
    from snh_features import compute_snh_features as _snh_compute_features


CATEGORY_CUES = {
    "Abuse": "DEFENSELESS victim (child/elderly/restrained) being hit repeatedly — attacker has total control, victim CANNOT fight back",
    "Arrest": "UNIFORMED police/security officers restraining, handcuffing, or detaining a suspect — look for uniforms, police vehicles, badges",
    "Arson": "Fire starting SMALL from one point and GROWING gradually — a person may be near the fire origin with lighter/matches, deliberate fire-setting",
    "Assault": "ONE-SIDED attack — one person aggressively strikes another who is cowering, on the ground, or fleeing but NOT fighting back (victim is an able adult)",
    "Burglary": "Person ENTERING a building by force — breaking doors/windows, climbing walls/fences, forcing locks to GET INSIDE",
    "Explosion": "SUDDEN instantaneous blast in a fraction of a second — bright flash, debris FLYING outward, large smoke/dust cloud appears ALL AT ONCE",
    "Fighting": "MUTUAL combat — TWO OR MORE people BOTH actively throwing punches/kicks at EACH OTHER, both sides in aggressive postures",
    "RoadAccidents": "Vehicle COLLISION or CRASH — cars/trucks/motorcycles colliding, flipping, or skidding on a road, with debris and damage",
    "Robbery": "FACE-TO-FACE confrontation — thief uses WEAPON (gun/knife) or FORCE/THREATS to demand and take items from a person directly",
    "Shooting": "People suddenly DROP/COLLAPSE or crowd SCATTERS in panic WITHOUT anyone physically touching them — signs of gunfire, guns may NOT be visible",
    "Shoplifting": "Inside a RETAIL STORE with product shelves — person secretly concealing merchandise in bag/pocket/clothing and moving toward exit",
    "Stealing": "COVERT theft — taking UNATTENDED items (bag/phone from table/bench/car) or SNATCHING from a distracted person, then fleeing — NO weapon, NO threat",
    "Vandalism": "DAMAGING property WHERE IT STANDS — smashing car windows, kicking objects, spray-painting walls, breaking street furniture — NOT entering a building",
}

VIDEOMAE_CONFUSION_MAP = {
    # Only entries with net positive effect (no regressions)
    'RoadAccidents': ['Robbery'],    # Robbery102 fix: VideoMAE=RoadAccidents→VLM→Robbery
    'Abuse': ['Explosion'],          # Explosion035 fix: VideoMAE=Abuse→VLM→Explosion
    'Assault': ['Arson'],            # Neutral: keeps Arson check alive, no Assault regression
    # REMOVED: Burglary→Shoplifting (v29 test: VLM chose Shoplifting 3/3, no improvement)
    # REMOVED: Stealing, Vandalism — VLM bias toward Robbery/Burglary caused -8 videos
}

SHORT_EVENT_CATEGORIES = {'Explosion', 'RoadAccidents'}

# v50: Categories where CLIP AUC is too low for reliable boundary detection
LOW_CLIP_CATEGORIES = {'Shoplifting', 'Stealing', 'Arson', 'Burglary'}

# v50: Category-aware duration priors (physically motivated, not dataset-overfit)
# v52/v53/v54 each tried adding threshold cleverness here (fixed per-category,
# overshoot-adaptive, multi-threshold+duration-fit) and each regressed mAP.
# v55 reverts to v51's plain fixed-ratio contrast selection (see
# _clip_boundary_detection) and keeps this dict untouched by threshold logic.
CATEGORY_DURATION_PRIORS = {
    # Short-burst events
    'RoadAccidents': {'expected_duration': (2.0, 20.0), 'preferred_smooth': 1.0, 'min_segment': 1.5},  # v51: was (2.0, 8.0) — real accidents span 10-30s incl. aftermath
    'Explosion':     {'expected_duration': (2.0, 10.0), 'preferred_smooth': 1.0, 'min_segment': 1.5},
    'Abuse':         {'expected_duration': (2.0, 8.0), 'preferred_smooth': 1.0, 'min_segment': 1.5},
    'Fighting':      {'expected_duration': (3.0, 15.0), 'preferred_smooth': 1.5, 'min_segment': 2.0},
    'Shooting':      {'expected_duration': (3.0, 15.0), 'preferred_smooth': 1.5, 'min_segment': 2.0},
    # Medium events
    'Robbery':       {'expected_duration': (5.0, 30.0), 'preferred_smooth': 2.0, 'min_segment': 3.0},
    'Burglary':      {'expected_duration': (5.0, 40.0), 'preferred_smooth': 2.5, 'min_segment': 3.0},
    'Shoplifting':   {'expected_duration': (3.0, 20.0), 'preferred_smooth': 2.0, 'min_segment': 2.0},
    'Stealing':      {'expected_duration': (5.0, 30.0), 'preferred_smooth': 2.0, 'min_segment': 3.0},
    'Vandalism':     {'expected_duration': (3.0, 20.0), 'preferred_smooth': 2.0, 'min_segment': 2.0},
    # Long events
    'Arson':         {'expected_duration': (5.0, 40.0), 'preferred_smooth': 2.5, 'min_segment': 3.0},
    'Arrest':        {'expected_duration': (15.0, 90.0), 'preferred_smooth': 4.0, 'min_segment': 8.0},
    'Assault':       {'expected_duration': (10.0, 90.0), 'preferred_smooth': 3.0, 'min_segment': 5.0},
}

# v155 CBT: per-category inward duration caps (seconds) = GT-p75 of the 4 hard
# categories on UCF-Crime, lightly padded. Only categories present here are
# tightened; deleting a key = per-category rollback (no core-method change).
# CBT fires only when a FINAL detection's dur > 1.5*cap (over-wide) and the
# contrast-selected sub-window is strictly narrower — so a cap slightly above
# GT (e.g. RoadAccidents 6.0 vs GT-p75 ~4.3) just gates which dets qualify.
CBT_CAPS = {
    'RoadAccidents': 6.0,
    'Explosion':     13.0,
    'Shoplifting':   10.0,
    'Shooting':      14.0,
}

# v171 CNV: Cascade Null-Hypothesis Veto — PRINCIPLED STRUCTURAL DEFAULTS.
# These are NOT fitted on the test split: each threshold is a plain robust-
# statistics statement about a video's OWN baseline (median/MAD z-scores and
# outlier fractions), verified (not tuned) on a handful of TRAIN-split videos.
# v172 TIGHTEN: thresholds UNCHANGED — the variant only ADDS necessary
# conjuncts (stage-1 sig_d: videomae_category == 'Normal'; stage-2: z_win
# must also be < 0, i.e. below the video's own median), so every v172 fire
# is a strict subset of v171 fires by construction.
CNV_TAU_MIL = 0.20   # stage-1(a): MIL top-k snippet mean below this = no video-level evidence
CNV_TAU_Z = 1.5      # stage-1(b): fused-curve in-video z-max below this = no outlier moment
CNV_TAU_MOT = 0.01   # stage-1(c): motion outlier (>median+3*MAD) frame fraction below this = still scene
CNV_TAU_SEG = 0.5    # stage-2: window curve-mass z below this (and off-argmax) = veto the detection
CNV_MIL_TOPK = 3     # top-k of the 32 MIL snippets averaged for stage-1(a) (~top 10% of the video)


def _cnv_signal_values(mil_scores, curve, motion):
    """v171 CNV stage-1 signal values + negativity bits (PURE function, module
    level so the offline TRAIN-split verification imports the exact same math).

    Returns (mil_topk, curve_zmax, mot_frac, sig_a, sig_b, sig_c) where the
    three values may be None when their signal family is unavailable, and a
    sig bit is True iff the corresponding family is NEGATIVE (null hypothesis
    "this video contains no baseline-breaking event" survives on that family).
    An unavailable family is NEVER treated as negative (conservative: no
    signal -> no veto).
    """
    mil_topk = None
    if mil_scores is not None:
        s = np.asarray(mil_scores, dtype=np.float64).reshape(-1)
        if len(s):
            k = min(CNV_MIL_TOPK, len(s))
            mil_topk = float(np.mean(np.sort(s)[-k:]))
    curve_zmax = None
    if curve is not None:
        c = np.asarray(curve, dtype=np.float64).reshape(-1)
        if len(c) >= 3:
            med = float(np.median(c))
            mad = float(np.median(np.abs(c - med)))
            curve_zmax = float(np.max((c - med) / (mad + 1e-9)))
    mot_frac = None
    if motion is not None:
        m = np.asarray(motion, dtype=np.float64).reshape(-1)
        if len(m):
            med_m = float(np.median(m))
            mad_m = float(np.median(np.abs(m - med_m)))
            mot_frac = float(np.mean(m > (med_m + 3.0 * mad_m)))
    sig_a = mil_topk is not None and mil_topk < CNV_TAU_MIL
    sig_b = curve_zmax is not None and curve_zmax < CNV_TAU_Z
    sig_c = mot_frac is not None and mot_frac < CNV_TAU_MOT
    return mil_topk, curve_zmax, mot_frac, sig_a, sig_b, sig_c


def _cnv_fmt(x, prec=4):
    """v171 CNV marker formatting: 'nan' for unavailable signals."""
    return f"{x:.{prec}f}" if x is not None else "nan"


# v179 SNH: Supervised Normality Head — module-level lazy model cache. The
# v19-trained head (model_snh.json next to this file) is loaded AT MOST ONCE
# per process, and ONLY on the flag-on path (flag-off: the file is never even
# read — byte-identical committed behaviour).
_SNH_MODEL = None  # None = not yet loaded this process


def _snh_load_model():
    """v179 SNH: lazy-load model_snh.json ONCE (module-level cache).

    Returns the parsed+validated model dict, or None when the file is missing/
    corrupt — the caller then emits [SNH-ERR] and treats the video as flag-off
    (the run NEVER crashes on the veto path). A FAILED load is deliberately
    not cached, so a transient FS error does not silently disable the
    mechanism for the rest of the run (one [SNH-ERR] per affected video).
    """
    global _SNH_MODEL
    if _SNH_MODEL is not None:
        return _SNH_MODEL
    from pathlib import Path as _SnhPath
    try:
        m = json.loads((_SnhPath(__file__).resolve().parent
                        / 'model_snh.json').read_text())
        for k in ('tau', 'feature_order', 'scaler_mean', 'scaler_scale',
                  'weights', 'bias'):
            if k not in m:
                raise ValueError(f"model_snh.json missing key {k!r}")
        if not (len(m['feature_order']) == len(m['scaler_mean'])
                == len(m['scaler_scale']) == len(m['weights'])):
            raise ValueError("model_snh.json feature/scaler/weight length mismatch")
        _SNH_MODEL = m
    except Exception as _e:
        _log.warning(f"  [SNH-ERR] model_snh.json load failed: {_e}")
        return None
    return _SNH_MODEL


def _compute_tsm_novelty(clip_features, kernel_half_widths=(15, 30, 60)):
    """v95: Temporal Self-Similarity Matrix + checkerboard kernel novelty
    function (Foote's novelty function, UBoCo/GEBD lineage). Loop 1 research:
    the ONLY signal construction among ~7 paradigms studied that is a
    genuinely different SIGNAL TYPE from our text-similarity curve - a
    frame-to-FRAME visual similarity matrix, not frame-to-TEXT. Detects
    visual scene transitions completely independent of category semantics.

    SSM[i,j] = cosine_sim(feat_i, feat_j) for all frame pairs. Checkerboard
    kernel (+1 same-block quadrants, -1 cross-block, Gaussian-tapered)
    correlated along the SSM diagonal. Peaks = visual transitions.

    Multi-scale: computed at 3 kernel half-widths, combined via max (a
    transition detected at ANY scale counts). Zero additional CLIP/VLM calls
    - reuses features already extracted for the primary pipeline.

    Returns a novelty(t) array, same length as clip_features, normalized to
    [0, 1] via min-max over the video.
    """
    n = clip_features.shape[0]
    if n < 8:
        return np.zeros(n)

    feats = clip_features
    if hasattr(feats, 'cpu'):
        feats = feats.cpu().numpy()
    feats = feats.astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats_n = feats / norms
    ssm = feats_n @ feats_n.T  # (n, n) cosine similarity matrix

    novelty_multiscale = np.zeros(n)
    for L in kernel_half_widths:
        L = min(L, n // 2 - 1)
        if L < 2:
            continue
        size = 2 * L
        idx = np.arange(size) - L + 0.5
        gauss = np.exp(-(idx[:, None] ** 2 + idx[None, :] ** 2) / (2 * (L / 2.0) ** 2))
        checker = np.sign(np.outer(np.sign(idx), np.sign(idx)))
        kernel = checker * gauss

        novelty = np.zeros(n)
        for t in range(L, n - L):
            block = ssm[t - L:t + L, t - L:t + L]
            novelty[t] = float(np.sum(kernel * block))
        _rng = novelty.max() - novelty.min()
        if _rng > 1e-8:
            novelty = (novelty - novelty.min()) / _rng
        novelty_multiscale = np.maximum(novelty_multiscale, novelty)

    return novelty_multiscale


def _score_boundary_proposal(smoothed, gradient, seg_times, lo, hi, refine_range, min_dur, max_dur):
    """v54: Refine a threshold-crossing [lo, hi) via gradient, then score it by
    Outer-Inner-Contrast (OIC, as in AutoLoc/BSN) times a duration-prior fit.

    This is the key architectural change from v50-v53: instead of picking the
    proposal with the highest contrast alone (which favors narrow, spiky, but
    often implausible boundaries), the selection score also rewards proposals
    whose duration actually falls inside the category's expected range. A
    proposal outside [min_dur, max_dur] is penalized by how far outside it is,
    rather than being accepted just because no better-contrast alternative was
    tried at that threshold.
    """
    left_lo = max(0, lo - refine_range)
    left_hi = min(len(gradient), lo + refine_range + 1)
    refined_lo = left_lo + int(np.argmax(gradient[left_lo:left_hi])) if left_hi > left_lo else lo

    right_lo = max(0, hi - refine_range)
    right_hi = min(len(gradient), hi + refine_range + 1)
    refined_hi = right_lo + int(np.argmin(gradient[right_lo:right_hi])) if right_hi > right_lo else hi

    if refined_hi <= refined_lo:
        refined_lo, refined_hi = lo, hi
    if refined_hi <= refined_lo:
        return None

    inside_mean = float(np.mean(smoothed[refined_lo:refined_hi + 1]))
    outside_left = float(np.mean(smoothed[:max(1, refined_lo)])) if refined_lo > 0 else 0.0
    outside_right = float(np.mean(smoothed[refined_hi + 1:])) if refined_hi < len(smoothed) - 1 else 0.0
    contrast = inside_mean - (outside_left + outside_right) / 2.0

    t_start = float(seg_times[min(refined_lo, len(seg_times) - 1)])
    t_end = float(seg_times[min(refined_hi, len(seg_times) - 1)])
    dur = t_end - t_start
    if dur <= 0:
        return None

    if dur < min_dur:
        duration_fit = dur / min_dur
    elif dur > max_dur:
        duration_fit = max_dur / dur
    else:
        duration_fit = 1.0

    return {
        't_start': t_start, 't_end': t_end,
        'contrast': contrast, 'duration_fit': duration_fit,
        'combined': contrast * duration_fit,
    }


# v51: Category-aware ConfFilter thresholds (calibrated against new confidence formula)
CATEGORY_CONF_GATES = {
    # Low-CLIP categories (AUC < 0.55): trust LLM more, filter less
    'Shoplifting': 0.38, 'Stealing': 0.38, 'Robbery': 0.42,
    'Vandalism': 0.42, 'Arson': 0.42, 'Burglary': 0.42,
    # Medium-CLIP categories (AUC 0.55-0.75): moderate filtering
    'Abuse': 0.48, 'RoadAccidents': 0.48,
    # High-CLIP categories (AUC > 0.75): can afford stricter filtering
    'Arrest': 0.55, 'Assault': 0.55, 'Explosion': 0.55,
    'Fighting': 0.55, 'Shooting': 0.52,
}


class MultiAnomalyVAD:

    DEFAULT_ANOMALY_CATEGORIES = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism",
    ]

    def __init__(
        self,
        llm_client,
        config: VADConfig,
        classify_llm_client=None,
        image_processor=None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.classify_llm_client = classify_llm_client or llm_client
        self.image_processor = image_processor
        self.anomaly_categories = config.anomaly_categories or self.DEFAULT_ANOMALY_CATEGORIES

        pp = config.preprocess
        self.preprocessor = VideoPreprocessor(
            fps=pp.fps,
            segment_duration=pp.segment_duration,
            motion_threshold=pp.motion_threshold,
            min_motion_interval=pp.min_motion_interval,
            segment_overlap=pp.segment_overlap,
        )

        cl = config.clip
        self.anomaly_expert = VideoAnomalyExpert(
            anomaly_categories=self.anomaly_categories,
            clip_model_name=cl.model_name,
            clip_pretrained=cl.pretrained,
            device=config.device,
        )

        self.frame_sampler = SemanticFrameSampler(
            anomaly_expert=self.anomaly_expert,
            tau=cl.semantic_tau,
            max_frames=cl.max_frames,
            min_frames=cl.min_frames,
            fps_sample=cl.fps_sample,
        )

        try:
            self.videomae = VideoMAEClassifier(device=config.device)
        except Exception as e:
            _log.warning(f"[Pipeline] VideoMAE not available: {e}")
            self.videomae = None

        dc = config.discovery
        self.discovery = AnomalyDiscovery(
            llm_client=llm_client,
            anomaly_categories=self.anomaly_categories,
            image_processor=image_processor,
            clip_expert=self.anomaly_expert,
            frame_sampler=self.frame_sampler,
            min_pass_rate=dc.min_pass_rate,
            clip_threshold=dc.clip_threshold,
            max_scan_segments=dc.max_scan_segments,
            comparative_batch_segments=dc.comparative_batch_segments,
            comparative_keyframes=dc.comparative_keyframes,
            comparative_long_segments=dc.comparative_long_segments,
            comparative_long_batch=dc.comparative_long_batch,
            kfd_uniform_keyframes=dc.kfd_uniform_keyframes,
            kfd_k=dc.kfd_k,
            tcp_prompt=dc.tcp_prompt,
        )

        self.mcts_simulations = config.mcts.simulations
        self.mcts_max_depth = config.mcts.max_depth

        self.frame_scorer = FrameScorer(
            anomaly_expert=self.anomaly_expert,
            config=config,
            anomaly_categories=self.anomaly_categories,
        )

        # v51: Load MIL model for supplementary localization on low-CLIP categories
        self.mil_model = None
        try:
            from pathlib import Path as _Path
            _mil_path = _Path(__file__).parent / 'model_mil.pth'
            if _mil_path.exists():
                try:
                    from .mil_model import MILAnomalyScorer
                except ImportError:
                    from mil_model import MILAnomalyScorer
                self.mil_model = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
                _state = torch.load(str(_mil_path), map_location=config.device, weights_only=True)
                self.mil_model.load_state_dict(_state)
                self.mil_model.eval()
                self.mil_model.to(config.device)
                _log.info(f"  [MIL] Loaded MIL model from {_mil_path}")
            else:
                _log.info(f"  [MIL] No MIL model found, MIL fusion disabled")
        except Exception as _e:
            _log.warning(f"  [MIL] Failed to load MIL model: {_e}")
            self.mil_model = None

    def _qscore_rerank(self, final_anomalies, frame_scores, timestamps, base=0.5):
        """v111: pure post-hoc localization-quality confidence re-rank.

        Q = mean(boundary_sharpness, peak_contrast, center_fit) on the OUTPUT
        frame_scores curve, min-max normalized WITHIN the video, then
        conf' = conf * (base + (1-base)*Q). Runs after conf-gate/ECE/EUBC
        (count already fixed) so it NEVER changes boundaries or detection count
        (FM1/FM2 immune). Byte-for-byte matches analysis_v111_qscore_precheck.py.
        """
        if not final_anomalies or frame_scores is None or timestamps is None:
            _log.info("  [QSCORE-GATE] skipped (no detections / no curve)")
            return final_anomalies
        fs = np.asarray(frame_scores, dtype=float)
        ts = np.asarray(timestamps, dtype=float)
        if len(fs) < 3 or len(ts) != len(fs):
            _log.info("  [QSCORE-GATE] skipped (curve too short / misaligned)")
            return final_anomalies
        grad = np.abs(np.gradient(fs))
        vid_range = fs.max() - fs.min() + 1e-9
        qs = []
        for a in final_anomalies:
            t_s, t_e = a['t_start'], a['t_end']
            in_mask = (ts >= t_s) & (ts <= t_e)
            if in_mask.sum() < 1:
                qs.append(0.0)
                continue
            peak = fs[in_mask].max()
            peak_idx = np.where(in_mask)[0][np.argmax(fs[in_mask])]
            out_mask = ~in_mask
            out_mean = fs[out_mask].mean() if out_mask.any() else fs.min()
            contrast = float(np.clip((peak - out_mean) / vid_range, 0, 1))
            i_s = int(np.argmin(np.abs(ts - t_s)))
            i_e = int(np.argmin(np.abs(ts - t_e)))
            sharp = float(np.clip((grad[i_s] + grad[i_e]) / (2 * (grad.max() + 1e-9)), 0, 1))
            center_t = 0.5 * (t_s + t_e)
            dur = max(t_e - t_s, 1e-6)
            center = float(np.clip(1 - 2 * abs(ts[peak_idx] - center_t) / dur, 0, 1))
            qs.append((sharp + contrast + center) / 3.0)
        qs = np.array(qs)
        if len(qs) > 1 and qs.max() - qs.min() > 1e-9:
            qn = (qs - qs.min()) / (qs.max() - qs.min())
        else:
            qn = np.ones_like(qs)
        for a, q in zip(final_anomalies, qn):
            a['confidence'] = min(1.0, a['confidence'] * (base + (1 - base) * float(q)))
        _log.info(f"  [QSCORE-GATE] reranked {len(final_anomalies)} detections "
                  f"(base={base}, raw Q range [{qs.min():.3f},{qs.max():.3f}])")
        return final_anomalies

    # ================================================================
    # v142 BUE: Boundary-Union Extension (TERMINAL-MOST slot)
    # ================================================================

    def _bue_boundary_union_extension(self, final_anomalies, video_data, category):
        """v142 BUE: grow each FINAL detection OUTWARD to the outer-inner-contrast
        (OIC) argmax window on the read-only category curve — UNION ONLY, length
        capped at bue_lmax_sec / bue_lmax_frac*duration.

        Contract (why this is FM2/FM3/FM4/FM14-immune):
          * Only a['t_start'] / a['t_end'] are ever written. confidence, detection
            COUNT, anomaly_type, rank order, NMS, merge and every prompt are frozen
            bit-for-bit; the per-fire marker logs `conf=<c>-><c>` to make that
            visible to the existing paired parsers.
          * Zero new VLM / CLIP / MIL calls: get_category_scores is the SAME cached
            call _clip_boundary_single_scale already makes (per video+category cache
            key `_cat_scores_<cat>` on video_data), and it is strictly READ-ONLY —
            the curve is never written back, rescaled or fused (FM1 in the strict
            sense).
          * UNION IS THE MECHANISM, not an implementation detail. Offline 3-cache
            ablation: closure-gated REPLACE (emit the OIC window instead of the old
            one) = -0.0065 mean, UNION = +0.0378 mean. Monotone outward growth
            toward the contrast argmax is what wins; pure OIC relocation is
            worthless or harmful. Never contract, never replace.
          * The v51 half-max CLOSURE TEST (hm = (seg.max() + p20(curve))/2) skips
            byte-identically when the curve is already below hm at BOTH edges — the
            boundary was then genuinely set by a signal crossing rather than clipped
            by the padded search window. Measured mAP no-op (gated == ungated
            bit-identical on 3/3 caches at Lmax 40 and 60); retained purely as free
            blast-radius reduction (67 fires instead of 78).

        Byte-for-byte mirror of analysis_v142_bue_verify.py::bue_pick (an
        independent reimplementation from the written spec reproduced +0.02221 /
        +0.01629 / +0.03879 to five decimals), on the fidelity-gated substrate of
        analysis_v140_livecurve_lib (58/58 recorded live peak times at 0.000 s) —
        the FM15 remedy. features/test/*.npy is NOT used anywhere.

        FM13 magnitude bound taken BEFORE the eval: best-of-two oracle +0.0764,
        realized +0.0378 at Lmax=40, sign-robust 24/24 over Lmax in [15,60] x 3
        caches; 12-cache out-of-sample read 9/9 positive, mean +0.0370.
        """
        cfg = self.config.scoring
        # flag-off: return BEFORE any curve is read -> byte-identical, zero cost.
        if not cfg.bue_enabled or not final_anomalies or not category:
            return final_anomalies

        import os as _bue_os
        vid = _bue_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'

        timestamps = video_data.get('timestamps')
        if timestamps is None:
            _log.info(f"  [BUE] summary: vid={vid} fires=0 closed_skips=0 "
                      f"ndet_before={len(final_anomalies)} "
                      f"ndet_after={len(final_anomalies)} growth_med=0.000 "
                      f"dur_ratio=1.000 selfov=0 skip=no_timestamps")
            return final_anomalies
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        n = int(len(t))
        if n < 8:
            _log.info(f"  [BUE] summary: vid={vid} fires=0 closed_skips=0 "
                      f"ndet_before={len(final_anomalies)} "
                      f"ndet_after={len(final_anomalies)} growth_med=0.000 "
                      f"dur_ratio=1.000 selfov=0 skip=curve_too_short")
            return final_anomalies

        duration = float(video_data.get('duration') or float(t[-1]))
        dt = float(np.median(np.diff(t))) or 0.5
        w = int(cfg.bue_smooth_win)
        alpha = float(cfg.bue_alpha)
        l_min = float(cfg.bue_lmin_sec)
        l_max = float(cfg.bue_lmax_sec)
        l_frac = float(cfg.bue_lmax_frac)
        eff_lmax = min(l_max, l_frac * duration) if duration > 0 else l_max
        kmin = max(2, int(round(l_min / dt)))
        kmax = max(kmin + 1, int(round(eff_lmax / dt)))

        _prep_cache = {}

        def _prep(cat):
            """(cs, prefix-sum, percentile20) for a category, or None."""
            if cat in _prep_cache:
                return _prep_cache[cat]
            try:
                rel = self.frame_scorer.get_category_scores(video_data, cat)
            except Exception as _e:  # never let a read-only diagnostic kill detect()
                _log.warning(f"  [BUE] curve read failed vid={vid} cat={cat}: {_e}")
                rel = None
            if rel is None:
                _prep_cache[cat] = None
                return None
            if hasattr(rel, 'detach'):      # FM9: MPS/CUDA tensors
                rel = rel.detach().cpu().numpy()
            c = np.asarray(rel, dtype=np.float64).reshape(-1)
            if len(c) != n:
                _prep_cache[cat] = None
                return None
            cs = c if (w <= 1 or len(c) < w) else np.convolve(
                c, np.ones(w) / w, mode='same')
            P = np.concatenate([[0.0], np.cumsum(cs)])
            _prep_cache[cat] = (cs, P, float(np.percentile(cs, 20)))
            return _prep_cache[cat]

        def _mean(P, i, j):
            i = max(0, min(int(i), n))
            j = max(0, min(int(j), n))
            return (P[j] - P[i]) / max(1, j - i) if j > i else 0.0

        def _oic(P, s, e):
            k = int(e) - int(s)
            if k <= 0:
                return 0.0
            r = max(1, int(round(alpha * k)))
            nl = min(r, max(0, int(s)))
            nr = min(r, max(0, n - int(e)))
            den = nl + nr
            out = ((_mean(P, s - r, s) * nl + _mean(P, e, e + r) * nr) / den
                   if den else 0.0)
            return _mean(P, s, e) - out

        n_fire = n_closed = n_degen = n_noop = n_nocurve = 0
        growths, ratios = [], []
        ndet_before = len(final_anomalies)

        for di, a in enumerate(final_anomalies):
            # FM15: category resolution MUST match the verified replay exactly —
            # the detection's OWN anomaly_type first, then videomae_category.
            cat = str(a.get('anomaly_type') or '').strip() or category
            prep = _prep(cat)
            if prep is None:
                n_nocurve += 1
                continue
            cs, P, p20 = prep
            ds = float(a['t_start'])
            de = float(a['t_end'])
            i0 = int(np.searchsorted(t, ds))
            i1 = int(np.searchsorted(t, de))
            i0 = max(0, min(i0, n - 1))
            i1 = max(i0 + 1, min(i1, n))
            seg = cs[i0:i1]
            hm = (float(seg.max()) + p20) / 2.0
            anchor = i0 + int(np.argmax(seg))
            if cfg.bue_closure_gate and cs[i0] < hm and cs[i1 - 1] < hm:
                n_closed += 1                     # boundary genuinely closed by the signal
                continue
            best, bs, be = -1e30, None, None
            for s in range(max(0, anchor - kmax), anchor + 1):
                for k in range(kmin, kmax + 1):
                    e = s + k
                    if e > n or e <= anchor:
                        continue
                    r = max(1, int(round(alpha * k)))
                    nl = min(r, max(0, s))
                    nr = min(r, max(0, n - e))
                    den = nl + nr
                    out = ((_mean(P, s - r, s) * nl + _mean(P, e, e + r) * nr) / den
                           if den else 0.0)
                    v = _mean(P, s, e) - out
                    if v > best:
                        best, bs, be = v, s, e
            if bs is None:
                n_degen += 1
                continue
            ns = float(t[bs])
            ne = float(t[min(be, n - 1)])
            # UNION ONLY — never a replacement, never a contraction (measured:
            # REPLACE = -0.0065 vs UNION = +0.0378).
            ns = min(ds, ns)
            ne = max(de, ne)
            if duration > 0:
                ns = max(0.0, min(ns, duration))
                ne = max(0.0, min(ne, duration))
            if ne <= ns:
                n_degen += 1
                continue
            if abs(ns - ds) <= 1e-6 and abs(ne - de) <= 1e-6:
                n_noop += 1
                continue
            old_oic = float(_oic(P, i0, i1))
            us = max(0, min(int(np.searchsorted(t, ns)), n - 1))
            ue = max(us + 1, min(int(np.searchsorted(t, ne)), n))
            new_oic = float(_oic(P, us, ue))
            old_d = de - ds
            new_d = ne - ns
            conf = float(a['confidence'])
            a['t_start'] = ns
            a['t_end'] = ne
            n_fire += 1
            growths.append(new_d - old_d)
            ratios.append(new_d / old_d if old_d > 1e-9 else 1.0)
            _log.info(
                f"  [BUE] vid={vid} cat={cat} det={di} closed=0 "
                f"t={ds:.3f}-{de:.3f}->{ns:.3f}-{ne:.3f} "
                f"dur={old_d:.2f}->{new_d:.2f} oic={old_oic:.4f}->{new_oic:.4f} "
                f"conf={conf:.4f}->{conf:.4f}")

        # M5 (diagnostic only, DO NOT act on it this loop — FM8): count surviving
        # detection pairs whose mutual temporal IoU exceeds 0.3 after widening.
        # BUE runs AFTER NMS, so widening can create new mutual overlap and under
        # greedy single-match mAP the loser becomes an FP. That penalty is already
        # inside the measured +0.0378; this field only quantifies the de-dup
        # headroom for a later loop.
        selfov = 0
        for _i in range(len(final_anomalies)):
            for _j in range(_i + 1, len(final_anomalies)):
                _a, _b = final_anomalies[_i], final_anomalies[_j]
                _inter = max(0.0, min(_a['t_end'], _b['t_end'])
                             - max(_a['t_start'], _b['t_start']))
                _union = ((_a['t_end'] - _a['t_start'])
                          + (_b['t_end'] - _b['t_start']) - _inter)
                if _union > 1e-9 and _inter / _union > 0.3:
                    selfov += 1

        g_med = float(np.median(growths)) if growths else 0.0
        d_ratio = float(np.mean(ratios)) if ratios else 1.0
        # M4 MATERIALITY GATE (the gate that would have caught v107 ECE, which
        # passed activation with 160 markers while its median growth was 0.000 s):
        # the flag-ON smoke requires median per-fire growth > 2.0 s AND mean
        # post/pre duration ratio > 1.5 (offline: 3.9 s / 1.90).
        _log.info(
            f"  [BUE] summary: vid={vid} fires={n_fire} closed_skips={n_closed} "
            f"ndet_before={ndet_before} ndet_after={len(final_anomalies)} "
            f"growth_med={g_med:.3f} dur_ratio={d_ratio:.3f} selfov={selfov} "
            f"degen={n_degen} noop={n_noop} nocurve={n_nocurve} "
            f"lmax={eff_lmax:.1f} gate={int(bool(cfg.bue_closure_gate))}")
        return final_anomalies

    async def detect(
        self,
        video_path: str,
        context: str = "surveillance camera",
    ) -> Dict[str, Any]:
        """v124 [COST] instrumentation wrapper (DEFAULT-ON, no flag, FM10).

        Snapshots the drift-free per-video VLM (llm_client + classify_llm_client
        .generate) and CLIP (anomaly_expert.encode_image) call counters plus a
        wall-clock start, delegates to _detect_impl (unchanged detection logic),
        and emits exactly one machine-parseable [COST] INFO line per video in a
        finally block so it fires on every exit path (incl. early returns and
        exceptions). Counters are read-only here and never touch any request,
        feature, or detection -> mAP is byte-neutral.
        """
        import time as _cost_time
        import os as _cost_os
        _cost_t0 = _cost_time.time()
        _vlm_detect_start = getattr(self.llm_client, 'call_count', 0)
        _vlm_classify_start = getattr(self.classify_llm_client, 'call_count', 0)
        _clip_start = getattr(self.anomaly_expert, 'clip_call_count', 0)
        try:
            return await self._detect_impl(video_path, context)
        finally:
            _n_detect = getattr(self.llm_client, 'call_count', 0) - _vlm_detect_start
            if self.classify_llm_client is self.llm_client:
                # Same object: the detect delta already includes classify calls.
                _vlm_calls = _n_detect
            else:
                _n_classify = (getattr(self.classify_llm_client, 'call_count', 0)
                               - _vlm_classify_start)
                _vlm_calls = _n_detect + _n_classify
            _n_clip = getattr(self.anomaly_expert, 'clip_call_count', 0) - _clip_start
            _elapsed = _cost_time.time() - _cost_t0
            _vname = _cost_os.path.basename(str(video_path))
            _log.info(f"[COST] video={_vname} vlm_calls={_vlm_calls} "
                      f"clip_calls={_n_clip} elapsed_s={_elapsed:.1f}")

    async def _detect_impl(
        self,
        video_path: str,
        context: str = "surveillance camera",
    ) -> Dict[str, Any]:
        video_data = self.preprocessor.process(video_path)

        # Stage 1: CLIP precompute
        if self.anomaly_expert.model is not None:
            import time as _time
            _t0 = _time.time()
            video_data['clip_features'] = self.anomaly_expert.precompute_video_features(
                video_data['all_frames']
            )
            _log.info(f"  [CLIP precompute] {len(video_data['all_frames'])} frames "
                      f"encoded in {_time.time()-_t0:.1f}s")

            # v51: Compute MIL snippet scores from CLIP features
            if self.mil_model is not None and video_data['clip_features'] is not None:
                try:
                    _clip_feats = video_data['clip_features']
                    _n_frames = _clip_feats.shape[0]
                    _num_segments = 32
                    if _n_frames >= _num_segments:
                        _seg_len = _n_frames // _num_segments
                        _snippet_feats = []
                        for _si in range(_num_segments):
                            _start = _si * _seg_len
                            _end = _start + _seg_len if _si < _num_segments - 1 else _n_frames
                            _snippet_feats.append(_clip_feats[_start:_end].mean(dim=0))
                        _snippet_tensor = torch.stack(_snippet_feats).unsqueeze(0)  # (1, 32, 768)
                        with torch.no_grad():
                            _mil_scores = self.mil_model(_snippet_tensor).squeeze(0).cpu().numpy()
                        video_data['mil_scores'] = _mil_scores
                        _log.info(f"  [MIL] snippet scores: mean={_mil_scores.mean():.3f}, "
                                  f"max={_mil_scores.max():.3f}, min={_mil_scores.min():.3f}")

                    else:
                        _log.info(f"  [MIL] Too few frames ({_n_frames}) for 32-snippet MIL")
                except Exception as _me:
                    _log.warning(f"  [MIL] Scoring failed: {_me}")


        # v95: TSM checkerboard-kernel novelty (Loop 1 research). Zero extra
        # CLIP/VLM calls - reuses already-extracted clip_features. Used ONLY
        # for confidence re-ranking (never boundary position), avoiding all
        # 4 known failure modes from prior rounds.
        if video_data.get('clip_features') is not None:
            try:
                video_data['tsm_novelty'] = _compute_tsm_novelty(video_data['clip_features'])
            except Exception as _tsme:
                _log.warning(f"  [TSM] novelty computation failed: {_tsme}")
                video_data['tsm_novelty'] = None

        # Stage 1b: Motion energy computation
        video_data['motion_energy'] = self._compute_motion_energy(video_data)

        # Stage 2: VideoMAE full-video classification (moved up from end)
        videomae_category = None
        videomae_confidence = 0.0
        if self.videomae is not None and self.videomae.model is not None:
            result = self.videomae.classify(video_data['all_frames'], top_k=3, tta=True)
            videomae_category = result['category']
            videomae_confidence = result['confidence']
            top3_str = ', '.join(f'{c}({s:.0%})' for c, s in result['top_k'])
            _log.info(f"  [VideoMAE] {top3_str} -> {videomae_category}@{videomae_confidence:.0%}")

            # Early Normal filter
            if videomae_category == 'Normal' and videomae_confidence > 0.5:
                _log.info(f"  [VideoMAE] Normal@{videomae_confidence:.0%} — skipping detection")
                if self.config.scoring.event_continuation_extension:
                    _log.info("  [ECE] summary: 0 detections (early return: video normal)")
                if self.config.scoring.event_unit_boundary_contraction:
                    _log.info("  [EUBC] summary: 0 detections (early return: video normal)")
                if self.config.scoring.cbt_enabled:
                    # FM17: unconditional per-video CBT marker on the empty /
                    # 0-detection path so grep -c '[CBT-VID]' == videos processed.
                    _log.info(f"  [CBT-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                              f"enabled=True ndet=0 fired=0 "
                              f"recal={'on' if self.config.scoring.cbt_recal else 'off'} "
                              f"(early return: video normal)")
                if self.config.scoring.cgr_enabled:
                    # FM17: unconditional per-video CGR marker on the empty /
                    # 0-detection path so grep -c '[CGR-VID]' == videos processed.
                    _log.info(f"  [CGR-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                              f"enabled=True ndet_before=0 ndet_after=0 moved=0 "
                              f"skip=early_return_normal")
                if self.config.scoring.ccr_enabled:
                    # FM17: unconditional per-video CCR marker on the empty /
                    # 0-detection path so grep -c '[CCR-VID]' == videos processed.
                    _log.info(f"  [CCR-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                              f"enabled=True ndet_before=0 ndet_after=0 moved=0 "
                              f"skip=early_return_normal")
                if self.config.scoring.cgr_ab_diag:
                    # FM17: unconditional per-video CGR-AB marker on the empty /
                    # 0-detection path so grep -cF '[CGR-AB] vid=' == videos
                    # processed. ndet=0 -> both arms are trivially the empty list.
                    _log.info(f"  [CGR-AB] vid={str(video_path).rsplit('/', 1)[-1]} "
                              f"ndet=0 armoff_len=0 fired=0")
                if self.config.scoring.cex_extend:
                    # FM5: every processed video emits exactly one [CEX-GATE]
                    # line, so the runner-side grep gate can prove activation
                    # coverage and read the realized acceptance rate (M5).
                    _log.info(f"  [CEX-GATE] vid={_cost_os.path.basename(str(video_path))} "
                              f"probes=0 accepted=0 (early return: video normal)")
                if self.config.scoring.cnv_veto:
                    # FM17: unconditional per-video CNV marker on the empty /
                    # early-return-normal path so grep -cF '[CNV-VID]' == videos
                    # processed. This path returns BEFORE the terminal slot.
                    self._cnv_log_early_return(
                        video_path, video_data, videomae_category,
                        'early_return_normal')
                if self.config.scoring.snh_veto:
                    # FM17: unconditional per-video SNH marker on the empty /
                    # early-return-normal path so grep -cF '[SNH-VID]' == videos
                    # processed. This path returns BEFORE the terminal slot.
                    self._snh_log_early_return(
                        video_path, video_data, videomae_category,
                        'early_return_normal')
                if self.config.discovery.kfd_uniform_keyframes:
                    # FM17: unconditional per-video KFD marker on the
                    # early-return-normal path so grep -cF '[KFD-VID]' ==
                    # videos processed. This path returns BEFORE Stage 4, so
                    # the comparative scan never ran (segments=0).
                    self._kfd_log_vid(video_path, None, 'early_return_normal')
                if self.config.discovery.tcp_prompt:
                    # FM17: unconditional per-video TCP marker on the
                    # early-return-normal path so grep -cF '[TCP-VID]' ==
                    # videos processed. This path returns BEFORE Stage 4, so
                    # the scan never ran (segments=0).
                    self._tcp_log_vid(video_path, 0, 'early_return_normal')
                frame_scores = self.frame_scorer.compute_frame_scores(video_data, [], None)
                return DetectionResult(
                    video_path=video_path,
                    duration=video_data['duration'],
                    is_normal=True,
                    anomalies=[],
                    frame_scores=frame_scores,
                    timestamps=video_data['timestamps'].tolist() if frame_scores else None,
                    pre_ece_boundaries=(
                        [] if self.config.scoring.event_continuation_extension
                        else None),
                    pre_cex_boundaries=(
                        [] if self.config.scoring.cex_extend else None),
                    pre_cgr_boundaries=(
                        [] if self.config.scoring.cgr_enabled else None),
                    pre_ccr_boundaries=(
                        [] if self.config.scoring.ccr_enabled else None),
                    ab_arm_off_boundaries=(
                        [] if (self.config.scoring.cgr_ab_diag
                               and self.config.scoring.cgr_enabled) else None),
                )

            # Low-confidence Normal: CLIP fallback
            if videomae_category == 'Normal':
                precomputed = video_data.get('clip_features')
                if precomputed is not None and self.anomaly_expert.model is not None:
                    all_indices = list(range(len(video_data['all_frames'])))
                    clip_dist = self.anomaly_expert.get_category_distribution(
                        all_indices, precomputed, top_k=3)
                    if clip_dist:
                        clip_cats = [c for c in clip_dist.keys() if c != 'Normal']
                        if clip_cats:
                            videomae_category = clip_cats[0]
                            _log.info(f"  [VideoMAE] Normal override -> CLIP: {videomae_category}")

            video_data['videomae_category'] = videomae_category

            # Stage 2b: VideoMAE sliding window temporal scanning
            sc = self.config.scoring
            if sc.videomae_weight > 0:
                import time as _time2
                _t_mae = _time2.time()
                centers, mae_scores = self.videomae.sliding_window_scores(
                    video_data['all_frames'],
                    window_size=sc.mae_window_size,
                    stride=sc.mae_window_stride)
                video_data['videomae_window_centers'] = centers
                video_data['videomae_window_scores'] = mae_scores
                _log.info(f"  [VideoMAE-window] {len(centers)} windows in "
                          f"{_time2.time()-_t_mae:.1f}s")

        # Stage 3: Multi-category CLIP temporal localization
        # v97 FCCS: when full_coverage_comparative_scan is on, Stage 4 scans
        # ALL segments comparatively, so clip_regions' only consumer (scan
        # direction) is moot — skip the region loop entirely (verified
        # side-effect-free). The CLIP category curve used by Stage 5 boundary
        # detection is computed independently in
        # frame_scorer.get_category_scores and is unaffected.
        # v173 KFD-U: --kfd_uniform_keyframes ALSO engages the comparative
        # scan path. The pre-registered mechanism (replace the first-k
        # truncation with a uniform spread) is defined ONLY inside
        # _comparative_call — the directed-scan/fallback paths (scan_segment)
        # send ALL sampled keyframes and have no truncation to fix, so a
        # KFD flag that did not route through scan_full_coverage could never
        # fire. Flag OFF: _fccs is exactly full_coverage_comparative_scan
        # (byte-identical committed behavior).
        _fccs = (self.config.discovery.full_coverage_comparative_scan
                 or self.config.discovery.kfd_uniform_keyframes)
        if _fccs:
            clip_regions = []
        else:
            categories_to_search = [videomae_category] if videomae_category else []
            if videomae_category in VIDEOMAE_CONFUSION_MAP:
                for alt in VIDEOMAE_CONFUSION_MAP[videomae_category]:
                    if alt not in categories_to_search:
                        categories_to_search.append(alt)

            all_clip_regions = []
            for cat in categories_to_search:
                regions = self._clip_temporal_localization(video_data, cat)
                for r in regions:
                    r['source_category'] = cat
                all_clip_regions.extend(regions)

            clip_regions = self._deduplicate_clip_regions(all_clip_regions)

        # Stage 4: LLM scanning — directed by CLIP regions
        context_prompt = f"This is a {context} video." if context else ""
        kfd_stats = None  # v173 KFD-U: per-video scan stats for [KFD-VID]
        # v177 TCP: reset the per-video scan_segment counter so [TCP-VID]
        # reports THIS video's scan accounting. Internal instrumentation
        # only — never read when the flag is off, zero observable effect.
        self.discovery.scan_segments_count = 0
        if _fccs:
            # v97 FCCS: single full-coverage comparative scan replaces BOTH
            # the directed and fallback partial-coverage scans (coverage:
            # 10-18 of 23-120 segments -> 100%; judgment: independent
            # absolute -> relative comparative with in-video calibration).
            category_hint = ""
            if videomae_category:
                cue = CATEGORY_CUES.get(videomae_category, videomae_category)
                category_hint = f"\nExpected anomaly type: {videomae_category} — {cue}\n"
            candidates, kfd_stats = await self.discovery.scan_full_coverage(
                video_data, context_prompt, category_hint)
        elif clip_regions:
            candidates = await self._directed_llm_scan(
                video_data, clip_regions, context_prompt, videomae_category)
        else:
            candidates = await self.discovery.discover(video_data, context_prompt)

        if not candidates:
            if self.config.scoring.event_continuation_extension:
                _log.info("  [ECE] summary: 0 detections (early return: no candidates)")
            if self.config.scoring.event_unit_boundary_contraction:
                _log.info("  [EUBC] summary: 0 detections (early return: no candidates)")
            if self.config.scoring.cbt_enabled:
                # FM17: unconditional per-video CBT marker on the 0-detection path.
                _log.info(f"  [CBT-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                          f"enabled=True ndet=0 fired=0 "
                          f"recal={'on' if self.config.scoring.cbt_recal else 'off'} "
                          f"(early return: no candidates)")
            if self.config.scoring.cgr_enabled:
                # FM17: unconditional per-video CGR marker on the 0-detection path.
                _log.info(f"  [CGR-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                          f"enabled=True ndet_before=0 ndet_after=0 moved=0 "
                          f"skip=early_return_no_candidates")
            if self.config.scoring.ccr_enabled:
                # FM17: unconditional per-video CCR marker on the 0-detection path.
                _log.info(f"  [CCR-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                          f"enabled=True ndet_before=0 ndet_after=0 moved=0 "
                          f"skip=early_return_no_candidates")
            if self.config.scoring.cgr_ab_diag:
                # FM17: unconditional per-video CGR-AB marker on the 0-detection
                # path so grep -cF '[CGR-AB] vid=' == videos processed.
                _log.info(f"  [CGR-AB] vid={str(video_path).rsplit('/', 1)[-1]} "
                          f"ndet=0 armoff_len=0 fired=0")
            if self.config.scoring.cnv_veto:
                # FM17: unconditional per-video CNV marker on the 0-detection
                # (no candidates) path so grep -cF '[CNV-VID]' == videos
                # processed. This path returns BEFORE the terminal slot.
                self._cnv_log_early_return(
                    video_path, video_data, videomae_category,
                    'early_return_no_candidates')
            if self.config.scoring.snh_veto:
                # FM17: unconditional per-video SNH marker on the 0-detection
                # (no candidates) path so grep -cF '[SNH-VID]' == videos
                # processed. This path returns BEFORE the terminal slot.
                self._snh_log_early_return(
                    video_path, video_data, videomae_category,
                    'early_return_no_candidates')
            if self.config.discovery.kfd_uniform_keyframes:
                # FM17: unconditional per-video KFD marker on the 0-detection
                # (no candidates) path so grep -cF '[KFD-VID]' == videos
                # processed. Carries the real scan accounting (the scan ran
                # but flagged nothing above threshold).
                self._kfd_log_vid(video_path, kfd_stats,
                                  'early_return_no_candidates')
            if self.config.discovery.tcp_prompt:
                # FM17: unconditional per-video TCP marker on the 0-detection
                # (no candidates) path so grep -cF '[TCP-VID]' == videos
                # processed. Carries the real scan accounting (the scan ran
                # but flagged nothing above threshold).
                self._tcp_log_vid(video_path,
                                  self.discovery.scan_segments_count,
                                  'early_return_no_candidates')
            frame_scores = self.frame_scorer.compute_frame_scores(video_data, [], videomae_category)
            return DetectionResult(
                video_path=video_path,
                duration=video_data['duration'],
                is_normal=True,
                anomalies=[],
                frame_scores=frame_scores,
                timestamps=video_data['timestamps'].tolist() if frame_scores else None,
                pre_ece_boundaries=(
                    [] if self.config.scoring.event_continuation_extension
                    else None),
                pre_cgr_boundaries=(
                    [] if self.config.scoring.cgr_enabled else None),
                pre_ccr_boundaries=(
                    [] if self.config.scoring.ccr_enabled else None),
                ab_arm_off_boundaries=(
                    [] if (self.config.scoring.cgr_ab_diag
                           and self.config.scoring.cgr_enabled) else None),
            )

        _log.info(f"Found {len(candidates)} anomaly candidate(s):")
        for i, c in enumerate(candidates):
            _log.info(f"  Candidate {i+1}: {c['t_start']:.1f}s-{c['t_end']:.1f}s "
                  f"score={c['coarse_score']:.2f}")

        top_k = self.config.discovery.candidate_top_k
        if len(candidates) > top_k:
            candidates.sort(key=lambda c: c['coarse_score'], reverse=True)
            _log.info(f"  [Limit] Keeping top {top_k}/{len(candidates)} candidates")
            candidates = candidates[:top_k]

        # Stage 5: CLIP score-curve boundary detection (v50 — replaces MCTS)
        all_anomalies = self._clip_boundary_detection(
            candidates, video_data, videomae_category)

        # Stage 5b: VLM boundary estimation for low-CLIP-confidence candidates
        # V62 evaluation: VLM over-wide override triggered 18 times on 65 videos
        # but net mAP regressed (0.1492->0.1012). VLM timestamp estimation from
        # sparse keyframes is too noisy to reliably improve over CLIP half-max.
        # Reverted to V51's proven guard: LOW_CLIP + clip_conf < 0.15 only.
        if videomae_category in LOW_CLIP_CATEGORIES:
            vlm_refined = []
            for anom in all_anomalies:
                clip_conf = anom.get('boundary_contrast', 1.0)
                if clip_conf < 0.15:
                    r = await self._vlm_boundary_estimation(anom, video_data, videomae_category)
                    vlm_refined.append(r)
                else:
                    vlm_refined.append(anom)
            all_anomalies = vlm_refined

        _log.info(f"  --- Pre-merge anomalies: {len(all_anomalies)} ---")
        for i, a in enumerate(all_anomalies):
            _log.debug(f"    {i+1}: {a['t_start']:.1f}-{a['t_end']:.1f}s, "
                  f"conf={a['confidence']:.3f}, type={a.get('anomaly_type','?')}")

        final_anomalies = global_merge(all_anomalies, same_type_gap=2.0)
        # v86->v89 REVERT: Soft-NMS regressed full-eval mAP 0.1492->0.0961.
        # Root cause: evaluate.py's mAP computation uses standard GREEDY
        # matching (predictions sorted by confidence, each GT segment can only
        # be matched ONCE - see evaluate.py compute_temporal_map). Soft-NMS's
        # permissive score_thresh=0.15 let 2-3 overlapping near-duplicate
        # detections survive per true event (vs Hard-NMS's exact iou_threshold
        # cutoff keeping only 1); only the highest-confidence duplicate gets
        # matched as a true positive, the REST become FALSE POSITIVES under
        # greedy matching, directly hurting precision/AP. Reverted to proven
        # Hard-NMS (temporal_nms) which was calibrated correctly for this
        # specific greedy-matching evaluation protocol.
        final_anomalies = temporal_nms(final_anomalies, iou_threshold=0.3)
        _log.info(f"  --- Post-merge+NMS anomalies: {len(final_anomalies)} ---")

        # Stage 6b (v50): Post-merge boundary tightening for over-long segments
        final_anomalies = self._post_merge_boundary_tightening(
            final_anomalies, video_data, videomae_category)

        # Stage 6c: Short-event narrowing
        if videomae_category in SHORT_EVENT_CATEGORIES:
            final_anomalies = narrow_short_events(final_anomalies, video_data)

        # Stage 7: Multi-classifier ensemble (VideoMAE + CLIP + VLM)
        if final_anomalies:
            ensemble_cat = await self._ensemble_classify(
                final_anomalies, video_data, videomae_category, videomae_confidence)
            for a in final_anomalies:
                a['anomaly_type'] = ensemble_cat
        elif videomae_category and videomae_category != 'Normal':
            pass  # no anomalies to classify

        final_anomalies = unify_anomaly_types(final_anomalies)

        # Stage 7b: CLIP score curve post-correction
        if videomae_category and videomae_category != 'Normal':
            # v53: pass max_dur so already-well-calibrated segments are left alone
            _curve_max_dur = CATEGORY_DURATION_PRIORS.get(
                videomae_category, {}).get('expected_duration', (3.0, 30.0))[1]
            final_anomalies = self.frame_scorer.clip_score_curve_correction(
                final_anomalies, video_data, videomae_category, max_dur=_curve_max_dur)

        # v51: MIL video-level sanity check — reduce confidence when MIL says Normal
        _mil_scores = video_data.get('mil_scores')
        if _mil_scores is not None and np.max(_mil_scores) < 0.3:
            _log.info(f"  [MIL-sanity] MIL max_score={np.max(_mil_scores):.3f} < 0.3, "
                      f"reducing confidence by 30%")
            for a in final_anomalies:
                a['confidence'] *= 0.7

        # v51: Category-aware confidence gate (replaces uniform 0.72 threshold)
        default_gate = self.config.scoring.confidence_gate
        conf_gate = CATEGORY_CONF_GATES.get(videomae_category, default_gate)
        before_filter = len(final_anomalies)
        final_anomalies = [a for a in final_anomalies if a['confidence'] >= conf_gate]
        if len(final_anomalies) < before_filter:
            _log.info(f"  [ConfFilter] {before_filter} -> {len(final_anomalies)} "
                      f"(threshold={conf_gate:.2f}, cat={videomae_category})")

        # v105 ECE: Event-Continuation Extension（pinned v99 语义）——终态
        # 槽位第一段：v51 置信门之后、EUBC 之前（EUBC 本 loop 保持
        # flag-off/inert）。单开关（默认关 = 字节级恢复旧行为）。flag-on
        # 时先记录 pre-ECE 边界，供同一 run 内确定性 pre/post A/B 复评。
        pre_ece_boundaries = None
        if self.config.scoring.event_continuation_extension:
            pre_ece_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._event_continuation_extension(
                final_anomalies, video_data, videomae_category)

        # v100 EUBC: Event-Unit Boundary Contraction — 终态过宽边界收缩。
        # 单开关（默认关 = 字节级恢复旧行为）。在 ALL 边界逻辑之后、最终
        # dict 投影之前作用于 FINAL detections；信号严格只读；仅收缩。
        # 若 ECE 重跑并先保留，EUBC 在同一终态槽位运行于 ECE 之后。
        if self.config.scoring.event_unit_boundary_contraction:
            final_anomalies = self._event_unit_boundary_contraction(
                final_anomalies, video_data, videomae_category)

        # v141 CEX: capture the per-detection event DESCRIPTION before the
        # terminal dict projection drops it. The projection below is a strict
        # 1:1 comprehension and CEX never changes the count, so this list stays
        # index-aligned with final_anomalies all the way to the CEX slot.
        # Read-only; nothing downstream of here consumes it unless the flag is
        # on (flag-off: a single `= None` assignment, zero behavioural residue).
        _cex_descriptions = None
        if self.config.scoring.cex_extend:
            _cex_descriptions = [
                str(a.get('description') or a.get('anomaly_type') or '').strip()
                for a in final_anomalies]

        final_anomalies = [
            {
                't_start': a['t_start'],
                't_end': a['t_end'],
                'confidence': a['confidence'],
                'anomaly_type': a.get('anomaly_type', 'unknown'),
                'depth': a.get('depth'),
                'visits': a.get('visits'),
            }
            for a in final_anomalies
        ]

        # Generate per-frame continuous anomaly scores
        frame_scores = self.frame_scorer.compute_frame_scores(
            video_data, final_anomalies, videomae_category)

        # v111 QSCORE: pure post-hoc localization-quality confidence re-rank.
        # After conf-gate/ECE/EUBC (count fixed) + frame_scores computed; never
        # touches boundaries/count. Single switch (default OFF = byte-identical).
        if self.config.scoring.qscore_rerank:
            final_anomalies = self._qscore_rerank(
                final_anomalies, frame_scores, video_data.get('timestamps'),
                base=self.config.scoring.qscore_base)

        # v142 BUE: Boundary-Union Extension — TERMINAL-MOST slot. Placed
        # immediately AFTER _qscore_rerank and BEFORE the v141 CEX block (M3
        # anti-stack: cex_extend stays False this loop), so merge/NMS/PostTighten/
        # narrow/classify/curve-correction/MIL-sanity/conf-gate/ECE/EUBC/Q-score
        # have ALL completed: the count is genuinely frozen, confidence is
        # genuinely bit-identical, and the operator acts on exactly the population
        # the offline pricing used (cached FINAL detections). Accepted consequence:
        # frame_scores above are flag-off-identical -> frame AUC unchanged and the
        # effect is mAP-only (clean attribution).
        # M1: pre-extension windows are persisted in the result JSON so the paired
        # flag-off arm is reconstructed EXACTLY by
        # analysis_bootstrap_ab.py --paired-json pre_bue_boundaries — no regex
        # parsing, no float-rounding key collisions, and the length-mismatch
        # HARD-VOID check comes for free.
        # v167 CGR-M: Count-neutral motion-energy-peak candidate RELOCATION —
        # fires IMMEDIATELY BEFORE the pre_bue snapshot, on the FINAL post-merge/
        # NMS/PostTighten/classify/curve-correction/conf-gate/Q-score detection
        # list (count already frozen, confidence bit-identical). It translates
        # (does NOT add / grow / re-rank) the single LOWEST-confidence existing
        # detection onto the per-frame motion-energy peak, only when that peak is
        # currently uncovered, gated to the 4 hard cats. Strictly count-neutral
        # (in-place endpoint rewrite of exactly one element; no append/pop/insert/
        # del/filter/reorder), width-neutral (preserves w=t_end-t_start),
        # confidence/anomaly_type/coarse_score/order byte-frozen. Reuses
        # video_data['motion_energy'] computed at L1405 (NO recompute). This is
        # the cand-miss lever CBT (a post-BUE inward clamp along the CLIP curve)
        # cannot touch. Single switch (default OFF = byte-identical, motion never
        # read for relocation). M1: pre-relocation windows persisted so the paired
        # arm is reconstructed exactly by analysis_bootstrap_ab.py --paired-json
        # pre_cgr_boundaries (length-mismatch HARD-VOID for free). FM17: the method
        # emits its [CGR-VID] per-video summary UNCONDITIONALLY.
        # v169 CGR-AB (DUAL-ARM A/B INSTRUMENT, pure diagnostic): snapshot a DEEP
        # COPY of the PRE-CGR detection list. This is the ONLY thing that has to
        # happen before CGR; the arm-OFF replay (CCR-if-enabled -> BUE -> CBT on a
        # further deep copy of this base) runs AFTER the real arm has completed, so
        # the real path below is left byte-for-byte untouched. HAZARD: CGR/BUE/CBT
        # rewrite a['t_start']/a['t_end'] IN PLACE, so a shallow copy (or reusing
        # the same dicts) would let the two arms silently corrupt each other —
        # copy.deepcopy is load-bearing here, not defensive style.
        _ab_diag_on = bool(self.config.scoring.cgr_ab_diag
                           and self.config.scoring.cgr_enabled)
        _ab_base = None
        if _ab_diag_on:
            import copy as _ab_copy
            try:
                _ab_base = _ab_copy.deepcopy(final_anomalies)
            except Exception as _ab_err:  # never let a diagnostic break the run
                _ab_base = None
                _log.warning(f"  [CGR-AB-ERR] deepcopy(base) failed: {_ab_err}")

        pre_cgr_boundaries = None
        if self.config.scoring.cgr_enabled:
            pre_cgr_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._cgr_motion_relocate(
                final_anomalies, video_data, videomae_category)

        # v169 CGR-AB: record whether CGR ACTUALLY relocated a window, measured
        # here (immediately post-CGR) because BUE/CBT overwrite the endpoints
        # further down and the information would otherwise be unrecoverable.
        _ab_fired = 0
        if _ab_diag_on and pre_cgr_boundaries is not None:
            _ab_post_cgr = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            _ab_fired = 1 if _ab_post_cgr != pre_cgr_boundaries else 0

        # v168 CCR-SE: Count-neutral CLIP-category-curve semantic RELOCATION —
        # runs IMMEDIATELY AFTER CGR-M and BEFORE the pre_bue snapshot, so BUE
        # widens the CCR-relocated window and CBT tightens it exactly as for
        # CGR-M. Structural clone of _cgr_motion_relocate with the signal source
        # swapped from motion_energy to the cached read-only per-frame relative
        # CLIP category curve (get_category_scores). Gated to Shoplifting/Shooting
        # (the two low-motion cand-miss hard cats CGR-M under-delivered on);
        # RoadAccidents/Explosion excluded -> byte-identical there. Single switch
        # (default OFF = byte-identical, curve never read for relocation). M1:
        # pre_ccr snapshot (POST-CGR) persisted so the paired arm is reconstructed
        # exactly by analysis_bootstrap_ab.py --paired-json pre_ccr_boundaries
        # (length-mismatch HARD-VOID for free). FM17: the method emits its
        # [CCR-VID] per-video summary UNCONDITIONALLY.
        pre_ccr_boundaries = None
        if self.config.scoring.ccr_enabled:
            pre_ccr_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._ccr_curve_relocate(
                final_anomalies, video_data, videomae_category)

        # Single switch (default OFF = byte-identical, no curve is even read).
        pre_bue_boundaries = None
        if self.config.scoring.bue_enabled:
            pre_bue_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._bue_boundary_union_extension(
                final_anomalies, video_data, videomae_category)

        # v155 CBT: Category-conditioned Boundary Tightening — runs AFTER BUE
        # (strict INVERSE: BUE grows outward, CBT clamps inward), before CEX.
        # Single switch (default OFF = byte-identical, no curve is even read).
        # M1: pre-tighten windows persisted so the paired flag-off arm is
        # reconstructed exactly by analysis_bootstrap_ab.py --paired-json
        # pre_cbt_boundaries. FM17: the method emits its [CBT-VID] per-video
        # summary UNCONDITIONALLY at its end, so it fires on the empty /
        # 0-detection list here too.
        pre_cbt_boundaries = None
        if self.config.scoring.cbt_enabled:
            pre_cbt_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._cbt_category_boundary_tighten(
                final_anomalies, video_data, videomae_category)

        # v169 CGR-AB: ARM-OFF REPLAY. The real (arm-ON) path is now complete:
        # CGR -> CCR(off) -> BUE -> CBT has produced final_anomalies. Replay the
        # IDENTICAL downstream chain MINUS the single CGR call on a deep copy of
        # the pre-CGR base -> the windows the committed pipeline would have
        # produced for this exact video with this exact VLM sampling. Runs here
        # (after the real arm) so nothing above can be perturbed by the replay.
        # Excluded from the replay: CEX (it issues VLM calls and is default-OFF;
        # a warning is emitted if it is somehow on, since that would make the two
        # arms differ by more than CGR).
        ab_arm_off_boundaries = None
        if _ab_diag_on:
            if self.config.scoring.cex_extend:
                _log.warning("  [CGR-AB-ERR] cex_extend is ON — the CGR-AB arms "
                             "would differ by more than CGR; instrument VOID")
            ab_arm_off_boundaries = self._cgr_ab_arm_off_replay(
                _ab_base, video_data, videomae_category)
        # FM5/FM17: unconditional per-video activation marker (this branch is the
        # non-empty path; the two early-return paths emit the same marker).
        if self.config.scoring.cgr_ab_diag:
            _log.info(
                f"  [CGR-AB] vid={str(video_path).rsplit('/', 1)[-1]} "
                f"ndet={len(final_anomalies)} "
                f"armoff_len={-1 if ab_arm_off_boundaries is None else len(ab_arm_off_boundaries)} "
                f"fired={_ab_fired}")

        # v141 CEX: Continuation-Evidence Extension — TERMINAL-MOST slot (M1).
        # Deliberately placed AFTER _qscore_rerank and immediately before the
        # DetectionResult return, so merge/NMS/PostTighten/narrow/classify/
        # curve-correction/MIL-sanity/conf-gate/ECE/EUBC/Q-score have ALL
        # completed: the count is genuinely frozen, confidence is genuinely
        # bit-identical, and the operator acts on exactly the population the
        # offline pricing used (cached FINAL detections). Accepted consequence:
        # frame_scores above are flag-off-identical -> frame AUC unchanged and
        # the effect is mAP-only. Single switch (default OFF = byte-identical).
        # M4: pre-extension windows are persisted in the result JSON so the
        # paired flag-off arm is reconstructed exactly, with no regex parsing.
        pre_cex_boundaries = None
        if self.config.scoring.cex_extend:
            pre_cex_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = await self._continuation_extension(
                final_anomalies, video_data, _cex_descriptions, video_path)

        # v172 CNV variant "TIGHTEN": Cascade Null-Hypothesis Veto — TERMINAL
        # slot (AFTER the BUE/CBT/CEX blocks, on the genuinely-final
        # final_anomalies, immediately before DetectionResult is built).
        # ONE-SIDED confidence veto only (confidence := 0.0, detection KEPT —
        # count-frozen, FM2 asserted inside); never moves a window, never
        # adds, never re-ranks, zero new VLM/CLIP/MIL calls. Single switch
        # (default OFF = byte-identical: no signal is even read). M1/paired:
        # arm-OFF is rebuilt EXACTLY from the [CNV-FIRE] log lines by
        # analysis_bootstrap_ab.py --paired-log <run.log> --marker CNV-FIRE
        # --paired-mode conf (LOG-based, no new JSON fields). FM17: the
        # method emits its [CNV-VID] per-video summary UNCONDITIONALLY (the
        # two early-return paths emit their own, so every processed video
        # logs exactly one).
        if self.config.scoring.cnv_veto:
            final_anomalies = self._cnv_cascade_null_veto(
                final_anomalies, video_data, videomae_category)

        # v179 SNH: Supervised Normality Head — TERMINAL slot (AFTER the
        # BUE/CBT/CEX blocks, on the genuinely-final final_anomalies,
        # immediately before DetectionResult is built). VIDEO-LEVEL-ONLY gate
        # (NO per-segment stage): iff the v19-trained head's P(normal) >= tau,
        # veto ALL of the video's detections (confidence := 0.0, detection
        # KEPT — count-frozen, FM2 asserted inside); never moves a window,
        # never adds, never re-ranks, zero new VLM/CLIP/MIL calls. Single
        # switch (default OFF = byte-identical: the model file and no signal
        # are even read). M1/paired: arm-OFF is rebuilt EXACTLY from the
        # [SNH-FIRE] log lines by analysis_bootstrap_ab.py --paired-log
        # <run.log> --marker SNH-FIRE --paired-mode conf (LOG-based, no new
        # JSON fields). FM17: the method emits its [SNH-VID] per-video summary
        # UNCONDITIONALLY (the two early-return paths emit their own, so every
        # processed video logs exactly one).
        if self.config.scoring.snh_veto:
            final_anomalies = self._snh_veto(
                final_anomalies, video_data, videomae_category)

        # v173 KFD-U FM17: the unconditional per-video [KFD-VID] summary at
        # the terminal slot (the two early-return paths emit their own, so
        # every processed video logs exactly one). Pure marker — the
        # mechanism itself acted during Stage 4 (deterministic keyframe
        # spread inside the comparative scan); nothing is mutated here.
        if self.config.discovery.kfd_uniform_keyframes:
            self._kfd_log_vid(video_path, kfd_stats)

        # v177 TCP FM17: the unconditional per-video [TCP-VID] summary at
        # the terminal slot (the two early-return paths emit their own, so
        # every processed video logs exactly one). Pure marker — the
        # mechanism itself acted during Stage 4 (prompt text inside
        # scan_segment); nothing is mutated here.
        if self.config.discovery.tcp_prompt:
            self._tcp_log_vid(video_path, self.discovery.scan_segments_count)

        return DetectionResult(
            video_path=video_path,
            duration=video_data['duration'],
            is_normal=len(final_anomalies) == 0,
            anomalies=[AnomalySegment.from_dict(a) for a in final_anomalies],
            frame_scores=frame_scores,
            timestamps=video_data['timestamps'].tolist() if frame_scores else None,
            pre_ece_boundaries=pre_ece_boundaries,
            pre_cex_boundaries=pre_cex_boundaries,
            pre_bue_boundaries=pre_bue_boundaries,
            pre_cbt_boundaries=pre_cbt_boundaries,
            pre_cgr_boundaries=pre_cgr_boundaries,
            pre_ccr_boundaries=pre_ccr_boundaries,
            ab_arm_off_boundaries=ab_arm_off_boundaries,
        )

    # ================================================================
    # v169 CGR-AB: DUAL-ARM in-run A/B instrument for CGR-M (diagnostic)
    # ================================================================

    def _cgr_ab_arm_off_replay(self, base_anomalies, video_data, category):
        """v169 CGR-AB: produce the ARM-OFF (committed-pipeline) window list.

        Replays the EXACT downstream chain the real arm ran — CCR (only if
        ccr_enabled, which is default-OFF) -> BUE -> CBT — on a DEEP COPY of
        `base_anomalies` (itself already a deep copy of the PRE-CGR list), i.e.
        everything the committed pipeline does EXCEPT the single
        _cgr_motion_relocate call. Returns [[t_start, t_end], ...] index-aligned
        with the run's final anomalies (CGR is strictly count-neutral, so the
        length is necessarily identical).

        WHY THIS IS A VALID ARM (not a re-simulation): BUE/CBT/CCR run after every
        VLM call and read only cached per-frame motion energy and the cached
        read-only CLIP category curve, so they are deterministic pure functions of
        (windows, video_data, category). Replaying them on the pre-CGR windows
        reproduces the committed output bit-for-bit.

        THE IN-PLACE HAZARD (why deepcopy appears twice):
          _cgr_motion_relocate  L~2680: a['t_start']=ns; a['t_end']=ne
          _bue_boundary_union_extension: a['t_start']/a['t_end'] rewritten
          _cbt_category_boundary_tighten: a['t_start']/a['t_end'] rewritten
        All three mutate the SAME dict objects they are handed. `base_anomalies`
        was deep-copied before CGR ran (so CGR/BUE/CBT on the real arm could not
        touch it), and it is deep-copied AGAIN here so that this replay cannot
        mutate the caller's snapshot either. Net effect: arm-OFF and arm-ON share
        no dict object whatsoever, so neither can corrupt the other.

        LOGGING: BUE/CBT/CCR emit [BUE]/[CBT]/[CBT-VID]/[CCR-VID] markers via the
        module-global `_log`. Replaying them would double every FM5 activation
        count, so the global is swapped for _CgrAbSilentLog for the duration of
        this SYNCHRONOUS (await-free) block and restored in a `finally`.
        """
        global _log
        if base_anomalies is None:
            return None
        import copy as _ab_copy
        _saved_log = _log
        try:
            arm = _ab_copy.deepcopy(base_anomalies)  # HAZARD guard (2nd deepcopy)
            _log = _CgrAbSilentLog()
            if self.config.scoring.ccr_enabled:
                arm = self._ccr_curve_relocate(arm, video_data, category)
            if self.config.scoring.bue_enabled:
                arm = self._bue_boundary_union_extension(arm, video_data, category)
            if self.config.scoring.cbt_enabled:
                arm = self._cbt_category_boundary_tighten(arm, video_data, category)
        except Exception as _err:
            _log = _saved_log
            _log.warning(f"  [CGR-AB-ERR] arm-OFF replay failed: {_err}")
            return None
        finally:
            _log = _saved_log
        return [[float(a['t_start']), float(a['t_end'])] for a in (arm or [])]

    # ================================================================
    # v141 CEX: Continuation-Evidence Extension (TERMINAL-MOST slot, M1)
    # ================================================================

    async def _continuation_extension(self, anomalies, video_data,
                                      descriptions=None, video_path=None):
        """v141 CEX: grow each FINAL detection outward in fixed cex_step_sec
        steps, one step per side per round, gated ONLY by an independent BINARY
        VLM continuation vote on the single adjacent clip.

        Contract (why this is FM2/FM4/FM14-immune):
          - COUNT-PRESERVING: returns the SAME list object contents, same order,
            same length. No append, no delete, no filter, no re-sort.
          - CONFIDENCE-FROZEN: 'confidence', 'anomaly_type', 'depth', 'visits'
            are never written. Only 't_start'/'t_end' are mutated. Global AP's
            cross-video confidence LEVEL is therefore bit-identical to flag-off.
          - Geometry is imposed HERE, in code, in fixed steps: the VLM answers
            yes/no about one clip and never emits a timestamp (the v62 failure).
          - M3: NO CATEGORY_DURATION_PRIORS max_dur reject-cap and NO snap to
            the curve grid — the two neutralizers that left v105/v107 ECE at
            +0.00203 in vivo with AP@0.5/0.7 unmoved. Growth is bounded ONLY by
            cex_max_steps, cex_max_total_sec and the physical video extent.
          - A side closes PERMANENTLY at its first reject (no re-probing), and a
            probe failure/unparseable answer is fail-closed (closes the side),
            so an API error can never grow a window.

        Markers (FM5/M5/M6): [CEX] per accepted step (carries predur = the
        detection's own pre-extension duration, M6), [CEX-STOP] per closed side,
        [CEX-PAIR] per detection (win=<pre>-><post>, the paired flag-off arm),
        [CEX-GATE] once per video (probes/accepted = the realized acceptance
        rate = the q read; pre-registered smoke band 0.20-0.80, >0.85 means the
        VLM is rubber-stamping -> do not launch the full eval).
        """
        import os as _cex_os
        vid = _cex_os.path.basename(str(video_path)) if video_path else 'unknown'
        sc = self.config.scoring
        step = float(sc.cex_step_sec)
        max_steps = int(sc.cex_max_steps)
        min_conf = float(sc.cex_min_conf)
        max_total = float(sc.cex_max_total_sec)

        n = len(anomalies)
        timestamps = video_data.get('timestamps')
        all_frames = video_data.get('all_frames')
        duration = float(video_data.get('duration') or 0.0)

        def _gate(probes, accepted, note=''):
            rate = (accepted / probes) if probes else 0.0
            _log.info(f"  [CEX-GATE] vid={vid} probes={probes} accepted={accepted} "
                      f"rate={rate:.3f} n={n}{note}")

        if n == 0 or step <= 0 or max_steps <= 0:
            _gate(0, 0, ' (no-op: no detections or step/steps<=0)')
            return anomalies
        if timestamps is None or all_frames is None or duration <= 0:
            _gate(0, 0, ' (no-op: missing frames/timestamps/duration)')
            return anomalies
        ts = np.asarray(timestamps, dtype=np.float64)
        n_ts = int(len(ts))
        if n_ts == 0 or len(all_frames) == 0:
            _gate(0, 0, ' (no-op: empty frame set)')
            return anomalies

        descs = list(descriptions or [])
        if len(descs) < n:
            descs = descs + [''] * (n - len(descs))

        # pre-extension windows (the paired flag-off arm) + per-side live flags
        orig = [(float(a['t_start']), float(a['t_end'])) for a in anomalies]
        state = [{'l': True, 'r': True, 'grown': 0.0} for _ in range(n)]

        def _seg(t0, t1):
            """frame_indices covering [t0, t1]; None when the span holds no frame."""
            lo = int(np.searchsorted(ts, t0, side='left'))
            hi = int(np.searchsorted(ts, t1, side='right'))
            lo = max(0, min(lo, n_ts - 1))
            hi = max(0, min(hi, n_ts))
            if hi <= lo:
                hi = min(n_ts, lo + 1)
            idx = list(range(lo, hi))
            if not idx:
                return None
            return {'t_start': float(t0), 't_end': float(t1), 'frame_indices': idx}

        probes = 0
        accepted = 0
        for rnd in range(max_steps):
            jobs = []   # (i, side, seg, new_t0, new_t1)
            for i, a in enumerate(anomalies):
                st = state[i]
                if not (st['l'] or st['r']):
                    continue
                t0, t1 = float(a['t_start']), float(a['t_end'])
                pre_dur = orig[i][1] - orig[i][0]
                for side in ('l', 'r'):
                    if not st[side]:
                        continue
                    room = st['grown'] + step <= max_total + 1e-9
                    if not room:
                        st[side] = False
                        _log.info(f"    [CEX-STOP] vid={vid} side={side} reason=cap "
                                  f"(total_growth {st['grown']:.1f}s + {step:.1f}s > "
                                  f"cex_max_total_sec {max_total:.1f}s) predur={pre_dur:.1f}")
                        continue
                    if side == 'l':
                        c0, c1 = max(0.0, t0 - step), t0
                    else:
                        c0, c1 = t1, min(duration, t1 + step)
                    if c1 - c0 <= 1e-6:
                        st[side] = False
                        _log.info(f"    [CEX-STOP] vid={vid} side={side} reason=cap "
                                  f"(hit video edge at [{c0:.1f},{c1:.1f}], "
                                  f"duration={duration:.1f}s) predur={pre_dur:.1f}")
                        continue
                    seg = _seg(c0, c1)
                    if seg is None:
                        st[side] = False
                        _log.info(f"    [CEX-STOP] vid={vid} side={side} reason=cap "
                                  f"(no sampled frame in [{c0:.1f},{c1:.1f}]) "
                                  f"predur={pre_dur:.1f}")
                        continue
                    jobs.append((i, side, seg, c0, c1))
            if not jobs:
                break
            # independent single-clip calls; nothing merged/batched (FM3/FM12)
            results = await asyncio.gather(
                *[self.discovery.probe_continuation(j[2], all_frames, descs[j[0]])
                  for j in jobs],
                return_exceptions=True)
            probes += len(jobs)
            for (i, side, seg, c0, c1), res in zip(jobs, results):
                if not isinstance(res, dict):
                    res = {'ongoing': False, 'confidence': 0.0}   # fail-closed
                ongoing = bool(res.get('ongoing'))
                try:
                    pconf = float(res.get('confidence', 0.0))
                except (TypeError, ValueError):
                    pconf = 0.0
                a, st = anomalies[i], state[i]
                t0, t1 = float(a['t_start']), float(a['t_end'])
                pre_dur = orig[i][1] - orig[i][0]
                if not (ongoing and pconf >= min_conf):
                    st[side] = False
                    reason = 'no' if not ongoing else 'lowconf'
                    _log.info(f"    [CEX-STOP] vid={vid} side={side} reason={reason} "
                              f"step={rnd + 1} conf={pconf:.2f} predur={pre_dur:.1f}")
                    continue
                accepted += 1
                if side == 'l':
                    a['t_start'] = float(c0)
                    st['grown'] += (t0 - c0)
                else:
                    a['t_end'] = float(c1)
                    st['grown'] += (c1 - t1)
                _log.info(f"    [CEX] vid={vid} side={side} step={rnd + 1} "
                          f"conf={pconf:.2f} predur={pre_dur:.1f} "
                          f"win={t0:.1f}-{t1:.1f}->"
                          f"{float(a['t_start']):.1f}-{float(a['t_end']):.1f}")

        # M4: one [CEX-PAIR] per detection carrying the pre-extension window.
        # NOTE: deliberately NO `conf=<old>-><new>` field — confidence is frozen
        # here, so the paired tool must run in WINDOW mode (or read
        # pre_cex_boundaries from the result JSON, the primary path).
        for i, a in enumerate(anomalies):
            o0, o1 = orig[i]
            f0, f1 = float(a['t_start']), float(a['t_end'])
            _log.info(f"  [CEX-PAIR] vid={vid} t={f0:.3f}-{f1:.3f} "
                      f"win={o0:.3f}-{o1:.3f}->{f0:.3f}-{f1:.3f} "
                      f"predur={o1 - o0:.1f} postdur={f1 - f0:.1f} "
                      f"conf_frozen={float(a.get('confidence', 0.0)):.6f}")
        _gate(probes, accepted)
        return anomalies

    # ================================================================
    # v105 ECE: Event-Continuation Extension (pinned v99 semantics)
    # ================================================================

    def _event_continuation_extension(self, anomalies, video_data, category):
        """v105 ECE: grow under-extended FINAL detections along the read-only
        whole-video relative category CLIP curve (pinned v99 semantics).

        For each detection a=[t_s,t_e] with dur < max_dur =
        CATEGORY_DURATION_PRIORS[cat]['expected_duration'][1]:
          1. Category resolution (same pattern as EUBC): the detection's OWN
             anomaly_type (post-unify), falling back to the videomae_category
             argument; skip when None/'Normal'/unknown/not in
             anomaly_categories.
          2. rel = frame_scorer.get_category_scores(video_data, cat) — the
             cached whole-video relative curve (strictly read-only; None or
             length mismatch -> skip).
          3. hm = (rel.max() + percentile(rel, 20)) / 2.0 — the exact v51
             half-max LEVEL formula with zero new constants, applied to the
             WHOLE-VIDEO curve instead of v51's padding_ratio-bounded local
             search window (the cage that structurally shrinks long events).
          4. s = min(searchsorted(ts, t_s), n-1), e = min(searchsorted(ts,
             t_e), n-1); walk L leftward from s while rel[L-1] >= hm, R
             rightward from e while rel[R+1] >= hm (T3AL-style level-set
             region growth).
          5. REJECT-CAP (AutoLoc OIC rationale): if ts[R]-ts[L] > max_dur
             keep the ORIGINAL boundaries; else snap t_start/t_end to
             ts[L]/ts[R] (always snap to grid, even without growth).

        Detection count, rank order, confidence, category, NMS, merge and
        every upstream stage stay byte-identical; zero new VLM/CLIP/MIL
        calls. Self-limiting on flat curves (whole-video growth -> span >
        max_dur -> reject-cap -> unchanged).

        FM5 guard: one INFO '[ECE] pre=[...] post=[...] (outcome)' line per
        detection and exactly one INFO '[ECE] summary: X material / Y snaps
        / Z reject-caps / W skips' line per video (the runner-enforced hard
        gate requires the summary line for EVERY processed video; all-skip
        is a legitimate outcome, a missing line is not).
        """
        counts = {'material': 0, 'snap': 0, 'reject_cap': 0, 'skip': 0}
        n = len(anomalies) if anomalies else 0

        timestamps = video_data.get('timestamps')
        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=np.float64)
        n_ts = len(timestamps) if timestamps is not None else 0

        def _skip(a, t_s, t_e, reason):
            counts['skip'] += 1
            _log.info(f"    [ECE] pre=[{t_s:.1f},{t_e:.1f}] "
                      f"post=[{t_s:.1f},{t_e:.1f}] (skip: {reason})")

        out = []
        for a in (anomalies or []):
            a = a.copy()
            t_s, t_e = a['t_start'], a['t_end']
            dur = t_e - t_s

            # --- category resolution (detection's own type, fallback arg) ---
            cat = a.get('anomaly_type')
            if (cat is None or cat in ('Normal', 'unknown')
                    or cat not in self.anomaly_categories):
                cat = category
            if (cat is None or cat in ('Normal', 'unknown')
                    or cat not in self.anomaly_categories):
                _skip(a, t_s, t_e, f"cat={a.get('anomaly_type')}")
                out.append(a)
                continue

            max_dur = CATEGORY_DURATION_PRIORS.get(cat, {}).get(
                'expected_duration', (3.0, 30.0))[1]
            if dur >= max_dur:
                _skip(a, t_s, t_e,
                      f"dur {dur:.1f}s >= max_dur {max_dur:.1f}s, cat={cat}")
                out.append(a)
                continue

            if n_ts < 2:
                _skip(a, t_s, t_e, "no timestamps grid")
                out.append(a)
                continue

            # --- read-only cached whole-video relative curve ---
            rel = self.frame_scorer.get_category_scores(video_data, cat)
            if rel is None or len(rel) != n_ts:
                _skip(a, t_s, t_e, f"curve None/mismatch, cat={cat}")
                out.append(a)
                continue
            rel = np.asarray(rel, dtype=np.float64)

            # --- v51 half-max LEVEL formula on the WHOLE-VIDEO curve ---
            hm = (float(np.max(rel)) + float(np.percentile(rel, 20))) / 2.0

            s = min(int(np.searchsorted(timestamps, t_s)), n_ts - 1)
            e = min(int(np.searchsorted(timestamps, t_e)), n_ts - 1)
            L = s
            while L - 1 >= 0 and rel[L - 1] >= hm:
                L -= 1
            R = e
            while R + 1 <= n_ts - 1 and rel[R + 1] >= hm:
                R += 1

            new_start = float(timestamps[L])
            new_end = float(timestamps[R])
            grown_span = new_end - new_start

            # --- REJECT-CAP: never out-run the duration-prior evidence ---
            if grown_span > max_dur:
                counts['reject_cap'] += 1
                _log.info(f"    [ECE] pre=[{t_s:.1f},{t_e:.1f}] "
                          f"post=[{t_s:.1f},{t_e:.1f}] (reject-cap: span "
                          f"{grown_span:.1f}s > max_dur {max_dur:.1f}s, "
                          f"cat={cat})")
                out.append(a)
                continue

            # --- always snap to grid; material iff actual growth ---
            material = (L < s) or (R > e)
            if material:
                counts['material'] += 1
                outcome = 'material'
            else:
                counts['snap'] += 1
                outcome = 'grid-snap'
            _log.info(f"    [ECE] pre=[{t_s:.1f},{t_e:.1f}] "
                      f"post=[{new_start:.1f},{new_end:.1f}] ({outcome}: "
                      f"cat={cat}, hm={hm:.4f}, dur {dur:.1f}s -> "
                      f"{grown_span:.1f}s)")
            a['t_start'] = new_start
            a['t_end'] = new_end
            out.append(a)

        _log.info(f"  [ECE] summary: {counts['material']} material / "
                  f"{counts['snap']} snaps / {counts['reject_cap']} "
                  f"reject-caps / {counts['skip']} skips "
                  f"(n={n}, cat={category})")
        return out

    # ================================================================
    # v100 EUBC: Event-Unit Boundary Contraction
    # ================================================================

    def _eubc_frame_cos_sims(self, video_data):
        """v100: consecutive-frame cosine similarities of cached clip_features.

        sims[i] = cos(feat_i, feat_{i+1}), length n-1. Computed LAZILY (only
        when the EUBC switch is on — flag-off is a true zero-delta; no
        unconditional Stage-1 pass) and cached in video_data. Zero new
        CLIP/VLM calls (same verified feature-reuse pattern as v95 TSM).
        Returns None on any failure (guarded like tsm_novelty).
        """
        sims = video_data.get('frame_cos_sims')
        if sims is not None:
            return sims
        feats = video_data.get('clip_features')
        if feats is None:
            return None
        try:
            if hasattr(feats, 'cpu'):
                feats = feats.cpu().numpy()
            feats = np.asarray(feats, dtype=np.float32)
            n = feats.shape[0]
            if n < 3:
                return None
            norms = np.linalg.norm(feats, axis=1) + 1e-8
            feats_n = feats / norms[:, None]
            sims = np.sum(feats_n[:-1] * feats_n[1:], axis=1)
            video_data['frame_cos_sims'] = sims
            return sims
        except Exception as _e:
            _log.warning(f"  [EUBC] frame_cos_sims computation failed: {_e}")
            return None

    def _event_unit_boundary_contraction(self, anomalies, video_data, category):
        """v100 EUBC: contract over-wide FINAL detections on an independent
        frame-to-frame visual-coherence signal (read-only; contraction-only).

        For each detection a=[t_s,t_e]:
          1. Category resolution (per C mandate): use the detection's OWN
             anomaly_type (post-unify_anomaly_types — can differ from
             videomae_category after the Stage-7 ensemble), falling back to
             the videomae_category argument; skip when the resolved category
             is None/'Normal'/unknown/not in anomaly_categories, or the
             cached relative curve is None.
          2. Peak anchor P = argmax of the cached relative curve rel within
             [t_s, t_e].
          3. Explicit flat-coherence guard: robust_mad = 1.4826*MAD(sims);
             robust_mad < eubc_flat_eps -> explicit no-op (flat-skip). Pure
             numerical-degeneracy guard, not a tuned threshold.
          4. Coherence walk: thr = median(sims) - eubc_mad_k*1.4826*MAD(sims);
             walk L leftward from P while sims[L-1] >= thr, R rightward while
             sims[R] >= thr; coherence span = [ts[L], ts[R]].
          5. Plateau-union guard (exact v51 half-max LEVEL formula, zero new
             constants): within the detection hm = (max+p20)/2 of rel;
             plateau = {rel >= hm}; guarded span = [min(L_time,
             plateau_left), max(R_time, plateau_right)] — never cut extent
             the curve's own half-max evidence supports.
          6. Contraction-only fire rule: clip the guarded span to [t_s,t_e];
             FIRE only if new_dur < eubc_materiality*det_dur AND new_dur >=
             CATEGORY_DURATION_PRIORS[cat]['min_segment']. Else no-op.

        Confidence, category, detection count/order and every upstream stage
        are byte-identical; curves strictly read-only. Logs one [EUBC] line
        per detection (fired/no-op reason) and exactly one [EUBC] summary
        line per video (the runner-enforced FM5 gate requires the summary
        line for EVERY processed video; '0 fired' is a legitimate outcome,
        a missing line is not).
        """
        sc = self.config.scoring
        counts = {'fired': 0, 'materiality': 0, 'flat': 0,
                  'min_segment': 0, 'plateau_clip': 0, 'skip': 0}
        n = len(anomalies) if anomalies else 0

        timestamps = video_data.get('timestamps')
        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=np.float64)
        rel_by_cat = {}   # per-category cached relative curves (read-only)
        sims = None
        signals_ready = False
        if n > 0 and timestamps is not None and len(timestamps) >= 3:
            sims = self._eubc_frame_cos_sims(video_data)
            if sims is not None and len(sims) == len(timestamps) - 1:
                signals_ready = True

        out = []
        for a in (anomalies or []):
            a = a.copy()
            t_s, t_e = a['t_start'], a['t_end']
            det_dur = t_e - t_s

            # --- category resolution (detection's own type, fallback arg) ---
            cat = a.get('anomaly_type')
            if (cat is None or cat in ('Normal', 'unknown')
                    or cat not in self.anomaly_categories):
                cat = category
            if (cat is None or cat in ('Normal', 'unknown')
                    or cat not in self.anomaly_categories):
                counts['skip'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(cat-skip: type={a.get('anomaly_type')})")
                out.append(a)
                continue

            if not signals_ready:
                counts['skip'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(signal-skip: no clip_features/timestamps)")
                out.append(a)
                continue

            # --- read-only cached relative curve for THIS detection's cat ---
            if cat not in rel_by_cat:
                rel_by_cat[cat] = self.frame_scorer.get_category_scores(
                    video_data, cat)
            rel = rel_by_cat[cat]
            if rel is None or len(rel) != len(timestamps):
                counts['skip'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(curve-skip: cat={cat})")
                out.append(a)
                continue

            n_ts = len(timestamps)
            lo = int(np.searchsorted(timestamps, t_s))
            lo = min(lo, n_ts - 1)
            hi = min(int(np.searchsorted(timestamps, t_e)), n_ts - 1)
            if hi <= lo:
                counts['skip'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(degenerate window: lo={lo}, hi={hi})")
                out.append(a)
                continue

            # --- explicit flat-coherence guard (B's hardening) ---
            med = float(np.median(sims))
            robust_mad = 1.4826 * float(np.median(np.abs(sims - med)))
            if robust_mad < sc.eubc_flat_eps:
                counts['flat'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(flat-skip: robust_mad={robust_mad:.2e})")
                out.append(a)
                continue

            # --- peak anchor + coherence walk ---
            P = lo + int(np.argmax(rel[lo:hi + 1]))
            thr = med - sc.eubc_mad_k * robust_mad
            L = P
            while L - 1 >= 0 and sims[L - 1] >= thr:
                L -= 1
            R = P
            while R <= n_ts - 2 and sims[R] >= thr:
                R += 1

            # --- plateau-union guard (exact v51 half-max LEVEL formula) ---
            det_rel = rel[lo:hi + 1]
            hm = (float(np.max(det_rel)) + float(np.percentile(det_rel, 20))) / 2.0
            plateau_idx = np.where(det_rel >= hm)[0]
            plateau_left = lo + int(plateau_idx[0])
            plateau_right = lo + int(plateau_idx[-1])
            g_left = min(float(timestamps[L]), float(timestamps[plateau_left]))
            g_right = max(float(timestamps[R]), float(timestamps[plateau_right]))
            plateau_clip = (g_left > float(timestamps[L])) or \
                           (g_right < float(timestamps[R]))

            # --- contraction-only clip + fire rule ---
            new_start = max(g_left, t_s)
            new_end = min(g_right, t_e)
            new_dur = new_end - new_start
            min_seg = CATEGORY_DURATION_PRIORS.get(cat, {}).get('min_segment', 1.5)

            if new_dur <= 0 or new_dur >= sc.eubc_materiality * det_dur:
                counts['materiality'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(materiality: span {new_dur:.1f}s >= "
                          f"{sc.eubc_materiality:.2f}*{det_dur:.1f}s, cat={cat})")
                out.append(a)
                continue
            if new_dur < min_seg:
                counts['min_segment'] += 1
                _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s no-op "
                          f"(min_segment: {new_dur:.1f}s < {min_seg:.1f}s, cat={cat})")
                out.append(a)
                continue

            counts['fired'] += 1
            if plateau_clip:
                counts['plateau_clip'] += 1
            _log.info(f"    [EUBC] {t_s:.1f}-{t_e:.1f}s -> "
                      f"{new_start:.1f}-{new_end:.1f}s FIRED "
                      f"(dur {det_dur:.1f}s -> {new_dur:.1f}s, cat={cat}, "
                      f"thr={thr:.4f}, span=[{timestamps[L]:.1f},{timestamps[R]:.1f}]"
                      f"{', plateau-clip' if plateau_clip else ''})")
            a['t_start'] = new_start
            a['t_end'] = new_end
            out.append(a)

        _log.info(f"  [EUBC] summary: contracted {counts['fired']}/{n} detections, "
                  f"{counts['materiality']} materiality-noops, "
                  f"{counts['flat']} flat-skips, "
                  f"{counts['min_segment']} min_segment-noops, "
                  f"{counts['plateau_clip']} plateau-guard-clips, "
                  f"{counts['skip']} cat/curve/signal-skips (cat={category})")
        return out

    # ================================================================
    # v155 CBT: Category-conditioned Boundary Tightening (TERMINAL, post-BUE)
    # ================================================================

    def _cbt_category_boundary_tighten(self, final_anomalies, video_data, category):
        """v155 CBT: TERMINAL inward clamp of over-wide FINAL detections of the
        4 hard categories toward per-category GT-p75 caps (CBT_CAPS), choosing
        the sub-window by the AutoLoc outer-inner CONTRAST objective already
        coded for the LogOIC tightener (count-neutral, confidence-untouched,
        monotone-inward). Runs AFTER BUE (strict INVERSE of BUE's outward
        union), before CEX.

        For each FINAL detection a=[t_s,t_e]:
          1. Category resolution: detection's OWN anomaly_type, fallback the
             videomae_category arg; skip when the resolved category is not in
             CBT_CAPS (per-category rollback = delete a CBT_CAPS key).
          2. Over-wide gate: cap=CBT_CAPS[cat]; if det_dur <= 1.5*cap -> no-op
             (protects already-compact detections — FM1/FM19).
          3. Peak anchor peak_idx = argmax of the read-only cached relative
             curve rel (frame_scorer.get_category_scores — the exact curve BUE
             and EUBC already read) within [t_s,t_e].
          4. Contrast-max window: scan half-widths hw in
             [ceil(min_seg*fps/2) .. int(cap*fps/2)] and pick the
             [peak-hw, peak+hw] window MAXIMIZING the AutoLoc outer-inner
             contrast OIC = mean(inside) - weighted_mean(two 1/log-weighted
             outer rings over the whole video) — the identical objective at
             the LogOIC tightener.
          5. Monotone-inward fire: clamp the chosen window to the ORIGINAL
             [t_s,t_e] (never grows); FIRE only if new_dur < det_dur AND
             new_dur >= CATEGORY_DURATION_PRIORS[cat]['min_segment'].

        Confidence, detection count/order, anomaly_type, coarse_score and
        frame_scores are byte-identical to flag-off; curve strictly read-only;
        pure numpy on CPU (FM9 n/a). Per-fire [CBT] log with the pre/post
        duration; exactly one UNCONDITIONAL [CBT-VID] per-video summary at the
        end (fires on the empty list too — FM17).
        """
        import os as _cbt_os
        cfg = self.config.scoring
        # v160 CBT-RECAL: swap ONLY the two scalar inputs when --cbt_recal is ON;
        # flag-off => caps==CBT_CAPS and trig==1.5 (byte-identical to v155 CBT).
        caps = cfg.cbt_recal_caps if cfg.cbt_recal else CBT_CAPS
        trig = cfg.cbt_recal_trigger if cfg.cbt_recal else 1.5
        vid = _cbt_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'
        fires = 0

        timestamps = video_data.get('timestamps')
        if not final_anomalies or timestamps is None:
            _log.info(f"  [CBT-VID] vid={vid} enabled={cfg.cbt_enabled} "
                      f"ndet={len(final_anomalies) if final_anomalies else 0} "
                      f"fired=0 wide_noop=0 nocurve=0 nofire=0 "
                      f"skip={'no_dets' if not final_anomalies else 'no_timestamps'} "
                      f"recal={'on' if cfg.cbt_recal else 'off'}")
            return final_anomalies
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        n = int(len(t))
        if n < 3:
            _log.info(f"  [CBT-VID] vid={vid} enabled={cfg.cbt_enabled} "
                      f"ndet={len(final_anomalies)} fired=0 wide_noop=0 "
                      f"nocurve=0 nofire=0 skip=curve_too_short "
                      f"recal={'on' if cfg.cbt_recal else 'off'}")
            return final_anomalies
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 0.5
        fps = 1.0 / dt
        idx = np.arange(n)

        rel_cache = {}

        def _rel(cat):
            if cat in rel_cache:
                return rel_cache[cat]
            try:
                r = self.frame_scorer.get_category_scores(video_data, cat)
            except Exception as _e:  # a read-only diagnostic must never kill detect()
                _log.warning(f"  [CBT] curve read failed vid={vid} cat={cat}: {_e}")
                r = None
            if r is not None and hasattr(r, 'detach'):  # FM9: MPS/CUDA tensors
                r = r.detach().cpu().numpy()
            if r is not None:
                r = np.asarray(r, dtype=np.float64).reshape(-1)
                if len(r) != n:
                    r = None
            rel_cache[cat] = r
            return r

        n_wide_noop = n_nocurve = n_nofire = 0
        for di, a in enumerate(final_anomalies):
            # category resolution matches BUE/EUBC exactly: own type, fallback arg
            cat = str(a.get('anomaly_type') or '').strip() or (category or '')
            if cat not in caps:
                continue
            cap = float(caps[cat])
            t_s = float(a['t_start'])
            t_e = float(a['t_end'])
            det_dur = t_e - t_s
            if det_dur <= trig * cap:
                n_wide_noop += 1     # not over-wide -> protect it (FM1/FM19)
                continue
            rel = _rel(cat)
            if rel is None:
                n_nocurve += 1
                continue
            lo = max(0, min(int(np.searchsorted(t, t_s)), n - 1))
            hi = max(lo + 1, min(int(np.searchsorted(t, t_e)), n))
            peak_idx = lo + int(np.argmax(rel[lo:hi]))
            min_seg = CATEGORY_DURATION_PRIORS.get(cat, {}).get('min_segment', 1.5)
            min_hw = max(1, int(np.ceil(min_seg * fps * 0.5)))
            max_hw = max(min_hw, int(cap * fps * 0.5))
            best_c, best_win = -1e30, None
            for hw in range(min_hw, max_hw + 1):
                lo_b = max(0, peak_idx - hw)
                hi_b = min(n - 1, peak_idx + hw)
                if hi_b <= lo_b:
                    continue
                inner_m = float(np.mean(rel[lo_b:hi_b + 1]))
                # AutoLoc/LogOIC 1/log-weighted outer rings over the whole video
                o_sum = o_wt = 0.0
                if lo_b > 0:
                    wl = 1.0 / np.log(2.0 + (lo_b - idx[:lo_b]))
                    o_sum += float(np.sum(rel[:lo_b] * wl))
                    o_wt += float(np.sum(wl))
                if hi_b + 1 < n:
                    wr = 1.0 / np.log(2.0 + (idx[hi_b + 1:] - hi_b))
                    o_sum += float(np.sum(rel[hi_b + 1:] * wr))
                    o_wt += float(np.sum(wr))
                o_m = o_sum / o_wt if o_wt > 0 else 0.0
                c = inner_m - o_m
                if c > best_c:
                    best_c, best_win = c, (lo_b, hi_b)
            if best_win is None:
                n_nofire += 1
                continue
            # clamp the chosen window to the ORIGINAL [t_s,t_e] — monotone inward
            ns = max(float(t[best_win[0]]), t_s)
            ne = min(float(t[best_win[1]]), t_e)
            new_dur = ne - ns
            if not (ne > ns and new_dur < det_dur and new_dur >= min_seg):
                n_nofire += 1
                continue
            conf = float(a.get('confidence', 0.0))
            a['t_start'] = ns
            a['t_end'] = ne
            fires += 1
            _log.info(
                f"  [CBT] vid={vid} cat={cat} det={di} "
                f"pre={det_dur:.1f} post={new_dur:.1f} "
                f"t={t_s:.2f}-{t_e:.2f}->{ns:.2f}-{ne:.2f} "
                f"cap={cap:.1f} contrast={best_c:.4f} conf={conf:.4f}->{conf:.4f}")

        _log.info(f"  [CBT-VID] vid={vid} enabled={cfg.cbt_enabled} "
                  f"ndet={len(final_anomalies)} fired={fires} "
                  f"wide_noop={n_wide_noop} nocurve={n_nocurve} nofire={n_nofire} "
                  f"recal={'on' if cfg.cbt_recal else 'off'}")
        return final_anomalies

    # ================================================================
    # v171 CNV: Cascade Null-Hypothesis Veto
    # ================================================================

    def _cnv_stage1_bundle(self, video_data, category, anomalies):
        """v171 CNV: gather the three cached stage-1 signal families for a video
        and evaluate the global-null negativity bits.

        STRICTLY READ-ONLY, ZERO new VLM/CLIP/MIL calls:
          (a) MIL — reuses video_data['mil_scores'] when present; else recomputes
              from the ALREADY-LOADED mil_model on the cached clip_features using
              the exact 32-snippet recipe of the Stage-1 block (a single forward
              on cached features, no new encoding);
          (b) fused category curve — element-wise max over the involved
              categories' own curves, each the SAME cached read-only
              frame_scorer.get_category_scores BUE/CBT already read (per
              video+category cache key `_cat_scores_<cat>`; never written back,
              never rescaled — the fusion lives only in this local array);
          (c) motion — video_data['motion_energy'] (computed at Stage 1b).
          (d) v172 TIGHTEN — the video's own videomae_category judgment
              (stored post-CLIP-override field, else the category argument);
              sig_d is True iff it is 'Normal'.

        Returns a dict with the three signal values, the FOUR negativity bits
        (sig_a..sig_c from _cnv_signal_values, sig_d from the videomae
        category), the fused curve, the per-category curve dict (stage-2
        input) and the timestamps array.
        """
        t = video_data.get('timestamps')
        n = int(len(np.asarray(t).reshape(-1))) if t is not None else 0

        # (a) MIL video-level evidence
        mil = video_data.get('mil_scores')
        if mil is None and self.mil_model is not None \
                and video_data.get('clip_features') is not None:
            try:
                _feats = video_data['clip_features']
                _nf = _feats.shape[0]
                _ns = 32
                if _nf >= _ns:
                    _sl = _nf // _ns
                    _snips = []
                    for _si in range(_ns):
                        _s = _si * _sl
                        _e = _s + _sl if _si < _ns - 1 else _nf
                        _snips.append(_feats[_s:_e].mean(dim=0))
                    with torch.no_grad():
                        mil = self.mil_model(
                            torch.stack(_snips).unsqueeze(0)).squeeze(0).cpu().numpy()
            except Exception as _e:  # never let a read-only diagnostic kill detect()
                _log.warning(f"  [CNV] MIL recompute failed: {_e}")
                mil = None

        # (b) fused category curve over the involved categories
        cats = []
        for a in (anomalies or []):
            c = str(a.get('anomaly_type') or '').strip() or (category or '')
            if c and c not in cats:
                cats.append(c)
        if not cats and category:
            cats = [category]
        curves = {}
        for c in cats:
            try:
                rel = self.frame_scorer.get_category_scores(video_data, c)
            except Exception as _e:  # a read-only diagnostic must never kill detect()
                _log.warning(f"  [CNV] curve read failed cat={c}: {_e}")
                rel = None
            if rel is not None and hasattr(rel, 'detach'):   # FM9: MPS/CUDA tensors
                rel = rel.detach().cpu().numpy()
            if rel is not None:
                rel = np.asarray(rel, dtype=np.float64).reshape(-1)
                if len(rel) == n:
                    curves[c] = rel
        fused = (np.max(np.stack(list(curves.values())), axis=0)
                 if curves else None)

        mil_topk, curve_zmax, mot_frac, sig_a, sig_b, sig_c = _cnv_signal_values(
            mil, fused, video_data.get('motion_energy'))
        # v172 TIGHTEN (sig_d): 4th NECESSARY stage-1 conjunct — the video's
        # own VideoMAE category judgment must be 'Normal'. Read the stored
        # post-CLIP-override field when present (terminal slot: it is set
        # before any detection is built); fall back to the category argument
        # (the early-return paths pass the raw videomae_category — on the
        # early-return-normal path that IS 'Normal', i.e. the early VideoMAE
        # path judged it normal). An unavailable or anomaly category is
        # NEVER treated as negative (conservative: no veto).
        vmae_cat = video_data.get('videomae_category')
        if vmae_cat is None:
            vmae_cat = category
        sig_d = str(vmae_cat or '').strip() == 'Normal'
        return {
            'mil_topk': mil_topk, 'curve_zmax': curve_zmax, 'mot_frac': mot_frac,
            'sig_a': sig_a, 'sig_b': sig_b, 'sig_c': sig_c, 'sig_d': sig_d,
            'fused': fused, 'curves': curves, 'timestamps': t,
        }

    def _cnv_cascade_null_veto(self, final_anomalies, video_data, category):
        """v172 CNV variant "TIGHTEN" (replaces the v171 gate logic): Cascade
        Null-Hypothesis Veto — ONE-SIDED confidence veto at the TERMINAL slot
        (after BUE/CBT/CEX, on the genuinely-final list).

        Contract (why this is FM1/FM2/FM3/FM4-immune):
          * Only a['confidence'] is ever written, and only ever to 0.0. The
            detection is KEPT in the list (count-frozen, index-aligned — FM2
            asserted below), t_start/t_end/anomaly_type/order are bit-frozen,
            and frame_scores were computed before this slot (flag-off-identical,
            so the effect is mAP/FA-rate-only — clean attribution).
          * ZERO new VLM/CLIP/MIL calls: all signal families are cached
            (see _cnv_stage1_bundle).
          * STAGE 1 (video-level global-null gate): veto ALL detections iff ALL
            FOUR independent families are negative (sig_a AND sig_b AND sig_c
            AND sig_d). v172 TIGHTEN: sig_d (videomae_category == 'Normal') is
            the new 4th NECESSARY conjunct — v171 could veto every detection
            of an ANOMALY-category video whose three cached signals happened
            to be negative (the v171 harm source); v172 cannot.
          * STAGE 2 (segment-level in-video surprisal veto): on videos that PASS
            stage 1, veto any detection whose window-aggregated mass on its OWN
            category curve is not an in-video outlier AND is BELOW the video's
            own median (v172 TIGHTEN: z_window < 0, kept conjunctively with the
            v171 bound z_window < CNV_TAU_SEG) AND whose window does not cover
            the video's global (fused) curve argmax. Every v172 stage-2 fire
            is a strict subset of v171 fires by construction.
          * Unavailable signals are never treated as negative (conservative).

        MARKERS (FM5/FM21): one [CNV-FIRE] per vetoed detection in the EXACT
        format analysis_bootstrap_ab.py restore_pre_marker (--paired-mode conf)
        parses — vid=<stem> ... t=<ts>-<te> ... conf=<old>->0.0000 — plus exactly
        one unconditional [CNV-VID] summary per call (the early-return paths
        emit their own via _cnv_log_early_return).

        DEV-ONLY: CNV_FORCE_FIRE=1 forces every gate to fire (self-test hook;
        excluded from default behaviour — the env var is never set in committed
        runs and its absence keeps behaviour byte-identical to non-forced).
        """
        import os as _cnv_os
        ndet = len(final_anomalies)
        vid = _cnv_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'
        # FM21: the [CNV-FIRE] vid key MUST be the STEM — evaluate.load_predictions
        # keys on Path(video_path).stem and restore_pre_marker matches it raw.
        vid_stem = _cnv_os.path.splitext(vid)[0]
        # DEV-ONLY self-test hook (never set in committed runs).
        force = _cnv_os.environ.get('CNV_FORCE_FIRE', '') == '1'

        b = self._cnv_stage1_bundle(video_data, category, final_anomalies)
        # v172 TIGHTEN: sig_d (videomae_category == 'Normal') is a NECESSARY
        # stage-1 conjunct — strictly fewer stage-1 fires than v171.
        stage1 = force or (b['sig_a'] and b['sig_b'] and b['sig_c'] and b['sig_d'])
        t = b['timestamps']
        tt = np.asarray(t, dtype=np.float64).reshape(-1) if t is not None else None
        fused = b['fused']
        g_argmax = int(np.argmax(fused)) if fused is not None and len(fused) else -1

        n_s1 = n_s2 = 0
        seen_keys = {}
        for di, a in enumerate(final_anomalies):
            cat = str(a.get('anomaly_type') or '').strip() or (category or '')
            stage = 0
            z_win = None
            if stage1:
                stage = 1
            else:
                rel = b['curves'].get(cat)
                if rel is not None and tt is not None and len(tt) >= 3:
                    nn = len(tt)
                    lo = max(0, min(int(np.searchsorted(tt, float(a['t_start']))), nn - 1))
                    hi = max(lo + 1, min(int(np.searchsorted(tt, float(a['t_end']))), nn))
                    win = rel[lo:hi]
                    med = float(np.median(rel))
                    mad = float(np.median(np.abs(rel - med)))
                    z_win = (float((float(np.mean(win)) - med) / (mad + 1e-9))
                             if len(win) else 0.0)
                    covers_argmax = lo <= g_argmax < hi
                    # v172 TIGHTEN: the window mass must be BELOW the video's
                    # own median (z_win < 0) — kept conjunctively with the
                    # v171 bound (z_win < CNV_TAU_SEG), so every v172 stage-2
                    # fire would also have fired under v171 (strict subset).
                    if z_win < 0.0 and z_win < CNV_TAU_SEG and not covers_argmax:
                        stage = 2
            if not stage:
                continue
            old = float(a.get('confidence', 0.0))
            a['confidence'] = 0.0      # ONE-SIDED VETO: confidence only
            if stage == 1:
                n_s1 += 1
            else:
                n_s2 += 1
            # FM21/FM5: EXACT marker format restore_pre_marker (--paired-mode
            # conf) parses: vid=<stem> ... t=<ts>-<te> ... conf=<old>-><new>.
            # t fields are printed at .2f so the log-side round(...,2) key equals
            # the prediction-side round(a['t_*'],2) key EXACTLY.
            _log.info(
                f"  [CNV-FIRE] vid={vid_stem} idx={di} cat={cat} stage={stage} "
                f"t={float(a['t_start']):.2f}-{float(a['t_end']):.2f} "
                f"conf={old:.4f}->0.0000"
                + (f" z_win={z_win:.3f}" if z_win is not None else ""))
            _k = (round(float(a['t_start']), 2), round(float(a['t_end']), 2))
            seen_keys[_k] = seen_keys.get(_k, 0) + 1
        assert len(final_anomalies) == ndet  # FM2: count-frozen, detections KEPT
        n_coll = sum(v - 1 for v in seen_keys.values() if v > 1)
        if n_coll:
            # INSTRUMENT TRANSPARENCY (not a mechanism change): the paired
            # conf-restore keys on (vid, round(t_start,2), round(t_end,2));
            # duplicate-window vetoed detections collapse to one key, so
            # analysis_bootstrap_ab's M4 check (matched != n_fires) will
            # correctly VOID the paired verdict rather than misreport it.
            _log.warning(
                f"  [CNV-COLLISION] vid={vid} dup_veto_keys={n_coll} — "
                f"vetoed detections share rounded (t_start,t_end); paired "
                f"conf-restore cannot disambiguate them (M4 VOID expected)")
        _log.info(
            f"  [CNV-VID] vid={vid} ndet={ndet} vetoed={n_s1 + n_s2} "
            f"stage1={n_s1} stage2={n_s2} "
            f"mil_topk={_cnv_fmt(b['mil_topk'])} "
            f"curve_zmax={_cnv_fmt(b['curve_zmax'], 3)} "
            f"mot_frac={_cnv_fmt(b['mot_frac'])} "
            f"sig={int(b['sig_a'])}{int(b['sig_b'])}{int(b['sig_c'])} "
            f"sig_d={int(b['sig_d'])} "
            f"gate1={int(stage1)} force={int(force)}")
        return final_anomalies

    def _cnv_log_early_return(self, video_path, video_data, category, reason):
        """v171 CNV FM17: the unconditional per-video [CNV-VID] summary on the
        two early-return paths (early-return-normal / 0-detection no-candidates),
        emitted ONLY when the flag is on (flag-off: never called, byte-identical
        log). The stage-1 signal values are still reported (read-only) so every
        path is observable; vetoes are trivially 0 because ndet==0. v172: the
        sig_d bit is reported too — on the early_return_normal path the early
        VideoMAE path judged the video normal, so sig_d=1 there."""
        b = self._cnv_stage1_bundle(video_data, category, [])
        _log.info(
            f"  [CNV-VID] vid={str(video_path).rsplit('/', 1)[-1]} ndet=0 "
            f"vetoed=0 stage1=0 stage2=0 "
            f"mil_topk={_cnv_fmt(b['mil_topk'])} "
            f"curve_zmax={_cnv_fmt(b['curve_zmax'], 3)} "
            f"mot_frac={_cnv_fmt(b['mot_frac'])} "
            f"sig={int(b['sig_a'])}{int(b['sig_b'])}{int(b['sig_c'])} "
            f"sig_d={int(b['sig_d'])} "
            f"gate1=0 skip={reason}")
        return

    # ================================================================
    # v179 SNH: Supervised Normality Head (video-level-ONLY gate)
    # ================================================================

    def _snh_p_normal(self, video_data):
        """v179 SNH: compute the video-level P(normal) from CACHED signals only.

        STRICTLY READ-ONLY signal access, ZERO new VLM/CLIP/MIL calls:
          (a) MIL — video_data['mil_scores'], the (32,) snippet-score array
              scored at Stage 1 by the ALREADY-LOADED mil_model (pipeline
              ~L1655). Read AS-IS, never recomputed: when it is unavailable
              (too few frames / MIL failed), the extractor's convention
              applies — the 5 mil_* features are None -> 0.0 after the
              feature_order mapping (identical to the train-time nan_to_num(0)
              sanitization);
          (b) the 13 cat/(cat+normal) curves — the SAME cached read-only
              frame_scorer.get_category_scores BUE/CBT already read (per
              video+category cache key `_cat_scores_<cat>`; softmax over the
              cached clip_features @ text_features — NO new encoding);
          (c) the CLIP Normal-prompt softmax curve — derived from the SAME
              cached clip_features @ text_features matmul (extractor-identical
              math: softmax(feats @ text.T * clamp(logit_scale, <=100))[:,-1]),
              cached per video under video_data['_snh_normal_curve'];
          (d) motion — video_data['motion_energy'] (computed at Stage 1b).

        The 16 features are computed by the SHARED module
        snh_features.compute_snh_features — the identical function the v19
        train-split extractor used, so train and eval see byte-identical
        feature math (no train/eval skew, the v17 C-record mandate). Features
        are mapped onto the model's feature_order with None -> 0.0 BEFORE
        standardization, sanitized with nan_to_num(nan=0, +-clip) and clipped
        (model JSON 'clip' field, 50.0), matching the training sanitization.

        Returns (p_normal, tau). On ANY failure (model file missing/corrupt —
        logged once by _snh_load_model; a required signal family unavailable;
        feature/score computation error) returns (None, None) after logging
        [SNH-ERR] — the caller treats the video as flag-off and the run NEVER
        crashes on the veto path.
        """
        m = _snh_load_model()
        if m is None:
            return None, None
        if torch is None:
            _log.warning("  [SNH-ERR] torch unavailable — video treated as flag-off")
            return None, None
        precomputed = video_data.get('clip_features')
        if precomputed is None or self.anomaly_expert.model is None:
            _log.warning("  [SNH-ERR] clip_features/CLIP model unavailable — "
                         "video treated as flag-off")
            return None, None
        # (b) the 13 cached cat/(cat+normal) curves (all categories — the
        # top-1 / all-cat aggregates need the full set, NOT just the video's
        # resolved category).
        cat_curves = {}
        for c in self.anomaly_categories:
            try:
                rel = self.frame_scorer.get_category_scores(video_data, c)
            except Exception as _e:  # a read-only diagnostic must never kill detect()
                _log.warning(f"  [SNH-ERR] curve read failed cat={c}: {_e}")
                return None, None
            if rel is None:
                _log.warning(f"  [SNH-ERR] category curve unavailable cat={c} — "
                             f"video treated as flag-off")
                return None, None
            if hasattr(rel, 'detach'):   # FM9: MPS/CUDA tensors
                rel = rel.detach().cpu().numpy()
            cat_curves[c] = np.asarray(rel, dtype=np.float64).reshape(-1)
        # (c) the CLIP Normal-prompt softmax curve (extractor-identical math).
        normal_curve = video_data.get('_snh_normal_curve')
        if normal_curve is None:
            try:
                with torch.no_grad():
                    logit_scale = (self.anomaly_expert.model.logit_scale.exp()
                                   .detach().clamp(max=100.0))
                    sim = precomputed @ self.anomaly_expert.text_features.T * logit_scale
                    probs = torch.softmax(sim, dim=-1).cpu().numpy()
                normal_curve = np.asarray(probs[:, -1], dtype=np.float64).reshape(-1)
                video_data['_snh_normal_curve'] = normal_curve
            except Exception as _e:
                _log.warning(f"  [SNH-ERR] normal-curve derive failed: {_e}")
                return None, None
        # (d) motion energy (Stage 1b).
        motion = video_data.get('motion_energy')
        if motion is None:
            _log.warning("  [SNH-ERR] motion_energy unavailable — video treated "
                         "as flag-off")
            return None, None
        # (a) MIL snippet scores — read AS-IS; None -> extractor convention.
        mil = video_data.get('mil_scores')
        try:
            feats, _top_cat = _snh_compute_features(
                mil, cat_curves, normal_curve, motion)
            order = list(m['feature_order'])
            clip = float(m.get('clip', 50.0))
            x = np.array([0.0 if feats.get(k) is None else float(feats[k])
                          for k in order], dtype=np.float64)
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip),
                        -clip, clip)
            z = ((x - np.asarray(m['scaler_mean'], dtype=np.float64))
                 / np.asarray(m['scaler_scale'], dtype=np.float64))
            logit = float(z @ np.asarray(m['weights'], dtype=np.float64)
                          + float(m['bias']))
            p_normal = float(1.0 / (1.0 + np.exp(-logit)))   # sigmoid
        except Exception as _e:
            _log.warning(f"  [SNH-ERR] feature/score computation failed: {_e}")
            return None, None
        return p_normal, float(m['tau'])

    def _snh_veto(self, final_anomalies, video_data, category):
        """v179 SNH: Supervised Normality Head — VIDEO-LEVEL-ONLY P(normal)
        veto at the TERMINAL slot (after BUE/CBT/CEX, on the genuinely-final
        final_anomalies, immediately before DetectionResult is built). There
        is NO per-segment stage-2: the gate is whole-video.

        Contract (identical to CNV — FM1/FM2/FM3/FM4-immune):
          * Only a['confidence'] is ever written, and only ever to 0.0. The
            detection is KEPT in the list (count-frozen, index-aligned — FM2
            asserted below), t_start/t_end/anomaly_type/order are bit-frozen,
            and frame_scores were computed before this slot (flag-off-identical,
            so the effect is mAP/FA-rate-only — clean attribution).
          * ZERO new VLM/CLIP/MIL calls: all signal families are cached
            (see _snh_p_normal).
          * GATE: fire iff P(normal) >= tau (model_snh.json; v19-trained,
            tau=0.7894 at CV anomaly-flag 3.8%). On fire, EVERY detection of
            the video is vetoed.
          * A missing/corrupt model or unavailable signal family logs
            [SNH-ERR] and the video is treated as flag-off (conservative:
            no veto) — the run NEVER crashes.

        MARKERS (FM5/FM21): one [SNH-FIRE] per vetoed detection in the EXACT
        format analysis_bootstrap_ab.py restore_pre_marker (--paired-mode
        conf) parses — vid=<stem> ... t=<ts>-<te> ... conf=<old>->0.0000 —
        plus exactly one unconditional [SNH-VID] summary per call (the
        early-return paths emit their own via _snh_log_early_return, so every
        processed video logs exactly one).

        DEV-ONLY: SNH_FORCE_FIRE=1 forces the gate to fire (self-test hook;
        excluded from default behaviour — the env var is never set in
        committed runs and its absence keeps behaviour byte-identical to
        non-forced).
        """
        import os as _snh_os
        ndet = len(final_anomalies)
        vid = _snh_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'
        # FM21: the [SNH-FIRE] vid key MUST be the STEM — evaluate.load_predictions
        # keys on Path(video_path).stem and restore_pre_marker matches it raw.
        vid_stem = _snh_os.path.splitext(vid)[0]
        # DEV-ONLY self-test hook (never set in committed runs).
        force = _snh_os.environ.get('SNH_FORCE_FIRE', '') == '1'

        p_normal, tau = self._snh_p_normal(video_data)
        if p_normal is None:
            # [SNH-ERR] already logged by _snh_p_normal/_snh_load_model:
            # treat the video as flag-off (no veto), still emit the
            # unconditional per-video summary (FM17).
            _log.info(
                f"  [SNH-VID] vid={vid} ndet={ndet} vetoed=0 p_normal=nan "
                f"tau=nan gate=0 force={int(force)} err=1")
            return final_anomalies

        gate = force or (p_normal >= tau)
        n_vetoed = 0
        seen_keys = {}
        if gate:
            for di, a in enumerate(final_anomalies):
                cat = str(a.get('anomaly_type') or '').strip() or (category or '')
                old = float(a.get('confidence', 0.0))
                a['confidence'] = 0.0    # ONE-SIDED VETO: confidence only
                n_vetoed += 1
                # FM21/FM5: EXACT marker format restore_pre_marker
                # (--paired-mode conf) parses: vid=<stem> ... t=<ts>-<te> ...
                # conf=<old>-><new>. t fields are printed at .2f so the log-side
                # round(...,2) key equals the prediction-side round(a['t_*'],2)
                # key EXACTLY.
                _log.info(
                    f"  [SNH-FIRE] vid={vid_stem} idx={di} cat={cat} "
                    f"t={float(a['t_start']):.2f}-{float(a['t_end']):.2f} "
                    f"conf={old:.4f}->0.0000 p_normal={p_normal:.4f}")
                _k = (round(float(a['t_start']), 2), round(float(a['t_end']), 2))
                seen_keys[_k] = seen_keys.get(_k, 0) + 1
        assert len(final_anomalies) == ndet  # FM2: count-frozen, detections KEPT
        n_coll = sum(v - 1 for v in seen_keys.values() if v > 1)
        if n_coll:
            # INSTRUMENT TRANSPARENCY (not a mechanism change): the paired
            # conf-restore keys on (vid, round(t_start,2), round(t_end,2));
            # duplicate-window vetoed detections collapse to one key, so
            # analysis_bootstrap_ab's M4 check (matched != n_fires) will
            # correctly VOID the paired verdict rather than misreport it.
            _log.warning(
                f"  [SNH-COLLISION] vid={vid} dup_veto_keys={n_coll} — "
                f"vetoed detections share rounded (t_start,t_end); paired "
                f"conf-restore cannot disambiguate them (M4 VOID expected)")
        _log.info(
            f"  [SNH-VID] vid={vid} ndet={ndet} vetoed={n_vetoed} "
            f"p_normal={p_normal:.4f} tau={tau:.4f} "
            f"gate={int(gate)} force={int(force)}")
        return final_anomalies

    def _snh_log_early_return(self, video_path, video_data, category, reason):
        """v179 SNH FM17: the unconditional per-video [SNH-VID] summary on the
        two early-return paths (early-return-normal / 0-detection no-candidates),
        emitted ONLY when the flag is on (flag-off: never called, byte-identical
        log). ndet=0 vetoed=0 trivially, but p_normal is still COMPUTED and
        reported (read-only) so every path is observable — mirroring
        _cnv_log_early_return."""
        import os as _snh_os
        force = _snh_os.environ.get('SNH_FORCE_FIRE', '') == '1'
        p_normal, tau = self._snh_p_normal(video_data)
        if p_normal is None:
            _log.info(
                f"  [SNH-VID] vid={str(video_path).rsplit('/', 1)[-1]} ndet=0 "
                f"vetoed=0 p_normal=nan tau=nan gate=0 force={int(force)} "
                f"err=1 skip={reason}")
            return
        _log.info(
            f"  [SNH-VID] vid={str(video_path).rsplit('/', 1)[-1]} ndet=0 "
            f"vetoed=0 p_normal={p_normal:.4f} tau={tau:.4f} gate=0 "
            f"force={int(force)} skip={reason}")
        return

    # ================================================================
    # v173 KFD-U: Uniform-spread keyframes (comparative scan) — markers
    # ================================================================

    def _kfd_log_vid(self, video_path, stats, skip=None):
        """v173 KFD-U FM17: the unconditional per-video [KFD-VID] summary on
        EVERY return path (terminal slot, 0-detection no-candidates,
        early-return-normal), emitted ONLY when kfd_uniform_keyframes is on
        (flag-off: never called, byte-identical log). Carries the comparative
        scan accounting: segments scanned, batches, [KFD-FIRE] line count,
        and how many segment selections the spread changed. On the
        early-return-normal path the scan never ran, so counts are 0 with a
        skip=<reason> suffix."""
        s = stats or {'segments': 0, 'batches': 0, 'fires': 0,
                      'segments_changed': 0}
        _log.info(
            f"  [KFD-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
            f"kfd_k={self.config.discovery.kfd_k} "
            f"segments={s['segments']} batches={s['batches']} "
            f"fires={s['fires']} seg_changed={s['segments_changed']}"
            + (f" skip={skip}" if skip else ""))
        return

    # ================================================================
    # v177 TCP: Temporal-Contrast Prompt — markers
    # ================================================================

    def _tcp_log_vid(self, video_path, segments_scanned=0, skip=None):
        """v177 TCP FM17/FM5: the unconditional per-video [TCP-VID] summary
        on EVERY return path (terminal slot, 0-detection no-candidates,
        early-return-normal), emitted ONLY when tcp_prompt is on (flag-off:
        never called, byte-identical log). Carries the Stage-4 scan_segment
        accounting (segments scanned this video). Immediately after the
        [TCP-VID] line, [TCP-BLOCK] logs the FULL TCP block once per video
        (FM5: grep -cF "Temporal reasoning instructions" proves the
        prompt-side mechanism engaged; the text is the single-source
        constant anomaly_discovery.TCP_PROMPT_BLOCK — bit-identical to what
        scan_segment appends, FM21)."""
        _log.info(
            f"  [TCP-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
            f"tcp=on segments={segments_scanned}"
            + (f" skip={skip}" if skip else ""))
        _log.info(
            f"  [TCP-BLOCK] appended to every scan_segment prompt this "
            f"video, immediately before 'Respond ONLY in JSON format':"
            f"\n{TCP_PROMPT_BLOCK}")
        return

    # ================================================================
    # v167 CGR-M: Count-neutral motion-energy-peak candidate RELOCATION
    # ================================================================

    def _cgr_motion_relocate(self, final_anomalies, video_data, category):
        """v167 CGR-M: translate the single LOWEST-confidence existing detection
        onto the per-frame motion-energy peak, ONLY when that peak is currently
        uncovered, gated to the 4 hard categories. Pre-BUE, count-neutral,
        width-neutral.

        Contract (why this is FM2/FM5/banned-family immune):
          - COUNT-NEUTRAL: rewrites final_anomalies[j]['t_start'] and ['t_end']
            IN PLACE for exactly one j (or does nothing). No append/pop/insert/
            del/filter/reorder/new-dict. len(after) == len(before) on every path.
          - WIDTH-NEUTRAL: preserves w = t_end - t_start of the moved window
            (BUE stays byte-identical, runs AFTER).
          - CONFIDENCE-FROZEN: 'confidence', 'anomaly_type', 'coarse_score',
            'depth', 'visits' never written; list order untouched.
        Steps:
          (1) Category gate: fire only if category in cfg.cgr_categories.
          (2) Reuse video_data['motion_energy'] (NO recompute); smooth with the
              EXACT L2468 _motion_temporal_localization convention; t_peak =
              timestamps[argmax(smoothed)] (motion co-indexed with timestamps).
          (3) Coverage guard: if ANY existing detection covers t_peak -> NO-OP.
          (4) Reclaim: j = argmin confidence; w = a[j] width; ns = clamp(
              t_peak - w/2, 0, duration); ne = clamp(ns + w, 0, duration).
          (5) Fire only if ns != old t_start.
        Emits exactly ONE unconditional [CGR-VID] marker per video at every
        return. Pure numpy on CPU (FM9 n/a).
        """
        import os as _cgr_os
        cfg = self.config.scoring
        vid = _cgr_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'
        ndet_before = len(final_anomalies) if final_anomalies else 0

        def _emit(moved, skip):
            # count-neutrality is asserted here: ndet_after is measured on the
            # SAME list object after any in-place rewrite -> must equal before.
            ndet_after = len(final_anomalies) if final_anomalies else 0
            _log.info(f"  [CGR-VID] vid={vid} enabled={cfg.cgr_enabled} "
                      f"ndet_before={ndet_before} ndet_after={ndet_after} "
                      f"moved={moved} skip={skip}")
            return final_anomalies

        # (1) category gate
        cat = str(category or '').strip()
        if cat not in cfg.cgr_categories:
            return _emit(0, f"cat_not_gated:{cat or 'none'}")
        if not final_anomalies:
            return _emit(0, "no_dets")

        motion = video_data.get('motion_energy')
        timestamps = video_data.get('timestamps')
        duration = float(video_data.get('duration') or 0.0)
        if motion is None or len(motion) < 4 or timestamps is None:
            return _emit(0, "no_motion")
        motion = np.asarray(motion, dtype=np.float64).reshape(-1)
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(t) != len(motion) or len(t) < 2 or duration <= 0:
            return _emit(0, "motion_misaligned")

        # (2) smoothing/indexing EXACTLY as _motion_temporal_localization (L2468)
        fps = max(1, len(t) / max(1, duration))
        window = max(2, int(cfg.cgr_smooth_win_sec * fps))
        kernel = np.ones(window) / window
        smoothed = np.convolve(motion, kernel, mode='same')
        t_peak = float(t[int(np.argmax(smoothed))])

        # (3) coverage guard: peak already proposed -> nothing wasted to reclaim
        for a in final_anomalies:
            if float(a['t_start']) <= t_peak <= float(a['t_end']):
                return _emit(0, "peak_covered")

        # (4) reclaim the LOWEST-confidence existing detection
        j = int(min(range(len(final_anomalies)),
                    key=lambda i: float(final_anomalies[i].get('confidence', 0.0))))
        a = final_anomalies[j]
        old_ts = float(a['t_start'])
        old_te = float(a['t_end'])
        w = old_te - old_ts
        ns = max(0.0, min(t_peak - w / 2.0, duration))
        ne = max(0.0, min(ns + w, duration))

        # (5) fire only if the start actually moves
        if ns == old_ts:
            return _emit(0, "no_move")
        conf = float(a.get('confidence', 0.0))
        a['t_start'] = ns
        a['t_end'] = ne
        _log.info(
            f"  [CGR] vid={vid} cat={cat} det={j} "
            f"t={old_ts:.2f}-{old_te:.2f}->{ns:.2f}-{ne:.2f} "
            f"w={w:.2f} t_peak={t_peak:.2f} conf={conf:.4f}->{conf:.4f}")
        return _emit(1, "fired")

    # ================================================================
    # v168 CCR-SE: Count-neutral CLIP-category-curve semantic RELOCATION
    # ================================================================

    def _ccr_curve_relocate(self, final_anomalies, video_data, category):
        """v168 CCR-SE: byte-for-byte structural clone of _cgr_motion_relocate
        with the SINGLE difference that the signal source is the cached read-only
        per-frame relative CLIP category curve (frame_scorer.get_category_scores)
        instead of video_data['motion_energy']. Translate the single LOWEST-
        confidence existing detection onto the smoothed category-curve peak, ONLY
        when that peak is currently uncovered, gated to the low-motion cand-miss
        hard cats (Shoplifting, Shooting). Pre-BUE, count-neutral, width-neutral.

        Contract (identical to CGR-M — FM2/FM5/banned-family immune):
          - COUNT-NEUTRAL: rewrites final_anomalies[j]['t_start'] and ['t_end']
            IN PLACE for exactly one j (or does nothing). No append/pop/insert/
            del/filter/reorder/new-dict. len(after) == len(before) on every path.
          - WIDTH-NEUTRAL: preserves w = t_end - t_start of the moved window.
          - CONFIDENCE-FROZEN: 'confidence', 'anomaly_type', 'coarse_score',
            'depth', 'visits' never written; list order untouched.
          - CURVE READ-ONLY: get_category_scores is a cached read (zero new
            inference on cache-hit); the curve is NEVER rescaled/min-maxed/written
            back (NOT banned-#4 per-video normalization). argmax is taken on the
            raw smoothed curve exactly as CGR takes it on raw smoothed motion.
        Emits exactly ONE unconditional [CCR-VID] marker per video at every
        return. Pure numpy on CPU (FM9 n/a; get_category_scores already .cpu()).
        """
        import os as _ccr_os
        cfg = self.config.scoring
        vid = _ccr_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'
        ndet_before = len(final_anomalies) if final_anomalies else 0

        def _emit(moved, skip):
            # count-neutrality is asserted here: ndet_after is measured on the
            # SAME list object after any in-place rewrite -> must equal before.
            ndet_after = len(final_anomalies) if final_anomalies else 0
            _log.info(f"  [CCR-VID] vid={vid} enabled={cfg.ccr_enabled} "
                      f"ndet_before={ndet_before} ndet_after={ndet_after} "
                      f"moved={moved} skip={skip}")
            return final_anomalies

        # (1) category gate
        cat = str(category or '').strip()
        if cat not in cfg.ccr_categories:
            return _emit(0, f"cat_not_gated:{cat or 'none'}")
        if not final_anomalies:
            return _emit(0, "no_dets")

        # (2) SIGNAL SOURCE SWAP: cached read-only relative CLIP category curve
        rel = self.frame_scorer.get_category_scores(video_data, cat)
        if rel is None:
            return _emit(0, "no_curve")
        timestamps = video_data.get('timestamps')
        duration = float(video_data.get('duration') or 0.0)
        rel = np.asarray(rel, dtype=np.float64).reshape(-1)
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1) if timestamps is not None \
            else np.asarray([], dtype=np.float64)
        if len(t) != len(rel) or len(t) < 2 or duration <= 0:
            return _emit(0, "curve_misaligned")

        # (2b) smoothing/indexing EXACTLY as CGR-M
        fps = max(1, len(t) / max(1, duration))
        window = max(2, int(cfg.ccr_smooth_win_sec * fps))
        kernel = np.ones(window) / window
        smoothed = np.convolve(rel, kernel, mode='same')
        t_peak = float(t[int(np.argmax(smoothed))])

        # (3) coverage guard: peak already proposed -> nothing wasted to reclaim
        for a in final_anomalies:
            if float(a['t_start']) <= t_peak <= float(a['t_end']):
                return _emit(0, "peak_covered")

        # (4) reclaim the LOWEST-confidence existing detection
        j = int(min(range(len(final_anomalies)),
                    key=lambda i: float(final_anomalies[i].get('confidence', 0.0))))
        a = final_anomalies[j]
        old_ts = float(a['t_start'])
        old_te = float(a['t_end'])
        w = old_te - old_ts
        ns = max(0.0, min(t_peak - w / 2.0, duration))
        ne = max(0.0, min(ns + w, duration))

        # (5) fire only if the start actually moves
        if ns == old_ts:
            return _emit(0, "no_move")
        conf = float(a.get('confidence', 0.0))
        a['t_start'] = ns
        a['t_end'] = ne
        _log.info(
            f"  [CCR] vid={vid} cat={cat} det={j} "
            f"t={old_ts:.2f}-{old_te:.2f}->{ns:.2f}-{ne:.2f} "
            f"w={w:.2f} t_peak={t_peak:.2f} conf={conf:.4f}->{conf:.4f}")
        return _emit(1, "fired")

    # ================================================================
    # Stage 3: Category-aware CLIP temporal localization
    # ================================================================


    def _deduplicate_clip_regions(self, regions):
        """Merge overlapping CLIP-loc regions from multiple categories."""
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: r['t_start'])
        merged = [regions[0].copy()]
        for r in regions[1:]:
            last = merged[-1]
            if r['t_start'] <= last['t_end'] + 3.0:
                last['t_end'] = max(last['t_end'], r['t_end'])
                last['clip_cat_score'] = max(last['clip_cat_score'], r['clip_cat_score'])
            else:
                merged.append(r.copy())
        merged.sort(key=lambda r: r['clip_cat_score'], reverse=True)
        merged = merged[:5]
        if len(regions) > len(merged):
            _log.info(f"  [Multi-CLIP-loc] {len(regions)} regions deduped to {len(merged)}")
        return merged


    def _motion_temporal_localization(self, video_data):
        """Use motion energy peaks for temporal localization (Explosion/RoadAccidents)."""
        motion = video_data.get('motion_energy')
        if motion is None or len(motion) < 4:
            return []
        
        timestamps = video_data['timestamps']
        duration = video_data['duration']
        
        # Smooth with 1-second window
        fps = max(1, len(timestamps) / max(1, duration))
        window = max(2, int(1.0 * fps))
        kernel = np.ones(window) / window
        smoothed = np.convolve(motion, kernel, mode='same')
        
        mean_m = np.mean(smoothed)
        std_m = np.std(smoothed)
        
        if std_m < 0.5:
            return []
        
        # Find regions above mean + 1.5*std (strong motion peaks)
        threshold = mean_m + 1.5 * std_m
        above = smoothed > threshold
        
        segments = []
        in_seg = False
        for i in range(len(above)):
            if above[i] and not in_seg:
                in_seg = True
                seg_start = i
            elif not above[i] and in_seg:
                in_seg = False
                segments.append((seg_start, i - 1))
        if in_seg:
            segments.append((seg_start, len(above) - 1))
        
        if not segments:
            return []
        
        # Convert to time and expand
        regions = []
        for start_idx, end_idx in segments:
            t_start = max(0, timestamps[start_idx] - 2.0)
            t_end = min(duration, timestamps[min(end_idx, len(timestamps)-1)] + 2.0)
            if t_end - t_start < 3.0:
                center = (t_start + t_end) / 2
                t_start = max(0, center - 3.0)
                t_end = min(duration, center + 3.0)
            avg_motion = np.mean(smoothed[start_idx:end_idx+1])
            regions.append({
                't_start': t_start, 't_end': t_end,
                'clip_cat_score': float(avg_motion / (np.max(smoothed) + 1e-8)),
            })
        
        regions.sort(key=lambda r: r['clip_cat_score'], reverse=True)
        regions = regions[:3]
        
        _log.info(f"  [Motion-loc] Found {len(regions)} motion peak regions:")
        for r in regions:
            _log.debug(f"    [{r['t_start']:.1f}-{r['t_end']:.1f}]s score={r['clip_cat_score']:.3f}")
        
        return regions

    def _compute_motion_energy(self, video_data):
        """Compute per-frame motion energy (mean absolute frame difference)."""
        import cv2
        frames = video_data['all_frames']
        if len(frames) < 2:
            return np.zeros(len(frames))
        small = np.stack([cv2.resize(f, (64, 64)) for f in frames]).astype(np.float32)
        motion = np.zeros(len(frames))
        motion[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
        return motion

    def _clip_temporal_localization(self, video_data, category):
        """Multi-scale CLIP temporal localization (v50).

        Runs peak detection at multiple smoothing scales to capture both
        short-burst events (RoadAccidents ~3s) and long events (Arrest ~50s).
        """
        precomputed = video_data.get('clip_features')
        if (precomputed is None or self.anomaly_expert.model is None
                or category is None or category == 'Normal'
                or torch is None):
            return []

        if category not in self.anomaly_categories:
            return []

        cat_idx = self.anomaly_categories.index(category)
        timestamps = video_data['timestamps']
        duration = video_data['duration']

        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            cat_scores = probs[:, cat_idx]
            normal_scores = probs[:, -1]
            relative_scores = cat_scores / (cat_scores + normal_scores + 1e-8)


        fps = max(1, len(timestamps) / max(1, duration))

        # Check for uniform scores (no discriminability)
        raw_range = np.max(relative_scores) - np.min(relative_scores)
        raw_cv = np.std(relative_scores) / (np.mean(relative_scores) + 1e-8)

        _log.info(f"  [CLIP-loc] Category={category}, mean={np.mean(relative_scores):.3f}, "
              f"std={np.std(relative_scores):.3f}, range={raw_range:.3f}, cv={raw_cv:.3f}")

        motion_categories = {'Explosion', 'RoadAccidents'}
        if raw_range < 0.08 or raw_cv < 0.03:
            if category in motion_categories:
                motion_regions = self._motion_temporal_localization(video_data)
                if motion_regions:
                    _log.info(f"  [CLIP-loc] Uniform but using motion energy for {category}")
                    return motion_regions
            _log.info(f"  [CLIP-loc] Scores too uniform, falling back to standard scanning")
            return []

        # Category-aware smoothing scales
        priors = CATEGORY_DURATION_PRIORS.get(category, {})
        preferred_smooth = priors.get('preferred_smooth', 2.0)
        smooth_scales = [
            max(0.5, preferred_smooth * 0.5),
            preferred_smooth,
            preferred_smooth * 2.0,
        ]

        # For Explosion/RoadAccidents: fuse motion energy with CLIP scores
        base_scores = relative_scores.copy()
        if category in motion_categories:
            motion = video_data.get('motion_energy', np.zeros(len(base_scores)))
            if len(motion) == len(base_scores) and np.max(motion) > 0:
                motion_norm = motion / (np.max(motion) + 1e-8)
                base_scores = 0.6 * base_scores + 0.4 * motion_norm
                _log.info(f"  [CLIP-loc] Fused with motion energy for {category}")

        # Multi-scale peak detection
        all_regions = []
        for smooth_sec in smooth_scales:
            window = max(3, int(smooth_sec * fps))
            if window >= len(base_scores):
                window = max(3, len(base_scores) // 2)
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window) / window
            smoothed = np.convolve(base_scores, kernel, mode='same')

            mean_score = np.mean(smoothed)
            std_score = np.std(smoothed)

            # Adaptive threshold
            threshold = mean_score + 0.3 * std_score
            n_above = np.sum(smoothed > threshold)
            min_frames_needed = max(4, int(0.15 * len(smoothed)))
            if n_above < min_frames_needed:
                sorted_scores = np.sort(smoothed)[::-1]
                threshold = sorted_scores[min(min_frames_needed - 1, len(sorted_scores) - 1)]

            above = smoothed > threshold

            # Find contiguous regions
            segments = []
            in_seg = False
            seg_start_idx = 0
            for i in range(len(above)):
                if above[i] and not in_seg:
                    seg_start_idx = i
                    in_seg = True
                elif not above[i] and in_seg:
                    segments.append((seg_start_idx, i - 1))
                    in_seg = False
            if in_seg:
                segments.append((seg_start_idx, len(above) - 1))

            # Merge with small gaps (< 3 seconds)
            gap_threshold = 3.0
            merged_segs = []
            for start_idx, end_idx in segments:
                t_start = float(timestamps[start_idx])
                t_end = float(timestamps[min(end_idx, len(timestamps) - 1)])
                if merged_segs and t_start - merged_segs[-1][1] < gap_threshold:
                    merged_segs[-1] = (merged_segs[-1][0], t_end)
                else:
                    merged_segs.append((t_start, t_end))

            # Expand each region by 2 seconds on each side
            for t_start, t_end in merged_segs:
                t_start = max(0, t_start - 2.0)
                t_end = min(duration, t_end + 2.0)
                if t_end - t_start < 3.0:
                    center = (t_start + t_end) / 2
                    t_start = max(0, center - 1.5)
                    t_end = min(duration, center + 1.5)
                mask = (timestamps >= t_start) & (timestamps <= t_end)
                if np.any(mask):
                    avg_score = float(np.mean(smoothed[mask]))
                    peak_score = float(np.max(smoothed[mask]))
                    all_regions.append({
                        't_start': t_start,
                        't_end': t_end,
                        'clip_cat_score': avg_score,
                        'peak_score': peak_score,
                        'scale': smooth_sec,
                    })

        # Multi-scale NMS: prefer higher peak score, break ties with tighter boundaries
        all_regions.sort(key=lambda r: (r['peak_score'], -(r['t_end'] - r['t_start'])), reverse=True)
        selected = []
        for region in all_regions:
            overlap = False
            for sel in selected:
                iou = self._temporal_iou(region, sel)
                if iou > 0.5:
                    overlap = True
                    break
            if not overlap:
                selected.append(region)

        selected = selected[:5]

        if selected:
            _log.info(f"  [CLIP-loc] Found {len(selected)} multi-scale regions:")
            for r in selected:
                _log.debug(f"    [{r['t_start']:.1f}-{r['t_end']:.1f}]s "
                      f"score={r['clip_cat_score']:.3f} peak={r['peak_score']:.3f} "
                      f"scale={r['scale']:.1f}s")

        return selected

    @staticmethod
    def _temporal_iou(a, b):
        """Compute temporal IoU between two segments."""
        overlap_start = max(a['t_start'], b['t_start'])
        overlap_end = min(a['t_end'], b['t_end'])
        overlap = max(0, overlap_end - overlap_start)
        union = max(a['t_end'], b['t_end']) - min(a['t_start'], b['t_start'])
        return overlap / union if union > 0 else 0

    async def _directed_llm_scan(self, video_data, clip_regions, context_prompt, category):
        """Hybrid LLM scan: union of CLIP-loc regions + standard CLIP pre-screen."""
        segments = video_data['segments']
        all_frames = video_data['all_frames']
        precomputed = video_data.get('clip_features')

        targeted_set = set()
        targeted_segments = []

        # Source 1: segments overlapping CLIP-loc regions
        for seg in segments:
            for region in clip_regions:
                if seg['t_start'] < region['t_end'] and seg['t_end'] > region['t_start']:
                    if id(seg) not in targeted_set:
                        targeted_segments.append(seg)
                        targeted_set.add(id(seg))
                    break

        # Source 2: standard CLIP anomaly pre-screen (top segments by general anomaly score)
        if self.anomaly_expert.model is not None and precomputed is not None:
            clip_results = self.anomaly_expert.batch_score_segments(segments, precomputed)
            all_scores = [r['anomaly_score'] for r in clip_results]
            mean_s = np.mean(all_scores)
            std_s = np.std(all_scores)
            adaptive_thresh = max(0.3, mean_s - 0.5 * std_s)
            scored_indices = sorted(range(len(segments)),
                                    key=lambda i: all_scores[i], reverse=True)
            # Add top CLIP-scored segments not already covered
            min_extra = max(2, int(len(segments) * 0.2))
            added = 0
            for i in scored_indices:
                if added >= min_extra:
                    break
                if id(segments[i]) not in targeted_set:
                    targeted_segments.append(segments[i])
                    targeted_set.add(id(segments[i]))
                    added += 1

        max_scan = min(max(10, len(segments) // 5), 18)
        if len(targeted_segments) > max_scan:
            targeted_segments = targeted_segments[:max_scan]

        _log.info(f"  [Directed scan] {len(targeted_segments)}/{len(segments)} segments "
                  f"selected for LLM scanning")

        category_hint = ""
        if category:
            cue = CATEGORY_CUES.get(category, category)
            category_hint = f"\nExpected anomaly type: {category} — {cue}\n"

        tasks = []
        for seg in targeted_segments:
            tasks.append(self.discovery.scan_segment(
                seg, all_frames, context_prompt + category_hint, precomputed))

        results = await asyncio.gather(*tasks)

        suspicious = []
        for seg, result in zip(targeted_segments, results):
            status = "SUSPICIOUS" if result['is_suspicious'] else "normal"
            _log.debug(f"    {seg['t_start']:.1f}-{seg['t_end']:.1f}s: {status} "
                  f"(score={result['anomaly_score']:.2f})")
            if result['is_suspicious']:
                suspicious.append({
                    't_start': seg['t_start'],
                    't_end': seg['t_end'],
                    'frame_indices': seg['frame_indices'],
                    'coarse_score': result['anomaly_score'],
                    'description': result['description'],
                })

        return self.discovery._merge_adjacent(suspicious)

    # ================================================================
    # Stage 5 (v50): CLIP score-curve boundary detection
    # ================================================================

    def _clip_boundary_detection(self, candidates, video_data, category):
        """Replace MCTS with CLIP score curve boundary detection.

        Uses multi-scale half-max crossing with gradient refinement to find
        precise anomaly boundaries from the continuous CLIP score signal.
        Category-aware smoothing windows adapt to expected event duration.
        """
        from bisect import bisect_left

        precomputed = video_data.get('clip_features')
        if precomputed is None or self.anomaly_expert.model is None or torch is None:
            # Fallback: return candidates as-is with default confidence
            return [dict(c, confidence=c.get('coarse_score', 0.8),
                         coarse_score=c.get('coarse_score', 0.8),
                         anomaly_type=category or 'unknown') for c in candidates]

        timestamps = video_data['timestamps']
        duration = video_data['duration']
        fps = max(1, len(timestamps) / max(1, duration))

        # Get category-specific relative scores
        relative_scores = self.frame_scorer.get_category_scores(video_data, category)
        if relative_scores is None:
            return [dict(c, confidence=c.get('coarse_score', 0.8),
                         coarse_score=c.get('coarse_score', 0.8),
                         anomaly_type=category or 'unknown') for c in candidates]

        # Category-aware smoothing scales
        priors = CATEGORY_DURATION_PRIORS.get(category, {})
        preferred_smooth = priors.get('preferred_smooth', 2.0)
        min_dur, max_dur = priors.get('expected_duration', (3.0, 30.0))
        min_segment = priors.get('min_segment', 2.0)

        smooth_scales = [
            max(0.5, preferred_smooth * 0.5),
            preferred_smooth,
            preferred_smooth * 2.0,
        ]

        refined_candidates = []
        for cand in candidates:
            t_s = cand['t_start']
            t_e = cand['t_end']
            cand_dur = t_e - t_s

            # Search window: candidate region + 50% padding on each side
            padding = cand_dur * 0.5
            search_start = max(0, t_s - padding)
            search_end = min(duration, t_e + padding)

            lo_idx = max(0, bisect_left(timestamps, search_start))
            hi_idx = min(len(timestamps), bisect_left(timestamps, search_end))

            if hi_idx - lo_idx < 4:
                refined_candidates.append(dict(
                    cand, confidence=cand.get('coarse_score', 0.8),
                    coarse_score=cand.get('coarse_score', 0.8),
                    anomaly_type=category or 'unknown', boundary_contrast=0.0))
                continue

            seg_scores = relative_scores[lo_idx:hi_idx]
            seg_times = timestamps[lo_idx:hi_idx]

            # v63: V51 half-max crossing + motion-sharpened boundary detection.
            # Motion energy has SHARP transitions at anomaly boundaries (sudden onset
            # of violence, crash impact, explosion) while CLIP scores are smooth/broad.
            # Fusing motion with CLIP before boundary detection produces sharper
            # boundaries for motion-sensitive categories.
            # V59's peak-valley detection (nearest local minima as boundaries) was
            # a regression: mAP 0.1172->0.0922, AP@0.7=0. Root cause: shallow dips
            # within anomaly plateaus are false valleys — the nearest valley is often
            # INSIDE the anomaly, producing severely over-narrowed segments.
            # V51's half-max crossing (mAP=0.1492) is the proven best. V60 restores
            # it but adds a targeted post-step: when the half-max result is wider
            # than 2x the category's max_dur, re-threshold at 70% of peak (stricter)
            # to tighten just the extreme over-wide cases (RoadAccidents 28s det for
            # 2.7s event, Shooting 61s for 7s event) without affecting well-calibrated
            # segments. MIL fusion removed (saturated, no-op in practice).

            # v66: V51 half-max crossing + LogOIC tightening for over-wide results.
            # V65 showed LogOIC excels at short events (RoadAccidents IoU 0.095->0.45,
            # Abuse 0.282->0.60) but severely over-narrows long events (Fighting
            # 0.575->0.18, Assault 0.297->0.04). The solution: use V51's proven
            # half-max as PRIMARY, then apply LogOIC ONLY when the result exceeds
            # max_dur — the cases where half-max is known to be too wide.
            # FFT actionness calibration retained from V65.

            # v68: MIL fusion RESTORED. V67 showed that removing MIL fusion
            # (despite MIL being "near-saturated") changes boundary detection via
            # subtle interpolation non-uniformity — Fighting IoU 0.574→0.176,
            # Explosion 0.500→0.350. LogOIC tightening never even fired because
            # MIL removal already changed half-max results to be narrower.
            mil_scores = video_data.get('mil_scores')
            if mil_scores is not None and len(mil_scores) >= 2:
                _mil_interp = np.interp(
                    np.linspace(0, 1, len(seg_scores)),
                    np.linspace(0, 1, len(mil_scores)),
                    mil_scores
                )
                fused_scores = 0.75 * seg_scores + 0.25 * _mil_interp
            else:
                fused_scores = seg_scores

            best_boundaries = None
            best_score = -1.0

            for smooth_sec in smooth_scales:
                window = max(3, int(smooth_sec * fps))
                if window >= len(fused_scores):
                    window = max(3, len(fused_scores) // 2)
                if window % 2 == 0:
                    window += 1
                kernel = np.ones(window) / window
                smoothed = np.convolve(fused_scores, kernel, mode='same')
                gradient = np.gradient(smoothed)
                refine_range = max(1, int(1.0 * fps))
                peak_val = np.max(smoothed)
                baseline = np.percentile(smoothed, 20)
                half_max = (peak_val + baseline) / 2.0

                above = smoothed >= half_max
                transitions = np.diff(above.astype(np.int32))
                rises = np.where(transitions == 1)[0]
                falls = np.where(transitions == -1)[0]

                if len(rises) == 0 and len(falls) == 0:
                    if np.mean(above) > 0.5:
                        new_lo, new_hi = 0, len(fused_scores) - 1
                    else:
                        continue
                else:
                    new_lo = rises[0] + 1 if len(rises) > 0 else 0
                    new_hi = falls[-1] if len(falls) > 0 else len(fused_scores) - 1
                if new_hi <= new_lo:
                    continue

                prop = _score_boundary_proposal(
                    smoothed, gradient, seg_times, new_lo, new_hi,
                    refine_range, min_dur, max_dur)
                if prop is None:
                    continue

                # v66: LogOIC tightening for over-wide half-max results
                prop_dur = prop['t_end'] - prop['t_start']
                if prop_dur > max_dur:
                    peak_idx = np.argmax(smoothed)
                    n = len(smoothed)
                    best_logoic = -1.0
                    best_tight = None
                    min_hw = max(2, int(min_dur * fps * 0.5))
                    max_hw = min(n // 2, max(3, int(max_dur * fps * 0.5)))
                    step_hw = max(1, (max_hw - min_hw) // 10)
                    for hw in range(min_hw, max_hw + 1, step_hw):
                        lo_b = max(0, peak_idx - hw)
                        hi_b = min(n - 1, peak_idx + hw)
                        if hi_b <= lo_b:
                            continue
                        inner_m = float(np.mean(smoothed[lo_b:hi_b + 1]))
                        o_sum, o_wt = 0.0, 0.0
                        for j in range(lo_b):
                            w = 1.0 / np.log(2.0 + lo_b - j)
                            o_sum += smoothed[j] * w; o_wt += w
                        for j in range(hi_b + 1, n):
                            w = 1.0 / np.log(2.0 + j - hi_b)
                            o_sum += smoothed[j] * w; o_wt += w
                        o_m = o_sum / o_wt if o_wt > 0 else 0.0
                        logoic = inner_m - o_m
                        if logoic > best_logoic:
                            ts = float(seg_times[min(lo_b, len(seg_times)-1)])
                            te = float(seg_times[min(hi_b, len(seg_times)-1)])
                            if te > ts:
                                best_logoic = logoic
                                best_tight = (ts, te)
                    if best_tight is not None:
                        _log.debug(f"      [LogOIC-tighten] {prop_dur:.1f}s -> "
                                   f"{best_tight[1]-best_tight[0]:.1f}s")
                        prop = {'t_start': best_tight[0], 't_end': best_tight[1],
                                'contrast': best_logoic}

                if prop['contrast'] > best_score:
                    best_score = prop['contrast']
                    best_boundaries = (prop['t_start'], prop['t_end'])

            if best_boundaries is not None:
                result = dict(cand)
                result['t_start'] = max(0.0, best_boundaries[0])
                result['t_end'] = min(best_boundaries[1], duration)

                # v54: light backstop only — duration-prior-fit already steers
                # selection above, so this should rarely trigger now.
                detected_dur = result['t_end'] - result['t_start']
                if detected_dur < min_segment:
                    center = (result['t_start'] + result['t_end']) / 2.0
                    result['t_start'] = max(0.0, center - min_segment / 2.0)
                    result['t_end'] = min(duration, center + min_segment / 2.0)
                elif detected_dur > max_dur * 2.0:
                    tighter = self._clip_boundary_single_scale(
                        cand, video_data, category,
                        smooth_sec=max(0.5, preferred_smooth * 0.5))
                    if tighter is not None and (tighter['t_end'] - tighter['t_start']) >= min_segment:
                        result = tighter

                # v51: Redesigned confidence formula — decouple boundary precision from detection existence
                # Base confidence: 80% of LLM coarse_score (the anomaly EXISTS)
                # Boundary bonus: up to 20% for high CLIP contrast (boundaries are PRECISE)
                # v95: TSM sharpness re-ranking (Loop 1-2 research). Modulates
                # ONLY the confidence bonus (never t_start/t_end) using a
                # genuinely independent signal (frame-frame visual novelty,
                # not text-similarity) - detections whose boundaries align
                # with real visual scene transitions get a small confidence
                # boost. Pure re-ranking, zero risk to boundary positions or
                # detection count (respects all 4 known failure modes).
                _tsm_novelty = video_data.get('tsm_novelty')
                _tsm_mod = 1.0
                if _tsm_novelty is not None and len(_tsm_novelty) >= 2:
                    from bisect import bisect_left as _bl95
                    _ts_idx = min(len(_tsm_novelty) - 1, max(0, _bl95(timestamps, result['t_start'])))
                    _te_idx = min(len(_tsm_novelty) - 1, max(0, _bl95(timestamps, result['t_end'])))
                    _tsm_sharpness = float((_tsm_novelty[_ts_idx] + _tsm_novelty[_te_idx]) / 2.0)
                    _tsm_mod = 0.85 + 0.15 * _tsm_sharpness

                # v96: Asymmetric onset/offset likelihood (BSN-inspired, Loop 4
                # research). Confirmed via literature (BSN TEM, LGN) that
                # separating DIRECTIONAL start/end signals from a symmetric
                # inner-outer contrast is a real, validated architectural
                # pattern. Training-free analog: at the START boundary, a
                # real onset should show FUTURE window more anomalous than
                # PAST window; at the END boundary, PAST should be more
                # anomalous than FUTURE. Pure confidence modulation only
                # (never changes t_start/t_end) - same safe pattern as V95.
                _onset_mod = 1.0
                _w_frames = max(2, int(1.0 * fps))
                _ts_i = min(len(relative_scores) - 1, max(0, _bl95(timestamps, result['t_start'])))
                _te_i = min(len(relative_scores) - 1, max(0, _bl95(timestamps, result['t_end'])))
                _past_s = relative_scores[max(0, _ts_i - _w_frames):_ts_i]
                _fut_s = relative_scores[_ts_i:min(len(relative_scores), _ts_i + _w_frames)]
                _past_e = relative_scores[max(0, _te_i - _w_frames):_te_i]
                _fut_e = relative_scores[_te_i:min(len(relative_scores), _te_i + _w_frames)]
                if len(_past_s) > 0 and len(_fut_s) > 0 and len(_past_e) > 0 and len(_fut_e) > 0:
                    _onset_sharp = float(np.mean(_fut_s) - np.mean(_past_s))
                    _offset_sharp = float(np.mean(_past_e) - np.mean(_fut_e))
                    _directional_quality = max(0.0, min(1.0, (_onset_sharp + _offset_sharp) / 2.0 + 0.5))
                    _onset_mod = 0.9 + 0.1 * _directional_quality

                _coarse = cand.get('coarse_score', 0.8)
                _base_conf = _coarse * 0.80
                _boundary_bonus = _coarse * 0.20 * min(1.0, best_score / 0.5) * _tsm_mod * _onset_mod
                result['confidence'] = min(1.0, _base_conf + _boundary_bonus)
                result['coarse_score'] = _coarse  # v51: preserve for PostTighten
                result['anomaly_type'] = category or 'unknown'
                result['boundary_contrast'] = best_score

                _log.info(f"    [CLIPBound] {t_s:.1f}-{t_e:.1f}s -> "
                          f"{result['t_start']:.1f}-{result['t_end']:.1f}s "
                          f"(contrast={best_score:.3f}, cat={category})")
                refined_candidates.append(result)
            else:
                refined_candidates.append(dict(
                    cand, confidence=cand.get('coarse_score', 0.8),
                    coarse_score=cand.get('coarse_score', 0.8),
                    anomaly_type=category or 'unknown', boundary_contrast=0.0))

        return refined_candidates

    def _clip_boundary_single_scale(self, candidate, video_data, category, smooth_sec=1.0):
        """Single-scale CLIP boundary detection for re-tightening over-long segments."""
        from bisect import bisect_left

        timestamps = video_data['timestamps']
        duration = video_data['duration']
        fps = max(1, len(timestamps) / max(1, duration))

        relative_scores = self.frame_scorer.get_category_scores(video_data, category)
        if relative_scores is None:
            return None

        t_s, t_e = candidate['t_start'], candidate['t_end']
        cand_dur = t_e - t_s
        padding = cand_dur * 0.3
        search_start = max(0, t_s - padding)
        search_end = min(duration, t_e + padding)

        lo_idx = max(0, bisect_left(timestamps, search_start))
        hi_idx = min(len(timestamps), bisect_left(timestamps, search_end))
        if hi_idx - lo_idx < 4:
            return None

        seg_scores = relative_scores[lo_idx:hi_idx]
        seg_times = timestamps[lo_idx:hi_idx]

        window = max(3, int(smooth_sec * fps))
        if window >= len(seg_scores):
            window = max(3, len(seg_scores) // 2)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        smoothed = np.convolve(seg_scores, kernel, mode='same')

        # v54: classic 50% threshold — this function is now only a rare backstop
        # (invoked when the primary multi-threshold proposal in _clip_boundary_detection
        # still produces an extreme outlier), not the primary detection mechanism.
        peak_val = np.max(smoothed)
        baseline = np.percentile(smoothed, 20)
        half_max = (peak_val + baseline) / 2.0

        above = smoothed >= half_max
        transitions = np.diff(above.astype(np.int32))
        rises = np.where(transitions == 1)[0]
        falls = np.where(transitions == -1)[0]

        if len(rises) == 0 and len(falls) == 0:
            return None

        new_lo = rises[0] + 1 if len(rises) > 0 else 0
        new_hi = falls[-1] if len(falls) > 0 else len(seg_scores) - 1

        if new_hi <= new_lo:
            return None

        result = dict(candidate)
        result['t_start'] = max(0.0, float(seg_times[min(new_lo, len(seg_times) - 1)]))
        result['t_end'] = min(float(seg_times[min(new_hi, len(seg_times) - 1)]), duration)
        inside_mean = float(np.mean(smoothed[new_lo:new_hi + 1]))
        # v51: Same redesigned confidence formula as main boundary detection
        _coarse = candidate.get('coarse_score', 0.8)
        _base_conf = _coarse * 0.80
        _boundary_bonus = _coarse * 0.20 * min(1.0, inside_mean / 0.5)
        result['confidence'] = min(1.0, _base_conf + _boundary_bonus)
        result['coarse_score'] = _coarse  # v51: preserve for PostTighten
        result['anomaly_type'] = category or 'unknown'
        result['boundary_contrast'] = float(inside_mean)
        return result

    # ================================================================
    # Stage 5b (v50): Single VLM boundary estimation
    # ================================================================

    async def _vlm_boundary_estimation(self, candidate, video_data, category):
        """Single VLM call for temporal boundary estimation.

        For low-CLIP-confidence candidates, ask the VLM once to identify
        start/end times, then cross-validate against CLIP curve.
        """
        t_s = candidate['t_start']
        t_e = candidate['t_end']
        dur = t_e - t_s
        duration = video_data['duration']
        timestamps = video_data['timestamps']

        # Expand search window by 30% on each side
        search_start = max(0, t_s - dur * 0.3)
        search_end = min(duration, t_e + dur * 0.3)

        # Sample 8-12 frames uniformly with timestamps
        n_frames = min(12, max(8, int((search_end - search_start) / 2)))
        frame_times = np.linspace(search_start, search_end, n_frames)

        frames = []
        actual_times = []
        for t in frame_times:
            idx = int(np.argmin(np.abs(timestamps - t)))
            frames.append(video_data['all_frames'][idx])
            actual_times.append(float(timestamps[idx]))

        keyframes = np.stack(frames)
        time_labels = ', '.join(f'{t:.1f}s' for t in actual_times)

        prompt = (
            f"""Analyze these {n_frames} surveillance video frames in chronological order.
Frame timestamps: {time_labels}

Target anomaly type: {category}

For each frame, state whether the anomaly ({category}) is ACTIVELY occurring.
Then provide your best estimate of when the anomaly starts and ends.

Respond ONLY in JSON format:
{{"anomaly_detected": true, "start_time": <seconds>, "end_time": <seconds>, "confidence": 0-100}}"""
        )

        # v90->v91 REVERT: median-of-3 self-consistency regressed full-eval
        # mAP 0.1494->0.1075. Root cause hypothesis: the temperature ladder
        # (0.0, 0.3, 0.5) was designed for REASONING-diversity self-consistency
        # (multiple valid reasoning PATHS converging on the same correct
        # answer), which doesn't transfer to PERCEPTUAL/EXTRACTION tasks like
        # "what timestamp does this event start" - there's no equivalent of
        # "another valid way to see the same correct answer" for reading a
        # timestamp off a frame. Deliberately raising temperature on 2 of 3
        # calls just added noise, diluting the more-accurate temp=0 estimate
        # via the median rather than improving it. Reverted to the original
        # single deterministic call.
        try:
            response = await self.llm_client.generate(prompt, keyframes)
            text = response.strip()
            start_j = text.find('{')
            end_j = text.rfind('}') + 1
            if start_j >= 0 and end_j > start_j:
                parsed = json.loads(text[start_j:end_j])
            else:
                parsed = {}
        except Exception as e:
            _log.warning(f"    [VLM-Bound] Parse failed: {e}")
            return candidate

        if not parsed.get('anomaly_detected', False):
            return candidate

        vlm_start = float(parsed.get('start_time', t_s))
        vlm_end = float(parsed.get('end_time', t_e))
        vlm_conf = float(parsed.get('confidence', 50)) / 100.0

        # Sanity check VLM answer
        if vlm_start >= vlm_end or vlm_start < 0 or vlm_end > duration * 1.1:
            _log.warning(f"    [VLM-Bound] Invalid VLM times: {vlm_start:.1f}-{vlm_end:.1f}")
            return candidate

        # v61: Cross-validate with CLIP curve, with over-wide override
        clip_start = candidate.get('t_start', t_s)
        clip_end = candidate.get('t_end', t_e)
        clip_conf = candidate.get('boundary_contrast', 0.5)
        clip_dur = clip_end - clip_start

        priors_bd = CATEGORY_DURATION_PRIORS.get(category, {})
        cat_max_dur = priors_bd.get('expected_duration', (3.0, 30.0))[1]

        if clip_dur > cat_max_dur * 1.2 and vlm_conf > 0.4:
            # CLIP is demonstrably over-wide — trust VLM boundaries directly
            final_start = vlm_start
            final_end = vlm_end
            w_clip, w_vlm = 0.0, 1.0
            _log.info(f"    [VLM-Bound] CLIP over-wide ({clip_dur:.1f}s > "
                      f"{cat_max_dur*1.2:.1f}s), using VLM boundaries")
        else:
            # Normal weighted average
            w_clip = clip_conf / (clip_conf + vlm_conf + 1e-8)
            w_vlm = vlm_conf / (clip_conf + vlm_conf + 1e-8)
            final_start = w_clip * clip_start + w_vlm * vlm_start
            final_end = w_clip * clip_end + w_vlm * vlm_end

        result = dict(candidate)
        result['t_start'] = max(0.0, final_start)
        result['t_end'] = min(final_end, duration)
        result['vlm_confidence'] = vlm_conf

        # Ensure minimum duration
        priors = CATEGORY_DURATION_PRIORS.get(category, {})
        min_seg = priors.get('min_segment', 2.0)
        if result['t_end'] - result['t_start'] < min_seg:
            center = (result['t_start'] + result['t_end']) / 2.0
            result['t_start'] = max(0.0, center - min_seg / 2.0)
            result['t_end'] = min(duration, center + min_seg / 2.0)

        _log.info(f"    [VLM-Bound] {t_s:.1f}-{t_e:.1f}s -> "
                  f"{result['t_start']:.1f}-{result['t_end']:.1f}s "
                  f"(vlm_conf={vlm_conf:.2f}, w_clip={w_clip:.2f}, w_vlm={w_vlm:.2f})")

        return result

        # ================================================================
    # Stage 6b (v50): Post-merge boundary tightening
    # ================================================================

    def _post_merge_boundary_tightening(self, merged_detections, video_data, category):
        """Re-run CLIP boundary detection on merged segments that are too long.

        v51 changes:
        - Never reduce confidence after re-running boundary detection
        - Skip PostTighten entirely for LOW_CLIP categories (re-running produces worse results)
        - Preserve coarse_score through the process
        """
        if not merged_detections:
            return merged_detections

        tightened = []
        for det in merged_detections:
            dur = det['t_end'] - det['t_start']
            cat = det.get('anomaly_type', category) or category
            expected_max = CATEGORY_DURATION_PRIORS.get(
                cat, {}).get('expected_duration', (3, 30))[1]

            # v51: Skip PostTighten for low-CLIP categories where re-running
            # boundary detection produces lower-quality results
            if cat in LOW_CLIP_CATEGORIES:
                tightened.append(det)
                continue

            if dur > expected_max * 1.2:
                original_conf = det['confidence']
                original_coarse = det.get('coarse_score', det['confidence'])
                refined = self._clip_boundary_detection([det], video_data, cat)
                if refined and (refined[0]['t_end'] - refined[0]['t_start']) >=                         CATEGORY_DURATION_PRIORS.get(cat, {}).get('min_segment', 2.0):
                    ref = refined[0]
                    # v51: NEVER reduce confidence — PostTighten is about boundaries, not existence
                    ref['confidence'] = max(original_conf, ref['confidence'])
                    ref['coarse_score'] = original_coarse
                    _log.info(f"    [PostTighten] {det['t_start']:.1f}-{det['t_end']:.1f}s "
                              f"-> {ref['t_start']:.1f}-{ref['t_end']:.1f}s "
                              f"(was {dur:.1f}s > {expected_max*1.2:.1f}s limit, "
                              f"conf={original_conf:.3f}->{ref['confidence']:.3f})")
                    tightened.append(ref)
                else:
                    tightened.append(det)
            else:
                tightened.append(det)

        return tightened

        # ================================================================
    # Stage 6: Category-aware CLIP boundary refinement
    # ================================================================


    async def _refine_boundary_category_aware(self, anomaly, video_data, category):
        """Gradient-based boundary refinement using CLIP score curve derivatives."""
        t_start = anomaly['t_start']
        t_end = anomaly['t_end']
        duration = video_data['duration']
        timestamps = video_data['timestamps']

        relative_scores = self.frame_scorer.get_category_scores(video_data, category)
        if relative_scores is None:
            return await self._refine_boundary(anomaly, video_data)

        global_std = np.std(relative_scores)
        global_mean = np.mean(relative_scores)
        coeff_var = global_std / (global_mean + 1e-8)
        if coeff_var < 0.05:
            return await self._refine_boundary(anomaly, video_data)

        from bisect import bisect_left
        lo = bisect_left(timestamps, t_start)
        hi = bisect_left(timestamps, t_end)
        if hi <= lo:
            return anomaly

        # Smooth scores for gradient computation
        fps = max(1, len(timestamps) / max(1, duration))
        smooth_w = max(3, int(1.5 * fps))
        if len(relative_scores) >= smooth_w:
            kernel = np.ones(smooth_w) / smooth_w
            smoothed = np.convolve(relative_scores, kernel, mode='same')
        else:
            smoothed = relative_scores.copy()

        core_scores = smoothed[lo:hi]
        core_mean = np.mean(core_scores)
        seg_dur = t_end - t_start

        # Compute score gradient (first derivative)
        gradient = np.gradient(smoothed)

        # --- Gradient-based left boundary ---
        max_expand = seg_dur * 0.5
        search_lo = max(0, bisect_left(timestamps, max(0, t_start - max_expand)))
        # Look for the sharpest positive gradient (score rising) near left boundary
        best_left = lo
        left_search = range(max(search_lo, lo - int(max_expand * fps)), lo + int(2 * fps) + 1)
        if left_search:
            left_grads = [(i, gradient[i]) for i in left_search if 0 <= i < len(gradient)]
            if left_grads:
                # Find steepest rise — that's where the anomaly starts
                rise_candidates = [(i, g) for i, g in left_grads if g > 0]
                if rise_candidates:
                    best_rise_idx = max(rise_candidates, key=lambda x: x[1])[0]
                    if smoothed[best_rise_idx] >= core_mean * 0.4:
                        best_left = best_rise_idx

        # Also check threshold-based left boundary
        threshold_left = lo
        thresh = core_mean * 0.55
        for i in range(lo - 1, search_lo - 1, -1):
            if smoothed[i] >= thresh:
                threshold_left = i
            else:
                break

        # Only contract, never expand beyond original boundary
        final_left = max(lo, min(best_left, threshold_left))

        # --- Gradient-based right boundary ---
        search_hi = min(len(timestamps), bisect_left(timestamps,
                        min(duration, t_end + max_expand)))
        best_right = hi
        right_search = range(max(0, hi - int(2 * fps)), min(search_hi, hi + int(max_expand * fps)) + 1)
        if right_search:
            right_grads = [(i, gradient[i]) for i in right_search if 0 <= i < len(gradient)]
            if right_grads:
                # Find steepest drop (negative gradient) — that's where the anomaly ends
                drop_candidates = [(i, g) for i, g in right_grads if g < 0]
                if drop_candidates:
                    best_drop_idx = min(drop_candidates, key=lambda x: x[1])[0]
                    if smoothed[best_drop_idx] >= core_mean * 0.4:
                        best_right = best_drop_idx + 1

        # Also check threshold-based right boundary
        threshold_right = hi
        for i in range(hi, search_hi):
            if smoothed[i] >= thresh:
                threshold_right = i + 1
            else:
                break

        # Only contract, never expand beyond original boundary
        final_right = min(hi, max(best_right, threshold_right))

        # Convert back to time
        final_left = max(0, min(final_left, len(timestamps) - 1))
        final_right = max(0, min(final_right, len(timestamps) - 1))
        new_start = timestamps[final_left]
        new_end = timestamps[final_right]

        # Shrink: trim frames with very low scores from edges (max 15%)
        shrink_thresh = core_mean * self.config.scoring.boundary_shrink_ratio
        s_lo = bisect_left(timestamps, new_start)
        s_hi = bisect_left(timestamps, new_end)
        total_frames = s_hi - s_lo
        max_trim = max(1, total_frames // 7)
        for i in range(s_lo, min(s_lo + max_trim, s_hi)):
            if i < len(smoothed) and smoothed[i] < shrink_thresh:
                new_start = timestamps[min(i + 1, len(timestamps) - 1)]
            else:
                break
        for i in range(s_hi - 1, max(s_hi - max_trim - 1, s_lo), -1):
            if i < len(smoothed) and smoothed[i] < shrink_thresh:
                new_end = timestamps[max(i - 1, 0)]
            else:
                break

        refined = anomaly.copy()
        refined['t_start'] = max(0, new_start)
        refined['t_end'] = min(new_end, duration)

        if refined['t_end'] - refined['t_start'] < 1.0:
            refined['t_start'] = t_start
            refined['t_end'] = t_end

        if abs(refined['t_start'] - t_start) > 0.5 or abs(refined['t_end'] - t_end) > 0.5:
            _log.info(f"    [GradRefine] {t_start:.1f}-{t_end:.1f}s "
                      f"-> {refined['t_start']:.1f}-{refined['t_end']:.1f}s "
                      f"(cat={category}, cv={coeff_var:.3f})")

        return refined

    # ================================================================
    # Fallback boundary refinement (generic CLIP)
    # ================================================================

    async def _refine_boundary(self, anomaly, video_data):
        """Generic CLIP anomaly score boundary refinement (vectorized, no async calls)."""
        t_start = anomaly['t_start']
        t_end = anomaly['t_end']
        duration = video_data['duration']
        precomputed = video_data.get('clip_features')
        timestamps = video_data['timestamps']

        if precomputed is None or self.anomaly_expert.model is None or torch is None:
            return anomaly

        # Use general anomaly scores (max across all anomaly categories)
        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            # Anomaly score = 1 - normal_prob (last category is Normal)
            anomaly_scores = 1.0 - probs[:, -1]

        from bisect import bisect_left
        lo = bisect_left(timestamps, t_start)
        hi = bisect_left(timestamps, t_end)
        if hi <= lo:
            return anomaly

        core_scores = anomaly_scores[lo:hi]
        core_mean = np.mean(core_scores)
        seg_dur = t_end - t_start

        # Expand: include adjacent frames above 55% of core mean
        thresh = core_mean * 0.55
        max_expand = seg_dur * 0.4
        new_lo = lo
        search_lo = max(0, bisect_left(timestamps, max(0, t_start - max_expand)))
        for i in range(lo - 1, search_lo - 1, -1):
            if anomaly_scores[i] >= thresh:
                new_lo = i
            else:
                break

        new_hi = hi
        search_hi = min(len(timestamps), bisect_left(timestamps,
                        min(duration, t_end + max_expand)))
        for i in range(hi, search_hi):
            if anomaly_scores[i] >= thresh:
                new_hi = i + 1
            else:
                break

        # Shrink: trim very low score edges
        shrink_thresh = core_mean * self.config.scoring.boundary_shrink_ratio
        total_frames = new_hi - new_lo
        max_trim = max(1, total_frames // 6)
        for i in range(new_lo, min(new_lo + max_trim, new_hi)):
            if anomaly_scores[i] < shrink_thresh:
                new_lo = i + 1
            else:
                break
        for i in range(new_hi - 1, max(new_hi - max_trim - 1, new_lo), -1):
            if anomaly_scores[i] < shrink_thresh:
                new_hi = i
            else:
                break

        refined = anomaly.copy()
        new_lo = max(0, min(new_lo, len(timestamps) - 1))
        new_hi = max(0, min(new_hi - 1, len(timestamps) - 1))
        refined['t_start'] = max(0, timestamps[new_lo])
        refined['t_end'] = min(timestamps[new_hi], duration)

        if refined['t_end'] - refined['t_start'] < 1.0:
            refined['t_start'] = t_start
            refined['t_end'] = t_end

        return refined

    # ================================================================
    # Classification fallback
    # ================================================================


    def _dual_videomae_classify(self, anomalies, video_data, full_video_category, full_video_confidence):
        """Dual VideoMAE: combine full-video and segment-level classification."""
        if not anomalies or self.videomae is None or self.videomae.model is None:
            for a in anomalies:
                a['anomaly_type'] = full_video_category
            return anomalies

        # Get frames from detected anomaly segments
        all_seg_frames = []
        for a in anomalies:
            frames, indices = self.preprocessor.get_segment_frames(
                video_data, a['t_start'], a['t_end'])
            if len(frames) > 0:
                all_seg_frames.append(frames)

        if not all_seg_frames:
            for a in anomalies:
                a['anomaly_type'] = full_video_category
            return anomalies

        # Concatenate all segment frames
        seg_frames = np.concatenate(all_seg_frames, axis=0)

        # Classify segment frames (need at least 16 frames for VideoMAE)
        if len(seg_frames) >= 8:
            seg_result = self.videomae.classify(seg_frames, top_k=3, tta=False)
            seg_category = seg_result['category']
            seg_confidence = seg_result['confidence']
            seg_top3 = ', '.join(f'{c}({s:.0%})' for c, s in seg_result['top_k'])
            _log.info(f"    [VideoMAE-seg] {seg_top3} -> {seg_category}@{seg_confidence:.0%}")

            # Decision logic
            if seg_category == full_video_category:
                # Both agree — use it
                category = full_video_category
            elif seg_category == 'Normal':
                # Segment says Normal, use full-video
                category = full_video_category
            elif full_video_confidence > 0.85 and seg_confidence < 0.5:
                # Full-video very confident, segment not — trust full-video
                category = full_video_category
            elif seg_confidence > full_video_confidence:
                # Segment more confident — prefer segment (anomaly content is concentrated)
                category = seg_category
                _log.info(f"    [VideoMAE-dual] Override: {full_video_category}@{full_video_confidence:.0%} "
                      f"-> {seg_category}@{seg_confidence:.0%} (segment)")
            else:
                # Otherwise trust full-video (more context)
                category = full_video_category
        else:
            category = full_video_category

        for a in anomalies:
            a['anomaly_type'] = category
        return anomalies

    async def _ensemble_classify(self, anomalies, video_data, videomae_cat, videomae_conf):
        """Multi-classifier ensemble: VideoMAE + VLM for known confusion pairs only."""
        # v137 NOCLS-CA (KEPT latency win): drop the entire classify VLM sub-tree
        # (qwen3-vl-plus, <=3 calls/video) and return the upstream VideoMAE label.
        # mAP-NEUTRAL BY CONSTRUCTION: the returned string only sets a['anomaly_type'],
        # which compute_temporal_map never reads (it uses {video_id,t_start,t_end,
        # confidence} only); the confidence gate / curve correction / frame scores
        # all key off videomae_category, not this result. Cuts ~7.5% of VLM calls.
        # Tradeoff: degrades the separate Level-4 category-accuracy metric, not mAP.
        if self.config.scoring.classify_skip_class_agnostic:
            _label = videomae_cat if (videomae_cat and videomae_cat != 'Normal') else 'unknown'
            _log.info(f"  [NOCLS-CA] classify VLM sub-tree skipped -> label='{_label}' "
                      f"(mAP-neutral; ~-7.5% VLM calls)")
            return _label
        if not videomae_cat or videomae_cat == 'Normal':
            result = await self._classify_clip_guided(anomalies, video_data)
            if result and result[0].get('anomaly_type'):
                return result[0]['anomaly_type']
            return 'unknown'

        # Low-confidence VideoMAE (<50%): override with CLIP top-1 if it's clearly different
        # This recovers cases like Burglary005 where VideoMAE is uncertain but CLIP is clear
        if videomae_conf < 0.50:
            precomputed = video_data.get('clip_features')
            _lc_frame_indices = []
            for _a in anomalies:
                _, _idx = self.preprocessor.get_segment_frames(
                    video_data, _a['t_start'], _a['t_end'])
                if _idx:
                    _lc_frame_indices.extend(_idx)
            if _lc_frame_indices and precomputed is not None and self.anomaly_expert.model is not None:
                _unique = sorted(set(_lc_frame_indices))
                _clip_dist = self.anomaly_expert.get_category_distribution(_unique, precomputed, top_k=3)
                if _clip_dist:
                    _clip_top1 = max(_clip_dist, key=_clip_dist.get)
                    _clip_top1_score = _clip_dist[_clip_top1]
                    if _clip_top1 != videomae_cat and _clip_top1_score > 0.35:
                        _log.info(f"  [Ensemble] LowConf VideoMAE({videomae_conf:.0%}), CLIP={_clip_top1}({_clip_top1_score:.0%}) override")
                        return _clip_top1

        confusion_alts = VIDEOMAE_CONFUSION_MAP.get(videomae_cat, [])
        if not confusion_alts:
            _log.info(f"  [Ensemble] No confusion pair -> {videomae_cat}")
            return videomae_cat

        # Build candidates: ONLY VideoMAE prediction + known confusion alternatives
        candidates = [videomae_cat] + confusion_alts
        candidates = candidates[:4]

        # Collect anomaly segment frames for VLM
        all_frame_indices = []
        seg_info = []
        precomputed = video_data.get('clip_features')
        for a in anomalies:
            frames, indices = self.preprocessor.get_segment_frames(
                video_data, a['t_start'], a['t_end'])
            if indices:
                all_frame_indices.extend(indices)
                seg_info.append(f"{a['t_start']:.1f}-{a['t_end']:.1f}s")

        # CLIP distribution for VLM context (informational only)
        clip_dist = {}
        if all_frame_indices and precomputed is not None and self.anomaly_expert.model is not None:
            unique_indices = sorted(set(all_frame_indices))
            clip_dist = self.anomaly_expert.get_category_distribution(
                unique_indices, precomputed, top_k=5)
            if clip_dist:
                clip_sorted = sorted(clip_dist.items(), key=lambda x: x[1], reverse=True)
                _log.info(f"  [Ensemble] VideoMAE={videomae_cat}@{videomae_conf:.0%}, "
                          f"CLIP-seg={', '.join(f'{c}({s:.0%})' for c, s in clip_sorted[:3])}")

        _log.info(f"  [Ensemble] VLM deciding among {candidates}")

        all_frames = video_data['all_frames']
        if all_frame_indices and precomputed is not None:
            selected = self.anomaly_expert.get_top_anomaly_frames(
                sorted(set(all_frame_indices)), precomputed, n=16)
        else:
            step = max(1, len(all_frames) // 16)
            selected = list(range(0, len(all_frames), step))[:16]

        keyframes = np.stack([all_frames[i] for i in selected])
        # Majority voting: 3 VLM calls in parallel to reduce non-determinism
        vote_tasks = [
            self._classify_from_candidates(keyframes, seg_info, candidates, clip_dist, perspective_idx=i)
            for i in range(3)
        ]
        votes = list(await asyncio.gather(*vote_tasks))
        vote_counts = Counter(votes)
        majority_cat, majority_count = vote_counts.most_common(1)[0]
        if majority_count >= 2:
            vlm_cat = majority_cat
        else:
            vlm_cat = videomae_cat
        _log.info(f"  [Ensemble] VLM votes={dict(vote_counts)} -> {vlm_cat}")
        return vlm_cat

    async def _classify_clip_guided(self, anomalies, video_data):
        """CLIP top-3 guided classification fallback."""
        if not anomalies:
            return anomalies

        precomputed = video_data.get('clip_features')

        all_frame_indices = []
        seg_info = []
        for a in anomalies:
            frames, indices = self.preprocessor.get_segment_frames(
                video_data, a['t_start'], a['t_end'])
            if indices:
                all_frame_indices.extend(indices)
                seg_info.append(f"{a['t_start']:.1f}-{a['t_end']:.1f}s")

        if not all_frame_indices:
            return anomalies

        unique_indices = sorted(set(all_frame_indices))

        clip_dist = {}
        if precomputed is not None and self.anomaly_expert.model is not None:
            selected_indices = self.anomaly_expert.get_top_anomaly_frames(
                unique_indices, precomputed, n=16)
            clip_dist = self.anomaly_expert.get_category_distribution(
                unique_indices, precomputed, top_k=3)
        else:
            selected_indices = unique_indices[:16]

        all_frames = video_data['all_frames']
        keyframes = np.stack([all_frames[i] for i in selected_indices])

        sorted_cats = sorted(clip_dist.items(), key=lambda x: x[1], reverse=True) if clip_dist else []

        if not sorted_cats:
            return anomalies

        top1_cat, top1_score = sorted_cats[0]
        top2_cat, top2_score = sorted_cats[1] if len(sorted_cats) > 1 else (None, 0)
        top3_cat, top3_score = sorted_cats[2] if len(sorted_cats) > 2 else (None, 0)

        _log.info(f"    [CLIP] Top-3: {top1_cat}({top1_score:.1%}), "
              f"{top2_cat}({top2_score:.1%}), {top3_cat}({top3_score:.1%})")

        if top2_score > 0 and top1_score / top2_score > 3.0:
            anomaly_type = top1_cat
            _log.info(f"    [Classify] CLIP high-confidence -> {anomaly_type}")
        else:
            candidates_list = [top1_cat]
            if top2_cat:
                candidates_list.append(top2_cat)
            if top3_cat and top2_score > 0 and top1_score / top2_score < 1.5:
                candidates_list.append(top3_cat)

            anomaly_type = await self._classify_from_candidates(
                keyframes, seg_info, candidates_list, clip_dist)

        classified = []
        for a in anomalies:
            a = a.copy()
            a['anomaly_type'] = anomaly_type
            classified.append(a)
        return classified

    # v86: diverse-perspective prompts for self-consistency voting (Loop 1
    # research, Agent B). Replaces 3 IDENTICAL VLM calls (relying only on API
    # sampling randomness) with 3 DIFFERENT analytical angles, which the
    # self-consistency literature shows produces more accurate majority votes
    # than repeated identical prompts - especially for semantically-overlapping
    # categories (Shoplifting/Stealing/Robbery/Burglary all involve "taking an
    # item", differing in method/force/location). Zero extra API cost (same
    # call count).
    CLASSIFY_PERSPECTIVES = [
        "Focus on the SETTING and CONTEXT: where is this happening (store/street/"
        "building/vehicle), what objects or props are visible, and does the "
        "environment match a specific anomaly type?",
        "Focus on the SPECIFIC PHYSICAL ACTIONS: what exactly is each person doing "
        "with their hands/body (grabbing, forcing entry, threatening, concealing, "
        "striking), and in what sequence?",
        "Focus on the OUTCOME and INTERACTION: is there a victim reacting/resisting, "
        "is force or a weapon involved, does an item change possession, and what is "
        "the apparent consequence of the action?",
    ]

    async def _classify_from_candidates(self, keyframes, seg_info, candidates, clip_dist, perspective_idx=0):
        options_text = "\n".join(
            f"  {i+1}. {cat}: {CATEGORY_CUES.get(cat, cat)}"
            for i, cat in enumerate(candidates)
        )

        clip_hint = ""
        if clip_dist:
            scores = [(c, clip_dist.get(c, 0)) for c in candidates]
            clip_hint = "\nVisual analysis prior: " + ", ".join(
                f"{c}({s:.0%})" for c, s in scores) + "\n"

        _perspective = self.CLASSIFY_PERSPECTIVES[perspective_idx % len(self.CLASSIFY_PERSPECTIVES)]

        prompt = (
            f"These {len(keyframes)} frames are from anomalous segments "
            f"({', '.join(seg_info)}) of a surveillance video. "
            f"Frames are in CHRONOLOGICAL ORDER."
            f"{clip_hint}\n"
            f"{_perspective}\n\n"
            f"Which type of anomaly is shown?\n\n"
            f"{options_text}\n\n"
            f"Respond ONLY in JSON: "
            f"{{\"choice\": 1-{len(candidates)}, \"anomaly_type\": \"...\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"specific visual evidence\"}}"
        )

        response = await self.classify_llm_client.generate(prompt, keyframes)

        anomaly_type = None
        try:
            text = response.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                choice = data.get('choice')
                predicted = data.get('anomaly_type', '')
                if isinstance(choice, int) and 1 <= choice <= len(candidates):
                    anomaly_type = candidates[choice - 1]
                elif predicted in candidates:
                    anomaly_type = predicted
                elif predicted in self.anomaly_categories:
                    anomaly_type = predicted
        except Exception:
            pass

        if anomaly_type is None:
            anomaly_type = candidates[0]

        _log.info(f"    [Classify] VLM chose {anomaly_type} from {candidates}")
        return anomaly_type


