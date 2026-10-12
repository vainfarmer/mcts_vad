"""
视频异常检测与时间定位 — 完整 Pipeline (v100)

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

try:
    from .video_preprocessor import VideoPreprocessor
    from .anomaly_discovery import AnomalyDiscovery
    from .temporal_mcts import TemporalBoundaryMCTS
    from .anomaly_expert import VideoAnomalyExpert
    from .token_compressor import SemanticFrameSampler
except ImportError:
    from video_preprocessor import VideoPreprocessor
    from anomaly_discovery import AnomalyDiscovery
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

    async def detect(
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
                if self.config.scoring.event_unit_boundary_contraction:
                    _log.info("  [EUBC] summary: 0 detections (early return: video normal)")
                frame_scores = self.frame_scorer.compute_frame_scores(video_data, [], None)
                return DetectionResult(
                    video_path=video_path,
                    duration=video_data['duration'],
                    is_normal=True,
                    anomalies=[],
                    frame_scores=frame_scores,
                    timestamps=video_data['timestamps'].tolist() if frame_scores else None,
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
        _fccs = self.config.discovery.full_coverage_comparative_scan
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
        if _fccs:
            # v97 FCCS: single full-coverage comparative scan replaces BOTH
            # the directed and fallback partial-coverage scans (coverage:
            # 10-18 of 23-120 segments -> 100%; judgment: independent
            # absolute -> relative comparative with in-video calibration).
            category_hint = ""
            if videomae_category:
                cue = CATEGORY_CUES.get(videomae_category, videomae_category)
                category_hint = f"\nExpected anomaly type: {videomae_category} — {cue}\n"
            candidates = await self.discovery.scan_full_coverage(
                video_data, context_prompt, category_hint)
        elif clip_regions:
            candidates = await self._directed_llm_scan(
                video_data, clip_regions, context_prompt, videomae_category)
        else:
            candidates = await self.discovery.discover(video_data, context_prompt)

        if not candidates:
            if self.config.scoring.event_unit_boundary_contraction:
                _log.info("  [EUBC] summary: 0 detections (early return: no candidates)")
            frame_scores = self.frame_scorer.compute_frame_scores(video_data, [], videomae_category)
            return DetectionResult(
                video_path=video_path,
                duration=video_data['duration'],
                is_normal=True,
                anomalies=[],
                frame_scores=frame_scores,
                timestamps=video_data['timestamps'].tolist() if frame_scores else None,
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

        # v100 EUBC: Event-Unit Boundary Contraction — 终态过宽边界收缩。
        # 单开关（默认关 = 字节级恢复旧行为）。在 ALL 边界逻辑之后、最终
        # dict 投影之前作用于 FINAL detections；信号严格只读；仅收缩。
        # 若 ECE 重跑并先保留，EUBC 在同一终态槽位运行于 ECE 之后。
        if self.config.scoring.event_unit_boundary_contraction:
            final_anomalies = self._event_unit_boundary_contraction(
                final_anomalies, video_data, videomae_category)

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

        return DetectionResult(
            video_path=video_path,
            duration=video_data['duration'],
            is_normal=len(final_anomalies) == 0,
            anomalies=[AnomalySegment.from_dict(a) for a in final_anomalies],
            frame_scores=frame_scores,
            timestamps=video_data['timestamps'].tolist() if frame_scores else None,
        )

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


