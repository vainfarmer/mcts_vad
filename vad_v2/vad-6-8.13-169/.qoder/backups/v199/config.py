"""
VAD 统一配置 (VADConfig)

解决的核心问题：阈值/权重等 magic number 散落在各方法内部
（_compute_frame_level_scores 里的 0.35、detect() 里的 0.72、sigma 的 1.5 倍因子、
MCTS padding 的 0.75、candidate 截断 3 等），改一个参数要在代码里全局搜索。

本模块定义单一 VADConfig dataclass 集中管理所有超参数：
  - 不同实验用不同 config（YAML/JSON），不改代码
  - config 可随 evaluation_report 一起持久化，保证实验可复现
  - from_args() 适配现有 argparse 命令行入口（run.py / batch_evaluate.py）

默认值严格对齐当前 v33 行为，避免引入回归。
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PreprocessConfig:
    """Stage 1 视频预处理参数。"""
    fps: int = 2                # 采样帧率
    segment_duration: float = 6.0   # 初始分段时长（秒）
    segment_overlap: float = 0.25   # 段重叠比例
    motion_threshold: float = 1.0   # 运动过滤阈值（0 禁用）
    min_motion_interval: float = 2.0  # 静态区最低保留间隔（秒）


@dataclass
class CLIPConfig:
    """CLIP 零样本专家 + 帧采样参数。"""
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    max_frames: int = 200       # 单次 LLM 请求帧数硬上限
    min_frames: int = 3         # 帧数下限（保证 VLM 能识别）
    fps_sample: float = 3.0     # 动态帧数基准采样率
    semantic_tau: float = 0.90  # 语义去重相似度阈值


@dataclass
class MCTSConfig:
    """Stage 5 时间边界参数（v50: MCTS replaced by CLIP boundary detection）。"""
    simulations: int = 12       # 保留用于兼容 batch_evaluate CLI 参数
    max_depth: int = 4          # 保留用于兼容
    padding_ratio: float = 0.75  # 搜索区间外扩比例（v50: 用于 CLIP boundary padding）
    early_stop_agree: int = 4   # v50 未使用，保留兼容
    early_stop_min_sim: int = 5  # v50 未使用，保留兼容


@dataclass
class ScoringConfig:
    """Stage 7b/8 帧级评分 + 边界修正参数（v33 核心参数）。"""
    clip_base_weight: float = 0.35     # CLIP base 信号幅度（cat_scores * 0.35）
    gaussian_sigma_factor: float = 1.5  # 平滑核 sigma = factor * fps_est
    gaussian_sigma_min: float = 1.0
    confidence_gate: float = 0.50      # v51: lowered from 0.72 (category-aware overrides in pipeline)
    curve_smooth_seconds: float = 2.0  # score curve 卷积窗 = 2.0s * fps
    curve_half_max_weight: float = 0.5  # 半峰值阈值 = (core_max + global_mean)*0.5
    curve_max_shift_ratio: float = 0.4  # 边界最大移动 = seg_dur * 0.4
    curve_core_signal_floor: float = 0.3  # core_max < mean + 0.3*std 则不修正
    boundary_shrink_ratio: float = 0.35  # 边界收缩阈值 = core_mean * 0.35
    videomae_weight: float = 0.15      # VideoMAE 滑动窗口信号权重
    motion_weight: float = 0.10        # motion energy 信号权重
    mae_window_size: int = 32          # VideoMAE 滑动窗口帧数
    mae_window_stride: int = 16        # 滑动步长
    # v105 ECE: Event-Continuation Extension（单开关；False = 字节级恢复旧行为）
    # 终态 post-processing（EUBC 之前的同一终态槽位）：对 FINAL detections
    # 在整视频相对类别 CLIP 曲线上做 v51 半峰值 LEVEL 区域生长（pinned v99
    # 语义），CATEGORY_DURATION_PRIORS max_dur reject-cap；零新常数。
    event_continuation_extension: bool = False
    # v100 EUBC: Event-Unit Boundary Contraction（单开关；False = 字节级恢复旧行为）
    # 终态 post-processing：对 FINAL detections 按帧间视觉一致性收缩过宽边界。
    event_unit_boundary_contraction: bool = False
    eubc_mad_k: float = 1.0            # coherence 阈值 = median - k*1.4826*MAD(sims)
    eubc_materiality: float = 0.75     # 仅当 new_dur < 0.75*det_dur 才收缩（微小修剪为 no-op）
    eubc_flat_eps: float = 1e-6        # 数值退化防护（flat coherence 显式 no-op，非调参阈值）
    # v111 QSCORE: localization-quality Q-score confidence re-rank（单开关；
    # False = 字节级恢复旧行为）。PURE post-hoc 置信度重排：在 conf-gate/ECE/
    # EUBC 之后、最终 dict 投影之前，作用于 FINAL detections——绝不改变边界或
    # 数量（FM1/FM2 免疫）。Q = mean(boundary_sharpness, peak_contrast,
    # center_fit)，在输出 frame_scores 曲线上计算，视频内 min-max 归一化后
    # conf' = conf * (base + (1-base)*Q)。离线在 v108 真实检测上确定性 A/B
    # 复评 +0.0087（无 VLM 噪声），broad-based（AP@0.3/0.5/0.7 全升）。
    qscore_rerank: bool = True  # v114 KEEP-FLAG-ON: clean fresh-detection A/B delta +0.00619 (>=+0.005). New baseline map=0.15632.
    qscore_base: float = 0.5
    # v142 BUE: Boundary-Union Extension（单开关；False = 字节级恢复旧行为）。
    # TERMINAL-MOST 槽位（_qscore_rerank 之后、v141 CEX 块与 DetectionResult
    # 返回之前）：对 FINAL detections 在 verdict-INDEPENDENT 的只读类别 CLIP
    # 曲线（frame_scorer.get_category_scores，与 _clip_boundary_single_scale
    # 完全同一次调用、与 analysis_v140_livecurve_lib 58/58 live-peak 保真的
    # 同一 substrate）上：平滑 w=3 -> 取该检测自身的曲线 argmax 作 anchor ->
    # 跑 v51 半峰值 CLOSURE TEST（hm = (seg.max()+percentile(curve,20))/2；
    # 两端都已低于 hm 说明边界确由信号交叉决定，字节级跳过）-> 否则网格搜索
    # 含 anchor 的 (s,e)，长度 ∈ [bue_lmin_sec, min(bue_lmax_sec,
    # bue_lmax_frac*duration)]，最大化 AutoLoc 式 outer-inner contrast
    # OIC(s,e) = mean(inside) - weighted_mean(两侧宽 alpha*(e-s) 的 outer ring)
    # -> 发射「旧窗口 ∪ OIC argmax 窗口」。
    # UNION 是机制本身：离线 3-cache 消融 REPLACE = -0.0065 vs UNION = +0.0378。
    # 只写 t_start/t_end；confidence / 检测数量 / anomaly_type / 排序 / NMS /
    # merge / 所有 prompt 全部逐位冻结，零新增 VLM/CLIP/MIL 调用
    # （FM2/FM3/FM4/FM14 免疫）。
    bue_enabled: bool = True           # v8: SHIPPED default-ON after 290-video paired verdict (+0.0216 mAP, P=0.938); --no_bue_extension to disable
    bue_alpha: float = 1.0             # outer ring 宽度 = alpha * 窗口长度
    bue_lmin_sec: float = 6.0          # 候选窗口最短秒数（= 既有 scan segment 长度）
    bue_lmax_sec: float = 40.0         # 候选窗口最长秒数（M6: 40 与 50 逐位等价，
                                       # 因为 0.8*duration 在 40 以上先绑定；
                                       # [15,60] 内 24/24 arm-cache 为正；90 塌到 -0.00117）
    bue_lmax_frac: float = 0.8         # 候选窗口长度相对视频时长的上限比例
    bue_smooth_win: int = 3            # 曲线平滑窗（帧）
    bue_closure_gate: bool = True      # v51 半峰值 closure test（mAP no-op，纯 blast-radius 缩减）
    # v155 CBT: Category-conditioned Boundary Tightening（单开关；False = 字节级
    # 恢复旧行为）。TERMINAL 槽位（BUE 之后、CEX 之前）：对 4 个 hard category
    # (Shooting/RoadAccidents/Shoplifting/Explosion) 中 det_dur > 1.5*cap 的过宽
    # FINAL 检测，沿 read-only 类别曲线 (frame_scorer.get_category_scores) 选取
    # AutoLoc outer-inner contrast 最大的子窗口 [peak-hw,peak+hw]，clamp 到原始
    # [t_start,t_end]（单调内缩，永不外扩），仅当 new_dur<det_dur 且
    # new_dur>=min_segment 时发射。confidence / 检测数量 / anomaly_type / 排序 /
    # coarse_score / frame_scores 全部逐位冻结（count-neutral，FM2 免疫）。
    # CBT_CAPS（GT-p75，轻度 padding）见 pipeline.py。BUE 是严格外扩、CBT 是
    # 严格内缩，二者互不干涉（CBT 在 BUE 之后运行，BUE 字节不变）。
    cbt_enabled: bool = True  # PROMOTED default-ON (v164 @290 paired marginal +0.02283, P(delta>0)=0.918; crossrun hard +0.0688); --no_cbt_tighten to disable
    # v160 CBT-RECAL（单开关 --cbt_recal，与 --cbt_tighten 同用；默认 OFF =
    # 字节级恢复 v155 CBT-on 行为）。仅替换 _cbt_category_boundary_tighten 的两个
    # 标量输入：caps -> GT-p75 recal 值，over-wide 触发倍率 1.5 -> cbt_recal_trigger。
    # 方法逻辑（contrast objective / monotone-inward / count-neutral /
    # confidence-frozen）完全不变。flag-off 时 caps==CBT_CAPS 且 trig==1.5。
    cbt_recal: bool = True  # PROMOTED default-ON with CBT (v164 @290 paired marginal +0.02283, P=0.918); --no_cbt_recal to disable
    cbt_recal_caps: dict = field(default_factory=lambda: {
        'RoadAccidents': 4.5, 'Explosion': 13.0,
        'Shoplifting': 9.0, 'Shooting': 13.5})
    cbt_recal_trigger: float = 1.1
    # v167 CGR-M: Count-neutral motion-energy-peak candidate RELOCATION（单开关
    # --cgr_relocate；默认 OFF = 字节级恢复旧行为，motion 从不为重定位读取）。
    # PRE-BUE 槽位（在 pre_bue snapshot 之前）：对 FINAL 检测列表（count 已冻结、
    # confidence 逐位固定），若类别命中 cgr_categories 且当前无任何检测覆盖
    # per-frame motion-energy peak，则把 confidence 最低的那条检测 IN-PLACE 平移
    # 到 motion peak（保宽 w=t_end-t_start，仅改一条的 t_start/t_end）。严格
    # count-neutral（无 append/pop/insert/del/filter/reorder），confidence /
    # anomaly_type / coarse_score / 排序全部不变。paired 判据字段 pre_cgr_boundaries
    # 由 pipeline 填充。这是 CBT（post-BUE 沿 CLIP 曲线的内缩）无法触及的 cand-miss
    # lever：把一条被浪费的窗口重定位到 scan 漏掉的 kinematic 信号上。
    cgr_enabled: bool = False  # v169 CGR-M: MEASURED-CLOSED, default-OFF PERMANENTLY. DO NOT RE-LITIGATE. Closed by an EXACT PAIRED end-to-end dual-arm A/B (single run, zero cross-run VLM noise; arm-OFF = BUE+CBT on a deepcopy of the pre-CGR list, arm-ON = CGR then BUE+CBT = the run's real output): hard-88 point_delta -0.00487 on FINAL mAP (arm-ON 0.09743 vs arm-OFF 0.10231), CI90 [-0.02491, +0.01140], P(delta>0)=0.224. Mechanism DID fire (30/134 dets = 22.4% relocated) and was count-neutral (134 dets both arms, 0 length mismatches), so this is a real negative, not a no-op. Per-cat: RoadAccidents +0.0293 (4 relocs) is more than cancelled by Shoplifting -0.0318 (11 relocs); Shooting +0.0000, Explosion +0.0011. Damage is at loose IoU (AP@0.3 -0.00859, AP@0.5 -0.00603) and AP@0.7 is UNCHANGED (0.02020 both arms) => CGR buys no tight-localization gain, its stated purpose. HISTORY: the v15 +0.06105 "paired" figure was INFLATED (pre_cgr_boundaries is snapshotted PRE-BUE, so that delta bundled the committed BUE+CBT gains); the offline CGR-only reconstruction (+0.01919, P=0.886) could not see the BUE/CBT interaction and did NOT transfer (FM6). Code + the reusable dual-arm instrument (cgr_ab_diag) stay in the tree; --cgr_relocate still enables for research only.
    cgr_categories: tuple = ('Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion')
    cgr_smooth_win_sec: float = 1.0
    # v169 CGR-AB: DUAL-ARM in-run A/B INSTRUMENT for CGR-M（单开关
    # --cgr_ab_diag；默认 OFF = 字节级旧行为；仅在 cgr_enabled 同时为 True 时生效）。
    # 纯 DIAGNOSTIC：不改变本次 run 的任何最终输出（final_anomalies /
    # confidence / count / frame_scores 全部逐位不变），只额外产出一个
    # per-detection 字段 ab_arm_off_boundaries。
    # 为什么单 run 就够：CGR -> CCR -> BUE -> CBT 全部在所有 VLM 调用之后执行，
    # 只读 cached per-frame motion energy 与 cached CLIP 类别曲线，是 DETERMINISTIC
    # post-processing。因此同一次 run 内可以同时产出两条 arm：
    #   arm-OFF = 在 pre-CGR 检测列表的 DEEP COPY 上跑同样的 downstream chain
    #             （CCR if enabled -> BUE -> CBT）= committed pipeline 的输出
    #   arm-ON  = 正常路径（CGR 之后同样的 chain）= 本次 run 的真实最终输出
    # 同视频、同 VLM 采样、同检测数量 -> EXACT PAIRED A/B，ZERO cross-run VLM
    # 噪声；两条 arm 只差 EXACTLY ONE 次 _cgr_motion_relocate 调用。
    # v15 的教训（为什么必须这样测）：pre_cgr_boundaries 的快照点在 pre-BUE，
    # 因此那条 paired delta(+0.06105) 把 committed BUE+CBT 的收益一起打包进去了，
    # 是 INFLATED、不是 CGR-only；offline 重建出的 CGR-only 是 +0.01919
    # (P=0.886)，但 offline 又无法 capture BUE/CBT 的交互（CGR 移窗之后 BUE 从
    # 新位置向外生长、CBT 顶着类别曲线的 DIFFERENT 区段内缩）。CGR 对 final mAP
    # 的真实效应恰恰就是这个 end-to-end 量 —— 本 flag 直接测它。
    # 判据：analysis_bootstrap_ab.py --paired-json ab_arm_off_boundaries
    # （长度不一致 HARD-VOID 免费继承）。
    cgr_ab_diag: bool = False  # DEFAULT OFF (v169 CGR-AB dual-arm instrument); --cgr_ab_diag to enable (requires --cgr_relocate)
    # v168 CCR-SE: Count-neutral CLIP-CATEGORY-CURVE semantic candidate
    # RELOCATION（单开关 --ccr_relocate；默认 OFF = 字节级恢复旧行为，曲线从不
    # 为重定位读取）。这是 v167 CGR-M（motion-energy peak）的结构性克隆，唯一
    # 差异是信号源从 video_data['motion_energy'] 换成 read-only 的 per-frame
    # 相对 CLIP 类别曲线 frame_scorer.get_category_scores（cache key
    # _cat_scores_<cat>）。运行在 PRE-BUE 槽位、CGR 之后、pre_bue snapshot 之前：
    # 对 FINAL 检测列表（count 冻结、confidence 逐位固定），若类别命中
    # ccr_categories 且当前无任何检测覆盖平滑后曲线 argmax，则把 confidence 最低
    # 的那条检测 IN-PLACE 平移到 curve peak（保宽，仅改一条 t_start/t_end）。严格
    # count-neutral（无 append/pop/insert/del/filter/reorder），confidence /
    # anomaly_type / coarse_score / 排序全部不变。曲线只读，永不 rescale/写回
    # （非 banned-#4 per-video normalization）。gated 到 motion-only CGR-M
    # under-delivered 的两个 low-motion cand-miss 类别（Shoplifting/Shooting），
    # RoadAccidents/Explosion 显式排除 -> 那两个类别 AP 逐位不变（保护 floor）。
    # paired 判据字段 pre_ccr_boundaries 由 pipeline 填充。
    ccr_enabled: bool = False  # DEFAULT OFF (v168 CCR-SE); --ccr_relocate to enable
    ccr_categories: tuple = ('Shoplifting', 'Shooting')
    ccr_smooth_win_sec: float = 1.0
    # v172 CNV variant "TIGHTEN"（REPLACES v171 gate 逻辑；单开关 --cnv_veto；
    # 默认 OFF = 字节级旧行为，任何信号都不读取）。TERMINAL 槽位（BUE/CBT/CEX
    # 之后、DetectionResult 返回之前的 genuinely-final final_anomalies 上）的
    # ONE-SIDED 置信度否决：只写 confidence := 0.0，检测保留在列表中
    # （count-frozen、index-aligned），永不移动窗口、永不新增检测、永不重排。
    # STAGE 1（视频级 global-null gate）：当且仅当四个独立条件同时为负时否决
    # 该视频的全部检测——(a) MIL 视频级证据 top-k mean(_mil_scores) < 0.20；
    # (b) 视频自身 fused 类别曲线的 in-video outlierness max((x-median)/MAD)
    # < 1.5；(c) 运动能量超过 (median+3*MAD) 的帧占比 < 0.01；(d) v172 TIGHTEN
    # 新增 NECESSARY 合取：视频自身 videomae_category 必须为 'Normal'（terminal
    # 槽位读 post-CLIP-override 存储字段；early-return-normal 路径上 early
    # VideoMAE 已判 normal）。STAGE 2（segment 级 in-video surprisal veto）：
    # 对通过 stage 1 的视频，否决窗口曲线质量低于视频自身中位数的检测
    # （v172 TIGHTEN：z_window < 0，与 v171 上界 z_window < 0.5 合取保留，
    # 且窗口不含视频全局曲线 argmax）。每个 v172 fire 都是 v171 fire 的严格
    # 子集（构造性更保守）。零新增 VLM/CLIP/MIL 调用（全部读 cached 信号）。
    # paired 判据：LOG-based —— 每条被否决的检测产生一行 [CNV-FIRE]（vid +
    # t_start + t_end + conf=old->new），arm-OFF 由
    # analysis_bootstrap_ab.py --paired-log <run.log> --marker CNV-FIRE
    # --paired-mode conf EXACT 重建。FM17：[CNV-VID] per-video 摘要无条件发射
    # （含 0-detection 路径与 early-return-normal 路径）。
    cnv_veto: bool = False  # DEFAULT OFF (v172 CNV TIGHTEN); --cnv_veto to enable
    # v179 SNH: Supervised Normality Head（单开关 --snh_veto；默认 OFF = 字节级
    # 旧行为，model_snh.json 从不被读取，任何信号都不读取）。v17-record-mandated、
    # v19-GO-gated 的 NORMAL-video false-positive lever。mechanism = video-level
    # P(normal)>=tau veto from model_snh.json（v19 训练产物：16 个 cached-signal
    # 特征经 SHARED 模块 snh_features.compute_snh_features 计算（train/eval 无
    # skew），z-standardize 后 P(normal)=sigmoid(w·z+b)，tau=0.7894，CV AUC
    # 0.996，CV anomaly-flag 3.8% / normal-flag 94%）。TERMINAL 槽位 ONE-SIDED
    # 置信度否决：gate 命中时该视频全部检测 confidence := 0.0，检测保留
    # （count-frozen、index-aligned，FM2 assert），永不移动窗口、永不新增、永不
    # 重排。VIDEO-LEVEL-ONLY gate —— 无 per-segment stage-2。零新增 VLM/CLIP/MIL
    # 调用（全部读 cached 信号：video_data['mil_scores'] / 13 条 cached 类别曲线
    # frame_scorer.get_category_scores / cached clip_features 派生的 Normal
    # softmax 曲线 / video_data['motion_energy']）。model 文件缺失/损坏 ->
    # [SNH-ERR] 且该视频按 flag-off 处理，NEVER crash。paired adjudication =
    # --paired-log --marker SNH-FIRE --paired-mode conf
    # （analysis_bootstrap_ab.py restore_pre_marker 从 [SNH-FIRE] 行 EXACT 重建
    # arm-OFF）。FM17: [SNH-VID] per-video 摘要无条件发射（含 0-detection 与
    # early-return-normal 路径，两者 ndet=0 vetoed=0 + 实算 p_normal）。
    # DEV-ONLY: SNH_FORCE_FIRE=1 强制 gate 命中（self-test hook）。
    snh_veto: bool = False  # DEFAULT OFF (v179 SNH); --snh_veto to enable
    # v21 CLOSED 2026-08-16 (summary_v21_snh-eval_20260816.md): G4 GT-harm FAIL 5>1 (RoadAccidents128 IoU 0.733), G1 paired VOID (M4 key collisions); code stays in-tree inert, default-OFF.
    # v196 VLC: VLM video-Level Calibration（单开关 --vlc_calib；默认 OFF =
    # 字节级旧行为，不发起任何额外 VLM 调用、不写任何 marker）。动机：跨视频
    # confidence-LEVEL oracle 是 +0.0518（v11 diagnostic）；mAP 按 confidence
    # 全局排序，因此校准每个视频的 LEVEL 会移动 mAP。MIL-signal 变体已 CLOSED
    # （v183），但 VLM-READ 的 video-level 信号尚未测试。TERMINAL 槽位
    # （CNV/SNH 之后、DetectionResult 返回之前的 genuinely-final
    # final_anomalies 上）的 VIDEO-LEVEL 置信度校准：对 ndet>=1 的视频，跨
    # 整段时长均匀抽 16 帧，向 detect-client (qwen-vl-plus) 发起 ONE 次
    # video-level 调用（"Does this video contain any abnormal activity..."），
    # 鲁棒解析出 v∈[0,1]（任何失败 -> v=0.5 且记 [VLC-PARSE-FALLBACK]），
    # 然后对每条检测做 conf_new = conf_old * (0.5 + 0.5*v)。count-frozen、
    # windows-frozen、order-frozen（只写 confidence，永不移动/新增/重排）。
    # 0-detection 视频 SKIP 该 VLM 调用（省成本）但仍记 [VLC-VID]。paired
    # 判据：LOG-based —— 每条被校准的检测产生一行 [VLC-FIRE]（vid + idx +
    # cat + t=ts-te + conf=old->new），arm-OFF 由 analysis_bootstrap_ab.py
    # --paired-log <run.log> --marker VLC-FIRE --paired-mode conf EXACT 重建。
    # FM17：[VLC-VID] per-video 摘要无条件发射（含 0-detection 与
    # early-return-normal 路径；未发起调用时 v=NA）。COST：ndet>=1 的视频
    # 恰好多 +1 次 detect VLM 调用（自动计入 [COST] 的 vlm_calls delta）。
    vlc_calib: bool = False  # DEFAULT OFF (v196 VLC); --vlc_calib to enable
    # v196 CLOSED 2026-08-22 (summary_v196_vlc_20260822.md): paired verdict VOID (M4 key collisions — 134 [VLC-FIRE] fires occupy 121 unique (vid,rounded-window) keys, matched=134 != n_fires=121); observed arm-ON hard-88 mAP 0.13229 not adjudicable, no delta reported; VLM-read level signal NOT refuted; code stays in-tree inert, default-OFF.
    # v197 DSV: Dual-model Second-opinion Verification（单开关 --dsv_verify；
    # 默认 OFF = 字节级旧行为，不发起任何额外 VLM 调用、不写任何 marker）。
    # 动机：scan VLM (qwen-vl-plus) 只看孤立的 6s 段、绝对置信度未校准
    # （v143 根因）；用一个 INDEPENDENT 第二模型（classify client
    # qwen3-vl-plus —— 与 scan 不同的模型）对每条 FINAL 检测窗口做
    # detection-level（非 v196 的 video-level）二次判读，加入一路独立证据。
    # TERMINAL 槽位（SNH/VLC 之后、DetectionResult 返回之前的
    # genuinely-final final_anomalies 上）：对每条 FINAL 检测，在
    # [t_start, t_end] 内均匀抽 <=8 帧，向 CLASSIFY client (qwen3-vl-plus)
    # 发起一次调用（"these frames come from a clip flagged as <category>...
    # Rate whether this clip genuinely shows <category>-type abnormal
    # activity. Answer ONLY JSON {\"score\": a}, a in [0,1]."），鲁棒解析出
    # a∈[0,1]（任何失败 -> a=0.5 且记 [DSV-PARSE-FALLBACK]），然后
    # conf_new = conf_old * (0.6 + 0.4*a)。count-frozen、windows-frozen、
    # order-frozen（只写 confidence）。COST CAP：>8 条检测的视频只对
    # confidence top-8 重打分，其余不动（无 FIRE 行）。0-detection 视频
    # 不发起调用但仍记 [DSV-VID]。paired 判据：LOG-based —— 每条被重打分的
    # 检测产生一行 [DSV-FIRE]（vid + idx + cat + t=ts-te + conf=old->new），
    # arm-OFF 由 analysis_bootstrap_ab.py --paired-log <run.log> --marker
    # DSV-FIRE --paired-mode conf EXACT 重建。FM17：[DSV-VID] per-video
    # 摘要（vid + ncalls + ndet）无条件发射（含 0-detection 与
    # early-return-normal 路径）。COST：每个被重打分的检测恰好多 +1 次
    # classify VLM 调用（自动计入 [COST] 的 vlm_calls delta）。
    dsv_verify: bool = False  # DEFAULT OFF (v197 DSV); --dsv_verify to enable
    # v197 CLOSED 2026-08-22 (summary_v197_dsv_20260822.md): paired verdict VOID (M4 key collisions — 134 [DSV-FIRE] fires occupy 126 unique (vid,rounded-window) keys, matched=134 != n_fires=126, last-write-wins on 8 dup keys); observed arm-ON hard-88 mAP 0.17092 not adjudicable, no delta reported; independent second-model detection-level verification NOT refuted; code stays in-tree inert, default-OFF.
    # v198 MCS: MaxCat curve Selection for BUE/CBT（单开关 --mcs_curves；默认
    # OFF = 字节级旧行为，不读取任何额外曲线、不写任何 marker/字段）。
    # 动机：BUE 外扩与 CBT 内缩都沿「resolved category」的 CLIP 曲线操作；若
    # resolved category 错了（或事件在另一类曲线上读数更强），边界算子跟随
    # 的是错误信号。MCS 把两个算子的曲线换成全部 13 条类别曲线的 POINTWISE
    # MAX（maxcat(t) = max_c curve_c(t)）：确定性 post-VLM（全部读 cached
    # clip_features 的 get_category_scores，零新增 VLM/CLIP 调用），
    # count-frozen、conf-frozen（BUE/CBT 本就只写 t_start/t_end，不变）。
    # CBT-RECAL 的 per-category caps 在两条 arm 里都仍按 RESOLVED category
    # 查表（只有曲线更换）。DUAL-ARM：pre-BUE 检测列表 deep-copy 一份跑
    # committed BUE+CBT(+recal)（resolved-category 曲线）-> arm-OFF 窗口，
    # 写入 per-video 顶层 JSON 字段 pre_mcs_boundaries（与 anomalies
    # index-aligned，v141 pre_cex_boundaries 同款 plumbing）；真实列表跑
    # BUE+CBT(+recal)（curve_mc）-> arm-ON 窗口 = 本次 run 的真实输出。
    # paired 判据：analysis_bootstrap_ab.py --paired-json pre_mcs_boundaries
    # （长度不一致 HARD-VOID 免费继承）。FM17：[MCS-VID] per-video 摘要
    # （vid + ndet + n_changed）无条件发射（含 0-detection 与
    # early-return-normal 路径，0 检测时字段以 [] 写出）；[MCS-FIRE] 每条
    # 窗口发生变化的检测一行（vid + idx + old->new window），仅观测用途。
    mcs_curves: bool = False  # DEFAULT OFF (v198 MCS); --mcs_curves to enable
    # v198 CLOSED 2026-08-22 (summary_v198_mcs_20260822.md): paired-json[pre_mcs_boundaries] REVERT — point_delta -0.03507, CI90 [-0.07411,-0.00531], P(delta>0)=0.024, new88 mAP 0.08192 (all 3 gates FAIL); harm mechanism: maxcat single-peak collapses multiple detections onto one window (Explosion002 both dets -> 51.5-54.5) and pulls tight-window cats onto foreign curves (Shooting 0.10030->0.00631); resolved-category curve VALIDATED as the committed BUE/CBT signal; code stays in-tree inert, default-OFF.
    # v141 CEX: Continuation-Evidence Extension（单开关；False = 字节级恢复旧
    # 行为）。TERMINAL-MOST 槽位（_qscore_rerank 之后、DetectionResult 返回
    # 之前）：对 FINAL detections 每一侧发起一次 BINARY、NUMBER-FREE 的
    # VLM 续延投票（"同一事件在这段 6s 相邻片段里是否仍在进行？"），仅在
    # confident yes 时把该侧向外扩 cex_step_sec，首次 no 永久关闭该侧。
    # 几何由代码以固定步长施加，VLM 永不输出时间戳（v62 失败模式的根因）。
    # 检测数量 / coarse_score / confidence / 排序 / 半峰值代码全部不变，
    # 只有已有窗口的两个端点向外移动（FM2/FM4/FM14 免疫）。
    # M3: 生长仅由 cex_max_steps 与 cex_max_total_sec 约束——不继承 v105/v107
    # ECE 的两个中和器（CATEGORY_DURATION_PRIORS max_dur reject-cap 与
    # curve-grid snap），它们是 v107 in-vivo 只剩 +0.00203 的直接原因。
    cex_extend: bool = False
    cex_step_sec: float = 6.0          # 单步外扩秒数（= 既有 scan segment 长度）
    cex_max_steps: int = 3             # 每侧最多轮数
    cex_min_conf: float = 0.7          # accept 阈值（直接设定 false-extension 率 q）
    cex_max_total_sec: float = 60.0    # 单条检测总生长绝对上限
    # v137 NOCLS-CA (KEPT latency win): skip the Stage-7 _ensemble_classify VLM
    # sub-tree (qwen3-vl-plus, <=3 calls/video) and return the upstream VideoMAE
    # label. mAP-NEUTRAL BY CONSTRUCTION — compute_temporal_map reads only
    # {video_id, t_start, t_end, confidence}, never anomaly_type; the confidence
    # gate / curve correction / frame scores all key off videomae_category, not
    # the classify result. Verified ~-7.5% per-video VLM calls (v132 in-vivo).
    # TRADEOFF: degrades the separate Level-4 category-accuracy metric (NOT mAP).
    # Opt out with --no_classify_skip_class_agnostic.
    classify_skip_class_agnostic: bool = True
    # v199 FDS: Frame-Density Scan（单开关 --fds_scan；默认 OFF = 字节级恢复旧
    # 行为，directed scan 行为逐位不动）。flag ON 且 pipeline 解析的
    # videomae_category ∈ {Shooting, RoadAccidents, Shoplifting, Explosion}
    # （pipeline.FDS_CATEGORIES）时，Stage 4 DIRECTED 扫描（两条 scan_segment
    # 路径：_directed_llm_scan 与 discovery.discover 兜底）把每个【已选中】的
    # 6s 段送入 VLM 的帧数从 ~12（2fps）翻倍到 ~24（4fps）——同一 6s 窗口、
    # 同一均匀放置、密度翻倍。实现：按 2×preprocess.fps 重解码得到稠密帧池
    # （其余预处理参数不变），_sample_keyframes 按时间段从稠密池重取帧；
    # scan_segment 的 call/prompt/JSON schema 不动。candidate build（CLIP-loc
    # 定向 + 通用预筛 + min_pass/max_scan 上限）在原始 2fps 网格上计算
    # （选择与 flag-off 逐位一致），merge/NMS/classify/BUE/CBT 全部不动；
    # video_data['segments'] 本体不被改动；FCCS comparative 路径不受影响
    # （FDS 仅在非 _fccs 时 arm）。Markers: [FDS-FIRE] 每个 GATED 视频恰好一行
    # （vid, category, frames_per_segment base -> dense）；[FDS-VID] 每视频
    # 无条件一行（含 0-detection 与 early-return-normal，vid, gated=0|1,
    # n_segments）。
    fds_scan: bool = False  # DEFAULT OFF (v199 FDS); --fds_scan to enable


@dataclass
class DiscoveryConfig:
    """Stage 2 粗粒度异常发现参数。"""
    clip_threshold: float = 0.3
    min_pass_rate: float = 0.2   # 至少扫描段比例
    max_scan_segments: int = 6   # 扫描段数硬上限（pipeline 会按段数动态调整）
    candidate_top_k: int = 3     # 保留得分最高的候选数
    # v97 FCCS: 全覆盖比较式 VLM 扫描（单开关；False = 字节级恢复旧行为）
    full_coverage_comparative_scan: bool = False
    comparative_batch_segments: int = 8    # 每次比较式调用打包的连续段数
    comparative_keyframes: int = 2         # 每段关键帧数（长视频降为 1）
    comparative_long_segments: int = 60    # 段数超过此值走长视频批量规则
    comparative_long_batch: int = 16       # 长视频每次调用打包段数（1 帧/段）
    # v173 KFD-U: Uniform-spread keyframes in the comparative scan（单开关
    # --kfd_uniform_keyframes；默认 OFF = 字节级恢复旧行为，first-k 截断路径
    # 逐位不动）。pre-registered by the v17 council：_comparative_call 把
    # _sample_keyframes 返回的 ~12 帧截断成前 keyframes_per_segment 帧（6s 段
    # = 前 2 帧 = t+0.0s/t+0.5s；>60 段长视频规则 = 前 1 帧），VLM 对每个 6s
    # 段最多看到其前 ~0.5s。KFD-U 用确定性均匀展开替换该截断：k=2 取时间戳
    # 最接近 {t_start, t_start+3.0s} 的两帧；1-帧规则取段中点最近帧；k=3
    # （fallback knob）取 {t_start, t_start+2.5s, t_start+5.0s}。段网格 /
    # 批量几何（8x2 / 16x1）/ 每次调用 16 帧 / 调用次数 / prompt / 阈值 /
    # merge / top-k 全部不变 —— 零延迟成本。注意：该截断只存在于比较式扫描
    # 路径（scan_full_coverage -> _scan_comparative_batch_guarded ->
    # _comparative_call；committed pipeline 的 directed-scan 路径不做任何
    # first-k 截断），因此 --kfd_uniform_keyframes 置 ON 时 pipeline 经由该
    # 比较式路径执行扫描（机制有定义的唯一路径）；flag OFF 时行为与
    # committed pipeline 逐位一致。Markers: [KFD-FIRE] 每个发生选择的批量
    # 一行；[KFD-VID] 每视频无条件一行（含 0-detection 与 early-return-normal）。
    kfd_uniform_keyframes: bool = False  # DEFAULT OFF (v173 KFD-U); --kfd_uniform_keyframes to enable
    kfd_k: int = 2                       # pre-registered arms: 2={t,t+3.0s}; 3={t,t+2.5,t+5} (fallback); 1=midpoint
    # v177 TCP: Temporal-Contrast Prompt（单开关 --tcp_prompt；默认 OFF =
    # 字节级恢复旧行为，scan_segment prompt 逐位不动）。机制为 prompt-side
    # ONLY：committed directed scan 已把每 6s 段的全部 ~12 帧展示给 VLM
    # （非 frame-blind），但 RA009（GT 7-8s，落在段 4.5-10.5 内）与 RA021
    # （GT 1-3s，落在段 0-5.2 内）在全帧展示下仍被 VLM 判负 —— 假设是
    # 整段绝对 prompt（"Determine if this segment contains any abnormal or
    # suspicious activity"）诱发 clip-AVERAGING，使 1-3s 短暂事件在大量
    # 正常帧中稀释。flag ON 时 scan_segment 在 "Respond ONLY in JSON
    # format" 行之前紧接追加 pre-registered TCP 块（模块级常量
    # anomaly_discovery.TCP_PROMPT_BLOCK；context_prompt 与其余一切不变）：
    # 同样的帧、同样的调用次数、同样的 directed 路径、同样的 JSON schema
    # （is_anomalous/confidence/description）—— 只有 prompt 文本变化，零
    # 延迟成本。不属于任何 banned family（非 abs-conf gate、非 merge/batch、
    # 非 scan-density）。Markers: [TCP-VID] 每视频无条件一行（含 0-detection
    # 与 early-return-normal，镜像 CNV/KFD call sites），紧随其后 [TCP-BLOCK]
    # 每视频记录完整 TCP 块一次（FM5: grep -cF "Temporal reasoning
    # instructions"），[TCP-PROMPT] 每视频记录首个段的完整实际 prompt 一次。
    tcp_prompt: bool = False  # DEFAULT OFF (v177 TCP); --tcp_prompt to enable
    # v182 FSD: Fine Segment Division（单开关 --fsd_cats；默认 OFF = 字节级
    # 恢复旧行为，directed scan 行为逐位不动）。flag ON 且 pipeline 解析的
    # videomae_category ∈ {Shoplifting, RoadAccidents}（pipeline.FSD_CATEGORIES）
    # 时，Stage 4 DIRECTED 扫描（两条 scan_segment 路径：_directed_llm_scan
    # 与 discovery.discover 兜底）把每个【已选中】的 6s 段切成 2 个按时间
    # 顺序的 3s 半段（anomaly_discovery.fsd_split_segment，单一来源；2fps
    # 下各 ~6 帧；尾半段可更短——>=3 帧即保留，<3 帧丢弃；两个半段都不足
    # 3 帧时原段整段保留）。切分发生在段选择【之后】、扫描迭代处：candidate
    # build（CLIP-loc 定向 + 通用预筛 + min_pass/max_scan 上限）在原始 6s
    # 网格上计算（选择与 flag-off 逐位一致），每个子段走同样的 scan_segment
    # 调用/prompt/JSON schema；merge/NMS/classify/BUE/CBT 全部不动。
    # video_data['segments'] 本体不被改动，>60 段长视频规则
    # （comparative_long_segments，FCCS 专用）等一切按段数计算的规则仍在
    # 原始 6s 网格上；_fccs 开关不受影响（保持 OFF；FSD 不作用于
    # comparative 路径）。逐视频 arming：discovery.fsd_split（默认 False）。
    # Markers: [FSD-FIRE] 每个 GATED 视频恰好一行（vid, category,
    # n_segments_6s -> n_segments_3s）；[FSD-VID] 每视频无条件一行（含
    # 0-detection 与 early-return-normal，vid, gated=0|1, n_segments）。
    fsd_cats: bool = False  # DEFAULT OFF (v182 FSD); --fsd_cats to enable
    # v182 CLOSED 2026-08-18 (session v24): hard-88 cross-run REVERT — point_delta -0.02083, P(delta>0)=0.16, purpose population NOT converted; flag stays OFF in-tree inert.


@dataclass
class VADConfig:
    """VAD pipeline 完整配置（聚合所有子配置 + 全局项）。"""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    device: str = "cpu"
    anomaly_categories: List[str] = field(default_factory=list)

    # ---- 扁平化访问（便于 pipeline 内部直接取值）----
    @property
    def fps(self) -> int: return self.preprocess.fps
    @property
    def segment_duration(self) -> float: return self.preprocess.segment_duration
    @property
    def segment_overlap(self) -> float: return self.preprocess.segment_overlap
    @property
    def clip_model_name(self) -> str: return self.clip.model_name
    @property
    def clip_pretrained(self) -> str: return self.clip.pretrained
    @property
    def mcts_simulations(self) -> int: return self.mcts.simulations
    @property
    def mcts_max_depth(self) -> int: return self.mcts.max_depth
    @property
    def semantic_tau(self) -> float: return self.clip.semantic_tau
    @property
    def max_frames(self) -> int: return self.clip.max_frames
    @property
    def min_frames(self) -> int: return self.clip.min_frames
    @property
    def fps_sample(self) -> float: return self.clip.fps_sample
    @property
    def motion_threshold(self) -> float: return self.preprocess.motion_threshold
    @property
    def min_motion_interval(self) -> float: return self.preprocess.min_motion_interval

    # ---- 工厂 ----
    @classmethod
    def from_args(cls, args: Any) -> "VADConfig":
        """从 argparse Namespace 构造（适配 run.py / batch_evaluate.py）。"""
        cats = list(getattr(args, "anomaly_categories", None) or [])
        return cls(
            preprocess=PreprocessConfig(
                fps=getattr(args, "fps", 2),
                segment_duration=getattr(args, "segment_duration", 6.0),
                segment_overlap=getattr(args, "segment_overlap", 0.25),
                motion_threshold=getattr(args, "motion_threshold", 1.0),
                min_motion_interval=getattr(args, "min_motion_interval", 2.0),
            ),
            clip=CLIPConfig(
                model_name=getattr(args, "clip_model", "ViT-L-14"),
                pretrained=getattr(args, "clip_pretrained", "openai"),
                max_frames=getattr(args, "max_frames", 200),
                min_frames=getattr(args, "min_frames", 3),
                fps_sample=getattr(args, "fps_sample", 3.0),
            ),
            mcts=MCTSConfig(
                simulations=getattr(args, "mcts_simulations", 12),
                max_depth=getattr(args, "mcts_max_depth", 4),
            ),
            scoring=ScoringConfig(
                # v100 EUBC 开关（默认 False = 旧行为）。
                # FM5 根因修复：此前 from_args 完全不构造 scoring 对象，
                # ScoringConfig 上的开关永远无法到达 pipeline（v98/v99 激活
                # 失败的直接原因）。必须显式 threading；flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                event_unit_boundary_contraction=getattr(
                    args, "event_unit_boundary_contraction", False),
                # v105 ECE 开关（默认 False = 旧行为）；同一 FM5 根因修复
                # 模式：显式 threading，flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                event_continuation_extension=getattr(
                    args, "event_continuation_extension", False),
                # v124 BASELINE-FLIP: v114 KEEP-FLAG-ON completed. Q-score
                # re-rank is now the committed baseline (map=0.15632), so the
                # from_args default is True. Opt-out via --no_qscore_rerank.
                qscore_rerank=getattr(args, "qscore_rerank", True),
                # v137 NOCLS-CA (KEPT latency win): default-ON classify-path
                # DROP; mAP-neutral by construction. Opt out via
                # --no_classify_skip_class_agnostic.
                classify_skip_class_agnostic=getattr(
                    args, "classify_skip_class_agnostic", True),
                # v141 CEX 开关（默认 False = 旧行为）；同一 FM5 根因修复
                # 模式：全部 5 个参数显式 threading，flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                cex_extend=getattr(args, "cex_extend", False),
                cex_step_sec=getattr(args, "cex_step_sec", 6.0),
                cex_max_steps=getattr(args, "cex_max_steps", 3),
                cex_min_conf=getattr(args, "cex_min_conf", 0.7),
                cex_max_total_sec=getattr(args, "cex_max_total_sec", 60.0),
                # v142 BUE 开关（默认 False = 旧行为）；同一 FM5 根因修复
                # 模式（v99/v105/v141 激活失败的记录在案根因就是漏 threading）：
                # 全部 7 个参数显式 threading，flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                bue_enabled=getattr(args, "bue_enabled", True),
                bue_alpha=getattr(args, "bue_alpha", 1.0),
                bue_lmin_sec=getattr(args, "bue_lmin_sec", 6.0),
                bue_lmax_sec=getattr(args, "bue_lmax_sec", 40.0),
                bue_lmax_frac=getattr(args, "bue_lmax_frac", 0.8),
                bue_smooth_win=getattr(args, "bue_smooth_win", 3),
                bue_closure_gate=getattr(args, "bue_closure_gate", True),
                # v155 CBT 开关（默认 False = 旧行为）；同一 FM5 根因修复模式：
                # 显式 threading，flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                cbt_enabled=getattr(args, "cbt_enabled", True),
                # v160 CBT-RECAL 开关（默认 False = 字节级 v155 CBT-on）；同一
                # FM5 显式 threading 模式。caps/trigger 保留 dataclass 默认值。
                cbt_recal=getattr(args, "cbt_recal", True),
                # v167 CGR-M 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                cgr_enabled=getattr(args, "cgr_enabled", False),
                # v169 CGR-AB dual-arm A/B instrument 开关（默认 False = 字节级
                # 旧行为）；同一 FM5 显式 threading 模式。flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                cgr_ab_diag=getattr(args, "cgr_ab_diag", False),
                # v168 CCR-SE 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                ccr_enabled=getattr(args, "ccr_enabled", False),
                # v172 CNV TIGHTEN 开关（默认 False = 字节级旧行为）；同一 FM5
                # 显式 threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                cnv_veto=getattr(args, "cnv_veto", False),
                # v179 SNH 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                snh_veto=getattr(args, "snh_veto", False),
                # v196 VLC 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                vlc_calib=getattr(args, "vlc_calib", False),
                # v197 DSV 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                dsv_verify=getattr(args, "dsv_verify", False),
                # v198 MCS 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。
                mcs_curves=getattr(args, "mcs_curves", False),
                # v199 FDS 开关（默认 False = 字节级旧行为，directed scan 逐位
                # 不动）；同一 FM5 显式 threading 模式。flag-off 时
                # from_args(empty).scoring == ScoringConfig()（字节级等价）。
                fds_scan=getattr(args, "fds_scan", False),
            ),
            discovery=DiscoveryConfig(
                # v97 FCCS 开关（默认 False = 旧行为）
                full_coverage_comparative_scan=getattr(args, "full_coverage_scan", False),
                # v173 KFD-U 开关（默认 False = 字节级旧行为）；同一 FM5 显式
                # threading 模式。flag-off 时 from_args(empty).discovery ==
                # DiscoveryConfig()（字节级等价）。
                kfd_uniform_keyframes=getattr(args, "kfd_uniform_keyframes", False),
                kfd_k=getattr(args, "kfd_k", 2),
                # v177 TCP 开关（默认 False = 字节级旧行为，prompt 逐位不动）；
                # 同一 FM5 显式 threading 模式。flag-off 时
                # from_args(empty).discovery == DiscoveryConfig()（字节级等价）。
                tcp_prompt=getattr(args, "tcp_prompt", False),
                # v182 FSD 开关（默认 False = 字节级旧行为，6s 段网格不动）；
                # 同一 FM5 显式 threading 模式。flag-off 时
                # from_args(empty).discovery == DiscoveryConfig()（字节级等价）。
                fsd_cats=getattr(args, "fsd_cats", False),
            ),
            device=getattr(args, "device", "cpu"),
            anomaly_categories=cats,
        )

    # ---- 序列化（实验复现）----
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "VADConfig":
        """从 JSON 加载配置（覆盖默认值，缺字段保留默认）。"""
        data = json.loads(Path(path).read_text())
        return cls(
            preprocess=PreprocessConfig(**data.get("preprocess", {})),
            clip=CLIPConfig(**data.get("clip", {})),
            mcts=MCTSConfig(**data.get("mcts", {})),
            scoring=ScoringConfig(**data.get("scoring", {})),
            discovery=DiscoveryConfig(**data.get("discovery", {})),
            device=data.get("device", "cpu"),
            anomaly_categories=data.get("anomaly_categories", []),
        )


__all__ = [
    "VADConfig", "PreprocessConfig", "CLIPConfig", "MCTSConfig",
    "ScoringConfig", "DiscoveryConfig",
]
