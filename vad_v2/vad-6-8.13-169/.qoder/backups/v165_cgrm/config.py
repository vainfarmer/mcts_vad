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
    # v164 CGR-R: Category-curve Guided count-neutral candidate Relocation（单开关
    # --cgr；默认 OFF = 字节级恢复 committed BUE+CBT+recal 行为，不读曲线）。PRE-BUE
    # 槽位：对 4 个 hard category 视频，若类别曲线全局峰所在窗口未被任何现有检测覆盖，
    # 且存在一个「低置信 + 自身曲线区间远低于峰 + 非局部极大」的可牺牲检测，则把该
    # 检测就地移动到峰窗口（RELOCATE-ONLY，count 严格不变，confidence 逐位冻结）。
    # 随后未改动的 BUE 向外生长、CBT 向内收紧。FM2 因零新增检测而 BY CONSTRUCTION 免疫。
    cgr_enabled: bool = False
    # M1: 真实峰显著度阈值（vacuous percentileofscore 的替代）——
    # prom = (cs.max()-median(cs)) / (cs.max()-cs.min()+eps)；平坦曲线（Shoplifting
    # near-uniform，即 11/25 cand-miss 的根因）prom 低 -> 不发射，避免在近随机峰上重定位。
    cgr_min_prom: float = 0.35
    cgr_win: float = 3.5               # 重定位窗口半宽（秒），窗口 = [t_peak-w, t_peak+w]
    # M2: 尺度相对的源区间保护余量（曲线动态范围的比例，替代 ABS 常数，因
    # get_category_scores 数值尺度未确立）。margin = cgr_margin_frac*(cs.max()-cs.min())。
    cgr_margin_frac: float = 0.15
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
                # v164 CGR 开关（默认 False = committed 行为）；同一 FM5 显式
                # threading 模式：flag-off 时 from_args(empty).scoring ==
                # ScoringConfig()（字节级等价）。cgr_min_prom/cgr_win/
                # cgr_margin_frac 保留 dataclass 默认值。
                cgr_enabled=getattr(args, "cgr_enabled", False),
            ),
            discovery=DiscoveryConfig(
                # v97 FCCS 开关（默认 False = 旧行为）
                full_coverage_comparative_scan=getattr(args, "full_coverage_scan", False),
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
