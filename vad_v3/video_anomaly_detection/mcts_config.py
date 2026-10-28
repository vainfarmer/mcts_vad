"""Validated configuration shared by the single-video and batch entry points."""
from dataclasses import dataclass, fields
import argparse
import math


@dataclass
class MCTSConfig:
    simulations: int = 12
    max_depth: int = 4
    padding_ratio: float = 0.75  # Legacy search() only.
    early_stop_agree: int = 4
    early_stop_min_sim: int = 5
    # off不搜索；shadow搜索但不应用；refine允许应用通过全部门槛的修改。
    mode: str = 'off'
    action_set: str = 'shrink'
    search_strategy: str = 'mcts'
    # best_valid只比较最佳合格提案；reward_gain还要求超过根节点的最小奖励增益。
    comparison_gate: str = 'best_valid'
    # relative_nonloss允许原事件不完整且比较打平，但必须保留原区间已有事件内容。
    acceptance_policy: str = 'strict'
    # all_valid取消对比阈值筛选，但仍要求证据有效、存在合法动作并遵守预算。
    trigger_policy: str = 'low_contrast'
    # 仅语义全部通过时允许一次引用纠正，仍占原请求预算；0可关闭。
    citation_repair_attempts: int = 1
    # 以下是逻辑预算（非HTTP重试次数）；根评价和最终A/B比较也消耗预算。
    max_candidates_per_video: int = 2
    max_vlm_evaluations_per_candidate: int = 10
    max_vlm_evaluations_per_video: int = 20
    timeout_seconds_per_candidate: float = 90.0
    # 触发阈值作用于原始内外差值，不是异常概率，也不是归一化后的quality。
    contrast_threshold: float = 0.15
    search_padding_ratio: float = 0.25
    max_inward_shift_ratio: float = 0.25
    max_outward_shift_ratio: float = 0.25
    step_ratio: float = 0.10
    min_interval_seconds: float = 0.5
    # 图像预算和时间覆盖约束共同决定可搜索性，增加采样率可能增加所需图片数。
    max_images_per_evaluation: int = 16
    context_seconds: float = 2.0
    max_boundary_gap_seconds: float = 1.0
    max_evidence_gap_seconds: float = 2.0
    min_unique_evaluations: int = 4
    patience: int = 3
    min_reward_gain: float = 0.02
    max_curve_quality_drop: float = 0.05
    ucb_c: float = 1.0
    ucb_decay: float = 0.0
    random_seed: int = 42
    trace_enabled: bool = False
    # 配对模式共享上游，独立运行三条尾流程；整次token不是单独on分支成本。
    paired_eval: bool = False
    # 最终去重对off/on均生效；保护仅作用于MCTS已接受来源。默认保持旧流程，消融后再决定启用。
    final_nms_enabled: bool = False
    final_nms_iou: float = 0.3
    boundary_guard_enabled: bool = False
    guard_min_retained_ratio: float = 0.5
    guard_max_shift_ratio: float = 0.25
    guard_tolerance_seconds: float = 0.5
    postprocess_ablation: bool = False
    dense_fps: float = 0.0  # 0: conservative original frame pool.
    max_dense_frames: int = 512
    max_enumerated_states: int = 10000

    def __post_init__(self):
        for key, allowed in {
            'mode': ('off', 'shadow', 'refine'),
            'action_set': ('shrink', 'shrink_shift', 'shrink_expand_shift'),
            'search_strategy': ('mcts', 'greedy', 'uniform'),
            'comparison_gate': ('best_valid', 'reward_gain'),
            'acceptance_policy': ('strict', 'relative_nonloss'),
            'trigger_policy': ('low_contrast', 'all_valid'),
        }.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f'{key} must be one of {allowed}')
        if type(self.citation_repair_attempts) is not int or self.citation_repair_attempts not in (0,1):
            raise ValueError('citation_repair_attempts must be 0 or 1')
        integers = ('simulations', 'max_depth', 'max_candidates_per_video',
                    'max_vlm_evaluations_per_candidate', 'max_vlm_evaluations_per_video',
                    'max_images_per_evaluation', 'min_unique_evaluations', 'patience',
                    'max_dense_frames', 'max_enumerated_states')
        for key in integers:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(f.default, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{f.name} must be finite and nonnegative')
        for key in ('timeout_seconds_per_candidate', 'step_ratio', 'min_interval_seconds',
                    'context_seconds', 'max_boundary_gap_seconds', 'max_evidence_gap_seconds'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if not 0 <= self.max_inward_shift_ratio < 1:
            raise ValueError('max_inward_shift_ratio must be in [0,1)')
        if self.max_outward_shift_ratio > self.search_padding_ratio:
            raise ValueError('outward displacement exceeds search padding')
        if self.max_vlm_evaluations_per_candidate < 2 or self.max_vlm_evaluations_per_video < 2:
            raise ValueError('Budget must reserve root evaluation and final comparison')
        if self.paired_eval and self.mode != 'refine':
            raise ValueError('paired_eval requires mode=refine')
        if self.postprocess_ablation and not self.paired_eval:
            raise ValueError('postprocess_ablation requires paired_eval')
        if self.postprocess_ablation and not (self.final_nms_enabled and self.boundary_guard_enabled):
            raise ValueError('postprocess_ablation requires final_nms_enabled and boundary_guard_enabled for main arm')
        for key in ('final_nms_iou', 'guard_min_retained_ratio'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in (0,1]')

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, 'mcts_' + f.name, f.default) for f in fields(cls)})


def add_mcts_arguments(parser):
    for f in fields(MCTSConfig):
        option = '--mcts_' + f.name
        if option in parser._option_string_actions:
            continue
        kw = {'default': f.default}
        if isinstance(f.default, bool):
            kw['action'] = argparse.BooleanOptionalAction
        else:
            kw['type'] = type(f.default)
        parser.add_argument(option, **kw)
    parser.add_argument('--mcts_trace', dest='mcts_trace_enabled', action='store_true',
                        default=argparse.SUPPRESS)
