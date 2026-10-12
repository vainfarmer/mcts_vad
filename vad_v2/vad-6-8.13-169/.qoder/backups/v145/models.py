"""
VAD 数据模型 (dataclasses)

解决的核心问题：pipeline 之前全程用裸 dict 传递，导致字段遗漏类 bug
（如 frame_scores=None 在某些返回路径下丢失）。

本模块定义带类型的领域模型，作为 pipeline 的输入/输出契约：
  - Segment          : 视频初始分段（Stage 1 产物）
  - VideoData        : 单视频上下文（贯穿全 pipeline 的核心数据结构）
  - AnomalySegment   : 检测到的异常段（含时间/置信度/类别）
  - DetectionResult  : detect() 的最终返回

设计原则：
  1. 输出模型 (AnomalySegment / DetectionResult) 强类型，杜绝字段遗漏
  2. VideoData 保持 dict 兼容（外部模块 mcts/discovery/expert 仍按 dict 访问），
     通过 __getitem__/__setitem__/get/contains 适配，可渐进迁移
  3. 所有模型提供 to_dict() 用于 JSON 序列化，保持与现有 batch_evaluate.py 兼容
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@dataclass
class Segment:
    """视频初始分段 — Stage 1 预处理产物。

    Attributes:
        t_start: 段起始时间（秒）
        t_end:   段结束时间（秒）
        frame_indices: 该段在 all_frames 中的全局帧索引
    """
    t_start: float
    t_end: float
    frame_indices: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "frame_indices": list(self.frame_indices),
        }


@dataclass
class AnomalySegment:
    """检测到的异常段（pipeline 输出的核心单元）。

    必填字段（t_start/t_end/confidence/anomaly_type）保证每条异常完整，
    可选字段（depth/visits/description/coarse_score）用于追溯调试。

    Attributes:
        t_start: 异常起始时间（秒）
        t_end:   异常结束时间（秒）
        confidence: 异常置信度 [0,1]，>= 0.72 通过 confidence gate
        anomaly_type: 异常类别（UCF-Crime 13 类之一）
        depth:  MCTS 树深度（追溯用，可选）
        visits: MCTS 访问次数（追溯用，可选）
        description: CLIP/VLM 给出的异常描述（可选）
        coarse_score: Stage 2 粗粒度分数（可选）
    """
    t_start: float
    t_end: float
    confidence: float
    anomaly_type: str = "unknown"
    depth: Optional[int] = None
    visits: Optional[int] = None
    description: Optional[str] = None
    coarse_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "depth": self.depth,
            "visits": self.visits,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnomalySegment":
        """从 dict 构造（兼容 pipeline 内部 dict 表示）。"""
        return cls(
            t_start=float(d["t_start"]),
            t_end=float(d["t_end"]),
            confidence=float(d.get("confidence", 0.0)),
            anomaly_type=d.get("anomaly_type", "unknown"),
            depth=d.get("depth"),
            visits=d.get("visits"),
            description=d.get("description"),
            coarse_score=d.get("coarse_score"),
        )


@dataclass
class DetectionResult:
    """detect() 的最终返回 — 替代之前的裸 dict。

    frame_scores / timestamps 设为 Optional 显式表达「可能为空」，
    配合 compute_frame_level_scores 返回 None 的场景，杜绝字段遗漏 bug。

    Attributes:
        video_path: 输入视频路径
        duration:   视频时长（秒）
        is_normal:  是否判定为正常视频（无异常）
        anomalies:  检测到的异常段列表
        frame_scores: 逐帧连续异常分数 [0,1]（与 timestamps 等长）
        timestamps:   frame_scores 对应的时间戳（秒）
        processing_time: 处理耗时（秒，可选，由调用方填充）
        videomae_category: VideoMAE 全视频分类结果（调试用）
    """
    video_path: str
    duration: float
    is_normal: bool
    anomalies: List[AnomalySegment] = field(default_factory=list)
    frame_scores: Optional[List[float]] = None
    timestamps: Optional[List[float]] = None
    processing_time: Optional[float] = None
    videomae_category: Optional[str] = None
    # v105 ECE: ECE 生效前的 [t_start, t_end] 列表（与 anomalies 顺序一一
    # 对应）。仅当 ECE 开关开启时由 pipeline 填充，用于同一 run 内的
    # 确定性 pre/post A/B 复评（噪声免疫的 keep/revert 判据）。
    # None（flag-off）时不序列化 —— 输出 JSON 与旧行为字节级一致。
    pre_ece_boundaries: Optional[List[List[float]]] = None
    # v141 CEX (M4): CEX 生效前的 [t_start, t_end] 列表（与 anomalies 顺序
    # index-aligned；CEX 数量不变，因此长度必然等于 len(anomalies)）。仅当
    # cex_extend 开启时由 pipeline 填充。用途：paired flag-off arm 直接从
    # result JSON 精确重建，无需正则解析 run.log、无浮点取整键冲突。
    # None（flag-off）时不序列化 —— 输出 JSON 与旧行为字节级一致。
    pre_cex_boundaries: Optional[List[List[float]]] = None
    # v142 BUE (M1): BUE 生效前的 [t_start, t_end] 列表（与 anomalies 顺序
    # index-aligned；BUE 数量不变，因此长度必然等于 len(anomalies)）。仅当
    # bue_enabled 开启时由 pipeline 填充。用途：paired flag-off arm 直接从
    # result JSON 精确重建（analysis_bootstrap_ab.py --paired-json
    # pre_bue_boundaries），无需正则解析 run.log、无浮点取整键冲突，并免费
    # 继承该工具的长度不一致 HARD-VOID 检查。
    # None（flag-off）时不序列化 —— 输出 JSON 与旧行为字节级一致。
    pre_bue_boundaries: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict，供 batch_evaluate.py 写 JSON。

        anomalies 只保留 t_start/t_end/confidence/anomaly_type/depth/visits，
        与 pipeline v33 原有输出字段完全一致，保持向后兼容。
        v105: pre_ece_boundaries 仅在非 None（ECE flag-on）时写出。
        """
        d = {
            "video_path": self.video_path,
            "duration": self.duration,
            "is_normal": self.is_normal,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "frame_scores": self.frame_scores,
            "timestamps": self.timestamps,
        }
        if self.pre_ece_boundaries is not None:
            d["pre_ece_boundaries"] = self.pre_ece_boundaries
        if self.pre_cex_boundaries is not None:
            d["pre_cex_boundaries"] = self.pre_cex_boundaries
        if self.pre_bue_boundaries is not None:
            d["pre_bue_boundaries"] = self.pre_bue_boundaries
        return d


class VideoData(dict):
    """单视频上下文（贯穿全 pipeline 的核心数据结构）。

    保持 dict 兼容：pipeline 内部大量按 video_data['key'] 访问，
    外部模块 (AnomalyDiscovery / TemporalMCTS / VideoAnomalyExpert) 也按 dict 取值。
    继承 dict + 文档化 schema，允许渐进迁移而不破坏现有调用。

    标准键:
        video_path (str):            输入视频路径
        duration (float):            视频时长（秒）
        all_frames (np.ndarray):     采样后的帧数组 [N,H,W,3] RGB uint8
        timestamps (np.ndarray):     每帧时间戳（秒）
        segments (list[Segment|dict]):初始分段列表
        video_fps (float):           原始视频帧率

    运行期动态扩展键:
        clip_features (Tensor):      Stage 1 CLIP 预计算特征
        motion_energy (np.ndarray):  Stage 1b 运动能量
        videomae_category (str):     Stage 2 VideoMAE 分类结果
        cat_scores_<category> (np.ndarray): CLIP 类别分数缓存
    """

    # 声明 schema 供类型检查器/文档参考（运行期为普通 dict）
    video_path: str
    duration: float
    all_frames: np.ndarray
    timestamps: np.ndarray
    segments: List[Any]
    video_fps: float

    @property
    def n_frames(self) -> int:
        return len(self.get("all_frames", []))


__all__ = ["Segment", "AnomalySegment", "DetectionResult", "VideoData"]
