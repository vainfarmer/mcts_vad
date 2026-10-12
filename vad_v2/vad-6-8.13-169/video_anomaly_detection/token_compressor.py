"""
语义帧采样模块（Token 压缩器）

解决的核心问题：VLM（如 qwen-vl-max）的 token 预算有限（通常 ~200 帧），
但视频段的帧数可能远超上限。直接均匀下采样会发送大量内容相似的冗余帧，
浪费 token 预算且可能淹没关键帧的信息。

两阶段选帧策略：
  阶段 1（动态帧数）：根据时间段长度 + MCTS 搜索深度动态计算目标帧数 K
    公式：K = clamp(ceil(duration × fps_effective), min_frames, max_for_depth)
    长段多帧覆盖全貌，短段少帧但保证下限（min_frames），深层节点密采样做精判

  阶段 2（语义去重）：用 CLIP 特征做贪心聚类，将相似度 > tau 的帧归为同簇，
    每簇只保留最独特的一帧（与簇内其他帧平均相似度最低的那个）
    典型效果：监控视频中大量静态背景帧被聚为一簇，只保留一帧代表

整体效果：在不丢失关键视觉信息的前提下，将发送给 LLM 的帧数压缩 2-5x。
"""

import math
import numpy as np
from typing import List, Optional


class SemanticFrameSampler:
    """
    动态帧数 + 语义去重的帧采样器

    被 AnomalyDiscovery（Stage 2）和 TemporalBoundaryMCTS（Stage 3）共用：
      - Stage 2: 每个初始段采样后发送给 LLM 做异常判断
      - Stage 3: MCTS 每次 simulation 采样后发送给 LLM 确认异常是否还在

    复用 anomaly_expert 的 CLIP 编码能力做语义去重，不引入额外模型。
    当 CLIP 不可用时自动降级为均匀采样（_uniform_fallback）。
    """

    def __init__(self, anomaly_expert=None, tau: float = 0.92,
                 max_frames: int = 200, min_frames: int = 3,
                 fps_sample: float = 1.0):
        """
        Args:
            anomaly_expert: VideoAnomalyExpert 实例，复用其 CLIP 编码器做帧特征提取。
                           若为 None 则跳过语义去重，退化为均匀采样
            tau: 帧间余弦相似度阈值（0-1），高于此值的帧对被视为冗余。
                 0.92 适合监控场景（连续静态帧相似度通常 > 0.95）
            max_frames: 帧数硬上限，防止超过 LLM 的 token 承载能力
            min_frames: 帧数下限，保证 LLM 有足够视觉信息做出判断。
                       设为 3 因为低于 3 帧 LLM 几乎无法感知时间变化
            fps_sample: 每秒目标采样帧数，用于动态帧数计算的基准值
        """
        self.expert = anomaly_expert
        self.tau = tau
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.fps_sample = fps_sample

    def compute_target_frames(self, duration: float,
                              depth: int = 0, n_shrinks: int = 0) -> int:
        """
        根据时间段长度 + MCTS 搜索状态动态计算目标帧数

        策略：
          - 浅层（depth 小）：段长，粗采样做概览
          - 深层（depth 大 + 多次 shrink）：段短，密采样做精判

        公式:
          fps_effective = fps_sample × (1 + 0.5 × n_shrinks)
          max_for_depth = max_frames / (1 + depth × 0.3)
          K = clamp(ceil(duration × fps_effective), min_frames, max_for_depth)
        """
        # 深层节点段更短但需要密采样做精判，所以随 shrink 次数提高采样率
        fps_effective = self.fps_sample * (1.0 + 0.5 * n_shrinks)
        # 深层节点帧数上限随深度递减，避免浅层就用完 token 预算
        depth_max = max(self.min_frames, int(self.max_frames / (1 + depth * 0.3)))
        k = math.ceil(duration * fps_effective)
        return max(self.min_frames, min(k, depth_max))

    def select_keyframes(self, frames: np.ndarray, duration: float = 0.0,
                         depth: int = 0, n_shrinks: int = 0,
                         frame_indices=None,
                         precomputed_features=None,
                         return_positions=False) -> np.ndarray:
        """
        从帧数组中选出语义最多样的关键帧（主入口）

        两步流程：
          1. 先用 compute_target_frames 确定目标帧数 K，均匀下采样到 K 帧
          2. 对这 K 帧做 CLIP 语义去重，去除视觉冗余帧

        当 CLIP 不可用或去重抛异常时，降级返回均匀采样的 K 帧。

        Args:
            frames: np.ndarray [N, H, W, 3]，原始帧数组
            duration: 时间段长度（秒），用于计算动态帧数
            depth: MCTS 当前节点深度（影响帧数上限）
            n_shrinks: 路径上执行的 shrink 操作次数（影响采样密度）
            frame_indices: 帧在全局 all_frames 中的索引（用于查表预计算特征）
            precomputed_features: 预计算的全局 CLIP 特征矩阵

        Returns:
            np.ndarray [K', H, W, 3]，K' <= K，语义去重后的关键帧
        """
        if duration > 0:
            target = self.compute_target_frames(duration, depth, n_shrinks)
        else:
            target = min(len(frames), self.max_frames)
        target = min(target, len(frames))

        # 帧数已在目标以下，无需采样
        if len(frames) <= target:
            if return_positions:
                return frames, np.arange(len(frames))
            return frames

        # Step 1: 均匀预采样到目标帧数（保证时间覆盖均匀）
        pre_indices = np.linspace(0, len(frames) - 1, target, dtype=int)
        pre_sampled = frames[pre_indices]

        # 无 CLIP 时直接返回均匀采样结果
        if self.expert is None or self.expert.model is None:
            if return_positions:
                return pre_sampled, pre_indices
            return pre_sampled

        # 映射到全局帧索引，用于查表预计算的 CLIP 特征
        if precomputed_features is not None and frame_indices is not None:
            sub_indices = [frame_indices[i] for i in pre_indices]
        else:
            sub_indices = None

        # Step 2: CLIP 语义去重（失败时降级为均匀采样）
        try:
            selected_positions = self._semantic_deduplicate_indices(
                pre_sampled, sub_indices, precomputed_features
            )
            selected = pre_sampled[selected_positions]
        except Exception:
            if return_positions:
                return pre_sampled, pre_indices
            return pre_sampled
        if return_positions:
            return selected, pre_indices[np.asarray(selected_positions, dtype=int)]
        return selected

    def _semantic_deduplicate(self, frames: np.ndarray,
                              frame_indices=None,
                              precomputed_features=None) -> np.ndarray:
        """语义去重，返回去重后的帧数组（行为与旧版逐位一致）。

        v173 KFD-U: 实际逻辑移至 _semantic_deduplicate_indices（返回被选中
        帧的整数位置），本方法只是 frames[indices] 的薄封装。"""
        return frames[self._semantic_deduplicate_indices(
            frames, frame_indices, precomputed_features)]

    def _semantic_deduplicate_indices(self, frames: np.ndarray,
                                      frame_indices=None,
                                      precomputed_features=None) -> List[int]:
        """
        对预采样帧做 CLIP 语义去重，返回被选中帧在 frames 中的位置列表
        （升序）。逻辑与旧版 _semantic_deduplicate 完全一致，仅返回值从
        frames[selected_indices] 改为 selected_indices（v173 KFD-U 需要
        位置来查时间戳做确定性均匀展开）。

        流程：
          1. 获取帧的 CLIP 特征（优先用预计算特征查表，否则实时编码）
          2. 计算帧间余弦相似度矩阵（L2 归一化后矩阵乘法即余弦相似度）
          3. 贪心聚类：相似度 > tau 的帧归入同簇
          4. 每簇选一个代表帧（簇内最独特的那个）

        保底：去重后帧数不足 min_frames 时，从未选中帧中补充
        """
        import torch
        import torch.nn.functional as F

        # 特征获取：预计算模式下通过索引查表（O(1)），否则实时编码
        if precomputed_features is not None and frame_indices is not None:
            features = precomputed_features[frame_indices]
        else:
            features = self.expert._encode_frames(frames, max_frames=len(frames))
            features = F.normalize(features, p=2, dim=-1)
        # 特征已 L2 归一化，矩阵乘法直接得到余弦相似度
        sim_matrix = (features @ features.T).cpu().numpy()

        # 贪心聚类而非层次/谱聚类：帧数少（通常 <200）且需确定性结果，贪心足够
        clusters = self._greedy_cluster(sim_matrix)
        selected_indices = self._pick_representatives(clusters, sim_matrix)

        # 保证去重后不低于 min_frames
        if len(selected_indices) < self.min_frames and len(frames) >= self.min_frames:
            remaining = [i for i in range(len(frames)) if i not in selected_indices]
            while len(selected_indices) < self.min_frames and remaining:
                selected_indices.append(remaining.pop(0))

        selected_indices.sort()
        return selected_indices

    def _greedy_cluster(self, sim_matrix: np.ndarray) -> List[List[int]]:
        """
        贪心聚类：按帧顺序扫描，将与簇心相似度 > tau 的帧归入同簇

        算法：对每个未分配的帧 i，以 i 为簇心，向后扫描所有未分配帧 j，
        若 sim(i,j) > tau 则 j 归入 i 的簇。O(N²) 但 N 通常 < 200。

        选择贪心而非层次聚类/谱聚类的原因：
          - 帧数少（< 200），复杂算法的精度优势可忽略
          - 结果确定性（相同输入必定相同输出），便于缓存和调试
          - 实现简单，无需引入 sklearn 等额外依赖
        """
        n = sim_matrix.shape[0]
        assigned = [False] * n
        clusters = []

        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if not assigned[j] and sim_matrix[i, j] > self.tau:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)

        return clusters

    def _pick_representatives(
        self, clusters: List[List[int]], sim_matrix: np.ndarray
    ) -> List[int]:
        """
        从每个簇中选一个代表帧

        策略：选与簇内其他帧平均相似度最低的帧（最独特的那个），
        这样保留了最多信息量。如果簇只有一个帧则直接选。
        """
        representatives = []
        for cluster in clusters:
            if len(cluster) == 1:
                representatives.append(cluster[0])
            else:
                sub_sim = sim_matrix[np.ix_(cluster, cluster)]
                avg_sim = sub_sim.mean(axis=1)
                # 选簇内平均相似度最低的帧：它与其他帧最不同，携带最多独特信息
                best_local = int(np.argmin(avg_sim))
                representatives.append(cluster[best_local])
        return representatives

    def _uniform_fallback(self, frames: np.ndarray) -> np.ndarray:
        """无 CLIP 时降级为均匀采样"""
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
        return frames[indices]
