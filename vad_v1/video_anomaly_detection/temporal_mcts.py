"""
时间维度 MCTS 搜索模块

核心类比 DyFo 的空间 MCTS → 时间 MCTS：

  DyFo (空间):
    state   = region_coords (x1, y1, x2, y2)
    action  = crop (repeat_question) / zoom_out
    reward  = 1 - valid_area_ratio  (if all objects present)
    expert  = Lang_SAM → bounding boxes
    目标     = 包含所有目标的最小图像区域

  VAD (时间):
    state   = time_segment (t_start, t_end)
    action  = focus_half / focus_third / expand / shift
    reward  = (1 - temporal_ratio) * llm_confidence + expert_bonus
    expert  = CLIP zero-shot → anomaly score + peak_time
    目标     = 包含特定异常的最短时间段

对每一个异常候选独立运行一次 MCTS，收敛到该异常的精确时间边界。
"""

import math
import random
import re
from typing import List, Dict, Any, Optional, Tuple


class TemporalMCTSNode:
    """时间维度 MCTS 树节点"""

    def __init__(self, state, parent=None, available_actions=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.leaf_reward = 0
        self.untried_actions = available_actions.copy() if available_actions else []
        self.temporal_ratio = state.get('temporal_ratio', 1.0)
        self.anomaly_info = None
        self.q_hat = {}
        self.action_visits = {}
        self.action_rewards = {}


class TemporalBoundaryMCTS:
    """
    对单个异常候选做时间边界精确定位的 MCTS 搜索

    优化点：
      - Expert 引导 expansion（peak_time 决定聚焦方向）
      - 三等分动作提升搜索精度
      - LLM 结果缓存避免重复调用
      - Prompt 返回 confidence 融入 reward
    """

    def __init__(self, anomaly_description, video_data, llm_client,
                 anomaly_expert=None, frame_sampler=None, image_processor=None,
                 anomaly_categories=None, max_images=4):
        """
        Args:
            anomaly_description: 要定位的异常类型描述（来自 Stage 2 的 LLM 判断）
            video_data: VideoPreprocessor.process() 的输出字典
            llm_client: VLM 客户端（async generate）
            anomaly_expert: CLIP 异常专家，提供异常分数和 peak_time 引导搜索方向
            frame_sampler: 语义帧采样器，控制发送给 LLM 的帧数
            image_processor: 图像预处理器（可选）
            anomaly_categories: 异常类别列表（用于 LLM prompt）
            max_images: 无 frame_sampler 时的最大帧数
        """
        self.anomaly_desc = anomaly_description
        self.video_data = video_data
        self.llm = llm_client
        self.anomaly_expert = anomaly_expert
        self.frame_sampler = frame_sampler
        self.image_processor = image_processor
        self.anomaly_categories = anomaly_categories or []
        self.max_images = max_images

        # MCTS 超参数
        self.max_depth = 5           # 搜索树最大深度，限制收缩次数
        self.c_puct = 1.0            # UCB 探索系数（基准值，会随模拟次数衰减）
        self.n_simulations = 12      # MCTS 模拟次数（由 Pipeline 覆盖）
        self.min_duration = 0.5      # 时间段最短长度（秒），防止过度收缩
        self.adaptive_lambda = 0.1   # UCB 探索衰减速率
        self._current_sim = 0        # 当前模拟轮次，用于自适应 UCB

        # 动作空间：只有两个收缩方向，通过深度自适应比例控制每次收缩量
        self.actions = [
            "shrink_left",   # 左边界右移，丢弃左侧内容
            "shrink_right",  # 右边界左移，丢弃右侧内容
        ]

        self.root = None
        # LLM 结果缓存：以 (t_start, t_end) 四舍五入到 0.1s 为键
        # 不同模拟路径可能到达相同时间段，缓存可节省大量 API 调用
        self._llm_cache = {}

    # ================================================================
    # 动作执行器（4 个正交边界操作）
    # ================================================================

    def _shrink_ratio(self, depth):
        """深度自适应收缩比例：浅层大步收缩，深层精细收缩"""
        # depth=0 → 40% 收缩；depth=3 → ~21%，避免深层过度裁剪丢失异常
        return 0.4 / (1 + 0.3 * depth)

    async def _execute_shrink_left(self, node):
        """
        左边界右移：丢弃左侧 ratio 比例的内容

        例：段 [0, 10]，ratio=0.4 → 新段 [4, 10]，丢弃了 0-4s 的内容
        如果异常在右侧，此操作可快速排除左侧无关内容
        """
        duration = node.state['t_end'] - node.state['t_start']
        ratio = self._shrink_ratio(node.state['depth'])
        t_start = node.state['t_start'] + duration * ratio
        return await self._create_child(node, t_start, node.state['t_end'],
                                         "shrink_left")

    async def _execute_shrink_right(self, node):
        """
        右边界左移：丢弃右侧 ratio 比例的内容

        例：段 [0, 10]，ratio=0.4 → 新段 [0, 6]，丢弃了 6-10s 的内容
        如果异常在左侧，此操作可快速排除右侧无关内容
        """
        duration = node.state['t_end'] - node.state['t_start']
        ratio = self._shrink_ratio(node.state['depth'])
        t_end = node.state['t_end'] - duration * ratio
        return await self._create_child(node, node.state['t_start'], t_end,
                                         "shrink_right")


    async def _execute_shift_left(self, node):
        """整体窗口左移：保持长度不变，向左平移 20% 的段长"""
        duration = node.state['t_end'] - node.state['t_start']
        shift_amount = duration * 0.2
        orig_start = node.state['original_t_start']
        t_start = max(orig_start, node.state['t_start'] - shift_amount)
        t_end = t_start + duration
        if t_end > node.state['original_t_end']:
            t_end = node.state['original_t_end']
            t_start = t_end - duration
        return await self._create_child(node, t_start, t_end, "shift_left")

    async def _execute_shift_right(self, node):
        """整体窗口右移：保持长度不变，向右平移 20% 的段长"""
        duration = node.state['t_end'] - node.state['t_start']
        shift_amount = duration * 0.2
        orig_end = node.state['original_t_end']
        t_end = min(orig_end, node.state['t_end'] + shift_amount)
        t_start = t_end - duration
        if t_start < node.state['original_t_start']:
            t_start = node.state['original_t_start']
            t_end = t_start + duration
        return await self._create_child(node, t_start, t_end, "shift_right")

    async def _create_child(self, parent, t_start, t_end, action_name):
        """
        创建子节点：获取新时间段的帧，用 CLIP expert 评估，构建状态

        流程：
          1. 从预处理数据中获取 [t_start, t_end] 范围内的帧和索引
          2. 用 anomaly_expert 评估该段的 CLIP 异常分数和 peak_time
          3. 计算 temporal_ratio（当前段长度 / 原始段长度），用于 reward 计算
          4. 构建新 state 并确定可用动作（排除会导致段过短的动作）
        """
        try:
            from .video_preprocessor import VideoPreprocessor
        except ImportError:
            from video_preprocessor import VideoPreprocessor
        frames, frame_indices = VideoPreprocessor.get_segment_frames(
            self.video_data, t_start, t_end
        )

        expert_result = None
        if self.anomaly_expert is not None:
            precomputed = self.video_data.get('clip_features')
            expert_result = await self.anomaly_expert.score_segment(
                frames, t_start, t_end,
                frame_indices=frame_indices,
                precomputed_features=precomputed,
            )

        orig_duration = parent.state['original_t_end'] - parent.state['original_t_start']
        temporal_ratio = (t_end - t_start) / orig_duration

        new_state = {
            't_start': t_start,
            't_end': t_end,
            'depth': parent.state['depth'] + 1,
            'frame_indices': frame_indices,
            'action_history': parent.state['action_history'] + [action_name],
            'original_t_start': parent.state['original_t_start'],
            'original_t_end': parent.state['original_t_end'],
            'temporal_ratio': temporal_ratio,
        }

        available = [a for a in self.actions if self._is_action_valid(a, new_state)]
        child = TemporalMCTSNode(new_state, parent=parent, available_actions=available)
        child.temporal_ratio = temporal_ratio
        child.anomaly_info = expert_result
        return child

    def _is_action_valid(self, action, state):
        """
        检查动作是否合法：收缩后的段长度必须 > min_duration

        防止过度收缩导致段太短（< 0.5s），此时帧数过少无法做有效判断。
        当两个方向的 shrink 都不合法时，该节点成为叶节点，不再扩展。
        """
        duration = state['t_end'] - state['t_start']
        depth = state.get('depth', 0)
        ratio = self._shrink_ratio(depth)
        if action in ("shrink_left", "shrink_right"):
            return duration * (1 - ratio) > self.min_duration
        return True

    # ================================================================
    # MCTS 四阶段
    # ================================================================

    def selection(self, node):
        """
        MCTS 选择阶段：从根节点沿树向下选择最有价值的路径

        递归地选择 UCB 分数最高的子节点，直到到达：
          - 有未尝试动作的节点（可以展开）
          - 叶节点（无子节点，需要评估）

        UCB 公式：score = exploit + c × explore
          exploit: 该动作的估计价值（q_hat，Welford 在线均值）
          explore: 探索奖励，鼓励访问次数少的分支
          c: 探索系数，随模拟轮次指数衰减（前期探索，后期利用）
        """
        if node.untried_actions:
            return node
        if not node.children:
            return node

        # 探索系数随模拟轮次指数衰减：前期鼓励探索不同分支，后期偏向利用已知最优
        # sim=0 → c=1.0, sim=10 → c≈0.37, sim=20 → c≈0.14
        adaptive_c = self.c_puct * math.exp(-self.adaptive_lambda * self._current_sim)

        def ucb_score(action, child):
            if child.visits == 0:
                return float('inf')
            # 优先用增量更新的 q_hat（Welford 在线均值，更稳定），回退到累加均值
            exploit = node.q_hat.get(action, child.value / child.visits)
            explore = math.sqrt(2 * math.log(node.visits + 1) / (child.visits + 1e-8))
            return exploit + adaptive_c * explore

        best_action = max(node.children.keys(),
                          key=lambda a: ucb_score(a, node.children[a]))
        return self.selection(node.children[best_action])

    async def expansion(self, node):
        """
        MCTS 扩展阶段：从当前节点的未尝试动作中选一个执行，创建子节点

        扩展终止条件：达到 max_depth 或无可用动作。
        动作选择通过 _select_action_with_expert 实现 expert 引导，
        而非传统 MCTS 的随机选择——利用 CLIP peak_time 和 LLM 边界信息
        优先选择更有希望的方向。
        """
        if node.state['depth'] >= self.max_depth or not node.untried_actions:
            return node

        # Expert 引导：优先选包含 peak_time 的方向
        action = self._select_action_with_expert(node)
        node.untried_actions.remove(action)

        executors = {
            "shrink_left": self._execute_shrink_left,
            "shrink_right": self._execute_shrink_right,
        }
        child = await executors[action](node)
        node.children[action] = child
        return child

    def _select_action_with_expert(self, node):
        """用 expert peak_time + LLM 边界信息引导动作选择"""
        preferred = []

        starts_at_begin = node.state.get('starts_at_beginning', True)
        ends_at_end = node.state.get('ends_at_end', True)

        # LLM 判断异常不从头开始 → 左边有冗余，优先收缩左侧
        if not starts_at_begin:
            preferred.append("shrink_left")
        if not ends_at_end:
            preferred.append("shrink_right")

        # 无边界信息时退回 CLIP peak_time 启发式：保留包含峰值的一侧
        if not preferred and node.anomaly_info and node.anomaly_info.get('peak_time') is not None:
            peak = node.anomaly_info['peak_time']
            t_start = node.state['t_start']
            t_end = node.state['t_end']
            duration = t_end - t_start
            relative_pos = (peak - t_start) / duration if duration > 0 else 0.5

            # 峰值偏左(<0.4) → 右侧多余可裁；偏右(>0.6) → 左侧可裁；居中 → 两边都试
            if relative_pos < 0.4:
                preferred = ["shrink_right"]
            elif relative_pos > 0.6:
                preferred = ["shrink_left"]
            else:
                preferred = ["shrink_left", "shrink_right"]

        for action in preferred:
            if action in node.untried_actions:
                return action

        return random.choice(node.untried_actions)

    async def simulation(self, node):
        """
        MCTS 模拟阶段：调用 LLM 评估当前时间段是否仍包含目标异常

        这是 MCTS 中最耗时的操作（需要 API 调用），因此引入了缓存机制：
        以 (t_start, t_end) 四舍五入到 0.1s 为缓存键，相同时间段只调用一次 LLM。

        评估结果用于计算 reward，reward 融合三个信号：
          1. CLIP 异常分数（0.8 权重）：连续值，比 LLM 二值判断更精确
          2. 收缩奖励 shrink_bonus：temporal_ratio 越小（段越短）奖励越高
          3. 边界松紧度 boundary_bonus：贯穿整段 +0.05，两端有余 -0.05

        LLM 同时返回边界信息（starts_at_beginning / ends_at_end），
        指导后续扩展时优先收缩哪一侧。
        """
        t_start = node.state['t_start']
        t_end = node.state['t_end']
        frames = self.video_data['all_frames'][node.state['frame_indices']]

        # 检查缓存
        cache_key = (round(t_start, 1), round(t_end, 1))
        if cache_key in self._llm_cache:
            cached = self._llm_cache[cache_key]
            node.state['anomaly_present'] = cached['anomaly_present']
            node.state['llm_response'] = cached['response']
            node.state['starts_at_beginning'] = cached.get('starts_at_beginning', True)
            node.state['ends_at_end'] = cached.get('ends_at_end', True)
            return cached['reward_base']

        duration = t_end - t_start
        if self.frame_sampler is not None:
            depth = node.state.get('depth', 0)
            actions = node.state.get('action_history', [])
            n_shrinks = sum(1 for a in actions if a.startswith('shrink'))
            precomputed = self.video_data.get('clip_features')
            sampled_frames = self.frame_sampler.select_keyframes(
                frames, duration=duration, depth=depth, n_shrinks=n_shrinks,
                frame_indices=node.state['frame_indices'],
                precomputed_features=precomputed,
            )
        elif len(frames) > self.max_images:
            import numpy as np
            indices = np.linspace(0, len(frames) - 1, self.max_images, dtype=int)
            sampled_frames = frames[indices]
        else:
            sampled_frames = frames

        response = await self.llm.generate(
            self._build_prompt(t_start, t_end), sampled_frames
        )

        anomaly_present, llm_confidence, starts_at_begin, ends_at_end = \
            self._parse_response(response)
        node.state['anomaly_present'] = anomaly_present
        node.state['llm_response'] = response
        node.state['starts_at_beginning'] = starts_at_begin
        node.state['ends_at_end'] = ends_at_end

        if anomaly_present:
            # CLIP 异常分：连续值 [0,1]，区分度高，作为主信号
            clip_score = (node.anomaly_info.get('anomaly_score', 0.5)
                          if node.anomaly_info else 0.5)
            # LLM confidence：语义判断更准确，融入 reward 避免浪费 API 信号
            # 时间段越短 temporal_ratio 越小 → 奖励越高，激励精确定位
            shrink_bonus = (1.0 - node.temporal_ratio) * 0.15

            # 边界松紧度信号：贯穿整段说明边界合适，两端有余说明还能继续收缩
            boundary_bonus = 0.0
            if starts_at_begin and ends_at_end:
                boundary_bonus = 0.05
            elif not starts_at_begin and not ends_at_end:
                boundary_bonus = -0.05

            reward = clip_score * 0.4 + llm_confidence * 0.4 + shrink_bonus + boundary_bonus
            reward = max(0.0, min(reward, 1.0))
        else:
            reward = 0

        # 缓存结果
        self._llm_cache[cache_key] = {
            'anomaly_present': anomaly_present,
            'response': response,
            'reward_base': reward,
            'starts_at_beginning': starts_at_begin,
            'ends_at_end': ends_at_end,
        }

        return reward

    def _parse_response(self, response):
        """解析 LLM 返回的 answer + confidence + boundary info"""
        lower = response.lower()
        first_line = lower.split('\n')[0] if '\n' in lower else lower
        anomaly_present = bool(re.search(r'\byes\b', first_line))

        confidence_match = re.search(r'confidence[:\s]*(\d+)', lower)
        if confidence_match:
            llm_confidence = min(int(confidence_match.group(1)), 100) / 100.0
        else:
            llm_confidence = 0.7 if anomaly_present else 0.3

        # 解析边界信息
        starts_match = re.search(r'starts_at_beginning[:\s]*(yes|no)', lower)
        ends_match = re.search(r'ends_at_end[:\s]*(yes|no)', lower)
        starts_at_beginning = starts_match.group(1) == 'yes' if starts_match else True
        ends_at_end = ends_match.group(1) == 'yes' if ends_match else True

        return anomaly_present, llm_confidence, starts_at_beginning, ends_at_end

    def _build_prompt(self, t_start, t_end):
        categories_hint = ""
        if self.anomaly_categories:
            categories_hint = (
                f"Known anomaly categories: {', '.join(self.anomaly_categories)}\n"
            )
        return (
            f"You are analyzing a surveillance video segment "
            f"({t_start:.1f}s - {t_end:.1f}s) for temporal anomaly localization.\n"
            f"{categories_hint}"
            f"Target anomaly: \"{self.anomaly_desc}\"\n"
            f"The frames are shown in chronological order.\n\n"
            f"Answer in this exact format:\n"
            f"Answer: yes or no\n"
            f"Confidence: 0-100\n"
            f"Starts_at_beginning: yes or no (does the anomaly appear to "
            f"already be happening in the first few frames?)\n"
            f"Ends_at_end: yes or no (does the anomaly appear to still be "
            f"happening in the last few frames?)"
        )

    def backpropagation(self, node, reward):
        """
        MCTS 回传阶段：从叶节点向上回溯，沿途更新每个祖先的统计量

        更新内容：
          - node.visits: 总访问次数
          - node.value: 累计 reward（用于计算均值 = value/visits）
          - node.q_hat[action]: 该动作的 Welford 在线均值估计
            公式：q += (r - q) / n，比 value/visits 更数值稳定
          - node.action_visits[action]: 该动作被选择的次数
          - node.action_rewards[action]: 该动作的 reward 历史列表

        回溯方向：叶节点 → 父节点 → ... → 根节点
        """
        child = None
        while node:
            node.visits += 1
            node.value += reward
            if child is not None:
                for action, ch in node.children.items():
                    if ch is child:
                        node.action_visits[action] = node.action_visits.get(action, 0) + 1
                        rewards_list = node.action_rewards.get(action, [])
                        rewards_list.append(reward)
                        node.action_rewards[action] = rewards_list
                        n = node.action_visits[action]
                        # q_hat 增量更新（Welford 在线均值）：q += (r - q) / n
                        node.q_hat[action] = node.q_hat.get(action, 0) + (reward - node.q_hat.get(action, 0)) / n
                        break
            child = node
            node = node.parent

    # ================================================================
    # 主搜索流程
    # ================================================================

    async def search(self, t_start, t_end) -> List[Dict[str, Any]]:
        """
        对指定时间范围执行 MCTS 搜索，返回精确定位的异常段列表

        主循环执行 n_simulations 次迭代，每次迭代包含 MCTS 四阶段：
          1. Selection: 从根节点选择最有价值的路径
          2. Expansion: 在选中节点上尝试新动作，创建子节点
          3. Simulation: 调用 LLM 评估子节点的时间段
          4. Backpropagation: 将 reward 回传到祖先节点

        搜索结束后，从整棵树中选择 leaf_reward 最高的异常节点作为结果。

        Args:
            t_start: 搜索范围起始时间（秒）
            t_end: 搜索范围结束时间（秒）

        Returns:
            list[dict]: 精确定位的异常段，每项含 t_start, t_end, confidence,
                       anomaly_type, depth, visits
        """
        try:
            from .video_preprocessor import VideoPreprocessor
        except ImportError:
            from video_preprocessor import VideoPreprocessor
        frames, frame_indices = VideoPreprocessor.get_segment_frames(
            self.video_data, t_start, t_end
        )

        initial_state = {
            't_start': t_start,
            't_end': t_end,
            'depth': 0,
            'frame_indices': frame_indices,
            'action_history': [],
            'original_t_start': t_start,
            'original_t_end': t_end,
            'temporal_ratio': 1.0,
        }

        self.root = TemporalMCTSNode(
            initial_state,
            available_actions=[a for a in self.actions
                              if self._is_action_valid(a, initial_state)]
        )

        # 用 expert 评估根节点
        if self.anomaly_expert is not None:
            precomputed = self.video_data.get('clip_features')
            self.root.anomaly_info = await self.anomaly_expert.score_segment(
                frames, t_start, t_end,
                frame_indices=frame_indices,
                precomputed_features=precomputed,
            )

        root_reward = await self.simulation(self.root)
        self.root.leaf_reward = root_reward
        self.root.visits = 1
        self.root.value = root_reward
        print(f"    [MCTS] Root eval: {t_start:.1f}-{t_end:.1f}s, "
              f"anomaly_present={self.root.state.get('anomaly_present')}, "
              f"reward={root_reward:.3f}")

        for sim_i in range(self.n_simulations):
            self._current_sim = sim_i
            node = self.selection(self.root)
            if node.state['depth'] < self.max_depth:
                node = await self.expansion(node)
            reward = await self.simulation(node)
            node.leaf_reward = reward
            self.backpropagation(node, reward)
            print(f"    [MCTS] Sim {sim_i+1}: {node.state['t_start']:.1f}-"
                  f"{node.state['t_end']:.1f}s, depth={node.state['depth']}, "
                  f"anomaly={node.state.get('anomaly_present')}, reward={reward:.3f}")

        results = self._extract_results()
        print(f"    [MCTS] Results: {len(results)} anomaly segment(s)")
        for r in results:
            print(f"      -> {r['t_start']:.1f}-{r['t_end']:.1f}s, "
                  f"conf={r['confidence']:.3f}, type={r['anomaly_type']}")
        return results

    def _extract_results(self) -> List[Dict[str, Any]]:
        """
        从搜索树中提取最终结果

        遍历整棵树（BFS），筛选所有 anomaly_present=True 且 reward>0 的节点，
        选择其中 leaf_reward 最高的作为最终结果。

        选择 leaf_reward 而非 visits 最多的原因：
          - leaf_reward 融合了 CLIP 分数 + 时间收缩奖励 + 边界信号
          - visits 只反映探索频率，可能因为 UCB 探索项而偏向不佳的分支
          - 深层节点的 reward 更可靠（段更精确），但 visits 通常更少
        """
        all_nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)
            queue.extend(node.children.values())

        anomalous = [
            n for n in all_nodes
            if n.state.get('anomaly_present', False) and n.leaf_reward > 0
        ]

        if not anomalous:
            return []

        # 选 leaf_reward 最高的节点（而非 visits 最多），因为 reward 已融合 clip+边界信号
        best = max(anomalous, key=lambda n: n.leaf_reward)

        # expert 的 CLIP 分类结果比初始文本描述更准确，优先使用
        anomaly_type = self.anomaly_desc
        if best.anomaly_info and best.anomaly_info.get('predicted_type'):
            anomaly_type = best.anomaly_info['predicted_type']

        return [{
            't_start': best.state['t_start'],
            't_end': best.state['t_end'],
            'confidence': best.leaf_reward,
            'anomaly_type': anomaly_type,
            'depth': best.state['depth'],
            'visits': best.visits,
        }]

    def serialize_tree(self, node=None):
        """
        将搜索树序列化为嵌套字典（用于调试和可视化）

        递归遍历整棵树，每个节点包含 state 摘要、统计量和子节点。
        可以用 json.dumps(mcts.serialize_tree(), indent=2) 输出到文件分析搜索过程。
        """
        if node is None:
            node = self.root
        return {
            'state': {
                't_start': node.state['t_start'],
                't_end': node.state['t_end'],
                'depth': node.state['depth'],
                'action_history': node.state['action_history'],
                'temporal_ratio': node.state['temporal_ratio'],
                'anomaly_present': node.state.get('anomaly_present'),
            },
            'visits': node.visits,
            'value': node.value,
            'leaf_reward': node.leaf_reward,
            'temporal_ratio': node.temporal_ratio,
            'children': {
                action: self.serialize_tree(child)
                for action, child in node.children.items()
            },
        }
