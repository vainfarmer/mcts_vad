"""Budgeted local tree search; legacy TemporalBoundaryMCTS.search stays unchanged."""
from dataclasses import dataclass, field
import math
import random
import numpy as np


# 树节点表示一条搜索路径上的区间；相同区间可从不同历史到达，不能随意合并节点。
@dataclass
class Node:
    interval: tuple
    depth: int = 0
    parent: object = None
    children: list = field(default_factory=list)
    pending: object = None
    visits: int = 0
    value: float = 0.
    evaluation: object = None
    exhausted: bool = False


class LocalBoundarySearch:
    def __init__(self, seed, scorer, config, candidate_id=0):
        self.seed, self.scorer, self.cfg = tuple(seed), scorer, config
        self.rng = random.Random(config.random_seed + candidate_id)
        self.root = Node(self.seed)
        self.nodes = [self.root]
        self.records = []
        self.stop_reason = 'simulation_limit'
        self.d = seed[1]-seed[0]
        self.bounds = (max(0., seed[0]-config.search_padding_ratio*self.d),
                       min(scorer.duration, seed[1]+config.search_padding_ratio*self.d))
        # 端点对齐可用时间戳及种子/范围边界，不能假定运动过滤后的帧等间隔。
        self.grid = sorted(set([*self.seed, *self.bounds] + [float(t) for t in scorer.times if self.bounds[0] <= t <= self.bounds[1]]))

    # 位移限制始终相对原始 seed，避免多层小步累积越界。
    def legal(self, x):
        s,e=x; a,b=self.seed; c=self.cfg
        return (self.bounds[0] <= s < e <= self.bounds[1]
                and e-s >= c.min_interval_seconds
                and a-c.max_outward_shift_ratio*self.d-1e-9 <= s <= a+c.max_inward_shift_ratio*self.d+1e-9
                and b-c.max_inward_shift_ratio*self.d-1e-9 <= e <= b+c.max_outward_shift_ratio*self.d+1e-9)

    # 默认 shrink 只收缩左右端点；shrink_shift 额外允许等时长平移，
    # 不是任意扩张动作。步长不小于附近采样分辨率，且禁止返回祖先形成循环。
    def successors(self, node):
        if node.depth >= self.cfg.max_depth:
            return []
        s,e=node.interval
        def resolution(t):
            i=int(np.searchsorted(self.scorer.times,t))
            ds=np.diff(self.scorer.times[max(0,i-2):min(len(self.scorer.times),i+3)])
            return float(np.median(ds)) if len(ds) else float('inf')
        dl=max(resolution(s),self.cfg.step_ratio*self.d/(1+.3*node.depth))
        dr=max(resolution(e),self.cfg.step_ratio*self.d/(1+.3*node.depth))
        choices=[]
        for action, ideal, values in (
                ('shrink_left', s+dl, [(t,e) for t in self.grid if t>s]),
                ('shrink_right', e-dr, [(s,t) for t in self.grid if t<e])):
            valid=[x for x in values if self.legal(x)]
            if valid:
                axis=0 if action=='shrink_left' else 1
                choices.append(min(valid,key=lambda x:(abs(x[axis]-ideal),abs(x[axis]-node.interval[axis]))))
        # 可选的外扩动作给“原区间已经截断事件”的候选提供补全机会。
        # 与收缩共享相对seed位移、背景覆盖、最终完整性检查，不允许无限外扩。
        if self.cfg.action_set=='shrink_expand_shift':
            for axis,ideal,values in (
                    (0,s-dl,[(t,e) for t in self.grid if t<s]),
                    (1,e+dr,[(s,t) for t in self.grid if t>e])):
                valid=[x for x in values if self.legal(x)]
                if valid:
                    choices.append(min(valid,key=lambda x:(abs(x[axis]-ideal),abs(x[axis]-node.interval[axis]))))
        if self.cfg.action_set in ('shrink_shift','shrink_expand_shift'):
            grid_lookup={round(t,9):t for t in self.grid}
            for direction in (-1,1):
                valid=[]
                for ns in self.grid:
                    shift=ns-s
                    ne=grid_lookup.get(round(e+shift,9))
                    if shift*direction>0 and ne is not None and abs((ne-ns)-(e-s))<1e-8 and self.legal((ns,ne)):
                        valid.append((ns,ne))
                if valid:
                    ideal=direction*max(dl,dr)
                    choices.append(min(valid,key=lambda x:(abs(x[0]-s-ideal),abs(x[0]-s))))
        ancestors=set(); p=node
        while p is not None:
            ancestors.add(p.interval); p=p.parent
        return list(dict.fromkeys(x for x in choices if x not in ancestors))

    def ordered(self, node):
        def quality(x):
            try: return self.scorer.quality(x)['quality']
            except ValueError: return -float('inf')
        return sorted(self.successors(node),key=lambda x:(-quality(x),x))

    def select(self, node, iteration):
        if node.pending is None:
            node.pending=self.ordered(node)
        if node.pending:
            return node
        while True:
            available=[c for c in node.children if not c.exhausted]
            if not available:
                node.exhausted=True
                return None
            # UCB = 平均奖励 + 探索项；无访问子节点优先探索。衰减默认0，即不衰减。
            coefficient=self.cfg.ucb_c*math.exp(-self.cfg.ucb_decay*iteration)
            def ucb(c):
                return (float('inf') if not c.visits else c.value/c.visits + coefficient*math.sqrt(2*math.log(node.visits+1)/c.visits))
            found=self.select(max(available,key=ucb),iteration)
            if found is not None:
                return found

    def backprop(self, node, evaluation):
        node.evaluation=evaluation
        # 无效证据/API失败不当作零分回传，否则会把技术失败误判成差边界。
        if evaluation is None:
            return
        while node is not None:
            node.visits+=1; node.value+=evaluation['reward']; node=node.parent

    def enumerate_nodes(self):
        pending=[Node(self.seed)]; found=[]
        while pending:
            n=pending.pop(0)
            # Path history also affects ancestor-cycle checks. Enumerate complete paths;
            # only the final uniform draw deduplicates intervals, not expansion states.
            found.append(n)
            if len(found)>self.cfg.max_enumerated_states:
                raise ValueError('enumeration_limit')
            pending.extend(Node(x,n.depth+1,n) for x in self.successors(n))
        return found

    # 三种策略共用外部 evaluate 和请求预算：mcts 用UCB，greedy逐层选最好，
    # uniform在可达区间中固定种子抽样。simulations不是HTTP请求次数。
    async def run(self, evaluate, can_evaluate):
        self.backprop(self.root,await evaluate(self.root))
        if self.root.evaluation is None:
            self.stop_reason='invalid_root'; return self
        seen={self.seed}; best=self.root.evaluation['reward']; stale=0
        uniform=[]; greedy=[self.root]; round_nodes=[]
        if self.cfg.search_strategy=='uniform':
            try:
                reachable=self.enumerate_nodes()
            except ValueError:
                self.stop_reason='enumeration_limit'; return self
            by_interval={}
            for n in sorted(reachable,key=lambda n:n.depth):
                if n.depth: by_interval.setdefault(n.interval,n)
            uniform=list(by_interval.values()); self.rng.shuffle(uniform)
        for iteration in range(self.cfg.simulations):
            if not can_evaluate():
                self.stop_reason='request_budget'; break
            if self.cfg.search_strategy=='uniform':
                if not uniform: self.stop_reason='tree_exhausted'; break
                node=uniform.pop()
            elif self.cfg.search_strategy=='greedy':
                parent=greedy[0]
                if parent.pending is None: parent.pending=self.ordered(parent)
                if not parent.pending:
                    valid=[n for n in round_nodes if n.evaluation is not None]
                    if not valid: self.stop_reason='tree_exhausted'; break
                    greedy=[max(valid,key=lambda n:n.evaluation['reward'])]; round_nodes=[]
                    parent=greedy[0]; parent.pending=self.ordered(parent)
                    if not parent.pending: self.stop_reason='tree_exhausted'; break
                node=Node(parent.pending.pop(0),parent.depth+1,parent)
                parent.children.append(node); round_nodes.append(node)
            else:
                parent=self.select(self.root,iteration)
                if parent is None: self.stop_reason='tree_exhausted'; break
                node=Node(parent.pending.pop(0),parent.depth+1,parent)
                parent.children.append(node)
            self.nodes.append(node)
            self.backprop(node,await evaluate(node))
            if node.interval not in seen and node.evaluation is not None:
                seen.add(node.interval)
                reward=node.evaluation['reward']
                if reward>=best+self.cfg.min_reward_gain:
                    best=reward; stale=0
                else: stale+=1
                # 早停先要求足够的不同有效区间和根动作覆盖，避免重复缓存误触发收敛。
                root_explored=set(self.successors(self.root)) <= seen
                if len(seen)>=self.cfg.min_unique_evaluations and root_explored and stale>=self.cfg.patience:
                    self.stop_reason='patience'; break
        return self
