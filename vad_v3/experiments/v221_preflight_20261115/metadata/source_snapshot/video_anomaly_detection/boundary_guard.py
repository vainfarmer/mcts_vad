"""后处理边界保护：只检查已接受MCTS来源，违约时恢复该阶段之前的完整检测。"""
import copy


class BoundaryGuard:
    def __init__(self, seeds, protections, config):
        self.cfg = config
        self.protections = {int(k): v for k, v in protections.items()}
        self.seeds = {i: (a['t_start'], a['t_end']) for i, a in enumerate(seeds)}
        self.previous = {}
        self.remember(seeds)

    @staticmethod
    def key(item):
        return tuple(sorted(item.get('source_candidate_ids', [])))

    def remember(self, items):
        self.previous = {self.key(a): copy.deepcopy(a) for a in items if self.key(a)}

    def apply(self, items, stage):
        events = []
        for i, item in enumerate(items):
            ids = self.key(item)
            protected = [self.protections[k] for k in ids if k in self.protections]
            if not protected:
                continue
            previous = self.previous.get(ids)
            if previous is None:
                events.append(dict(stage=stage, source_candidate_ids=list(ids), reason='missing_previous_identity', reverted=False))
                continue
            s, e = item['t_start'], item['t_end']
            tol = self.cfg.guard_tolerance_seconds
            reasons = []
            if not s < e:
                reasons.append('invalid_interval')
            # 多来源合并时保留每个已保护事件；外扩上限按所有来源的包络计算，
            # 不能拿单个MCTS区间限制整个合并结果，截掉未搜索的另一个事件。
            references = [self.protections[k]['accepted_interval'] if k in self.protections
                          else self.seeds[k] for k in ids if k in self.seeds or k in self.protections]
            lo, hi = min(x[0] for x in references), max(x[1] for x in references)
            limit = max(tol, (hi-lo)*self.cfg.guard_max_shift_ratio)
            if abs(s-lo)>limit or abs(e-hi)>limit:
                reasons.append('endpoint_shift')
            bounds = [p['evidence_bounds'] for p in protected] + [self.seeds[k] for k in ids if k not in self.protections and k in self.seeds]
            if s < min(x[0] for x in bounds)-tol or e > max(x[1] for x in bounds)+tol:
                reasons.append('outside_evidence')
            for p in protected:
                a,b = p['accepted_interval']
                if max(0.,min(e,b)-max(s,a))/(b-a) < self.cfg.guard_min_retained_ratio:
                    reasons.append('insufficient_retained_interval')
                if not s-tol <= p['anchor_timestamp'] <= e+tol:
                    reasons.append('lost_curve_anchor')
            if reasons:
                items[i] = copy.deepcopy(previous)
                events.append(dict(stage=stage, source_candidate_ids=list(ids), reasons=sorted(set(reasons)),
                                   proposed=[s,e], restored=[previous['t_start'],previous['t_end']], reverted=True))
        return events
