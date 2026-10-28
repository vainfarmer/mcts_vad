"""Time-weighted interval scoring and conservative, timestamped visual evidence.

No model/network calls. Evidence is selected before a request, never silently truncated.
"""
from dataclasses import dataclass
import json
import math
import numpy as np


class EvidenceError(ValueError):
    pass


@dataclass
class Evidence:
    frames: np.ndarray
    indices: list
    timestamps: list
    roles: list
    coverage: dict


# 返回 a 中不属于 b 的区间，最多两段；统一使用半开区间 [start,end)。
def interval_difference(a, b):
    parts = []
    if a[0] < min(a[1], b[0]):
        parts.append((a[0], min(a[1], b[0])))
    if max(a[0], b[1]) < a[1]:
        parts.append((max(a[0], b[1]), a[1]))
    return parts


class BoundaryEvidenceScorer:
    def __init__(self, video_data, curve, config, expected_duration=(3., 30.)):
        self.data, self.cfg = video_data, config
        self.times = np.asarray(video_data['timestamps'], dtype=float)
        self.curve = np.asarray(curve, dtype=float)
        self.duration = float(video_data['duration'])
        self.expected = expected_duration
        if (len(self.times) < 2 or self.curve.shape != self.times.shape
                or len(video_data['all_frames']) != len(self.times)
                or not np.isfinite(self.times).all() or not np.isfinite(self.curve).all()
                or np.any(np.diff(self.times) <= 0)
                or np.any(self.times < 0) or np.any(self.times > self.duration)
                or np.any(self.curve < 0) or np.any(self.curve > 1)):
            raise EvidenceError('invalid_curve_or_timestamps')

    def validate(self, interval):
        s, e = interval
        if not (math.isfinite(s) and math.isfinite(e) and 0 <= s < e <= self.duration):
            raise EvidenceError('invalid_interval')

    # 对真实时间轴积分求平均，避免密集帧比稀疏帧获得更高权重。
    def mean(self, s, e):
        if e <= s:
            raise EvidenceError('empty_curve_region')
        # Only short video-edge extrapolation is allowed, never long missing spans.
        if s < self.times[0] and (s != 0 or self.times[0] > self.cfg.max_boundary_gap_seconds):
            raise EvidenceError('missing_curve_start')
        if e > self.times[-1] and (e != self.duration or e-self.times[-1] > self.cfg.max_boundary_gap_seconds):
            raise EvidenceError('missing_curve_end')
        lo = max(0, np.searchsorted(self.times, s, side='right') - 1)
        hi = min(len(self.times)-1, np.searchsorted(self.times, e))
        if np.any(np.diff(self.times[lo:hi+1]) > self.cfg.max_evidence_gap_seconds):
            raise EvidenceError('curve_gap')
        points = np.r_[s, self.times[(self.times > s) & (self.times < e)], e]
        return float(np.trapezoid(np.interp(points, self.times, self.curve), points)/(e-s))

    # contrast_raw=区间内均值-左右背景均值，用于0.15触发阈值。
    # quality=(contrast_raw+1)/2 用于奖励，二者尺度不同，不要混用。
    def quality(self, interval):
        self.validate(interval)
        s, e = interval
        inside = self.mean(s, e)
        outside = []
        if s > 0:
            outside.append(self.mean(max(0, s-self.cfg.context_seconds), s))
        if e < self.duration:
            outside.append(self.mean(e, min(self.duration, e+self.cfg.context_seconds)))
        if not outside:
            raise EvidenceError('no_external_curve')
        contrast = inside - float(np.mean(outside))
        return {'contrast_raw': contrast, 'quality': float(np.clip((contrast+1)/2, 0, 1))}

    # 固定原区间首帧/末帧/峰值帧为核心参考，防止搜索转移到另一个事件。
    def core_indices(self, seed):
        ids = np.flatnonzero((self.times >= seed[0]) & (self.times < seed[1]))
        if not len(ids):
            raise EvidenceError('empty_seed')
        return sorted(set(int(i) for i in (ids[0], ids[-1], ids[np.argmax(self.curve[ids])])) )

    # 图片按时间排序并去重，同一图可承担多个角色；索引随后在请求内重新编号。
    # 要同时覆盖核心、端点内外、区间内部以及相对seed新增/删除的区域。
    def evidence(self, seed, intervals, core, max_images):
        chosen = {int(i): {'core_reference'} for i in core}
        coverage = {}

        def add(i, role):
            chosen.setdefault(int(i), set()).add(role)

        def cover(s, e, role):
            if e <= s:
                return
            ids = np.flatnonzero((self.times >= s) & (self.times < e))
            if not len(ids):
                raise EvidenceError('empty_' + role)
            # Deterministic temporal cover; selected gaps including endpoints are bounded.
            used = [int(ids[0])]
            for idx in ids[1:]:
                if self.times[idx] - self.times[used[-1]] >= self.cfg.max_evidence_gap_seconds / 2:
                    used.append(int(idx))
            used.append(int(ids[-1]))
            used = sorted(set(used))
            gap = float(np.max(np.diff(np.r_[s, self.times[used], e])))
            if gap > self.cfg.max_evidence_gap_seconds:
                raise EvidenceError('coverage_gap_' + role)
            for i in used:
                add(i, role)
            coverage[role] = {'start': float(s), 'end': float(e), 'max_gap': gap}

        for n, (s, e) in enumerate(intervals):
            self.validate((s, e))
            # At an exact end timestamp, that frame is outside [s,e).
            masks = (self.times < s, (self.times >= s) & (self.times < e),
                     (self.times >= s) & (self.times < e), self.times >= e)
            for side, mask, target, first in zip(
                    ('left_out', 'left_in', 'right_in', 'right_out'), masks,
                    (s, s, e, e), (False, True, False, True)):
                if (side == 'left_out' and s == 0) or (side == 'right_out' and e == self.duration):
                    continue
                ids = np.flatnonzero(mask)
                if not len(ids):
                    raise EvidenceError('missing_' + side)
                i = int(ids[0] if first else ids[-1])
                if abs(self.times[i]-target) > self.cfg.max_boundary_gap_seconds:
                    raise EvidenceError('boundary_gap_' + side)
                add(i, f'{n}_{side}')
            cover(s, e, f'{n}_inside')
            for j, part in enumerate(interval_difference(seed, (s, e))):
                cover(*part, f'{n}_removed_{j}')
            for j, part in enumerate(interval_difference((s, e), seed)):
                cover(*part, f'{n}_added_{j}')
        ids = sorted(chosen)
        # 不静默截断图片：截断会使已通过的覆盖检查失效。长区间可能因此直接回退。
        if len(ids) > max_images:
            raise EvidenceError('image_budget_insufficient')
        return Evidence(self.data['all_frames'][ids], ids,
                        [float(self.times[i]) for i in ids],
                        [sorted(chosen[i]) for i in ids], coverage)

    # P目标置信度/B边界语义/E完整性/C曲线质量/F时长先验；奖励只用于搜索排序。
    def reward(self, interval, answer):
        if not answer['target_present_inside']:
            return 0., {'present': 0.}
        if 'unknown' in (answer['start_status'], answer['end_status'], answer['event_completeness']):
            raise EvidenceError('unknown_semantic_evidence')
        if not answer['same_event_as_reference']:
            raise EvidenceError('event_identity_mismatch')
        levels = {'well_bounded': 1., 'excess_background': .5, 'event_cut': 0.}
        b = (levels[answer['start_status']]+levels[answer['end_status']])/2
        c = self.quality(interval)['quality']
        d = interval[1]-interval[0]
        low, high = self.expected
        f = min(1., d/low, high/d)
        parts = {'P': answer['target_confidence'], 'B': b,
                 'E': float(answer['event_completeness'] == 'complete'), 'C': c, 'F': f}
        return sum(parts[k]*w for k, w in zip('PBECF', (.15,.35,.25,.20,.05))), parts


# 严格校验JSON、布尔类型、枚举和图像索引；不把字符串'true'自动当成真。
def parse_answer(text, image_count, pair=False):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise EvidenceError('invalid_json_object')
    if pair:
        if obj.get('winner') not in ('A', 'B', 'tie', 'unknown'):
            raise EvidenceError('invalid_winner')
        for key in ('same_event', 'complete_without_loss', 'removed_contains_target', 'added_same_event'):
            if type(obj.get(key)) is not bool:
                raise EvidenceError('invalid_' + key)
    else:
        for key in ('target_present_inside', 'same_event_as_reference'):
            if type(obj.get(key)) is not bool:
                raise EvidenceError('invalid_' + key)
        p = obj.get('target_confidence')
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise EvidenceError('invalid_confidence')
        for key in ('start_status', 'end_status'):
            if obj.get(key) not in ('well_bounded', 'excess_background', 'event_cut', 'unknown'):
                raise EvidenceError('invalid_' + key)
        if obj.get('event_completeness') not in ('complete', 'partial', 'unknown'):
            raise EvidenceError('invalid_completeness')
    ids = obj.get('evidence_image_indices')
    if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < image_count for i in ids):
        raise EvidenceError('invalid_evidence_indices')
    if pair:
        for key in ('removed_evidence_indices', 'added_evidence_indices'):
            ids = obj.get(key)
            if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < image_count for i in ids):
                raise EvidenceError('invalid_' + key)
    return obj
