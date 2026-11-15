"""
后处理模块 (post_processor)

从 pipeline 抽离的纯函数集合 — 异常段的合并/统一/收窄，不依赖 CLIP/LLM：
  - global_merge         : 按 IoU + 同类间隔合并重叠/相邻异常段
  - unify_anomaly_types  : 多个异常统一为加权置信度最高的类别
  - narrow_short_events  : 用运动能量峰值收窄过宽的短事件预测

纯函数（输入 dict 列表 -> 输出 dict 列表），易单测，pipeline 直接调用。
"""

from __future__ import annotations
from bisect import bisect_left
from collections import Counter
from typing import Dict, List

import numpy as np

try:
    import vad_logger as _vl
    _log = _vl.get("vad.postproc")
except Exception:
    import logging
    _log = logging.getLogger("vad.postproc")


def global_merge(anomalies: List[Dict], iou_threshold: float = 0.3,
                 same_type_gap: float = 2.0) -> List[Dict]:
    """合并重叠/相邻的异常段。

    同类段：有重叠或间隔 <= same_type_gap 即合并（保留更高 confidence 者的追溯字段）。
    异类段：仅当 IoU > iou_threshold 时合并（保留更高 confidence 者的类别）。
    迭代直到无变化（一次合并可能产生新的相邻关系）。
    """
    if not anomalies:
        return []
    merged = sorted([a.copy() for a in anomalies], key=lambda a: a['t_start'])
    changed = True
    while changed:
        changed = False
        new_merged = [merged[0]]
        for a in merged[1:]:
            did_merge = False
            for i, existing in enumerate(new_merged):
                overlap_start = max(existing['t_start'], a['t_start'])
                overlap_end = min(existing['t_end'], a['t_end'])
                overlap = max(0, overlap_end - overlap_start)
                gap = a['t_start'] - existing['t_end']
                same_type = (existing.get('anomaly_type') == a.get('anomaly_type'))
                if same_type and (overlap > 0 or gap <= same_type_gap):
                    keep = existing if existing['confidence'] >= a['confidence'] else a
                    new_merged[i] = {
                        't_start': min(existing['t_start'], a['t_start']),
                        't_end': max(existing['t_end'], a['t_end']),
                        'confidence': max(existing['confidence'], a['confidence']),
                        'coarse_score': max(existing.get('coarse_score', 0.8),
                                            a.get('coarse_score', 0.8)),  # v51
                        'anomaly_type': existing.get('anomaly_type'),
                        'depth': keep.get('depth'), 'visits': keep.get('visits'),
                    }
                    if 'source_candidate_ids' in existing or 'source_candidate_ids' in a:
                        new_merged[i]['source_candidate_ids'] = sorted(set(existing.get('source_candidate_ids', [])) | set(a.get('source_candidate_ids', [])))
                    did_merge = True; changed = True; break
                elif not same_type and overlap > 0:
                    union = max(existing['t_end'], a['t_end']) - min(existing['t_start'], a['t_start'])
                    if overlap / union > iou_threshold:
                        keep = existing if existing['confidence'] >= a['confidence'] else a
                        new_merged[i] = {
                            't_start': min(existing['t_start'], a['t_start']),
                            't_end': max(existing['t_end'], a['t_end']),
                            'confidence': max(existing['confidence'], a['confidence']),
                            'coarse_score': max(existing.get('coarse_score', 0.8),
                                                a.get('coarse_score', 0.8)),  # v51
                            'anomaly_type': keep.get('anomaly_type'),
                            'depth': keep.get('depth'), 'visits': keep.get('visits'),
                        }
                        if 'source_candidate_ids' in existing or 'source_candidate_ids' in a:
                            new_merged[i]['source_candidate_ids'] = sorted(set(existing.get('source_candidate_ids', [])) | set(a.get('source_candidate_ids', [])))
                        did_merge = True; changed = True; break
            if not did_merge:
                new_merged.append(a.copy())
        merged = sorted(new_merged, key=lambda a: a['t_start'])
    return merged


def unify_anomaly_types(anomalies: List[Dict]) -> List[Dict]:
    """多异常段统一为单一类别（置信度×时长加权投票的胜者）。"""
    if len(anomalies) <= 1:
        return anomalies
    types = set(a['anomaly_type'] for a in anomalies)
    if len(types) == 1:
        return anomalies
    type_weight = Counter()
    for a in anomalies:
        duration = a['t_end'] - a['t_start']
        type_weight[a['anomaly_type']] += a['confidence'] * duration
    best_type = type_weight.most_common(1)[0][0]
    _log.info(f"    [Unify] types={dict(type_weight)}, winner={best_type}")
    unified = []
    for a in anomalies:
        a = a.copy()
        a['anomaly_type'] = best_type
        unified.append(a)
    return unified


def narrow_short_events(anomalies: List[Dict], video_data) -> List[Dict]:
    """用运动能量峰值收窄过宽的短事件预测（Explosion/RoadAccidents 等）。

    v51 changes:
    - Increased skip threshold from 10s to 15s (don't narrow already-short segments)
    - Use wider, duration-proportional window instead of fixed +/-5s
    - Added 60% floor: never shrink a segment below 60% of its original duration
    """
    motion = video_data.get('motion_energy')
    timestamps = video_data['timestamps']
    duration = video_data['duration']
    if motion is None or len(motion) < 4:
        return anomalies
    narrowed = []
    for a in anomalies:
        a = a.copy()
        seg_dur = a['t_end'] - a['t_start']
        if seg_dur <= 15.0:  # v51: was 10.0 — don't narrow already-short segments
            narrowed.append(a)
            continue
        lo = bisect_left(timestamps, a['t_start'])
        hi = bisect_left(timestamps, a['t_end'])
        if hi <= lo + 2:
            narrowed.append(a)
            continue
        seg_motion = motion[lo:hi]
        peak_local = np.argmax(seg_motion)
        peak_global = peak_local + lo
        peak_val = motion[peak_global]
        if peak_val > np.mean(motion) + np.std(motion):
            peak_time = timestamps[min(peak_global, len(timestamps) - 1)]
            # v51: wider, proportional window instead of fixed +/-5s
            half_window = max(5.0, min(seg_dur * 0.4, 10.0))
            new_start = max(0, peak_time - half_window)
            new_end = min(duration, peak_time + half_window)
            # v51: 60% floor — never shrink below 60% of original duration
            new_dur = new_end - new_start
            if new_dur < seg_dur * 0.6:
                expand = (seg_dur * 0.6 - new_dur) / 2
                new_start = max(0, new_start - expand)
                new_end = min(duration, new_end + expand)
            if new_end - new_start >= 3.0:
                _log.info(f"    [Short-narrow] {a['t_start']:.1f}-{a['t_end']:.1f}s "
                          f"-> {new_start:.1f}-{new_end:.1f}s (peak@{peak_time:.1f}s)")
                a['t_start'] = new_start
                a['t_end'] = new_end
        narrowed.append(a)
    return narrowed



def temporal_nms(anomalies: List[Dict], iou_threshold: float = 0.3, audit=None) -> List[Dict]:
    """IoU-based Non-Maximum Suppression for temporal segments."""
    if len(anomalies) <= 1:
        return anomalies
    sorted_anoms = sorted(anomalies, key=lambda a: a.get('confidence', 0), reverse=True)
    keep = []
    while sorted_anoms:
        best = sorted_anoms.pop(0)
        keep.append(best)
        remaining = []
        for a in sorted_anoms:
            overlap_start = max(best['t_start'], a['t_start'])
            overlap_end = min(best['t_end'], a['t_end'])
            overlap = max(0, overlap_end - overlap_start)
            union = (max(best['t_end'], a['t_end']) - min(best['t_start'], a['t_start']))
            iou = overlap / union if union > 0 else 0
            if iou < iou_threshold:
                remaining.append(a)
            elif audit is not None:
                audit.append({'winner': best.get('source_candidate_ids', []),
                              'suppressed': a.get('source_candidate_ids', []), 'iou': iou})
        sorted_anoms = remaining
    return keep


__all__ = ["global_merge", "unify_anomaly_types", "narrow_short_events", "temporal_nms"]
