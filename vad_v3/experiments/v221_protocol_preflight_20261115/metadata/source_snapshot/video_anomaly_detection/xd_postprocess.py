"""XD-only, class-aware tail ablations on immutable shared MCTS proposals.

These are experimental adaptations, NOT the UCF GT-duration-calibrated tail.
No labels, annotations, output confidence overlays or per-video min/max scaling
are used as boundary evidence. Geometry finishes before Q-score is computed.
"""
from dataclasses import dataclass, asdict
import copy
import numpy as np

try:
    from .xd_config import XD_CLASSES
    from .post_processor import temporal_nms
except ImportError:
    from xd_config import XD_CLASSES
    from post_processor import temporal_nms

ARMS = {
    'mcts': (),
    'mcts_qscore': ('qscore',),
    'mcts_qscore_cbt': ('cbt', 'qscore'),
    'mcts_qscore_cbt_bue': ('cbt', 'bue', 'qscore'),
    'mcts_cbt_bue': ('cbt', 'bue'),
}


@dataclass(frozen=True)
class XDTailConfig:
    # Frozen before evaluation; no category-specific GT duration caps.
    min_curve_contrast: float = .05
    context_seconds: float = 2.
    cbt_max_shrink_ratio: float = .25
    bue_max_expand_ratio: float = .5
    bue_max_expand_seconds: float = 8.
    min_interval_seconds: float = .5
    qscore_base: float = .5
    qscore_contrast_scale: float = .25
    nms_iou: float = .5


def interval(row):
    return [float(row['t_start']), float(row['t_end'])]


def curve_mean(times, curve, start, end):
    """Time integral, with only endpoint extrapolation to the source duration."""
    if end <= start:
        return None
    points = np.r_[start, times[(times > start) & (times < end)], end]
    return float(np.trapezoid(np.interp(points, times, curve), points)/(end-start))


def boundary_operation(event, times, curve, duration, operation, cfg):
    row = copy.deepcopy(event)
    start, end = interval(row)
    audit = dict(operator=operation, source_candidate_ids=row['source_candidate_ids'],
                 before=[start, end], after=[start, end], applied=False)
    length = end-start
    core = np.flatnonzero((times >= start) & (times < end))
    radius = min(cfg.bue_max_expand_seconds, length*cfg.bue_max_expand_ratio)
    lo, hi = max(0., start-radius), min(duration, end+radius)
    # Estimate background beyond the allowed expansion region as well. Using
    # only that region makes a truncated, high-support plateau look "flat" and
    # prevents BUE from reaching its actual falling edge.
    local = np.flatnonzero((times >= max(0., lo-cfg.context_seconds)) &
                           (times < min(duration, hi+cfg.context_seconds)))
    if len(core) < 2 or len(local) < 3:
        audit['reason'] = 'insufficient_samples'
        return row, audit
    # Half-height relative to a LOCAL background, not a video min/max range.
    peak = float(np.max(curve[core]))
    background = float(np.percentile(curve[local], 20))
    if peak-background < cfg.min_curve_contrast:
        audit['reason'] = 'flat_or_weak_class_curve'
        return row, audit
    threshold = background+.5*(peak-background)
    audit.update(threshold=threshold, peak=peak, background=background)
    step = float(np.median(np.diff(times)))
    new_start, new_end = start, end
    if operation == 'cbt':
        # Trim low-support edges only. Preserve ALL internal high-support
        # islands; selecting only the tallest peak would cut a multi-part event.
        supported = core[curve[core] >= threshold]
        new_start = max(start, min(float(times[supported[0]]), start+length*cfg.cbt_max_shrink_ratio))
        new_end = min(end, max(float(times[supported[-1]]+step), end-length*cfg.cbt_max_shrink_ratio))
        if curve_mean(times, curve, new_start, new_end) <= curve_mean(times, curve, start, end)+1e-6:
            new_start, new_end = start, end
    elif operation == 'bue':
        # Union with continuous class support at each existing boundary. Stop
        # at the first low-score sample; do not jump a gap to a different event.
        left, right = int(core[0]), int(core[-1])
        if curve[left] >= threshold:
            while left > 0 and times[left-1] >= lo and curve[left-1] >= threshold:
                left -= 1
            new_start = min(start, max(lo, float(times[left])))
        if curve[right] >= threshold:
            while right+1 < len(times) and times[right+1] < hi and curve[right+1] >= threshold:
                right += 1
            new_end = max(end, min(hi, float(times[right]+step)))
    else:
        raise ValueError('Unknown boundary operation: '+operation)
    if new_end-new_start < cfg.min_interval_seconds:
        audit['reason'] = 'minimum_duration'
        return row, audit
    row.update(t_start=new_start, t_end=new_end)
    audit.update(after=interval(row), applied=interval(row) != [start, end],
                 reason='changed' if interval(row) != [start, end] else 'already_supported_boundary')
    return row, audit


def quality_score(event, times, curve, duration, cfg):
    """Fixed [0,1] quality scale, comparable across videos and class curves.

    Q = mean(class support, positive inner/outer contrast, boundary contrast).
    It reorders confidence only; it is neither a probability nor a class gate.
    """
    start, end = interval(event)
    inside = curve_mean(times, curve, start, end)
    context = min(cfg.context_seconds, (end-start)/2)
    outside, edges = [], []
    for out_start, out_end, in_start, in_end in (
            (max(0., start-context), start, start, min(end, start+context)),
            (end, min(duration, end+context), max(start, end-context), end)):
        out = curve_mean(times, curve, out_start, out_end)
        if out is not None:
            outside.append(out)
            edges.append(max(0., curve_mean(times, curve, in_start, in_end)-out))
    contrast = max(0., inside-float(np.mean(outside))) if outside else 0.
    parts = dict(support=float(np.clip(inside, 0, 1)),
                 contrast=float(np.clip(contrast/cfg.qscore_contrast_scale, 0, 1)),
                 boundary=float(np.clip(np.mean(edges)/cfg.qscore_contrast_scale, 0, 1)) if edges else 0.)
    q = float(np.mean(list(parts.values())))
    return q, parts


def merge_adjacent(events):
    """Same adjacency rule as XD baseline, retaining provenance through unions."""
    merged = []
    for category in XD_CLASSES:
        rows = sorted((copy.deepcopy(e) for e in events if e['anomaly_type'] == category),
                      key=lambda e: (e['t_start'], e['source_candidate_ids']))
        group = []
        for row in rows:
            if group:
                prev = group[-1]
                gap = row['t_start']-prev['t_end']
                shorter = min(prev['t_end']-prev['t_start'], row['t_end']-row['t_start'])
                # The reviewer already separated independent events and merged
                # proven fragments. Do not undo that decision by proximity alone.
                reviewed = 'verification' in prev or 'verification' in row
                if not reviewed and -.25*shorter <= gap <= .5 and max(0., -gap) < .25*shorter:
                    prev['t_end'] = max(prev['t_end'], row['t_end'])
                    prev['confidence'] = max(prev['confidence'], row['confidence'])
                    prev['source_candidate_ids'] = sorted(set(prev['source_candidate_ids']+row['source_candidate_ids']))
                    prev['description'] = '; '.join(filter(None, [prev.get('description'), row.get('description')]))
                    continue
            group.append(row)
        merged.extend(group)
    return merged


def make_arms(events, times, curves, duration, cfg=None):
    """Small event dictionaries are copied; frames/features are shared read-only."""
    cfg = cfg or XDTailConfig()
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid tail timestamps')
    curves = {c: np.asarray(curves[c], dtype=float) for c in XD_CLASSES}
    for c in XD_CLASSES:
        if curves[c].shape != times.shape or not np.isfinite(curves[c]).all():
            raise ValueError('Invalid class curve: '+c)
    arms = {}
    for name, operators in ARMS.items():
        rows, audit = copy.deepcopy(events), []
        for operator in operators:
            if operator == 'qscore':
                continue  # Geometry/adjacency merge must finish first.
            changed = []
            for event in rows:
                row, record = boundary_operation(event, times, curves[event['anomaly_type']], duration, operator, cfg)
                changed.append(row); audit.append(record)
            rows = changed
        count_before_merge = len(rows)
        rows = merge_adjacent(rows)
        if 'qscore' in operators:
            for row in rows:
                original = row['confidence']
                q, parts = quality_score(row, times, curves[row['anomaly_type']], duration, cfg)
                row['confidence'] = float(original*(cfg.qscore_base+(1-cfg.qscore_base)*q))
                row['qscore'] = dict(value=q, components=parts, confidence_before=original,
                                     evaluated_interval=interval(row))
                audit.append(dict(operator='qscore', source_candidate_ids=row['source_candidate_ids'],
                                  before=original, after=row['confidence'], applied=row['confidence'] != original))
        final, suppression = [], []
        for category in XD_CLASSES:
            final.extend(temporal_nms([r for r in rows if r['anomaly_type'] == category], cfg.nms_iou, suppression))
        final.sort(key=lambda e: (e['t_start'], e['anomaly_type'], e['source_candidate_ids']))
        arms[name] = dict(anomalies=final, is_normal=not final,
                          operation_order=[o for o in operators if o != 'qscore']+['adjacent_merge']+
                              (['qscore'] if 'qscore' in operators else [])+['classwise_nms'],
                          audit=audit, merged_count=count_before_merge-len(rows), nms_suppression=suppression,
                          tail_config=asdict(cfg))
    return arms
