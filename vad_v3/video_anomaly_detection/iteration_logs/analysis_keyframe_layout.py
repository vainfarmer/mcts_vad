"""Keyframe-layout comparison on cached features: which per-segment frame
layout maximizes cue coverage for the VLM comparative scan?

Layouts compared per segment (k=2 unless noted):
  A first2      : frames[0], frames[1]              (current FCCS behavior)
  B endpoints   : frames[0], frames[-1]
  C peak2       : top-2 motion frames (greedy, +-2 exclusion)
  D end+peak    : frames[0] + top-1 motion frame (if distinct, else frames[-1])
  E mid (k=1)   : frames[len//2]                    (long-video rule analog)
  F first (k=1) : frames[0]                         (current long-video rule)

Coverage metric: for each GT-overlapping segment, does any shown frame fall
inside GT(+/-0.5s)? Also reports distance from shown frames to GT center.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth

ROOT = Path('/Users/zzl/PycharmProjects/vad')
FEAT_CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_keyframe_feats.pkl'

gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
cache = pickle.load(open(FEAT_CACHE, 'rb'))
print(f'{len(cache)} cached videos')


def layout_times(fi, ts, motion, name):
    n = len(fi)
    if n == 0:
        return []
    m = {gi: motion[gi] for gi in fi}
    if name == 'A_first2':
        picks = fi[:2]
    elif name == 'B_endpoints':
        picks = [fi[0], fi[-1]] if n > 1 else [fi[0]]
    elif name == 'C_peak2':
        order = sorted(fi, key=lambda g: -m[g])
        picks = []
        for gi in order:
            if all(abs(gi - p) > 2 for p in picks):
                picks.append(gi)
            if len(picks) == 2:
                break
    elif name == 'D_end+peak':
        top = max(fi, key=lambda g: m[g])
        picks = [fi[0]] if top != fi[0] else [fi[0], fi[-1]]
        if top != fi[0]:
            picks.append(top)
    elif name == 'E_mid':
        picks = [fi[n // 2]]
    elif name == 'F_first':
        picks = [fi[0]]
    else:
        raise ValueError(name)
    return [float(ts[gi]) for gi in picks]


rows = []
for vid, e in cache.items():
    ts = e['ts']; motion = e['motion']
    gt_t = [(a / 30.0, b / 30.0) for a, b in gt_all[vid]['segments']]
    for seg in e['segments']:
        fi = seg['frame_indices']
        if not fi:
            continue
        seg_gt = [g for g in gt_t if seg['t_end'] > g[0] and seg['t_start'] < g[1]]
        if not seg_gt:
            continue
        row = {'vid': vid}
        for name in ['A_first2', 'B_endpoints', 'C_peak2', 'D_end+peak', 'E_mid', 'F_first']:
            times = layout_times(fi, ts, motion, name)
            hit = any(a - 0.5 <= t <= b + 0.5 for t in times for (a, b) in seg_gt)
            # distance of nearest shown frame to GT center (discrimination proxy)
            ctr = [(a + b) / 2 for (a, b) in seg_gt]
            dmin = min(abs(t - c) for t in times for c in ctr)
            row[name] = (hit, dmin)
        rows.append(row)

names = ['A_first2', 'B_endpoints', 'C_peak2', 'D_end+peak', 'E_mid', 'F_first']
print(f'GT-overlapping segments: {len(rows)}')
for name in names:
    cov = np.mean([r[name][0] for r in rows])
    dmean = np.mean([r[name][1] for r in rows])
    print(f'  {name:12s} coverage={cov:.3f}  mean-dist-to-GT-center={dmean:.2f}s')

# category split for the wrong-location property crimes
prop = [r for r in rows if r['vid'].startswith(('Shoplifting', 'Stealing', 'Burglary'))]
print(f'\nproperty-crime subset: {len(prop)} segments')
for name in names:
    cov = np.mean([r[name][0] for r in prop])
    print(f'  {name:12s} coverage={cov:.3f}')
