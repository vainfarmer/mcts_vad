"""v164 offline PAIRED CBT-recal marginal @290.

The v160_full JSONs DO carry pre_cbt_boundaries (the FM20 premise is stale):
  ON  arm = anomalies as written (post CBT+recal)
  OFF arm = same anomalies with [t_start,t_end] restored from pre_cbt_boundaries
Same detection set, same VLM run => a clean paired marginal with ZERO VLM noise.
This is a deterministic transform on the boundaries only (conf/order/count fixed).
"""
import json
import glob
import sys
import numpy as np

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, compute_temporal_map, _extract_video_id

LABEL = '/Users/zzl/Movies/ucf-crime/data'
V160 = '/Users/zzl/PycharmProjects/vad/results/batch_eval_v160_full'
HARD = {'Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion'}

gt = load_ground_truth(LABEL, 'test')

pred_on, pred_off = {}, {}
for f in glob.glob(V160 + '/*.json'):
    d = json.load(open(f))
    if not isinstance(d, dict) or 'anomalies' not in d:
        continue
    vid = _extract_video_id(d.get('video_path', f))
    an = d['anomalies']
    pc = d.get('pre_cbt_boundaries')
    dur = d.get('duration', 0)
    on = [{'t_start': a['t_start'], 't_end': a['t_end'],
           'confidence': a.get('confidence', 0.0)} for a in an]
    if pc is not None and len(pc) == len(an):
        off = [{'t_start': pc[i][0], 't_end': pc[i][1],
                'confidence': an[i].get('confidence', 0.0)} for i in range(len(an))]
    else:
        off = [dict(x) for x in on]
    pred_on[vid] = {'anomalies': on, 'duration': dur}
    pred_off[vid] = {'anomalies': off, 'duration': dur}


def mp(gt_sub, pred):
    ps = {k: pred[k] for k in gt_sub if k in pred}
    return compute_temporal_map(gt_sub, ps)['mAP']


def subset(cats=None, invert=False):
    if cats is None:
        return dict(gt)
    return {k: v for k, v in gt.items() if (v['category'] in cats) != invert}


gt_all = subset()
gt_hard = subset(HARD)

on_all, off_all = mp(gt_all, pred_on), mp(gt_all, pred_off)
on_hard, off_hard = mp(gt_hard, pred_on), mp(gt_hard, pred_off)

print('PAIRED @290 FULL : on(CBT+recal)=%.5f  off(no-CBT)=%.5f  delta=%+.5f'
      % (on_all, off_all, on_all - off_all))
print('PAIRED @88  HARD : on=%.5f  off=%.5f  delta=%+.5f'
      % (on_hard, off_hard, on_hard - off_hard))

# video-matched paired bootstrap on the FULL 290 (the promotion question is @290)
vids = [k for k in gt_all if k in pred_on]
rng = np.random.default_rng(12345)
N = 2000
deltas = np.empty(N)
gt_list = gt_all
for b in range(N):
    samp = rng.choice(len(vids), size=len(vids), replace=True)
    gsub = {}
    p_on = {}
    p_off = {}
    for j, si in enumerate(samp):
        v = vids[si]
        key = f'{v}#{j}'          # unique key so duplicates count independently
        gsub[key] = gt_list[v]
        p_on[key] = pred_on[v]
        p_off[key] = pred_off[v]
    deltas[b] = (compute_temporal_map(gsub, p_on)['mAP']
                 - compute_temporal_map(gsub, p_off)['mAP'])

lo, hi = np.percentile(deltas, [5, 95])
p_gt0 = float(np.mean(deltas > 0))
print('BOOTSTRAP @290 paired delta: point=%+.5f  ci90=[%+.5f,%+.5f]  P(delta>0)=%.4f'
      % (on_all - off_all, lo, hi, p_gt0))

# per hard cat paired
print('--- per hard cat paired ---')
for c in sorted(HARD):
    g = subset({c})
    print('%-14s on=%.5f off=%.5f d=%+.5f' % (c, mp(g, pred_on), mp(g, pred_off), mp(g, pred_on) - mp(g, pred_off)))
