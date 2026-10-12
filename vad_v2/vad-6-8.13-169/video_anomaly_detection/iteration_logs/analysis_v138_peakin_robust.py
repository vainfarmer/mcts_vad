"""
v138 ROBUSTNESS BATTERY — raw-curve PEAK-CONTAINMENT confidence modulation.

Candidate mechanism (verdict-independent, pure post-hoc re-ranking):
    conf' = conf * (1 - w + w * peak_in)
where peak_in = 1 iff the global argmax of the RAW category-conditioned CLIP curve
(frame_scorer.get_category_scores, reconstructed offline per analysis_v138_actionness_lib)
falls inside the detection window. Boundaries / count / detection set untouched.

Why a battery and not a single A/B: the first dual-cache probe gave +0.0099 (v130) and
+0.0116 (v132) for w=0.4, but that is one arm chosen out of six on the same 65-video set.
This script therefore stresses the SAME mechanism along the three axes that would expose
a lucky pick:
  1. CACHE (8 independent production-formula full evals) -> is the sign stable?
  2. WEIGHT w in {0.15,0.2,0.3,0.4,0.5}                  -> is it a knife-edge parameter?
  3. TEMPORAL RESOLUTION of the offline curve (8/16/32 snippets) -> would the denser
     in-vivo curve (the real pipeline has ~130 frames, not 32 snippets) keep the gain?
Plus a per-video broadness report for the headline arm (a gain carried by 1-2 videos is
not promotable) and a continuous variant (1+peak_gap) as a graceful-degradation check.
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))

import analysis_v138_actionness_lib as L  # noqa: E402
from evaluate import (  # noqa: E402
    load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou,
)

LABEL_PATH = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
# production-confidence-formula caches only (Q-score era, no experimental re-rank arm
# baked into the cached confidences)
CACHES = ['batch_eval_v114_full', 'batch_eval_v124_full', 'batch_eval_v125_full',
          'batch_eval_v127_full', 'batch_eval_v128_full', 'batch_eval_v130_full',
          'batch_eval_v131_full', 'batch_eval_v132_full']


def downsample(curve, times, n_target):
    """Average-pool the 32-snippet curve down to n_target bins (resolution probe)."""
    if curve is None or n_target >= len(curve):
        return curve, times
    idx = np.array_split(np.arange(len(curve)), n_target)
    c = np.array([curve[i].mean() for i in idx])
    t = np.array([times[i].mean() for i in idx])
    return c, t


def build(cache, resolution=32):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(ROOT / 'results' / cache))
    gte = {k: v for k, v in gt.items() if k in pred}
    curves, rows = {}, []
    for vid, a, dur in L.iter_cached_detections(ROOT / 'results' / cache):
        cat = a.get('anomaly_type') or ''
        ck = (vid, cat, round(dur, 2))
        if ck not in curves:
            curves[ck] = L.category_curve(vid, cat, dur)
        curve, times = curves[ck]
        if resolution != 32:
            curve, times = downsample(curve, times, resolution)
        f = L.boundary_features(curve, times, a['t_start'], a['t_end'])
        iou = max([_temporal_iou(a, s, FPS) for s in gte[vid]['segments']] + [0.0]) \
            if vid in gte else 0.0
        rows.append(dict(vid=vid, key=(vid, round(a['t_start'], 3), round(a['t_end'], 3)),
                         conf=float(a.get('confidence', 0.0)), iou=iou, feats=f))
    return gte, pred, rows


def apply_arm(pred, rows, fn):
    fmap = {r['key']: r['feats'] for r in rows}
    p2 = {}
    for vid, e in pred.items():
        outs = []
        for a in e['anomalies']:
            k = (vid, round(a['t_start'], 3), round(a['t_end'], 3))
            f = fmap.get(k, {})
            c = float(a.get('confidence', 0.0))
            v = fn(c, f) if f else c
            a2 = dict(a)
            a2['confidence'] = float(c if v is None or np.isnan(v) else v)
            outs.append(a2)
        p2[vid] = dict(e, anomalies=outs)
    return p2


def main():
    print('=' * 100)
    print('AXIS 1+2: cache x weight, resolution=32 (peak_in)')
    header = f'{"cache":10s}' + ''.join(f'{f"w={w}":>12s}' for w in
                                        (0.15, 0.2, 0.3, 0.4, 0.5)) + f'{"base":>10s}'
    print(header)
    per_cache = {}
    built = {}
    for cache in CACHES:
        gte, pred, rows = build(cache)
        built[cache] = (gte, pred, rows)
        base = compute_temporal_map(gte, pred, fps=FPS)
        line = f'{cache[11:15]:10s}'
        per_cache[cache] = {}
        for w in (0.15, 0.2, 0.3, 0.4, 0.5):
            r = compute_temporal_map(
                gte, apply_arm(pred, rows,
                               lambda c, f, w=w: c * (1 - w + w * f['peak_in'])), fps=FPS)
            d = r['mAP'] - base['mAP']
            per_cache[cache][w] = dict(mAP=r['mAP'], d=d, ap7=r['AP@0.7'],
                                       dap7=r['AP@0.7'] - base['AP@0.7'])
            line += f'{d:+12.5f}'
        line += f'{base["mAP"]:10.5f}'
        print(line)
    print('-' * 100)
    for w in (0.15, 0.2, 0.3, 0.4, 0.5):
        ds = [per_cache[c][w]['d'] for c in CACHES]
        dap7 = [per_cache[c][w]['dap7'] for c in CACHES]
        print(f'  w={w}: mean dmAP={np.mean(ds):+.5f}  median={np.median(ds):+.5f}  '
              f'positive {sum(1 for x in ds if x > 0)}/{len(ds)}  '
              f'min={min(ds):+.5f} max={max(ds):+.5f}  mean dAP@0.7={np.mean(dap7):+.5f}')

    print('=' * 100)
    print('AXIS 3: temporal resolution of the offline curve (w=0.4)')
    for res in (8, 16, 32):
        ds = []
        for cache in CACHES:
            gte, pred, rows = build(cache, resolution=res)
            base = compute_temporal_map(gte, pred, fps=FPS)
            r = compute_temporal_map(
                gte, apply_arm(pred, rows,
                               lambda c, f: c * (0.6 + 0.4 * f['peak_in'])), fps=FPS)
            ds.append(r['mAP'] - base['mAP'])
        print(f'  resolution={res:2d} snippets: mean dmAP={np.mean(ds):+.5f}  '
              f'positive {sum(1 for x in ds if x > 0)}/{len(ds)}  min={min(ds):+.5f}')

    print('=' * 100)
    print('CONTINUOUS VARIANT (1+peak_gap in [0,1]) vs binary peak_in, w=0.4')
    for label, fn in [('binary peak_in', lambda c, f: c * (0.6 + 0.4 * f['peak_in'])),
                      ('cont 1+peak_gap', lambda c, f: c * (0.6 + 0.4 * (1.0 + f['peak_gap']))),
                      ('cont act_max_norm', lambda c, f: c * (0.6 + 0.4 * f['act_max_norm']))]:
        ds = []
        for cache in CACHES:
            gte, pred, rows = built[cache]
            base = compute_temporal_map(gte, pred, fps=FPS)
            r = compute_temporal_map(gte, apply_arm(pred, rows, fn), fps=FPS)
            ds.append(r['mAP'] - base['mAP'])
        print(f'  {label:18s} mean={np.mean(ds):+.5f}  positive {sum(1 for x in ds if x > 0)}'
              f'/{len(ds)}  min={min(ds):+.5f} max={max(ds):+.5f}')

    print('=' * 100)
    print('BROADNESS (w=0.4, resolution=32): how many detections are demoted, and does the'
          ' demotion land on misses?')
    tot_fire = tot_det = 0
    hit_dem = miss_dem = 0
    for cache in CACHES:
        gte, pred, rows = built[cache]
        fired = [r for r in rows if r['feats'] and r['feats']['peak_in'] < 0.5]
        tot_fire += len(fired)
        tot_det += len(rows)
        hit_dem += sum(1 for r in fired if r['iou'] >= 0.3)
        miss_dem += sum(1 for r in fired if r['iou'] < 0.3)
    print(f'  demoted (peak OUTSIDE window): {tot_fire}/{tot_det} detections '
          f'({100 * tot_fire / tot_det:.1f}%) across {len(CACHES)} caches')
    print(f'    of those, IoU>=0.3 (wrongly demoted) = {hit_dem}, IoU<0.3 (correctly '
          f'demoted) = {miss_dem}  -> precision of demotion = {miss_dem / max(1, tot_fire):.3f}')
    n_hit = sum(1 for c in CACHES for r in built[c][2] if r['iou'] >= 0.3)
    n_miss = sum(1 for c in CACHES for r in built[c][2] if r['iou'] < 0.3)
    print(f'    base rates: hits {n_hit}, misses {n_miss} -> demotion recall on misses '
          f'{miss_dem / max(1, n_miss):.3f}, false-demotion rate on hits '
          f'{hit_dem / max(1, n_hit):.3f}')

    out = HERE / 'analysis_v138_peakin_robust.json'
    out.write_text(json.dumps({'per_cache': {c: {str(w): per_cache[c][w] for w in per_cache[c]}
                                            for c in per_cache}}, indent=2, default=float))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
