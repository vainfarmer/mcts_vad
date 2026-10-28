"""Agent C gate check: apply the MANDATED video-level bootstrap CI to BUE's own
OFFLINE paired delta. The proposal reports only point deltas, which the session
protocol forbids as a verdict statistic.

Arms: pre-registered Lmax=60 and test-selected Lmax=30/40.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map
from analysis_v142_bue_verify import bue_pick, curve, LABEL_PATH, FPS, THR

N = 300


def apply_bue(pred, **kw):
    out = {}
    for v, e in pred.items():
        na = []
        for a in e['anomalies']:
            cat = a.get('anomaly_type') or e.get('category') or 'unknown'
            c, t = curve(v, cat)
            b = dict(a)
            if c is not None and t is not None and len(t) == len(c):
                dur = e.get('duration') or float(t[-1])
                st, (ns, ne) = bue_pick(c, t, a['t_start'], a['t_end'], dur, **kw)
                if st == 'fire':
                    b['t_start'], b['t_end'] = ns, ne
            na.append(b)
        o = dict(e); o['anomalies'] = na; out[v] = o
    return out


def boot(gt, pa, pb, n=N, seed=0):
    vids = sorted(set(pa) & set(pb) & set(gt))
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        samp = rng.choice(vids, size=len(vids), replace=True)
        g, a, b = {}, {}, {}
        for i, v in enumerate(samp):
            k = f'{v}#{i}'
            g[k], a[k], b[k] = gt[v], pa[v], pb[v]
        ds.append(compute_temporal_map(g, b, fps=FPS, iou_thresholds=THR)['mAP']
                  - compute_temporal_map(g, a, fps=FPS, iou_thresholds=THR)['mAP'])
    return np.array(ds)


ARMS = {
    'Lmax=60 PRE-REGISTERED': dict(alpha=1.0, gate=True, union=True, Lmax=60.0),
    'Lmax=40 SELECTED':       dict(alpha=1.0, gate=True, union=True, Lmax=40.0),
    'Lmax=30 SELECTED':       dict(alpha=1.0, gate=True, union=True, Lmax=30.0),
}

if __name__ == '__main__':
    gt_all = load_ground_truth(LABEL_PATH, split='test')
    for d in [Path(x) for x in sys.argv[1:]]:
        pred = load_predictions(str(d))
        gt = {k: v for k, v in gt_all.items() if k in pred and gt_all[k]['segments']}
        pred = {k: v for k, v in pred.items() if k in gt}
        base = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)['mAP']
        print(f'\n=== {d.name}  base mAP={base:.5f}  nvid={len(pred)}', flush=True)
        for nm, kw in ARMS.items():
            p2 = apply_bue(pred, **kw)
            pt = compute_temporal_map(gt, p2, fps=FPS, iou_thresholds=THR)['mAP'] - base
            ds = boot(gt, pred, p2)
            lo, hi = np.percentile(ds, [5, 95])
            print(f'  {nm:24s} point={pt:+.5f} boot_mean={ds.mean():+.5f} '
                  f'90%CI=[{lo:+.5f},{hi:+.5f}] halfwidth={(hi-lo)/2:.5f} '
                  f'P(d>0)={(ds > 0).mean():.3f}', flush=True)
