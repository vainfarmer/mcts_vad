"""Agent C gate: (a) verify Lmax=40 claim, (b) sign robustness on ALL available
full caches (proposal used only 3 of 12), (c) signal-free control battery.
Reuses analysis_v142_bue_verify.bue_pick verbatim (the reproduced spec).
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
import analysis_v142_bue_verify as V

FPS = 30; THR = [0.3, 0.5, 0.7]
LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data')


def arms_for(mode):
    if mode == 'lmax':
        return {f'BUE L<={L}': dict(alpha=1.0, gate=True, union=True, Lmax=float(L))
                for L in (15, 20, 25, 30, 35, 40, 50, 60, 90)}
    return {'BUE L<=40': dict(alpha=1.0, gate=True, union=True, Lmax=40.0)}


def evaluate_arm(gt, pred, kw, ctrl=None, seed=0):
    rng = np.random.RandomState(seed)
    np2, fired = {}, 0
    for v, e in pred.items():
        na = []
        for a in e['anomalies']:
            cat = a.get('anomaly_type') or e.get('category') or 'unknown'
            c, t = V.curve(v, cat)
            b = dict(a)
            if c is not None and t is not None and len(t) == len(c):
                dur = e.get('duration') or float(t[-1])
                st, (ns, ne) = V.bue_pick(c, t, a['t_start'], a['t_end'], dur, **kw)
                if st == 'fire' and (abs(ns - a['t_start']) > 1e-6 or abs(ne - a['t_end']) > 1e-6):
                    if ctrl is None:
                        b['t_start'], b['t_end'] = ns, ne
                    else:
                        L = ne - ns
                        ods, ode = a['t_start'], a['t_end']
                        if ctrl == 'C1':   # duration-matched symmetric dilation
                            ctr = 0.5 * (ods + ode)
                            b['t_start'] = max(0.0, ctr - L / 2); b['t_end'] = min(dur, ctr + L / 2)
                        elif ctrl == 'C2':  # union w/ RANDOM window of same length
                            rs = rng.uniform(0, max(1e-6, dur - L))
                            b['t_start'] = max(0.0, min(ods, rs))
                            b['t_end'] = min(dur, max(ode, rs + L))
                        elif ctrl == 'C4':  # union w/ window of same length, same SIDE bias, random offset
                            rs = rng.uniform(0, max(1e-6, dur - L))
                            b['t_start'] = max(0.0, min(ods, rs)); b['t_end'] = min(dur, max(ode, rs + L))
                    fired += 1
            na.append(b)
        o = dict(e); o['anomalies'] = na; np2[v] = o
    m = compute_temporal_map(gt, np2, fps=FPS, iou_thresholds=THR)
    return m, fired, np2


def main():
    mode = sys.argv[1]
    dirs = [Path(x) for x in sys.argv[2:]]
    gt_all = load_ground_truth(LABEL_PATH, split='test')
    agg = {}
    for rd in dirs:
        if not rd.exists():
            continue
        pred = load_predictions(str(rd))
        gt = {k: v for k, v in gt_all.items() if k in pred and gt_all[k]['segments']}
        pred = {k: v for k, v in pred.items() if k in gt}
        if len(pred) < 40:
            print(f'{rd.name}: SKIP only {len(pred)} vids'); continue
        base = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)
        print(f'\n=== {rd.name}: {len(pred)}v base mAP={base["mAP"]:.5f}', flush=True)
        if mode == 'ctrl':
            m, f, _ = evaluate_arm(gt, pred, dict(alpha=1.0, gate=True, union=True, Lmax=40.0))
            print(f'  {"BUE L<=40":22s} d={m["mAP"]-base["mAP"]:+.5f} fire={f}', flush=True)
            agg.setdefault('BUE L<=40', []).append(m['mAP'] - base['mAP'])
            for ctrl in ('C1', 'C2'):
                ds = []
                seeds = (0,) if ctrl == 'C1' else (0, 1, 2, 3, 4)
                for s in seeds:
                    mc, _, _ = evaluate_arm(gt, pred, dict(alpha=1.0, gate=True, union=True, Lmax=40.0), ctrl=ctrl, seed=s)
                    ds.append(mc['mAP'] - base['mAP'])
                print(f'  {ctrl:22s} d={np.mean(ds):+.5f} (seeds {[round(x,4) for x in ds]})', flush=True)
                agg.setdefault(ctrl, []).append(float(np.mean(ds)))
        else:
            for nm, kw in arms_for(mode).items():
                m, f, _ = evaluate_arm(gt, pred, kw)
                d = m['mAP'] - base['mAP']
                print(f'  {nm:22s} d={d:+.5f} mAP={m["mAP"]:.5f} fire={f}', flush=True)
                agg.setdefault(nm, []).append(d)
    V.CACHE.write_bytes(pickle.dumps(V._CV))
    print('\n=== MEAN ===')
    for k, v in agg.items():
        print(f'  {k:22s} mean={np.mean(v):+.5f} n+={sum(1 for x in v if x>0)}/{len(v)} {[round(x,4) for x in v]}')


main()
