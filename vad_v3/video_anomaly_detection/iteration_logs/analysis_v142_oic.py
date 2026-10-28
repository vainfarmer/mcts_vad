"""v142 OIC duration-selection precheck on the FIDELITY-GATED live curve.

Mechanism family under test: keep each final detection's ANCHOR, re-select its DURATION
(and optionally its two edges independently) by maximizing outer-inner contrast
   OIC(s,e) = mean(curve[s:e]) - mean(curve on [s-aL,s) U (e,e+aL])
on the verdict-INDEPENDENT get_category_scores curve. Count-preserving,
confidence-frozen. No curve modification -> not FM1's family.

Substrate: analysis_v140_livecurve_lib.live_curve (58/58 live-peak fidelity gate).
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
import analysis_v140_livecurve_lib as LC
LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]
_CV = {}


def curve(vid, cat):
    k = (vid, cat)
    if k in _CV: return _CV[k]
    if not (LC.CURVE_CACHE / f'{vid}.npz').exists():
        _CV[k] = (None, None); return _CV[k]
    try:
        c, t = LC.live_curve(vid, cat)
    except Exception:
        c, t = None, None
    _CV[k] = (c, t); return _CV[k]


def smooth(c, w=3):
    if w <= 1 or len(c) < w: return c
    return np.convolve(c, np.ones(w) / w, mode='same')


def oic_pick(c, t, ds, de, alpha, Lmin, Lmax, mode, sm=3):
    """returns (s,e) seconds maximizing OIC, or None."""
    n = len(t)
    if n < 8: return None
    cs = smooth(np.asarray(c, float), sm)
    P = np.concatenate([[0.0], np.cumsum(cs)])
    dt = float(np.median(np.diff(t))) or 0.5
    def mean(i, j):  # [i,j)
        i = max(0, min(i, n)); j = max(0, min(j, n))
        return (P[j] - P[i]) / max(1, j - i) if j > i else 0.0
    i0 = int(np.searchsorted(t, ds)); i1 = int(np.searchsorted(t, de))
    i1 = max(i0 + 1, min(i1, n))
    anchor = i0 + int(np.argmax(cs[i0:i1])) if i1 > i0 else i0
    kmin = max(2, int(round(Lmin / dt))); kmax = max(kmin + 1, int(round(Lmax / dt)))
    best, bs, be = -1e9, None, None
    if mode == 'sym':          # center fixed at anchor, search L only (== ORACLE_dur structure)
        for k in range(kmin, kmax + 1):
            s = anchor - k // 2; e = s + k
            if s < 0 or e > n: continue
            r = max(1, int(round(alpha * k)))
            out = (mean(s - r, s) * min(r, max(0, s)) + mean(e, e + r) * min(r, max(0, n - e)))
            den = min(r, max(0, s)) + min(r, max(0, n - e))
            v = mean(s, e) - (out / den if den else 0.0)
            if v > best: best, bs, be = v, s, e
    else:                      # independent edges, must contain anchor
        for s in range(max(0, anchor - kmax), anchor + 1):
            for k in range(kmin, kmax + 1):
                e = s + k
                if e > n or e <= anchor: continue
                r = max(1, int(round(alpha * k)))
                nl = min(r, max(0, s)); nr = min(r, max(0, n - e))
                out = mean(s - r, s) * nl + mean(e, e + r) * nr
                den = nl + nr
                v = mean(s, e) - (out / den if den else 0.0)
                if v > best: best, bs, be = v, s, e
    if bs is None: return None
    return float(t[bs]), float(t[min(be, n - 1)])


def run(rd, arms):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(rd))
    gt = {k: v for k, v in gt.items() if k in pred and gt[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}
    base = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)
    ndet = sum(len(e['anomalies']) for e in pred.values())
    print(f'\n=== {rd.name}: {len(pred)} vids {ndet} det base mAP={base["mAP"]:.5f}')
    out = {}
    for nm, kw in arms.items():
        np2, fired, cov = {}, 0, 0
        for v, e in pred.items():
            na = []
            for a in e['anomalies']:
                cat = a.get('anomaly_type') or e.get('category') or 'unknown'
                c, t = curve(v, cat)
                b = dict(a)
                if c is not None:
                    cov += 1
                    vd = e.get('duration') or float(t[-1])
                    r = oic_pick(c, t, a['t_start'], a['t_end'],
                                 kw['alpha'], kw['Lmin'], min(kw['Lmax'], 0.8 * vd), kw['mode'])
                    if r and r[1] > r[0] and (abs(r[0] - a['t_start']) > 1e-6 or abs(r[1] - a['t_end']) > 1e-6):
                        b['t_start'], b['t_end'] = r; fired += 1
                na.append(b)
            o = dict(e); o['anomalies'] = na; np2[v] = o
        m = compute_temporal_map(gt, np2, fps=FPS, iou_thresholds=THR)
        ious = [max([_temporal_iou(a, s, FPS) for s in gt[v]['segments']] or [0.0])
                for v, e in np2.items() for a in e['anomalies']]
        ious = np.array(ious)
        print(f'  {nm:34s} d={m["mAP"]-base["mAP"]:+.5f} mAP={m["mAP"]:.5f} '
              f'AP@.3={m["AP@0.3"]:.4f} .5={m["AP@0.5"]:.4f} .7={m["AP@0.7"]:.4f} '
              f'h.3={int((ious>=.3).sum())} h.5={int((ious>=.5).sum())} h.7={int((ious>=.7).sum())} '
              f'zero={int((ious<=0).sum())} fired={fired}/{cov}')
        out[nm] = m['mAP'] - base['mAP']
    return out


ARMS = {}
for mode in ('sym', 'free'):
    for alpha in (0.25, 0.5):
        for Lmax in (30, 60):
            ARMS[f'{mode} a={alpha} L=[6,{Lmax}]'] = dict(mode=mode, alpha=alpha, Lmin=6, Lmax=Lmax)

if __name__ == '__main__':
    dirs = [Path(x) for x in (sys.argv[1:] or [str(ROOT / 'results/batch_eval_v124_full')])]
    agg = {}
    for d in dirs:
        for k, v in run(d, ARMS).items(): agg.setdefault(k, []).append(v)
    print('\n=== mean dmAP ===')
    for k, v in sorted(agg.items(), key=lambda x: -np.mean(x[1])):
        print(f'  {k:34s} mean={np.mean(v):+.5f} n+={sum(1 for x in v if x>0)}/{len(v)} {[round(x,4) for x in v]}')
