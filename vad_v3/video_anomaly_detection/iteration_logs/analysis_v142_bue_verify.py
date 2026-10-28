"""C-team independent verification of IDEA B (BUE) exactly as specified.

Spec under test (verbatim from the proposal):
  - read verdict-independent curve via live_curve (fidelity-gated substrate)
  - smooth w=3
  - CLOSURE TEST: hm = (cs[i0:i1].max() + percentile(cs,20))/2 ;
    if cs[i0] < hm and cs[i1-1] < hm  -> SKIP (byte-identical)
  - else grid search (s,e), s<=anchor<e, L in [6s, min(60s, 0.8*dur)]
    maximizing OIC = mean_in - weighted_mean(outer rings width alpha*L), alpha=1.0
  - emit UNION of old window and OIC window
Also runs: closure-gated REPLACE (no union), ungated union, and the "old" alphas
for comparison, plus a per-arm hit census.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
import analysis_v140_livecurve_lib as LC

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]
CACHE = HERE / 'analysis_v142_bue_curvecache.pkl'
_CV = {}
if CACHE.exists():
    _CV = pickle.loads(CACHE.read_bytes())


def curve(vid, cat):
    k = (vid, cat)
    if k in _CV:
        return _CV[k]
    if not (LC.CURVE_CACHE / f'{vid}.npz').exists():
        _CV[k] = (None, None); return _CV[k]
    try:
        c, t = LC.live_curve(vid, cat)
    except Exception:
        c, t = None, None
    _CV[k] = (np.asarray(c, float) if c is not None else None,
              np.asarray(t, float) if t is not None else None)
    return _CV[k]


def smooth(c, w=3):
    if w <= 1 or len(c) < w:
        return np.asarray(c, float)
    return np.convolve(np.asarray(c, float), np.ones(w) / w, mode='same')


def bue_pick(c, t, ds, de, dur, alpha=1.0, Lmin=6.0, Lmax=60.0,
             lmax_frac=0.8, sm=3, gate=True, union=True):
    """returns ('skip'|'fire', (s,e)) per the BUE spec."""
    n = len(t)
    if n < 8:
        return 'skip', (ds, de)
    cs = smooth(c, sm)
    P = np.concatenate([[0.0], np.cumsum(cs)])

    def mean(i, j):
        i = max(0, min(i, n)); j = max(0, min(j, n))
        return (P[j] - P[i]) / max(1, j - i) if j > i else 0.0

    dt = float(np.median(np.diff(t))) or 0.5
    i0 = int(np.searchsorted(t, ds)); i1 = int(np.searchsorted(t, de))
    i0 = max(0, min(i0, n - 1)); i1 = max(i0 + 1, min(i1, n))
    seg = cs[i0:i1]
    hm = (float(seg.max()) + float(np.percentile(cs, 20))) / 2.0
    anchor = i0 + int(np.argmax(seg))
    if gate and cs[i0] < hm and cs[i1 - 1] < hm:
        return 'skip', (ds, de)          # boundary genuinely closed by the curve
    eff_lmax = min(Lmax, lmax_frac * dur) if dur and dur > 0 else Lmax
    kmin = max(2, int(round(Lmin / dt))); kmax = max(kmin + 1, int(round(eff_lmax / dt)))
    best, bs, be = -1e30, None, None
    for s in range(max(0, anchor - kmax), anchor + 1):
        for k in range(kmin, kmax + 1):
            e = s + k
            if e > n or e <= anchor:
                continue
            r = max(1, int(round(alpha * k)))
            nl = min(r, max(0, s)); nr = min(r, max(0, n - e))
            den = nl + nr
            out = (mean(s - r, s) * nl + mean(e, e + r) * nr) / den if den else 0.0
            v = mean(s, e) - out
            if v > best:
                best, bs, be = v, s, e
    if bs is None:
        return 'skip', (ds, de)
    ns, ne = float(t[bs]), float(t[min(be, n - 1)])
    if union:
        ns, ne = min(ds, ns), max(de, ne)
    if dur and dur > 0:
        ns = max(0.0, min(ns, dur)); ne = max(0.0, min(ne, dur))
    if ne <= ns:
        return 'skip', (ds, de)
    return 'fire', (ns, ne)


def census(gt, pred):
    ious = []
    for v, e in pred.items():
        for a in e['anomalies']:
            ious.append(max([_temporal_iou(a, s, FPS) for s in gt[v]['segments']] or [0.0]))
    io = np.array(ious) if ious else np.zeros(0)
    return (int((io >= .3).sum()), int((io >= .5).sum()),
            int((io >= .7).sum()), int((io <= 0).sum()), len(io))


def run(rd, arms):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(rd))
    gt = {k: v for k, v in gt.items() if k in pred and gt[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}
    base = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)
    h = census(gt, pred)
    print(f'\n=== {rd.name}: {len(pred)} vids {h[4]} det base mAP={base["mAP"]:.5f} '
          f'AP.3={base["AP@0.3"]:.4f} .5={base["AP@0.5"]:.4f} .7={base["AP@0.7"]:.4f} '
          f'h.3={h[0]} h.5={h[1]} h.7={h[2]} zero={h[3]}', flush=True)
    res = {}
    for nm, kw in arms.items():
        np2, fired, cov, skipped = {}, 0, 0, 0
        for v, e in pred.items():
            na = []
            for a in e['anomalies']:
                cat = a.get('anomaly_type') or e.get('category') or 'unknown'
                c, t = curve(v, cat)
                b = dict(a)
                if c is not None and t is not None and len(t) == len(c):
                    cov += 1
                    dur = e.get('duration') or float(t[-1])
                    st, (ns, ne) = bue_pick(c, t, a['t_start'], a['t_end'], dur, **kw)
                    if st == 'fire' and (abs(ns - a['t_start']) > 1e-6 or abs(ne - a['t_end']) > 1e-6):
                        b['t_start'], b['t_end'] = ns, ne; fired += 1
                    elif st == 'skip':
                        skipped += 1
                na.append(b)
            o = dict(e); o['anomalies'] = na; np2[v] = o
        m = compute_temporal_map(gt, np2, fps=FPS, iou_thresholds=THR)
        hh = census(gt, np2)
        print(f'  {nm:30s} d={m["mAP"]-base["mAP"]:+.5f} mAP={m["mAP"]:.5f} '
              f'AP.3={m["AP@0.3"]:.4f} .5={m["AP@0.5"]:.4f} .7={m["AP@0.7"]:.4f} '
              f'h.3={hh[0]} h.5={hh[1]} h.7={hh[2]} zero={hh[3]} '
              f'fire={fired}/{cov} closed_skip={skipped}', flush=True)
        res[nm] = m['mAP'] - base['mAP']
    return res


ARMS = {
    'BUE spec (gate,union,a=1)': dict(alpha=1.0, gate=True, union=True),
    'gate,REPLACE,a=1':          dict(alpha=1.0, gate=True, union=False),
    'UNGATED,union,a=1':         dict(alpha=1.0, gate=False, union=True),
    'gate,union,a=0.5':          dict(alpha=0.5, gate=True, union=True),
    'gate,union,a=1,L<=30':      dict(alpha=1.0, gate=True, union=True, Lmax=30.0),
}

if __name__ == '__main__':
    dirs = [Path(x) for x in (sys.argv[1:] or [str(ROOT / 'results/batch_eval_v124_full')])]
    agg = {}
    for d in dirs:
        for k, v in run(d, ARMS).items():
            agg.setdefault(k, []).append(v)
    CACHE.write_bytes(pickle.dumps(_CV))
    print('\n=== mean dmAP ===')
    for k, v in sorted(agg.items(), key=lambda x: -np.mean(x[1])):
        print(f'  {k:30s} mean={np.mean(v):+.5f} n+={sum(1 for x in v if x>0)}/{len(v)} '
              f'{[round(x,4) for x in v]}')
