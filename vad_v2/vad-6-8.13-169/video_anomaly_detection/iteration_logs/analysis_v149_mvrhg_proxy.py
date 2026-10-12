"""v149 MVR-HG offline mAP-PROXY GATE (C-mandated decision gate, zero eval cost).

Extends the v146 proxy with the MVR-HG DOUBLE gate:
  (a) HEAD-FREEZE: only relocate dets in the bottom within-video confidence-rank tier
      (rank_pct <= rank_pct_thr). High-conf head (precision checkpoints) never moves.
  (b) NON-DEGRADATION: relocate only if current-window MIL support < within-video P50
      AND target run mean support > within-video P75 AND center-gap > min_gap.
  WIDTH-PRESERVING translation onto the target run's support-weighted centroid.
  count/conf/rank/type frozen.
Reports projected ABSOLUTE mAP. C's bar: must project >= ~0.245 to justify eval spend.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
import analysis_v140_livecurve_lib as LC  # noqa
from mil_model import MILAnomalyScorer

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]
BASE_DIR = ROOT / 'results/batch_eval_v144c_bue65ref'
NPZ = HERE / 'analysis_v140_livecurves'
CURVE_CACHE = HERE / 'analysis_v142_bue_curvecache.pkl'
_CV = pickle.loads(CURVE_CACHE.read_bytes()) if CURVE_CACHE.exists() else {}

_dev = 'cpu'
_mil = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
_mil.load_state_dict(torch.load(ROOT / 'video_anomaly_detection/model_mil.pth', map_location=_dev))
_mil.eval().to(_dev)


def mil_scores_for(vid):
    f = NPZ / f'{vid}.npz'
    if not f.exists():
        return None, None
    z = np.load(f, allow_pickle=True)
    cf = np.asarray(z['clip_features'], dtype=np.float32)
    ts = np.asarray(z['timestamps'], dtype=np.float64)
    n = cf.shape[0]; NS = 32
    if n < NS:
        return None, None
    seg = n // NS
    feats, snip_t = [], []
    for si in range(NS):
        a = si * seg; b = a + seg if si < NS - 1 else n
        feats.append(cf[a:b].mean(axis=0))
        snip_t.append((float(ts[a]), float(ts[b - 1])))
    tens = torch.from_numpy(np.stack(feats)).unsqueeze(0).to(_dev)
    with torch.no_grad():
        s = _mil(tens).squeeze(0).detach().cpu().numpy()
    return s.astype(np.float64), snip_t


def smooth(c, w=3):
    if w <= 1 or len(c) < w:
        return np.asarray(c, float)
    return np.convolve(np.asarray(c, float), np.ones(w) / w, mode='same')


def bue_pick(c, t, ds, de, dur, alpha=1.0, Lmin=6.0, Lmax=60.0,
             lmax_frac=0.8, sm=3, gate=True, union=True):
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
        return 'skip', (ds, de)
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


def snip_center_time(snip_t, i):
    return 0.5 * (snip_t[i][0] + snip_t[i][1])


def mvrhg_relocate(ds, de, conf_rank_pct, s, snip_t, min_gap, hmfrac,
                   rank_pct_thr, lo_pct, hi_pct):
    """Width-preserving translation with head-freeze + valley double gate."""
    if s is None:
        return ds, de, False
    # (a) head-freeze: only bottom-tier confidence dets eligible
    if conf_rank_pct > rank_pct_thr:
        return ds, de, False
    n = len(s)
    p50 = float(np.percentile(s, lo_pct)); p75 = float(np.percentile(s, hi_pct))
    smax = float(s.max()); p20 = float(np.percentile(s, 20))
    hm = ((smax + p20) / 2.0) * hmfrac
    # current-window support
    ci0 = min(range(n), key=lambda i: abs(snip_center_time(snip_t, i) - ds))
    ci1 = min(range(n), key=lambda i: abs(snip_center_time(snip_t, i) - de))
    ci0, ci1 = min(ci0, ci1), max(ci0, ci1)
    cur_sup = float(s[ci0:ci1 + 1].mean()) if ci1 >= ci0 else float(s[ci0])
    # (b) non-degradation: current window must be MIL-weak
    if cur_sup >= p50:
        return ds, de, False
    # target = max-mean above-half-max contiguous run
    above = s >= hm
    runs = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return ds, de, False
    best = max(runs, key=lambda r: s[r[0]:r[1] + 1].mean())
    lo, hi = best
    run_mean = float(s[lo:hi + 1].mean())
    if run_mean <= p75:
        return ds, de, False
    # support-weighted centroid of the run
    idxs = np.arange(lo, hi + 1)
    w = s[lo:hi + 1] - p20
    w = np.clip(w, 1e-6, None)
    cen_i = float((idxs * w).sum() / w.sum())
    ci_lo = int(np.floor(cen_i)); frac = cen_i - ci_lo
    ci_hi = min(ci_lo + 1, n - 1)
    run_center_t = (1 - frac) * snip_center_time(snip_t, ci_lo) + frac * snip_center_time(snip_t, ci_hi)
    det_center = 0.5 * (ds + de)
    if abs(run_center_t - det_center) <= min_gap:
        return ds, de, False
    if ds <= run_center_t <= de:
        return ds, de, False
    width = de - ds
    ns = run_center_t - width / 2.0
    ne = run_center_t + width / 2.0
    return ns, ne, True


def best_iou(a, segs):
    return max([_temporal_iou({'t_start': a[0], 't_end': a[1]}, sgm, FPS) for sgm in segs] or [0.0])


def run(min_gap, hmfrac, rank_pct_thr, lo_pct, hi_pct):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(BASE_DIR))
    gt = {k: v for k, v in gt.items() if k in pred and gt[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}

    base_pred, mvr_pred = {}, {}
    fired = coverage = mh = hm_reg = 0
    net_iou = 0.0
    mh_ex, hm_ex = [], []
    for v, e in pred.items():
        anomalies = e['anomalies']
        pbb = e.get('pre_bue_boundaries')
        dur = e.get('duration') or 0.0
        s, snip_t = mil_scores_for(v)
        segs = gt[v]['segments']
        base_out, mvr_out = [], []
        # within-video confidence rank percentile (0=lowest conf .. 1=highest)
        confs = [a['confidence'] for a in anomalies]
        order = np.argsort(np.argsort(confs))  # rank ascending
        m = len(anomalies)
        rank_pct = [(order[i] / (m - 1)) if m > 1 else 0.0 for i in range(m)]
        for idx, a in enumerate(anomalies):
            cat = a.get('anomaly_type') or e.get('category') or 'unknown'
            c, t = _CV.get((v, cat), (None, None))
            if pbb is not None and idx < len(pbb):
                ds0, de0 = float(pbb[idx][0]), float(pbb[idx][1])
            else:
                ds0, de0 = float(a['t_start']), float(a['t_end'])
            bstart, bend = ds0, de0
            if c is not None and t is not None and len(t) == len(c):
                st, (ns, ne) = bue_pick(c, t, ds0, de0, dur)
                if st == 'fire':
                    bstart, bend = ns, ne
            base_out.append((bstart, bend))
            mds, mde = ds0, de0
            did_fire = False
            if s is not None:
                coverage += 1
                mds, mde, did_fire = mvrhg_relocate(ds0, de0, rank_pct[idx], s, snip_t,
                                                     min_gap, hmfrac, rank_pct_thr, lo_pct, hi_pct)
            mstart, mend = mds, mde
            if c is not None and t is not None and len(t) == len(c):
                st, (ns, ne) = bue_pick(c, t, mds, mde, dur)
                if st == 'fire':
                    mstart, mend = ns, ne
            mvr_out.append((mstart, mend))
            if did_fire:
                fired += 1
                io_b = best_iou((bstart, bend), segs)
                io_m = best_iou((mstart, mend), segs)
                net_iou += (io_m - io_b)
                if io_b < 0.3 <= io_m:
                    mh += 1; mh_ex.append((v, round(io_b, 2), round(io_m, 2)))
                if io_b >= 0.3 > io_m:
                    hm_reg += 1; hm_ex.append((v, round(io_b, 2), round(io_m, 2)))
        base_pred[v] = dict(e, anomalies=[{'t_start': x[0], 't_end': x[1], 'confidence': a['confidence'],
                                           'anomaly_type': a.get('anomaly_type', 'unknown')}
                                          for x, a in zip(base_out, anomalies)])
        mvr_pred[v] = dict(e, anomalies=[{'t_start': x[0], 't_end': x[1], 'confidence': a['confidence'],
                                          'anomaly_type': a.get('anomaly_type', 'unknown')}
                                         for x, a in zip(mvr_out, anomalies)])

    mb = compute_temporal_map(gt, base_pred, fps=FPS, iou_thresholds=THR)
    mm = compute_temporal_map(gt, mvr_pred, fps=FPS, iou_thresholds=THR)
    print(f'### min_gap={min_gap} rank_pct<={rank_pct_thr} lo={lo_pct} hi={hi_pct} hmfrac={hmfrac}')
    print(f'  BUE-reapply baseline  mAP={mb["mAP"]:.5f} .3={mb["AP@0.3"]:.4f} .5={mb["AP@0.5"]:.4f} .7={mb["AP@0.7"]:.4f}')
    print(f'  MVR-HG arm  ABS mAP  ={mm["mAP"]:.5f} .3={mm["AP@0.3"]:.4f} .5={mm["AP@0.5"]:.4f} .7={mm["AP@0.7"]:.4f}')
    print(f'  >>> dmAP={mm["mAP"]-mb["mAP"]:+.5f}  ABS-vs-0.21932={mm["mAP"]-0.21932:+.5f}')
    print(f'  CENSUS: fired={fired}/cov={coverage} miss->hit={mh} hit->miss={hm_reg} net_dIoU={net_iou:+.3f}')
    print(f'    mh_ex={mh_ex[:6]}  hm_ex={hm_ex[:6]}')
    return mm["mAP"]


if __name__ == '__main__':
    for rp in [0.4, 0.6, 0.8]:
        for mg in [6.0, 10.0]:
            run(mg, 1.0, rp, 50, 75)
        print()
