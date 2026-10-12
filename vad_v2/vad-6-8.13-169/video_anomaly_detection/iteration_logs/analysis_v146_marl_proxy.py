"""v146 MARL offline mAP-PROXY GATE (zero eval cost).

Protocol (C-mandated): on the 65 cached BUE-on baseline JSONs,
  baseline arm  = pre_bue_boundaries -> BUE(bue_pick)          (should reproduce cached anomalies)
  MARL arm      = pre_bue_boundaries -> MARL relocate -> BUE   (MARL-before-BUE live ordering)
mil_scores reproduced EXACTLY from cached LIVE clip_features (the pipeline's own substrate,
analysis_v140_livecurves/<vid>.npz) mean-pooled into 32 snippets through model_mil.pth (v1).
Reports mean-mAP lift + per-threshold + TP-regression census (miss->hit / hit->miss).
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
import analysis_v140_livecurve_lib as LC
from mil_model import MILAnomalyScorer

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]
BASE_DIR = ROOT / 'results/batch_eval_v144c_bue65ref'
NPZ = HERE / 'analysis_v140_livecurves'
CURVE_CACHE = HERE / 'analysis_v142_bue_curvecache.pkl'
_CV = pickle.loads(CURVE_CACHE.read_bytes()) if CURVE_CACHE.exists() else {}

# ---- v1 MIL model ----
_dev = 'cpu'
_mil = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
_mil.load_state_dict(torch.load(ROOT / 'video_anomaly_detection/model_mil.pth', map_location=_dev))
_mil.eval().to(_dev)


def mil_scores_for(vid):
    """Reproduce pipeline video_data['mil_scores'] from cached LIVE clip_features."""
    f = NPZ / f'{vid}.npz'
    if not f.exists():
        return None, None
    z = np.load(f, allow_pickle=True)
    cf = np.asarray(z['clip_features'], dtype=np.float32)  # (n_frames, 768)
    ts = np.asarray(z['timestamps'], dtype=np.float64)
    n = cf.shape[0]
    NS = 32
    if n < NS:
        return None, None
    seg = n // NS
    feats, snip_t = [], []
    for si in range(NS):
        a = si * seg
        b = a + seg if si < NS - 1 else n
        feats.append(cf[a:b].mean(axis=0))
        snip_t.append((float(ts[a]), float(ts[b - 1])))  # (start_t, end_t) of snippet
    tens = torch.from_numpy(np.stack(feats)).unsqueeze(0).to(_dev)  # (1,32,768)
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


def marl_relocate(ds, de, s, snip_t, min_gap, hmfrac):
    """Return (relocated_start, relocated_end, fired) using v1 mil_scores s (len32)."""
    smax = float(s.max()); p20 = float(np.percentile(s, 20))
    hm = ((smax + p20) / 2.0) * hmfrac
    above = s >= hm
    amax = int(np.argmax(s))
    # contiguous run containing argmax
    lo = amax
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = amax
    while hi + 1 < len(s) and above[hi + 1]:
        hi += 1
    run_start_t = snip_t[lo][0]
    run_end_t = snip_t[hi][1]
    run_center = 0.5 * (run_start_t + run_end_t)
    run_sup = float(s[lo:hi + 1].mean() - s.mean())
    det_center = 0.5 * (ds + de)
    gap = abs(run_center - det_center)
    already = (ds <= run_center <= de)
    if gap > min_gap and run_sup > 0 and not already and run_end_t > run_start_t:
        return run_start_t, run_end_t, True, gap, run_sup
    return ds, de, False, gap, run_sup


def best_iou(a, segs):
    return max([_temporal_iou({'t_start': a[0], 't_end': a[1]}, sgm, FPS) for sgm in segs] or [0.0])


def curve_support(c, t, ds, de):
    """Return (edge_supported_bool, mean_support_rel). edge_supported = curve is high at
    at least one window edge (the BUE closure test inverted). Weakly-anchored windows
    (both edges below hm) are the safe-to-relocate candidates."""
    if c is None or t is None or len(t) != len(c) or len(t) < 8:
        return True, 0.0  # unknown -> treat as supported (protect: do not relocate)
    cs = smooth(c, 3)
    n = len(t)
    i0 = int(np.searchsorted(t, ds)); i1 = int(np.searchsorted(t, de))
    i0 = max(0, min(i0, n - 1)); i1 = max(i0 + 1, min(i1, n))
    seg = cs[i0:i1]
    hm = (float(seg.max()) + float(np.percentile(cs, 20))) / 2.0
    edge_supported = (cs[i0] >= hm) or (cs[i1 - 1] >= hm)
    mean_rel = float(seg.mean() - np.percentile(cs, 20))
    return edge_supported, mean_rel


def run(min_gap, hmfrac, curve_gate=False):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(BASE_DIR))
    gt = {k: v for k, v in gt.items() if k in pred and gt[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}

    base_pred, marl_pred = {}, {}
    fired = coverage = mh = hm_reg = 0
    net_iou = 0.0
    miss_hit_ex, hit_miss_ex = [], []
    for v, e in pred.items():
        anomalies = e['anomalies']
        pbb = e.get('pre_bue_boundaries')
        dur = e.get('duration') or 0.0
        s, snip_t = mil_scores_for(v)
        segs = gt[v]['segments']
        base_out, marl_out = [], []
        # index-aligned pre_bue windows; fall back to final if missing
        for idx, a in enumerate(anomalies):
            cat = a.get('anomaly_type') or e.get('category') or 'unknown'
            c, t = _CV.get((v, cat), (None, None))
            if pbb is not None and idx < len(pbb):
                ds0, de0 = float(pbb[idx][0]), float(pbb[idx][1])
            else:
                ds0, de0 = float(a['t_start']), float(a['t_end'])
            # baseline: pre_bue -> BUE
            bstart, bend = ds0, de0
            if c is not None and t is not None and len(t) == len(c):
                st, (ns, ne) = bue_pick(c, t, ds0, de0, dur)
                if st == 'fire':
                    bstart, bend = ns, ne
            base_out.append((bstart, bend))
            # MARL: pre_bue -> MARL -> BUE
            mds, mde = ds0, de0
            did_fire = False
            if s is not None:
                coverage += 1
                mds, mde, did_fire, gap, rsup = marl_relocate(ds0, de0, s, snip_t, min_gap, hmfrac)
                if did_fire and curve_gate:
                    edge_sup, _msr = curve_support(c, t, ds0, de0)
                    if edge_sup:  # curve anchors this window -> protect TP, do NOT relocate
                        mds, mde, did_fire = ds0, de0, False
            mstart, mend = mds, mde
            if c is not None and t is not None and len(t) == len(c):
                st, (ns, ne) = bue_pick(c, t, mds, mde, dur)
                if st == 'fire':
                    mstart, mend = ns, ne
            marl_out.append((mstart, mend))
            if did_fire:
                fired += 1
                io_b = best_iou((bstart, bend), segs)
                io_m = best_iou((mstart, mend), segs)
                net_iou += (io_m - io_b)
                if io_b < 0.3 <= io_m:
                    mh += 1; miss_hit_ex.append((v, idx, round(io_b, 3), round(io_m, 3)))
                if io_b >= 0.3 > io_m:
                    hm_reg += 1; hit_miss_ex.append((v, idx, round(io_b, 3), round(io_m, 3)))
        base_pred[v] = dict(e, anomalies=[{'t_start': x[0], 't_end': x[1], 'confidence': a['confidence'],
                                           'anomaly_type': a.get('anomaly_type', 'unknown')}
                                          for x, a in zip(base_out, anomalies)])
        marl_pred[v] = dict(e, anomalies=[{'t_start': x[0], 't_end': x[1], 'confidence': a['confidence'],
                                           'anomaly_type': a.get('anomaly_type', 'unknown')}
                                          for x, a in zip(marl_out, anomalies)])

    cached = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)
    mb = compute_temporal_map(gt, base_pred, fps=FPS, iou_thresholds=THR)
    mm = compute_temporal_map(gt, marl_pred, fps=FPS, iou_thresholds=THR)
    print(f'\n### min_gap={min_gap} hmfrac={hmfrac}')
    print(f'  CACHED anomalies mAP={cached["mAP"]:.5f} (fidelity ref)')
    print(f'  BUE-reapply baseline  mAP={mb["mAP"]:.5f} '
          f'AP.3={mb["AP@0.3"]:.4f} .5={mb["AP@0.5"]:.4f} .7={mb["AP@0.7"]:.4f}')
    print(f'  MARL arm              mAP={mm["mAP"]:.5f} '
          f'AP.3={mm["AP@0.3"]:.4f} .5={mm["AP@0.5"]:.4f} .7={mm["AP@0.7"]:.4f}')
    print(f'  >>> dmAP(MARL-baseBUE) = {mm["mAP"]-mb["mAP"]:+.5f}   '
          f'per-thr d: .3={mm["AP@0.3"]-mb["AP@0.3"]:+.4f} '
          f'.5={mm["AP@0.5"]-mb["AP@0.5"]:+.4f} .7={mm["AP@0.7"]-mb["AP@0.7"]:+.4f}')
    print(f'  CENSUS: fired={fired}/cov={coverage}  miss->hit={mh}  hit->miss(TPreg)={hm_reg}  '
          f'net_dIoU={net_iou:+.3f}')
    print(f'    miss->hit ex: {miss_hit_ex[:8]}')
    print(f'    hit->miss ex: {hit_miss_ex[:8]}')
    return mm["mAP"] - mb["mAP"]


if __name__ == '__main__':
    print('===== UNGATED (v146 MARL as proposed) =====')
    run(6.0, 1.0, curve_gate=False)
    print('\n===== CURVE-SUPPORT GATED (only relocate curve-weak windows) =====')
    for mg in [6.0, 10.0]:
        run(mg, 1.0, curve_gate=True)
