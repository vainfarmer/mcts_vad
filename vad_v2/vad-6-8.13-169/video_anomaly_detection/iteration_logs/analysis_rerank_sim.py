"""Deterministic confidence re-ranking simulation on v97 full predictions.

Modifies ONLY the 'confidence' field of cached v97 detections with read-only
structural signals (duration-prior fit, curve-peak containment, window rank,
window mean from cached CLIP curves), then re-scores with the REAL
compute_temporal_map. Baseline must reproduce 0.16155 (0.21823/0.15005/0.11636).
"""
import sys, json, pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, _temporal_iou, compute_temporal_map
from pipeline import CATEGORY_DURATION_PRIORS

ROOT = Path('/Users/zzl/PycharmProjects/vad')
CURVES = ROOT / 'video_anomaly_detection/iteration_logs/analysis_verify_curves.pkl'

gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
pred = {k: v for k, v in load_predictions(str(ROOT / 'results/batch_eval_v97_full')).items() if k in gt_all}
gt = {k: v for k, v in gt_all.items() if k in pred}
curves = pickle.load(open(CURVES, 'rb'))

# ---- per-detection read-only features ----
def feats_for(vid, a):
    cat = gt[vid]['category']
    dur = a['t_end'] - a['t_start']
    pri = CATEGORY_DURATION_PRIORS.get(cat, {}).get('expected_duration', (3.0, 30.0))
    lo, hi = pri
    if lo <= dur <= hi:
        dur_fit = 1.0
    elif dur < lo:
        dur_fit = float(dur / lo)
    else:
        dur_fit = float(hi / dur)
    peak_contain, wrank, in_mean = 0.0, 0.5, 0.0
    e = curves.get(vid)
    if e is not None:
        curve = e['rel'].get(cat)
        if curve is not None:
            ts = e['ts']
            li = int(np.searchsorted(ts, a['t_start']))
            hi_i = int(min(len(ts), np.searchsorted(ts, a['t_end'])))
            if hi_i <= li:
                hi_i = li + 1
            peak_contain = 1.0 if li <= int(np.argmax(curve)) <= hi_i else 0.0
            in_mean = float(curve[li:hi_i].mean())
            wlen = hi_i - li
            if len(curve) > wlen:
                cs = np.cumsum(np.concatenate([[0.0], curve]))
                wmeans = (cs[wlen:] - cs[:-wlen]) / wlen
                wrank = float((wmeans < in_mean).mean())
            else:
                wrank = 1.0
    return dur_fit, peak_contain, wrank, in_mean

rows = []
for vid, entry in pred.items():
    segs = gt[vid]['segments']
    for a in entry['anomalies']:
        best = max([_temporal_iou(a, s, 30) for s in segs] + [0.0])
        df, pc, wr, im = feats_for(vid, a)
        rows.append({'vid': vid, 'conf': a['confidence'], 'iou': best,
                     'dur_fit': df, 'peak_contain': pc, 'wrank': wr, 'in_mean': im,
                     'is_tp': best >= 0.3, 'is_fp': best < 0.1})

tp = [r for r in rows if r['is_tp']]
fp = [r for r in rows if r['is_fp']]
def sep_auc(key):
    vt = [r[key] for r in tp]; vf = [r[key] for r in fp]
    c = sum(1 for t in vt for f in vf if t > f) + 0.5 * sum(1 for t in vt for f in vf if t == f)
    return c / (len(vt) * len(vf))
print(f'TP={len(tp)} FP={len(fp)}')
for k in ['conf', 'dur_fit', 'peak_contain', 'wrank', 'in_mean']:
    print(f'  sep-AUC {k:14s}: {sep_auc(k):.3f} (TP mean={np.mean([r[k] for r in tp]):.3f} FP mean={np.mean([r[k] for r in fp]):.3f})')

# ---- mAP simulation ----
def clone_pred():
    return {vid: {'anomalies': [dict(a) for a in entry['anomalies']]} for vid, entry in pred.items()}

def run(mod_fn, name):
    p = clone_pred()
    for vid, entry in p.items():
        for a in entry['anomalies']:
            df, pc, wr, im = feats_for(vid, a)
            a['confidence'] = float(np.clip(mod_fn(a['confidence'], df, pc, wr, im), 0.0, 1.0))
    res = compute_temporal_map(gt, p, 30)
    print(f"  {name:34s} mAP={res['mAP']:.5f} AP03={res['AP@0.3']:.5f} AP05={res['AP@0.5']:.5f} AP07={res['AP@0.7']:.5f}")
    return res

base = compute_temporal_map(gt, clone_pred(), 30)
print(f"\nBASELINE repro: mAP={base['mAP']:.5f} AP03={base['AP@0.3']:.5f} AP05={base['AP@0.5']:.5f} AP07={base['AP@0.7']:.5f}")
print('(target repro: mAP=0.16155 AP03=0.21823 AP05=0.15005 AP07=0.11636)\n')

run(lambda c, df, pc, wr, im: c * (0.9 + 0.2 * df), 'M1 dur_fit [0.9,1.1]')
run(lambda c, df, pc, wr, im: c * (0.95 + 0.1 * pc), 'M2 peak_contain +-5%')
run(lambda c, df, pc, wr, im: c * (0.9 + 0.2 * wr), 'M3 wrank [0.9,1.1]')
run(lambda c, df, pc, wr, im: c * (0.9 + 0.2 * min(1.0, im / 0.9)), 'M4 in_mean/0.9 [0.9,1.1]')
run(lambda c, df, pc, wr, im: c * (0.9 + 0.2 * df) * (0.95 + 0.1 * pc), 'M5 dur_fit x peak_contain')
run(lambda c, df, pc, wr, im: c * (0.9 + 0.2 * df) * (0.9 + 0.2 * wr), 'M6 dur_fit x wrank')
run(lambda c, df, pc, wr, im: c * (0.9 + 0.2 * df) * (0.95 + 0.1 * pc) * (0.9 + 0.2 * wr), 'M7 df x pc x wr')
run(lambda c, df, pc, wr, im: c * (0.8 + 0.4 * df), 'M8 dur_fit strong [0.8,1.2]')
