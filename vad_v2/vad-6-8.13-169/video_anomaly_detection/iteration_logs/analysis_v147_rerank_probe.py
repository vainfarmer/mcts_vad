"""v147 offline RE-RANK probe (zero eval cost).
Tests whether a verdict-independent SUPERVISED MIL window-score can re-order the
GLOBAL confidence ranking to raise temporal mAP, without moving windows or changing count.
All arms use the SAME cached FINAL anomaly windows; only the confidence *ordering* changes.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
from mil_model import MILAnomalyScorer

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]
BASE_DIR = ROOT / 'results/batch_eval_v144c_bue65ref'
NPZ = HERE / 'analysis_v140_livecurves'

def load_mil(path, **kw):
    m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, dropout=0.6, **kw)
    m.load_state_dict(torch.load(path, map_location='cpu')); m.eval()
    return m
_v1 = load_mil(ROOT/'video_anomaly_detection/model_mil.pth', k=3)
try:
    _v6 = load_mil(ROOT/'video_anomaly_detection/model_mil_v6.pth', k=3)
except Exception as ex:
    print('v6 load fail', ex); _v6 = None

def mil_for(vid, model):
    f = NPZ / f'{vid}.npz'
    if not f.exists(): return None, None
    z = np.load(f, allow_pickle=True)
    cf = np.asarray(z['clip_features'], np.float32); ts = np.asarray(z['timestamps'], np.float64)
    n = cf.shape[0]; NS = 32
    if n < NS: return None, None
    seg = n // NS; feats, snip_t = [], []
    for si in range(NS):
        a = si*seg; b = a+seg if si < NS-1 else n
        feats.append(cf[a:b].mean(0)); snip_t.append((float(ts[a]), float(ts[b-1])))
    with torch.no_grad():
        s = model(torch.from_numpy(np.stack(feats)).unsqueeze(0)).squeeze(0).cpu().numpy()
    return s.astype(np.float64), snip_t

def win_mil(ds, de, s, snip_t):
    if s is None: return None
    idx = [i for i,(a,b) in enumerate(snip_t) if b >= ds and a <= de]
    if not idx:
        c = 0.5*(ds+de); idx = [int(np.argmin([abs(0.5*(a+b)-c) for a,b in snip_t]))]
    return float(np.mean(s[idx]))

def best_iou(seg, segs):
    return max([_temporal_iou(seg, g, FPS) for g in segs] or [0.0])

def mapof(gt, pred): return compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)

def build(gt, recs, key):
    pred = {}
    for v, items in recs.items():
        pred[v] = {'anomalies': [{'t_start': it['ds'], 't_end': it['de'],
                    'confidence': it[key], 'anomaly_type': it.get('at','unknown')} for it in items]}
    return pred

def main():
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(BASE_DIR))
    gt = {k:v for k,v in gt.items() if k in pred and gt[k]['segments']}
    recs = {}
    tp_v1, fp_v1, tp_conf, fp_conf = [], [], [], []
    for v in gt:
        segs = gt[v]['segments']
        s1, st1 = mil_for(v, _v1)
        s6, st6 = mil_for(v, _v6) if _v6 else (None, None)
        items = []
        for a in pred[v]['anomalies']:
            ds, de = float(a['t_start']), float(a['t_end']); conf = float(a['confidence'])
            m1 = win_mil(ds, de, s1, st1); m6 = win_mil(ds, de, s6, st6)
            iou = best_iou({'t_start': ds, 't_end': de}, segs)
            it = {'ds': ds, 'de': de, 'conf': conf, 'at': a.get('anomaly_type','unknown'),
                  'm1': m1 if m1 is not None else 0.0, 'm6': m6 if m6 is not None else 0.0,
                  'iou': iou, 'or18': -iou}  # or18: oracle key = -iou so higher iou ranks first via conf-desc? handled below
            items.append(it)
            (tp_v1 if iou>=0.3 else fp_v1).append(it['m1'])
            (tp_conf if iou>=0.3 else fp_conf).append(conf)
        recs[v] = items
    # discrimination
    import numpy as _n
    print('DISCRIMINATION (TP=best_iou>=0.3 vs FP):')
    print(f'  v1 win-MIL  TP mean={_n.mean(tp_v1):.4f} FP mean={_n.mean(fp_v1):.4f}  n_tp={len(tp_v1)} n_fp={len(fp_v1)}')
    print(f'  conf        TP mean={_n.mean(tp_conf):.4f} FP mean={_n.mean(fp_conf):.4f}')
    # normalize m1,m6 globally to conf-like scale for blends
    allm1 = _n.array([it['m1'] for v in recs for it in recs[v]])
    allm6 = _n.array([it['m6'] for v in recs for it in recs[v]])
    def z(x, arr): return (x-arr.mean())/(arr.std()+1e-9)
    for v in recs:
        for it in recs[v]:
            it['zm1'] = z(it['m1'], allm1); it['zm6'] = z(it['m6'], allm6)
            it['oracle'] = it['iou']  # rank by iou desc
    base = mapof(gt, build(gt, recs, 'conf'))
    orc  = mapof(gt, build(gt, recs, 'oracle'))
    m1r  = mapof(gt, build(gt, recs, 'm1'))
    print(f"\nBASELINE (conf)        mAP={base['mAP']:.5f} .3={base['AP@0.3']:.4f} .5={base['AP@0.5']:.4f} .7={base['AP@0.7']:.4f}")
    print(f"ORACLE (rank by IoU)   mAP={orc['mAP']:.5f} .3={orc['AP@0.3']:.4f} .5={orc['AP@0.5']:.4f} .7={orc['AP@0.7']:.4f}")
    print(f"PURE v1 win-MIL rank   mAP={m1r['mAP']:.5f} .3={m1r['AP@0.3']:.4f} .5={m1r['AP@0.5']:.4f} .7={m1r['AP@0.7']:.4f}")
    if _v6 is not None:
        m6r = mapof(gt, build(gt, recs, 'm6'))
        print(f"PURE v6 win-MIL rank   mAP={m6r['mAP']:.5f} .3={m6r['AP@0.3']:.4f} .5={m6r['AP@0.5']:.4f} .7={m6r['AP@0.7']:.4f}")
    # blends: new_conf = conf + lam * zm  (preserves cross-video level of conf, nudges order)
    for lam in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for tag, zk in [('v1','zm1')] + ([('v6','zm6')] if _v6 is not None else []):
            for v in recs:
                for it in recs[v]:
                    it['blend'] = it['conf'] + lam*it[zk]
            b = mapof(gt, build(gt, recs, 'blend'))
            print(f"  BLEND conf+{lam}*z({tag})  mAP={b['mAP']:.5f} d={b['mAP']-base['mAP']:+.5f} "
                  f".3={b['AP@0.3']:.4f} .5={b['AP@0.5']:.4f} .7={b['AP@0.7']:.4f}")

if __name__ == '__main__':
    main()
