"""v146 pivot: MIL-support SCORE re-rank proxy (no window move, no add, no CLIP curve).
For each cached FINAL detection compute v1-MIL support at its window (raw sigmoid mean over
overlapping snippets, reproduced from cached LIVE clip_features via model_mil.pth), then re-score
confidence by several fusions and recompute temporal mAP. Baseline = cached anomalies mAP.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import torch
ROOT = Path('/Users/zzl/PycharmProjects/vad'); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
from mil_model import MILAnomalyScorer
LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30; THR = [0.3, 0.5, 0.7]
BASE_DIR = ROOT / 'results/batch_eval_v144c_bue65ref'; NPZ = HERE / 'analysis_v140_livecurves'
_mil = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
_mil.load_state_dict(torch.load(ROOT / 'video_anomaly_detection/model_mil.pth', map_location='cpu'))
_mil.eval()
_MC = {}

def mil_for(vid):
    if vid in _MC: return _MC[vid]
    f = NPZ / f'{vid}.npz'
    if not f.exists(): _MC[vid] = (None, None); return _MC[vid]
    z = np.load(f, allow_pickle=True); cf = np.asarray(z['clip_features'], np.float32); ts = np.asarray(z['timestamps'], np.float64)
    n = cf.shape[0]; NS = 32
    if n < NS: _MC[vid] = (None, None); return _MC[vid]
    seg = n // NS; feats, st = [], []
    for si in range(NS):
        a = si*seg; b = a+seg if si < NS-1 else n
        feats.append(cf[a:b].mean(0)); st.append((float(ts[a]), float(ts[b-1])))
    with torch.no_grad():
        s = _mil(torch.from_numpy(np.stack(feats)).unsqueeze(0)).squeeze(0).numpy().astype(np.float64)
    _MC[vid] = (s, st); return _MC[vid]

def support(s, st, ds, de):
    if s is None: return None
    ov = [s[i] for i,(a,b) in enumerate(st) if min(b,de) > max(a,ds)]
    if not ov:  # window between snippet centers: nearest snippet
        c = 0.5*(ds+de); i = int(np.argmin([abs(0.5*(a+b)-c) for a,b in st])); return float(s[i])
    return float(np.mean(ov))

def mil_gap(s, st, ds, de):
    """seconds from window center to the MIL argmax-snippet center (displacement)."""
    if s is None: return None
    amax = int(np.argmax(s)); pc = 0.5*(st[amax][0]+st[amax][1]); dc = 0.5*(ds+de)
    return abs(pc - dc)

def evalp(gt, pred):
    return compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)

def apply_rerank(pred, fn):
    out = {}
    for v, e in pred.items():
        s, st = mil_for(v); na = []
        for a in e['anomalies']:
            b = dict(a); sup = support(s, st, a['t_start'], a['t_end'])
            if sup is not None:
                b['confidence'] = float(fn(a['confidence'], sup))
            na.append(b)
        out[v] = dict(e, anomalies=na)
    return out

def main():
    gt = load_ground_truth(LABEL_PATH, split='test'); pred = load_predictions(str(BASE_DIR))
    gt = {k:v for k,v in gt.items() if k in pred and gt[k]['segments']}; pred = {k:v for k,v in pred.items() if k in gt}
    base = evalp(gt, pred)
    print(f'BASELINE cached mAP={base["mAP"]:.5f} .3={base["AP@0.3"]:.4f} .5={base["AP@0.5"]:.4f} .7={base["AP@0.7"]:.4f}')
    # coverage of MIL support
    sups = [support(*mil_for(v), a['t_start'], a['t_end']) for v,e in pred.items() for a in e['anomalies']]
    sv = [x for x in sups if x is not None]
    print(f'MIL support over {len(sv)} dets: mean={np.mean(sv):.3f} med={np.median(sv):.3f} p10={np.percentile(sv,10):.3f} p90={np.percentile(sv,90):.3f}')
    schemes = {
        'A pure MIL':          lambda c,s: s,
        'B conf*sup':          lambda c,s: c*s,
        'C conf*(.5+.5sup)':   lambda c,s: c*(0.5+0.5*s),
        'D conf+0.3(sup-.5)':  lambda c,s: c+0.3*(s-0.5),
        'E conf+0.5(sup-.5)':  lambda c,s: c+0.5*(s-0.5),
        'F conf*(.3+.7sup)':   lambda c,s: c*(0.3+0.7*s),
        'G penalize low(<p25)':lambda c,s: c if s>=np.percentile(sv,25) else c*(s/np.percentile(sv,25)),
        'H conf+1.0(sup-.5)':  lambda c,s: c+1.0*(s-0.5),
    }
    for nm, fn in schemes.items():
        m = evalp(gt, apply_rerank(pred, fn))
        print(f'  {nm:22s} dmAP={m["mAP"]-base["mAP"]:+.5f} mAP={m["mAP"]:.5f} '
              f'.3={m["AP@0.3"]:.4f} .5={m["AP@0.5"]:.4f} .7={m["AP@0.7"]:.4f}')
    # ---- displacement-based re-rank (down-rank large MIL-gap dets = likely mislocated FP) ----
    gaps = [mil_gap(*mil_for(v), a['t_start'], a['t_end']) for v,e in pred.items() for a in e['anomalies']]
    gv = [x for x in gaps if x is not None]
    print(f'\nMIL gap(s) over {len(gv)} dets: mean={np.mean(gv):.1f} med={np.median(gv):.1f} p25={np.percentile(gv,25):.1f} p75={np.percentile(gv,75):.1f} max={np.max(gv):.1f}')
    def rr_gap(fn):
        out={}
        for v,e in pred.items():
            s,st=mil_for(v); na=[]
            for a in e['anomalies']:
                b=dict(a); g=mil_gap(s,st,a['t_start'],a['t_end'])
                if g is not None: b['confidence']=float(fn(a['confidence'],g))
                na.append(b)
            out[v]=dict(e,anomalies=na)
        return out
    for tau in [8.0, 15.0, 25.0]:
        m=evalp(gt, rr_gap(lambda c,g,tau=tau: c*np.exp(-g/tau)))
        print(f'  GAP conf*exp(-g/{tau:>4}) dmAP={m["mAP"]-base["mAP"]:+.5f} mAP={m["mAP"]:.5f} .3={m["AP@0.3"]:.4f} .5={m["AP@0.5"]:.4f} .7={m["AP@0.7"]:.4f}')
    for thr in [10.0, 20.0]:
        m=evalp(gt, rr_gap(lambda c,g,thr=thr: c if g<=thr else c*0.5))
        print(f'  GAP halve if>{thr:>4}s     dmAP={m["mAP"]-base["mAP"]:+.5f} mAP={m["mAP"]:.5f} .3={m["AP@0.3"]:.4f} .5={m["AP@0.5"]:.4f} .7={m["AP@0.7"]:.4f}')

if __name__ == '__main__':
    main()
