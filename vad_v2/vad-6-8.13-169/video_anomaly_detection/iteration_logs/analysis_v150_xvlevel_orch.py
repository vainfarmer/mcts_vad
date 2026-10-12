"""
v150 LEAD-A ORACLE PRE-CHECK (offline, zero API).
Q1: reconfirm the CROSS-VIDEO-LEVEL oracle on the current BUE-on cache.
Q2: how much does a REAL model_mil_v6 video-level prior recover?
Q3: is the MIL video score monotone with per-video best-detection IoU (the
    thing global AP actually rewards)?
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import (load_ground_truth, load_predictions, compute_temporal_map,
                      _temporal_iou)
from mil_model import MILAnomalyScorer

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data')
FEAT = ROOT / 'features' / 'test'
FPS = 30
REF = ROOT / 'results' / 'batch_eval_v144c_bue65ref'


def best_iou(a, gt_entry):
    return max((_temporal_iou(a, s, FPS) for s in gt_entry['segments']), default=0.0)


def mil_video_scores(vids, model_path):
    dev = 'cpu'
    m = MILAnomalyScorer().to(dev)
    sd = torch.load(model_path, map_location=dev)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m.load_state_dict(sd)
    m.eval()
    out = {}
    with torch.no_grad():
        for v in vids:
            fp = FEAT / f'{v}.npy'
            if not fp.exists():
                out[v] = None; continue
            x = np.load(fp).astype(np.float32)
            if x.ndim == 2:
                x = x[None]
            s = m(torch.from_numpy(x).to(dev)).squeeze(0).cpu().numpy()
            out[v] = s  # (32,)
    return out


def main():
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(REF))
    gt_eval = {k: v for k, v in gt.items() if k in pred}
    base = compute_temporal_map(gt_eval, pred, fps=FPS)
    print(f'BASE mAP={base["mAP"]:.5f} AP@0.3={base["AP@0.3"]:.5f} '
          f'AP@0.5={base["AP@0.5"]:.5f} AP@0.7={base["AP@0.7"]:.5f}')

    # per-video: intra order (by original conf), best iou per det, max/mean iou
    vinfo = {}
    for v, e in pred.items():
        an = e['anomalies']
        if not an:
            vinfo[v] = dict(n=0); continue
        confs = np.array([float(a.get('confidence', 0.0)) for a in an])
        ious = np.array([best_iou(a, gt_eval[v]) if v in gt_eval else 0.0 for a in an])
        vinfo[v] = dict(n=len(an), confs=confs, ious=ious,
                        maxiou=float(ious.max()), meaniou=float(ious.mean()))

    def relevel(level_fn):
        p2 = {}
        for v, e in pred.items():
            an = e['anomalies']
            outs = []
            info = vinfo[v]
            if info['n'] == 0:
                p2[v] = dict(e, anomalies=[]); continue
            L = level_fn(v, info)
            # keep intra order: rank within video, tiny epsilon step below level
            order = np.argsort(-info['confs'])  # descending
            rank = np.empty(info['n'], dtype=float)
            rank[order] = np.arange(info['n'])
            for i, a in enumerate(an):
                a2 = dict(a)
                a2['confidence'] = float(L - 1e-4 * rank[i])
                outs.append(a2)
            p2[v] = dict(e, anomalies=outs)
        return p2

    def score(name, p2):
        r = compute_temporal_map(gt_eval, p2, fps=FPS)
        print(f'{name:38s} mAP={r["mAP"]:.5f} (d={r["mAP"]-base["mAP"]:+.5f}) '
              f'AP@0.5={r["AP@0.5"]:.5f}')
        return r["mAP"] - base["mAP"]

    # ORACLE level definitions (keep intra order)
    score('ORACLE level=maxIoU', relevel(lambda v, i: i['maxiou']))
    score('ORACLE level=meanIoU', relevel(lambda v, i: i['meaniou']))

    # ---- REAL MIL v6 prior ----
    vids = [v for v in pred if vinfo[v]['n'] > 0]
    mv6 = mil_video_scores(vids, ROOT / 'video_anomaly_detection' / 'model_mil_v6.pth')
    mv1 = mil_video_scores(vids, ROOT / 'video_anomaly_detection' / 'model_mil.pth')

    def prior_stats(mil, tag):
        smax = {v: (float(np.max(mil[v])) if mil[v] is not None else None) for v in vids}
        smean = {v: (float(np.mean(mil[v])) if mil[v] is not None else None) for v in vids}
        # monotonicity vs per-video maxIoU (what level oracle rewards)
        vv = [v for v in vids if smax[v] is not None]
        mx = np.array([smax[v] for v in vv]); mn = np.array([smean[v] for v in vv])
        oi = np.array([vinfo[v]['maxiou'] for v in vv])
        def sp(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])
        print(f'  [{tag}] spearman(MILmax, maxIoU)={sp(mx,oi):+.4f}  '
              f'spearman(MILmean, maxIoU)={sp(mn,oi):+.4f}')
        return smax, smean

    for tag, mil in [('v6', mv6), ('v1', mv1)]:
        smax, smean = prior_stats(mil, tag)
        # re-level by MIL prior (rank among videos -> level), keep intra order
        def lvl_from(prior):
            vals = {v: prior[v] for v in vids if prior[v] is not None}
            arr = np.array(list(vals.values()))
            def fn(v, i):
                if v not in vals:
                    return float(np.median(arr))
                return float(vals[v])
            return fn
        score(f'REAL prior {tag} MILmax level', relevel(lvl_from(smax)))
        score(f'REAL prior {tag} MILmean level', relevel(lvl_from(smean)))
        # blended: keep original band center, nudge by prior (bounded)
        def blend(prior, w):
            vals = {v: prior[v] for v in vids if prior[v] is not None}
            arr = np.array(list(vals.values()))
            lo, hi = arr.min(), arr.max()
            def fn(v, i):
                base_c = float(np.mean(i['confs']))
                if v not in vals or hi <= lo:
                    return base_c
                p = (vals[v]-lo)/(hi-lo)
                return base_c * (1 - w + w * p)
            return fn
        score(f'REAL prior {tag} blend w=0.5', relevel(blend(smax, 0.5)))


if __name__ == '__main__':
    main()
