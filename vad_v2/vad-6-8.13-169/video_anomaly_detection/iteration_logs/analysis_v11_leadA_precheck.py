"""
v11 LEAD A pre-check (offline, zero API):
1. Reconfirm the cross-video-LEVEL oracle on the BUE-on 65 cache (keep intra-video
   order, set each video's LEVEL optimally) -> is it ~+0.0518?
2. Compute model_mil_v6 video-level scores (max & mean of 32-snippet curve) per video.
3. Measure whether the MIL video score correlates with the quantity the cross-video
   oracle rewards: per-video DETECTION IoU QUALITY (max/mean best-IoU of its dets).
4. Estimate REALIZED cross-video re-leveling by the MIL score (the actual mechanism).
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import (load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou)
from mil_model import MILAnomalyScorer

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data')
FEAT = ROOT / 'features' / 'test'
FPS = 30
CACHE = ROOT / 'results' / 'batch_eval_v144c_bue65ref'

def best_iou(a, gt_entry):
    return max((_temporal_iou(a, s, FPS) for s in gt_entry['segments']), default=0.0)

gt = load_ground_truth(LABEL_PATH, split='test')
pred = load_predictions(str(CACHE))
gt_eval = {k: v for k, v in gt.items() if k in pred}
base = compute_temporal_map(gt_eval, pred, fps=FPS)
print(f'BASE cached mAP={base["mAP"]:.5f}  AP@0.3={base["AP@0.3"]:.5f} AP@0.5={base["AP@0.5"]:.5f} AP@0.7={base["AP@0.7"]:.5f}')

def relevel(level_fn):
    # keep intra-video order via tiny epsilon on the ORIGINAL confidence rank
    p2 = {}
    for vid, entry in pred.items():
        L = level_fn(vid, entry)
        outs = []
        an = entry['anomalies']
        # rank within video by original conf (preserve intra order)
        order = np.argsort([-x.get('confidence', 0.0) for x in an])
        rankpos = {i: r for r, i in enumerate(order)}
        for i, a in enumerate(an):
            a2 = dict(a); a2['confidence'] = float(L) - 1e-4 * rankpos[i]
            outs.append(a2)
        p2[vid] = dict(entry, anomalies=outs)
    return compute_temporal_map(gt_eval, p2, fps=FPS)

# ---- cross-video LEVEL oracle: level = per-video MAX best-IoU of its detections ----
def vid_maxiou(vid, entry):
    if vid not in gt_eval: return 0.0
    return max((best_iou(a, gt_eval[vid]) for a in entry['anomalies']), default=0.0)
def vid_meaniou(vid, entry):
    if vid not in gt_eval or not entry['anomalies']: return 0.0
    return float(np.mean([best_iou(a, gt_eval[vid]) for a in entry['anomalies']]))

o_max = relevel(vid_maxiou)
o_mean = relevel(vid_meaniou)
print(f'ORACLE level=maxIoU   mAP={o_max["mAP"]:.5f}  (d={o_max["mAP"]-base["mAP"]:+.5f})')
print(f'ORACLE level=meanIoU  mAP={o_mean["mAP"]:.5f}  (d={o_mean["mAP"]-base["mAP"]:+.5f})')

# ---- MIL v6 video-level scores ----
dev = 'cpu'
model = MILAnomalyScorer()
sd = torch.load(ROOT/'video_anomaly_detection'/'model_mil_v6.pth', map_location=dev)
if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
model.load_state_dict(sd); model.eval()

mil_max, mil_mean = {}, {}
for vid in pred:
    fp = FEAT / f'{vid}.npy'
    if not fp.exists():
        mil_max[vid]=mil_mean[vid]=None; continue
    feat = np.load(fp).astype(np.float32)  # (32,768)
    with torch.no_grad():
        s = model(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).cpu().numpy()
    mil_max[vid]=float(s.max()); mil_mean[vid]=float(s.mean())

# ---- correlation: MIL video score vs per-video detection IoU quality ----
rows=[]
for vid, entry in pred.items():
    if mil_max[vid] is None or vid not in gt_eval: continue
    rows.append((vid, mil_max[vid], mil_mean[vid], vid_maxiou(vid,entry), vid_meaniou(vid,entry),
                 max((x.get('confidence',0.0) for x in entry['anomalies']), default=0.0),
                 gt_eval[vid].get('is_normal', False)))
arr = rows
mm = np.array([r[1] for r in arr]); qmax = np.array([r[3] for r in arr])
qmean = np.array([r[4] for r in arr]); curconf = np.array([r[5] for r in arr])
def sp(x,y):
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx,ry)[0,1])
print(f'\nn videos with feats+gt = {len(arr)}')
print(f'MIL_max vs per-video maxIoU  : pearson={np.corrcoef(mm,qmax)[0,1]:+.4f} spearman={sp(mm,qmax):+.4f}')
print(f'MIL_max vs per-video meanIoU : pearson={np.corrcoef(mm,qmean)[0,1]:+.4f} spearman={sp(mm,qmean):+.4f}')
print(f'curconf vs per-video maxIoU  : pearson={np.corrcoef(curconf,qmax)[0,1]:+.4f} spearman={sp(curconf,qmax):+.4f}')

# ---- REALIZED re-leveling by MIL score (the actual mechanism) ----
# blend current per-video level with MIL score at several weights
cur_level = {vid: max((x.get('confidence',0.0) for x in entry['anomalies']), default=0.0)
             for vid,entry in pred.items()}
# normalize mil to [0,1] comparable to conf range
mmall = np.array([mil_max[v] for v in pred if mil_max[v] is not None])
lo,hi = mmall.min(), mmall.max()
def milnorm(v):
    if mil_max[v] is None: return cur_level[v]
    return (mil_max[v]-lo)/(hi-lo+1e-9)
for w in [0.25,0.5,0.75,1.0]:
    def lvl(vid,entry,w=w):
        return (1-w)*cur_level[vid] + w*milnorm(vid)
    r = relevel(lvl)
    print(f'REALIZED MIL re-level w={w:.2f}  mAP={r["mAP"]:.5f}  (d={r["mAP"]-base["mAP"]:+.5f})')

# ---- bounded DOWN-ONLY MIL suppression (generalize the pipeline.py:1579 sanity gate) ----
print('\n-- bounded down-only MIL multiplier conf*=(a+(1-a)*milnorm), count-neutral, intra-order kept --')
for a in [0.5,0.7,0.85]:
    p2={}
    for vid,entry in pred.items():
        f = a + (1-a)*milnorm(vid)
        outs=[dict(x, confidence=x.get('confidence',0.0)*f) for x in entry['anomalies']]
        p2[vid]=dict(entry,anomalies=outs)
    r=compute_temporal_map(gt_eval,p2,fps=FPS)
    print(f'  a={a:.2f} mAP={r["mAP"]:.5f} (d={r["mAP"]-base["mAP"]:+.5f})')
# hard sanity-gate replica sweep: multiply by g if MIL_max < thr
print('-- hard MIL-gate: conf*=g if MIL_max<thr (v1579 is g=0.7,thr=0.3) --')
for thr,g in [(0.3,0.7),(0.4,0.7),(0.5,0.5),(0.3,0.5)]:
    p2={}
    for vid,entry in pred.items():
        f = g if (mil_max[vid] is not None and mil_max[vid]<thr) else 1.0
        outs=[dict(x, confidence=x.get('confidence',0.0)*f) for x in entry['anomalies']]
        p2[vid]=dict(entry,anomalies=outs)
    r=compute_temporal_map(gt_eval,p2,fps=FPS)
    nfire=sum(1 for v in pred if mil_max[v] is not None and mil_max[v]<thr)
    print(f'  thr={thr} g={g} fires={nfire} mAP={r["mAP"]:.5f} (d={r["mAP"]-base["mAP"]:+.5f})')
