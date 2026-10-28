"""Quantify the Normal-video false-positive problem on the 290-video BUE run.
Oracle = zero out predictions on Normal (empty-GT) videos, recompute mAP => headroom ceiling."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map
from analysis_bootstrap_ab import LABEL_PATH, FPS

D = ROOT / 'results/batch_eval_full290_bue'
gt_all = load_ground_truth(LABEL_PATH, split='test')
pred = load_predictions(str(D))
common = [k for k in pred if k in gt_all]
print(f'common videos: {len(common)}')

# classify Normal (empty GT) vs anomaly (>=1 GT segment)
def is_normal(k):
    segs = gt_all[k].get('segments', [])
    return len(segs) == 0
normals = [k for k in common if is_normal(k)]
anoms   = [k for k in common if not is_normal(k)]
print(f'Normal (empty-GT) videos: {len(normals)} | anomaly videos: {len(anoms)}')

# how many Normal videos fire >=1 detection, confidence stats
nfire = 0; nconf = []
for k in normals:
    dets = pred[k].get('anomalies', pred[k]) if isinstance(pred[k], dict) else pred[k]
    segs = pred[k].get('segments') if isinstance(pred[k], dict) else None
    plist = pred[k]['predictions'] if isinstance(pred[k], dict) and 'predictions' in pred[k] else None
    # be robust to structure
    cand = None
    for key in ('predictions','anomalies','segments'):
        if isinstance(pred[k], dict) and key in pred[k] and isinstance(pred[k][key], list):
            cand = pred[k][key]; break
    if cand is None: cand = []
    confs = [c.get('confidence', c.get('score', 0)) for c in cand if isinstance(c, dict)]
    if confs:
        nfire += 1; nconf.extend(confs)
print(f'Normal videos with >=1 detection: {nfire}/{len(normals)} ({100*nfire/max(1,len(normals)):.1f}%)')
if nconf:
    a = np.array(nconf)
    print(f'Normal-video detection confidences: n={len(a)} mean={a.mean():.3f} median={np.median(a):.3f} max={a.max():.3f} | >=0.5: {(a>=0.5).sum()} >=0.7: {(a>=0.7).sum()} >=0.9: {(a>=0.9).sum()}')

# mAP baseline vs oracle (drop all Normal-video predictions)
gt = {k: gt_all[k] for k in common}
base = compute_temporal_map(gt, {k: pred[k] for k in common}, fps=FPS)
pred_drop = {}
for k in common:
    if is_normal(k):
        p = dict(pred[k]) if isinstance(pred[k], dict) else {}
        for key in ('predictions','anomalies','segments'):
            if key in p and isinstance(p[key], list): p[key] = []
        pred_drop[k] = p
    else:
        pred_drop[k] = pred[k]
orc = compute_temporal_map(gt, pred_drop, fps=FPS)
print()
print(f'BASELINE (BUE, all preds)      mAP={base["mAP"]:.5f}  AP@0.3={base.get("AP@0.3",-1):.5f} AP@0.5={base.get("AP@0.5",-1):.5f} AP@0.7={base.get("AP@0.7",-1):.5f}')
print(f'ORACLE (drop Normal-video FPs) mAP={orc["mAP"]:.5f}  AP@0.3={orc.get("AP@0.3",-1):.5f} AP@0.5={orc.get("AP@0.5",-1):.5f} AP@0.7={orc.get("AP@0.7",-1):.5f}')
print(f'==> HEADROOM from perfect Normal suppression: dmAP = {orc["mAP"]-base["mAP"]:+.5f}')
