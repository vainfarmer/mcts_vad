"""
v138 ORCHESTRATOR RANKING DIAGNOSTIC (offline, zero API calls).

Question this answers, before any v138 proposal is gated:
  How much of the current full-eval mAP is actually PRODUCED by the global
  confidence ranking (evaluate.py:342 sorts ALL predictions globally), and how
  informative is the current coarse_score-monotone confidence formula
  (pipeline.py:2236-2239) as that ranking key?

Arms (identical detection set / boundaries / count -- pure re-ranking, the
historically SAFE class, v111 Q-score lineage):
  A. cached      : confidences exactly as saved  -> must reproduce the run's mAP
  B. constant    : every confidence = 1.0        -> global ranking destroyed
  C. per-video   : rank only within video (video-index-major ordering)
  D. oracle      : confidence = IoU with GT      -> upper bound of re-ranking
Plus: Spearman/Pearson correlation between cached confidence and per-detection
best IoU vs GT, and confidence-value resolution (# distinct values / ties),
because a heavily quantized key cannot express a good global order.

Usage: python3.14 analysis_v138_rankdiag_orch.py [result_dir ...]
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import (  # noqa: E402
    load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou,
)

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data')
FPS = 30


def _best_iou(anomaly, gt_entry):
    best = 0.0
    for seg in gt_entry['segments']:
        best = max(best, _temporal_iou(anomaly, seg, FPS))
    return best


def run(result_dir):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(result_dir))
    # restrict GT to evaluated videos the same way evaluate() does
    gt_eval = {k: v for k, v in gt.items() if k in pred}
    n_det = sum(len(p['anomalies']) for p in pred.values())
    print(f'\n=== {result_dir.name} : {len(pred)} videos / {n_det} detections ===')

    base = compute_temporal_map(gt_eval, pred, fps=FPS)
    print(f'A cached      mAP={base["mAP"]:.5f}  ' +
          '  '.join(f'{k}={v:.5f}' for k, v in base.items() if k != 'mAP'))

    def with_conf(fn):
        p2 = {}
        for vid, entry in pred.items():
            outs = []
            for i, a in enumerate(entry['anomalies']):
                a2 = dict(a)
                a2['confidence'] = fn(vid, i, a)
                outs.append(a2)
            p2[vid] = dict(entry, anomalies=outs)
        return p2

    arms = {
        'B constant   ': with_conf(lambda v, i, a: 1.0),
        'C per-video  ': with_conf(lambda v, i, a: 1.0 - 0.001 * i),
        'D oracle-IoU ': with_conf(
            lambda v, i, a: _best_iou(a, gt_eval[v]) if v in gt_eval else 0.0),
    }
    for name, p2 in arms.items():
        r = compute_temporal_map(gt_eval, p2, fps=FPS)
        print(f'{name} mAP={r["mAP"]:.5f}  (d={r["mAP"] - base["mAP"]:+.5f})  ' +
              '  '.join(f'{k}={v:.5f}' for k, v in r.items() if k != 'mAP'))

    # informativeness of the cached key
    confs, ious = [], []
    for vid, entry in pred.items():
        if vid not in gt_eval:
            continue
        for a in entry['anomalies']:
            confs.append(float(a.get('confidence', 0.0)))
            ious.append(_best_iou(a, gt_eval[vid]))
    confs, ious = np.array(confs), np.array(ious)
    if len(confs) > 2:
        def spearman(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])
        uniq = np.unique(np.round(confs, 4))
        print(f'    conf-vs-IoU  pearson={np.corrcoef(confs, ious)[0, 1]:+.4f}'
              f'  spearman={spearman(confs, ious):+.4f}')
        print(f'    conf resolution: {len(uniq)} distinct / {len(confs)} dets'
              f'  (max tie-block={int(np.max(np.unique(np.round(confs, 4), return_counts=True)[1]))})'
              f'  range=[{confs.min():.4f},{confs.max():.4f}]')
        hit = ious >= 0.5
        print(f'    dets with IoU>=0.5: {int(hit.sum())}/{len(ious)}'
              f'  mean conf hit={confs[hit].mean() if hit.any() else float("nan"):.4f}'
              f'  vs miss={confs[~hit].mean() if (~hit).any() else float("nan"):.4f}')


if __name__ == '__main__':
    dirs = sys.argv[1:] or [str(ROOT / 'results' / 'batch_eval_v132_full')]
    for d in dirs:
        run(Path(d))
