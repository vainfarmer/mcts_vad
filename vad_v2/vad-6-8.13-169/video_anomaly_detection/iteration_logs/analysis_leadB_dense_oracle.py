"""LEAD B FM13 precheck: GT-aware vs verdict-independent run-IoU oracle on a
DENSER localizer substrate (per-frame fps=2 CLIP features, 93-678 frames/video),
for the 17 zero-IoU wrong-location videos.

Dense anomaly score curves tested (verdict-independent, pre-calibratable):
  - dense MIL v1 (model_mil.pth) per-frame
  - dense MIL v6 (model_mil_v6.pth) per-frame
  - motion energy (cached)
  - CLIP category-actionness (max cos-sim to the 13 anomaly-class text prompts)

For each curve & video:
  GT-aware oracle:   best IoU over a threshold-sweep of contiguous above-thr runs
                     (upper bound: can the dense curve REACH the GT at all?)
  verdict-indep:     single best run picked WITHOUT GT (half-max run around argmax)
Reports @IoU>=0.3 hit counts. Must beat ~0/17 verdict-independent to keep LEAD B open.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth

ROOT = Path('/Users/zzl/PycharmProjects/vad')
CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_keyframe_feats.pkl'

def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0

def runs_above(curve, thr):
    out = []; st = None
    for i, v in enumerate(curve):
        if v >= thr and st is None:
            st = i
        elif v < thr and st is not None:
            out.append((st, i)); st = None
    if st is not None:
        out.append((st, len(curve)))
    return out

def best_iou_gtaware(curve, ts, gt_t):
    """Sweep thresholds, take best-IoU run vs any GT segment (GT-AWARE upper bound)."""
    best = 0.0
    qs = np.percentile(curve, [50, 60, 70, 80, 85, 90, 93, 95, 97])
    for thr in qs:
        for (s, e) in runs_above(curve, thr):
            r0, r1 = ts[s], ts[min(e, len(ts)-1)]
            for (g0, g1) in gt_t:
                best = max(best, iou(r0, r1, g0, g1))
    return best

def best_iou_vindep(curve, ts, gt_t):
    """No-GT selector: half-max contiguous run containing the global argmax."""
    hm = (curve.max() + np.percentile(curve, 20)) / 2.0
    ai = int(np.argmax(curve))
    # grow run around argmax while >= hm
    s = ai
    while s > 0 and curve[s-1] >= hm:
        s -= 1
    e = ai
    while e < len(curve)-1 and curve[e+1] >= hm:
        e += 1
    r0, r1 = ts[s], ts[min(e, len(ts)-1)]
    return max((iou(r0, r1, g0, g1) for (g0, g1) in gt_t), default=0.0)

def main():
    gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
    cache = pickle.load(open(CACHE, 'rb'))
    from mil_model import MILAnomalyScorer
    def load_mil(p):
        m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
        m.load_state_dict(torch.load(p, map_location='cpu')); m.eval(); return m
    mil1 = load_mil(ROOT/'video_anomaly_detection/model_mil.pth')
    mil6 = load_mil(ROOT/'video_anomaly_detection/model_mil_v6.pth')

    def dense_mil(model, feats):
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s = model(x).squeeze(0).cpu().numpy()
        return s

    curves = ['mil1', 'mil6', 'motion']
    ga = {c: [] for c in curves}; vi = {c: [] for c in curves}
    per = []
    for vid in cache:
        if vid not in gt_all:
            continue
        e = cache[vid]
        feats = e['feats']; ts = np.asarray(e['ts']); motion = np.asarray(e['motion'])
        gt_t = [(a/30.0, b/30.0) for a, b in gt_all[vid]['segments']]
        if not gt_t or len(ts) < 4:
            continue
        cur = {'mil1': dense_mil(mil1, feats), 'mil6': dense_mil(mil6, feats),
               'motion': motion.astype(np.float32)}
        row = {'vid': vid, 'n': len(ts)}
        for c in curves:
            g = best_iou_gtaware(cur[c], ts, gt_t)
            v = best_iou_vindep(cur[c], ts, gt_t)
            ga[c].append(g); vi[c].append(v)
            row[c] = (round(g, 2), round(v, 2))
        per.append(row)

    n = len(per)
    print(f"n videos = {n}")
    for c in curves:
        gh = sum(x >= 0.3 for x in ga[c]); vh = sum(x >= 0.3 for x in vi[c])
        gh5 = sum(x >= 0.5 for x in ga[c]); vh5 = sum(x >= 0.5 for x in vi[c])
        print(f"  {c:7s} GT-aware @0.3 {gh:2d}/{n}  @0.5 {gh5:2d}/{n} | "
              f"V-INDEP @0.3 {vh:2d}/{n}  @0.5 {vh5:2d}/{n} | "
              f"medIoU vindep {np.median(vi[c]):.2f}")
    print("\nper-video (gtaware, vindep):")
    for r in per:
        print(f"  {r['vid']:22s} n={r['n']:4d} mil1={r['mil1']} mil6={r['mil6']} motion={r['motion']}")

if __name__ == '__main__':
    main()
