"""Agent C: does BUE@Lmax=40 clear the MANDATED P(delta>0)>=0.90 gate?
Uses the OFFICIAL bootstrap function from analysis_bootstrap_ab.py at its
default n=1000, on the paired offline arms (exact, zero VLM noise).
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map
from analysis_bootstrap_ab import bootstrap, LABEL_PATH, FPS
from analysis_v142_C_bootstrap import apply_bue

KW = dict(alpha=1.0, gate=True, union=True, Lmax=40.0)

gt_all = load_ground_truth(LABEL_PATH, split='test')
rows = []
for d in [Path(x) for x in sys.argv[1:]]:
    pred = load_predictions(str(d))
    gt = {k: v for k, v in gt_all.items() if k in pred and gt_all[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}
    base = compute_temporal_map(gt, pred, fps=FPS)['mAP']
    p2 = apply_bue(pred, **KW)
    pt = compute_temporal_map(gt, p2, fps=FPS)['mAP'] - base
    ds, vids = bootstrap(gt, pred, p2, n=1000, seed=0)
    lo, hi = np.percentile(ds, [5, 95])
    P = float((ds > 0).mean())
    ok = 'PASS' if (pt >= 0.01 and P >= 0.90) else 'FAIL'
    print(f'{d.name:26s} base={base:.5f} point={pt:+.5f} '
          f'90%CI=[{lo:+.5f},{hi:+.5f}] hw={(hi-lo)/2:.4f} P(d>0)={P:.3f}  gate={ok}', flush=True)
    rows.append((d.name, pt, P, ok))
print('\n=== SUMMARY ===')
ps = [r[2] for r in rows]
print(f'n_caches={len(rows)}  mean point={np.mean([r[1] for r in rows]):+.5f}  '
      f'mean P={np.mean(ps):.3f}  P>=0.90 in {sum(1 for p in ps if p>=0.90)}/{len(ps)} caches')
