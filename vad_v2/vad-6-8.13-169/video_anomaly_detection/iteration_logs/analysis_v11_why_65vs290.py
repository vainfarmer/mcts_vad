"""Why does the 65-video harness score 0.219 but the full 290 scores 0.151?
Decompose the SAME 290-video BUE run over subsets — no VLM-noise confound."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad'); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map
from batch_evaluate import select_videos
from analysis_bootstrap_ab import LABEL_PATH, FPS

D = ROOT / 'results/batch_eval_full290_bue'
gt_all = load_ground_truth(LABEL_PATH, split='test')
pred = load_predictions(str(D))
common = [k for k in pred if k in gt_all]

def cat(k):  # category from filename
    import re; m = re.match(r'([A-Za-z]+)', k); return m.group(1) if m else '?'
def is_norm(k): return len(gt_all[k].get('segments', [])) == 0

# the 65-harness video set (--max_per_category 5)
h65 = select_videos('/Users/zzl/Movies/ucf-crime', 5, 'test')
h65_ids = set()
for r in h65:
    vp = r['video_path'] if isinstance(r, dict) else r
    h65_ids.add(Path(str(vp)).stem)
# match to prediction keys
key_by_stem = {Path(k).stem: k for k in common}
sub65 = [key_by_stem[s] for s in h65_ids if s in key_by_stem]

def mAP(keys):
    g = {k: gt_all[k] for k in keys}; p = {k: pred[k] for k in keys}
    r = compute_temporal_map(g, p, fps=FPS); return r['mAP'], r.get('AP@0.3',-1), r.get('AP@0.5',-1)

def show(name, keys):
    m, a3, a5 = mAP(keys); n_norm = sum(1 for k in keys if is_norm(k))
    print(f'{name:38s} n={len(keys):3d} (norm {n_norm:3d}) mAP={m:.5f} AP@0.3={a3:.5f} AP@0.5={a5:.5f}')

print("=== SAME 290 run, different subsets ===")
show('FULL 290', common)
show('290 anomaly-only (drop Normal)', [k for k in common if not is_norm(k)])
show('65-harness subset (max_per_cat 5)', sub65)
show('65-harness anomaly-only', [k for k in sub65 if not is_norm(k)])

print("\n=== per-category AP on the 290 run (anomaly cats, sorted by count) ===")
cats = {}
for k in common:
    if is_norm(k): continue
    cats.setdefault(cat(k), []).append(k)
rows = []
for c, ks in cats.items():
    m, a3, a5 = mAP(ks)
    rows.append((len(ks), c, m))
for n, c, m in sorted(rows, reverse=True):
    in65 = sum(1 for k in cats[c] if k in set(sub65))
    print(f'  {c:14s} 290-count={n:2d}  (in 65-harness: {in65})  cat-mAP={m:.4f}')
