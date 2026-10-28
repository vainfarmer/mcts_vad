"""v183 CLC — Cross-video Level Calibration pre-check (OFFLINE, zero eval spend).

Mechanism: the pipeline ranks ALL detections from ALL videos globally by
confidence for temporal mAP. CLC scales every detection confidence of video i
by a supervised video-level anomaly likelihood p_i:

    conf' = conf * (0.5 + 0.5 * p_i)

p_i comes from the pipeline's OWN MIL signal:
    s_i = mean of top-3 of the 32 snippet scores from model_mil.pth
    p_i = sigmoid(a * s_i + b),  (a, b) fitted by logistic regression on the
          TRAIN split ONLY (labels from filename: contains 'Normal' -> 0,
          else 1).

CLC is count-frozen and window-frozen (only conf changes), preserves
intra-video order, and does NOT normalize per-video (it sets cross-video
levels using a supervised signal — explicitly allowed).

MIL load idiom: copied from iteration_logs/analysis_v180_snhp_precheck.py
(MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6) +
model_mil.pth, eval mode, mps-or-cpu).

Metric: reused verbatim from video_anomaly_detection/evaluate.py
(load_ground_truth / load_predictions / compute_temporal_map, fps=30,
IoU [0.3, 0.5, 0.7], 11-point AP, global confidence ranking, per-video
greedy match). Baseline = committed full-290 run
results/batch_eval_v160_full (BUE+CBT+CBT-recal ON, everything else OFF),
report mAP 0.17254429208225996.

GO GATE (pre-registered):
  GO iff  full290 delta >= +0.005  AND  hard-88 delta >= +0.008.
Hard-88 slice: categories Shooting, RoadAccidents, Shoplifting, Explosion
(88 videos). Normal-FA (fraction of Normal videos with >=1 detection
conf>0) must be unchanged by construction — verified.

CAVEAT: cached .npy features are 32-uniform-snippet middle-frame CLIP
(extract_features.py), NOT pipeline-exact (fps=2 motion-filtered mean-pool)
— this precheck is a proxy; if GO is marginal (delta within 2x of a gate
threshold), a pipeline-exact feature recheck is required before
implementation.

Usage:
  python3.14 iteration_logs/analysis_v183_clc_precheck.py   (cwd=repo root)
Output: results/v183_clc_precheck.json
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import (  # noqa: E402
    load_ground_truth,
    load_predictions,
    compute_temporal_map,
)
from mil_model import MILAnomalyScorer  # noqa: E402

REPO = Path('/Users/zzl/PycharmProjects/vad')
FEAT_TRAIN = REPO / 'features' / 'train'
FEAT_TEST = REPO / 'features' / 'test'
RES_DIR = REPO / 'results' / 'batch_eval_v160_full'
GT_PATH = '/Users/zzl/Movies/ucf-crime/data'
OUT_JSON = REPO / 'results' / 'v183_clc_precheck.json'
MIL_PATH = Path(__file__).resolve().parent.parent / 'model_mil.pth'

REPORT_MAP = 0.17254429208225996  # committed evaluation_report.json value
SANITY_TOL = 0.001

HARD_CATS = ['Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion']
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
FPS = 30

GATE_D290 = 0.005
GATE_D88 = 0.008

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
BATCH = 64

CAVEAT = (
    'Cached .npy features are 32-uniform-snippet middle-frame CLIP '
    '(extract_features.py), NOT pipeline-exact (fps=2 motion-filtered '
    'mean-pool) — this precheck is a proxy; if GO is marginal (delta within '
    '2x of a gate threshold), a pipeline-exact feature recheck is required '
    'before implementation.'
)


# ---------------------------------------------------------------------------
# MIL signal
# ---------------------------------------------------------------------------

def load_mil():
    """Idiom copied verbatim from analysis_v180_snhp_precheck.py."""
    mil_model = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3,
                                 dropout=0.6)
    mil_model.load_state_dict(torch.load(str(MIL_PATH), map_location=DEV,
                                         weights_only=True))
    mil_model.eval().to(DEV)
    return mil_model


def mil_snippet_scores_batch(mil_model, feats):
    """feats: (N, 32, 768) float32 np array -> (N, 32) snippet scores."""
    out = []
    for i in range(0, len(feats), BATCH):
        t = torch.from_numpy(feats[i:i + BATCH]).to(DEV)
        with torch.no_grad():
            out.append(mil_model(t).cpu().numpy())
    return np.concatenate(out, axis=0)


def s_top3(scores32):
    """mean of top-3 of the 32 snippet scores."""
    return float(np.mean(np.sort(scores32)[-3:]))


def collect_s(mil_model, feat_dir):
    """One MIL forward per video -> {stem: s_i}. Deterministic order."""
    stems = sorted(p.stem for p in feat_dir.glob('*.npy'))
    feats = np.stack([np.load(feat_dir / f'{s}.npy').astype(np.float32)
                      for s in stems])
    scores = mil_snippet_scores_batch(mil_model, feats)  # (N, 32)
    return {s: s_top3(scores[i]) for i, s in enumerate(stems)}


def fit_logistic(train_s):
    """Fit p = sigmoid(a*s + b) on train split. Labels: 'Normal' in name -> 0."""
    stems = sorted(train_s.keys())
    X = np.array([train_s[s] for s in stems], dtype=np.float64).reshape(-1, 1)
    y = np.array([0 if 'Normal' in s else 1 for s in stems], dtype=np.int64)
    lr = LogisticRegression(random_state=0)
    lr.fit(X, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    p = 1.0 / (1.0 + np.exp(-(a * X.ravel() + b)))
    auc = float(roc_auc_score(y, p))
    print(f'[CLC] train fit: n={len(y)} pos={int(y.sum())} neg={int((1 - y).sum())} '
          f'a={a:.6f} b={b:.6f} train_auc={auc:.4f}', flush=True)
    return a, b, auc


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Evaluation helpers (metric functions imported from evaluate.py)
# ---------------------------------------------------------------------------

def matched_map(gt, pred):
    """evaluate() matching logic: intersect GT and prediction video sets."""
    matched = set(gt.keys()) & set(pred.keys())
    gt_m = {k: gt[k] for k in matched}
    pred_m = {k: pred[k] for k in matched}
    return compute_temporal_map(gt_m, pred_m, fps=FPS,
                                iou_thresholds=IOU_THRESHOLDS), matched


def clc_transform(pred, p_map):
    """Deep-copy pred; scale each detection conf by (0.5 + 0.5*p_i).

    Count-frozen, window-frozen: only 'confidence' changes. Videos without a
    fitted p_i are left unchanged.
    """
    out = copy.deepcopy(pred)
    n_scaled = 0
    for vid, entry in out.items():
        if vid not in p_map:
            continue
        f = 0.5 + 0.5 * p_map[vid]
        for anom in entry['anomalies']:
            anom['confidence'] = anom.get('confidence', 0.0) * f
        n_scaled += 1
    return out, n_scaled


def subset_by_cat(gt, pred, cats):
    vids = {k for k, v in gt.items() if v['category'] in cats}
    return ({k: gt[k] for k in vids}, {k: pred[k] for k in vids if k in pred})


def normal_fa_rate(gt, pred):
    """Fraction of Normal-category videos with >=1 detection conf>0."""
    n, fa = 0, 0
    for vid, g in gt.items():
        if g['category'] != 'Normal':
            continue
        n += 1
        entry = pred.get(vid)
        if entry and any(a.get('confidence', 0.0) > 0
                         for a in entry['anomalies']):
            fa += 1
    return (fa / n) if n else 0.0, fa, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    print(f'[CLC] device={DEV} mil={MIL_PATH.name}', flush=True)

    # --- MIL signal -------------------------------------------------------
    mil_model = load_mil()
    train_s = collect_s(mil_model, FEAT_TRAIN)
    test_s = collect_s(mil_model, FEAT_TEST)
    print(f'[CLC] s computed: train={len(train_s)} test={len(test_s)} '
          f'in {time.time() - t0:.1f}s', flush=True)

    a, b, train_auc = fit_logistic(train_s)
    p_map = {s: float(sigmoid(a * v + b)) for s, v in test_s.items()}
    pvals = np.array(list(p_map.values()))
    print(f'[CLC] test p_i: min={pvals.min():.4f} mean={pvals.mean():.4f} '
          f'max={pvals.max():.4f}', flush=True)

    # --- Load GT + baseline predictions -----------------------------------
    gt = load_ground_truth(GT_PATH, split='test')
    pred_base = load_predictions(str(RES_DIR))
    print(f'[CLC] gt={len(gt)} pred_files={len(pred_base)}', flush=True)

    base_res, matched = matched_map(gt, pred_base)
    baseline_map = base_res['mAP']
    print(f'[CLC] baseline mAP={baseline_map:.6f} (report {REPORT_MAP:.6f}) '
          f'matched={len(matched)}', flush=True)

    sanity_ok = abs(baseline_map - REPORT_MAP) <= SANITY_TOL
    if not sanity_ok:
        print(f'[CLC][FATAL] baseline mAP mismatch: recomputed={baseline_map:.6f} '
              f'vs report={REPORT_MAP:.6f} (|d|={abs(baseline_map - REPORT_MAP):.6f} '
              f'> {SANITY_TOL}). STOPPING.', flush=True)

    # --- CLC transform ------------------------------------------------------
    pred_clc, n_scaled = clc_transform(
        pred_base, {v: p_map[v] for v in matched if v in p_map})
    n_test_matched = sum(1 for v in matched if v in p_map)
    clc_res, _ = matched_map(gt, pred_clc)
    clc_map = clc_res['mAP']
    delta290 = clc_map - baseline_map
    print(f'[CLC] clc mAP={clc_map:.6f} delta290={delta290:+.6f} '
          f'(videos_scaled={n_scaled})', flush=True)

    # --- Hard-88 slice ------------------------------------------------------
    gt_m = {k: gt[k] for k in matched}
    pb_m = {k: pred_base[k] for k in matched}
    pc_m = {k: pred_clc[k] for k in matched}
    gt_h, pb_h = subset_by_cat(gt_m, pb_m, HARD_CATS)
    _, pc_h = subset_by_cat(gt_m, pc_m, HARD_CATS)
    print(f'[CLC] hard-88 slice: {len(gt_h)} videos', flush=True)
    base_h = compute_temporal_map(gt_h, pb_h, fps=FPS,
                                  iou_thresholds=IOU_THRESHOLDS)['mAP']
    clc_h = compute_temporal_map(gt_h, pc_h, fps=FPS,
                                 iou_thresholds=IOU_THRESHOLDS)['mAP']
    delta88 = clc_h - base_h
    print(f'[CLC] hard88 base={base_h:.6f} clc={clc_h:.6f} '
          f'delta88={delta88:+.6f}', flush=True)

    # --- Per-category deltas (4 hard cats) ----------------------------------
    per_category = {}
    for cat in HARD_CATS:
        g_c, b_c = subset_by_cat(gt_m, pb_m, [cat])
        _, c_c = subset_by_cat(gt_m, pc_m, [cat])
        mb = compute_temporal_map(g_c, b_c, fps=FPS,
                                  iou_thresholds=IOU_THRESHOLDS)['mAP']
        mc = compute_temporal_map(g_c, c_c, fps=FPS,
                                  iou_thresholds=IOU_THRESHOLDS)['mAP']
        per_category[cat] = {
            'n_videos': len(g_c),
            'baseline_map': float(mb),
            'clc_map': float(mc),
            'delta': float(mc - mb),
        }
        print(f'[CLC] cat={cat:15s} n={len(g_c):3d} base={mb:.6f} '
              f'clc={mc:.6f} delta={mc - mb:+.6f}', flush=True)

    # --- Normal-video FA rate -------------------------------------------------
    fa_base, fa_b_n, n_norm = normal_fa_rate(gt_m, pb_m)
    fa_clc, fa_c_n, _ = normal_fa_rate(gt_m, pc_m)
    print(f'[CLC] Normal FA: base={fa_base:.6f} ({fa_b_n}/{n_norm}) '
          f'clc={fa_clc:.6f} ({fa_c_n}/{n_norm})', flush=True)

    # --- GO gate --------------------------------------------------------------
    go = bool(delta290 >= GATE_D290 and delta88 >= GATE_D88 and sanity_ok)
    marginal = bool((delta290 < 2 * GATE_D290) or (delta88 < 2 * GATE_D88))
    print(f'[CLC][VERDICT] go={go} marginal={marginal} '
          f'(gates: d290>={GATE_D290}, d88>={GATE_D88})', flush=True)

    result = {
        'go': go,
        'marginal': marginal,
        'baseline_map': float(baseline_map),
        'clc_map': float(clc_map),
        'delta290': float(delta290),
        'baseline_map_hard88': float(base_h),
        'clc_map_hard88': float(clc_h),
        'delta88': float(delta88),
        'per_category': per_category,
        'normal_fa_base': float(fa_base),
        'normal_fa_clc': float(fa_clc),
        'normal_fa_base_videos': fa_b_n,
        'normal_fa_clc_videos': fa_c_n,
        'n_normal_videos': n_norm,
        'train_auc': float(train_auc),
        'a': a,
        'b': b,
        'n_test_matched': n_test_matched,
        'n_gt': len(gt),
        'n_matched': len(matched),
        'sanity_baseline_reproduced': sanity_ok,
        'baseline_map_report': REPORT_MAP,
        'baseline_dir': str(RES_DIR),
        'ap_detail': {
            'baseline': {k: float(v) for k, v in base_res.items()},
            'clc': {k: float(v) for k, v in clc_res.items()},
        },
        'gate': {'delta290_min': GATE_D290, 'delta88_min': GATE_D88},
        'caveat': CAVEAT,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[CLC] written {OUT_JSON} in {time.time() - t0:.1f}s', flush=True)

    if not sanity_ok:
        sys.exit(2)


if __name__ == '__main__':
    main()
