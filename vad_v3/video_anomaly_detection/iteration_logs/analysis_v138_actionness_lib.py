"""
v138 offline RAW-ACTIONNESS reconstruction library (zero API calls, CLIP-local only).

WHY THIS EXISTS
The cached per-video artifact's `frame_scores` is NOT the raw CLIP anomaly curve: it is
the FLOODED curve produced by frame_scorer.compute_frame_scores (frame_scorer.py:63+),
which overlays each detection's OWN segment confidence on top of 0.35*cat_scores and
gaussian-smooths. Any feature read off it is partly circular with `confidence`, which is
exactly why the orchestrator's LOO-CV ceiling test over cached features came out WORSE
than cached confidence alone (AUC 0.744/0.717 vs 0.771/0.742).

The verdict-INDEPENDENT curve the pipeline actually uses for boundary detection is
frame_scorer.get_category_scores (frame_scorer.py:41-61):
    probs = softmax(clip_feats @ text_features.T * logit_scale)
    cat_scores = probs[:, cat_idx];  normal = probs[:, -1]
    curve      = cat_scores / (cat_scores + normal + 1e-8)
It is NOT cached, but it is exactly reproducible offline from the pre-extracted
snippet features in features/test/<video_id>.npy (290 files, (32,768), L2-normalized,
ViT-L/14) plus the SAME text embeddings the pipeline builds (learned prompt weights
from prompt_weights.json). Temporal resolution is 32 snippets/video instead of the
~130 inference frames -- coarser, but the detections here span 4-90s, so interior /
exterior contrast is still resolvable. Any precheck built on it is verdict-independent
by construction: no VLM output enters the curve.

Provides:
  load_text_features()               -> (torch text_features, categories, logit_scale)
  category_curve(video_id, category) -> (curve[32], snippet_center_times) using duration
  boundary_features(curve, times, t_start, t_end) -> dict of actionness/completeness feats
"""
import sys
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
VAD = ROOT / 'video_anomaly_detection'
FEAT_TEST = ROOT / 'features' / 'test'
sys.path.insert(0, str(VAD))


@lru_cache(maxsize=1)
def load_expert():
    """Build the real VideoAnomalyExpert (CLIP text tower only is exercised)."""
    from run import UCF_CRIME_CATEGORIES
    from anomaly_expert import VideoAnomalyExpert
    exp = VideoAnomalyExpert(list(UCF_CRIME_CATEGORIES))
    if exp.text_features is None:
        raise RuntimeError('text_features unavailable (CLIP text tower failed to load)')
    return exp, list(UCF_CRIME_CATEGORIES)


@lru_cache(maxsize=512)
def _feats(video_id):
    p = FEAT_TEST / f'{video_id}.npy'
    if not p.exists():
        return None
    a = np.load(p).astype(np.float32)
    return a


def category_curve(video_id, category, duration):
    """Raw verdict-independent actionness curve, reproducing get_category_scores exactly.

    Returns (curve, times) or (None, None) when features/category are unavailable.
    """
    import torch
    f = _feats(video_id)
    if f is None or not category:
        return None, None
    exp, cats = load_expert()
    if category not in cats:
        return None, None
    ci = cats.index(category)
    with torch.no_grad():
        x = torch.from_numpy(f).to(exp.text_features.device, dtype=exp.text_features.dtype)
        logit_scale = exp.model.logit_scale.exp().detach().clamp(max=100.0)
        sim = x @ exp.text_features.T * logit_scale
        probs = torch.softmax(sim, dim=-1).cpu().numpy()
    cat_s, nor_s = probs[:, ci], probs[:, -1]
    curve = cat_s / (cat_s + nor_s + 1e-8)
    n = len(curve)
    # snippet i covers [i/n, (i+1)/n) of the video; use centers
    times = (np.arange(n) + 0.5) * (duration / n)
    return curve.astype(float), times


def boundary_features(curve, times, t_start, t_end):
    """Verdict-independent actionness / completeness / boundary-quality features.

    All are pure functions of the raw curve and the (unchanged) window -- no VLM
    verdict, no cached confidence, so using them as a ranking key cannot leak the
    quantity being predicted.
    """
    if curve is None or times is None or len(curve) < 4:
        return {}
    c = np.asarray(curve, dtype=float)
    t = np.asarray(times, dtype=float)
    inside = (t >= t_start) & (t <= t_end)
    if inside.sum() < 1:
        # window shorter than one snippet: attribute to nearest snippet
        idx = int(np.argmin(np.abs(t - 0.5 * (t_start + t_end))))
        inside = np.zeros(len(c), dtype=bool)
        inside[idx] = True
    ins, out = c[inside], c[~inside]
    lo, hi = float(c.min()), float(c.max())
    rng = max(1e-9, hi - lo)
    i0, i1 = int(np.argmax(inside)), int(len(c) - 1 - np.argmax(inside[::-1]))

    def _mean(a, d=0.0):
        return float(a.mean()) if len(a) else d

    left = c[max(0, i0 - 2):i0]
    right = c[i1 + 1:min(len(c), i1 + 3)]
    feats = {
        # --- actionness (is the window ON the anomalous evidence at all) ---
        'act_mean': _mean(ins),
        'act_max': float(ins.max()),
        'act_mean_norm': (_mean(ins) - lo) / rng,
        'act_max_norm': (float(ins.max()) - lo) / rng,
        # --- inner/outer contrast (completeness-style: BSN/BMN inner-outer) ---
        'contrast': _mean(ins) - _mean(out, _mean(ins)),
        'contrast_norm': (_mean(ins) - _mean(out, _mean(ins))) / rng,
        'ratio': _mean(ins) / max(1e-9, _mean(out, _mean(ins))),
        # --- peak containment / prominence ---
        'peak_in': 1.0 if bool(inside[int(np.argmax(c))]) else 0.0,
        'peak_gap': (float(ins.max()) - hi) / rng,          # 0 when window owns the peak
        # --- boundary sharpness (TriDet-style directional edges) ---
        'onset_step': (_mean(ins) - _mean(left, _mean(ins))) / rng,
        'offset_step': (_mean(ins) - _mean(right, _mean(ins))) / rng,
        # --- interior consistency (a complete event should stay elevated) ---
        'interior_min_norm': (float(ins.min()) - lo) / rng,
        'interior_flatness': 1.0 - float(ins.std()) / max(1e-9, float(c.std())),
        # --- coverage: fraction of above-half-max mass captured ---
        'halfmax_recall': (float(ins[ins >= lo + 0.5 * rng].sum())
                           / max(1e-9, float(c[c >= lo + 0.5 * rng].sum()))),
        'halfmax_precision': (float(ins[ins >= lo + 0.5 * rng].sum())
                              / max(1e-9, float(ins.sum()))),
        'n_snip_in': float(inside.sum()),
    }
    return feats


def iter_cached_detections(result_dir):
    """Yield (video_id, anomaly_dict, duration) for one cached full-eval run."""
    for p in sorted(Path(result_dir).glob('*.json')):
        if p.name == 'evaluation_report.json':
            continue
        d = json.loads(p.read_text())
        if 'anomalies' not in d:
            continue
        for a in d['anomalies']:
            yield p.stem, a, float(d['duration'])
