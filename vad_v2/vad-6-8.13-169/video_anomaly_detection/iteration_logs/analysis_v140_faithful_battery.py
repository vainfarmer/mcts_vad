"""
v140 FAITHFUL-SUBSTRATE BATTERY — re-run the re-rank arms on the LIVE curve.

Two questions, in order. The second is only meaningful if the first passes.

Q1 (SUBSTRATE VALIDATION — retro-prediction of known in-vivo outcomes):
    Does the live-faithful substrate (analysis_v140_livecurve_lib, fidelity 58/58 recorded live peak
    times reproduced at 0.000s) predict the two verdicts the features/test substrate got WRONG?
      * v138 dense peak_in, w=0.2   -> in-vivo paired dmAP -0.00287
      * v139 pooled-32 peak_in, w=0.2 -> in-vivo paired dmAP -0.00149
    The old substrate predicted +0.0191 (8/8 caches positive, every weight). If the faithful
    substrate instead lands near zero/negative across the SAME 8 caches, it retro-predicts both
    refutations and is thereby validated as predictive — and the old +0.0191 is confirmed to have
    been 100% substrate artifact.

Q2 (SCREENING — only if Q1 passes): does ANY verdict-independent curve feature still show a
    trustworthy positive on the faithful substrate? Gate unchanged: same-sign on all caches and
    mean dmAP >= +0.005, with AP@0.7 non-regressing.

All arms are pure post-hoc re-ranking (detections/boundaries/count frozen), so FM1/FM2 immune.
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))

import analysis_v140_livecurve_lib as LC  # noqa: E402
import analysis_v138_actionness_lib as OLD  # noqa: E402  (features/test substrate, for contrast)
from evaluate import (  # noqa: E402
    load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou,
)

LABEL_PATH = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
CACHES = ['batch_eval_v114_full', 'batch_eval_v124_full', 'batch_eval_v125_full',
          'batch_eval_v127_full', 'batch_eval_v128_full', 'batch_eval_v130_full',
          'batch_eval_v131_full', 'batch_eval_v132_full']


def build(cache, skip_uncached=True):
    """Per-detection features from the LIVE curve. skip_uncached avoids new CLIP encoding."""
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(ROOT / 'results' / cache))
    gte = {k: v for k, v in gt.items() if k in pred}
    curves, rows = {}, []
    for vid, a, dur in OLD.iter_cached_detections(ROOT / 'results' / cache):
        cat = a.get('anomaly_type') or ''
        ck = (vid, cat)
        if ck not in curves:
            if skip_uncached and not (LC.CURVE_CACHE / f'{vid}.npz').exists():
                curves[ck] = (None, None)
            else:
                curves[ck] = LC.live_curve(vid, cat)
        c, t = curves[ck]
        f = OLD.boundary_features(c, t, a['t_start'], a['t_end']) if c is not None else {}
        if c is not None:
            f['peak_in_dense'] = 1.0 if a['t_start'] <= LC.pooled_peak_time(c, t, 0) <= a['t_end'] else 0.0
            f['peak_in_p32'] = 1.0 if a['t_start'] <= LC.pooled_peak_time(c, t, 32) <= a['t_end'] else 0.0
            f['peak_in_p16'] = 1.0 if a['t_start'] <= LC.pooled_peak_time(c, t, 16) <= a['t_end'] else 0.0
        iou = max([_temporal_iou(a, s, FPS) for s in gte[vid]['segments']] + [0.0]) \
            if vid in gte else 0.0
        rows.append(dict(vid=vid, key=(vid, round(a['t_start'], 3), round(a['t_end'], 3)),
                         conf=float(a.get('confidence', 0.0)), iou=iou, feats=f))
    return gte, pred, rows


def apply_arm(pred, rows, fn):
    fmap = {r['key']: r['feats'] for r in rows}
    p2 = {}
    for vid, e in pred.items():
        outs = []
        for a in e['anomalies']:
            k = (vid, round(a['t_start'], 3), round(a['t_end'], 3))
            f, c = fmap.get(k, {}), float(a.get('confidence', 0.0))
            v = fn(c, f) if f else c
            a2 = dict(a)
            a2['confidence'] = float(c if v is None or np.isnan(v) else v)
            outs.append(a2)
        p2[vid] = dict(e, anomalies=outs)
    return p2


ARMS = {
    'peak_in DENSE  (=v138, in vivo -0.00287)': lambda c, f: c * (0.8 + 0.2 * f['peak_in_dense']),
    'peak_in POOL32 (=v139, in vivo -0.00149)': lambda c, f: c * (0.8 + 0.2 * f['peak_in_p32']),
    'peak_in POOL16': lambda c, f: c * (0.8 + 0.2 * f['peak_in_p16']),
    'act_max_norm': lambda c, f: c * (0.8 + 0.2 * f['act_max_norm']),
    'act_mean_norm': lambda c, f: c * (0.8 + 0.2 * f['act_mean_norm']),
    'halfmax_recall': lambda c, f: c * (0.8 + 0.2 * f['halfmax_recall']),
    'contrast_norm(clipped)': lambda c, f: c * (0.8 + 0.2 * min(1.0, max(0.0, f['contrast_norm'] + 0.5))),
    'interior_min_norm': lambda c, f: c * (0.8 + 0.2 * f['interior_min_norm']),
    'onset_step(clipped)': lambda c, f: c * (0.8 + 0.2 * min(1.0, max(0.0, f['onset_step'] + 0.5))),
}


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, float)
    ok = ~np.isnan(s)
    s, l = s[ok], l[ok]
    pos, neg = s[l > 0.5], s[l <= 0.5]
    if not len(pos) or not len(neg):
        return float('nan')
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))).astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    built, bases = {}, {}
    cov = []
    for cache in CACHES:
        gte, pred, rows = build(cache)
        built[cache] = (gte, pred, rows)
        bases[cache] = compute_temporal_map(gte, pred, fps=FPS)
        n_f = sum(1 for r in rows if r['feats'])
        cov.append((cache[11:15], n_f, len(rows)))
    print('coverage (detections with a LIVE curve / total): ' +
          '  '.join(f'{c}:{a}/{b}' for c, a, b in cov))
    print('=' * 104)
    print(f'{"arm":42s}' + ''.join(f'{c[11:15]:>9s}' for c in CACHES) + f'{"mean":>10s}{"pos":>6s}')
    out = {}
    for name, fn in ARMS.items():
        ds = []
        d7 = []
        for cache in CACHES:
            gte, pred, rows = built[cache]
            r = compute_temporal_map(gte, apply_arm(pred, rows, fn), fps=FPS)
            ds.append(r['mAP'] - bases[cache]['mAP'])
            d7.append(r['AP@0.7'] - bases[cache]['AP@0.7'])
        out[name] = dict(deltas=ds, mean=float(np.mean(ds)), dap7=float(np.mean(d7)),
                         pos=int(sum(1 for x in ds if x > 0)))
        print(f'{name:42s}' + ''.join(f'{x:+9.4f}' for x in ds) +
              f'{np.mean(ds):+10.5f}{out[name]["pos"]:>4d}/8')
    print('=' * 104)
    print('feature AUC for IoU>=0.3 on the LIVE substrate (pooled over all caches):')
    lab, feats = [], {}
    for cache in CACHES:
        for r in built[cache][2]:
            if not r['feats']:
                continue
            lab.append(1.0 if r['iou'] >= 0.3 else 0.0)
            for k, v in r['feats'].items():
                feats.setdefault(k, []).append(v)
            feats.setdefault('cached_conf', []).append(r['conf'])
    for k in sorted(feats, key=lambda k: -abs(auc(feats[k], lab) - 0.5)):
        print(f'   {k:22s} AUC={auc(feats[k], lab):.4f}')
    (HERE / 'analysis_v140_faithful_battery.json').write_text(
        json.dumps(out, indent=2, default=float))
    print('wrote analysis_v140_faithful_battery.json')


if __name__ == '__main__':
    main()
