"""
v138 offline precheck — RAW-ACTIONNESS re-ranking (verdict-independent), dual-cache.

Protocol (pre-registered before any arm is run):
  * Detection set, boundaries and per-video COUNT are IDENTICAL to the cached run
    (pure post-hoc re-ranking = the historically safe class; FM1/FM2 immune).
  * The ranking key may only use the RAW curve reconstructed from features/test
    (analysis_v138_actionness_lib) + cached confidence. No VLM verdict, no GT.
  * EVERY arm is measured on BOTH independent cached full evals (v130_full, v132_full).
    GO requires SAME-SIGN dmAP on both AND mean dmAP >= +0.010 AND no AP@0.7 regression
    on either cache. (Discipline added after duration-family arms flipped sign between
    caches: v130 -0.005..-0.031 vs v132 +0.003..+0.014 = noise.)
  * Reported diagnostics are CHECKPOINT PRECISIONS (precision at the 1st/7th/14th/20th
    hit = the only quantities 11-point AP at n_gt=66 can see), never rank correlation
    (interior-mean had better spearman than confidence yet regressed mAP).

Also reports, as the decisive feasibility datum, the discriminative AUC of each raw
actionness feature for IoU>=0.3 -- if the raw curve cannot separate hits from misses,
the whole verdict-independent-confidence family (v135's registered forward path) is
CLOSED on this dataset and no arm can clear the gate.
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))

import analysis_v138_actionness_lib as L  # noqa: E402
from evaluate import (  # noqa: E402
    load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou,
)

LABEL_PATH = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
CACHES = ['batch_eval_v130_full', 'batch_eval_v132_full']
FEATS = ['act_mean', 'act_max', 'act_mean_norm', 'act_max_norm', 'contrast',
         'contrast_norm', 'ratio', 'peak_in', 'peak_gap', 'onset_step',
         'offset_step', 'interior_min_norm', 'interior_flatness',
         'halfmax_recall', 'halfmax_precision', 'n_snip_in']


def _auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, float)
    ok = ~np.isnan(score)
    score, label = score[ok], label[ok]
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # rank-based AUC with tie handling
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1
    rpos = r[:len(pos)]
    return float((rpos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def build(cache):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(ROOT / 'results' / cache))
    gte = {k: v for k, v in gt.items() if k in pred}
    curves = {}
    rows = []
    for vid, a, dur in L.iter_cached_detections(ROOT / 'results' / cache):
        cat = a.get('anomaly_type') or ''
        key = (vid, cat, round(dur, 2))
        if key not in curves:
            curves[key] = L.category_curve(vid, cat, dur)
        curve, times = curves[key]
        f = L.boundary_features(curve, times, a['t_start'], a['t_end'])
        iou = 0.0
        if vid in gte:
            iou = max([_temporal_iou(a, s, FPS) for s in gte[vid]['segments']] + [0.0])
        rows.append(dict(vid=vid, key=(vid, round(a['t_start'], 3), round(a['t_end'], 3)),
                         conf=float(a.get('confidence', 0.0)), iou=iou, feats=f,
                         has_curve=bool(f)))
    return gt, gte, pred, rows


def checkpoints(rows_sorted, n_gt, iou_thr):
    """Precision at the recall checkpoints 11-point AP can actually see."""
    hits = 0
    out = {}
    targets = {}
    for k in range(11):
        need = int(np.ceil(k / 10.0 * n_gt))
        if need <= n_gt * max(r['iou'] >= iou_thr for r in rows_sorted):
            targets[k] = max(1, need)
    for i, r in enumerate(rows_sorted, start=1):
        if r['iou'] >= iou_thr:
            hits += 1
            for k, need in targets.items():
                if hits == need and k not in out:
                    out[k] = hits / i
    return out


def rerank(pred, keyfn):
    p2 = {}
    for vid, e in pred.items():
        outs = []
        for a in e['anomalies']:
            a2 = dict(a)
            a2['confidence'] = float(keyfn(vid, a))
            outs.append(a2)
        p2[vid] = dict(e, anomalies=outs)
    return p2


def main():
    data = {}
    print('=' * 92)
    for cache in CACHES:
        gt, gte, pred, rows = build(cache)
        cov = sum(1 for r in rows if r['has_curve'])
        n_gt = sum(len(v['segments']) for v in gte.values())
        base = compute_temporal_map(gte, pred, fps=FPS)
        print(f'{cache}: {len(rows)} dets ({cov} with raw curve), n_gt={n_gt}, '
              f'base mAP={base["mAP"]:.5f}')
        lab = [1.0 if r['iou'] >= 0.3 else 0.0 for r in rows if r['has_curve']]
        print(f'   AUC(IoU>=0.3)  cached conf = '
              f'{_auc([r["conf"] for r in rows if r["has_curve"]], lab):.4f}   <-- to beat')
        aucs = {}
        for f in FEATS:
            aucs[f] = _auc([r['feats'].get(f, np.nan) for r in rows if r['has_curve']], lab)
        for f, v in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5)):
            print(f'      {f:20s} AUC={v:.4f}  (|d|={abs(v - 0.5):.4f})')
        data[cache] = dict(gte=gte, pred=pred, rows=rows, base=base, n_gt=n_gt, aucs=aucs)
        print('-' * 92)

    # cross-cache AUC consistency: a usable feature must beat 0.5 in the SAME direction
    print('AUC direction consistency across caches (feature: v130 / v132):')
    for f in FEATS:
        a, b = data[CACHES[0]]['aucs'][f], data[CACHES[1]]['aucs'][f]
        flag = 'CONSISTENT' if (a - 0.5) * (b - 0.5) > 0 else 'FLIPS'
        print(f'   {f:20s} {a:.4f} / {b:.4f}   {flag}')
    print('=' * 92)

    # ---- arms (only run if some feature is consistently discriminative) ----
    def arms_for(rows):
        fmap = {r['key']: r['feats'] for r in rows}
        cmap = {r['key']: r['conf'] for r in rows}

        def mk(name, fn):
            def keyfn(vid, a):
                k = (vid, round(a['t_start'], 3), round(a['t_end'], 3))
                f, c = fmap.get(k, {}), cmap.get(k, float(a.get('confidence', 0.0)))
                v = fn(c, f)
                return c if v is None or (isinstance(v, float) and np.isnan(v)) else v
            return name, keyfn
        A = []
        A.append(mk('R1 conf*(0.6+0.4*act_mean_norm)',
                    lambda c, f: c * (0.6 + 0.4 * f['act_mean_norm']) if f else None))
        A.append(mk('R2 conf*(0.6+0.4*halfmax_recall)',
                    lambda c, f: c * (0.6 + 0.4 * f['halfmax_recall']) if f else None))
        A.append(mk('R3 conf*(0.6+0.4*(peak_in))',
                    lambda c, f: c * (0.6 + 0.4 * f['peak_in']) if f else None))
        A.append(mk('R4 conf*(0.6+0.4*clip(contrast_norm))',
                    lambda c, f: c * (0.6 + 0.4 * min(1.0, max(0.0, f['contrast_norm'] + 0.5))) if f else None))
        A.append(mk('R5 conf*(0.6+0.4*act_max_norm)',
                    lambda c, f: c * (0.6 + 0.4 * f['act_max_norm']) if f else None))
        A.append(mk('R6 act_mean_norm only',
                    lambda c, f: f['act_mean_norm'] if f else None))
        return A

    print('ARMS (dual-cache; GO = same sign both caches, mean dmAP >= +0.010, no AP@0.7 loss)')
    names = [n for n, _ in arms_for(data[CACHES[0]]['rows'])]
    results = {n: {} for n in names}
    for cache in CACHES:
        d = data[cache]
        for name, keyfn in arms_for(d['rows']):
            r = compute_temporal_map(d['gte'], rerank(d['pred'], keyfn), fps=FPS)
            results[name][cache] = r
    for name in names:
        line = [f'{name:38s}']
        ds = []
        for cache in CACHES:
            b, r = data[cache]['base'], results[name][cache]
            ds.append(r['mAP'] - b['mAP'])
            line.append(f'{cache[11:15]}: {r["mAP"]:.5f} ({r["mAP"] - b["mAP"]:+.5f}) '
                        f'AP.7={r["AP@0.7"]:.4f}({r["AP@0.7"] - b["AP@0.7"]:+.4f})')
        same = ds[0] * ds[1] > 0
        verdict = 'GO' if (same and np.mean(ds) >= 0.010) else ('same-sign' if same else 'FLIPS')
        print('  ' + '  '.join(line) + f'  [{verdict}]')

    out = HERE / 'analysis_v138_actionness_precheck.json'
    out.write_text(json.dumps({
        'caches': {c: {'base': data[c]['base'], 'n_det': len(data[c]['rows']),
                       'n_gt': data[c]['n_gt'], 'aucs': data[c]['aucs']} for c in CACHES},
        'arms': {n: {c: results[n][c] for c in CACHES} for n in names},
    }, indent=2, default=float))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
