"""
v138 diagnostic probe #2 (Agent B, research-only) — WHICH cross-video signal predicts
per-video localization quality (max IoU), and does re-leveling by it move mAP?

Consumes only saved artifacts (no VLM, no CLIP forward pass):
  results/batch_eval_v124_full/*.json  (66; anomalies + frame_scores + timestamps)
  features/test/*.npy                  (290; 32x768 pooled CLIP snippet features)
  /Users/zzl/Movies/ucf-crime/data     (parquet GT)
  video_anomaly_detection/model_mil.pth
"""
import sys, json, glob
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
THRESH = [0.3, 0.5, 0.7]
DDIR = ROOT / 'results/batch_eval_v124_full'


def iou(a, g):
    ps, pe = a['t_start'] * FPS, a['t_end'] * FPS
    gs, ge = g
    ov = max(0.0, min(pe, ge) - max(ps, gs))
    un = max(pe, ge) - min(ps, gs)
    return ov / un if un > 0 else 0.0


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or x.std() == 0 or y.std() == 0:
        return float('nan')
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    gt_all = load_ground_truth(LABEL, split='test')
    rows = {}
    for jf in sorted(glob.glob(str(DDIR / '*.json'))):
        vid = Path(jf).stem
        if vid not in gt_all:
            continue
        j = json.load(open(jf))
        anoms = j.get('anomalies', [])
        fs = np.asarray(j.get('frame_scores') or [], float)
        ts = np.asarray(j.get('timestamps') or [], float)
        segs = gt_all[vid]['segments']
        best = [max([iou(a, g) for g in segs], default=0.0) for a in anoms]
        # region outside ALL detections (un-overlaid background of the curve)
        if len(fs) and len(ts) == len(fs):
            inside = np.zeros(len(ts), bool)
            for a in anoms:
                inside |= (ts >= a['t_start']) & (ts <= a['t_end'])
            bg = fs[~inside]
        else:
            bg = np.array([])
        rows[vid] = dict(
            n_det=len(anoms), dur=j['duration'],
            conf_max=max([a['confidence'] for a in anoms], default=0.0),
            best_iou=max(best, default=0.0),
            det_dur=(anoms[0]['t_end'] - anoms[0]['t_start']) if anoms else 0.0,
            det_dur_max=max([a['t_end'] - a['t_start'] for a in anoms], default=0.0),
            cov_ratio=(sum(a['t_end'] - a['t_start'] for a in anoms) / max(1e-6, j['duration'])),
            bg_mean=float(bg.mean()) if bg.size else float('nan'),
            bg_p90=float(np.percentile(bg, 90)) if bg.size else float('nan'),
            bg_max=float(bg.max()) if bg.size else float('nan'),
            curve_std=float(fs.std()) if fs.size else float('nan'),
            curve_mean=float(fs.mean()) if fs.size else float('nan'),
            n_gt=len(segs), cat=gt_all[vid]['category'],
            anoms=anoms, fs=fs, ts=ts, best=best,
        )

    # ---- MIL absolute per-video prior (v109 replication, for the record) ----
    try:
        import torch
        from mil_model import MILAnomalyScorer
        m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
        st = torch.load(str(ROOT / 'video_anomaly_detection/model_mil.pth'), map_location='cpu')
        m.load_state_dict(st.get('model_state_dict', st) if isinstance(st, dict) else st)
        m.eval()
        for v in rows:
            fp = ROOT / f'features/test/{v}.npy'
            if fp.exists():
                f = np.load(fp)
                with torch.no_grad():
                    s = m(torch.tensor(f, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                rows[v]['mil_max'] = float(s.max())
                rows[v]['mil_top4'] = float(np.sort(s)[-4:].mean())
                rows[v]['mil_mean'] = float(s.mean())
                rows[v]['mil_spread'] = float(np.sort(s)[-4:].mean() - np.sort(s)[:8].mean())
    except Exception as e:
        print('MIL unavailable:', e)

    vids = sorted(rows)
    print(f"videos={len(vids)}  n_det={sum(rows[v]['n_det'] for v in vids)}")
    y = np.array([rows[v]['best_iou'] for v in vids])
    print(f"per-video best_iou: mean={y.mean():.3f}  ==0: {(y==0).sum()}/{len(y)}  "
          f">=0.3: {(y>=0.3).sum()}  >=0.5: {(y>=0.5).sum()}  >=0.7: {(y>=0.7).sum()}")
    lvl = np.array([rows[v]['conf_max'] for v in vids])
    print(f"per-video conf_max: min={lvl.min():.3f} med={np.median(lvl):.3f} max={lvl.max():.3f} "
          f"n_unique={len(set(np.round(lvl,6)))}")

    print("\n--- SPEARMAN(signal, per-video best_iou) over %d videos ---" % len(vids))
    keys = ['conf_max', 'n_det', 'dur', 'det_dur', 'det_dur_max', 'cov_ratio',
            'bg_mean', 'bg_p90', 'bg_max', 'curve_std', 'curve_mean',
            'mil_max', 'mil_top4', 'mil_mean', 'mil_spread']
    for k in keys:
        if k not in rows[vids[0]]:
            continue
        s = spear([rows[v].get(k, np.nan) for v in vids], y)
        print(f"  {k:<12} {s:+.4f}")

    # ---- offline re-rank trials: conf' = conf * mult(video-level signal) ----
    def run(mult_fn, name):
        pred, gt = {}, {}
        for v in vids:
            m_ = mult_fn(rows[v])
            pred[v] = {'anomalies': [dict(a, confidence=min(1.0, a['confidence'] * m_))
                                     for a in rows[v]['anoms']]}
            gt[v] = gt_all[v]
        r = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THRESH)
        return r, name

    base_r, _ = run(lambda r: 1.0, 'baseline')
    print(f"\nbaseline mAP={base_r['mAP']:.5f}  "
          + '  '.join(f"AP@{t}={base_r[f'AP@{t}']:.5f}" for t in THRESH))

    trials = {}
    # background-penalty family (AS-Norm-like cohort normalization, absolute scale)
    for a in [0.3, 0.5, 0.8, 1.0]:
        trials[f'bg_penalty a={a}'] = (
            lambda r, a=a: float(np.clip(1.0 - a * (r['bg_mean'] if np.isfinite(r['bg_mean']) else 0.0), 0.05, 1.0)))
    # absolute contrast level (peak_in - bg_mean), no within-video normalization
    def abs_contrast(r):
        if not r['anoms'] or r['fs'].size == 0 or not np.isfinite(r['bg_mean']):
            return 1.0
        pk = max(r['fs'].max(), 1e-6)
        return float(np.clip(0.5 + 0.5 * (pk - r['bg_mean']), 0.05, 1.0))
    trials['abs_contrast'] = abs_contrast
    # coverage-ratio penalty (detected fraction of the video; cross-video comparable)
    for a in [0.5, 1.0, 2.0]:
        trials[f'cov_penalty a={a}'] = (
            lambda r, a=a: float(np.clip(1.0 - a * r['cov_ratio'], 0.05, 1.0)))
    # n_det penalty
    trials['ndet_penalty'] = lambda r: 1.0 / (1.0 + 0.25 * max(0, r['n_det'] - 1))
    # video-duration difficulty prior
    trials['short_video_bonus'] = lambda r: float(np.clip(1.2 - r['dur'] / 300.0, 0.3, 1.0))
    # MIL spread (not the saturated max) as cross-video level
    if 'mil_spread' in rows[vids[0]]:
        for a in [0.3, 0.5]:
            trials[f'mil_spread a={a}'] = (
                lambda r, a=a: float(np.clip(1 - a + a * r.get('mil_spread', 1.0), 0.05, 1.0)))
    print()
    out = {'baseline': base_r['mAP']}
    for name, fn in sorted(trials.items()):
        r, _ = run(fn, name)
        out[name] = r['mAP']
        print(f"  {name:<22} mAP={r['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
              + f"  delta={r['mAP']-base_r['mAP']:+.5f}")

    json.dump(out, open(ROOT / 'video_anomaly_detection/iteration_logs/'
                        'analysis_v138_crossvideo_signal_probe.json', 'w'), indent=2)

    # ---- per-video dump for eyeballing ----
    print("\n--- worst offenders: high conf_max, zero IoU ---")
    bad = sorted([v for v in vids if rows[v]['best_iou'] == 0.0],
                 key=lambda v: -rows[v]['conf_max'])[:12]
    for v in bad:
        r = rows[v]
        print(f"  {v:<26} conf={r['conf_max']:.3f} n_det={r['n_det']} "
              f"det_dur={r['det_dur_max']:.1f}s vid={r['dur']:.0f}s cov={r['cov_ratio']:.2f} "
              f"bg={r['bg_mean']:.3f} mil={r.get('mil_max', float('nan')):.3f}")
    print("--- best: high IoU ---")
    good = sorted(vids, key=lambda v: -rows[v]['best_iou'])[:12]
    for v in good:
        r = rows[v]
        print(f"  {v:<26} iou={r['best_iou']:.3f} conf={r['conf_max']:.3f} n_det={r['n_det']} "
              f"det_dur={r['det_dur_max']:.1f}s vid={r['dur']:.0f}s cov={r['cov_ratio']:.2f} "
              f"bg={r['bg_mean']:.3f} mil={r.get('mil_max', float('nan')):.3f}")


if __name__ == '__main__':
    main()
