"""v138 SCRATCH ANALYSIS #2 (Agent A, research only).

The v138 ceiling analysis showed the ENTIRE remaining pure-re-ranking headroom is
CROSS-VIDEO (ORACLE-CROSS +0.061) and the intra-video axis is exhausted
(ORACLE-WITHIN +0.006 == exactly what v111 Q-score already banked).

This script asks: which VERDICT-INDEPENDENT, CROSS-VIDEO-COMPARABLE, offline-computable
scalar predicts a video's best achievable IoU (i.e. localization trustworthiness)?

Sources (all zero-API):
  A) saved per-video detection JSON (results/batch_eval_v124_full/*.json):
     frame_scores (contaminated INSIDE detections by the max-overlay), timestamps,
     duration, anomalies[t_start,t_end,confidence,anomaly_type]
  B) cached CLIP ViT-L/14 features features/test/<vid>.npy -> (32,768) L2-normalised,
     UNCONTAMINATED, same model as the pipeline -> clean absolute category curve
  C) model_mil.pth applied to (1,32,768) -> clean MIL snippet curve
"""
import sys, json, glob
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
VAD = ROOT / 'video_anomaly_detection'
sys.path.insert(0, str(VAD))
from evaluate import load_ground_truth, compute_temporal_map, _temporal_iou  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
RES = ROOT / 'results/batch_eval_v124_full'
FEAT = ROOT / 'features/test'
CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
        "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    import torch
    from anomaly_expert import VideoAnomalyExpert
    gt = load_ground_truth(LABEL, split='test')
    preds = {}
    for jf in sorted(glob.glob(str(RES / '*.json'))):
        d = json.load(open(jf))
        preds[Path(d.get('video_path', jf)).stem] = d
    gt = {k: v for k, v in gt.items() if k in preds}

    dev = 'cpu'
    exp = VideoAnomalyExpert(CATS, device=dev)
    assert exp.text_features is not None, 'CLIP text features unavailable'
    tf = exp.text_features
    logit_scale = exp.model.logit_scale.exp().detach().clamp(max=100.0)

    # MIL
    mil = None
    try:
        import mil_model as mm
        sd = torch.load(str(VAD / 'model_mil.pth'), map_location='cpu')
        for ctor in ('MILAnomalyScorer',):
            if hasattr(mm, ctor):
                try:
                    m = getattr(mm, ctor)()
                    m.load_state_dict(sd if not isinstance(sd, dict) or 'state_dict' not in sd
                                      else sd['state_dict'])
                    m.eval(); mil = m; print(f'[MIL] loaded via {ctor}'); break
                except Exception as e:
                    print(f'[MIL] {ctor} failed: {e}')
    except Exception as e:
        print(f'[MIL] unavailable: {e}')

    feats = {}
    rows = []
    missing = []
    for vid, d in preds.items():
        anoms = d.get('anomalies', [])
        if not anoms:
            continue
        gs = gt.get(vid, {}).get('segments', [])
        bious = [max([_temporal_iou(a, g, FPS) for g in gs], default=0.0) for a in anoms]
        fp = FEAT / f'{vid}.npy'
        if not fp.exists():
            missing.append(vid); continue
        X = torch.from_numpy(np.load(fp)).float()          # (32,768) L2-normalised
        with torch.no_grad():
            sim = X @ tf.T.float().cpu() * float(logit_scale)
            probs = torch.softmax(sim, dim=-1).numpy()
        cat = anoms[0].get('anomaly_type', 'unknown')
        ci = CATS.index(cat) if cat in CATS else int(np.argmax(probs[:, :-1].mean(0)))
        catc = probs[:, ci]; norm = probs[:, -1]
        rel = catc / (catc + norm + 1e-8)                   # SAME formula as frame_scorer:59
        anom_any = 1.0 - norm                               # any-anomaly evidence
        mil_c = None
        if mil is not None:
            with torch.no_grad():
                mil_c = mil(X.unsqueeze(0)).squeeze(0).numpy().astype(float).ravel()

        # ---- clean cross-video-comparable video-level scalars ----
        f = {}
        f['rel_max'] = float(rel.max())
        f['rel_mean'] = float(rel.mean())
        f['rel_std'] = float(rel.std())
        f['rel_range'] = float(rel.max() - rel.min())
        f['rel_p90_p50'] = float(np.percentile(rel, 90) - np.percentile(rel, 50))
        f['rel_max_minus_med'] = float(rel.max() - np.median(rel))
        f['rel_kurt'] = float(((rel - rel.mean()) ** 4).mean() / (rel.var() + 1e-12) ** 2)
        f['rel_topgap'] = float(np.sort(rel)[-1] - np.sort(rel)[-4])   # peak isolation
        f['rel_frac_hi'] = float(np.mean(rel > 0.5))                   # plateau breadth
        f['anom_max'] = float(anom_any.max())
        f['anom_mean'] = float(anom_any.mean())
        f['anom_max_minus_med'] = float(anom_any.max() - np.median(anom_any))
        f['catprob_max'] = float(catc.max())
        f['catprob_margin'] = float((catc - norm).max())               # absolute margin
        if mil_c is not None and mil_c.size >= 4:
            f['mil_max'] = float(mil_c.max()); f['mil_mean'] = float(mil_c.mean())
            f['mil_std'] = float(mil_c.std())
            f['mil_max_minus_med'] = float(mil_c.max() - np.median(mil_c))
            f['mil_frac_hi'] = float(np.mean(mil_c > 0.5))
            fu = 0.75 * rel + 0.25 * mil_c
            f['fus_max_minus_med'] = float(fu.max() - np.median(fu))
            f['fus_topgap'] = float(np.sort(fu)[-1] - np.sort(fu)[-4])

        # geometry / coverage (verdict-independent, absolute)
        dur = float(d.get('duration', 0)) or 1.0
        tot = sum(a['t_end'] - a['t_start'] for a in anoms)
        f['coverage'] = float(tot / dur)
        f['n_det'] = float(len(anoms))
        f['maxdet_dur'] = float(max(a['t_end'] - a['t_start'] for a in anoms))
        f['video_dur'] = dur
        # clean part of the SAVED curve (outside every detection)
        fs = np.asarray(d.get('frame_scores') or [], float)
        ts = np.asarray(d.get('timestamps') or [], float)
        if fs.size and fs.size == ts.size:
            out = np.ones(fs.size, bool)
            for a in anoms:
                out &= ~((ts >= a['t_start'] - 2.0) & (ts <= a['t_end'] + 2.0))
            if out.sum() >= 4:
                f['outp90'] = float(np.percentile(fs[out], 90))
                f['outmean'] = float(fs[out].mean())
            else:
                f['outp90'] = float('nan'); f['outmean'] = float('nan')
        else:
            f['outp90'] = float('nan'); f['outmean'] = float('nan')

        feats[vid] = f
        rows.append((vid, max(bious), f))

    print(f"videos with detections+features: {len(rows)}  missing feature file: {len(missing)} {missing[:6]}")
    keys = sorted({k for _, _, f in rows for k in f})
    y = np.array([r[1] for r in rows])
    print(f"\nbest-IoU per video: mean={y.mean():.3f} frac0={np.mean(y<=1e-9):.3f} "
          f"frac>=0.5={np.mean(y>=0.5):.3f}")
    print(f"\n{'signal':<24} {'spearman_vs_bestIoU':>20}  {'spearman_vs_(IoU>=0.5)':>24}")
    yb = (y >= 0.5).astype(float)
    ranked = []
    for k in keys:
        x = np.array([f.get(k, np.nan) for _, _, f in rows], float)
        s1 = spearman(x, y); s2 = spearman(x, yb)
        ranked.append((abs(s1) if not np.isnan(s1) else 0, k, s1, s2))
    for _, k, s1, s2 in sorted(ranked, reverse=True):
        print(f"{k:<24} {s1:>20.4f}  {s2:>24.4f}")

    # reference: how well does the CURRENT video-level confidence predict best IoU?
    curconf = np.array([max(a['confidence'] for a in preds[v]['anomalies']) for v, _, _ in rows])
    print(f"\n[REF] spearman(video max CURRENT confidence, bestIoU) = {spearman(curconf, y):.4f}")
    json.dump({'feats': feats, 'best_iou': {r[0]: r[1] for r in rows}},
              open(VAD / 'iteration_logs/analysis_v138_crossvideo_signals.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
