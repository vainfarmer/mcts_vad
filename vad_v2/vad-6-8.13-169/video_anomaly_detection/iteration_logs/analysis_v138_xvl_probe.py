"""
v138 diagnostic probe #4 (Agent B, research-only) — CROSS-VIDEO LEVEL (XVL) in
STACK form: conf'' = conf' * m(video), applied AFTER the committed v111/v114
within-video Q-score. Faithfully testable on saved v124 detections because the
stack multiplier is applied to exactly the confidences the live pipeline would
carry at the insertion point (pipeline.py:1288-1291, right after _qscore_rerank).

Signals under test (all cross-video comparable, i.e. NOT within-video normalized):
  XVL-TXT : absolute CLIP anomaly-vs-Normal text margin over the video's 32
            pooled snippets (features/test/*.npy) -- v123 SLM-CV lineage at video level
  XVL-CRV : absolute dispersion of the output score curve (curve_std)
  XVL-DUR : log detection duration (empirically strongest, but semantically empty)
  XVL-MIL : MIL video-level prior (v109 replication -- expected saturated/null)

Consumes: results/batch_eval_v124_full, features/test, model_mil.pth, parquet GT.
"""
import sys, json, glob
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS, THRESH = 30, [0.3, 0.5, 0.7]
DIRS = ['results/batch_eval_v124_full', 'results/batch_eval_v131_full',
        'results/batch_eval_v130_full', 'results/batch_eval_v132_full',
        'results/batch_eval_v114_full']


def iou(a, g):
    ps, pe, gs, ge = a['t_start'] * FPS, a['t_end'] * FPS, g[0], g[1]
    ov = max(0.0, min(pe, ge) - max(ps, gs)); un = max(pe, ge) - min(ps, gs)
    return ov / un if un > 0 else 0.0


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 5 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


# ---------------- XVL-TXT: absolute CLIP text margin per video ----------------
def build_txt_levels(vids):
    try:
        import torch, open_clip
        sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
        from anomaly_expert import VideoAnomalyExpert
        from pipeline import MultiAnomalyVAD
        CATS = MultiAnomalyVAD.DEFAULT_ANOMALY_CATEGORIES
    except Exception as e:
        print('  [XVL-TXT] unavailable:', e)
        return {}
    ex = VideoAnomalyExpert(list(CATS), device='cpu')
    if ex.model is None:
        return {}
    tf = ex.text_features.detach().cpu().numpy()          # (14, 768) L2-normed
    ls = float(ex.model.logit_scale.exp().detach().cpu().numpy().clip(max=100.0))
    out = {}
    for v in vids:
        fp = ROOT / f'features/test/{v}.npy'
        if not fp.exists():
            continue
        f = np.load(fp).astype(np.float64)                # (32, 768)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        sim = f @ tf.T * ls                               # (32, 14)
        e = np.exp(sim - sim.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        top3 = np.sort(p[:, :-1], axis=1)[:, -3:].sum(axis=1)
        rel = top3 / (top3 + p[:, -1] + 1e-9)             # per-snippet, ABSOLUTE [0,1]
        out[v] = dict(txt_max=float(rel.max()), txt_top4=float(np.sort(rel)[-4:].mean()),
                      txt_mean=float(rel.mean()),
                      txt_margin=float(np.sort(rel)[-4:].mean() - np.sort(rel)[:8].mean()))
    return out


def build_mil_levels(vids):
    try:
        import torch
        from mil_model import MILAnomalyScorer
        m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
        st = torch.load(str(ROOT / 'video_anomaly_detection/model_mil.pth'), map_location='cpu')
        m.load_state_dict(st.get('model_state_dict', st) if isinstance(st, dict) else st)
        m.eval()
    except Exception as e:
        print('  [XVL-MIL] unavailable:', e)
        return {}
    out = {}
    for v in vids:
        fp = ROOT / f'features/test/{v}.npy'
        if not fp.exists():
            continue
        with torch.no_grad():
            s = m(torch.tensor(np.load(fp), dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        out[v] = dict(mil_max=float(s.max()), mil_top4=float(np.sort(s)[-4:].mean()))
    return out


def main():
    gt_all = load_ground_truth(LABEL, split='test')
    allvids = sorted({Path(f).stem for f in glob.glob(str(ROOT / DIRS[0] / '*.json'))} & set(gt_all))
    print('building XVL-TXT levels ...')
    txt = build_txt_levels(allvids)
    print(f'  ok for {len(txt)}/{len(allvids)} videos')
    mil = build_mil_levels(allvids)

    for dname in DIRS:
        dets = {}
        for jf in sorted(glob.glob(str(ROOT / dname / '*.json'))):
            v = Path(jf).stem
            if v in gt_all:
                dets[v] = json.load(open(jf))
        if not dets:
            continue
        gt = {v: gt_all[v] for v in dets}
        vids = sorted(dets)
        best = {v: max([max([iou(a, g) for g in gt_all[v]['segments']], default=0.0)
                        for a in dets[v]['anomalies']], default=0.0) for v in vids}
        crv = {}
        for v in vids:
            fs = np.asarray(dets[v].get('frame_scores') or [], float)
            crv[v] = float(fs.std()) if fs.size else float('nan')
        dur = {v: max([a['t_end'] - a['t_start'] for a in dets[v]['anomalies']], default=0.0)
               for v in vids}

        print(f"\n================ {Path(dname).name} ({len(vids)} videos) ================")
        y = [best[v] for v in vids]
        for nm, sig in [('txt_max', {v: txt.get(v, {}).get('txt_max', np.nan) for v in vids}),
                        ('txt_top4', {v: txt.get(v, {}).get('txt_top4', np.nan) for v in vids}),
                        ('txt_margin', {v: txt.get(v, {}).get('txt_margin', np.nan) for v in vids}),
                        ('mil_top4', {v: mil.get(v, {}).get('mil_top4', np.nan) for v in vids}),
                        ('curve_std', crv), ('det_dur', dur)]:
            print(f"  spearman({nm:<10}, best_iou) = {spear([sig[v] for v in vids], y):+.4f}")

        def run(mfn):
            pred = {v: {'anomalies': [dict(a, confidence=min(1.0, a['confidence'] * mfn(v)))
                                      for a in dets[v]['anomalies']]} for v in vids}
            return compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THRESH)

        base = run(lambda v: 1.0)
        print(f"  baseline           mAP={base['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={base[f'AP@{t}']:.5f}" for t in THRESH))

        def lin(sig, b):
            arr = np.array([sig[v] for v in vids], float)
            fin = arr[np.isfinite(arr)]
            lo, hi = (fin.min(), fin.max()) if fin.size else (0.0, 1.0)
            rng = max(hi - lo, 1e-9)
            def f(v):
                x = sig[v]
                if not np.isfinite(x):
                    return 1.0
                return float(b + (1 - b) * (x - lo) / rng)
            return f

        trials = []
        for nm, sig in [('XVL-TXT(top4)', {v: txt.get(v, {}).get('txt_top4', np.nan) for v in vids}),
                        ('XVL-TXT(max)', {v: txt.get(v, {}).get('txt_max', np.nan) for v in vids}),
                        ('XVL-TXT(marg)', {v: txt.get(v, {}).get('txt_margin', np.nan) for v in vids}),
                        ('XVL-MIL(top4)', {v: mil.get(v, {}).get('mil_top4', np.nan) for v in vids}),
                        ('XVL-CRV(std)', crv),
                        ('XVL-DUR(log)', {v: float(np.log1p(dur[v])) for v in vids})]:
            for b in [0.7, 0.5, 0.3]:
                trials.append((f'{nm} b={b}', lin(sig, b)))
        for nm, fn in trials:
            r = run(fn)
            print(f"  {nm:<20} mAP={r['mAP']:.5f}  "
                  + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
                  + f"  delta={r['mAP']-base['mAP']:+.5f}")


if __name__ == '__main__':
    main()
