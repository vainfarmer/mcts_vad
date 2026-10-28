"""
v138 diagnostic probe (Agent B, research-only) — how much mAP headroom exists in
PURE CONFIDENCE RE-RANKING, and how it splits into within-video vs cross-video.

Reads REAL saved detections (no pipeline, no VLM, no CLIP) and re-scores them with
oracle / random / decomposed-oracle confidences. Boundaries and detection COUNT are
never touched, so every number is an admissible bound for any FM1/FM2-immune
confidence-only lever (which is what v107/v111/v115/v118/v119 and the v135 forward
path all are).

Artifacts consumed (verified present):
  results/batch_eval_v124_full/*.json   (66 files, per-video anomalies+frame_scores)
  results/batch_eval_v114_full/*.json   (66)
  results/batch_eval_v132_full/*.json   (63)
  /Users/zzl/Movies/ucf-crime/data      (parquet GT, test split)
"""
import sys, json, glob, random
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
THRESH = [0.3, 0.5, 0.7]


def load_dets(d):
    out = {}
    for jf in sorted(glob.glob(str(ROOT / d / '*.json'))):
        vid = Path(jf).stem
        j = json.load(open(jf))
        out[vid] = j
    return out


def iou(pred, gt):
    ps, pe = pred['t_start'] * FPS, pred['t_end'] * FPS
    gs, ge = gt
    ov = max(0.0, min(pe, ge) - max(ps, gs))
    un = max(pe, ge) - min(ps, gs)
    return ov / un if un > 0 else 0.0


def as_pred(dets):
    return {v: {'anomalies': j.get('anomalies', [])} for v, j in dets.items()}


def mp(gt, dets):
    r = compute_temporal_map(gt, as_pred(dets), fps=FPS, iou_thresholds=THRESH)
    return r


def clone(dets):
    return {v: {'anomalies': [dict(a) for a in j.get('anomalies', [])],
                'frame_scores': j.get('frame_scores'),
                'timestamps': j.get('timestamps')}
            for v, j in dets.items()}


def main():
    gt_all = load_ground_truth(LABEL, split='test')
    for dset in ['results/batch_eval_v124_full', 'results/batch_eval_v114_full']:
        dets = load_dets(dset)
        gt = {k: gt_all[k] for k in (set(gt_all) & set(dets))}
        print(f"\n================ {dset} ================")
        print(f"videos matched={len(gt)}  n_gt_segs={sum(len(v['segments']) for v in gt.values())}")
        ndet = sum(len(j.get('anomalies', [])) for v, j in dets.items() if v in gt)
        nvid_multi = sum(1 for v, j in dets.items()
                         if v in gt and len(j.get('anomalies', [])) > 1)
        print(f"n_detections={ndet}  videos_with_>1_det={nvid_multi}")

        base = mp(gt, dets)
        print(f"BASELINE          mAP={base['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={base[f'AP@{t}']:.5f}" for t in THRESH))

        # ---- best-IoU table per detection ----
        best = {}
        for v, j in dets.items():
            if v not in gt:
                continue
            segs = gt[v]['segments']
            best[v] = [max([iou(a, g) for g in segs], default=0.0)
                       for a in j.get('anomalies', [])]

        # recall ceiling: fraction of GT covered by ANY detection at each thresh
        for t in THRESH:
            cov = 0
            for v in gt:
                for gi, g in enumerate(gt[v]['segments']):
                    if any(iou(a, g) >= t for a in dets.get(v, {}).get('anomalies', [])):
                        cov += 1
            print(f"  recall-ceiling@{t}: {cov}/{sum(len(x['segments']) for x in gt.values())}"
                  f" = {cov/max(1,sum(len(x['segments']) for x in gt.values())):.3f}")

        # correlation conf vs bestIoU (global)
        cs, iw = [], []
        for v in best:
            for a, b in zip(dets[v]['anomalies'], best[v]):
                cs.append(a['confidence']); iw.append(b)
        cs, iw = np.array(cs), np.array(iw)
        def spearman(x, y):
            rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
            return float(np.corrcoef(rx, ry)[0, 1])
        print(f"  GLOBAL spearman(conf, bestIoU) = {spearman(cs, iw):+.4f}   n={len(cs)}")
        # within-video spearman (only multi-det videos)
        wsp = []
        for v in best:
            if len(best[v]) > 1 and np.std(best[v]) > 0 and np.std(
                    [a['confidence'] for a in dets[v]['anomalies']]) > 0:
                wsp.append(spearman(np.array([a['confidence'] for a in dets[v]['anomalies']]),
                                    np.array(best[v])))
        print(f"  WITHIN-VIDEO mean spearman = {np.mean(wsp):+.4f} over {len(wsp)} videos")

        # ---- ORACLE-FULL: conf = bestIoU (perfect global ordering) ----
        o = clone(dets)
        for v in best:
            for a, b in zip(o[v]['anomalies'], best[v]):
                a['confidence'] = float(b)
        r = mp(gt, o)
        print(f"ORACLE-FULL       mAP={r['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
              + f"   delta={r['mAP']-base['mAP']:+.5f}")

        # ---- ORACLE-WITHIN: permute each video's own confidences to match IoU order ----
        o = clone(dets)
        for v in best:
            confs = sorted([a['confidence'] for a in o[v]['anomalies']], reverse=True)
            order = np.argsort(-np.array(best[v]))
            for rank, idx in enumerate(order):
                o[v]['anomalies'][idx]['confidence'] = confs[rank]
        r = mp(gt, o)
        print(f"ORACLE-WITHIN     mAP={r['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
              + f"   delta={r['mAP']-base['mAP']:+.5f}")

        # ---- ORACLE-CROSSVIDEO: keep within-video ORDER, replace the video's
        # confidence LEVEL by the video's best IoU (all dets in a video get
        # level - eps*rank, preserving current intra order) ----
        o = clone(dets)
        for v in best:
            lvl = max(best[v], default=0.0)
            cur = [a['confidence'] for a in o[v]['anomalies']]
            rk = np.argsort(np.argsort(-np.array(cur)))  # 0 = current top
            for a, k in zip(o[v]['anomalies'], rk):
                a['confidence'] = float(lvl) - 1e-4 * float(k)
        r = mp(gt, o)
        print(f"ORACLE-CROSSVIDEO mAP={r['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
              + f"   delta={r['mAP']-base['mAP']:+.5f}")

        # ---- RANDOM ranking floor ----
        rr = []
        for seed in range(8):
            rnd = random.Random(seed)
            o = clone(dets)
            for v in o:
                for a in o[v]['anomalies']:
                    a['confidence'] = rnd.random()
            rr.append(mp(gt, o)['mAP'])
        print(f"RANDOM-RANK       mAP={np.mean(rr):.5f} +- {np.std(rr):.5f} (8 seeds)"
              f"   delta={np.mean(rr)-base['mAP']:+.5f}")

        # ---- COUNT oracle (FM2 reference, not a legal lever): keep best det/video
        o = clone(dets)
        for v in best:
            if len(best[v]) > 1:
                keep = int(np.argmax(best[v]))
                o[v]['anomalies'] = [o[v]['anomalies'][keep]]
        r = mp(gt, o)
        print(f"[ref] COUNT-ORACLE mAP={r['mAP']:.5f}   delta={r['mAP']-base['mAP']:+.5f}")

        # ---- per-video confidence z-score (a REAL candidate lever, param-free) ----
        for mode in ['zscore', 'rank', 'minmax_keeplevel']:
            o = clone(dets)
            for v in o:
                c = np.array([a['confidence'] for a in o[v]['anomalies']], dtype=float)
                if len(c) == 0:
                    continue
                if mode == 'zscore':
                    nc = (c - c.mean()) / (c.std() + 1e-6) if len(c) > 1 else np.zeros_like(c)
                elif mode == 'rank':
                    nc = (np.argsort(np.argsort(-c)) == 0).astype(float)
                else:
                    nc = c / (c.max() + 1e-9)
                for a, x in zip(o[v]['anomalies'], nc):
                    a['confidence'] = float(x)
            r = mp(gt, o)
            print(f"[cand] pervideo-{mode:<18} mAP={r['mAP']:.5f}"
                  f"   delta={r['mAP']-base['mAP']:+.5f}")


if __name__ == '__main__':
    main()
