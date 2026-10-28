"""v138 SCRATCH ANALYSIS (Agent A, research only — no pipeline code touched).

Question: how much temporal-mAP headroom exists in the CONFIDENCE RANKING alone,
given v124/v132's already-fixed detections (boundaries + count frozen)?

Decomposes the ceiling of any pure post-hoc re-ranking into:
  - ORACLE-GLOBAL   : conf := best IoU vs GT               (absolute ceiling)
  - ORACLE-WITHIN   : within each video, permute the EXISTING confidences so their
                      order matches IoU order (video-level scale untouched)
                      -> ceiling of intra-video re-ranking (v111/v115/v118/v119 family)
  - ORACLE-CROSS    : keep within-video order, replace each video's confidence LEVEL
                      by that video's best IoU (order within video preserved by
                      monotone remap) -> ceiling of cross-video calibration
  - RANDOM          : conf := U(0,1)                       (value of current ranking)
  - CONST           : conf := 1.0                          (no ranking at all)
Plus confidence distribution / variance decomposition and Spearman(conf, IoU).
"""
import sys, json, glob
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map, _temporal_iou  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30


def load_pred(result_dir):
    preds = {}
    for jf in sorted(glob.glob(str(Path(result_dir) / '*.json'))):
        d = json.load(open(jf))
        vid = Path(d.get('video_path', jf)).stem
        preds[vid] = d
    return preds


def best_iou(anom, gt_segs):
    if not gt_segs:
        return 0.0
    return max(_temporal_iou(anom, g, FPS) for g in gt_segs)


def mk(preds, fn):
    """fn(video_id, anomalies, gt_segs) -> list of new confidences"""
    out = {}
    for vid, d in preds.items():
        anoms = [dict(a) for a in d.get('anomalies', [])]
        out[vid] = {'anomalies': anoms, 'duration': d.get('duration', 0)}
    return out


def evaluate_conf(preds, gt, conf_fn):
    p2 = {}
    for vid, d in preds.items():
        anoms = [dict(a) for a in d.get('anomalies', [])]
        gt_segs = gt.get(vid, {}).get('segments', [])
        new = conf_fn(vid, anoms, gt_segs)
        for a, c in zip(anoms, new):
            a['confidence'] = float(c)
        p2[vid] = {'anomalies': anoms, 'duration': d.get('duration', 0)}
    return compute_temporal_map(gt, p2, fps=FPS)


def main(result_dir, tag):
    gt = load_ground_truth(LABEL, split='test')
    preds = load_pred(result_dir)
    # restrict gt to evaluated videos (evaluate() does the same via load_predictions keys)
    gt = {k: v for k, v in gt.items() if k in preds}
    print(f"\n===== {tag}: {result_dir} =====")
    print(f"videos={len(preds)}  gt_videos={len(gt)}")

    rows = []
    for vid, d in preds.items():
        for a in d.get('anomalies', []):
            rows.append((vid, a['confidence'], best_iou(a, gt.get(vid, {}).get('segments', []))))
    confs = np.array([r[1] for r in rows])
    ious = np.array([r[2] for r in rows])
    ndet = len(rows)
    n_gt = sum(len(v['segments']) for v in gt.values())
    print(f"detections={ndet}  gt_segments={n_gt}  "
          f"videos_with_det={len(set(r[0] for r in rows))}")
    print(f"conf: min={confs.min():.4f} p25={np.percentile(confs,25):.4f} "
          f"med={np.median(confs):.4f} p75={np.percentile(confs,75):.4f} "
          f"max={confs.max():.4f} std={confs.std():.4f} n_unique={len(np.unique(np.round(confs,6)))}")
    print(f"iou : mean={ious.mean():.4f} frac>=0.3={np.mean(ious>=0.3):.3f} "
          f"frac>=0.5={np.mean(ious>=0.5):.3f} frac>=0.7={np.mean(ious>=0.7):.3f} "
          f"frac==0={np.mean(ious<=1e-9):.3f}")

    # variance decomposition of confidence: between-video vs within-video
    byv = {}
    for v, c, i in rows:
        byv.setdefault(v, []).append((c, i))
    grand = confs.mean()
    ss_between = sum(len(x) * (np.mean([c for c, _ in x]) - grand) ** 2 for x in byv.values())
    ss_within = sum(sum((c - np.mean([cc for cc, _ in x])) ** 2 for c, _ in x) for x in byv.values())
    ss_tot = ss_between + ss_within
    print(f"conf variance: between-video={ss_between/ss_tot:.3f}  within-video={ss_within/ss_tot:.3f}")
    n_multi = sum(1 for x in byv.values() if len(x) > 1)
    print(f"videos with >1 detection: {n_multi}/{len(byv)}  "
          f"(detections in multi-det videos: {sum(len(x) for x in byv.values() if len(x)>1)}/{ndet})")

    # Spearman
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        if ra.std() == 0 or rb.std() == 0:
            return float('nan')
        return float(np.corrcoef(ra, rb)[0, 1])
    print(f"GLOBAL spearman(conf, iou) = {spearman(confs, ious):.4f}")
    ws = [spearman(np.array([c for c, _ in x]), np.array([i for _, i in x]))
          for x in byv.values() if len(x) > 1]
    ws = [w for w in ws if not np.isnan(w)]
    print(f"WITHIN-video spearman(conf,iou): mean={np.mean(ws):+.4f} n={len(ws)}")

    # ---- arms ----
    def f_base(vid, anoms, gs):
        return [a['confidence'] for a in anoms]

    def f_oracle_global(vid, anoms, gs):
        return [best_iou(a, gs) for a in anoms]

    def f_oracle_within(vid, anoms, gs):
        cs = np.array([a['confidence'] for a in anoms], dtype=float)
        iv = np.array([best_iou(a, gs) for a in anoms], dtype=float)
        # assign the video's own confidence VALUES to detections in IoU order
        order = np.argsort(-iv)  # best IoU first
        sorted_conf = np.sort(cs)[::-1]
        out = np.empty_like(cs)
        out[order] = sorted_conf
        return out

    def f_oracle_cross(vid, anoms, gs):
        cs = np.array([a['confidence'] for a in anoms], dtype=float)
        iv = np.array([best_iou(a, gs) for a in anoms], dtype=float)
        level = iv.max() if len(iv) else 0.0
        # preserve within-video order, set video level to oracle level
        if len(cs) > 1 and cs.max() - cs.min() > 1e-12:
            rel = (cs - cs.min()) / (cs.max() - cs.min())
        else:
            rel = np.ones_like(cs)
        return level * (0.90 + 0.10 * rel)

    rng = np.random.RandomState(0)

    def f_random(vid, anoms, gs):
        return rng.rand(len(anoms))

    def f_const(vid, anoms, gs):
        return np.ones(len(anoms))

    arms = [('BASELINE', f_base), ('ORACLE-GLOBAL', f_oracle_global),
            ('ORACLE-WITHIN', f_oracle_within), ('ORACLE-CROSS', f_oracle_cross),
            ('CONST-1.0', f_const)]
    print(f"\n{'arm':<16} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  delta")
    base_map = None
    res = {}
    for name, fn in arms:
        tm = evaluate_conf(preds, gt, fn)
        if base_map is None:
            base_map = tm['mAP']
        res[name] = tm
        print(f"{name:<16} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f}  {tm['mAP']-base_map:+.5f}")
    rmaps = []
    for s in range(8):
        rng = np.random.RandomState(s)
        tm = evaluate_conf(preds, gt, f_random)
        rmaps.append(tm['mAP'])
    print(f"{'RANDOM(x8)':<16} {np.mean(rmaps):>9.5f} "
          f"(sd={np.std(rmaps):.5f}, min={min(rmaps):.5f}, max={max(rmaps):.5f})"
          f"  {np.mean(rmaps)-base_map:+.5f}")
    res['RANDOM_mean'] = {'mAP': float(np.mean(rmaps))}
    return res


if __name__ == '__main__':
    out = {}
    for d, t in [(ROOT / 'results/batch_eval_v124_full', 'v124'),
                 (ROOT / 'results/batch_eval_v132_full', 'v132'),
                 (ROOT / 'results/batch_eval_v130_full', 'v130')]:
        if Path(d).exists():
            out[t] = {k: (v if isinstance(v, dict) else v) for k, v in main(str(d), t).items()}
    json.dump({k: {kk: vv['mAP'] for kk, vv in v.items()} for k, v in out.items()},
              open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v138_confbase_ceiling.json', 'w'),
              indent=2)
