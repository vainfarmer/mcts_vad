"""v142 BOUNDARY-GEOMETRY oracle battery (offline, ZERO API, FAITHFUL BY CONSTRUCTION).

Substrate = the cached prediction JSONs' own (t_start, t_end, confidence). Every arm is a pure
geometric transform of those numbers, so the offline computation IS the in-vivo computation
(modulo VLM run-to-run noise). No CLIP curve, no features/*.npy => FM15 cannot apply.

Answers:
 A) checkpoint census: hits at IoU>=.3/.5/.7 and which of the 9 attainable AP checkpoints bind.
 B) oracle ceilings for boundary-only edits (count & confidence & anchor preserved).
 C) blind geometric arms: additive dilation, target-duration snap, asymmetric growth.
 D) video-level bootstrap CI for the best arm.
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]


def load(rd):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(rd))
    gt = {k: v for k, v in gt.items() if k in pred and gt[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}
    return gt, pred


def mAP(gt, pred, vids=None):
    if vids is None:
        return compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)
    # bootstrap: allow duplicated videos by suffixing keys
    g2, p2 = {}, {}
    for n, v in enumerate(vids):
        g2[f'{v}#{n}'] = gt[v]; p2[f'{v}#{n}'] = pred[v]
    return compute_temporal_map(g2, p2, fps=FPS, iou_thresholds=THR)


def xf(pred, gt, fn):
    """apply boundary transform fn(det, gtsegs, dur)->(s,e); count/conf preserved."""
    out = {}
    for v, e in pred.items():
        dur = e.get('duration') or 0.0
        segs = [(s[0] / FPS, s[1] / FPS) for s in gt[v]['segments']]
        na = []
        for a in e['anomalies']:
            s, t = fn(a, segs, dur)
            if dur > 0:
                s = max(0.0, min(s, dur)); t = max(0.0, min(t, dur))
            if t <= s: s, t = a['t_start'], a['t_end']
            b = dict(a); b['t_start'] = float(s); b['t_end'] = float(t); na.append(b)
        o = dict(e); o['anomalies'] = na; out[v] = o
    return out


def census(gt, pred, tag):
    ious = []
    for v, e in pred.items():
        for a in e['anomalies']:
            ious.append(max([_temporal_iou(a, s, FPS) for s in gt[v]['segments']] or [0.0]))
    ious = np.array(ious); ng = sum(len(g['segments']) for g in gt.values())
    r = mAP(gt, pred)
    print(f'  [{tag}] mAP={r["mAP"]:.5f} AP@.3={r["AP@0.3"]:.4f} AP@.5={r["AP@0.5"]:.4f} '
          f'AP@.7={r["AP@0.7"]:.4f} | ndet={len(ious)} ngt={ng} '
          f'hits>=.3={int((ious>=.3).sum())} >=.5={int((ious>=.5).sum())} >=.7={int((ious>=.7).sum())} '
          f'| in[.1,.3)={int(((ious>=.1)&(ious<.3)).sum())} [.3,.5)={int(((ious>=.3)&(ious<.5)).sum())} '
          f'[.5,.7)={int(((ious>=.5)&(ious<.7)).sum())} zero={int((ious<=0).sum())}')
    return r['mAP']


ARMS = {}
ARMS['base'] = lambda a, g, d: (a['t_start'], a['t_end'])
# --- oracle: perfect boundaries for dets that already touch a GT (anchor kept) ---
def _or_touch(a, g, d):
    best, bi = 0.0, -1
    for i, s in enumerate(g):
        iou = _temporal_iou(a, (s[0]*FPS, s[1]*FPS), FPS)
        if iou > best: best, bi = iou, i
    return g[bi] if bi >= 0 else (a['t_start'], a['t_end'])
ARMS['ORACLE_touch(perfect bounds where IoU>0)'] = _or_touch
# --- oracle: perfect boundaries only where already IoU>=0.3 (upgrade to 1.0) ---
def _or_hi(a, g, d):
    best, bi = 0.0, -1
    for i, s in enumerate(g):
        iou = _temporal_iou(a, (s[0]*FPS, s[1]*FPS), FPS)
        if iou > best: best, bi = iou, i
    return g[bi] if best >= 0.3 else (a['t_start'], a['t_end'])
ARMS['ORACLE_hi(.3+ -> 1.0)'] = _or_hi
# --- oracle: perfect DURATION only, center kept (isolates duration vs offset error) ---
def _or_dur(a, g, d):
    best, bi = 0.0, -1
    for i, s in enumerate(g):
        iou = _temporal_iou(a, (s[0]*FPS, s[1]*FPS), FPS)
        if iou > best: best, bi = iou, i
    if bi < 0: return (a['t_start'], a['t_end'])
    c = 0.5 * (a['t_start'] + a['t_end']); L = g[bi][1] - g[bi][0]
    return (c - L / 2, c + L / 2)
ARMS['ORACLE_dur(perfect duration,center kept)'] = _or_dur
# --- oracle: perfect CENTER only, duration kept ---
def _or_cen(a, g, d):
    best, bi = 0.0, -1
    for i, s in enumerate(g):
        iou = _temporal_iou(a, (s[0]*FPS, s[1]*FPS), FPS)
        if iou > best: best, bi = iou, i
    if bi < 0: return (a['t_start'], a['t_end'])
    c = 0.5 * (g[bi][0] + g[bi][1]); L = a['t_end'] - a['t_start']
    return (c - L / 2, c + L / 2)
ARMS['ORACLE_cen(perfect center,dur kept)'] = _or_cen
# --- blind arms ---
for s in (2, 4, 6, 8, 12, 20):
    ARMS[f'dilate+-{s}s'] = (lambda s: lambda a, g, d: (a['t_start'] - s, a['t_end'] + s))(s)
for D in (12, 18, 24, 30, 40):
    ARMS[f'snapdur>={D}s'] = (lambda D: lambda a, g, d: (
        (a['t_start'], a['t_end']) if (a['t_end'] - a['t_start']) >= D else
        (0.5*(a['t_start']+a['t_end']) - D/2, 0.5*(a['t_start']+a['t_end']) + D/2)))(D)
for f in (1.5, 2.0, 3.0):
    ARMS[f'scale x{f}'] = (lambda f: lambda a, g, d: (
        0.5*(a['t_start']+a['t_end']) - f*(a['t_end']-a['t_start'])/2,
        0.5*(a['t_start']+a['t_end']) + f*(a['t_end']-a['t_start'])/2))(f)


def main(dirs, boot_arms, nboot=400):
    allres = {}
    for rd in dirs:
        gt, pred = load(rd)
        print(f'\n=== {rd.name}: {len(pred)} vids')
        base = census(gt, pred, 'base')
        for name, fn in ARMS.items():
            if name == 'base': continue
            m = census(gt, xf(pred, gt, fn), name)
            allres.setdefault(name, []).append(m - base)
    print('\n=== mean dmAP across caches ===')
    for k, v in sorted(allres.items(), key=lambda x: -np.mean(x[1])):
        print(f'  {k:42s} mean={np.mean(v):+.5f}  n+={sum(1 for x in v if x>0)}/{len(v)}  {[round(x,4) for x in v]}')
    # bootstrap on first dir
    gt, pred = load(dirs[0]); vids = sorted(pred)
    rng = np.random.default_rng(7)
    for name in boot_arms:
        pn = xf(pred, gt, ARMS[name]); ds = []
        for _ in range(nboot):
            vv = list(rng.choice(vids, size=len(vids), replace=True))
            ds.append(mAP(gt, pn, vv)['mAP'] - mAP(gt, pred, vv)['mAP'])
        ds = np.array(ds)
        print(f'  BOOT[{name}] on {dirs[0].name}: mean={ds.mean():+.5f} '
              f'90%CI=[{np.percentile(ds,5):+.4f},{np.percentile(ds,95):+.4f}] P(d>0)={(ds>0).mean():.2f}')


if __name__ == '__main__':
    args = sys.argv[1:]
    boot = [a.split('boot=')[1] for a in args if a.startswith('boot=')]
    dirs = [Path(a) for a in args if not a.startswith('boot=')]
    main(dirs or [ROOT / 'results/batch_eval_v124_full'], boot)
