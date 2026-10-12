"""
v138 diagnostic probe #6 (Agent B, research-only) — INTRA-TAIL ORDERING of the
dense bottom-ranked proposal append. All appended proposals live strictly below the
global confidence floor, so AP is monotone non-decreasing regardless of their order;
but the ORDER WITHIN the appended band sets the precision at the newly-reached
recall points, i.e. it is exactly where SCORE QUALITY converts into mAP.

Arms:
  ord-insert : generator order (probe #5 default)
  ord-qcurve : localization-quality score on the SAVED output curve (v111 Q components,
               computed on an ABSOLUTE cross-video scale -- no within-video min-max)
  ord-oracle : max IoU with GT (upper bound of this mechanism, NOT a legal lever)
Generators: dense multi-scale grid (deterministic) and hash-seeded pseudo-random.

Consumes only saved artifacts. No VLM / CLIP / model.
"""
import sys, json, glob, random
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map  # noqa
from pipeline import CATEGORY_DURATION_PRIORS as PRIORS  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS, THRESH = 30, [0.3, 0.5, 0.7]
DIRS = ['results/batch_eval_v124_full', 'results/batch_eval_v131_full',
        'results/batch_eval_v130_full', 'results/batch_eval_v132_full',
        'results/batch_eval_v114_full', 'results/batch_eval_v108_full']
TOP = 0.01     # top of the appended band; every real detection is >= 0.25


def load(d, gt_all):
    return {Path(f).stem: json.load(open(f))
            for f in sorted(glob.glob(str(ROOT / d / '*.json')))
            if Path(f).stem in gt_all}


def iou_t(s, e, g):
    ps, pe, gs, ge = s * FPS, e * FPS, g[0], g[1]
    ov = max(0.0, min(pe, ge) - max(ps, gs)); un = max(pe, ge) - min(ps, gs)
    return ov / un if un > 0 else 0.0


def gen_grid(j, cat, K):
    dur = j['duration']
    lo, hi = (PRIORS.get(cat) or {'expected_duration': (3.0, 30.0)})['expected_duration']
    props = []
    for L in [lo, float(np.sqrt(lo * hi)), hi, min(dur, hi * 2)]:
        L = float(min(max(L, 1.0), dur)); t = 0.0
        while t + L <= dur + 1e-6:
            props.append((round(t, 3), round(t + L, 3))); t += L / 2.0
    props = sorted(set(props))
    if len(props) > K:
        props = [props[i] for i in np.linspace(0, len(props) - 1, K).astype(int)]
    return props


def gen_random(j, cat, K):
    rnd = random.Random(abs(hash(j['video_path'])) % (2 ** 31))
    dur = j['duration']
    lo, hi = (PRIORS.get(cat) or {'expected_duration': (3.0, 30.0)})['expected_duration']
    out = []
    for _ in range(K):
        L = rnd.uniform(lo, min(max(hi, lo + 1), max(dur, lo + 1)))
        s = rnd.uniform(0, max(0.0, dur - L))
        out.append((s, s + L))
    return out


def qabs(j, s, e):
    """ABSOLUTE (cross-video comparable) localization quality of interval [s,e] on
    the saved output curve: peak-above-background contrast x centre fit."""
    fs = np.asarray(j.get('frame_scores') or [], float)
    ts = np.asarray(j.get('timestamps') or [], float)
    if fs.size < 3 or ts.size != fs.size:
        return 0.0
    m = (ts >= s) & (ts <= e)
    if m.sum() < 1:
        return 0.0
    pk = float(fs[m].max())
    bg = float(fs[~m].mean()) if (~m).any() else float(fs.min())
    pk_i = int(np.where(m)[0][int(np.argmax(fs[m]))])
    ctr = float(np.clip(1 - 2 * abs(ts[pk_i] - 0.5 * (s + e)) / max(e - s, 1e-6), 0, 1))
    return float(np.clip(pk - bg, 0, 1)) * (0.5 + 0.5 * ctr)


def build(dets, gt_all, gname, K, order):
    gfn = {'grid': gen_grid, 'random': gen_random}[gname]
    out = {}
    for v, j in dets.items():
        aa = [dict(a) for a in j['anomalies']]
        if aa:
            cat = aa[0].get('anomaly_type') or 'unknown'
            props = gfn(j, cat, K)
            if order == 'qcurve':
                props = sorted(props, key=lambda p: -qabs(j, p[0], p[1]))
            elif order == 'oracle':
                segs = gt_all[v]['segments']
                props = sorted(props, key=lambda p: -max(
                    [iou_t(p[0], p[1], g) for g in segs], default=0.0))
            for i, (s, e) in enumerate(props):
                aa.append({'t_start': float(s), 't_end': float(e),
                           'confidence': TOP - 1e-7 * i, 'anomaly_type': cat})
        out[v] = dict(j, anomalies=aa)
    return out


def mp(gt, d):
    return compute_temporal_map(gt, {v: {'anomalies': j['anomalies']} for v, j in d.items()},
                                fps=FPS, iou_thresholds=THRESH)


def main():
    gt_all = load_ground_truth(LABEL, split='test')
    summary = {}
    for dname in DIRS:
        dets = load(dname, gt_all)
        if not dets:
            continue
        gt = {v: gt_all[v] for v in dets}
        base = mp(gt, dets)
        print(f"\n===== {Path(dname).name} =====  baseline mAP={base['mAP']:.5f}")
        for gname in ['grid', 'random']:
            for K in [20, 40, 100]:
                line = f"  {gname:<6} K={K:<4}"
                for order in ['insert', 'qcurve', 'oracle']:
                    r = mp(gt, build(dets, gt_all, gname, K, order))
                    line += f"  {order}={r['mAP']:.5f}({r['mAP']-base['mAP']:+.4f})"
                    summary.setdefault(f'{gname}_K{K}_{order}', []).append(
                        r['mAP'] - base['mAP'])
                print(line)
    print("\n===== mean delta across all detection sets =====")
    for k, v in sorted(summary.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {k:<26} mean={np.mean(v):+.5f}  min={np.min(v):+.5f}  n={len(v)}")
    json.dump({k: float(np.mean(v)) for k, v in summary.items()},
              open(ROOT / 'video_anomaly_detection/iteration_logs/'
                   'analysis_v138_tailorder_probe.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
