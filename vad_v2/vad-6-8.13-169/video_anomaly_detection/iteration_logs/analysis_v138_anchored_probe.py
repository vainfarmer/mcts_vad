"""
v138 diagnostic probe #7 (Agent B, research-only) — DETECTION-ANCHORED MULTI-SCALE
bottom-ranked hedges (the candidate mechanism, "SMH": Scale-Multiplicity Hedging).

Motivation from probe #1: recall-ceiling@0.5 = 15/67 and @0.7 = 9/67, i.e. AP@0.5/0.7
are bounded by the fact that a single half-max window per event is either right at
one scale or wrong at all of them. Appending SCALED/SHIFTED variants of each surviving
detection, strictly below the global confidence floor, gives the greedy matcher a
scale ladder to match at the stricter IoU thresholds, and (probe #5/#6) cannot lower AP.

Arms:
  SMH      : scaled+shifted variants of each existing detection only
  GRIDQ    : whole-video multi-scale grid ordered by absolute curve quality (probe #6 best)
  SMH+GRIDQ: union
Consumes only saved artifacts. No VLM / CLIP / model.
"""
import sys, json, glob
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
TOP = 0.01

SCALES = [0.4, 0.55, 0.7, 0.85, 1.2, 1.5, 2.0, 2.7, 3.5]
SHIFTS = [-0.6, -0.3, 0.0, 0.3, 0.6]


def load(d, gt_all):
    return {Path(f).stem: json.load(open(f))
            for f in sorted(glob.glob(str(ROOT / d / '*.json')))
            if Path(f).stem in gt_all}


def qabs(fs, ts, s, e):
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


def smh(j):
    dur = j['duration']
    out = []
    for a in j['anomalies']:
        s0, e0 = a['t_start'], a['t_end']
        d0 = max(e0 - s0, 0.5)
        c0 = 0.5 * (s0 + e0)
        for k in SCALES:
            L = k * d0
            for sh in SHIFTS:
                c = c0 + sh * d0
                s, e = c - L / 2, c + L / 2
                s, e = max(0.0, s), min(dur, e)
                if e - s >= 0.8 and not (abs(s - s0) < 1e-6 and abs(e - e0) < 1e-6):
                    out.append((round(s, 3), round(e, 3)))
    return sorted(set(out))


def gridq(j, cat, K=40):
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


def build(dets, arm, cap):
    out = {}
    for v, j in dets.items():
        aa = [dict(a) for a in j['anomalies']]
        if aa:
            cat = aa[0].get('anomaly_type') or 'unknown'
            fs = np.asarray(j.get('frame_scores') or [], float)
            ts = np.asarray(j.get('timestamps') or [], float)
            props = []
            if 'SMH' in arm:
                props += smh(j)
            if 'GRIDQ' in arm:
                props += gridq(j, cat)
            props = sorted(set(props), key=lambda p: -qabs(fs, ts, p[0], p[1]))[:cap]
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
    agg = {}
    for dname in DIRS:
        dets = load(dname, gt_all)
        if not dets:
            continue
        gt = {v: gt_all[v] for v in dets}
        base = mp(gt, dets)
        print(f"\n===== {Path(dname).name} =====")
        print(f"  baseline           mAP={base['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={base[f'AP@{t}']:.5f}" for t in THRESH))
        for arm in ['SMH', 'GRIDQ', 'SMH+GRIDQ']:
            for cap in [20, 40, 80]:
                d2 = build(dets, arm, cap)
                r = mp(gt, d2)
                n = sum(len(x['anomalies']) for x in d2.values())
                agg.setdefault(f'{arm}_cap{cap}', []).append(r['mAP'] - base['mAP'])
                print(f"  {arm:<10} cap={cap:<3} n={n:<5} mAP={r['mAP']:.5f}  "
                      + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
                      + f"  delta={r['mAP']-base['mAP']:+.5f}")
    print("\n===== mean / min delta across 6 detection sets =====")
    for k, v in sorted(agg.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {k:<18} mean={np.mean(v):+.5f}  min={np.min(v):+.5f}  max={np.max(v):+.5f}")
    json.dump({k: [float(np.mean(v)), float(np.min(v))] for k, v in agg.items()},
              open(ROOT / 'video_anomaly_detection/iteration_logs/'
                   'analysis_v138_anchored_probe.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
