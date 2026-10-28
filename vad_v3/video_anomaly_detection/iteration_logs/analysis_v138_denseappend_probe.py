"""
v138 diagnostic probe #5 (Agent B, research-only) — DENSE BOTTOM-RANKED PROPOSAL
APPEND. Tests the monotone-non-decreasing property of evaluate._compute_ap under
appending detections STRICTLY BELOW the global confidence floor, and measures the
realizable mAP gain as a function of K (proposals appended per video) and of the
proposal generator (dense multi-scale grid vs unclaimed-CLIP-peak vs random).

Monotonicity argument being tested empirically:
  _compute_ap (evaluate.py:383-389) = (1/11) * sum_t max(precision[recall>=t]).
  Appending predictions after all existing ones (a) never removes an index from
  {recall>=t}, only adds tail indices, and max over a superset >= max over subset;
  (b) can only extend the reachable recall range. => AP non-decreasing.

Consumes only saved artifacts (5 detection sets + parquet GT). No VLM/CLIP/model.
"""
import sys, json, glob, random
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import (load_ground_truth, compute_temporal_map,  # noqa
                      compute_frame_auc, compute_video_level)
from pipeline import CATEGORY_DURATION_PRIORS as PRIORS  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS, THRESH = 30, [0.3, 0.5, 0.7]
DIRS = ['results/batch_eval_v124_full', 'results/batch_eval_v131_full',
        'results/batch_eval_v130_full', 'results/batch_eval_v132_full',
        'results/batch_eval_v114_full', 'results/batch_eval_v108_full']
CONF = 0.01   # v101 pinned bottom-rank confidence (real dets >= 0.25 after gate)


def load(dirname, gt_all):
    out = {}
    for jf in sorted(glob.glob(str(ROOT / dirname / '*.json'))):
        v = Path(jf).stem
        if v in gt_all:
            out[v] = json.load(open(jf))
    return out


def mp(gt, dets):
    return compute_temporal_map(
        gt, {v: {'anomalies': j['anomalies']} for v, j in dets.items()},
        fps=FPS, iou_thresholds=THRESH)


# ------------------------- proposal generators -------------------------------
def gen_grid(j, cat, K):
    """Dense multi-scale anchor grid over the whole video (signal-free, BSN/BMN
    style dense anchors). Scales from the category duration prior."""
    dur = j['duration']
    p = PRIORS.get(cat) or {'expected_duration': (3.0, 30.0)}
    lo, hi = p['expected_duration']
    scales = [lo, np.sqrt(lo * hi), hi, min(dur, hi * 2)]
    props = []
    for L in scales:
        L = float(min(max(L, 1.0), dur))
        stride = L / 2.0
        t = 0.0
        while t + L <= dur + 1e-6:
            props.append((t, t + L))
            t += stride
    # de-dup, then take a uniform subsample of size K (deterministic)
    props = sorted(set([(round(a, 3), round(b, 3)) for a, b in props]))
    if not props:
        return []
    if len(props) > K:
        idx = np.linspace(0, len(props) - 1, K).astype(int)
        props = [props[i] for i in idx]
    return props


def gen_curve(j, cat, K):
    """Unclaimed-peak half-max proposals off the SAVED output curve (v101 BRP,
    generalised to K)."""
    fs = np.asarray(j.get('frame_scores') or [], float)
    ts = np.asarray(j.get('timestamps') or [], float)
    if fs.size < 4 or ts.size != fs.size:
        return []
    p = PRIORS.get(cat) or {'expected_duration': (3.0, 30.0)}
    max_dur = p['expected_duration'][1]
    masked = np.zeros(fs.size, bool)
    for a in j['anomalies']:
        masked |= (ts >= a['t_start']) & (ts <= a['t_end'])
    base = float(np.percentile(fs, 20))
    props = []
    for _ in range(K):
        free = np.where(~masked)[0]
        if free.size == 0:
            break
        seed = free[int(np.argmax(fs[free]))]
        hm = (fs[seed] + base) / 2.0
        L = seed
        while L - 1 >= 0 and fs[L - 1] >= hm:
            L -= 1
        R = seed
        while R + 1 < fs.size and fs[R + 1] >= hm:
            R += 1
        t_s, t_e = float(ts[L]), float(ts[R])
        if t_e - t_s > max_dur:
            c = float(ts[seed])
            t_s, t_e = max(0.0, c - max_dur / 2), min(j['duration'], c + max_dur / 2)
        if t_e - t_s < 1.0:
            t_e = min(j['duration'], t_s + 1.0)
        props.append((t_s, t_e))
        masked |= (ts >= t_s) & (ts <= t_e)
    return props


def gen_random(j, cat, K, seed=0):
    rnd = random.Random(hash(j['video_path']) ^ seed)
    dur = j['duration']
    p = PRIORS.get(cat) or {'expected_duration': (3.0, 30.0)}
    lo, hi = p['expected_duration']
    out = []
    for _ in range(K):
        L = rnd.uniform(lo, min(hi, max(lo + 1, dur)))
        s = rnd.uniform(0, max(0.0, dur - L))
        out.append((s, s + L))
    return out


def append(dets, genfn, K):
    out = {}
    for v, j in dets.items():
        aa = [dict(a) for a in j['anomalies']]
        if aa:  # never create detections for a video that had none (is_normal safe)
            cat = aa[0].get('anomaly_type') or 'unknown'
            for i, (s, e) in enumerate(genfn(j, cat, K)):
                aa.append({'t_start': float(s), 't_end': float(e),
                           'confidence': CONF - 1e-6 * i,
                           'anomaly_type': cat, 'depth': None, 'visits': None})
        out[v] = dict(j, anomalies=aa)
    return out


def main():
    gt_all = load_ground_truth(LABEL, split='test')
    print("gen        K     " + '  '.join(f"{d.split('_')[-2]:>9}" for d in DIRS))
    for dname in DIRS:
        dets = load(dname, gt_all)
        if not dets:
            continue
        gt = {v: gt_all[v] for v in dets}
        base = mp(gt, dets)
        nd = sum(len(j['anomalies']) for j in dets.values())
        print(f"\n===== {Path(dname).name} ({len(dets)} videos, {nd} dets) =====")
        print(f"  baseline            mAP={base['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={base[f'AP@{t}']:.5f}" for t in THRESH))
        for gname, gfn in [('grid', gen_grid), ('curve', gen_curve), ('random', gen_random)]:
            for K in [2, 5, 10, 20, 40, 80]:
                d2 = append(dets, gfn, K)
                r = mp(gt, d2)
                n2 = sum(len(j['anomalies']) for j in d2.values())
                flag = '' if r['mAP'] >= base['mAP'] - 1e-12 else '  <<< NON-MONOTONE!'
                print(f"  {gname:<6} K={K:<3} n={n2:<5} mAP={r['mAP']:.5f}  "
                      + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
                      + f"  delta={r['mAP']-base['mAP']:+.5f}{flag}")

    # ---- side-metric invariance check on v124 (frame AUC / video level) ----
    print("\n--- side-metric invariance (v124, grid K=40) ---")
    dets = load('results/batch_eval_v124_full', gt_all)
    gt = {v: gt_all[v] for v in dets}
    def full(d):
        pr = {v: {'anomalies': j['anomalies'], 'duration': j['duration'],
                  'frame_scores': j.get('frame_scores'),
                  'timestamps': j.get('timestamps')} for v, j in d.items()}
        return (compute_frame_auc(gt, pr, FPS)['auc'], compute_video_level(gt, pr))
    a0, v0 = full(dets)
    a1, v1 = full(append(dets, gen_grid, 40))
    print(f"  frame AUC  {a0:.6f} -> {a1:.6f}   (delta {a1-a0:+.6f})")
    print(f"  video F1   {v0['f1']:.6f} -> {v1['f1']:.6f}   "
          f"video AUC {v0['auc']:.6f} -> {v1['auc']:.6f}")


if __name__ == '__main__':
    main()
