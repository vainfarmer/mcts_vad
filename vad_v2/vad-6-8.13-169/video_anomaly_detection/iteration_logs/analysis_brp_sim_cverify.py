"""Agent C independent verification of v101 BRP sim claims.

Reproduces the official 65-video temporal mAP baseline from cached v97
predictions, then measures the BRP append-only recovery-proposal delta
under the PINNED semantics from the joint proposal:
  - mask (coverage of current final detections) used ONLY for seed selection
  - half-max walk NOT blocked by the mask
  - hm = (rel[t2] + p20(whole-video rel)) / 2
  - fire gate: rel[t2] >= gate * rel.max()
  - symmetric max_dur clamp about ts[t2]; min_segment extension
  - K <= 2, conf = 0.01 fixed bottom rank
Variants: gt-cat / pred-cat, native pkl grid / live pred-JSON grid,
random-placement control.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, compute_temporal_map

ROOT = Path('/Users/zzl/PycharmProjects/vad')
PRED_DIR = ROOT / 'results/batch_eval_v97_full'
CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_verify_curves.pkl'

CATS = ["Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting",
        "RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"]

# mirrored from pipeline.py (read-only use)
CATEGORY_DURATION_PRIORS = {
    'RoadAccidents': {'expected_duration': (2.0, 20.0), 'min_segment': 1.5},
    'Explosion':     {'expected_duration': (2.0, 10.0), 'min_segment': 1.5},
    'Abuse':         {'expected_duration': (2.0, 8.0),  'min_segment': 1.5},
    'Fighting':      {'expected_duration': (3.0, 15.0), 'min_segment': 2.0},
    'Shooting':      {'expected_duration': (3.0, 15.0), 'min_segment': 2.0},
    'Robbery':       {'expected_duration': (5.0, 30.0), 'min_segment': 3.0},
    'Burglary':      {'expected_duration': (5.0, 40.0), 'min_segment': 3.0},
    'Shoplifting':   {'expected_duration': (3.0, 20.0), 'min_segment': 2.0},
    'Stealing':      {'expected_duration': (5.0, 30.0), 'min_segment': 3.0},
    'Vandalism':     {'expected_duration': (3.0, 20.0), 'min_segment': 2.0},
    'Arson':         {'expected_duration': (5.0, 40.0), 'min_segment': 3.0},
    'Arrest':        {'expected_duration': (15.0, 90.0),'min_segment': 8.0},
    'Assault':       {'expected_duration': (10.0, 90.0),'min_segment': 5.0},
}


def brp(ts, rel, dets, cat, K=2, gate=0.5, conf=0.01, rng=None, random_ctrl=False):
    """Pinned BRP semantics. dets: list of (t_start, t_end). Returns appended dicts."""
    if not dets:
        return []  # empty-anomalies no-op (per proposal test list)
    ts = np.asarray(ts, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    priors = CATEGORY_DURATION_PRIORS[cat]
    max_dur = priors['expected_duration'][1]
    min_seg = priors['min_segment']
    mask = np.zeros(len(ts), dtype=bool)
    for (s, e) in dets:
        mask |= (ts >= s) & (ts <= e)
    p20 = float(np.percentile(rel, 20))
    out = []
    for _ in range(K):
        avail = ~mask
        if not avail.any():
            break
        if random_ctrl:
            idx_avail = np.where(avail)[0]
            t2 = int(rng.choice(idx_avail))
        else:
            t2 = int(np.where(avail)[0][np.argmax(rel[avail])])
        if rel[t2] < gate * rel.max():
            break
        hm = (rel[t2] + p20) / 2.0
        L = t2
        while L - 1 >= 0 and rel[L - 1] >= hm:
            L -= 1
        R = t2
        while R + 1 < len(ts) and rel[R + 1] >= hm:
            R += 1
        s, e = float(ts[L]), float(ts[R])
        if e - s > max_dur:  # symmetric clamp about peak time
            s = float(ts[t2]) - max_dur / 2.0
            e = float(ts[t2]) + max_dur / 2.0
        if e - s < min_seg:  # extend to min_segment about center
            mid = (s + e) / 2.0
            s = mid - min_seg / 2.0
            e = mid + min_seg / 2.0
        s = max(0.0, s)
        out.append({'t_start': s, 't_end': e, 'confidence': conf, 'anomaly_type': cat})
        mask |= (ts >= s) & (ts <= e)
    return out


def resolve_cat(entry, gt_cat, mode):
    """pred-cat: highest-confidence detection's anomaly_type, fallback video cat."""
    if mode == 'gt':
        return gt_cat if gt_cat in CATS else None
    dets = entry.get('anomalies') or []
    best = None
    for a in dets:
        t = a.get('anomaly_type')
        if t in CATS and (best is None or a.get('confidence', 0) > best.get('confidence', 0)):
            best = a
    if best is not None:
        return best['anomaly_type']
    # live fallback would be videomae_category; best on-disk proxy is the GT cat
    return gt_cat if gt_cat in CATS else None


def run(mode='gt', grid='pkl', K=2, gate=0.5, random_ctrl=False, seed=0):
    gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
    pred_all = load_predictions(str(PRED_DIR))
    curves = pickle.load(open(CACHE, 'rb'))
    rng = np.random.default_rng(seed)

    pred = {}
    for vid, entry in pred_all.items():
        if vid not in gt_all:
            continue
        entry = dict(entry)
        anoms = [dict(a) for a in entry.get('anomalies', [])]
        cur = curves.get(vid)
        if cur is not None:
            gt_cat = gt_all[vid]['category']
            cat = resolve_cat(entry, gt_cat, mode)
            if cat is not None:
                if grid == 'pkl':
                    ts = cur['ts']
                    rel = cur['rel'][cat]
                else:  # live grid: pred JSON timestamps; nearest-index map
                    ts = np.array(entry.get('timestamps') or [], dtype=np.float64)
                    if len(ts) >= 3:
                        src_ts = cur['ts']
                        src_rel = cur['rel'][cat]
                        idx = np.clip(np.searchsorted(src_ts, ts), 0, len(src_ts) - 1)
                        # nearest neighbor refinement
                        idx_l = np.clip(idx - 1, 0, len(src_ts) - 1)
                        choose_l = np.abs(src_ts[idx_l] - ts) < np.abs(src_ts[idx] - ts)
                        idx = np.where(choose_l, idx_l, idx)
                        rel = src_rel[idx]
                    else:
                        ts, rel = None, None
                if ts is not None and len(ts) >= 3:
                    dets = [(a['t_start'], a['t_end']) for a in anoms]
                    anoms.extend(brp(ts, rel, dets, cat, K=K, gate=gate,
                                     rng=rng, random_ctrl=random_ctrl))
        entry['anomalies'] = anoms
        pred[vid] = entry
    gt = {k: v for k, v in gt_all.items() if k in pred}
    return compute_temporal_map(gt, pred, fps=30)


def main():
    gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
    pred_all = {k: v for k, v in load_predictions(str(PRED_DIR)).items() if k in gt_all}
    gt = {k: v for k, v in gt_all.items() if k in pred_all}
    base = compute_temporal_map(gt, pred_all, fps=30)
    print('BASELINE:', {k: round(v, 8) for k, v in base.items()})
    assert abs(base['mAP'] - 0.16154893439350917) < 1e-12, 'baseline mismatch!'
    print('baseline exact-match: OK')

    for grid in ['pkl', 'live']:
        for mode in ['gt', 'pred']:
            r = run(mode=mode, grid=grid)
            print(f'BRP grid={grid:4s} cat={mode:4s}: mAP={r["mAP"]:.6f} '
                  f'(delta {r["mAP"]-base["mAP"]:+.6f}) '
                  f'AP03={r["AP@0.3"]:.6f}({r["AP@0.3"]-base["AP@0.3"]:+.6f}) '
                  f'AP05={r["AP@0.5"]:.8f}({r["AP@0.5"]-base["AP@0.5"]:+.2e}) '
                  f'AP07={r["AP@0.7"]:.8f}({r["AP@0.7"]-base["AP@0.7"]:+.2e})')

    # random control, 5 seeds, live grid + pred cat (the strictest variant)
    deltas = []
    for seed in range(5):
        r = run(mode='pred', grid='live', random_ctrl=True, seed=seed)
        deltas.append(r['mAP'] - base['mAP'])
    print('RANDOM-CTRL live/pred deltas:', [f'{d:+.6f}' for d in deltas],
          f'mean={np.mean(deltas):+.6f}')

    # K and gate sensitivity on the pinned variant (pred/live)
    for K in [1, 2, 3, 4]:
        r = run(mode='pred', grid='live', K=K)
        print(f'K={K}: delta={r["mAP"]-base["mAP"]:+.6f}')
    for g in [0.4, 0.5, 0.6]:
        r = run(mode='pred', grid='live', gate=g)
        print(f'gate={g}: delta={r["mAP"]-base["mAP"]:+.6f}')


if __name__ == '__main__':
    main()
