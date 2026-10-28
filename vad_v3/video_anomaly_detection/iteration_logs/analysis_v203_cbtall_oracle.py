"""v203 CBT-ALL — Phase-Oracle ceiling bound (FM13) at the MANDATORY full-290
harness, reconstructed OFFLINE from the authoritative committed run
`results/batch_eval_v26_full290` (290/290, 0 failures, citable mAP 0.16682).

WHAT THIS BOUNDS. CBT-ALL extends the CBT caps dict KEY SET from the 4 hard
categories to all 13, with caps DERIVED from
CATEGORY_DURATION_PRIORS[cat]['expected_duration'][1] (pipeline.CBT_ALL_EXTRA_CAPS).
The operator is a MONOTONE-INWARD clamp: it can only ever replace a FINAL
detection window by a strictly narrower SUB-window, never move or grow it, and
never touches confidence or the detection count. Therefore the BEST POSSIBLE
version of this operator is: for every det whose resolved anomaly_type is one of
the 9 extended categories AND whose duration exceeds trigger * cap, pick the
inward sub-window that MAXIMISES IoU against that video's GT. That is what this
script computes.

The GT is used ONLY as the oracle selector (an upper bound), NEVER to pick the
shipped caps — those come from the duration priors alone.

Two oracle variants:
  O1 (loose)  : any inward sub-window with dur >= CATEGORY_DURATION_PRIORS min_segment
  O2 (method) : additionally dur <= cap (the real method's max half-width is
                int(cap*fps/2), so its widest emitted window is ~cap)

Run:  python3.14 video_anomaly_detection/iteration_logs/analysis_v203_cbtall_oracle.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'video_anomaly_detection'))

import evaluate  # noqa: E402
from pipeline import (CBT_ALL_EXTRA_CAPS, CATEGORY_DURATION_PRIORS,  # noqa: E402
                      CBT_CAPS)

RUN = str(REPO / 'results' / 'batch_eval_v26_full290')
GT_DIR = '/Users/zzl/Movies/ucf-crime/data'
TRIG = 1.1          # committed cbt_recal_trigger
GT_FPS = 30         # evaluate.compute_temporal_map default (GT segs are frames)
IOU_THR = [0.3, 0.5, 0.7]
GRID = 0.25         # seconds; finer than the 0.5s (2fps) timestamp grid


def iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def best_inward(win, gts, min_seg, width_cap):
    """Oracle sub-window: max IoU over all inward sub-windows on a GRID that
    satisfy min_seg <= dur < original dur (and dur <= width_cap when given)."""
    t_s, t_e = win
    det_dur = t_e - t_s
    best = (iou(win, g) for g in gts)
    best_iou = max(best) if gts else 0.0
    best_win = None
    n = int(round(det_dur / GRID))
    for i in range(n + 1):
        s = t_s + i * GRID
        for j in range(i + 1, n + 1):
            e = t_s + j * GRID
            dur = e - s
            if dur < min_seg or dur >= det_dur:
                continue
            if width_cap is not None and dur > width_cap:
                continue
            v = max(iou((s, e), g) for g in gts)
            if v > best_iou:
                best_iou, best_win = v, (s, e)
    return best_win, best_iou


def main():
    gt = evaluate.load_ground_truth(GT_DIR, split='test')
    pred = evaluate.load_predictions(RUN)
    missing = [k for k, v in gt.items() if v.get('segments') and k not in pred]
    print(f'run={os.path.basename(RUN)}  gt_videos={len(gt)}  pred_videos={len(pred)}')
    print(f'COVERAGE GATE: missing anomaly videos = {len(missing)} (must be 0)')

    base = evaluate.compute_temporal_map(gt, pred, iou_thresholds=IOU_THR)['mAP']
    print(f'baseline full-290 mAP = {base:.5f}')

    stats = {'addressable': 0, 'on_anomaly_video': 0, 'improved_o1': 0,
             'improved_o2': 0, 'hard88_video': 0}
    HARD88 = {'Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion'}
    import copy
    for name, variant in (('O1_loose', None), ('O2_method_width', 'cap')):
        p2 = copy.deepcopy(pred)
        n_fire = 0
        for vid, rec in p2.items():
            # GT segments are stored in 30fps FRAME numbers (evaluate._temporal_iou
            # multiplies the PREDICTION seconds by fps) -> convert to seconds.
            gts = [(float(s[0]) / GT_FPS, float(s[1]) / GT_FPS)
                   for s in (gt.get(vid, {}).get('segments') or [])]
            for a in rec['anomalies']:
                cat = str(a.get('anomaly_type') or '').strip()
                if cat not in CBT_ALL_EXTRA_CAPS:
                    continue
                cap = CBT_ALL_EXTRA_CAPS[cat]
                t_s, t_e = float(a['t_start']), float(a['t_end'])
                if (t_e - t_s) <= TRIG * cap:
                    continue
                if name == 'O1_loose':
                    stats['addressable'] += 1
                    if gts:
                        stats['on_anomaly_video'] += 1
                    if any(vid.startswith(h) for h in HARD88):
                        stats['hard88_video'] += 1
                if not gts:
                    continue      # Normal video: an inward clamp of an FP cannot
                    #                change 11-point AP (confidence is frozen)
                min_seg = CATEGORY_DURATION_PRIORS.get(cat, {}).get('min_segment', 1.5)
                w, _v = best_inward((t_s, t_e), gts, min_seg,
                                    cap if variant == 'cap' else None)
                if w is not None:
                    a['t_start'], a['t_end'] = w[0], w[1]
                    n_fire += 1
        m = evaluate.compute_temporal_map(gt, p2, iou_thresholds=IOU_THR)['mAP']
        stats['improved_o1' if name == 'O1_loose' else 'improved_o2'] = n_fire
        print(f'{name:16s} oracle mAP = {m:.5f}   delta = {m - base:+.5f}   '
              f'oracle_fires = {n_fire}')

    print('POPULATION: addressable dets (dur > 1.1*prior_cap, 9 extended cats) '
          f'= {stats["addressable"]}')
    print(f'            of which on ANOMALY videos (mAP-relevant) = '
          f'{stats["on_anomaly_video"]}')
    print(f'            of which sit on a hard-88 video = {stats["hard88_video"]} '
          '(=> hard-88 is NOT an exact zero, but ~5x under-powered)')
    print(f'CBT own committed caps = {CBT_CAPS} / extended = {CBT_ALL_EXTRA_CAPS}')


if __name__ == '__main__':
    main()
