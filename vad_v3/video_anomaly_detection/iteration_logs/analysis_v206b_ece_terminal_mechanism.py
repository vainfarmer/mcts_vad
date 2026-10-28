#!/usr/bin/env python3.14
"""v206b: mechanism scoping for the v206 ECE-at-terminal offline precheck.

Reuses analysis_v206_ece_terminal_precheck helpers verbatim (no re-implementation
of the ECE rule). Adds, all MEASURED not asserted:
  * hard-88-SCOPED cbt_undo (the decision harness), alongside the full-290 number
  * per-detection anatomy of the biggest AP losers (which side grew, by how much,
    and whether that side had been clamped inward by CBT)
  * a hard-88 leave-one-video-out jackknife on the offline delta, so the sign of
    the result is not a single-video artefact
  * IoU-threshold decomposition of the hard-88 delta
ZERO API calls, ZERO evals, cache-only curves.
"""
import sys
import json
from pathlib import Path

import numpy as np

VAD = Path('/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
sys.path.insert(0, str(VAD))
sys.path.insert(0, str(VAD / 'iteration_logs'))

import evaluate                                             # noqa: E402
import analysis_v206_ece_terminal_precheck as V206          # noqa: E402

HARD = V206.HARD


def main():
    V206.set_device('mps')
    gt = evaluate.load_ground_truth(V206.GT_PATH, split='test')
    recs = V206.load_run(V206.RUN)
    override, agg, traces, curve_miss = V206.replay_final(recs, allow_build=False)
    print(f'[ECE-TERMINAL] {agg} curve_miss={curve_miss}')

    base = V206.to_pred(recs)
    ece = V206.to_pred(recs, override)

    def scope(pred, keys):
        return {k: v for k, v in pred.items() if k in keys}

    hard_gt = {k: v for k, v in gt.items() if k.startswith(HARD)}
    hk = set(hard_gt)

    # ---- IoU decomposition on hard-88
    print('\n[IoU decomposition hard-88]')
    for thr in (0.3, 0.5, 0.7):
        b = evaluate.compute_temporal_map(hard_gt, scope(base, hk), fps=V206.FPS,
                                          iou_thresholds=[thr])['mAP']
        a = evaluate.compute_temporal_map(hard_gt, scope(ece, hk), fps=V206.FPS,
                                          iou_thresholds=[thr])['mAP']
        print(f'   IoU {thr}: base={b:.5f} ece={a:.5f} delta={a - b:+.5f}')

    m88_b = V206.mp(hard_gt, scope(base, hk))
    m88_a = V206.mp(hard_gt, scope(ece, hk))
    print(f'\n[hard-88] base={m88_b:.5f} ece={m88_a:.5f} delta={m88_a - m88_b:+.5f}')

    # ---- cbt_undo, scoped
    def cbt_undo(keys):
        tot_clamp = tot_undo = 0.0
        nd = 0
        rows = []
        for vid, d in sorted(recs.items()):
            if vid not in keys:
                continue
            pre_cbt = d.get('pre_cbt_boundaries')
            an = d.get('anomalies') or []
            if not pre_cbt or len(pre_cbt) != len(an):
                continue
            ov = override.get(vid)
            for i, a in enumerate(an):
                ps, pe = float(pre_cbt[i][0]), float(pre_cbt[i][1])
                fs, fe = float(a['t_start']), float(a['t_end'])
                clamp = (ps - fs) + (pe - fe)
                if clamp <= 1e-9:
                    continue
                nd += 1
                tot_clamp += clamp
                es, ee = (ov[i] if ov else (fs, fe))
                undo = max(0.0, fs - es) + max(0.0, ee - fe)
                tot_undo += min(undo, clamp)
                rows.append((vid, i, ps, pe, fs, fe, es, ee, clamp, min(undo, clamp)))
        frac = tot_undo / tot_clamp if tot_clamp > 0 else 0.0
        return nd, tot_clamp, tot_undo, frac, rows

    for label, keys in (('full-290', set(recs)), ('hard-88', hk)):
        nd, tc, tu, fr, rows = cbt_undo(keys)
        print(f'[cbt_undo {label}] clamped_dets={nd} clamp={tc:.1f}s undone={tu:.1f}s '
              f'frac={fr:.3f}')
        if label == 'hard-88':
            full_undo = sum(1 for r in rows if r[9] > 0.5 * r[8])
            any_undo = sum(1 for r in rows if r[9] > 1e-6)
            print(f'   of {nd} clamped dets: {any_undo} partially re-grown, '
                  f'{full_undo} re-grown past half the clamp')

    # ---- anatomy of losers / winners
    print('\n[anatomy] per-detection for videos whose AP moved most')
    per = []
    for vid in sorted(hk & set(base)):
        one = {vid: hard_gt[vid]}
        b = V206.mp(one, {vid: base[vid]})
        a = V206.mp(one, {vid: ece[vid]})
        if abs(a - b) > 1e-9:
            per.append((a - b, vid, b, a))
    per.sort()
    for dv, vid, b, a in per[:4] + per[-3:]:
        d = recs[vid]
        an = d['anomalies']
        pre_cbt = d.get('pre_cbt_boundaries')
        segs = hard_gt[vid].get('segments')
        print(f'  {dv:+.4f} {vid} AP {b:.3f}->{a:.3f}  GT={segs}')
        for i, aa in enumerate(an):
            fs, fe = aa['t_start'], aa['t_end']
            es, ee = override[vid][i]
            pc = pre_cbt[i] if pre_cbt and i < len(pre_cbt) else None
            tag = 'MATERIAL' if (abs(es - fs) > 1e-6 or abs(ee - fe) > 1e-6) else 'same'
            print(f'     #{i} cat={aa.get("anomaly_type")} pre_cbt={pc} '
                  f'final=[{fs:.1f},{fe:.1f}] -> ece=[{es:.1f},{ee:.1f}] {tag}')

    # ---- jackknife: is the negative sign a single-video artefact?
    vids = sorted(hk & set(base))
    d_full = m88_a - m88_b
    jk = []
    for v in vids:
        g = {k: x for k, x in hard_gt.items() if k != v}
        b = V206.mp(g, {k: x for k, x in base.items() if k in g})
        a = V206.mp(g, {k: x for k, x in ece.items() if k in g})
        jk.append((a - b, v))
    arr = np.array([x[0] for x in jk])
    print(f'\n[jackknife hard-88] full delta={d_full:+.5f}  LOO n={len(arr)} '
          f'min={arr.min():+.5f} max={arr.max():+.5f} median={np.median(arr):+.5f}')
    print(f'   LOO deltas >= +0.0048 (GO gate): {int((arr >= 0.0048).sum())}/{len(arr)}')
    print(f'   LOO deltas > 0: {int((arr > 0).sum())}/{len(arr)}')
    worst = sorted(jk)[-3:]
    print('   most-negative-contributing videos (dropping them helps most): '
          + ', '.join(f'{v}({d:+.4f})' for d, v in worst))

    json.dump({'m88_base': m88_b, 'm88_ece': m88_a, 'd88': d_full,
               'loo_min': float(arr.min()), 'loo_max': float(arr.max()),
               'loo_median': float(np.median(arr)),
               'loo_pass_gate': int((arr >= 0.0048).sum()),
               'loo_positive': int((arr > 0).sum()), 'loo_n': len(arr)},
              open(VAD / 'iteration_logs' / 'analysis_v206b_mechanism.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
