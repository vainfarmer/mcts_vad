#!/usr/bin/env python3.14
"""
v200c PHASE-ORACLE part 3: ORACLE -> REAL transfer calibration + the two
confound controls the part-2 numbers need.

Part 2 gave each roster mechanism its scoped oracle ceiling. A ceiling only
becomes a decision if we know what fraction of it a real (GT-blind) operator
converts. We have exactly one measured data point for that: CBT+cbt_recal,
whose paired marginal at 290 is +0.02283 (config.py:125/131). CBT's admissible
move set is IDENTICAL in form to v201/v203 (inward contraction, per-category
cap, over-wide trigger), so its oracle/real ratio is the right prior.

Also computes:
  * the pre_bue and pre_cbt FLOORS, so the v204 oracle numbers (which start
    from pre_bue and therefore drop CBT from the chain) are interpretable;
  * how many detections CBT currently fires on, to size v203's 10-det
    population against CBT's own.

Zero evals, zero API calls. Read-only. Never modifies evaluate.py.
"""
import sys
import copy
import json
from pathlib import Path

REPO = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(REPO / 'video_anomaly_detection'))

import evaluate  # noqa: E402
from pipeline import CATEGORY_DURATION_PRIORS  # noqa: E402

GT_PATH = '/Users/zzl/Movies/ucf-crime/data'
RUN = REPO / 'results' / 'batch_eval_v26_full290'
FPS, IOUS = 30, [0.3, 0.5, 0.7]
HARD = {'Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion'}
CBT_RECAL_CAPS = {'RoadAccidents': 4.5, 'Explosion': 13.0,
                  'Shoplifting': 9.0, 'Shooting': 13.5}
CBT_TRIG = 1.1
CBT_REAL_MARGINAL_290 = 0.02283   # config.py:125/131, paired at 290


def mp(gt, pred):
    return evaluate.compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=IOUS)


def ov(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def iou_sec(s, e, gs, ge):
    u = max(e, ge) - min(s, gs)
    return ov(s, e, gs, ge) / u if u > 0 else 0.0


def best_inward(s, e, gs, ge):
    i0 = ov(s, e, gs, ge)
    cur = iou_sec(s, e, gs, ge)
    if i0 <= 0:
        return (s, e), cur
    gl = ge - gs
    cand = i0 / gl if gl > 0 else 0.0
    if cand > cur:
        return (max(s, gs), min(e, ge)), cand
    return (s, e), cur


def main():
    gt = evaluate.load_ground_truth(GT_PATH, split='test')
    pred = evaluate.load_predictions(RUN)
    assert json.load(open(RUN / 'evaluation_report.json'))['timing']['failed'] == 0

    gt88 = {k: v for k, v in gt.items() if v['category'] in HARD}
    base = mp(gt, pred)['mAP']
    base88 = mp(gt88, {k: v for k, v in pred.items() if k in gt88})['mAP']
    print(f"[BASE] 290={base:.5f}  88={base88:.5f}")
    out = {'base290': base, 'base88': base88}

    snaps = {}
    for f in RUN.glob('*.json'):
        if f.name == 'evaluation_report.json':
            continue
        d = json.load(open(f))
        vid = Path(d.get('video_path', f.stem)).stem
        snaps[vid] = (d.get('pre_bue_boundaries'), d.get('pre_cbt_boundaries'))

    # ---------- FLOORS: replay the chain truncated at each snapshot ----------
    for name, slot in (('pre_bue', 0), ('pre_cbt', 1)):
        p = copy.deepcopy(pred)
        n = 0
        for vid, pe in p.items():
            sn = snaps.get(vid, (None, None))[slot]
            dets = pe.get('anomalies', [])
            if not sn or len(sn) != len(dets):
                continue
            for (s, e), d in zip(sn, dets):
                if (float(d['t_start']), float(d['t_end'])) != (float(s), float(e)):
                    n += 1
                d['t_start'], d['t_end'] = float(s), float(e)
        m = mp(gt, p)['mAP']
        m88 = mp(gt88, {k: v for k, v in p.items() if k in gt88})['mAP']
        print(f"[FLOOR {name}] windows rolled back on {n} dets -> "
              f"290 mAP={m:.5f} (d={m - base:+.5f})   88 mAP={m88:.5f} (d={m88 - base88:+.5f})")
        out[f'floor_{name}'] = {'mAP290': m, 'd290': m - base,
                                'mAP88': m88, 'd88': m88 - base88, 'rolled': n}

    # ---------- how many dets does CBT actually fire on today? ----------
    cbt_fired = 0
    cbt_gate_pop = 0
    for vid, gentry in gt.items():
        sn = snaps.get(vid, (None, None))[1]
        dets = pred.get(vid, {}).get('anomalies', [])
        if not sn or len(sn) != len(dets):
            continue
        for (s, e), d in zip(sn, dets):
            if (float(d['t_start']), float(d['t_end'])) != (float(s), float(e)):
                cbt_fired += 1
            cap = CBT_RECAL_CAPS.get(gentry['category'])
            if cap and (float(e) - float(s)) > CBT_TRIG * cap:
                cbt_gate_pop += 1
    print(f"[CBT-TODAY] dets whose window changed at the CBT slot: {cbt_fired}; "
          f"dets passing the recal over-wide gate: {cbt_gate_pop}")
    out['cbt_today'] = {'fired_dets': cbt_fired, 'gate_pop': cbt_gate_pop}

    # ---------- CBT ORACLE on its own population (hard-4, recal caps) -------
    p = copy.deepcopy(pred)
    fired = 0
    for vid, gentry in gt.items():
        segs = gentry['segments']
        cap = CBT_RECAL_CAPS.get(gentry['category'])
        sn = snaps.get(vid, (None, None))[1]
        if not segs or not cap or not sn:
            continue
        dets = p[vid]['anomalies']
        if len(sn) != len(dets):
            continue
        idxs = sorted(range(len(dets)), key=lambda i: -float(dets[i].get('confidence', 0.0)))
        free = set(range(len(segs)))
        for i in idxs:
            if not free:
                break
            s, e = float(sn[i][0]), float(sn[i][1])   # pre-CBT window
            if (e - s) <= CBT_TRIG * cap:
                continue                              # gate: not over-wide
            best = None
            for j in sorted(free):
                gs, ge = segs[j][0] / FPS, segs[j][1] / FPS
                (ns, ne), q = best_inward(s, e, gs, ge)
                if best is None or q > best[0]:
                    best = (q, j, ns, ne)
            q, j, ns, ne = best
            free.discard(j)
            if (ns, ne) != (float(dets[i]['t_start']), float(dets[i]['t_end'])):
                dets[i]['t_start'], dets[i]['t_end'] = ns, ne
                fired += 1
    m = mp(gt, p)['mAP']
    m88 = mp(gt88, {k: v for k, v in p.items() if k in gt88})['mAP']
    # oracle marginal is measured against the pre_cbt floor, matching how the
    # +0.02283 real marginal was measured (CBT-ON vs CBT-OFF, same run)
    fl = out['floor_pre_cbt']
    orc_marg_290 = m - fl['mAP290']
    print(f"[CBT-ORACLE] fired {fired} dets -> 290 mAP={m:.5f}  88 mAP={m88:.5f}")
    print(f"[CBT-ORACLE] oracle marginal vs pre_cbt floor @290 = {orc_marg_290:+.5f}")
    print(f"[CBT-REAL]   measured paired marginal @290          = {CBT_REAL_MARGINAL_290:+.5f}")
    ratio = CBT_REAL_MARGINAL_290 / orc_marg_290 if orc_marg_290 > 0 else float('nan')
    print(f"[TRANSFER]   real / oracle = {ratio:.3f}   <-- the FM6 transfer factor "
          f"for inward-contraction operators")
    out['cbt_oracle'] = {'mAP290': m, 'mAP88': m88, 'fired': fired,
                         'oracle_marginal_290': orc_marg_290,
                         'real_marginal_290': CBT_REAL_MARGINAL_290,
                         'transfer_ratio': ratio}

    # ---------- apply the transfer factor to the roster ceilings ----------
    ceilings = {
        'v201 eubc  (88, inward+min_seg)': 0.04913,
        'v201 eubc  (88, inward loose)':   0.07778,
        'v202 ece   (88, outward@prior)':  0.09829,
        'v203 cbtall(290, trig1.1)':       0.01694,
        'v204 bue   (88, percat lmax)':   -0.01654,
        'v204 bue   (88, free lmax)':      0.06864,
        'v205 snh   (290, perfect veto)':  0.00747,
    }
    print("\n[EXPECTED REALIZATION at the measured transfer factor]")
    for k, v in ceilings.items():
        exp = v * ratio
        print(f"  {k:34s} ceiling {v:+.5f} -> expected {exp:+.5f} "
              f"{'PASS' if exp >= 0.008 else 'FAIL'} vs +0.008 gate")
        out.setdefault('expected', {})[k] = {'ceiling': v, 'expected': exp,
                                             'pass': exp >= 0.008}
    print("\n" + json.dumps(out, indent=1, default=str))
    return out


if __name__ == '__main__':
    main()
