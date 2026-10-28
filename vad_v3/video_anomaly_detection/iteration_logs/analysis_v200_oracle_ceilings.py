#!/usr/bin/env python3.14
"""
v200 PHASE-ORACLE (FM13): bound the remaining headroom PER LAYER on the
committed full-290 detections, so the v201-v205 roster targets MEASURED
ceilings instead of hope.

ZERO new evals, ZERO API calls. Read-only over results/batch_eval_v26_full290
(290/290, timing.failed == 0 -- the fresh committed baseline substrate).
Never modifies evaluate.py; only calls its public loaders + scorer.

Three oracles, each holding the OTHER TWO layers FIXED:

  (a) BOUNDARY ceiling  -- detection set + confidence order frozen; each
      window replaced by the best achievable window (exact snap to a GT
      segment => IoU 1.0). Within a video the top-min(ndet,nGT) detections
      BY CONFIDENCE are snapped to distinct GT segments, so the maximum
      possible number of TPs land at the best possible ranks. TIGHT.
      Ceiling for EUBC / ECE / BUE-recal / CBT-ALL (all boundary operators).

  (b) CANDIDATE-SET ceiling -- boundaries + confidences of surviving
      detections frozen; optimally ADD windows for uncovered GT and/or DROP
      non-matching detections. Reported in three variants because the
      combined form is provably degenerate (see note in output).
      Ceiling for candidate-generation levers.

  (c) RANKING ceiling -- every window frozen; confidence reordered globally
      to the optimal ranking. Computed analytically from the per-video
      maximum bipartite matching at each IoU threshold (strict upper bound,
      per-threshold optimal), plus an achievable single-order realization.
      Ceiling for confidence operators (SNH lives here).

Plus: the CBT-ALL addressable population -- how many of the NON-hard-category
anomaly videos carry a detection whose duration exceeds its
CATEGORY_DURATION_PRIORS-implied cap. If ~0, loop v203 is pointless.
"""
import sys
import copy
import json
import statistics
from pathlib import Path

REPO = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(REPO / 'video_anomaly_detection'))

import evaluate  # noqa: E402
from pipeline import CATEGORY_DURATION_PRIORS, CBT_CAPS  # noqa: E402

GT_PATH = '/Users/zzl/Movies/ucf-crime/data'
RUN = str(REPO / 'results' / 'batch_eval_v26_full290')
FPS = 30
IOUS = [0.3, 0.5, 0.7]
HARD = {'Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion'}

# cbt_recal defaults, config.py:132-135
CBT_RECAL_TRIGGER = 1.1


def mp(gt, pred):
    return evaluate.compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=IOUS)


def iou(seg, gtseg):
    return evaluate._temporal_iou(seg, gtseg, FPS)


def max_matching(dets, gtsegs, thresh):
    """Maximum bipartite matching size between dets and gtsegs at IoU>=thresh."""
    adj = [[j for j, g in enumerate(gtsegs) if iou(d, g) >= thresh]
           for d in dets]
    match_g = {}

    def aug(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_g or aug(match_g[j], seen):
                match_g[j] = i
                return True
        return False

    k = 0
    for i in range(len(dets)):
        if aug(i, set()):
            k += 1
    return k, match_g


def ap11_all_tp_first(k, n_gt):
    """11-point AP when k TPs occupy the top ranks and every later pred is FP.
    precision==1.0 up to recall k/n_gt; beyond that the recall mask is empty."""
    if n_gt == 0:
        return 0.0
    hits = 0
    for t10 in range(11):
        t = t10 / 10.0
        if t <= k / n_gt + 1e-12:
            hits += 1
    return hits / 11.0


def main():
    out = {}

    # ---------- coverage gate (PROJECT_DOCUMENTATION 8.4) ----------
    rep = json.load(open(Path(RUN) / 'evaluation_report.json'))
    timing = rep.get('timing', {})
    assert timing.get('failed') == 0, f"COVERAGE GATE FAIL: failed={timing.get('failed')}"

    gt = evaluate.load_ground_truth(GT_PATH, split='test')
    pred = evaluate.load_predictions(RUN)
    missing = [k for k, v in gt.items() if v.get('segments') and k not in pred]
    absent = [k for k in gt if k not in pred]
    print(f"[GATE] report num_videos={timing.get('num_videos')} failed={timing.get('failed')}")
    print(f"[GATE] gt videos={len(gt)}  pred videos={len(pred)}  "
          f"missing_anomaly={len(missing)}  absent_any={len(absent)}")
    assert len(missing) == 0, f"missing anomaly videos: {missing[:5]}"
    # load_predictions globs *.json, so evaluation_report.json shows up as a
    # phantom key; it is not in gt and carries no 'anomalies', so it is inert.
    n_real = sum(1 for k in pred if k in gt)
    print(f"[GATE] pred keys that are real gt videos={n_real}")
    assert n_real == 290, f"real pred count {n_real} != 290"

    n_gt = sum(len(v['segments']) for v in gt.values())
    n_anom_vid = sum(1 for v in gt.values() if v['segments'])
    n_det = sum(len(p['anomalies']) for p in pred.values())
    print(f"[SUBSTRATE] n_gt_segments={n_gt}  anomaly_videos={n_anom_vid}  "
          f"total_detections={n_det}")
    out['n_gt_segments'] = n_gt
    out['n_anomaly_videos'] = n_anom_vid
    out['n_detections'] = n_det

    base = mp(gt, pred)
    print(f"[BASE] committed full-290 mAP={base['mAP']:.5f}  "
          + '  '.join(f"{k}={base[k]:.5f}" for k in sorted(base) if k != 'mAP'))
    out['baseline'] = base['mAP']

    # =========================================================
    # (a) BOUNDARY ceiling -- TIGHT
    # =========================================================
    pred_a = copy.deepcopy(pred)
    snapped = 0
    vids_touched = 0
    for vid, gentry in gt.items():
        segs = gentry['segments']
        if not segs or vid not in pred_a:
            continue
        dets = pred_a[vid]['anomalies']
        if not dets:
            continue
        order = sorted(range(len(dets)),
                       key=lambda i: -float(dets[i].get('confidence', 0.0)))
        fired = False
        for rank, i in enumerate(order):
            if rank >= len(segs):
                break
            s, e = segs[rank]
            dets[i]['t_start'] = s / FPS
            dets[i]['t_end'] = e / FPS
            snapped += 1
            fired = True
        if fired:
            vids_touched += 1
    ora_a = mp(gt, pred_a)
    print(f"[ORACLE-a BOUNDARY] mAP={ora_a['mAP']:.5f} "
          f"dmAP={ora_a['mAP'] - base['mAP']:+.5f} "
          f"(snapped {snapped} dets across {vids_touched} videos)")
    out['boundary'] = {'mAP': ora_a['mAP'], 'dmAP': ora_a['mAP'] - base['mAP'],
                       'snapped': snapped, 'videos': vids_touched,
                       **{k: ora_a[k] for k in sorted(ora_a) if k != 'mAP'}}
    # arm-independence self-check: a video with NO gt segs must be byte-identical
    ident = all(pred_a[v]['anomalies'] == pred[v]['anomalies']
                for v in gt if not gt[v]['segments'] and v in pred)
    print(f"[SELFTEST-a] normal-video detections byte-identical: {ident}")
    out['boundary']['selftest_normal_identical'] = bool(ident)

    # =========================================================
    # (c) RANKING ceiling -- windows frozen, global reorder
    # =========================================================
    per_thresh = {}
    for th in IOUS:
        k_tot = 0
        for vid, gentry in gt.items():
            segs = gentry['segments']
            if not segs or vid not in pred:
                continue
            dets = pred[vid]['anomalies']
            if not dets:
                continue
            k, _ = max_matching(dets, segs, th)
            k_tot += k
        per_thresh[th] = {'k_max': k_tot, 'ap_ceiling': ap11_all_tp_first(k_tot, n_gt)}
        print(f"[ORACLE-c] IoU {th}: max-matchable TPs={k_tot}/{n_gt} "
              f"-> AP ceiling {per_thresh[th]['ap_ceiling']:.5f} "
              f"(committed AP@{th}={base[f'AP@{th}']:.5f})")
    rank_ceiling = statistics.fmean(v['ap_ceiling'] for v in per_thresh.values())
    print(f"[ORACLE-c RANKING] mAP ceiling={rank_ceiling:.5f} "
          f"dmAP={rank_ceiling - base['mAP']:+.5f}  (per-threshold optimal order = "
          f"STRICT upper bound over all single orderings)")

    # achievable single-order realization: sort by best-IoU-to-any-GT desc
    pred_c = copy.deepcopy(pred)
    scored = []
    for vid, p in pred_c.items():
        segs = gt.get(vid, {}).get('segments', [])
        for d in p['anomalies']:
            best = max((iou(d, g) for g in segs), default=0.0)
            scored.append((best, d))
    scored.sort(key=lambda x: -x[0])
    n = len(scored)
    for r, (_, d) in enumerate(scored):
        d['confidence'] = 1.0 - r / (n + 1.0)
    ora_c = mp(gt, pred_c)
    print(f"[ORACLE-c realized] single-order (best-IoU desc) mAP={ora_c['mAP']:.5f} "
          f"dmAP={ora_c['mAP'] - base['mAP']:+.5f}")
    out['ranking'] = {
        'mAP_upper': rank_ceiling, 'dmAP_upper': rank_ceiling - base['mAP'],
        'mAP_realized': ora_c['mAP'], 'dmAP_realized': ora_c['mAP'] - base['mAP'],
        'per_threshold': {str(k): v for k, v in per_thresh.items()},
    }

    # =========================================================
    # (b) CANDIDATE-SET ceiling
    # =========================================================
    confs = [float(d.get('confidence', 0.0)) for p in pred.values()
             for d in p['anomalies']]
    med_conf = statistics.median(confs)
    print(f"[ORACLE-b] confidence distribution: n={len(confs)} "
          f"min={min(confs):.4f} med={med_conf:.4f} max={max(confs):.4f}")

    # b1 ADD-ONLY: every existing det untouched; add one perfect window per GT
    # segment not already matched at IoU>=0.7, at MEDIAN confidence (so the
    # addition does not trivially dominate the global ranking).
    pred_b1 = copy.deepcopy(pred)
    added = 0
    for vid, gentry in gt.items():
        segs = gentry['segments']
        if not segs:
            continue
        dets = pred_b1.setdefault(vid, {'anomalies': [], 'duration':
                                        pred.get(vid, {}).get('duration', 0)})['anomalies']
        _, mg = max_matching(dets, segs, 0.7)
        for j, (s, e) in enumerate(segs):
            if j in mg:
                continue
            dets.append({'t_start': s / FPS, 't_end': e / FPS,
                         'confidence': med_conf,
                         'anomaly_type': gentry['category']})
            added += 1
    ora_b1 = mp(gt, pred_b1)
    print(f"[ORACLE-b1 ADD-ONLY] added {added} perfect windows @med conf -> "
          f"mAP={ora_b1['mAP']:.5f} dmAP={ora_b1['mAP'] - base['mAP']:+.5f}")

    # b1h: same but at MAX confidence (added dets outrank everything)
    pred_b1h = copy.deepcopy(pred)
    for vid, gentry in gt.items():
        segs = gentry['segments']
        if not segs:
            continue
        dets = pred_b1h[vid]['anomalies']
        _, mg = max_matching(dets, segs, 0.7)
        for j, (s, e) in enumerate(segs):
            if j in mg:
                continue
            dets.append({'t_start': s / FPS, 't_end': e / FPS,
                         'confidence': 1.0, 'anomaly_type': gentry['category']})
    ora_b1h = mp(gt, pred_b1h)
    print(f"[ORACLE-b1h ADD-ONLY @conf 1.0] mAP={ora_b1h['mAP']:.5f} "
          f"dmAP={ora_b1h['mAP'] - base['mAP']:+.5f}")

    # b2 REMOVE-ONLY: drop every detection that cannot participate in the max
    # matching at the LOOSEST threshold (IoU>=0.3). Single candidate set that
    # serves all three thresholds. Boundaries + confidences untouched.
    pred_b2 = copy.deepcopy(pred)
    dropped = 0
    for vid, p in pred_b2.items():
        segs = gt.get(vid, {}).get('segments', [])
        dets = p['anomalies']
        if not dets:
            continue
        if not segs:
            dropped += len(dets)
            p['anomalies'] = []
            continue
        _, mg = max_matching(dets, segs, 0.3)
        keep = set(mg.values())
        dropped += len(dets) - len(keep)
        p['anomalies'] = [d for i, d in enumerate(dets) if i in keep]
    ora_b2 = mp(gt, pred_b2)
    print(f"[ORACLE-b2 REMOVE-ONLY] dropped {dropped}/{n_det} non-matching -> "
          f"mAP={ora_b2['mAP']:.5f} dmAP={ora_b2['mAP'] - base['mAP']:+.5f}")

    # b2n REMOVE-ONLY restricted to NORMAL videos (the SNH-as-deletion lever)
    pred_b2n = copy.deepcopy(pred)
    dn = 0
    for vid, p in pred_b2n.items():
        if gt.get(vid, {}).get('segments'):
            continue
        dn += len(p['anomalies'])
        p['anomalies'] = []
    ora_b2n = mp(gt, pred_b2n)
    print(f"[ORACLE-b2n REMOVE Normal-only] dropped {dn} dets on Normal videos -> "
          f"mAP={ora_b2n['mAP']:.5f} dmAP={ora_b2n['mAP'] - base['mAP']:+.5f}")

    # b3 COMBINED (spec form): add-all + remove-all
    pred_b3 = copy.deepcopy(pred_b2)
    for vid, gentry in gt.items():
        segs = gentry['segments']
        if not segs:
            continue
        dets = pred_b3[vid]['anomalies']
        _, mg = max_matching(dets, segs, 0.7)
        for j, (s, e) in enumerate(segs):
            if j in mg:
                continue
            dets.append({'t_start': s / FPS, 't_end': e / FPS,
                         'confidence': med_conf, 'anomaly_type': gentry['category']})
    ora_b3 = mp(gt, pred_b3)
    print(f"[ORACLE-b3 COMBINED add+remove] mAP={ora_b3['mAP']:.5f} "
          f"dmAP={ora_b3['mAP'] - base['mAP']:+.5f}  (DEGENERATE by construction)")

    out['candidate'] = {
        'combined_mAP': ora_b3['mAP'], 'combined_dmAP': ora_b3['mAP'] - base['mAP'],
        'add_only_med_mAP': ora_b1['mAP'], 'add_only_med_dmAP': ora_b1['mAP'] - base['mAP'],
        'add_only_max_mAP': ora_b1h['mAP'], 'add_only_max_dmAP': ora_b1h['mAP'] - base['mAP'],
        'remove_only_mAP': ora_b2['mAP'], 'remove_only_dmAP': ora_b2['mAP'] - base['mAP'],
        'remove_normal_mAP': ora_b2n['mAP'], 'remove_normal_dmAP': ora_b2n['mAP'] - base['mAP'],
        'added': added, 'dropped': dropped, 'dropped_normal': dn,
    }

    # =========================================================
    # v203 CBT-ALL addressable population (NON-hard anomaly cats)
    # =========================================================
    nonhard_vids = [v for v, g in gt.items()
                    if g['segments'] and g['category'] not in HARD]
    print(f"\n[v203] non-hard-category anomaly videos: {len(nonhard_vids)}")
    for trig in (1.0, 1.1, 1.5):
        hit_v, hit_d = 0, 0
        percat = {}
        for v in nonhard_vids:
            cap = CATEGORY_DURATION_PRIORS.get(
                gt[v]['category'], {}).get('expected_duration', (0, 1e9))[1]
            dets = pred.get(v, {}).get('anomalies', [])
            k = sum(1 for d in dets
                    if (float(d['t_end']) - float(d['t_start'])) > trig * cap)
            if k:
                hit_v += 1
                hit_d += k
                percat[gt[v]['category']] = percat.get(gt[v]['category'], 0) + 1
        print(f"[v203] trigger x{trig}: {hit_v}/{len(nonhard_vids)} videos, "
              f"{hit_d} over-wide dets  percat={dict(sorted(percat.items()))}")
        out.setdefault('cbt_all', {})[f'trigger_{trig}'] = {
            'videos': hit_v, 'dets': hit_d, 'percat': percat,
            'n_nonhard': len(nonhard_vids)}

    # how much of the boundary ceiling lives on the non-hard subset?
    pred_nh = copy.deepcopy(pred)
    for vid in nonhard_vids:
        segs = gt[vid]['segments']
        dets = pred_nh[vid]['anomalies']
        order = sorted(range(len(dets)),
                       key=lambda i: -float(dets[i].get('confidence', 0.0)))
        for rank, i in enumerate(order):
            if rank >= len(segs):
                break
            s, e = segs[rank]
            dets[i]['t_start'] = s / FPS
            dets[i]['t_end'] = e / FPS
    ora_nh = mp(gt, pred_nh)
    print(f"[v203] boundary oracle restricted to the {len(nonhard_vids)} non-hard "
          f"anomaly videos: mAP={ora_nh['mAP']:.5f} "
          f"dmAP={ora_nh['mAP'] - base['mAP']:+.5f}")
    out['cbt_all']['nonhard_boundary_oracle_mAP'] = ora_nh['mAP']
    out['cbt_all']['nonhard_boundary_oracle_dmAP'] = ora_nh['mAP'] - base['mAP']

    # and on the hard-4 subset (v201/v202/v204 territory), scored at 290
    hard_vids = [v for v, g in gt.items() if g['segments'] and g['category'] in HARD]
    pred_h = copy.deepcopy(pred)
    for vid in hard_vids:
        segs = gt[vid]['segments']
        dets = pred_h[vid]['anomalies']
        order = sorted(range(len(dets)),
                       key=lambda i: -float(dets[i].get('confidence', 0.0)))
        for rank, i in enumerate(order):
            if rank >= len(segs):
                break
            s, e = segs[rank]
            dets[i]['t_start'] = s / FPS
            dets[i]['t_end'] = e / FPS
    ora_h = mp(gt, pred_h)
    print(f"[hard-4] boundary oracle restricted to the {len(hard_vids)} hard "
          f"anomaly videos: mAP={ora_h['mAP']:.5f} "
          f"dmAP={ora_h['mAP'] - base['mAP']:+.5f}")
    out['hard_boundary_oracle_mAP'] = ora_h['mAP']
    out['hard_boundary_oracle_dmAP'] = ora_h['mAP'] - base['mAP']
    out['n_hard_anomaly_videos'] = len(hard_vids)

    # ---------- hard-88 harness view (where v201/v202/v204 are judged) ----------
    gt88 = {k: v for k, v in gt.items() if v['category'] in HARD}
    pred88 = {k: v for k, v in pred.items() if k in gt88}
    b88 = mp(gt88, pred88)
    p88a = {k: v for k, v in copy.deepcopy(pred_a).items() if k in gt88}
    o88a = mp(gt88, p88a)
    print(f"\n[hard-88] n={len(gt88)} committed mAP={b88['mAP']:.5f}  "
          f"boundary-oracle mAP={o88a['mAP']:.5f} "
          f"dmAP={o88a['mAP'] - b88['mAP']:+.5f}")
    n_gt88 = sum(len(v['segments']) for v in gt88.values())
    pt88 = {}
    for th in IOUS:
        kt = 0
        for vid, gentry in gt88.items():
            segs = gentry['segments']
            dets = pred88.get(vid, {}).get('anomalies', [])
            if not segs or not dets:
                continue
            k, _ = max_matching(dets, segs, th)
            kt += k
        pt88[th] = ap11_all_tp_first(kt, n_gt88)
    r88 = statistics.fmean(pt88.values())
    print(f"[hard-88] ranking ceiling={r88:.5f} dmAP={r88 - b88['mAP']:+.5f}")
    out['hard88'] = {'n': len(gt88), 'baseline': b88['mAP'],
                     'boundary_mAP': o88a['mAP'],
                     'boundary_dmAP': o88a['mAP'] - b88['mAP'],
                     'ranking_mAP': r88, 'ranking_dmAP': r88 - b88['mAP']}

    print("\n" + json.dumps(out, indent=1, default=str))
    return out


if __name__ == '__main__':
    main()
