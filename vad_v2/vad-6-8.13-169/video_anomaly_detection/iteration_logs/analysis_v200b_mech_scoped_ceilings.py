#!/usr/bin/env python3.14
"""
v200b PHASE-ORACLE part 2 (FM13): MECHANISM-SCOPED ceilings.

Part 1 (analysis_v200_oracle_ceilings.py) showed the three LAYER oracles are
all far above the +0.008 gate (boundary +0.665, ranking +0.136, candidate
+0.676 at 290). Those bounds are therefore TOO LOOSE to discriminate the
v201-v205 roster: they permit moves no real operator can make.

This script computes, for each roster mechanism, the ceiling reachable under
that mechanism's OWN admissible move set -- direction, per-category cap, and
fire gate all enforced -- with the GT used only as the oracle target. These
are the numbers that decide go/no-go, because an operator provably cannot
exceed its own move set.

Admissible move sets (all analytic, derived below):
  INWARD (EUBC v201, CBT-ALL v203): new window must be a SUBSET of the
    current one. Best subset W of [s,e] w.r.t. GT [gs,ge] is [s,e] cap GT,
    giving IoU = I0/gt_len (any smaller W has |W cap GT| <= I0 and the same
    union gt_len, so I0/gt_len is the max). Monotone: contraction can never
    beat the original when the det is already inside GT.
  OUTWARD-CAPPED (ECE v202, BUE-recal v204): new window must be a SUPERSET
    of the current one with |W| <= L. Growth toward GT raises the
    intersection by 1 per second while the union end-points stay pinned at
    min(s,gs)/max(e,ge), so every allocation of the growth budget
    B = L - (e-s) across the two deficits yields the SAME IoU:
    (I0 + min(B, deficit)) / U0.  <- exact, not a search.
  CONFIDENCE-ONLY (SNH v205): boundaries frozen; a perfect video-level
    P(normal) veto can only demote detections on TRUE Normal videos.

Zero evals, zero API calls. Read-only over results/batch_eval_v26_full290.
Never modifies evaluate.py.
"""
import sys
import copy
import json
import statistics
from pathlib import Path

REPO = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(REPO / 'video_anomaly_detection'))

import evaluate  # noqa: E402
from pipeline import CATEGORY_DURATION_PRIORS  # noqa: E402

GT_PATH = '/Users/zzl/Movies/ucf-crime/data'
RUN = REPO / 'results' / 'batch_eval_v26_full290'
FPS = 30
IOUS = [0.3, 0.5, 0.7]
HARD = {'Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion'}
CBT_TRIG = 1.1          # config.py:135 cbt_recal_trigger
BUE_LMAX_GLOBAL = 40.0  # config.py:109 bue_lmax_sec


def mp(gt, pred):
    return evaluate.compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=IOUS)


def ov(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def iou_sec(s, e, gs, ge):
    inter = ov(s, e, gs, ge)
    union = max(e, ge) - min(s, ge if False else gs)
    union = max(e, ge) - min(s, gs)
    return inter / union if union > 0 else 0.0


def best_inward(s, e, gs, ge):
    """Best SUBSET of [s,e] w.r.t. GT, and its IoU. Falls back to no-op."""
    i0 = ov(s, e, gs, ge)
    cur = iou_sec(s, e, gs, ge)
    if i0 <= 0:
        return (s, e), cur
    ns, ne = max(s, gs), min(e, ge)
    gl = ge - gs
    cand = i0 / gl if gl > 0 else 0.0
    if cand > cur:
        return (ns, ne), cand
    return (s, e), cur


def best_outward(s, e, gs, ge, L, dur):
    """Best SUPERSET of [s,e] with length <= L. Analytic (see docstring)."""
    cur = iou_sec(s, e, gs, ge)
    length = e - s
    B = L - length
    if B <= 0:
        return (s, e), cur
    left_need = max(0.0, min(s - gs, s))            # cannot grow past t=0
    right_need = max(0.0, min(ge - e, dur - e))     # nor past video end
    need = left_need + right_need
    gain = min(B, need)
    u0 = max(e, ge) - min(s, gs)
    new_iou = (ov(s, e, gs, ge) + gain) / u0 if u0 > 0 else 0.0
    if new_iou <= cur:
        return (s, e), cur
    take_l = min(left_need, gain)
    take_r = min(right_need, gain - take_l)
    return (s - take_l, e + take_r), new_iou


def apply_scoped(gt, pred, mover, vid_filter=None, det_filter=None):
    """Per video: greedily assign dets (confidence desc) to unclaimed GT segs,
    taking the pair with the highest reachable IoU, and apply `mover`."""
    out = copy.deepcopy(pred)
    fired_dets = 0
    fired_vids = 0
    for vid, gentry in gt.items():
        segs = gentry['segments']
        if not segs or vid not in out:
            continue
        if vid_filter and not vid_filter(vid, gentry):
            continue
        dets = out[vid]['anomalies']
        if not dets:
            continue
        dur = out[vid].get('duration', 0.0) or 0.0
        idxs = sorted(range(len(dets)),
                      key=lambda i: -float(dets[i].get('confidence', 0.0)))
        free = set(range(len(segs)))
        touched = False
        for i in idxs:
            if not free:
                break
            d = dets[i]
            s, e = float(d['t_start']), float(d['t_end'])
            if det_filter and not det_filter(vid, gentry, d):
                continue
            best = None
            for j in sorted(free):
                gs, ge = segs[j][0] / FPS, segs[j][1] / FPS
                (ns, ne), q = mover(s, e, gs, ge, dur, gentry)
                if best is None or q > best[0]:
                    best = (q, j, ns, ne)
            if best is None:
                continue
            q, j, ns, ne = best
            free.discard(j)
            if (ns, ne) != (s, e):
                d['t_start'], d['t_end'] = ns, ne
                fired_dets += 1
                touched = True
        if touched:
            fired_vids += 1
    return out, fired_dets, fired_vids


def cap_of(cat):
    return CATEGORY_DURATION_PRIORS.get(cat, {}).get('expected_duration', (0.0, 1e9))[1]


def main():
    gt = evaluate.load_ground_truth(GT_PATH, split='test')
    pred = evaluate.load_predictions(RUN)
    assert json.load(open(RUN / 'evaluation_report.json'))['timing']['failed'] == 0
    assert sum(1 for k in pred if k in gt) == 290

    base = mp(gt, pred)['mAP']
    gt88 = {k: v for k, v in gt.items() if v['category'] in HARD}
    pred88 = {k: v for k, v in pred.items() if k in gt88}
    base88 = mp(gt88, pred88)['mAP']
    print(f"[BASE] full-290 mAP={base:.5f}   hard-88 mAP={base88:.5f}  (n88={len(gt88)})")
    res = {'base290': base, 'base88': base88}

    def report(tag, newpred, fd, fv, harness='290'):
        if harness == '88':
            g, p, b = gt88, {k: v for k, v in newpred.items() if k in gt88}, base88
        else:
            g, p, b = gt, newpred, base
        m = mp(g, p)['mAP']
        print(f"[{tag}] harness={harness} fired {fd} dets / {fv} vids -> "
              f"mAP={m:.5f} dmAP={m - b:+.5f}"
              + ("   *** BELOW +0.008 GATE ***" if m - b < 0.008 else ""))
        return {'mAP': m, 'dmAP': m - b, 'fired_dets': fd, 'fired_vids': fv,
                'harness': harness}

    # ---- SELF-TEST: a no-op mover must be byte-identical (arm independence) --
    noop = lambda s, e, gs, ge, dur, ge2: ((s, e), 0.0)
    p_noop, fd, fv = apply_scoped(gt, pred, noop)
    ident = all(p_noop[v]['anomalies'] == pred[v]['anomalies'] for v in pred)
    print(f"[SELFTEST] no-op mover: fired={fd} byte-identical={ident}")
    assert ident and fd == 0
    # and a fires-everywhere mover must differ
    p_all, fd_all, _ = apply_scoped(gt, pred, lambda s, e, gs, ge, d, g: ((gs, ge), 1.0))
    print(f"[SELFTEST] snap-all mover: fired={fd_all} differs="
          f"{any(p_all[v]['anomalies'] != pred[v]['anomalies'] for v in pred)}")

    # =====================================================================
    # v201 EUBC -- pure INWARD contraction, ungated by category
    # =====================================================================
    mv_in = lambda s, e, gs, ge, dur, g: best_inward(s, e, gs, ge)
    p201, fd, fv = apply_scoped(gt, pred, mv_in)
    res['v201_eubc_88'] = report('v201 EUBC inward-oracle', p201, fd, fv, '88')
    res['v201_eubc_290'] = report('v201 EUBC inward-oracle', p201, fd, fv, '290')
    # min_segment-respecting variant (what the real operator must honour)
    def mv_in_ms(s, e, gs, ge, dur, g):
        (ns, ne), q = best_inward(s, e, gs, ge)
        ms = CATEGORY_DURATION_PRIORS.get(g['category'], {}).get('min_segment', 1.5)
        if ne - ns < ms:
            return (s, e), iou_sec(s, e, gs, ge)
        return (ns, ne), q
    p201b, fd, fv = apply_scoped(gt, pred, mv_in_ms)
    res['v201_eubc_88_minseg'] = report('v201 EUBC inward+min_segment', p201b, fd, fv, '88')

    # =====================================================================
    # v202 ECE -- OUTWARD growth capped by CATEGORY_DURATION_PRIORS max_dur
    # =====================================================================
    mv_out_prior = lambda s, e, gs, ge, dur, g: best_outward(
        s, e, gs, ge, cap_of(g['category']), dur)
    p202, fd, fv = apply_scoped(gt, pred, mv_out_prior)
    res['v202_ece_88'] = report('v202 ECE outward@prior-cap', p202, fd, fv, '88')
    res['v202_ece_290'] = report('v202 ECE outward@prior-cap', p202, fd, fv, '290')
    # loose variant: no cap at all (upper bound on ANY outward operator)
    mv_out_free = lambda s, e, gs, ge, dur, g: best_outward(s, e, gs, ge, 1e9, dur)
    p202f, fd, fv = apply_scoped(gt, pred, mv_out_free)
    res['v202_outward_uncapped_88'] = report('v202 outward UNCAPPED (loose)', p202f, fd, fv, '88')

    # =====================================================================
    # v203 CBT-ALL -- INWARD, only the 9 non-hard cats, only over-wide dets
    # =====================================================================
    vf_nh = lambda v, g: g['category'] not in HARD
    df_wide = lambda v, g, d: (float(d['t_end']) - float(d['t_start'])) > CBT_TRIG * cap_of(g['category'])
    p203, fd, fv = apply_scoped(gt, pred, mv_in, vid_filter=vf_nh, det_filter=df_wide)
    res['v203_cbt_all_290'] = report('v203 CBT-ALL inward@prior-cap trig1.1', p203, fd, fv, '290')
    # relaxed gate: trigger 1.0, and any non-hard det (no over-wide gate)
    df_wide10 = lambda v, g, d: (float(d['t_end']) - float(d['t_start'])) > 1.0 * cap_of(g['category'])
    p203b, fd, fv = apply_scoped(gt, pred, mv_in, vid_filter=vf_nh, det_filter=df_wide10)
    res['v203_trig10_290'] = report('v203 CBT-ALL trig1.0', p203b, fd, fv, '290')
    p203c, fd, fv = apply_scoped(gt, pred, mv_in, vid_filter=vf_nh)
    res['v203_ungated_290'] = report('v203 CBT-ALL ungated inward (loose)', p203c, fd, fv, '290')

    # =====================================================================
    # v204 BUE-recal -- re-tune lmax only. Reachable set = supersets of the
    # pre-BUE window with length <= the lmax actually chosen (<= 40s today).
    # =====================================================================
    prebue = {}
    for f in RUN.glob('*.json'):
        if f.name == 'evaluation_report.json':
            continue
        d = json.load(open(f))
        vid = Path(d.get('video_path', f.stem)).stem
        if d.get('pre_bue_boundaries'):
            prebue[vid] = d['pre_bue_boundaries']
    print(f"\n[v204] videos carrying pre_bue_boundaries: {len(prebue)}")
    moved = same = 0
    grow = []
    for vid, pb in prebue.items():
        dets = pred.get(vid, {}).get('anomalies', [])
        if len(pb) != len(dets):
            continue
        for (ps, pe), d in zip(pb, dets):
            if abs(ps - float(d['t_start'])) < 1e-9 and abs(pe - float(d['t_end'])) < 1e-9:
                same += 1
            else:
                moved += 1
                grow.append((float(d['t_end']) - float(d['t_start'])) - (pe - ps))
    print(f"[v204] dets whose final window == pre_bue: {same};  moved: {moved}")
    if grow:
        print(f"[v204] net length change (final - pre_bue): "
              f"median={statistics.median(grow):+.2f}s "
              f"min={min(grow):+.2f}s max={max(grow):+.2f}s")
    res['v204_prebue'] = {'videos': len(prebue), 'unmoved': same, 'moved': moved,
                          'median_len_delta': statistics.median(grow) if grow else None}

    # Oracle: replace each det by the best window that CONTAINS its pre_bue
    # window and is no longer than L. Sweep L over the per-category prior cap,
    # the global 40s, and an uncapped upper bound.
    def bue_oracle(L_fn, tag):
        out = copy.deepcopy(pred)
        fd = fv = 0
        for vid, gentry in gt.items():
            segs = gentry['segments']
            if not segs or vid not in out or vid not in prebue:
                continue
            dets = out[vid]['anomalies']
            pb = prebue[vid]
            if len(pb) != len(dets) or not dets:
                continue
            dur = out[vid].get('duration', 0.0) or 0.0
            idxs = sorted(range(len(dets)),
                          key=lambda i: -float(dets[i].get('confidence', 0.0)))
            free = set(range(len(segs)))
            touched = False
            for i in idxs:
                if not free:
                    break
                ps, pe = float(pb[i][0]), float(pb[i][1])
                L = L_fn(gentry['category'])
                best = None
                for j in sorted(free):
                    gs, ge = segs[j][0] / FPS, segs[j][1] / FPS
                    (ns, ne), q = best_outward(ps, pe, gs, ge, max(L, pe - ps), dur)
                    if best is None or q > best[0]:
                        best = (q, j, ns, ne)
                q, j, ns, ne = best
                free.discard(j)
                if (ns, ne) != (float(dets[i]['t_start']), float(dets[i]['t_end'])):
                    dets[i]['t_start'], dets[i]['t_end'] = ns, ne
                    fd += 1
                    touched = True
            if touched:
                fv += 1
        return report(tag, out, fd, fv, '88'), report(tag, out, fd, fv, '290')

    r88, r290 = bue_oracle(lambda c: cap_of(c), 'v204 BUE-recal percat-lmax oracle')
    res['v204_bue_percat_88'], res['v204_bue_percat_290'] = r88, r290
    r88g, _ = bue_oracle(lambda c: BUE_LMAX_GLOBAL, 'v204 BUE lmax=40 oracle (current)')
    res['v204_bue_lmax40_88'] = r88g
    r88u, _ = bue_oracle(lambda c: 1e9, 'v204 BUE lmax=inf oracle (loose)')
    res['v204_bue_uncapped_88'] = r88u

    # =====================================================================
    # v205 SNH -- CONFIDENCE ONLY. A perfect video-level P(normal) veto can
    # only touch detections on TRUE Normal videos. Two variants:
    #   demote  = confidence -> 0.0, detection KEPT (what SNH actually does)
    #   delete  = detection removed (banned family 9)
    # =====================================================================
    normals = [v for v, g in gt.items() if not g['segments']]
    p205 = copy.deepcopy(pred)
    nd = 0
    for v in normals:
        for d in p205.get(v, {}).get('anomalies', []):
            d['confidence'] = 0.0
            nd += 1
    res['v205_snh_demote_290'] = report('v205 SNH demote conf->0 (perfect veto)',
                                        p205, nd, len(normals), '290')
    p205d = copy.deepcopy(pred)
    for v in normals:
        if v in p205d:
            p205d[v]['anomalies'] = []
    res['v205_snh_delete_290'] = report('v205 SNH delete (family-9 form)',
                                        p205d, nd, len(normals), '290')
    # construction validation: no Normal video fired -> delta EXACTLY 0
    p205z = copy.deepcopy(pred)
    m0 = mp(gt, p205z)['mAP']
    print(f"[SELFTEST-v205] veto fires nowhere -> dmAP={m0 - base:+.9f} (must be 0)")
    assert abs(m0 - base) < 1e-12
    nd_vids = sum(1 for v in normals if pred.get(v, {}).get('anomalies'))
    print(f"[v205] Normal videos={len(normals)}  with >=1 detection={nd_vids}  "
          f"detections demoted={nd}")
    res['v205_pop'] = {'normal_videos': len(normals), 'with_det': nd_vids, 'dets': nd}
    # loose extra: perfect ranking WITHIN anomaly videos too (SNH cannot do this,
    # but it brackets the whole confidence layer once Normal FPs are gone)
    print("\n" + json.dumps(res, indent=1, default=str))
    return res


if __name__ == '__main__':
    main()
