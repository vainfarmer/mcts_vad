#!/usr/bin/env python3.14
"""
v206 ECE-AT-TERMINAL OFFLINE PRECHECK (FM13). ZERO API calls, ZERO evals.

QUESTION
--------
v202 established that ECE at its CURRENT front slot has an EXACT-ZERO effect on
final mAP (BUE runs after it, grows outward and takes the UNION, so it already
occupies ECE's move set). The fix under test is purely POSITIONAL: move ECE to
the terminal-most slot, AFTER CBT, where nothing downstream can overwrite it.

WHY THE OFFLINE REPLAY IS EXACT (not a ceiling)
-----------------------------------------------
_event_continuation_extension is a pure function of
  (boundaries, video_data['timestamps'], read-only category curve, cat, CATEGORY_DURATION_PRIORS)
It makes ZERO VLM/CLIP/MIL calls and writes ONLY a['t_start'] / a['t_end'].
At the terminal slot there is NO operator downstream of it, so replaying it
offline on a cached run's FINAL post-CBT boundaries reproduces exactly what an
in-run terminal ECE would have produced. This is the same argument that makes
the dual-arm rig exact (PROJECT_DOCUMENTATION 8.3.2). FM6 (offline gains do not
transfer) bites when downstream operators REACT to the change; here there is no
downstream.

THE ONE WAY IT CAN FAIL: the category curve must be reconstructed offline
bit-faithfully to the in-run frame_scorer.get_category_scores call. That is
gated, not assumed, by two checks:
  G1  timestamps grid: offline npz grid == the grid persisted in the run JSON
  G2  ECE END-TO-END: replay ECE on the v202 selftest's pre_ece_boundaries and
      require the output to equal that run's pre_bue_boundaries EXACTLY
      (EUBC/CGR/CCR are default-OFF and qscore touches only confidence, so
      pre_bue == post-ECE in that run, bit-for-bit).

Substrate: results/batch_eval_v26_full290 (290/290, failed=0, mAP 0.16682),
results/batch_eval_v202_ece_selftest, and the v140 live-faithful curve library
(local CLIP only, no VLM client constructible).
"""
import sys
import json
import copy
import argparse
from pathlib import Path

import numpy as np

REPO = Path('/Users/zzl/PycharmProjects/vad')
VAD = REPO / 'video_anomaly_detection'
sys.path.insert(0, str(VAD))
sys.path.insert(0, str(VAD / 'iteration_logs'))

import evaluate                                    # noqa: E402  (loaders + scorer only)
from pipeline import CATEGORY_DURATION_PRIORS      # noqa: E402
from run import UCF_CRIME_CATEGORIES               # noqa: E402
import analysis_v140_livecurve_lib as LC           # noqa: E402

GT_PATH = '/Users/zzl/Movies/ucf-crime/data'
RUN = REPO / 'results' / 'batch_eval_v26_full290'
SELFTEST = REPO / 'results' / 'batch_eval_v202_ece_selftest'
CURVE_CACHE = VAD / 'iteration_logs' / 'analysis_v140_livecurves'
FPS = 30
IOUS = [0.3, 0.5, 0.7]
HARD = ('Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion')
CATS = list(UCF_CRIME_CATEGORIES)


# ---------------------------------------------------------------- curve access
_curve_mem = {}


def curve_for(vid, cat, allow_build=False):
    """rel curve + timestamps from the v140 live-faithful substrate.

    allow_build=False => cache-only (returns None if the npz is absent), so the
    cheap phases never pay CLIP cost.
    """
    key = (vid, cat)
    if key in _curve_mem:
        return _curve_mem[key]
    if not allow_build and not (CURVE_CACHE / f'{vid}.npz').exists():
        _curve_mem[key] = (None, None)
        return None, None
    try:
        c, ts = LC.live_curve(vid, cat)
    except Exception as e:                                  # pragma: no cover
        print(f'  [curve-err] {vid}/{cat}: {e}')
        c, ts = None, None
    _curve_mem[key] = (c, ts)
    return c, ts


# ------------------------------------------------- exact ECE rule (pipeline.py L3481)
def ece_replay(windows, cats, vid, timestamps, fallback_cat=None,
               allow_build=False, trace=None):
    """Mirror of AnomalyPipeline._event_continuation_extension.

    windows: list[[t_start, t_end]]; cats: list[anomaly_type] index-aligned.
    Returns (new_windows, counts). Boundaries only; count frozen by construction.
    """
    counts = {'material': 0, 'snap': 0, 'reject_cap': 0, 'skip': 0}
    ts = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else None
    n_ts = len(ts) if ts is not None else 0
    out = []
    for i, (w, c0) in enumerate(zip(windows, cats)):
        t_s, t_e = float(w[0]), float(w[1])
        dur = t_e - t_s

        cat = c0
        if cat is None or cat in ('Normal', 'unknown') or cat not in CATS:
            cat = fallback_cat
        if cat is None or cat in ('Normal', 'unknown') or cat not in CATS:
            counts['skip'] += 1
            out.append([t_s, t_e])
            continue

        max_dur = CATEGORY_DURATION_PRIORS.get(cat, {}).get(
            'expected_duration', (3.0, 30.0))[1]
        if dur >= max_dur:
            counts['skip'] += 1
            out.append([t_s, t_e])
            continue

        if n_ts < 2:
            counts['skip'] += 1
            out.append([t_s, t_e])
            continue

        rel, _cts = curve_for(vid, cat, allow_build=allow_build)
        if rel is None or len(rel) != n_ts:
            counts['skip'] += 1
            out.append([t_s, t_e])
            if trace is not None:
                trace.append((i, 'curve-miss', t_s, t_e, t_s, t_e, cat, None))
            continue
        rel = np.asarray(rel, dtype=np.float64)

        hm = (float(np.max(rel)) + float(np.percentile(rel, 20))) / 2.0
        s = min(int(np.searchsorted(ts, t_s)), n_ts - 1)
        e = min(int(np.searchsorted(ts, t_e)), n_ts - 1)
        L = s
        while L - 1 >= 0 and rel[L - 1] >= hm:
            L -= 1
        R = e
        while R + 1 <= n_ts - 1 and rel[R + 1] >= hm:
            R += 1

        new_start = float(ts[L])
        new_end = float(ts[R])
        span = new_end - new_start

        if span > max_dur:
            counts['reject_cap'] += 1
            out.append([t_s, t_e])
            if trace is not None:
                trace.append((i, 'reject-cap', t_s, t_e, t_s, t_e, cat, hm))
            continue

        material = (L < s) or (R > e)
        counts['material' if material else 'snap'] += 1
        out.append([new_start, new_end])
        if trace is not None:
            trace.append((i, 'material' if material else 'grid-snap',
                          t_s, t_e, new_start, new_end, cat, hm))
    return out, counts


# ------------------------------------------------------------------ scoring util
def mp(gt, pred):
    return evaluate.compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=IOUS)['mAP']


def load_run(run_dir):
    recs = {}
    for p in sorted(Path(run_dir).glob('*.json')):
        if p.name == 'evaluation_report.json':
            continue
        recs[p.stem] = json.load(open(p))
    return recs


def to_pred(recs, override=None):
    """Build the dict evaluate.compute_temporal_map consumes.

    override: {vid: [[s,e], ...]} replacement windows, index-aligned.
    """
    pred = {}
    for vid, d in recs.items():
        an = d.get('anomalies') or []
        ov = (override or {}).get(vid)
        segs = []
        for i, a in enumerate(an):
            s, e = (ov[i] if ov is not None else (a['t_start'], a['t_end']))
            b = dict(a)
            b['t_start'] = float(s)
            b['t_end'] = float(e)
            segs.append(b)
        pred[vid] = {'anomalies': segs, 'is_normal': d.get('is_normal'),
                     'duration': d.get('duration')}
    return pred


# ---------------------------------------------------------------------- phases
def gate_coverage():
    rep = json.load(open(RUN / 'evaluation_report.json'))
    t = rep.get('timing', {})
    print(f'[GATE] {RUN.name}: num_videos={t.get("num_videos")} failed={t.get("failed")} '
          f'reported_mAP={rep.get("mAP", rep.get("overall", {}).get("mAP"))}')
    assert t.get('failed') == 0, f'COVERAGE GATE FAIL failed={t.get("failed")}'
    gt = evaluate.load_ground_truth(GT_PATH, split='test')
    pred = evaluate.load_predictions(str(RUN))
    missing = [k for k, v in gt.items() if v.get('segments') and k not in pred]
    print(f'[GATE] gt={len(gt)} pred={len(pred)} missing_anomaly={len(missing)}')
    assert not missing, f'missing anomaly videos {missing[:5]}'
    print(f'[GATE] recomputed full-290 mAP = {mp(gt, pred):.5f}  (doc: 0.16682)')
    return gt


def gate_grid(recs):
    ok = bad = miss = 0
    for vid, d in recs.items():
        f = CURVE_CACHE / f'{vid}.npz'
        if not f.exists():
            miss += 1
            continue
        z = np.load(f)
        a = np.asarray(d['timestamps'], dtype=float)
        b = np.asarray(z['timestamps'], dtype=float)
        if a.shape == b.shape and np.array_equal(a, b):
            ok += 1
        else:
            bad += 1
            print(f'  [G1-MISMATCH] {vid} run={a.shape} offline={b.shape}')
    print(f'[G1] timestamps grid exact-equal {ok} / mismatch {bad} / uncached {miss}')
    return bad == 0


def gate_ece_endtoend():
    """G2: replay ECE on v202 pre_ece_boundaries; require == pre_bue_boundaries."""
    recs = load_run(SELFTEST)
    tot = agree = disagree = 0
    vok = vbad = 0
    for vid, d in sorted(recs.items()):
        pre = d.get('pre_ece_boundaries')
        post_ref = d.get('pre_bue_boundaries')
        an = d.get('anomalies') or []
        if pre is None or post_ref is None:
            continue
        cats = [a.get('anomaly_type') for a in an]
        if len(cats) != len(pre):
            print(f'  [G2-SKIP] {vid} len(anomalies)={len(cats)} len(pre)={len(pre)}')
            continue
        trace = []
        got, counts = ece_replay(pre, cats, vid, d['timestamps'],
                                 fallback_cat=None, allow_build=True, trace=trace)
        # qscore_rerank may REORDER, so compare as sorted multisets of windows.
        a = sorted([round(x, 6), round(y, 6)] for x, y in got)
        b = sorted([round(float(x), 6), round(float(y), 6)] for x, y in post_ref)
        tot += len(a)
        if a == b:
            agree += len(a)
            vok += 1
        else:
            vbad += 1
            disagree += len(a)
            print(f'  [G2-MISMATCH] {vid}\n     replay={a}\n     in-run={b}\n     '
                  f'pre={pre} counts={counts}')
    print(f'[G2] ECE end-to-end on v202 selftest: {vok} videos exact / {vbad} mismatched '
          f'({agree}/{tot} detections)')
    return vbad == 0, vok, vbad


def build_curves(recs, only_hard=False, limit=None):
    """Pre-build the npz curve cache for every video that has a live ECE candidate."""
    need = []
    for vid, d in sorted(recs.items()):
        if only_hard and not vid.startswith(HARD):
            continue
        if (CURVE_CACHE / f'{vid}.npz').exists():
            continue
        for a in d.get('anomalies') or []:
            cat = a.get('anomaly_type')
            if cat not in CATS:
                continue
            md = CATEGORY_DURATION_PRIORS.get(cat, {}).get('expected_duration', (3., 30.))[1]
            if (a['t_end'] - a['t_start']) < md:
                need.append(vid)
                break
    if limit:
        need = need[:limit]
    print(f'[BUILD] {len(need)} curve caches to build', flush=True)
    for i, vid in enumerate(need, 1):
        try:
            LC.live_video_data(vid)
        except Exception as e:
            print(f'  [BUILD-ERR] {vid}: {e}', flush=True)
        if i % 10 == 0 or i == len(need):
            print(f'  [BUILD] {i}/{len(need)}', flush=True)


def replay_final(recs, allow_build=True):
    """Apply terminal-slot ECE to the cached run's FINAL post-CBT boundaries."""
    override = {}
    agg = {'material': 0, 'snap': 0, 'reject_cap': 0, 'skip': 0}
    traces = {}
    curve_miss = 0
    for vid, d in sorted(recs.items()):
        an = d.get('anomalies') or []
        if not an:
            continue
        win = [[a['t_start'], a['t_end']] for a in an]
        cats = [a.get('anomaly_type') for a in an]
        tr = []
        got, c = ece_replay(win, cats, vid, d['timestamps'], fallback_cat=None,
                            allow_build=allow_build, trace=tr)
        for k in agg:
            agg[k] += c[k]
        curve_miss += sum(1 for t in tr if t[1] == 'curve-miss')
        override[vid] = got
        traces[vid] = tr
        assert len(got) == len(win), f'FM2 VIOLATION {vid}: {len(win)} -> {len(got)}'
    return override, agg, traces, curve_miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='gates',
                    choices=['gates', 'build', 'score'])
    ap.add_argument('--build-limit', type=int, default=None)
    ap.add_argument('--hard-only', action='store_true')
    args = ap.parse_args()

    gt = gate_coverage()
    recs = load_run(RUN)
    print(f'[SUB] {len(recs)} per-video JSONs, '
          f'{sum(len(d.get("anomalies") or []) for d in recs.values())} detections')

    if args.phase == 'gates':
        gate_grid(recs)
        ok, vok, vbad = gate_ece_endtoend()
        print(f'\nG2 FAITHFUL = {ok}')
        return

    if args.phase == 'build':
        build_curves(recs, only_hard=args.hard_only, limit=args.build_limit)
        return

    # ---------------- score ----------------
    override, agg, traces, curve_miss = replay_final(recs, allow_build=True)
    print(f'[ECE-TERMINAL] material={agg["material"]} snaps={agg["snap"]} '
          f'reject_caps={agg["reject_cap"]} skips={agg["skip"]} curve_miss={curve_miss}')

    pred_base = to_pred(recs)
    pred_ece = to_pred(recs, override)

    hard_gt = {k: v for k, v in gt.items() if k.startswith(HARD)}
    hard_base = {k: v for k, v in pred_base.items() if k.startswith(HARD)}
    hard_ece = {k: v for k, v in pred_ece.items() if k.startswith(HARD)}

    m290_b, m290_a = mp(gt, pred_base), mp(gt, pred_ece)
    m88_b, m88_a = mp(hard_gt, hard_base), mp(hard_gt, hard_ece)
    print(f'\n[mAP] full-290  base={m290_b:.5f}  ECE-terminal={m290_a:.5f}  '
          f'delta={m290_a - m290_b:+.5f}')
    print(f'[mAP] hard-88   base={m88_b:.5f}  ECE-terminal={m88_a:.5f}  '
          f'delta={m88_a - m88_b:+.5f}')
    print(f'[HARNESS] hard gt videos={len(hard_gt)} pred videos={len(hard_base)}')

    # ---- n_fire (material extensions), split by harness
    n_fire_all = agg['material']
    n_fire_hard = sum(1 for v, tr in traces.items() if v.startswith(HARD)
                      for t in tr if t[1] == 'material')
    print(f'[n_fire] material extensions: all={n_fire_all} hard-88={n_fire_hard}')

    # ---- cbt_undo: how much of CBT's inward clamp does terminal ECE reverse?
    tot_clamp = tot_undo = 0.0
    nv = nd = 0
    for vid, d in sorted(recs.items()):
        pre_cbt = d.get('pre_cbt_boundaries')
        an = d.get('anomalies') or []
        if not pre_cbt or len(pre_cbt) != len(an):
            continue
        ov = override.get(vid)
        for i, a in enumerate(an):
            ps, pe = float(pre_cbt[i][0]), float(pre_cbt[i][1])
            fs, fe = float(a['t_start']), float(a['t_end'])
            clamp = (ps - fs) + (pe - fe)          # >0 == CBT pulled inward
            if clamp <= 1e-9:
                continue
            nd += 1
            tot_clamp += clamp
            es, ee = (ov[i] if ov else (fs, fe))
            undo = max(0.0, fs - es) + max(0.0, ee - fe)   # ECE grows back outward
            tot_undo += min(undo, clamp)
        nv += 1
    frac = (tot_undo / tot_clamp) if tot_clamp > 0 else 0.0
    print(f'[cbt_undo] CBT-clamped detections={nd} total inward clamp={tot_clamp:.1f}s  '
          f'reversed by terminal ECE={tot_undo:.1f}s  fraction={frac:.3f}')

    # ---- per-video hard-88 attribution of the delta
    deltas = []
    for vid in sorted(hard_base):
        one_gt = {vid: hard_gt[vid]} if vid in hard_gt else None
        if not one_gt:
            continue
        b = mp(one_gt, {vid: hard_base[vid]})
        a = mp(one_gt, {vid: hard_ece[vid]})
        if abs(a - b) > 1e-9:
            deltas.append((a - b, vid, b, a))
    deltas.sort()
    print(f'[per-video hard-88] {len(deltas)} videos changed AP')
    for dv, vid, b, a in deltas[:8]:
        print(f'   {dv:+.4f}  {vid}  {b:.4f} -> {a:.4f}')
    if len(deltas) > 16:
        print('   ...')
    for dv, vid, b, a in deltas[-8:]:
        print(f'   {dv:+.4f}  {vid}  {b:.4f} -> {a:.4f}')

    json.dump({'m290_base': m290_b, 'm290_ece': m290_a, 'd290': m290_a - m290_b,
               'm88_base': m88_b, 'm88_ece': m88_a, 'd88': m88_a - m88_b,
               'counts': agg, 'curve_miss': curve_miss,
               'n_fire_all': n_fire_all, 'n_fire_hard': n_fire_hard,
               'cbt_clamp_s': tot_clamp, 'cbt_undo_s': tot_undo, 'cbt_undo_frac': frac},
              open(VAD / 'iteration_logs' / 'analysis_v206_ece_terminal_precheck.json', 'w'),
              indent=2)


if __name__ == '__main__':
    main()
