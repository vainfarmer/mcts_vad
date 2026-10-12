"""
VIDEO-MATCHED BOOTSTRAP A/B — the standard verdict statistic from v141 onward.

WHY THIS REPLACES POINT DELTAS (v140 finding).
On this 65-video anomaly-only harness the video-level sampling error of a mAP delta has a 90% CI
half-width of ~+/-0.025 — five times the +0.005 gate used since v114. Point deltas therefore cannot
distinguish a real +0.01 lever from noise, and "N caches agree" is nearly worthless because the
cached runs share ~81% of their detection windows (mean pairwise Jaccard 0.813). Two consecutive
offline-validated mechanisms (v138, v139) were refuted in vivo, and the committed Q-score win
(+0.00619) is itself inside the noise band.

WHAT THIS DOES.
Both runs cover the SAME video set, so the comparison is matched per video. We resample VIDEOS with
replacement (the true sampling unit), recompute mAP for both arms on each resample, and report the
distribution of the difference. That removes video-sampling variance from the verdict; residual
VLM run-to-run variance remains for cross-run (non-paired) comparisons and is reported as a caveat.

Usage:
  # cross-run (a mechanism that changes detections): NEW vs REF directories
  python3.14 analysis_bootstrap_ab.py --new results/batch_eval_v141_full --ref results/batch_eval_v124_full
  # paired (a post-hoc mechanism logging conf_old->conf_new): one dir + its marker log
  python3.14 analysis_bootstrap_ab.py --new results/batch_eval_v141_full --paired-log results/batch_eval_v141_full/run.log --marker MYMARK

Verdict fields printed and written to <new>/bootstrap_ab.json:
  point_delta, ci90_low, ci90_high, p_gt0, per-IoU deltas, n_videos_common, and PASS/FAIL of the
  pre-registered gate (point_delta >= min_delta AND p_gt0 >= min_p AND new_mAP >= floor).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map  # noqa: E402

LABEL_PATH = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30


def restore_pre_marker(pred, log_path, marker):
    """Rebuild the flag-OFF arm from `<marker> ... conf=<old>-><new>` lines (paired mode)."""
    pat = re.compile(re.escape(marker) + r'.*?t=(?P<ts>[-\d.]+)-(?P<te>[-\d.]+).*?'
                     r'conf=(?P<old>[\d.]+)->(?P<new>[\d.]+)')
    vid_pat = re.compile(r'vid=(\S+)')
    fires = {}
    for line in Path(log_path).read_text(errors='replace').splitlines():
        m = pat.search(line)
        if not m:
            continue
        v = vid_pat.search(line)
        if not v:
            continue
        fires[(v.group(1), round(float(m.group('ts')), 2),
               round(float(m.group('te')), 2))] = float(m.group('old'))
    out, matched = {}, 0
    for vid, e in pred.items():
        outs = []
        for a in e['anomalies']:
            a2 = dict(a)
            k = (vid, round(a['t_start'], 2), round(a['t_end'], 2))
            if k in fires:
                a2['confidence'] = fires[k]
                matched += 1
            outs.append(a2)
        out[vid] = dict(e, anomalies=outs)
    return out, matched, len(fires)


def restore_pre_marker_window(pred, log_path, marker):
    """Rebuild the flag-OFF arm from `<marker> ... win=<a>-<b>-><c>-<d>` lines (paired WINDOW mode).

    v141 (CEX) need: the conf-mode restorer above keys on the FINAL (vid, t_start, t_end) and only
    rewrites 'confidence', so it CANNOT reconstruct a BOUNDARY change. Here we key on the FINAL
    (c, d) and restore t_start/t_end to the pre-mechanism (a, b). compute_temporal_map reads only
    {video_id, t_start, t_end, confidence} (evaluate.py), so this reproduces the flag-off arm exactly
    for any count-preserving, confidence-frozen window operator.
    """
    pat = re.compile(re.escape(marker) + r'.*?win=(?P<a>[-\d.]+)-(?P<b>[-\d.]+)->'
                     r'(?P<c>[-\d.]+)-(?P<d>[-\d.]+)')
    vid_pat = re.compile(r'vid=(\S+)')
    fires = {}
    for line in Path(log_path).read_text(errors='replace').splitlines():
        m = pat.search(line)
        if not m:
            continue
        v = vid_pat.search(line)
        if not v:
            continue
        vid = Path(v.group(1)).stem
        fires[(vid, round(float(m.group('c')), 2), round(float(m.group('d')), 2))] = (
            float(m.group('a')), float(m.group('b')))
    out, matched = {}, 0
    for vid, e in pred.items():
        outs = []
        for a in e['anomalies']:
            a2 = dict(a)
            k = (vid, round(a['t_start'], 2), round(a['t_end'], 2))
            if k in fires:
                a2['t_start'], a2['t_end'] = fires[k]
                matched += 1
            outs.append(a2)
        out[vid] = dict(e, anomalies=outs)
    return out, matched, len(fires)


def restore_pre_json(pred, new_dir, field='pre_cex_boundaries'):
    """Rebuild the flag-OFF arm from an index-aligned `<field>` list in the result JSONs (M4).

    PRIMARY paired path for v141 CEX: no regex parsing and no float-rounding key collisions. The
    field is written by the pipeline only when the mechanism's flag is on, is index-aligned with
    'anomalies', and holds the pre-mechanism [t_start, t_end]. Returns (ref, matched, n_fires,
    bad_videos) where bad_videos lists any video whose field length disagrees with its detection
    count — a HARD VOID condition (M4).
    """
    sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
    from evaluate import _extract_video_id  # noqa: E402
    pre = {}
    for jf in sorted(Path(new_dir).glob('*.json')):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict) or data.get(field) is None:
            continue
        pre[_extract_video_id(data.get('video_path', jf.stem))] = data[field]
    out, matched, n_fires, bad = {}, 0, 0, []
    for vid, e in pred.items():
        rows = pre.get(vid)
        if rows is None:
            out[vid] = e
            continue
        if len(rows) != len(e['anomalies']):
            bad.append(f'{vid}({len(rows)}!={len(e["anomalies"])})')
            out[vid] = e
            continue
        outs = []
        for a, row in zip(e['anomalies'], rows):
            a2 = dict(a)
            a2['t_start'], a2['t_end'] = float(row[0]), float(row[1])
            if (abs(a2['t_start'] - a['t_start']) > 1e-6
                    or abs(a2['t_end'] - a['t_end']) > 1e-6):
                matched += 1
            n_fires += 1
            outs.append(a2)
        out[vid] = dict(e, anomalies=outs)
    return out, matched, n_fires, bad


def bootstrap(gt, predA, predB, n=1000, seed=0):
    """Distribution of mAP(B) - mAP(A) under video-level resampling. A = reference arm."""
    vids = sorted(set(predA) & set(predB) & set(gt))
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        samp = rng.choice(vids, size=len(vids), replace=True)
        g, a, b = {}, {}, {}
        for i, v in enumerate(samp):
            k = f'{v}#{i}'
            g[k], a[k], b[k] = gt[v], predA[v], predB[v]
        ds.append(compute_temporal_map(g, b, fps=FPS)['mAP']
                  - compute_temporal_map(g, a, fps=FPS)['mAP'])
    return np.array(ds), vids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new', required=True)
    p.add_argument('--ref')
    p.add_argument('--paired-log')
    p.add_argument('--marker')
    p.add_argument('--paired-mode', choices=['conf', 'window'], default='conf',
                   help="what the marker restores: 'conf' (v138/v139 post-hoc confidence "
                        "transforms, unchanged default) or 'window' (v141 CEX boundary operator, "
                        "parses win=<a>-<b>-><c>-<d>)")
    p.add_argument('--paired-json', metavar='FIELD', nargs='?', const='pre_cex_boundaries',
                   help='PRIMARY paired path (M4): rebuild the flag-off arm from an index-aligned '
                        'pre-mechanism boundary list in the result JSONs (default field '
                        'pre_cex_boundaries). Needs only --new.')
    p.add_argument('--resamples', type=int, default=1000)
    p.add_argument('--min-delta', type=float, default=0.02)
    p.add_argument('--min-p', type=float, default=0.90)
    p.add_argument('--floor', type=float, default=0.15132)
    args = p.parse_args()

    gt = load_ground_truth(LABEL_PATH, split='test')
    new = load_predictions(str(args.new))
    if args.paired_json:
        mode = f'paired-json[{args.paired_json}]'
    elif args.paired_log:
        mode = f'paired-{args.paired_mode}'
    else:
        mode = 'cross-run'
    if args.paired_json:
        # M4 PRIMARY PATH: exact reconstruction from the result JSONs.
        ref, n_changed, n_rows, bad = restore_pre_json(new, args.new, args.paired_json)
        print(f'paired-json mode: {n_rows} index-aligned pre-mechanism rows, '
              f'{n_changed} detections whose window actually changed')
        if n_rows == 0:
            print(f'FM5 GATE FAILED: no {args.paired_json} rows — mechanism never ran. '
                  f'VERDICT: REVERT')
            return 1
        if bad:
            print(f'PAIRED VERDICT VOID (M4): {len(bad)} video(s) with a length mismatch between '
                  f'{args.paired_json} and anomalies: {", ".join(bad[:10])}')
            return 3
        if n_changed == 0:
            print('PAIRED VERDICT VOID (M4): mechanism logged rows but changed ZERO windows — '
                  'the two arms are identical, there is nothing to measure.')
            return 3
        if args.paired_log and args.marker:
            # integrity cross-check against the log markers (must agree exactly)
            _, m_log, n_log = restore_pre_marker_window(new, args.paired_log, args.marker)
            print(f'  cross-check vs [{args.marker}] log markers: {n_log} window fires, '
                  f'{m_log} matched')
            if m_log != n_changed:
                print(f'PAIRED VERDICT VOID (M4): JSON says {n_changed} changed windows but the '
                      f'log markers restore {m_log} — the arms cannot both be right.')
                return 3
    elif args.paired_log:
        if not args.marker:
            print('paired mode requires --marker')
            return 2
        if args.paired_mode == 'window':
            ref, matched, n_fires = restore_pre_marker_window(new, args.paired_log, args.marker)
            what = 'window'
        else:
            ref, matched, n_fires = restore_pre_marker(new, args.paired_log, args.marker)
            what = 'confidence'
        print(f'paired mode ({what}): {n_fires} marker fires, {matched} detections restored to '
              f'pre-mechanism {what}')
        if n_fires == 0:
            print('FM5 GATE FAILED: no marker lines — mechanism never ran. VERDICT: REVERT')
            return 1
        if matched != n_fires:
            print(f'PAIRED VERDICT VOID (M4): restored {matched} of {n_fires} recorded fires — '
                  f'the flag-off arm is incomplete, so no delta may be reported.')
            return 3
    else:
        if not args.ref:
            print('cross-run mode requires --ref')
            return 2
        ref = load_predictions(str(args.ref))

    gte = {k: v for k, v in gt.items() if k in new and k in ref}
    r_new = compute_temporal_map(gte, {k: new[k] for k in gte}, fps=FPS)
    r_ref = compute_temporal_map(gte, {k: ref[k] for k in gte}, fps=FPS)
    print(f'\nmode={mode}  common videos={len(gte)}')
    print(f'{"arm":6s} {"mAP":>9s} {"AP@0.3":>9s} {"AP@0.5":>9s} {"AP@0.7":>9s}')
    for nm, r in (('REF', r_ref), ('NEW', r_new)):
        print(f'{nm:6s} {r["mAP"]:9.5f} {r["AP@0.3"]:9.5f} {r["AP@0.5"]:9.5f} {r["AP@0.7"]:9.5f}')
    point = r_new['mAP'] - r_ref['mAP']
    print(f'{"delta":6s} {point:+9.5f} {r_new["AP@0.3"] - r_ref["AP@0.3"]:+9.5f} '
          f'{r_new["AP@0.5"] - r_ref["AP@0.5"]:+9.5f} {r_new["AP@0.7"] - r_ref["AP@0.7"]:+9.5f}')

    ds, vids = bootstrap(gt, {k: ref[k] for k in gte}, {k: new[k] for k in gte},
                         n=args.resamples, seed=0)
    lo, hi = float(np.percentile(ds, 5)), float(np.percentile(ds, 95))
    pgt = float((ds > 0).mean())
    print(f'\nvideo-level bootstrap ({args.resamples} resamples, n={len(vids)} videos):')
    print(f'  mean {ds.mean():+.5f}  90% CI [{lo:+.5f}, {hi:+.5f}]  P(delta>0)={pgt:.3f}  '
          f'sd {ds.std():.5f}')

    ok_d = point >= args.min_delta
    ok_p = pgt >= args.min_p
    ok_f = r_new['mAP'] >= args.floor
    verdict = 'KEEP' if (ok_d and ok_p and ok_f) else 'REVERT'
    print(f'\nGATE  point_delta >= {args.min_delta}: {ok_d} ({point:+.5f})'
          f'   P(delta>0) >= {args.min_p}: {ok_p} ({pgt:.3f})'
          f'   mAP >= floor {args.floor}: {ok_f} ({r_new["mAP"]:.5f})')
    print(f'VERDICT: {verdict}')
    if mode == 'cross-run':
        print('CAVEAT: cross-run mode still carries VLM run-to-run variance (~+/-0.019 unpaired in '
              'this ledger) ON TOP of the bootstrap CI. Prefer a paired design when the mechanism '
              'is post-hoc.')
    out = Path(args.new) / 'bootstrap_ab.json'
    out.write_text(json.dumps({'mode': mode, 'new': r_new, 'ref': r_ref, 'point_delta': point,
                               'ci90': [lo, hi], 'p_gt0': pgt, 'n_videos': len(vids),
                               'verdict': verdict}, indent=2, default=float))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
