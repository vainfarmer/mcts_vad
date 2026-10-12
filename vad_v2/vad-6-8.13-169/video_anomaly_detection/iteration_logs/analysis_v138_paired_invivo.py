"""
v138 PAIRED IN-VIVO A/B — exact GPC delta on live detections, zero VLM noise.

WHY PAIRED. The ledger's own history shows a ~+/-0.019 run-to-run full-eval mAP
spread between configurations that are mAP-identical BY CONSTRUCTION (v130 0.17708 /
v131 0.17163 / v132 0.15847, all Q-score-era with only inert or classify-only
changes). The expected GPC effect (+0.019 offline) is the SAME magnitude, so an
unpaired "v138 vs v124 REF" comparison cannot resolve it — that ambiguity is what
produced most of the session's "within noise band" verdicts.

GPC is a pure multiplicative post-hoc factor applied as the LAST confidence touch,
so the SAME live run contains both arms: each per-fire [GPC] log line records
conf_old -> conf_new for a detection whose boundaries and count are already final.
Restoring conf_old in the run's own cached artifacts reconstructs the flag-OFF arm
EXACTLY (byte-identical detections, same VLM verdicts, same boundaries), giving a
noise-free paired delta.

Usage:
  python3.14 analysis_v138_paired_invivo.py <run_log> <result_dir>
"""
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import (  # noqa: E402
    load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou,
)

LABEL_PATH = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
LINE = re.compile(
    r'\[GPC\] vid=(?P<vid>\S+) t=(?P<ts>[-\d.]+)-(?P<te>[-\d.]+) cat=(?P<cat>\S+) '
    r'peak=(?P<peak>[-\d.]+) in=(?P<inside>\d) '
    r'conf=(?P<old>[\d.]+)->(?P<new>[\d.]+) w=(?P<w>[\d.]+)')


def parse_log(log_path):
    """(video_id, t_start_2dp, t_end_2dp) -> {old, new, inside, peak}"""
    fires = {}
    n = 0
    for line in Path(log_path).read_text(errors='replace').splitlines():
        m = LINE.search(line)
        if not m:
            continue
        n += 1
        g = m.groupdict()
        fires[(g['vid'], round(float(g['ts']), 2), round(float(g['te']), 2))] = dict(
            old=float(g['old']), new=float(g['new']), inside=int(g['inside']),
            peak=float(g['peak']), w=float(g['w']))
    return fires, n


def main():
    log_path, result_dir = sys.argv[1], sys.argv[2]
    fires, n_lines = parse_log(log_path)
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(result_dir)
    gte = {k: v for k, v in gt.items() if k in pred}
    n_det = sum(len(e['anomalies']) for e in pred.values())
    print(f'[GPC] log lines: {n_lines} ({len(fires)} unique windows); '
          f'cached detections: {n_det} in {len(pred)} videos')
    if not fires:
        print('FM5 GATE FAILED: no [GPC] markers in the log — mechanism never ran.')
        return 1

    # rebuild the flag-OFF arm by restoring conf_old on matched detections
    matched = unmatched = 0
    off = {}
    for vid, e in pred.items():
        outs = []
        for a in e['anomalies']:
            k = (vid, round(a['t_start'], 2), round(a['t_end'], 2))
            f = fires.get(k)
            a2 = dict(a)
            if f is None:
                unmatched += 1
            else:
                matched += 1
                a2['confidence'] = f['old']
                if abs(f['new'] - float(a['confidence'])) > 1e-4:
                    print(f'  WARN {k}: log new={f["new"]:.6f} != cached '
                          f'{float(a["confidence"]):.6f} (later mutation?)')
            outs.append(a2)
        off[vid] = dict(e, anomalies=outs)
    print(f'matched {matched}/{n_det} detections to [GPC] fires '
          f'(unmatched {unmatched} — expected 0 when every detection had a usable '
          f'category+curve)')

    r_on = compute_temporal_map(gte, pred, fps=FPS)
    r_off = compute_temporal_map(gte, off, fps=FPS)
    print()
    print(f'{"arm":8s} {"mAP":>9s} {"AP@0.3":>9s} {"AP@0.5":>9s} {"AP@0.7":>9s}')
    for name, r in (('OFF', r_off), ('ON', r_on)):
        print(f'{name:8s} {r["mAP"]:9.5f} {r["AP@0.3"]:9.5f} {r["AP@0.5"]:9.5f} '
              f'{r["AP@0.7"]:9.5f}')
    d = {k: r_on[k] - r_off[k] for k in r_on}
    print(f'{"PAIRED d":8s} {d["mAP"]:+9.5f} {d["AP@0.3"]:+9.5f} {d["AP@0.5"]:+9.5f} '
          f'{d["AP@0.7"]:+9.5f}')

    # demotion diagnostics on live data
    dem = [f for f in fires.values() if f['inside'] == 0]
    print(f'\nfired: {len(fires)}  demoted (peak outside window): {len(dem)} '
          f'({100 * len(dem) / max(1, len(fires)):.1f}%)')
    hit = miss = fh = fm = 0
    for (vid, ts, te), f in fires.items():
        if vid not in gte:
            continue
        seg = {'t_start': ts, 't_end': te}
        iou = max([_temporal_iou(seg, s, FPS) for s in gte[vid]['segments']] + [0.0])
        if f['inside'] == 0:
            (hit if iou >= 0.3 else miss).__class__  # noqa
            if iou >= 0.3:
                fh += 1
            else:
                fm += 1
        else:
            if iou >= 0.3:
                hit += 1
            else:
                miss += 1
    print(f'  demoted & IoU>=0.3 (wrong demotions): {fh}   demoted & IoU<0.3 '
          f'(correct): {fm}  -> demotion precision {fm / max(1, fh + fm):.3f}')
    print(f'  kept & IoU>=0.3: {hit}   kept & IoU<0.3: {miss}')
    out = Path(result_dir) / 'gpc_paired_ab.json'
    out.write_text(json.dumps({'on': r_on, 'off': r_off, 'delta': d,
                               'n_fires': len(fires), 'n_demoted': len(dem),
                               'demotion_precision': fm / max(1, fh + fm)},
                              indent=2, default=float))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
