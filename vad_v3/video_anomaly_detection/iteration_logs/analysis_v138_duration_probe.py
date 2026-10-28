"""
v138 diagnostic probe #3 (Agent B, research-only) — is there a systematic
DETECTION-DURATION vs GT-DURATION scale bias, and does a count/order-frozen
temporal dilation of final detections move mAP?

Consumes only saved artifacts: results/batch_eval_v124_full (+ v114/v131/v132 for
replication) and the parquet GT. No VLM, no CLIP, no model.
"""
import sys, json, glob
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map  # noqa

LABEL = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30
THRESH = [0.3, 0.5, 0.7]


def iou_s(ps, pe, g):
    gs, ge = g[0] / FPS, g[1] / FPS
    ov = max(0.0, min(pe, ge) - max(ps, gs))
    un = max(pe, ge) - min(ps, gs)
    return ov / un if un > 0 else 0.0


def load(dirname, gt_all):
    out = {}
    for jf in sorted(glob.glob(str(ROOT / dirname / '*.json'))):
        v = Path(jf).stem
        if v not in gt_all:
            continue
        out[v] = json.load(open(jf))
    return out


def mp(gt, dets):
    return compute_temporal_map(
        gt, {v: {'anomalies': j['anomalies']} for v, j in dets.items()},
        fps=FPS, iou_thresholds=THRESH)


def main():
    gt_all = load_ground_truth(LABEL, split='test')

    # ---------- Part 1: scale-bias statistics on the MATCHED pairs ----------
    dets = load('results/batch_eval_v124_full', gt_all)
    gt = {v: gt_all[v] for v in dets}
    ratios, gtd, pdd = [], [], []
    for v, j in dets.items():
        for g in gt_all[v]['segments']:
            gd = (g[1] - g[0]) / FPS
            # the detection in this video with max IoU to this GT
            cands = [(iou_s(a['t_start'], a['t_end'], g), a) for a in j['anomalies']]
            if not cands:
                continue
            bi, a = max(cands, key=lambda x: x[0])
            dd = a['t_end'] - a['t_start']
            gtd.append(gd); pdd.append(dd)
            if gd > 0:
                ratios.append(dd / gd)
    ratios = np.array(ratios)
    print(f"GT durations   : mean={np.mean(gtd):.1f}s median={np.median(gtd):.1f}s "
          f"p10={np.percentile(gtd,10):.1f} p90={np.percentile(gtd,90):.1f}")
    print(f"DET durations  : mean={np.mean(pdd):.1f}s median={np.median(pdd):.1f}s "
          f"p10={np.percentile(pdd,10):.1f} p90={np.percentile(pdd,90):.1f}")
    print(f"det/gt duration ratio: mean={ratios.mean():.3f} median={np.median(ratios):.3f} "
          f"frac<1 (too short)={np.mean(ratios<1):.3f}  frac<0.5={np.mean(ratios<0.5):.3f}")

    # ---------- Part 2: count/order-frozen temporal dilation ----------
    def dilate(dets, k, mode='center', cap=True):
        out = {}
        for v, j in dets.items():
            dur = j['duration']
            aa = []
            for a in j['anomalies']:
                s, e = a['t_start'], a['t_end']
                d = e - s
                if mode == 'center':
                    c = 0.5 * (s + e)
                    ns, ne = c - k * d / 2, c + k * d / 2
                elif mode == 'right':
                    ns, ne = s, s + k * d
                elif mode == 'left':
                    ns, ne = e - k * d, e
                if cap:
                    ns, ne = max(0.0, ns), min(dur, ne)
                aa.append(dict(a, t_start=float(ns), t_end=float(ne)))
            out[v] = dict(j, anomalies=aa)
        return out

    print("\n--- count/order-FROZEN symmetric dilation (v124 detections) ---")
    base = mp(gt, dets)
    print(f"k=1.00 (baseline) mAP={base['mAP']:.5f}  "
          + '  '.join(f"AP@{t}={base[f'AP@{t}']:.5f}" for t in THRESH))
    tab = {}
    for k in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        r = mp(gt, dilate(dets, k))
        tab[f'center k={k}'] = r['mAP']
        print(f"k={k:<4} center    mAP={r['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
              + f"  delta={r['mAP']-base['mAP']:+.5f}")
    for mode in ['right', 'left']:
        for k in [1.5, 2.0, 3.0]:
            r = mp(gt, dilate(dets, k, mode))
            tab[f'{mode} k={k}'] = r['mAP']
            print(f"k={k:<4} {mode:<9} mAP={r['mAP']:.5f}  "
                  + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
                  + f"  delta={r['mAP']-base['mAP']:+.5f}")

    # additive-pad variant (fixed seconds, not proportional)
    def pad(dets, sec):
        out = {}
        for v, j in dets.items():
            dur = j['duration']
            aa = [dict(a, t_start=max(0.0, a['t_start'] - sec),
                       t_end=min(dur, a['t_end'] + sec)) for a in j['anomalies']]
            out[v] = dict(j, anomalies=aa)
        return out
    print()
    for sec in [1.0, 2.0, 3.0, 5.0, 8.0]:
        r = mp(gt, pad(dets, sec))
        tab[f'pad {sec}s'] = r['mAP']
        print(f"pad +-{sec:<4}s      mAP={r['mAP']:.5f}  "
              + '  '.join(f"AP@{t}={r[f'AP@{t}']:.5f}" for t in THRESH)
              + f"  delta={r['mAP']-base['mAP']:+.5f}")

    # ---------- Part 3: replication on other saved detection sets ----------
    print("\n--- replication of the best dilation on OTHER saved detection sets ---")
    for dname in ['results/batch_eval_v114_full', 'results/batch_eval_v131_full',
                  'results/batch_eval_v132_full', 'results/batch_eval_v130_full',
                  'results/batch_eval_v108_full']:
        d2 = load(dname, gt_all)
        if not d2:
            continue
        g2 = {v: gt_all[v] for v in d2}
        b2 = mp(g2, d2)
        line = f"  {Path(dname).name:<26} base={b2['mAP']:.5f}"
        for k in [1.5, 2.0, 2.5]:
            r2 = mp(g2, dilate(d2, k))
            line += f"   k={k}:{r2['mAP']:.5f}({r2['mAP']-b2['mAP']:+.5f})"
        print(line)

    json.dump(tab, open(ROOT / 'video_anomaly_detection/iteration_logs/'
                        'analysis_v138_duration_probe.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
