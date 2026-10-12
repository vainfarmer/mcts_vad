"""DECISIVE CONTROL for IDEA B (BUE): is the +0.0258 a LOCALIZATION signal,
or just "make windows longer"?

For every detection that BUE fires on, we record BUE's final duration D_new.
Then we build SIGNAL-FREE controls that reach the SAME duration without using
the OIC argmax location:
  C1 dur-matched symmetric dilation : keep old center, expand to D_new
  C2 dur-matched union w/ random win : union old win with a random window of
                                       the same length as BUE's OIC window
  C3 global constant duration snap   : every fired det -> centered window of the
                                       MEDIAN BUE duration (no per-det info at all)
If a control reproduces most of BUE's gain, the mechanism is a DURATION PRIOR,
not localization -> banned-family-5 (metric gaming) territory.
"""
import sys, pickle
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
import analysis_v142_bue_verify as BV

LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30
THR = [0.3, 0.5, 0.7]


def census(gt, pred):
    io = []
    for v, e in pred.items():
        for a in e['anomalies']:
            io.append(max([_temporal_iou(a, s, FPS) for s in gt[v]['segments']] or [0.0]))
    io = np.array(io) if io else np.zeros(0)
    return int((io >= .3).sum()), int((io >= .5).sum()), int((io >= .7).sum()), int((io <= 0).sum())


def build(rd):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(rd))
    gt = {k: v for k, v in gt.items() if k in pred and gt[k]['segments']}
    pred = {k: v for k, v in pred.items() if k in gt}
    return gt, pred


def apply_plan(pred, plan):
    out = {}
    for v, e in pred.items():
        na = []
        for i, a in enumerate(e['anomalies']):
            b = dict(a)
            if (v, i) in plan:
                b['t_start'], b['t_end'] = plan[(v, i)]
            na.append(b)
        o = dict(e); o['anomalies'] = na; out[v] = o
    return out


def run(rd, seeds=(0, 1, 2, 3, 4)):
    gt, pred = build(rd)
    base = compute_temporal_map(gt, pred, fps=FPS, iou_thresholds=THR)
    print(f'\n=== {rd.name} base mAP={base["mAP"]:.5f}', flush=True)

    bue, meta = {}, {}
    for v, e in pred.items():
        dur = e.get('duration') or 0.0
        for i, a in enumerate(e['anomalies']):
            cat = a.get('anomaly_type') or e.get('category') or 'unknown'
            c, t = BV.curve(v, cat)
            if c is None or t is None or len(t) != len(c):
                continue
            D = dur or float(t[-1])
            st, (ns, ne) = BV.bue_pick(c, t, a['t_start'], a['t_end'], D,
                                       alpha=1.0, gate=True, union=True)
            if st != 'fire' or (abs(ns - a['t_start']) < 1e-6 and abs(ne - a['t_end']) < 1e-6):
                continue
            # also get the raw (un-unioned) OIC window to know its length
            st2, (rs, re_) = BV.bue_pick(c, t, a['t_start'], a['t_end'], D,
                                         alpha=1.0, gate=True, union=False)
            bue[(v, i)] = (ns, ne)
            meta[(v, i)] = dict(old=(a['t_start'], a['t_end']), dur=D,
                                newlen=ne - ns,
                                oiclen=(re_ - rs) if st2 == 'fire' else (ne - ns))
    m = compute_temporal_map(gt, apply_plan(pred, bue), fps=FPS, iou_thresholds=THR)
    h = census(gt, apply_plan(pred, bue))
    print(f'  {"BUE (real OIC location)":34s} d={m["mAP"]-base["mAP"]:+.5f} '
          f'h.3={h[0]} h.5={h[1]} h.7={h[2]} zero={h[3]} n_fire={len(bue)}', flush=True)

    # C1: duration-matched symmetric dilation about the OLD centre
    p1 = {}
    for k, md in meta.items():
        o0, o1 = md['old']; ctr = 0.5 * (o0 + o1); L = md['newlen']
        s = max(0.0, ctr - L / 2); e = min(md['dur'], s + L)
        s = max(0.0, e - L)
        p1[k] = (s, e)
    m1 = compute_temporal_map(gt, apply_plan(pred, p1), fps=FPS, iou_thresholds=THR)
    h1 = census(gt, apply_plan(pred, p1))
    print(f'  {"C1 dur-matched sym dilation":34s} d={m1["mAP"]-base["mAP"]:+.5f} '
          f'h.3={h1[0]} h.5={h1[1]} h.7={h1[2]} zero={h1[3]}', flush=True)

    # C3: constant-duration snap (median BUE length), centred on old centre
    Lmed = float(np.median([md['newlen'] for md in meta.values()])) if meta else 0.0
    p3 = {}
    for k, md in meta.items():
        o0, o1 = md['old']; ctr = 0.5 * (o0 + o1)
        s = max(0.0, ctr - Lmed / 2); e = min(md['dur'], s + Lmed); s = max(0.0, e - Lmed)
        p3[k] = (s, e)
    m3 = compute_temporal_map(gt, apply_plan(pred, p3), fps=FPS, iou_thresholds=THR)
    h3 = census(gt, apply_plan(pred, p3))
    print(f'  {f"C3 const-dur snap L={Lmed:.0f}s":34s} d={m3["mAP"]-base["mAP"]:+.5f} '
          f'h.3={h3[0]} h.5={h3[1]} h.7={h3[2]} zero={h3[3]}', flush=True)

    # C2: union old window with a RANDOM window of the same length as OIC's
    ds = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        p2 = {}
        for k, md in meta.items():
            o0, o1 = md['old']; L = min(md['oiclen'], max(1.0, md['dur']))
            s = float(rng.uniform(0.0, max(1e-6, md['dur'] - L))); e = s + L
            p2[k] = (min(o0, s), max(o1, e))
        m2 = compute_temporal_map(gt, apply_plan(pred, p2), fps=FPS, iou_thresholds=THR)
        ds.append(m2['mAP'] - base['mAP'])
    print(f'  {"C2 union w/ RANDOM win (5 seeds)":34s} d={np.mean(ds):+.5f} '
          f'[{", ".join(f"{x:+.4f}" for x in ds)}]', flush=True)
    return dict(bue=m['mAP'] - base['mAP'], c1=m1['mAP'] - base['mAP'],
                c3=m3['mAP'] - base['mAP'], c2=float(np.mean(ds)))


if __name__ == '__main__':
    dirs = [Path(x) for x in (sys.argv[1:] or [str(ROOT / 'results/batch_eval_v124_full')])]
    agg = {}
    for d in dirs:
        for k, v in run(d).items():
            agg.setdefault(k, []).append(v)
    print('\n=== MEANS ===')
    for k, v in agg.items():
        print(f'  {k:6s} mean={np.mean(v):+.5f} {[round(x, 4) for x in v]}')
