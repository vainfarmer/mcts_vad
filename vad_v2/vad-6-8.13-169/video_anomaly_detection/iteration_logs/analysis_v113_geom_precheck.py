"""
v113 offline pre-check — orthogonal curve-geometry signals stacked on v111 Q-score.

Loop baseline = v111 Q3 (sharp+contrast+center, base 0.5) = 0.16703.
Candidate extra per-detection signals (all from the cached frame_scores curve):
  - oic:  inner_mean - outer_mean  (AutoLoc/LogOIC style; complements peak-based contrast)
  - prom: topographic prominence of the support peak (peak - max flanking valley)
  - asym: onset+offset sharpness (forward/backward contrast at the two boundaries)
Each tested as a 4th averaged component: Q4 = mean(sharp, contrast, center, X).
Deterministic re-score of v108 real detections; compare to 0.16703.
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
sys.path.insert(0, str(ROOT / 'video_anomaly_detection/iteration_logs'))
from evaluate import evaluate  # noqa
import analysis_v111_qscore_precheck as Q  # noqa

V108_DIR = ROOT / 'results/batch_eval_v108_full'


def extra_signal(kind, fs, ts, t_s, t_e):
    fs = np.asarray(fs, float); ts = np.asarray(ts, float)
    if len(fs) < 3:
        return 0.0
    inm = (ts >= t_s) & (ts <= t_e)
    if inm.sum() < 1:
        return 0.0
    rng = fs.max() - fs.min() + 1e-9
    out = ~inm
    if kind == 'oic':
        inner = fs[inm].mean(); outer = fs[out].mean() if out.any() else fs.min()
        return float(np.clip((inner - outer) / rng, 0, 1))
    if kind == 'prom':
        peak = fs[inm].max()
        idxs = np.where(inm)[0]
        li, ri = idxs[0], idxs[-1]
        left_valley = fs[:li].min() if li > 0 else fs.min()
        right_valley = fs[ri + 1:].min() if ri + 1 < len(fs) else fs.min()
        return float(np.clip((peak - max(left_valley, right_valley)) / rng, 0, 1))
    if kind == 'asym':
        idxs = np.where(inm)[0]
        li, ri = idxs[0], idxs[-1]
        w = max(2, (ri - li) // 4)
        onset = fs[li:li + w].mean() - fs[max(0, li - w):li].mean() if li > 0 else 0.0
        offset = fs[ri - w:ri].mean() - fs[ri:ri + w].mean() if ri + w < len(fs) else 0.0
        return float(np.clip((abs(onset) + abs(offset)) / (2 * rng), 0, 1))
    return 0.0


def build(out_dir, kind, base=0.5):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V108_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if fs and ts and anoms:
            comps = []
            for a in anoms:
                sh, co, ce = Q.qscore_components(fs, ts, a['t_start'], a['t_end'])
                if kind == 'Q3':
                    comps.append((sh + co + ce) / 3.0)
                else:
                    x = extra_signal(kind, fs, ts, a['t_start'], a['t_end'])
                    comps.append((sh + co + ce + x) / 4.0)
            comps = np.array(comps)
            if len(comps) > 1 and comps.max() - comps.min() > 1e-9:
                qn = (comps - comps.min()) / (comps.max() - comps.min())
            else:
                qn = np.ones_like(comps)
            for a, q in zip(anoms, qn):
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * (base + (1 - base) * float(q)))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def main():
    raw = Q.run_eval(V108_DIR)['mAP']
    d3 = ROOT / 'results/batch_eval_v113_anchor_q3'; build(d3, 'Q3'); tm3 = Q.run_eval(d3)
    print(f"[SANITY] raw v108={raw:.5f}  v111 Q3 anchor={tm3['mAP']:.5f} (expect 0.16703)")
    print(f"\n{'config':<16} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  vs-v111")
    print(f"{'v111 Q3':<16} {tm3['mAP']:>9.5f} {tm3['AP@0.3']:>9.5f} {tm3['AP@0.5']:>9.5f} {tm3['AP@0.7']:>9.5f}   ----")
    res = {'v111_Q3': tm3['mAP']}
    for kind in ['oic', 'prom', 'asym']:
        outd = ROOT / f'results/batch_eval_v113_{kind}'
        build(outd, kind); tm = Q.run_eval(outd); res[kind] = tm['mAP']
        print(f"{'+'+kind:<16} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f}   {tm['mAP']-tm3['mAP']:+.5f}")
    json.dump(res, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v113_geom_precheck.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
