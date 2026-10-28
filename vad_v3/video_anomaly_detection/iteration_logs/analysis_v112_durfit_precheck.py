"""
v112 offline pre-check — stack duration-prior fit onto the v111 Q-score re-rank.

Baseline for THIS loop = v111 QSCORE (3-component, base 0.5) = 0.16703, NOT raw v108.
Tests whether an INDEPENDENT signal (category duration-prior fit) stacks a further gain.

duration_fit(det) in [0,1]: 1 if det_dur within the category's expected [lo,hi],
else linear decay (dur/lo if too short, hi/dur if too long).

Variants:
  A. Q4 = mean(sharp, contrast, center, dur_fit)   -> 4-component averaged
  B. Q3 * dur_multiplier  (dur_fit as a separate gentle multiplier on top of v111)
Faithful/deterministic: re-scores v108's real detections; sanity vs v111 0.16703.
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
sys.path.insert(0, str(ROOT / 'video_anomaly_detection/iteration_logs'))
from evaluate import evaluate  # noqa
from pipeline import CATEGORY_DURATION_PRIORS as PRIORS  # noqa
import analysis_v111_qscore_precheck as Q  # reuse qscore_components + run_eval

V108_DIR = ROOT / 'results/batch_eval_v108_full'


def dur_fit(cat, dur):
    p = PRIORS.get(cat)
    if not p:
        return 1.0
    lo, hi = p['expected_duration']
    if dur < lo:
        return float(np.clip(dur / max(lo, 1e-6), 0, 1))
    if dur > hi:
        return float(np.clip(hi / max(dur, 1e-6), 0, 1))
    return 1.0


def build(out_dir, mode, base=0.5, dur_alpha=0.3):
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
                df = dur_fit(a.get('anomaly_type', ''), a['t_end'] - a['t_start'])
                if mode == 'Q4':
                    comps.append((sh + co + ce + df) / 4.0)
                else:  # Q3
                    comps.append((sh + co + ce) / 3.0)
            comps = np.array(comps)
            if len(comps) > 1 and comps.max() - comps.min() > 1e-9:
                qn = (comps - comps.min()) / (comps.max() - comps.min())
            else:
                qn = np.ones_like(comps)
            for a, q in zip(anoms, qn):
                mult = base + (1 - base) * float(q)
                if mode == 'Q3xDur':
                    df = dur_fit(a.get('anomaly_type', ''), a['t_end'] - a['t_start'])
                    mult *= (1 - dur_alpha + dur_alpha * df)
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * mult)
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def main():
    base = Q.run_eval(V108_DIR)
    # v111 anchor (Q3, base 0.5)
    d111 = ROOT / 'results/batch_eval_v112_anchor_q3'
    build(d111, 'Q3', 0.5)
    tm111 = Q.run_eval(d111)
    print(f"[SANITY] raw v108 mAP={base['mAP']:.5f}  v111 Q3 anchor mAP={tm111['mAP']:.5f} (expect ~0.16703)")
    print(f"\n{'config':<22} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  vs-v111")
    print(f"{'v111 Q3 (base)':<22} {tm111['mAP']:>9.5f} {tm111['AP@0.3']:>9.5f} {tm111['AP@0.5']:>9.5f} {tm111['AP@0.7']:>9.5f}   ----")
    res = {'raw': base['mAP'], 'v111_Q3': tm111['mAP']}
    trials = {
        'Q4 base0.5': ('Q4', 0.5, 0.0),
        'Q4 base0.3': ('Q4', 0.3, 0.0),
        'Q3xDur a0.3': ('Q3xDur', 0.5, 0.3),
        'Q3xDur a0.5': ('Q3xDur', 0.5, 0.5),
        'Q3xDur a0.2': ('Q3xDur', 0.5, 0.2),
    }
    for name, (mode, b, da) in trials.items():
        outd = ROOT / f"results/batch_eval_v112_{name.replace(' ', '_').replace('.', '')}"
        build(outd, mode, b, da)
        tm = Q.run_eval(outd)
        res[name] = tm['mAP']
        print(f"{name:<22} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f}   {tm['mAP']-tm111['mAP']:+.5f}")
    json.dump(res, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v112_durfit_precheck.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
