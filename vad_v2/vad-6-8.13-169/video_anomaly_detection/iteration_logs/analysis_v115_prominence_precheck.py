"""v115 offline A/B: topographic peak-prominence re-rank stacked on committed Q-score.

v114 cached detections ALREADY have Q-score applied live, so baseline = as-is.
Treatment = conf * (base + (1-base)*prom_n), prom normalized within-video.
Sanity: base=1.0 must reproduce baseline exactly.
Also reports within-video raw prominence spread (kill-gate for zero-variance trap).
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np
from scipy.signal import peak_prominences

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa

V_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'


def prom_raw(fs, ts, t_s, t_e):
    fs = np.asarray(fs, dtype=float); ts = np.asarray(ts, dtype=float)
    if len(fs) < 3:
        return 0.0
    in_mask = (ts >= t_s) & (ts <= t_e)
    if in_mask.sum() < 1:
        return 0.0
    peak_idx = int(np.where(in_mask)[0][np.argmax(fs[in_mask])])
    vid_range = fs.max() - fs.min() + 1e-9
    try:
        p = peak_prominences(fs, [peak_idx])[0][0]
    except Exception:
        p = 0.0
    return float(np.clip(p / vid_range, 0, 1))


def build(out_dir, base, apply=True):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    spreads = []
    for jf in glob.glob(str(V_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if apply and fs and ts and anoms:
            praw = np.array([prom_raw(fs, ts, a['t_start'], a['t_end']) for a in anoms])
            if len(praw) > 1:
                spreads.append((name, float(praw.min()), float(praw.max()), float(praw.max()-praw.min())))
            if len(praw) > 1 and praw.max()-praw.min() > 1e-9:
                pn = (praw - praw.min())/(praw.max()-praw.min())
            else:
                pn = np.ones_like(praw)
            for a, p in zip(anoms, pn):
                a['confidence'] = min(1.0, a.get('confidence', 0.0)*(base + (1-base)*float(p)))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))
    return spreads


def rune(rd):
    return evaluate(LABEL, str(rd), fps=30, split='test')['temporal_map']


def main():
    b = rune(V_DIR)
    print(f"[SANITY as-is baseline] mAP={b['mAP']:.5f} @0.3={b['AP@0.3']:.5f} @0.5={b['AP@0.5']:.5f} @0.7={b['AP@0.7']:.5f}")
    # sanity base=1.0 -> identical
    o1 = ROOT/'results/v115_base1'; build(o1, 1.0); t1 = rune(o1)
    print(f"[SANITY base=1.0]        mAP={t1['mAP']:.5f}  (must equal baseline)")
    spreads = None
    for base in [0.5, 0.3, 0.7]:
        od = ROOT/f'results/v115_prom_b{int(base*10)}'
        sp = build(od, base)
        if base == 0.5: spreads = sp
        t = rune(od)
        print(f"[PROM base={base}]  mAP={t['mAP']:.5f} @0.3={t['AP@0.3']:.5f} @0.5={t['AP@0.5']:.5f} @0.7={t['AP@0.7']:.5f}  delta={t['mAP']-b['mAP']:+.5f}")
    # spread report
    nz = [s for s in spreads if s[3] > 1e-6]
    print(f"\n[KILL-GATE] multi-detection videos={len(spreads)}  with-nonzero-spread={len(nz)}")
    for s in sorted(spreads, key=lambda x:-x[3])[:8]:
        print(f"   {s[0]:<28} prom[{s[1]:.3f},{s[2]:.3f}] spread={s[3]:.3f}")


if __name__ == '__main__':
    main()
