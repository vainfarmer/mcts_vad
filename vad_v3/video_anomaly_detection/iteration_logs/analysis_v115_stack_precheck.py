"""v115 STACK A/B: prominence stacked ON TOP of Q-score (the proposal's actual claim).
Cached v114 confidences are PRE-rerank raw. So:
  baseline_raw   = as-is
  Q-only         = apply Q (all-equal b0.5, reproduces committed v114=0.15632)
  prom-only      = apply prominence only
  Q+prom (STACK) = apply Q then prominence  <-- proposal's claim
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


def qcomp(fs, ts, t_s, t_e):
    fs = np.asarray(fs, float); ts = np.asarray(ts, float)
    if len(fs) < 3: return 0.0
    in_mask = (ts >= t_s) & (ts <= t_e)
    if in_mask.sum() < 1: return 0.0
    peak = fs[in_mask].max()
    peak_idx = int(np.where(in_mask)[0][np.argmax(fs[in_mask])])
    out_mask = ~in_mask
    out_mean = fs[out_mask].mean() if out_mask.any() else fs.min()
    vr = fs.max()-fs.min()+1e-9
    contrast = float(np.clip((peak-out_mean)/vr,0,1))
    grad = np.abs(np.gradient(fs))
    i_s = int(np.argmin(np.abs(ts-t_s))); i_e = int(np.argmin(np.abs(ts-t_e)))
    sharp = float(np.clip((grad[i_s]+grad[i_e])/(2*(grad.max()+1e-9)),0,1))
    ct = 0.5*(t_s+t_e); dur=max(t_e-t_s,1e-6)
    center = float(np.clip(1-2*abs(ts[peak_idx]-ct)/dur,0,1))
    return 0.34*sharp+0.33*contrast+0.33*center


def prom_raw(fs, ts, t_s, t_e):
    fs = np.asarray(fs, float); ts = np.asarray(ts, float)
    if len(fs) < 3: return 0.0
    in_mask = (ts >= t_s) & (ts <= t_e)
    if in_mask.sum() < 1: return 0.0
    peak_idx = int(np.where(in_mask)[0][np.argmax(fs[in_mask])])
    vr = fs.max()-fs.min()+1e-9
    try: p = peak_prominences(fs, [peak_idx])[0][0]
    except Exception: p = 0.0
    return float(np.clip(p/vr,0,1))


def norm(vals):
    v = np.array(vals)
    if len(v) > 1 and v.max()-v.min() > 1e-9:
        return (v-v.min())/(v.max()-v.min())
    return np.ones_like(v)


def build(out_dir, mode, base=0.5):
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V_DIR/'*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if fs and ts and anoms:
            if mode in ('q','qprom'):
                qn = norm([qcomp(fs,ts,a['t_start'],a['t_end']) for a in anoms])
                for a,q in zip(anoms,qn):
                    a['confidence'] = min(1.0, a.get('confidence',0.0)*(base+(1-base)*float(q)))
            if mode in ('prom','qprom'):
                pn = norm([prom_raw(fs,ts,a['t_start'],a['t_end']) for a in anoms])
                for a,p in zip(anoms,pn):
                    a['confidence'] = min(1.0, a.get('confidence',0.0)*(base+(1-base)*float(p)))
        json.dump(d, open(out_dir/f'{name}.json','w'))


def rune(rd): return evaluate(LABEL, str(rd), fps=30, split='test')['temporal_map']


def main():
    b = rune(V_DIR)
    print(f"raw baseline     mAP={b['mAP']:.5f} @3={b['AP@0.3']:.5f} @5={b['AP@0.5']:.5f} @7={b['AP@0.7']:.5f}")
    for mode,label in [('q','Q-only'),('prom','prom-only'),('qprom','Q+prom STACK')]:
        od = ROOT/f'results/v115_{mode}'
        build(od, mode)
        t = rune(od)
        ref = None
        print(f"{label:<16} mAP={t['mAP']:.5f} @3={t['AP@0.3']:.5f} @5={t['AP@0.5']:.5f} @7={t['AP@0.7']:.5f}")


if __name__ == '__main__':
    main()
