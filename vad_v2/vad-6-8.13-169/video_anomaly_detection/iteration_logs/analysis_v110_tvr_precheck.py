"""
v110 offline pre-check — TVR: per-detection VideoMAE-over-support confidence re-rank.

Targets the v109 root cause (intra-video mislocation): re-rank each detection by
the TRAINED VideoMAE sliding-window score sampled over its [t_s,t_e] support.
Per-detection (not per-video), so it has a lever on the 34/65 multi-detection videos
AND the global cross-video sort. Faithful offline: re-ranking cannot change
detections/boundaries/count, so re-scoring v108's real detections == v110 output.

conf' = conf * (1 - a + a * mae_norm),  mae_norm = min-max(mae_over_support) within video.

Usage: python3.14 analysis_v110_tvr_precheck.py [--limit N]
"""
import sys, json, glob, shutil, argparse
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa
from video_preprocessor import VideoPreprocessor  # noqa
from videomae_classifier import VideoMAEClassifier  # noqa

V108_DIR = ROOT / 'results/batch_eval_v108_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v110_mae_cache.json'


def compute_mae_signals(limit=None):
    """Per video: window center-times + scores. Cached to disk."""
    if CACHE.exists():
        return json.load(open(CACHE))
    pre = VideoPreprocessor(fps=2, segment_duration=6.0, motion_threshold=1.0, min_motion_interval=2.0)
    mae = VideoMAEClassifier(device='mps')
    out = {}
    files = sorted(glob.glob(str(V108_DIR / '*.json')))
    if limit:
        files = files[:limit]
    for i, jf in enumerate(files):
        name = Path(jf).stem
        d = json.load(open(jf))
        vp = d.get('video_path')
        if not vp:
            out[name] = None
            continue
        try:
            vd = pre.process(vp)
            frames = vd['all_frames']; ts = vd['timestamps']
            centers, scores = mae.sliding_window_scores(frames, window_size=32, stride=16)
            center_times = [float(ts[c]) if c < len(ts) else float(ts[-1]) for c in centers]
            out[name] = {'center_times': center_times, 'scores': [float(s) for s in scores]}
            print(f'[{i+1}/{len(files)}] {name}: {len(scores)} windows, score range '
                  f'[{min(scores):.3f},{max(scores):.3f}]', flush=True)
        except Exception as e:
            print(f'[{i+1}/{len(files)}] {name}: FAILED {e}', flush=True)
            out[name] = None
        if (i + 1) % 10 == 0:
            json.dump(out, open(CACHE, 'w'))
    json.dump(out, open(CACHE, 'w'))
    return out


def mae_over_support(sig, t_s, t_e):
    if not sig or not sig['scores']:
        return None
    ct = np.array(sig['center_times']); sc = np.array(sig['scores'])
    m = (ct >= t_s) & (ct <= t_e)
    if m.any():
        return float(sc[m].mean())
    # nearest window center to the detection midpoint
    mid = 0.5 * (t_s + t_e)
    return float(sc[np.argmin(np.abs(ct - mid))])


def build_reranked(out_dir, signals, alpha):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V108_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        sig = signals.get(name)
        anoms = d.get('anomalies', [])
        if sig and sig.get('scores') and anoms:
            lo, hi = min(sig['scores']), max(sig['scores'])
            rng = hi - lo
            for a in anoms:
                raw = mae_over_support(sig, a['t_start'], a['t_end'])
                mae_norm = (raw - lo) / rng if (raw is not None and rng > 1e-6) else 0.5
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * (1 - alpha + alpha * mae_norm))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    signals = compute_mae_signals(args.limit)
    base = run_eval(V108_DIR)
    print(f"\n[SANITY] v108 baseline mAP={base['mAP']:.5f}")
    print(f"\n{'alpha':<8} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  delta")
    print(f"{'baseline':<8} {base['mAP']:>9.5f} {base['AP@0.3']:>9.5f} {base['AP@0.5']:>9.5f} {base['AP@0.7']:>9.5f}")
    res = {'baseline': base['mAP']}
    for a in [0.15, 0.2, 0.3, 0.5]:
        outd = ROOT / f'results/batch_eval_v110_precheck_a{a}'
        build_reranked(outd, signals, a)
        tm = run_eval(outd)
        res[f'a{a}'] = tm['mAP']
        print(f"{a:<8} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f}  {tm['mAP']-base['mAP']:+.5f}")
    json.dump(res, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v110_tvr_precheck.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
