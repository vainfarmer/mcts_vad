"""
v111 offline pre-check — localization-quality Q-score confidence re-rank.

Targets the v109/v110 root cause (need PER-DETECTION intra-video discrimination).
Q-score is computed from the cached fused score curve (frame_scores + timestamps
in each v108 JSON) — no model, no VLM, instant. A correctly-located detection sits
on a sharp, centered, high-contrast peak of the curve; a wrongly-located one does not.

Q = w_sharp*boundary_sharpness + w_contrast*peak_contrast + w_center*center_fit
Normalized within each video across that video's detections, then:
  conf' = conf * (base + (1-base) * Q)

Faithful offline: re-ranking cannot change detections/boundaries/count, so re-scoring
v108's real detections == v111 output. Sanity: base=1.0 reproduces v108 exactly.
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa

V108_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'


def qscore_components(fs, ts, t_s, t_e):
    """fs: frame_scores array, ts: timestamps array (seconds). Returns (sharp, contrast, center)."""
    fs = np.asarray(fs, dtype=float); ts = np.asarray(ts, dtype=float)
    if len(fs) < 3:
        return 0.0, 0.0, 0.0
    in_mask = (ts >= t_s) & (ts <= t_e)
    if in_mask.sum() < 1:
        return 0.0, 0.0, 0.0
    peak = fs[in_mask].max()
    peak_idx = np.where(in_mask)[0][np.argmax(fs[in_mask])]
    out_mask = ~in_mask
    out_mean = fs[out_mask].mean() if out_mask.any() else fs.min()
    vid_range = fs.max() - fs.min() + 1e-9
    # peak contrast: how much the support peak stands above the outside baseline
    contrast = float(np.clip((peak - out_mean) / vid_range, 0, 1))
    # boundary sharpness: gradient magnitude at the two boundary indices
    grad = np.abs(np.gradient(fs))
    i_s = int(np.argmin(np.abs(ts - t_s))); i_e = int(np.argmin(np.abs(ts - t_e)))
    sharp = float(np.clip((grad[i_s] + grad[i_e]) / (2 * (grad.max() + 1e-9)), 0, 1))
    # center fit: is the peak near the detection center
    center_t = 0.5 * (t_s + t_e); dur = max(t_e - t_s, 1e-6)
    center = float(np.clip(1 - 2 * abs(ts[peak_idx] - center_t) / dur, 0, 1))
    return sharp, contrast, center


def build_reranked(out_dir, weights, base):
    ws, wc, wcen = weights
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V108_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if fs and ts and anoms:
            qs = []
            for a in anoms:
                sh, co, ce = qscore_components(fs, ts, a['t_start'], a['t_end'])
                qs.append(ws * sh + wc * co + wcen * ce)
            qs = np.array(qs)
            # normalize within video (min-max); single-detection videos -> Q=1 (no intra lever)
            if len(qs) > 1 and qs.max() - qs.min() > 1e-9:
                qn = (qs - qs.min()) / (qs.max() - qs.min())
            else:
                qn = np.ones_like(qs)
            for a, q in zip(anoms, qn):
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * (base + (1 - base) * float(q)))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def main():
    base_tm = run_eval(V108_DIR)
    print(f"[SANITY] v108 baseline mAP={base_tm['mAP']:.5f}")
    print(f"\n{'config':<28} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  delta")
    print(f"{'baseline':<28} {base_tm['mAP']:>9.5f} {base_tm['AP@0.3']:>9.5f} {base_tm['AP@0.5']:>9.5f} {base_tm['AP@0.7']:>9.5f}")
    # (w_sharp, w_contrast, w_center), base
    configs = {
        'all-equal b0.5': ((0.34, 0.33, 0.33), 0.5),
    }
    res = {'baseline': base_tm['mAP']}
    for name, (w, base) in configs.items():
        outd = ROOT / f"results/batch_eval_v111_precheck_{name.replace(' ', '_').replace('.', '')}"
        build_reranked(outd, w, base)
        tm = run_eval(outd)
        res[name] = tm['mAP']
        print(f"{name:<28} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f}  {tm['mAP']-base_tm['mAP']:+.5f}")
    json.dump(res, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v111_qscore_precheck.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
