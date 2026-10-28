"""
v113 offline pre-check — Q-score re-ranking applied to v97 FCCS detections.

Purpose: verify that the +0.009 Q-score gain (confirmed on v108 non-FCCS detections)
transfers to v97 FCCS detections. These are orthogonal mechanisms:
  - FCCS changes WHICH candidates are generated (upstream)
  - Q-score changes HOW they are RANKED (downstream post-hoc)

This is a faithful offline simulation: re-ranking cannot change boundaries/count,
so re-scoring v97's real detections == what v113 (FCCS+Q-score) would output.

Sweeps Q-score configs (same as v111 precheck) on v97 detections.
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa

V97_DIR = ROOT / 'results/batch_eval_v97_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'


def qscore_components(fs, ts, t_s, t_e):
    """Identical to pipeline._qscore_rerank and analysis_v111_qscore_precheck.py."""
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
    contrast = float(np.clip((peak - out_mean) / vid_range, 0, 1))
    grad = np.abs(np.gradient(fs))
    i_s = int(np.argmin(np.abs(ts - t_s))); i_e = int(np.argmin(np.abs(ts - t_e)))
    sharp = float(np.clip((grad[i_s] + grad[i_e]) / (2 * (grad.max() + 1e-9)), 0, 1))
    center_t = 0.5 * (t_s + t_e); dur = max(t_e - t_s, 1e-6)
    center = float(np.clip(1 - 2 * abs(ts[peak_idx] - center_t) / dur, 0, 1))
    return sharp, contrast, center


def build_reranked(src_dir, out_dir, weights, base):
    ws, wc, wcen = weights
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    reranked_count = 0
    for jf in glob.glob(str(src_dir / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if fs and ts and anoms:
            qs = []
            for a in anoms:
                sh, co, ce = qscore_components(fs, ts, a['t_start'], a['t_end'])
                qs.append(ws * sh + wc * co + wcen * ce)
            qs = np.array(qs)
            if len(qs) > 1 and qs.max() - qs.min() > 1e-9:
                qn = (qs - qs.min()) / (qs.max() - qs.min())
                reranked_count += 1
            else:
                qn = np.ones_like(qs)
            for a, q in zip(anoms, qn):
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * (base + (1 - base) * float(q)))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))
    return reranked_count


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def main():
    base_tm = run_eval(V97_DIR)
    print(f"[SANITY] v97 FCCS baseline mAP={base_tm['mAP']:.5f} (expect 0.1615)")
    print(f"  AP@0.3={base_tm['AP@0.3']:.5f}  AP@0.5={base_tm['AP@0.5']:.5f}  AP@0.7={base_tm['AP@0.7']:.5f}")
    print()

    # Analyze Q-score variance on FCCS detections (diagnostic)
    multi_det_videos = 0; total_anoms = 0; q_ranges = []
    for jf in glob.glob(str(V97_DIR / '*.json')):
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if fs and ts and len(anoms) > 1:
            multi_det_videos += 1
            qs = []
            for a in anoms:
                sh, co, ce = qscore_components(fs, ts, a['t_start'], a['t_end'])
                qs.append((sh + co + ce) / 3.0)
            q_ranges.append(max(qs) - min(qs))
        total_anoms += len(anoms)
    print(f"[DIAGNOSTIC] Total videos with anoms: {sum(1 for f in glob.glob(str(V97_DIR / '*.json')) if json.load(open(f)).get('anomalies'))}")
    print(f"  Multi-detection videos: {multi_det_videos}")
    print(f"  Total detections: {total_anoms}")
    print(f"  Q-score range in multi-det videos: mean={np.mean(q_ranges):.4f} std={np.std(q_ranges):.4f} max={np.max(q_ranges):.4f}" if q_ranges else "  No multi-det videos!")
    print()

    configs = {
        'all-equal b0.5':       ((0.34, 0.33, 0.33), 0.5),   # best from v111
        'all-equal b0.3':       ((0.34, 0.33, 0.33), 0.3),
        'all-equal b0.7':       ((0.34, 0.33, 0.33), 0.7),
        'contrast-only b0.6':   ((0.0, 1.0, 0.0), 0.6),
        'contrast-only b0.4':   ((0.0, 1.0, 0.0), 0.4),
        'sharp+contrast b0.5':  ((0.4, 0.6, 0.0), 0.5),
        'contrast+center b0.5': ((0.0, 0.6, 0.4), 0.5),
    }

    print(f"{'config':<28} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  {'delta':>8}  reranked")
    print(f"{'v97 FCCS baseline':<28} {base_tm['mAP']:>9.5f} {base_tm['AP@0.3']:>9.5f} {base_tm['AP@0.5']:>9.5f} {base_tm['AP@0.7']:>9.5f}  {'---':>8}")
    results = {'baseline_v97_fccs': base_tm}
    for name, (w, base) in configs.items():
        outd = ROOT / f"results/batch_eval_v113_precheck_{name.replace(' ', '_').replace('.', '')}"
        n_reranked = build_reranked(V97_DIR, outd, w, base)
        tm = run_eval(outd)
        delta = tm['mAP'] - base_tm['mAP']
        results[name] = {**tm, 'delta': delta, 'reranked_videos': n_reranked}
        print(f"{name:<28} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f}  {delta:>+8.5f}  {n_reranked}")

    # Summary
    best_name = max((k for k in results if k != 'baseline_v97_fccs'), 
                    key=lambda k: results[k]['mAP'])
    best = results[best_name]
    print(f"\n[BEST] {best_name}: mAP={best['mAP']:.5f} (delta={best['delta']:+.5f})")
    print(f"  Combined FCCS+Q-score would give: mAP={best['mAP']:.4f}")
    
    json.dump(results, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v113_qscore_on_fccs_precheck.json', 'w'), indent=2, default=str)
    print(f"\nResults saved.")

if __name__ == '__main__':
    main()
