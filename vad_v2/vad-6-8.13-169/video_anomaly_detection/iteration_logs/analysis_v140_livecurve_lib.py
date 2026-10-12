"""
v140 LIVE-FAITHFUL CURVE SUBSTRATE (zero API cost; local CLIP only).

WHY THIS EXISTS — the v139 closure.
Two consecutive in-vivo refutations (v138 dense-peak -0.00287, v139 pooled-peak -0.00149) came from
an offline battery that agreed 8/8 caches at every weight with two decisive placebos. The root cause
was NOT noise and NOT temporal resolution alone: the offline probe's curve came from
`features/test/<vid>.npy` (32 pre-extracted snippets from a separate extraction script), while the
pipeline's curve comes from LIVE per-frame CLIP encoding of motion-filtered, fps-sampled frames.
Same formula, different substrate ⇒ the same rule measured +0.01424 on the snippet substrate and
-0.00149 in vivo.

This module removes the substrate gap by reproducing the in-vivo pipeline EXACTLY, using the
pipeline's own classes and the same config the batch runner uses:

    VideoPreprocessor(fps=2, segment_duration=6.0, motion_threshold=1.0,
                      min_motion_interval=2.0, segment_overlap=0.25)   # PreprocessConfig defaults
      -> video_data{all_frames, timestamps, duration}
    VideoAnomalyExpert(ViT-L-14/openai, device).precompute_video_features(all_frames)
      -> clip_features                                # identical to pipeline.py:987
    FrameScorer.get_category_scores(video_data, category)
      -> the verdict-INDEPENDENT curve used by the boundary logic  # frame_scorer.py:41-61

No VLM/LLM client is constructed and no API call is possible: the curve depends only on the video,
CLIP, and the learned text embeddings. Cost is local CLIP encoding (~130 frames/video).

FIDELITY IS VALIDATED, NOT ASSUMED. `validate_against_log()` compares this module's dense argmax
against the 99 live peak times recorded in the v139 run's `[GPC] ... peak=<t>` markers (and the 93 in
v138's). A substrate that cannot reproduce those recorded peaks is rejected before any arm is scored
— which is precisely the check the v138/v139 battery never had.
"""
import sys
import re
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
VAD = ROOT / 'video_anomaly_detection'
sys.path.insert(0, str(VAD))

DATASET = Path('/Users/zzl/Movies/ucf-crime')
CURVE_CACHE = Path(__file__).resolve().parent / 'analysis_v140_livecurves'

PEAK_LINE = re.compile(r'\[GPC\] vid=(?P<vid>\S+) t=(?P<ts>[-\d.]+)-(?P<te>[-\d.]+) '
                       r'cat=(?P<cat>\S+) peak=(?P<peak>[-\d.]+)')


@lru_cache(maxsize=1)
def _components(device='cpu'):
    """Build the pipeline's own preprocessor / expert / frame-scorer, no LLM client."""
    from config import VADConfig
    from video_preprocessor import VideoPreprocessor
    from anomaly_expert import VideoAnomalyExpert
    from frame_scorer import FrameScorer
    from run import UCF_CRIME_CATEGORIES

    cfg = VADConfig()
    cfg.anomaly_categories = list(UCF_CRIME_CATEGORIES)
    cfg.device = device
    pp = cfg.preprocess
    pre = VideoPreprocessor(fps=pp.fps, segment_duration=pp.segment_duration,
                            motion_threshold=pp.motion_threshold,
                            min_motion_interval=pp.min_motion_interval,
                            segment_overlap=pp.segment_overlap)
    exp = VideoAnomalyExpert(anomaly_categories=cfg.anomaly_categories,
                             clip_model_name=cfg.clip.model_name,
                             clip_pretrained=cfg.clip.pretrained,
                             device=device)
    fs = FrameScorer(exp, cfg, cfg.anomaly_categories)
    return pre, exp, fs


def find_video(video_id):
    hits = list((DATASET / 'videos').rglob(f'{video_id}.mp4'))
    if not hits:
        hits = list(DATASET.rglob(f'{video_id}.mp4'))
    return hits[0] if hits else None


def live_video_data(video_id, device='cpu'):
    """Reproduce the pipeline's video_data + clip_features for one video (cached on disk)."""
    CURVE_CACHE.mkdir(exist_ok=True)
    cache = CURVE_CACHE / f'{video_id}.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return dict(timestamps=z['timestamps'], clip_features=z['clip_features'],
                    duration=float(z['duration'][0]))
    path = find_video(video_id)
    if path is None:
        return None
    pre, exp, _ = _components(device)
    vd = pre.process(str(path))
    feats = exp.precompute_video_features(vd['all_frames'])
    if feats is None:
        return None
    f = feats.detach().cpu().numpy() if hasattr(feats, 'detach') else np.asarray(feats)
    ts = np.asarray(vd['timestamps'], dtype=float)
    np.savez_compressed(cache, timestamps=ts, clip_features=f.astype(np.float32),
                        duration=np.array([float(vd['duration'])]))
    return dict(timestamps=ts, clip_features=f, duration=float(vd['duration']))


def live_curve(video_id, category, device='cpu'):
    """The verdict-independent category curve, byte-faithful to frame_scorer.get_category_scores."""
    import torch
    d = live_video_data(video_id, device)
    if d is None or not category:
        return None, None
    _, exp, fs = _components(device)
    if category not in fs.anomaly_categories:
        return None, None
    vd = {'timestamps': d['timestamps'],
          'clip_features': torch.from_numpy(np.asarray(d['clip_features'])).to(
              exp.text_features.device, dtype=exp.text_features.dtype),
          'duration': d['duration']}
    curve = fs.get_category_scores(vd, category)
    if curve is None:
        return None, None
    return np.asarray(curve, dtype=float), d['timestamps']


def pooled_peak_time(curve, timestamps, bins=0):
    """Peak time; bins>0 average-pools into `bins` bins first (v139 semantics)."""
    c, t = np.asarray(curve, float), np.asarray(timestamps, float)
    if bins and bins > 0 and len(c) > bins:
        idx = np.array_split(np.arange(len(c)), bins)
        return float(np.array([t[i].mean() for i in idx])[
            int(np.argmax(np.array([c[i].mean() for i in idx])))])
    return float(t[int(np.argmax(c))])


def validate_against_log(log_path, tol=1.0, limit=None, device='cpu'):
    """Reject-or-accept gate: reproduce the live peak times recorded in a [GPC] run log."""
    recs = {}
    for line in Path(log_path).read_text(errors='replace').splitlines():
        m = PEAK_LINE.search(line)
        if m:
            g = m.groupdict()
            recs[(g['vid'], g['cat'])] = float(g['peak'])
    items = sorted(recs.items())
    if limit:
        items = items[:limit]
    ok = bad = miss = 0
    diffs = []
    for (vid, cat), peak in items:
        c, t = live_curve(vid, cat, device)
        if c is None:
            miss += 1
            continue
        got = pooled_peak_time(c, t, bins=0)
        diffs.append(abs(got - peak))
        if abs(got - peak) <= tol:
            ok += 1
        else:
            bad += 1
            print(f'  MISMATCH {vid}/{cat}: live-log peak={peak:.2f} reproduced={got:.2f}')
    print(f'FIDELITY: {ok} reproduced within {tol}s, {bad} mismatched, {miss} unavailable '
          f'({len(items)} recorded video+category pairs)')
    if diffs:
        d = np.array(diffs)
        print(f'  |peak delta|: median={np.median(d):.3f}s mean={d.mean():.3f}s max={d.max():.3f}s')
    return ok, bad, miss


if __name__ == '__main__':
    log = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / 'results' / 'batch_eval_v139_full' / 'run.log')
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    validate_against_log(log, limit=lim)
