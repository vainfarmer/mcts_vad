"""v19 SNH — TRAIN-split feature extraction (OFFLINE, zero VLM, zero eval spend).

Standalone extractor that reuses pipeline modules pipeline-exactly:
  - VideoPreprocessor(fps=2, motion_threshold=1.0, min_motion_interval=2.0)
    -> motion-filtered frames + timestamps   (config.py defaults)
  - VideoAnomalyExpert CLIP ViT-L-14/openai on mps -> precompute_video_features
    (batch 32, L2-normalized; fp16 only on cuda -> mps is fp32, as in pipeline)
  - text_features: 13 category ensembles (learned prompt_weights.json, v93)
    + Normal mean-ensemble -> softmax(feats @ text.T * clamp(logit_scale,<=100))
    -> 13 cat/(cat+normal) curves + Normal softmax curve (pipeline L3524-3530)
  - MILAnomalyScorer(model_mil.pth): 32 consecutive-chunk mean-pool of the KEPT
    frames' CLIP feats -> 32 snippet scores (pipeline L1609-1632 recipe)
  - motion energy: 64x64 mean-abs frame diff on the KEPT frames
    (pipeline _compute_motion_energy L3494-3503 recipe)

Features per video: video_anomaly_detection/snh_features.py (SHARED module —
the identical function is reused inside the pipeline at eval time).

Sample: TRAIN split only (data/train-00000-of-00001.parquet, 1610 videos:
800 Training_Normal + 810 anomaly). ~50 normals + 4 per anomaly category
(13 cats -> 52), seeded (19). Labels are video-level only (train split has NO
frame GT; annotation columns are -1). Output: ONE JSONL line per video:
  {video, label (1=normal iff Training_Normal else 0), category,
   top_clip_category, n_frames, duration_s, features{...16 dims...}}

Usage:
  python3.14 iteration_logs/analysis_v19_snh_featextract.py [--limit N]
Output: /Users/zzl/PycharmProjects/vad/results/snh_train_features.jsonl
Resumable: videos already present in the output JSONL are skipped.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from video_preprocessor import VideoPreprocessor          # noqa: E402
from anomaly_expert import VideoAnomalyExpert             # noqa: E402
from mil_model import MILAnomalyScorer                    # noqa: E402
from snh_features import compute_snh_features, FEATURE_NAMES  # noqa: E402

ROOT = Path('/Users/zzl/Movies/ucf-crime')
TRAIN_PARQUET = ROOT / 'data/train-00000-of-00001.parquet'
OUT_JSONL = Path('/Users/zzl/PycharmProjects/vad/results/snh_train_features.jsonl')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]

DEV = 'mps'
SEED = 19
N_NORMAL = 50
PER_CAT = 4          # 13 cats * 4 = 52 anomaly videos
NUM_SNIPPETS = 32    # pipeline MIL recipe


def sample_videos():
    df = pd.read_parquet(TRAIN_PARQUET)
    normals = df[df.category == 'Normal']['video_path'].sample(
        n=N_NORMAL, random_state=SEED).tolist()
    anomaly = []
    for cat in CATS:
        pool = df[df.category == cat]['video_path']
        anomaly += pool.sample(n=min(PER_CAT, len(pool)),
                               random_state=SEED).tolist()
    vids = sorted(normals) + sorted(anomaly)
    label = {vp: (1 if vp.split('/')[1] == 'Training_Normal' else 0)
             for vp in vids}
    cat = dict(zip(df['video_path'], df['category']))
    return vids, label, cat


def motion_energy_pipeline(frames):
    """pipeline.py L3494 _compute_motion_energy, verbatim recipe."""
    import cv2
    if len(frames) < 2:
        return np.zeros(len(frames))
    small = np.stack([cv2.resize(f, (64, 64)) for f in frames]).astype(np.float32)
    motion = np.zeros(len(frames))
    motion[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    return motion


def mil_snippet_scores(mil_model, clip_feats):
    """pipeline.py L1609-1632: 32 consecutive chunks, mean-pool, one forward."""
    n_frames = clip_feats.shape[0]
    if n_frames < NUM_SNIPPETS:
        return None
    seg_len = n_frames // NUM_SNIPPETS
    snip = []
    for si in range(NUM_SNIPPETS):
        s = si * seg_len
        e = s + seg_len if si < NUM_SNIPPETS - 1 else n_frames
        snip.append(clip_feats[s:e].mean(dim=0))
    tensor = torch.stack(snip).unsqueeze(0).to(DEV)  # (1, 32, 768)
    with torch.no_grad():
        scores = mil_model(tensor).squeeze(0).cpu().numpy()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N videos (smoke test)')
    args = ap.parse_args()

    vids, label, cat_of = sample_videos()
    if args.limit:
        vids = vids[:args.limit]

    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['video'])
                except Exception:
                    pass
    todo = [v for v in vids if v not in done]
    print(f'[SNH-EXTRACT] total={len(vids)} done={len(done)} todo={len(todo)} '
          f'(seed={SEED}, normals={N_NORMAL}, per_cat={PER_CAT})', flush=True)
    if not todo:
        print('[SNH-EXTRACT] nothing to do', flush=True)
        return

    pre = VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.5)
    expert = VideoAnomalyExpert(CATS, clip_model_name='ViT-L-14',
                                clip_pretrained='openai', device=DEV)
    assert expert.model is not None, 'CLIP failed to load'
    logit_scale = expert.model.logit_scale.exp().detach().clamp(max=100.0)

    mil_model = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3,
                                 dropout=0.6)
    mil_path = Path(__file__).resolve().parent.parent / 'model_mil.pth'
    mil_model.load_state_dict(torch.load(str(mil_path), map_location=DEV,
                                         weights_only=True))
    mil_model.eval().to(DEV)
    print(f'[SNH-EXTRACT] models loaded (CLIP ViT-L-14/openai + {mil_path.name}) '
          f'on {DEV}', flush=True)

    fout = open(OUT_JSONL, 'a', buffering=1)
    n_fail = 0
    for i, vp in enumerate(todo):
        t0 = time.time()
        try:
            vd = pre.process(str(ROOT / vp))
            frames, ts = vd['all_frames'], vd['timestamps']
            feats = expert.precompute_video_features(frames)  # (N,768) norm
            with torch.no_grad():
                sims = feats @ expert.text_features.T * logit_scale
                probs = torch.softmax(sims, dim=-1).cpu().numpy()
            normal_curve = probs[:, -1]
            cat_curves = {}
            for ci, c in enumerate(CATS):
                cs = probs[:, ci]
                cat_curves[c] = cs / (cs + normal_curve + 1e-8)
            mil = mil_snippet_scores(mil_model, feats)
            motion = motion_energy_pipeline(frames)
            features, top_cat = compute_snh_features(
                mil, cat_curves, normal_curve, motion)
            rec = {
                'video': vp,
                'label': label[vp],
                'category': cat_of[vp],
                'top_clip_category': top_cat,
                'n_frames': int(len(ts)),
                'duration_s': round(float(vd['duration']), 2),
                'mil_ok': mil is not None,
                'features': features,
            }
            fout.write(json.dumps(rec) + '\n')
            print(f'[SNH-EXTRACT] {i + 1}/{len(todo)} {vp} '
                  f'label={label[vp]} frames={len(ts)} '
                  f'top1={top_cat} mil={"ok" if mil is not None else "NA"} '
                  f'in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            n_fail += 1
            print(f'[SNH-EXTRACT-ERROR] {vp}: {e}', flush=True)
            traceback.print_exc()
    fout.close()
    print(f'[SNH-EXTRACT] DONE todo={len(todo)} failures={n_fail} '
          f'out={OUT_JSONL} feature_dims={len(FEATURE_NAMES)}', flush=True)


if __name__ == '__main__':
    main()
