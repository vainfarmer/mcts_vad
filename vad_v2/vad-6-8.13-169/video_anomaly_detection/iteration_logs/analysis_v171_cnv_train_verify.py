"""v171 CNV — TRAIN-split threshold VERIFICATION (sanity, NOT fitting).

Verifies the PRINCIPLED STRUCTURAL DEFAULTS of the CNV stage-1 global-null gate
(tau_mil=0.20, tau_z=1.5, tau_mot=0.01) on a handful of TRAIN-split videos:
3 anomaly (with GT segments) + 3 normal. The stage-1 conjunct (all three
signal families negative) should ESSENTIALLY NEVER fire on anomaly videos and
is expected to fire on normal ones (that is the mechanism's premise).

NO test-split numbers are touched here. NO VLM calls: only the cached-signal
recipes (CLIP features -> 32-snippet MIL, per-category relative CLIP curve,
motion energy) replicated EXACTLY as pipeline.py computes them, and the SAME
_cnv_signal_values pure function the pipeline imports.

Usage:
  python3.14 video_anomaly_detection/iteration_logs/analysis_v171_cnv_train_verify.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))

from video_preprocessor import VideoPreprocessor          # noqa: E402
from anomaly_expert import VideoAnomalyExpert             # noqa: E402
from run import UCF_CRIME_CATEGORIES                      # noqa: E402
from pipeline import (_cnv_signal_values, CNV_TAU_MIL,    # noqa: E402
                      CNV_TAU_Z, CNV_TAU_MOT)
from mil_model import MILAnomalyScorer                    # noqa: E402

DATASET = Path('/Users/zzl/Movies/ucf-crime')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def pick_videos():
    df = pd.read_parquet(DATASET / 'data/train-00000-of-00001.parquet')
    picks = []
    # TRAIN split is weakly labeled (video-level category only; the temporal
    # annotation columns are all -1), so the category IS the label we verify
    # against. Take the first video of three diverse anomaly categories.
    anom = df[df.category != 'Normal']
    for cat in ['RoadAccidents', 'Explosion', 'Robbery']:
        sub = anom[anom.category == cat]
        if len(sub):
            r = sub.iloc[0]
            picks.append((r.video_path, r.category))
    for r in df[df.category == 'Normal'].head(3).itertuples():
        picks.append((r.video_path, r.category))
    return picks


def main():
    print(f'device={DEVICE}')
    print(f'thresholds: tau_mil={CNV_TAU_MIL} tau_z={CNV_TAU_Z} tau_mot={CNV_TAU_MOT}')
    pre = VideoPreprocessor()
    expert = VideoAnomalyExpert(anomaly_categories=UCF_CRIME_CATEGORIES, device=DEVICE)
    mil = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
    mil.load_state_dict(torch.load(str(ROOT / 'video_anomaly_detection/model_mil.pth'),
                                   map_location=DEVICE, weights_only=True))
    mil.eval().to(DEVICE)

    for vp, cat in pick_videos():
        vd = pre.process(str(DATASET / vp))
        feats = expert.precompute_video_features(vd['all_frames'])
        # MIL 32-snippet recipe — byte-for-byte the pipeline Stage-1 block.
        nf = feats.shape[0]
        ns = 32
        sl = nf // ns
        snips = []
        for si in range(ns):
            s = si * sl
            e = s + sl if si < ns - 1 else nf
            snips.append(feats[s:e].mean(dim=0))
        with torch.no_grad():
            mil_scores = mil(torch.stack(snips).unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
        # motion energy — byte-for-byte _compute_motion_energy.
        frames = vd['all_frames']
        small = np.stack([cv2.resize(f, (64, 64)) for f in frames]).astype(np.float32)
        motion = np.zeros(len(frames))
        motion[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
        # category curve — byte-for-byte frame_scorer.get_category_scores;
        # anomaly video -> its GT category; normal video -> fused max over ALL
        # anomaly categories (the most conservative any-event outlierness).
        with torch.no_grad():
            logit_scale = expert.model.logit_scale.exp().detach().clamp(max=100.0)
            sim = feats @ expert.text_features.T * logit_scale
            probs = torch.softmax(sim, dim=-1).cpu().numpy()
        normal_scores = probs[:, -1]
        if cat == 'Normal':
            rels = []
            for ci in range(len(UCF_CRIME_CATEGORIES)):
                cs = probs[:, ci]
                rels.append(cs / (cs + normal_scores + 1e-8))
            curve = np.max(np.stack(rels), axis=0)
        else:
            ci = UCF_CRIME_CATEGORIES.index(cat)
            cs = probs[:, ci]
            curve = cs / (cs + normal_scores + 1e-8)
        mil_topk, zmax, mfrac, sa, sb, sc = _cnv_signal_values(mil_scores, curve, motion)
        fires = sa and sb and sc
        print(f"{vp} cat={cat} frames={len(frames)} "
              f"mil_topk={mil_topk:.4f} curve_zmax={zmax:.3f} mot_frac={mfrac:.4f} "
              f"sig={int(sa)}{int(sb)}{int(sc)} conjunct={'FIRES' if fires else 'no'}")


if __name__ == '__main__':
    main()
