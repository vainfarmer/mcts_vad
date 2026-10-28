"""v18 DESIGN-SESSION offline pre-check (NO eval spend, local CLIP only).

Question: on the flat-curve hard cats (Shoplifting / RoadAccidents), does the
category-curve QUALITY diagnostic (in-GT-window z-score of the pipeline's own
cat/(cat+normal) curve) have headroom above the committed v93 learned prompt
weights, via
  (a) uniform mean ensemble,
  (b) best single prompt (oracle per video),
  (c) PMX-style per-frame max over prompts (v159 reference — measured-CLOSED
      online, reproduced here offline),
  (d) PW-TC-lite: a weakly-supervised TEMPORAL-CONTRAST refit of the 6-prompt
      weights on TRAIN-split videos only (video-level labels; top-decile
      prominence objective + Training_Normal penalty). This is the candidate
      mechanism's offline proxy.

Pipeline fidelity: frames sampled exactly like VideoPreprocessor(fps=2,
motion_threshold=1.0, min_motion_interval=2.0); CLIP ViT-L-14/openai on mps;
curve = softmax(feats @ text.T * clamp(logit_scale,<=100)) -> cat/(cat+normal).
Other category columns always use the COMMITTED learned weights so the softmax
competition matches the committed pipeline; only the target cat's column varies.

GT: test parquet annotations are frame indices at 30 fps -> seconds /30.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anomaly_expert import VideoAnomalyExpert  # noqa: E402  (prompts only)

import open_clip  # noqa: E402

ROOT = Path('/Users/zzl/Movies/ucf-crime')
CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]
WEIGHTS = json.load(open(Path(__file__).resolve().parent.parent / 'prompt_weights.json'))

TEST_VIDEOS = [
    'videos/Shoplifting/Shoplifting017_x264.mp4',
    'videos/Shoplifting/Shoplifting031_x264.mp4',
    'videos/Shoplifting/Shoplifting033_x264.mp4',
    'videos/Shoplifting/Shoplifting001_x264.mp4',
    'videos/RoadAccidents/RoadAccidents017_x264.mp4',
    'videos/RoadAccidents/RoadAccidents128_x264.mp4',
]
TRAIN_FIT = {  # video-level labels only (train split has NO frame GT)
    'Shoplifting': ['videos/Shoplifting/Shoplifting003_x264.mp4',
                    'videos/Shoplifting/Shoplifting006_x264.mp4',
                    'videos/Shoplifting/Shoplifting008_x264.mp4'],
    'RoadAccidents': ['videos/RoadAccidents/RoadAccidents027_x264.mp4',
                      'videos/RoadAccidents/RoadAccidents065_x264.mp4',
                      'videos/RoadAccidents/RoadAccidents043_x264.mp4'],
}
TRAIN_NORMAL = ['videos/Training_Normal/Normal_Videos001_x264.mp4',
                'videos/Training_Normal/Normal_Videos002_x264.mp4']

DEV = 'mps'
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai', device=DEV)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')
LOGIT_SCALE = model.logit_scale.exp().detach().clamp(max=100.0).item()


def encode_texts(texts):
    with torch.no_grad():
        f = model.encode_text(tokenizer(texts).to(DEV))
    return (f / f.norm(dim=-1, keepdim=True)).float().cpu()


# per-prompt features: {cat: (6, d)} plus 'Normal'
PROMPT_FEATS = {}
for cat in CATS:
    PROMPT_FEATS[cat] = encode_texts(VideoAnomalyExpert.CATEGORY_PROMPTS[cat])
PROMPT_FEATS['Normal'] = encode_texts(VideoAnomalyExpert.NORMAL_PROMPTS)


def cat_embedding(cat, w):
    e = (PROMPT_FEATS[cat] * torch.tensor(w).unsqueeze(-1)).sum(dim=0)
    return e / e.norm()


def build_text_matrix(target_cat, w):
    """14 x d text matrix; target cat column uses w, others committed."""
    cols = []
    for cat in CATS:
        if cat == target_cat and w is not None:
            cols.append(cat_embedding(cat, w))
        else:
            cols.append(cat_embedding(cat, WEIGHTS[cat]))
    nm = PROMPT_FEATS['Normal'].mean(dim=0)
    cols.append(nm / nm.norm())
    return torch.stack(cols, dim=0)  # (14, d)


def sample_frames(video_path, fps=2, motion_threshold=1.0, min_motion_interval=2.0):
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps
    step = max(1, int(video_fps / fps))
    frames, ts, idx = [], [], 0
    while True:
        if idx % step == 0:
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ts.append(idx / video_fps)
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()
    frames = np.array(frames)
    ts = np.array(ts)
    if motion_threshold > 0 and len(frames) > 1:
        small = np.stack([cv2.resize(f, (64, 64)) for f in frames])
        ms = np.zeros(len(frames))
        ms[1:] = np.abs(small[1:].astype(np.float32) - small[:-1].astype(np.float32)).mean(axis=(1, 2, 3))
        keep = np.zeros(len(frames), dtype=bool)
        keep[0] = True
        last = ts[0]
        for i in range(1, len(frames)):
            if ms[i] > motion_threshold or ts[i] - last >= min_motion_interval:
                keep[i] = True
                last = ts[i]
        frames, ts = frames[keep], ts[keep]
    return frames, ts, duration


_FEAT_CACHE = {}


def video_feats(vp):
    if vp in _FEAT_CACHE:
        return _FEAT_CACHE[vp]
    t0 = time.time()
    frames, ts, dur = sample_frames(ROOT / vp)
    feats = []
    from PIL import Image
    for i in range(0, len(frames), 32):
        batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i:i + 32]]).to(DEV)
        with torch.no_grad():
            f = model.encode_image(batch)
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    feats = torch.cat(feats, dim=0)
    _FEAT_CACHE[vp] = (feats, ts, dur)
    print(f'  [encode] {vp.split("/")[-1]}: {len(frames)} frames in {time.time() - t0:.1f}s', flush=True)
    return _FEAT_CACHE[vp]


def curve_for(feats, target_cat, w=None, maxpool=False):
    if maxpool:
        # PMX-style: category column = per-frame max over the 6 prompt logits;
        # other columns committed. (offline reproduction of v159 PMX)
        cols = []
        for cat in CATS:
            if cat == target_cat:
                cols.append(None)
            else:
                cols.append(cat_embedding(cat, WEIGHTS[cat]))
        nm = PROMPT_FEATS['Normal'].mean(dim=0)
        cols.append(nm / nm.norm())
        sims = feats @ torch.stack([c if c is not None else torch.zeros_like(cols[-1])
                                    for c in cols], dim=0).T * LOGIT_SCALE
        tgt_idx = CATS.index(target_cat)
        sims[:, tgt_idx] = (feats @ PROMPT_FEATS[target_cat].T).max(dim=-1).values * LOGIT_SCALE
    else:
        tm = build_text_matrix(target_cat, w)
        sims = feats @ tm.T * LOGIT_SCALE
    probs = torch.softmax(sims, dim=-1).numpy()
    ci = CATS.index(target_cat)
    cat_s, nor_s = probs[:, ci], probs[:, -1]
    return cat_s / (cat_s + nor_s + 1e-8)


def mad_z(x, curve):
    med = np.median(curve)
    mad = np.median(np.abs(curve - med)) * 1.4826 + 1e-8
    return (np.mean(x) - med) / mad


def gt_mask(ts, row):
    m = np.zeros(len(ts), dtype=bool)
    for s, e in [(row.annotation_start1, row.annotation_end1),
                 (row.annotation_start2, row.annotation_end2)]:
        if s >= 0 and e >= 0:
            m |= (ts >= s / 30.0) & (ts <= e / 30.0)
    return m


def metrics(curve, mask):
    if mask.sum() == 0:
        return None
    z_in = mad_z(curve[mask], curve)
    sep = float(curve[mask].mean() - curve[~mask].mean()) if (~mask).any() else float('nan')
    peak_in = bool(mask[int(np.argmax(curve))])
    return z_in, sep, peak_in


def prominence(curve, top_frac=0.1):
    k = max(1, int(len(curve) * top_frac))
    top = np.sort(curve)[-k:]
    return mad_z(top, curve)


def main():
    gt = pd.read_parquet(ROOT / 'data/test-00000-of-00001.parquet').set_index('video_path')

    variants = ['committed_w', 'uniform_mean', 'best_single', 'worst_single', 'maxpool_pmx']
    print('== TEST-VIDEO CURVE QUALITY (in-GT z / in-out sep / peak-in-GT) ==', flush=True)
    rows_out = []
    for vp in TEST_VIDEOS:
        cat = vp.split('/')[1]
        feats, ts, dur = video_feats(vp)
        mask = gt_mask(ts, gt.loc[vp])
        print(f'\n{vp}  dur={dur:.1f}s frames={len(ts)} gt_frames={int(mask.sum())}', flush=True)
        curves = {
            'committed_w': curve_for(feats, cat, WEIGHTS[cat]),
            'uniform_mean': curve_for(feats, cat, [1.0] * 6),
            'maxpool_pmx': curve_for(feats, cat, maxpool=True),
        }
        singles = [curve_for(feats, cat, [1.0 if i == k else 0.0 for i in range(6)])
                   for k in range(6)]
        s_metrics = [metrics(c, mask)[0] for c in singles]
        curves['best_single'] = singles[int(np.argmax(s_metrics))]
        curves['worst_single'] = singles[int(np.argmin(s_metrics))]
        for name in variants:
            m = metrics(curves[name], mask)
            rows_out.append((vp.split('/')[-1], cat, name, *m))
            print(f'  {name:14s} z_in={m[0]:+.3f} sep={m[1]:+.4f} peak_in={int(m[2])}', flush=True)
        print(f'  per-prompt z_in: {["%+.2f" % z for z in s_metrics]}  (weights={WEIGHTS[cat]})', flush=True)

    # ---- PW-TC-lite refit on TRAIN (video-level labels only) ----
    print('\n== PW-TC-LITE TRAIN REFIT (weak supervision, train split only) ==', flush=True)
    normal_prom = {}
    for vp in TRAIN_NORMAL:
        if not (ROOT / vp).exists():
            print(f'  [skip] {vp} missing', flush=True)
            continue
        feats, ts, dur = video_feats(vp)
        normal_prom[vp] = (feats, ts, dur)
    refit = {}
    for cat, vids in TRAIN_FIT.items():
        data = [video_feats(vp) for vp in vids if (ROOT / vp).exists()]
        logits = torch.tensor([0.0] * 6, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.1)
        for step in range(300):
            opt.zero_grad()
            w = torch.softmax(logits, dim=0)
            loss = 0.0
            for feats, ts, dur in data:
                e = (PROMPT_FEATS[cat].to(feats.dtype) * w.unsqueeze(-1)).sum(dim=0)
                e = e / e.norm()
                cols = [cat_embedding(c, WEIGHTS[c]) if c != cat else e for c in CATS]
                nm = PROMPT_FEATS['Normal'].mean(dim=0)
                cols.append(nm / nm.norm())
                sims = feats @ torch.stack(cols).T * LOGIT_SCALE
                probs = torch.softmax(sims, dim=-1)
                cs = probs[:, CATS.index(cat)]
                curve = cs / (cs + probs[:, -1] + 1e-8)
                k = max(1, int(len(curve) * 0.1))
                top = torch.topk(curve, k).values.mean()
                med = curve.median()
                mad = (curve - med).abs().median() * 1.4826 + 1e-8
                loss = loss - (top - med) / mad  # maximize anomaly-video prominence
            for vp, (feats, ts, dur) in normal_prom.items():
                e = (PROMPT_FEATS[cat].to(feats.dtype) * w.unsqueeze(-1)).sum(dim=0)
                e = e / e.norm()
                cols = [cat_embedding(c, WEIGHTS[c]) if c != cat else e for c in CATS]
                nm = PROMPT_FEATS['Normal'].mean(dim=0)
                cols.append(nm / nm.norm())
                sims = feats @ torch.stack(cols).T * LOGIT_SCALE
                probs = torch.softmax(sims, dim=-1)
                cs = probs[:, CATS.index(cat)]
                curve = cs / (cs + probs[:, -1] + 1e-8)
                k = max(1, int(len(curve) * 0.1))
                top = torch.topk(curve, k).values.mean()
                med = curve.median()
                mad = (curve - med).abs().median() * 1.4826 + 1e-8
                loss = loss + 0.5 * (top - med) / mad  # minimize normal-video prominence
            loss.backward()
            opt.step()
        w_final = torch.softmax(logits.detach(), dim=0).numpy()
        refit[cat] = w_final
        print(f'  {cat}: refit weights={np.round(w_final, 4).tolist()} '
              f'(committed={WEIGHTS[cat]})', flush=True)

    print('\n== REFIT -> TEST TRANSFER ==', flush=True)
    for vp in TEST_VIDEOS:
        cat = vp.split('/')[1]
        if cat not in refit:
            continue
        feats, ts, dur = video_feats(vp)
        mask = gt_mask(ts, gt.loc[vp])
        c_commit = curve_for(feats, cat, WEIGHTS[cat])
        c_refit = curve_for(feats, cat, refit[cat])
        m0 = metrics(c_commit, mask)
        m1 = metrics(c_refit, mask)
        print(f'  {vp.split("/")[-1]}: committed z_in={m0[0]:+.3f} peak={int(m0[2])} | '
              f'refit z_in={m1[0]:+.3f} peak={int(m1[2])}  (delta {m1[0] - m0[0]:+.3f})', flush=True)


if __name__ == '__main__':
    main()
