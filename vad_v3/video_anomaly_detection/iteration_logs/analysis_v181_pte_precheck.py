"""v181 PTE (Paraphrase-Table Ensemble) — TRAIN-split offline pre-check.

OFFLINE ONLY: zero VLM API, zero batch_evaluate, zero eval spend. Local CLIP only.

Question: does replacing the committed per-category text embedding (learned-weight
ensemble of the 6 hand-crafted prompts, prompt_weights.json v93) with a FROZEN
K=4 paraphrase-table ensemble (iteration_logs/v181_pte_prompts.json, uniform
mean of 4 normalized prompt embeddings, renormalized) improve category-curve
GT-window prominence on TRAIN videos?

Pipeline fidelity (same as v18/v19): frames sampled exactly like
VideoPreprocessor(fps=2, motion_threshold=1.0, min_motion_interval=2.0) —
sample_frames() below is verbatim from analysis_v18_pwtc_precheck.py; CLIP
ViT-L-14/openai on mps, fp32, batch 32, L2-normalized image feats (the
precompute_video_features recipe of analysis_v19_snh_featextract.py); curve =
softmax(feats @ text.T * clamp(logit_scale,<=100)) -> cat/(cat+normal).
For each comparison only the TARGET cat's text column varies (committed vs
PTE); all other 12 cat columns + Normal stay at the COMMITTED representation
so the softmax competition matches the committed pipeline (v18 recipe).

GT: train split has NO frame-level GT (verified 2026-08-05: 0/1610 rows of
data/train-00000-of-00001.parquet have annotation_start/end >= 0). Anomaly
train videos are therefore treated as anomalous throughout: GT interval =
FULL video (video-level label), so GT-window prominence = max-z of the whole
curve = (max(curve) - median(curve)) / (1.4826*MAD(curve) + eps) — the v18
median/MAD z recipe applied to a full-video GT window.

Metric per video+cat: prominence(curve) as above.
Normal-video inflation: for each Normal video x each hard cat, prominence of
that cat's curve (false prominence); inflation = mean(PTE - committed).

GO gate (pre-registered): mean prominence delta >= +0.3 z on >= 3/4 hard cats
AND no hard cat delta < -0.1 AND normal max-z inflation < +0.5 mean.

Sample (seed-fixed, short videos ~0.3-3 min, durations probed via cv2):
  Shooting:     003 (75.9s) 012 (62.9s) 017 (31.4s) 009 (172.2s)
  RoadAccidents:003 (49.4s) 006 (56.4s) 007 (36.7s) 008 (44.2s)
  Shoplifting:  006 (105.2s) 009 (173.4s) 013 (67.4s) 018 (71.1s)
  Explosion:    006 (64.0s) 014 (43.1s) 009 (36.7s) 015 (29.3s)
  Normal:       002 (55.4s) 008 (85.6s) 007 (32.8s) 004 (30.6s)
= 16 anomaly (4 per hard cat) + 4 Normal = 20 videos.

Per-frame CLIP features are computed ONCE per video and cached to
results/v181_pte_cache/<name>.npz (frames kept in memory only; both text
variants are scored from the same cached feats).

Usage:
  python3 iteration_logs/analysis_v181_pte_precheck.py
Output:
  /Users/zzl/PycharmProjects/vad/results/v181_pte_precheck.json
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anomaly_expert import VideoAnomalyExpert  # noqa: E402  (prompt source of truth)

import open_clip  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path('/Users/zzl/Movies/ucf-crime')
PTE_PATH = Path(__file__).resolve().parent / 'v181_pte_prompts.json'
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'prompt_weights.json'
CACHE_DIR = Path('/Users/zzl/PycharmProjects/vad/results/v181_pte_cache')
OUT_JSON = Path('/Users/zzl/PycharmProjects/vad/results/v181_pte_precheck.json')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]
HARD_CATS = ["Shooting", "RoadAccidents", "Shoplifting", "Explosion"]

SAMPLE = {
    'Shooting': ['videos/Shooting/Shooting003_x264.mp4',
                 'videos/Shooting/Shooting012_x264.mp4',
                 'videos/Shooting/Shooting017_x264.mp4',
                 'videos/Shooting/Shooting009_x264.mp4'],
    'RoadAccidents': ['videos/RoadAccidents/RoadAccidents003_x264.mp4',
                      'videos/RoadAccidents/RoadAccidents006_x264.mp4',
                      'videos/RoadAccidents/RoadAccidents007_x264.mp4',
                      'videos/RoadAccidents/RoadAccidents008_x264.mp4'],
    'Shoplifting': ['videos/Shoplifting/Shoplifting006_x264.mp4',
                    'videos/Shoplifting/Shoplifting009_x264.mp4',
                    'videos/Shoplifting/Shoplifting013_x264.mp4',
                    'videos/Shoplifting/Shoplifting018_x264.mp4'],
    'Explosion': ['videos/Explosion/Explosion006_x264.mp4',
                  'videos/Explosion/Explosion014_x264.mp4',
                  'videos/Explosion/Explosion009_x264.mp4',
                  'videos/Explosion/Explosion015_x264.mp4'],
    'Normal': ['videos/Training_Normal/Normal_Videos002_x264.mp4',
               'videos/Training_Normal/Normal_Videos008_x264.mp4',
               'videos/Training_Normal/Normal_Videos007_x264.mp4',
               'videos/Training_Normal/Normal_Videos004_x264.mp4'],
}

DEV = 'mps'
EPS = 1e-8


def sample_frames(video_path, fps=2, motion_threshold=1.0, min_motion_interval=2.0):
    """Verbatim from analysis_v18_pwtc_precheck.py (pipeline-exact sampling)."""
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


def encode_texts(model, tokenizer, texts):
    with torch.no_grad():
        f = model.encode_text(tokenizer(texts).to(DEV))
    return (f / f.norm(dim=-1, keepdim=True)).float().cpu()


def video_feats(model, preprocess, vp):
    """Per-frame CLIP feats computed ONCE per video; disk-cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (vp.split('/')[-1].replace('.mp4', '') + '.npz')
    if cache.exists():
        z = np.load(cache)
        return torch.from_numpy(z['feats']), z['ts'], float(z['duration'])
    t0 = time.time()
    frames, ts, dur = sample_frames(ROOT / vp)
    feats = []
    for i in range(0, len(frames), 32):
        batch = torch.stack([preprocess(Image.fromarray(f))
                             for f in frames[i:i + 32]]).to(DEV)
        with torch.no_grad():
            f = model.encode_image(batch)
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    feats = torch.cat(feats, dim=0)
    np.savez(cache, feats=feats.numpy(), ts=ts, duration=dur)
    print(f'  [encode] {vp.split("/")[-1]}: {len(frames)} frames '
          f'in {time.time() - t0:.1f}s', flush=True)
    return feats, ts, dur


def prominence(curve):
    """v18 median/MAD z recipe, GT window = full video (train has no frame GT)."""
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) * 1.4826 + EPS
    return (float(curve.max()) - med) / mad


def cat_curve(feats, text_matrix, cat, logit_scale):
    probs = torch.softmax(feats @ text_matrix.T * logit_scale, dim=-1).numpy()
    ci = CATS.index(cat)
    cs, ns = probs[:, ci], probs[:, -1]
    return cs / (cs + ns + EPS)


def main():
    pte = json.load(open(PTE_PATH))['categories']
    weights = json.load(open(WEIGHTS_PATH))

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=DEV)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    logit_scale = model.logit_scale.exp().detach().clamp(max=100.0).item()

    # Committed embeddings: learned-weight ensemble (prompt_weights.json), v93.
    # PTE embeddings: uniform mean of the K=4 frozen paraphrases, renormalized
    # (cleanly bypasses prompt_weights.json: len-4 != len-6 -> uniform mean).
    committed_cols, pte_cols = {}, {}
    for cat in CATS:
        pf = encode_texts(model, tokenizer, VideoAnomalyExpert.CATEGORY_PROMPTS[cat])
        w = torch.tensor(weights[cat]).unsqueeze(-1)
        e = (pf * w).sum(dim=0)
        committed_cols[cat] = e / e.norm()
        qf = encode_texts(model, tokenizer, pte[cat])
        pe = qf.mean(dim=0)
        pte_cols[cat] = pe / pe.norm()
    nf = encode_texts(model, tokenizer, VideoAnomalyExpert.NORMAL_PROMPTS)
    ne = nf.mean(dim=0)
    normal_col = ne / ne.norm()

    def text_matrix(target_cat, use_pte):
        cols = []
        for cat in CATS:
            if cat == target_cat and use_pte:
                cols.append(pte_cols[cat])
            else:
                cols.append(committed_cols[cat])
        cols.append(normal_col)
        return torch.stack(cols, dim=0)

    TM_COMMIT = {c: text_matrix(c, False) for c in HARD_CATS}
    TM_PTE = {c: text_matrix(c, True) for c in HARD_CATS}
    print('[PTE-PRECHECK] text side ready (committed vs frozen K=4 PTE)', flush=True)

    per_video = {}
    for grp, vids in SAMPLE.items():
        for vp in vids:
            feats, ts, dur = video_feats(model, preprocess, vp)
            cats_to_test = [grp] if grp != 'Normal' else HARD_CATS
            recs = {}
            for cat in cats_to_test:
                c0 = cat_curve(feats, TM_COMMIT[cat], cat, logit_scale)
                c1 = cat_curve(feats, TM_PTE[cat], cat, logit_scale)
                recs[cat] = {'committed_z': round(prominence(c0), 4),
                             'pte_z': round(prominence(c1), 4)}
            per_video[vp] = {'group': grp, 'n_frames': int(len(ts)),
                             'duration_s': round(dur, 1), 'curves': recs}
            print(f'[PTE-PRECHECK] {grp:15s} {vp.split("/")[-1]:35s} '
                  + ' '.join(f'{c}: {r["committed_z"]:+.2f}->{r["pte_z"]:+.2f}'
                             for c, r in recs.items()), flush=True)

    # Aggregate
    per_cat = {}
    for cat in HARD_CATS:
        vids = [v for v, r in per_video.items()
                if r['group'] == cat and cat in r['curves']]
        zc = np.mean([per_video[v]['curves'][cat]['committed_z'] for v in vids])
        zp = np.mean([per_video[v]['curves'][cat]['pte_z'] for v in vids])
        per_cat[cat] = {'n': len(vids), 'committed': round(float(zc), 4),
                        'pte': round(float(zp), 4), 'delta': round(float(zp - zc), 4)}

    norm_deltas = []
    for v, r in per_video.items():
        if r['group'] != 'Normal':
            continue
        for cat in HARD_CATS:
            cr = r['curves'][cat]
            norm_deltas.append(cr['pte_z'] - cr['committed_z'])
    normal_inflation = float(np.mean(norm_deltas))

    deltas = [per_cat[c]['delta'] for c in HARD_CATS]
    gate_ge03 = sum(1 for d in deltas if d >= 0.3)
    gate_nodec = all(d >= -0.1 for d in deltas)
    gate_infl = normal_inflation < 0.5
    go = bool(gate_ge03 >= 3 and gate_nodec and gate_infl)

    out = {'prompt_table': str(PTE_PATH),
           'gt_note': 'train split has no frame GT (0/1610 annotated); '
                      'GT interval = full video (video-level label)',
           'per_video': per_video, 'per_cat': per_cat,
           'normal_inflation_mean_delta_z': round(normal_inflation, 4),
           'gate': {'cats_delta>=+0.3': gate_ge03, 'no_cat_delta<-0.1': gate_nodec,
                    'normal_inflation<+0.5': gate_infl, 'go': go}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n== PER-CAT (hard cats, TRAIN) ==', flush=True)
    for cat in HARD_CATS:
        pc = per_cat[cat]
        print(f'  {cat:15s} n={pc["n"]} committed={pc["committed"]:+.3f} '
              f'pte={pc["pte"]:+.3f} delta={pc["delta"]:+.3f}', flush=True)
    print(f'  normal inflation (mean delta z over 4 normals x 4 cats): '
          f'{normal_inflation:+.4f}', flush=True)
    print(f'  GATE: >=+0.3 on {gate_ge03}/4 cats (need 3) | no delta<-0.1: '
          f'{gate_nodec} | inflation<+0.5: {gate_infl} -> GO={go}', flush=True)
    print(f'[PTE-PRECHECK] wrote {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
