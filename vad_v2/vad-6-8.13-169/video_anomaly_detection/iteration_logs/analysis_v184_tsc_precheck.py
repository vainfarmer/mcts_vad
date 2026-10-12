"""v184 TSC (Temporal-Smoothed Curves) — TRAIN-split offline precheck.

OFFLINE ONLY: zero VLM API, zero batch_evaluate, zero eval spend. Local CLIP only,
cache-only (reuses results/v181_pte_cache/*.npz — no video re-encoding).

Mechanism: the pipeline's per-frame CLIP category curves (FrameScorer recipe:
softmax(feats @ text.T * clamp(exp(logit_scale),<=100)) -> cat/(cat+normal)) are
consumed RAW by BUE/CBT/frame_scores/qscore; only the FUSED frame_scores get
Gaussian smoothing today. TSC applies ONE fixed Gaussian smooth (sigma = 1.5
frames, pre-registered, no fitting, reflect padding) to each category curve
before any downstream consumer. Text side is UNCHANGED (committed prompt
ensemble — the v181 committed arm; PTE/v181 paraphrases are NOT used here).

Question: does smoothing the category curves improve GT-window prominence
(boundary-op friendliness proxy) on TRAIN hard-cat videos without inflating
normal-video false prominence?

Smoothing detail: scipy.ndimage.gaussian_filter1d(curve, sigma=1.5,
mode='reflect', truncate=4.0) if scipy is importable; otherwise a hand-rolled
normalized Gaussian kernel (radius = int(4.0*sigma + 0.5), weights
exp(-0.5*(x/sigma)^2) normalized to sum 1, np.pad mode='reflect', correlate) —
deterministic, no fitting. NOTE: sampled frames are ~2 fps with motion-gated
skips, so inter-frame spacing is non-uniform in seconds; sigma=1.5 is in
SAMPLE-INDEX units (pre-registered).

GT caveat (same as v181): train split has NO frame-level GT (0/1610 rows
annotated), so GT interval = FULL video for anomaly videos (video-level label);
prominence = (max(curve) - median(curve)) / (1.4826*MAD(curve) + eps),
eps = 1e-8. Normal videos are used only for the inflation guard: per normal
video, max-z across ALL 13 cat curves, raw vs tsc; inflation = mean(tsc - raw).

PROXY CAVEAT: prominence-peakiness is a proxy for boundary-op friendliness.
Gaussian smoothing trades peak height against noise reduction — the median/MAD
z recipe captures the net effect (max shrinks, but MAD typically shrinks too).

GO gate (pre-registered, identical shape to v181): GO iff mean prominence
delta (tsc - raw) >= +0.3 z on >= 3 of the 4 hard cats AND no hard cat delta
< -0.1 AND normal max-z inflation < +0.5.

Sample (identical 20 cached train videos as v181):
  4 each of Shooting / RoadAccidents / Shoplifting / Explosion + 4 Normal.

Usage:
  python3.14 iteration_logs/analysis_v184_tsc_precheck.py
Output:
  /Users/zzl/PycharmProjects/vad/results/v184_tsc_precheck.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anomaly_expert import VideoAnomalyExpert  # noqa: E402  (prompt source of truth)

import open_clip  # noqa: E402

try:
    from scipy.ndimage import gaussian_filter1d as _scipy_gf1d
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

CACHE_DIR = Path('/Users/zzl/PycharmProjects/vad/results/v181_pte_cache')
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'prompt_weights.json'
OUT_JSON = Path('/Users/zzl/PycharmProjects/vad/results/v184_tsc_precheck.json')

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
SIGMA = 1.5          # frames (sample-index units), pre-registered
TRUNCATE = 4.0       # scipy gaussian_filter1d default


def smooth_tsc(curve):
    """ONE fixed Gaussian smooth, sigma=1.5 frames, reflect padding, no fitting."""
    if HAVE_SCIPY:
        return _scipy_gf1d(curve, sigma=SIGMA, mode='reflect', truncate=TRUNCATE)
    # Hand-rolled fallback: normalized Gaussian kernel, radius from truncate=4.0,
    # np.pad 'reflect' == scipy mode='reflect' (mirror about edge sample).
    radius = int(TRUNCATE * SIGMA + 0.5)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / SIGMA) ** 2)
    k /= k.sum()
    padded = np.pad(curve, radius, mode='reflect')
    return np.correlate(padded, k, mode='valid')


def encode_texts(model, tokenizer, texts):
    with torch.no_grad():
        f = model.encode_text(tokenizer(texts).to(DEV))
    return (f / f.norm(dim=-1, keepdim=True)).float().cpu()


def load_cached_feats(vp):
    """Cache-only: NEVER re-encodes video (zero spend)."""
    cache = CACHE_DIR / (vp.split('/')[-1].replace('.mp4', '') + '.npz')
    if not cache.exists():
        raise FileNotFoundError(f'missing v181 cache: {cache}')
    z = np.load(cache)
    return torch.from_numpy(z['feats']), z['ts'], float(z['duration'])


def prominence(curve):
    """v18 median/MAD z recipe, GT window = full video (train has no frame GT)."""
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) * 1.4826 + EPS
    return (float(curve.max()) - med) / mad


def cat_curves_from_probs(probs):
    """All 13 cat curves: cat/(cat+normal) from the shared 14-way softmax."""
    ns = probs[:, -1]
    return {cat: probs[:, ci] / (probs[:, ci] + ns + EPS)
            for ci, cat in enumerate(CATS)}


def main():
    t_start = time.time()
    weights = json.load(open(WEIGHTS_PATH))

    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=DEV)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    logit_scale = model.logit_scale.exp().detach().clamp(max=100.0).item()

    # COMMITTED text side (v181 committed arm): learned-weight ensemble of the
    # 6 hand-crafted prompts (prompt_weights.json v93); Normal = mean of
    # NORMAL_PROMPTS, renormalized. NO PTE paraphrases.
    cols = []
    for cat in CATS:
        pf = encode_texts(model, tokenizer, VideoAnomalyExpert.CATEGORY_PROMPTS[cat])
        w = torch.tensor(weights[cat]).unsqueeze(-1)
        e = (pf * w).sum(dim=0)
        cols.append(e / e.norm())
    nf = encode_texts(model, tokenizer, VideoAnomalyExpert.NORMAL_PROMPTS)
    ne = nf.mean(dim=0)
    cols.append(ne / ne.norm())
    TM = torch.stack(cols, dim=0)  # single committed 14-column text matrix
    print(f'[TSC-PRECHECK] text side ready (committed only) | logit_scale='
          f'{logit_scale:.2f} | scipy={HAVE_SCIPY}', flush=True)

    per_video = {}
    for grp, vids in SAMPLE.items():
        for vp in vids:
            feats, ts, dur = load_cached_feats(vp)
            probs = torch.softmax(feats @ TM.T * logit_scale, dim=-1).numpy()
            curves = cat_curves_from_probs(probs)
            cats_to_test = [grp] if grp != 'Normal' else CATS
            recs = {}
            for cat in cats_to_test:
                raw = curves[cat]
                tsc = smooth_tsc(raw)
                recs[cat] = {'raw_z': round(prominence(raw), 4),
                             'tsc_z': round(prominence(tsc), 4)}
            rec = {'group': grp, 'n_frames': int(len(ts)),
                   'duration_s': round(dur, 1), 'curves': recs}
            if grp == 'Normal':
                raw_maxz = max(r['raw_z'] for r in recs.values())
                tsc_maxz = max(r['tsc_z'] for r in recs.values())
                rec['maxz_raw'] = round(raw_maxz, 4)
                rec['maxz_tsc'] = round(tsc_maxz, 4)
                rec['maxz_cat_raw'] = max(recs, key=lambda c: recs[c]['raw_z'])
                rec['maxz_cat_tsc'] = max(recs, key=lambda c: recs[c]['tsc_z'])
            per_video[vp] = rec
            if grp == 'Normal':
                print(f'[TSC-PRECHECK] {grp:15s} {vp.split("/")[-1]:35s} '
                      f'maxz13: {raw_maxz:+.2f}->{tsc_maxz:+.2f}', flush=True)
            else:
                r = recs[grp]
                print(f'[TSC-PRECHECK] {grp:15s} {vp.split("/")[-1]:35s} '
                      f'{grp}: {r["raw_z"]:+.2f}->{r["tsc_z"]:+.2f}', flush=True)

    # Aggregate per hard cat: mean prominence raw vs tsc over its 4 videos.
    per_cat = {}
    for cat in HARD_CATS:
        vids = [v for v, r in per_video.items()
                if r['group'] == cat and cat in r['curves']]
        zr = np.mean([per_video[v]['curves'][cat]['raw_z'] for v in vids])
        zt = np.mean([per_video[v]['curves'][cat]['tsc_z'] for v in vids])
        per_cat[cat] = {'n': len(vids), 'raw': round(float(zr), 4),
                        'tsc': round(float(zt), 4),
                        'delta': round(float(zt - zr), 4)}

    # Normal guard: mean over the 4 normals of (max-z across all 13 cats), tsc - raw.
    norm_infl = float(np.mean(
        [r['maxz_tsc'] - r['maxz_raw'] for r in per_video.values()
         if r['group'] == 'Normal']))

    deltas = [per_cat[c]['delta'] for c in HARD_CATS]
    gate_ge03 = sum(1 for d in deltas if d >= 0.3)
    gate_nodec = all(d >= -0.1 for d in deltas)
    gate_infl = norm_infl < 0.5
    go = bool(gate_ge03 >= 3 and gate_nodec and gate_infl)

    out = {'mechanism': 'TSC: one fixed Gaussian smooth (sigma=1.5 frames) on each '
                        'category curve before any downstream consumer; text side '
                        'unchanged (committed prompts)',
           'sigma_frames': SIGMA,
           'kernel': ('scipy.ndimage.gaussian_filter1d(sigma=1.5, mode=reflect, '
                      'truncate=4.0)' if HAVE_SCIPY else
                      'hand-rolled normalized Gaussian kernel, radius='
                      'int(4.0*1.5+0.5)=6, np.pad reflect, correlate'),
           'frame_spacing_note': 'frames ~2fps with motion-gated skips; sigma is '
                                 'in sample-index units (pre-registered)',
           'gt_note': 'train split has no frame GT (0/1610 annotated); GT interval '
                      '= full video for anomaly videos (video-level label); normals '
                      'used only for the inflation guard',
           'proxy_caveat': 'prominence-peakiness is a proxy for boundary-op '
                           'friendliness: smoothing trades peak height against '
                           'noise reduction — the median/MAD z recipe captures '
                           'the net (max shrinks, MAD typically shrinks too)',
           'per_video': per_video, 'per_cat': per_cat,
           'normal_inflation_maxz13': round(norm_infl, 4),
           'gate': {'cats_delta>=+0.3': gate_ge03,
                    'no_cat_delta<-0.1': gate_nodec,
                    'normal_inflation<+0.5': gate_infl, 'go': go},
           'elapsed_s': round(time.time() - t_start, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n== PER-CAT (hard cats, TRAIN) ==', flush=True)
    for cat in HARD_CATS:
        pc = per_cat[cat]
        print(f'  {cat:15s} n={pc["n"]} raw={pc["raw"]:+.3f} '
              f'tsc={pc["tsc"]:+.3f} delta={pc["delta"]:+.3f}', flush=True)
    print(f'  normal max-z13 inflation (mean over 4 normals): '
          f'{norm_infl:+.4f}', flush=True)
    print(f'  GATE: >=+0.3 on {gate_ge03}/4 cats (need 3) | no delta<-0.1: '
          f'{gate_nodec} | inflation<+0.5: {gate_infl} -> GO={go}', flush=True)
    print(f'[TSC-PRECHECK] wrote {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
