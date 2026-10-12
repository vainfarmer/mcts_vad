"""v188 SZM — spatial-zoom motion-crop oracle precheck (OFFLINE, zero VLM, zero eval spend).

QUESTION: v185 (DSO) proved the dense full-frame CLIP category curve has NO
usable signal at the 54 pipeline-missed GT intervals (loose 8/54, strict
proposable 1/54) at NATIVE scale. UCF-Crime anomalies are often
small-in-frame. v188 tests the SPATIAL hypothesis: does a motion-bbox CROP
(zoom) make the missed events visible to the SAME CLIP encoder?
"Change WHAT the model sees spatially, not temporally."

BASELINE (not recomputed): results/v185_dso_oracle.json missed_rows carry the
full-frame peak z per missed GT (own-cat loose row). 51 videos / 54 missed
GTs. Baseline loose rate = 8/54 (14.8%).

CROP RULE (PRE-REGISTERED, deterministic): per kept frame (pipeline-exact
sampling, see below), compute abs frame-diff vs the previous KEPT frame at
64x64 grayscale (uint8, 0-255); motion mask = diff > 25; largest connected
component (cv2.connectedComponentsWithStats, connectivity=8, background
label 0 excluded) by area; require area > 64 px at 64x64 scale else FULL
FRAME. bbox = component rect mapped back to full resolution
(cell i covers [i/64*W, (i+1)/64*W)), dilated 25% of the bbox side on EACH
side (i.e. 1.5x per axis), clamped to frame; then enforce min side 96 px per
axis (expand symmetrically about the bbox center, re-clamped); if the final
bbox area > 90% of the frame area -> FULL FRAME. First kept frame has no
previous kept frame -> FULL FRAME. Every FULL-FRAME use is counted as a
fallback. The crop (or full frame on fallback) is encoded with the same
CLIP ViT-L-14/openai via the committed preprocess (as in
VideoAnomalyExpert.precompute_video_features).

CURVE: identical to v185 / FrameScorer.get_category_scores — logits =
clip_features @ text_features.T * clamp(exp(logit_scale), <=100); softmax
over 14 columns; curve_c = p_c / (p_c + p_normal + 1e-8). Text features are
the COMMITTED ones (anomaly_expert.VideoAnomalyExpert + prompt_weights.json).

COMPUTE: per missed GT, peak z_crop = (max-in-GT of the GT-own-cat CROP
curve - median) / (1.4826*MAD), MAD-scale eps 1e-8 (if scale < eps: z = inf
when peak > median else 0.0, same convention as v185). GUARD: per video,
max-z of the same crop curve over NON-GT frames (complement of ALL GT
intervals of that video, from the parquet), then p90 across the 51 videos.

GO GATE (pre-registered): GO iff
  (a) >= 40% of the 54 missed GTs reach z_crop >= 2.0 (GT-own-cat), AND
  (b) median uplift (z_crop - z_full) over the 54 >= +1.0, AND
  (c) guard: non-GT p90 max-z on crop curves < 4.0.

Resumable: per-video rows (curve, timestamps, fallback count, check) are
cached inside the single output JSON results/v188_szm_precheck.json.

Usage:
  python3.14 iteration_logs/analysis_v188_szm_precheck.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from video_preprocessor import VideoPreprocessor          # noqa: E402
from anomaly_expert import VideoAnomalyExpert             # noqa: E402

RES_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v182_ref88')
V185 = Path('/Users/zzl/PycharmProjects/vad/results/v185_dso_oracle.json')
PARQ = Path('/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet')
DATASET = Path('/Users/zzl/Movies/ucf-crime')
OUT = Path('/Users/zzl/PycharmProjects/vad/results/v188_szm_precheck.json')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]

DEV = 'mps'
GT_FPS = 30.0
Z_MIN = 2.0
MAD_EPS = 1e-8
GO_RATE = 0.40        # gate (a)
GO_UPLIFT = 1.0       # gate (b)
GO_GUARD = 4.0        # gate (c)

# pre-registered crop-rule constants
DIFF_THR = 25         # motion mask threshold on 0-255 uint8 diff
CC_MIN_AREA = 64      # min component area at 64x64 scale (strictly above)
DILATE = 0.25         # dilation fraction applied to EACH side
MIN_SIDE = 96         # min crop side in full-res pixels
MAX_AREA_FRAC = 0.90  # max crop area as fraction of frame area


# ---------------------------------------------------------------- data

def load_gt():
    df = pd.read_parquet(PARQ)
    gt = {}
    for _, r in df.iterrows():
        segs = []
        for sc, ec in [('annotation_start1', 'annotation_end1'),
                       ('annotation_start2', 'annotation_end2')]:
            if r[sc] >= 0 and r[ec] >= 0:
                segs.append((r[sc] / GT_FPS, r[ec] / GT_FPS))
        stem = Path(str(r['video_path'])).stem
        gt[stem] = {'cat': r['category'], 'segs': segs,
                    'video_path': str(r['video_path'])}
    return gt


def load_committed():
    """stem -> {timestamps, video_path, duration} from the committed run
    (validity cross-check + path resolution only; no curves recomputed)."""
    out = {}
    for f in sorted(RES_DIR.glob('*.json')):
        if f.name == 'evaluation_report.json':
            continue
        d = json.load(open(f))
        out[f.stem] = {
            'timestamps': d.get('timestamps'),
            'duration': float(d['duration']),
            'video_path': d.get('video_path'),
        }
    return out


def load_v185():
    d = json.load(open(V185))
    rows = d['missed_rows']
    by_vid = {}
    for r in rows:
        by_vid.setdefault(r['video'], []).append(r)
    return rows, by_vid


# ---------------------------------------------------------------- crop rule

def _expand_min_side(a0, a1, hi, min_len):
    """Expand [a0, a1] symmetrically about its center to >= min_len,
    clamped to [0, hi]. Deterministic."""
    if a1 - a0 >= min_len:
        return a0, a1
    c = 0.5 * (a0 + a1)
    n0, n1 = c - 0.5 * min_len, c + 0.5 * min_len
    if n0 < 0.0:
        n1 -= n0
        n0 = 0.0
    if n1 > hi:
        n0 -= (n1 - hi)
        n1 = hi
    n0 = max(0.0, n0)
    return n0, min(n1, hi)


def motion_bbox(prev_g64, g64, H, W):
    """Pre-registered crop rule. Returns (x0, y0, x1, y1) float box or None
    (None -> full-frame fallback)."""
    diff = cv2.absdiff(g64, prev_g64)                    # uint8 64x64
    mask = (diff > DIFF_THR).astype(np.uint8)
    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if int(areas[best - 1]) <= CC_MIN_AREA:
        return None
    cx, cy = int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP])
    cw, ch = int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT])
    # map 64x64 cell rect back to full resolution
    x0, x1 = cx / 64.0 * W, (cx + cw) / 64.0 * W
    y0, y1 = cy / 64.0 * H, (cy + ch) / 64.0 * H
    # dilate 25% of the bbox side on each side
    bw, bh = x1 - x0, y1 - y0
    x0 -= DILATE * bw
    x1 += DILATE * bw
    y0 -= DILATE * bh
    y1 += DILATE * bh
    # clamp to frame
    x0, x1 = max(0.0, x0), min(float(W), x1)
    y0, y1 = max(0.0, y0), min(float(H), y1)
    # enforce min side 96 px (per axis, centered, re-clamped)
    x0, x1 = _expand_min_side(x0, x1, float(W), MIN_SIDE)
    y0, y1 = _expand_min_side(y0, y1, float(H), MIN_SIDE)
    # max area 90% of frame -> full frame
    if (x1 - x0) * (y1 - y0) > MAX_AREA_FRAC * W * H:
        return None
    return x0, y0, x1, y1


def compute_crops(frames):
    """frames: (N, H, W, 3) uint8 RGB kept frames. Returns (crops list,
    n_fallback). Fallback frames use the full frame."""
    H, W = int(frames.shape[1]), int(frames.shape[2])
    crops = []
    n_fb = 0
    prev = None
    for i in range(len(frames)):
        f = frames[i]
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64),
                       interpolation=cv2.INTER_AREA)
        box = motion_bbox(prev, g, H, W) if prev is not None else None
        if box is None:
            crops.append(f)
            n_fb += 1
        else:
            x0, y0, x1, y1 = box
            xi0, yi0 = int(round(x0)), int(round(y0))
            xi1, yi1 = int(round(x1)), int(round(y1))
            xi1, yi1 = max(xi1, xi0 + 1), max(yi1, yi0 + 1)
            crops.append(f[yi0:yi1, xi0:xi1])
        prev = g
    return crops, n_fb


def encode_crops(expert, crops, batch_size=32):
    """Mirror VideoAnomalyExpert.precompute_video_features (PIL -> committed
    preprocess -> encode_image -> L2-normalize); crops have variable shapes
    so they cannot be stacked into one ndarray beforehand."""
    feats = []
    for i in range(0, len(crops), batch_size):
        chunk = crops[i:i + batch_size]
        imgs = [expert.preprocess(Image.fromarray(c)) for c in chunk]
        t = torch.stack(imgs).to(expert.device)
        with torch.no_grad():
            f = expert.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats)


# ---------------------------------------------------------------- z stats

def curve_med_mad(curve):
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med)))
    return med, mad


def z_of(peak, med, mad):
    scale = 1.4826 * mad
    if scale < MAD_EPS:
        return float('inf') if peak > med else 0.0
    return (peak - med) / scale


def gt_peak_z(curve, ts, gseg):
    med, mad = curve_med_mad(curve)
    gs, ge = gseg
    idx = np.where((ts >= gs) & (ts <= ge))[0]
    if len(idx) == 0:  # GT shorter than frame spacing: nearest frame to center
        idx = np.array([int(np.argmin(np.abs(ts - 0.5 * (gs + ge))))])
    return z_of(float(np.max(curve[idx])), med, mad)


def nongt_max_z(curve, ts, all_segs):
    med, mad = curve_med_mad(curve)
    mask = np.ones(len(ts), dtype=bool)
    for gs, ge in all_segs:
        mask &= ~((ts >= gs) & (ts <= ge))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(ts))
    return z_of(float(np.max(curve[idx])), med, mad)


# ---------------------------------------------------------------- extraction

def extract(todo, missed_by_vid, committed, gt, cache):
    pre = VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.25)
    expert = VideoAnomalyExpert(CATS, clip_model_name='ViT-L-14',
                                clip_pretrained='openai', device=DEV)
    assert expert.model is not None, 'CLIP failed to load'
    assert expert.text_features is not None
    print(f'[SZM] CLIP ViT-L-14/openai loaded on {DEV}, '
          f'text_features={tuple(expert.text_features.shape)} '
          f'(committed path, prompt_weights.json)', flush=True)

    for i, stem in enumerate(todo):
        t0 = time.time()
        info = committed.get(stem, {})
        vp = Path(info['video_path']) if info.get('video_path') else None
        if vp is None or not vp.exists():
            vp = DATASET / gt[stem]['video_path']
        try:
            vd = pre.process(str(vp))
            frames = vd['all_frames']
            ts = np.asarray(vd['timestamps'], dtype=np.float64)
            duration = float(vd['duration'])

            crops, n_fb = compute_crops(frames)
            feats = encode_crops(expert, crops)

            # FrameScorer.get_category_scores verbatim (as v185)
            with torch.no_grad():
                logit_scale = expert.model.logit_scale.exp().detach().clamp(max=100.0)
                sim = feats @ expert.text_features.T * logit_scale
                probs = torch.softmax(sim, dim=-1).cpu().numpy()  # (N, 14)
            p_norm = probs[:, -1]

            cats_here = {r['category'] for r in missed_by_vid[stem]}
            assert len(cats_here) == 1, f'{stem}: multiple cats {cats_here}'
            cat = next(iter(cats_here))
            ci = CATS.index(cat)
            p_c = probs[:, ci]
            curve = (p_c / (p_c + p_norm + 1e-8)).astype(np.float64)

            # validity cross-check: recomputed sampling vs committed run
            ev_ts = info.get('timestamps')
            chk = {'n_frames': int(len(ts)), 'duration': duration,
                   'eval_n_frames': len(ev_ts) if ev_ts else None,
                   'eval_duration': info.get('duration')}
            if ev_ts and len(ev_ts) == len(ts):
                chk['max_abs_ts_diff'] = float(np.max(np.abs(
                    np.asarray(ev_ts, dtype=np.float64) - ts)))
            else:
                chk['max_abs_ts_diff'] = None

            cache[stem] = {'check': chk, 'category': cat,
                           'curve': curve.tolist(),
                           'timestamps': ts.tolist(),
                           'n_frames': int(len(ts)),
                           'n_fallback': int(n_fb)}
            save(cache)
            print(f'[SZM] {i + 1}/{len(todo)} {stem} cat={cat} '
                  f'frames={len(ts)} fallback={n_fb}/{len(ts)} '
                  f'ts_diff={chk["max_abs_ts_diff"]} '
                  f'in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            cache[stem] = {'error': str(e)}
            save(cache)
            print(f'[SZM-ERROR] {stem}: {e}', flush=True)
            traceback.print_exc()


def save(cache):
    tmp = OUT.with_suffix('.json.tmp')
    payload = {'per_video': cache}
    if OUT.exists():
        try:
            old = json.load(open(OUT))
            for k in old:
                if k != 'per_video':
                    payload[k] = old[k]
        except Exception:
            pass
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    tmp.replace(OUT)


# ---------------------------------------------------------------- analysis

def analyze(v185_rows, gt, committed, cache):
    rows = []
    per_vid_guard = {}
    for r in v185_rows:
        stem, cat = r['video'], r['category']
        gseg = (float(r['gt'][0]), float(r['gt'][1]))
        z_full = r['own']['z']
        entry = cache.get(stem, {})
        base = {'video': stem, 'cat': cat,
                'gt': [round(gseg[0], 3), round(gseg[1], 3)],
                'gt_dur': round(gseg[1] - gseg[0], 3),
                'z_full': z_full,
                'crop_fallback_frames': entry.get('n_fallback'),
                'crop_n_frames': entry.get('n_frames')}
        if 'curve' not in entry:
            rows.append({**base, 'z_crop': None, 'z_crop_inf': False,
                         'uplift': None, 'error': True})
            continue
        curve = np.asarray(entry['curve'], dtype=np.float64)
        ts = np.asarray(entry['timestamps'], dtype=np.float64)
        zc = gt_peak_z(curve, ts, gseg)
        zc_inf = bool(not np.isfinite(zc))
        uplift = (None if zc_inf or z_full is None
                  else float(zc - z_full))
        rows.append({**base, 'z_crop': None if zc_inf else float(zc),
                     'z_crop_inf': zc_inf, 'uplift': uplift})
        if stem not in per_vid_guard:
            per_vid_guard[stem] = nongt_max_z(curve, ts, gt[stem]['segs'])

    done = [r for r in rows if not r.get('error')]
    reach = sum(1 for r in done
                if r['z_crop_inf'] or (r['z_crop'] is not None
                                       and r['z_crop'] >= Z_MIN))
    rate = reach / len(done) if done else 0.0

    upl = [r['uplift'] for r in done if r['uplift'] is not None]
    med_uplift = float(np.median(upl)) if upl else None

    gz = [z for z in per_vid_guard.values() if np.isfinite(z)]
    n_guard_inf = sum(1 for z in per_vid_guard.values() if not np.isfinite(z))
    guard_p90 = float(np.percentile(gz, 90)) if gz else None

    fb_frames = sum(c.get('n_fallback', 0) for c in cache.values()
                    if 'n_fallback' in c)
    tot_frames = sum(c.get('n_frames', 0) for c in cache.values()
                     if 'n_frames' in c)

    gate_a = {'n_ge2': reach, 'n_evaluated': len(done), 'rate': rate,
              'threshold': GO_RATE, 'pass': bool(rate >= GO_RATE)}
    gate_b = {'median_uplift': med_uplift, 'n_finite': len(upl),
              'threshold': GO_UPLIFT,
              'pass': bool(med_uplift is not None and med_uplift >= GO_UPLIFT)}
    gate_c = {'guard_p90': guard_p90, 'threshold': GO_GUARD,
              'n_videos': len(per_vid_guard), 'n_guard_inf': n_guard_inf,
              'per_video_nongt_maxz': {k: (None if not np.isfinite(v) else v)
                                       for k, v in sorted(per_vid_guard.items())},
              'pass': bool(guard_p90 is not None and guard_p90 < GO_GUARD)}
    go = bool(gate_a['pass'] and gate_b['pass'] and gate_c['pass']
              and len(done) == len(v185_rows))

    validity = {
        'ts_mismatch_videos': sorted(
            s for s, c in cache.items()
            if c.get('check') and (
                c['check'].get('max_abs_ts_diff') is None
                or c['check']['max_abs_ts_diff'] > 1e-6
                or c['check']['n_frames'] != c['check']['eval_n_frames'])),
        'errors': sorted(s for s, c in cache.items() if 'error' in c),
        'z_crop_inf_rows': [f"{r['video']}|{r['gt']}" for r in done
                            if r['z_crop_inf']],
    }

    return {
        'experiment': 'v188_szm_precheck',
        'hypothesis': 'spatial: motion-bbox crop (zoom) makes missed '
                      'small-in-frame events visible to the same CLIP encoder',
        'baseline': {'source': str(V185), 'full_frame_loose_rate': 8 / 54,
                     'full_frame_loose': 8, 'n_missed': 54},
        'crop_rule': {
            'diff': 'abs diff vs previous KEPT frame, 64x64 grayscale uint8',
            'mask': f'diff > {DIFF_THR}',
            'component': 'largest cv2.connectedComponentsWithStats '
                         f'component (connectivity=8), area > {CC_MIN_AREA} '
                         'px at 64x64 else full frame',
            'bbox': f'map to full res, dilate {DILATE:.0%} each side, clamp; '
                    f'min side {MIN_SIDE}px (centered expand, re-clamp); '
                    f'area > {MAX_AREA_FRAC:.0%} of frame -> full frame',
            'first_kept_frame': 'full frame (no previous kept frame)',
        },
        'gates': {'z_min': Z_MIN, 'a_rate_min': GO_RATE,
                  'b_median_uplift_min': GO_UPLIFT, 'c_guard_p90_max': GO_GUARD},
        'n_missed': len(v185_rows),
        'n_videos': len({r['video'] for r in v185_rows}),
        'gate_a': gate_a,
        'gate_b': gate_b,
        'gate_c': gate_c,
        'go': go,
        'fallback': {'frames_total': tot_frames, 'frames_fallback': fb_frames,
                     'rate': (fb_frames / tot_frames) if tot_frames else None,
                     'per_video': {s: [c['n_fallback'], c['n_frames']]
                                   for s, c in sorted(cache.items())
                                   if 'n_fallback' in c}},
        'rows': rows,
        'validity': validity,
    }


def main():
    v185_rows, missed_by_vid = load_v185()
    gt = load_gt()
    committed = load_committed()
    stems = sorted(missed_by_vid)
    print(f'[SZM] videos={len(stems)} missed_gts={len(v185_rows)} '
          f'(baseline full-frame loose=8/54=0.1481)', flush=True)

    cache = {}
    if OUT.exists():
        try:
            cache = json.load(open(OUT)).get('per_video', {})
        except Exception:
            cache = {}
    todo = [v for v in stems if v not in cache or 'curve' not in cache[v]]
    print(f'[SZM] cached={len(stems) - len(todo)} todo={len(todo)}', flush=True)
    if todo:
        extract(todo, missed_by_vid, committed, gt, cache)

    cache = json.load(open(OUT)).get('per_video', {}) if OUT.exists() else cache
    res = analyze(v185_rows, gt, committed, cache)
    res['per_video'] = cache

    a, b, c = res['gate_a'], res['gate_b'], res['gate_c']
    print('\n[SZM-ANALYSIS] ' + '=' * 60, flush=True)
    print(f"[SZM] gate_a: {a['n_ge2']}/{a['n_evaluated']} z_crop>=2.0 "
          f"rate={a['rate']:.4f} (need>={GO_RATE}) pass={a['pass']}", flush=True)
    print(f"[SZM] gate_b: median_uplift={b['median_uplift']} "
          f"(need>={GO_UPLIFT}) pass={b['pass']}", flush=True)
    print(f"[SZM] gate_c: guard_p90={c['guard_p90']} "
          f"(need<{GO_GUARD}) pass={c['pass']}", flush=True)
    fb = res['fallback']
    print(f"[SZM] fallback: {fb['frames_fallback']}/{fb['frames_total']} "
          f"frames full-frame ({(fb['rate'] or 0):.4f})", flush=True)
    fin = [r for r in res['rows'] if r.get('uplift') is not None]
    for r in sorted(fin, key=lambda r: -r['uplift'])[:5]:
        print(f"  top-uplift {r['video']} {r['cat']} gt={r['gt']} "
              f"z_full={r['z_full']:.3f} z_crop={r['z_crop']:.3f} "
              f"uplift=+{r['uplift']:.3f}", flush=True)
    print(f"[SZM] validity: ts_mismatch={res['validity']['ts_mismatch_videos']} "
          f"errors={res['validity']['errors']} "
          f"z_inf={len(res['validity']['z_crop_inf_rows'])}", flush=True)
    print(f"[SZM-VERDICT] go={res['go']}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[SZM] result written to {OUT}', flush=True)


if __name__ == '__main__':
    main()
