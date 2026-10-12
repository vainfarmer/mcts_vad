"""v185 DSO — dense-substrate oracle precheck (OFFLINE, zero VLM, zero eval spend).

QUESTION (last unrefuted recall pivot from the v11 audit): on hard-category
videos where the committed pipeline's candidate generation MISSED the GT
entirely, does the DENSE per-frame CLIP category curve contain a usable peak
at the missed GT? If even a GT-aware oracle finds no signal there, the
"denser localizer substrate for recall" direction is honestly closed.

DATA:
  - Committed hard-88 run: results/batch_eval_v182_ref88/*.json
    (BUE+CBT+CBT-recal ON; per-video anomalies[{t_start,t_end,...}], duration)
  - GT: /Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet
    (video_path, category, annotation_start/end1/2 in 30fps frame numbers;
     -1 = empty). Videos at /Users/zzl/Movies/ucf-crime/<video_path>.

PIPELINE-EXACT local recompute (mirrors v180 precheck / pipeline.py Stage-1):
  - VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
    min_motion_interval=2.0, segment_overlap=0.25)   (config.py defaults)
  - VideoAnomalyExpert CLIP ViT-L-14/openai on mps -> precompute_video_features
  - Category curves replicate FrameScorer.get_category_scores
    (frame_scorer.py L41): logits = clip_features @ text_features.T *
    clamp(exp(logit_scale), <=100); softmax over the 14 columns
    (13 cats + normal); curve_c = p_c / (p_c + p_normal + 1e-8).
  - Text features: COMMITTED embeddings via VideoAnomalyExpert's own
    _precompute_text_features (13 cats x 6 prompts + learned
    prompt_weights.json combination) — instantiated, not reimplemented.

COVERAGE (step 1): a GT interval is COVERED iff any committed detection has
IoU >= 0.3 with it, else MISSED.

ORACLE RULE (PRE-REGISTERED, GT-aware — the curve is chosen by the GT's own
category): a missed GT is "curve-proposable" iff the curve has a contiguous
supra-threshold region R (threshold = median(curve) + 1.0*MAD(curve),
MAD = median absolute deviation; if MAD ~ 0 fall back to median + 0.05
absolute) with IoU(R, GT) >= 0.3 AND peak z = (max-in-GT - median) /
(1.4826*MAD) >= 2.0. Secondary (looser) row: peak-z-only (z >= 2.0 anywhere
in GT). Step 3 variant: curve = max over all 13 category curves
(GT-agnostic curve choice, verdict-independent).

GO GATE (pre-registered): GO iff >= 40% of missed GTs are curve-proposable
under the GT-own-category rule AND the max-over-cats variant proposable
rate is >= 30%.

Resumable: per-video rows are cached inside the single output JSON
results/v185_dso_oracle.json (no other files are written).

Usage:
  python3.14 iteration_logs/analysis_v185_dso_oracle.py
"""

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

RES_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v182_ref88')
PARQ = Path('/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet')
DATASET = Path('/Users/zzl/Movies/ucf-crime')
OUT = Path('/Users/zzl/PycharmProjects/vad/results/v185_dso_oracle.json')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]

DEV = 'mps'
GT_FPS = 30.0
IOU_COVER = 0.3        # committed-detection coverage threshold (step 1)
REGION_IOU_MIN = 0.3   # oracle region-vs-GT IoU threshold
Z_MIN = 2.0            # oracle peak-z threshold
MAD_EPS = 1e-8         # "MAD ~ 0" guard
FALLBACK_MARGIN = 0.05  # absolute threshold fallback when MAD ~ 0
GO_OWN = 0.40
GO_MAXCAT = 0.30


# ---------------------------------------------------------------- utils

def iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uni if uni > 0 else 0.0


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


def load_dets():
    """stem -> {dets, duration, timestamps, video_path} from the committed run."""
    out = {}
    for f in sorted(RES_DIR.glob('*.json')):
        if f.name == 'evaluation_report.json':
            continue
        d = json.load(open(f))
        out[f.stem] = {
            'dets': [(a['t_start'], a['t_end']) for a in d.get('anomalies', [])],
            'duration': float(d['duration']),
            'timestamps': d.get('timestamps'),
            'video_path': d.get('video_path'),
        }
    return out


# ---------------------------------------------------------------- oracle rule

def curve_stats(curve):
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med)))
    fallback = mad < MAD_EPS
    thr = med + FALLBACK_MARGIN if fallback else med + 1.0 * mad
    return med, mad, thr, fallback


def supra_regions(curve, thr):
    """Maximal contiguous frame-index runs with curve >= thr -> [(lo, hi)]."""
    above = curve >= thr
    regions = []
    i = 0
    n = len(curve)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            regions.append((i, j))
            i = j + 1
        else:
            i += 1
    return regions


def region_time(lo, hi, ts, duration):
    r_s = float(ts[lo])
    r_e = float(ts[hi + 1]) if hi + 1 < len(ts) else float(duration)
    return r_s, r_e


def eval_curve_on_gt(curve, ts, duration, gt_seg):
    """Apply the pre-registered oracle rule to one curve vs one GT interval."""
    med, mad, thr, fallback = curve_stats(curve)
    gs, ge = gt_seg

    # peak z: max of curve over frames whose timestamp lies inside the GT
    idx = np.where((ts >= gs) & (ts <= ge))[0]
    if len(idx) == 0:  # GT shorter than frame spacing: nearest frame to center
        idx = np.array([int(np.argmin(np.abs(ts - 0.5 * (gs + ge))))])
    peak = float(np.max(curve[idx]))
    scale = 1.4826 * mad
    if scale < MAD_EPS:
        z = float('inf') if peak > med else 0.0
    else:
        z = (peak - med) / scale

    # contiguous supra-threshold regions, best IoU vs GT
    best_iou = 0.0
    best_region = None
    for lo, hi in supra_regions(curve, thr):
        rs, re = region_time(lo, hi, ts, duration)
        v = iou((rs, re), (gs, ge))
        if v > best_iou:
            best_iou = v
            best_region = (rs, re)

    return {
        'median': med, 'mad': mad, 'thr': thr, 'mad_fallback': bool(fallback),
        'peak_in_gt': peak, 'z': z if np.isfinite(z) else None,
        'z_inf': bool(not np.isfinite(z)),
        'region_iou': best_iou,
        'best_region': list(best_region) if best_region else None,
        'loose': bool(z >= Z_MIN),   # peak-z-only secondary row
        'proposable': bool(best_iou >= REGION_IOU_MIN and z >= Z_MIN),
    }


# ---------------------------------------------------------------- recompute

def extract(todo, missed_by_vid, dets, cache, gt):
    pre = VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.25)
    expert = VideoAnomalyExpert(CATS, clip_model_name='ViT-L-14',
                                clip_pretrained='openai', device=DEV)
    assert expert.model is not None, 'CLIP failed to load'
    assert expert.text_features is not None
    print(f'[DSO] CLIP ViT-L-14/openai loaded on {DEV}, '
          f'text_features={tuple(expert.text_features.shape)} '
          f'(13 cats x 6 prompts + prompt_weights.json, committed path)',
          flush=True)

    for i, stem in enumerate(todo):
        t0 = time.time()
        info = dets[stem]
        vp = Path(info['video_path']) if info['video_path'] else None
        if vp is None or not vp.exists():
            vp = DATASET / gt[stem]['video_path']
        try:
            vd = pre.process(str(vp))
            frames = vd['all_frames']
            ts = np.asarray(vd['timestamps'], dtype=np.float64)
            duration = float(vd['duration'])
            feats = expert.precompute_video_features(frames)

            # FrameScorer.get_category_scores verbatim (frame_scorer.py L41)
            with torch.no_grad():
                logit_scale = expert.model.logit_scale.exp().detach().clamp(max=100.0)
                sim = feats @ expert.text_features.T * logit_scale
                probs = torch.softmax(sim, dim=-1).cpu().numpy()  # (N, 14)
            p_norm = probs[:, -1]
            cat_curves = {}
            for ci, cat in enumerate(CATS):
                p_c = probs[:, ci]
                cat_curves[cat] = p_c / (p_c + p_norm + 1e-8)
            curve_max = np.max(np.stack([cat_curves[c] for c in CATS]), axis=0)

            # validity cross-check: recomputed sampling vs committed run
            ev_ts = info['timestamps']
            chk = {'n_frames': int(len(ts)), 'duration': duration,
                   'eval_n_frames': len(ev_ts) if ev_ts else None,
                   'eval_duration': info['duration']}
            if ev_ts and len(ev_ts) == len(ts):
                chk['max_abs_ts_diff'] = float(np.max(np.abs(
                    np.asarray(ev_ts, dtype=np.float64) - ts)))
            else:
                chk['max_abs_ts_diff'] = None

            rows = []
            for gseg in missed_by_vid[stem]:
                cat = gt[stem]['cat']
                own = eval_curve_on_gt(cat_curves[cat], ts, duration, gseg)
                mx = eval_curve_on_gt(curve_max, ts, duration, gseg)
                # which category attains the max-curve peak inside the GT
                idx = np.where((ts >= gseg[0]) & (ts <= gseg[1]))[0]
                if len(idx) == 0:
                    idx = np.array([int(np.argmin(np.abs(
                        ts - 0.5 * (gseg[0] + gseg[1]))))])
                pk_fr = idx[int(np.argmax(curve_max[idx]))]
                mx['peak_cat'] = CATS[int(np.argmax(
                    np.array([cat_curves[c][pk_fr] for c in CATS])))]
                best_det = max([iou(d, gseg) for d in info['dets']],
                               default=0.0)
                rows.append({'video': stem, 'category': cat,
                             'gt': [round(gseg[0], 3), round(gseg[1], 3)],
                             'gt_dur': round(gseg[1] - gseg[0], 3),
                             'n_dets': len(info['dets']),
                             'best_det_iou': round(best_det, 4),
                             'own': own, 'maxcat': mx})
            cache[stem] = {'check': chk, 'rows': rows}
            save(cache)
            print(f'[DSO] {i + 1}/{len(todo)} {stem} frames={len(ts)} '
                  f'dur={duration:.2f} missed_gts={len(rows)} '
                  f'prop_own={sum(r["own"]["proposable"] for r in rows)} '
                  f'prop_max={sum(r["maxcat"]["proposable"] for r in rows)} '
                  f'ts_diff={chk["max_abs_ts_diff"]} '
                  f'in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            cache[stem] = {'error': str(e), 'rows': []}
            save(cache)
            print(f'[DSO-ERROR] {stem}: {e}', flush=True)
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

def analyze(all_vids, gt, dets, cache):
    per_cat = {}
    missed_rows = []
    covered_total = 0
    for stem in all_vids:
        cat = gt[stem]['cat']
        per_cat.setdefault(cat, {'covered': 0, 'missed': 0})
        for gseg in gt[stem]['segs']:
            best = max([iou(d, gseg) for d in dets[stem]['dets']],
                       default=0.0)
            if best >= IOU_COVER:
                per_cat[cat]['covered'] += 1
                covered_total += 1
            else:
                per_cat[cat]['missed'] += 1
                rows = cache.get(stem, {}).get('rows', [])
                row = next((r for r in rows
                            if abs(r['gt'][0] - gseg[0]) < 0.05
                            and abs(r['gt'][1] - gseg[1]) < 0.05), None)
                if row is None:
                    row = {'video': stem, 'category': cat,
                           'gt': [round(gseg[0], 3), round(gseg[1], 3)],
                           'gt_dur': round(gseg[1] - gseg[0], 3),
                           'n_dets': len(dets[stem]['dets']),
                           'best_det_iou': round(best, 4),
                           'own': None, 'maxcat': None, 'error': True}
                missed_rows.append(row)

    done = [r for r in missed_rows if r.get('own') is not None]
    n_missed = len(missed_rows)

    def rate(rows, key, field):
        if not rows:
            return 0.0
        return sum(1 for r in rows if r[key][field]) / len(rows)

    res = {
        'n_videos': len(all_vids),
        'per_category': per_cat,
        'covered_total': covered_total,
        'missed_total': n_missed,
        'missed_videos': sorted({r['video'] for r in missed_rows}),
        'n_missed_evaluated': len(done),
        'own_cat': {
            'proposable': sum(1 for r in done if r['own']['proposable']),
            'rate': rate(done, 'own', 'proposable'),
            'loose_z_only': sum(1 for r in done if r['own']['loose']),
            'loose_rate': rate(done, 'own', 'loose'),
        },
        'max_over_cats': {
            'proposable': sum(1 for r in done if r['maxcat']['proposable']),
            'rate': rate(done, 'maxcat', 'proposable'),
            'loose_z_only': sum(1 for r in done if r['maxcat']['loose']),
            'loose_rate': rate(done, 'maxcat', 'loose'),
        },
        'missed_rows': missed_rows,
        'validity': {
            'ts_mismatch_videos': sorted(
                s for s, c in cache.items()
                if c.get('check') and (
                    c['check'].get('max_abs_ts_diff') is None
                    or c['check']['max_abs_ts_diff'] > 1e-6
                    or c['check']['n_frames'] != c['check']['eval_n_frames'])),
            'errors': sorted(s for s, c in cache.items() if 'error' in c),
        },
    }
    res['go'] = bool(res['own_cat']['rate'] >= GO_OWN
                     and res['max_over_cats']['rate'] >= GO_MAXCAT
                     and len(done) == n_missed)
    return res


def main():
    gt = load_gt()
    dets = load_dets()
    all_vids = sorted(dets.keys())
    print(f'[DSO] videos={len(all_vids)}', flush=True)

    # step 1: coverage, pick videos with >= 1 MISSED GT
    missed_by_vid = {}
    for stem in all_vids:
        miss = []
        for gseg in gt[stem]['segs']:
            best = max([iou(d, gseg) for d in dets[stem]['dets']],
                       default=0.0)
            if best < IOU_COVER:
                miss.append(gseg)
        if miss:
            missed_by_vid[stem] = miss
    print(f'[DSO] videos_with_missed_gt={len(missed_by_vid)} '
          f'missed_gts={sum(len(v) for v in missed_by_vid.values())}',
          flush=True)

    cache = {}
    if OUT.exists():
        try:
            cache = json.load(open(OUT)).get('per_video', {})
        except Exception:
            cache = {}
    todo = [v for v in sorted(missed_by_vid) if v not in cache]
    print(f'[DSO] cached={len(cache)} todo={len(todo)}', flush=True)
    if todo:
        extract(todo, missed_by_vid, dets, cache, gt)

    cache = json.load(open(OUT)).get('per_video', {}) if OUT.exists() else cache
    res = analyze(all_vids, gt, dets, cache)
    res['per_video'] = cache

    print('\n[DSO-ANALYSIS] ' + '=' * 60, flush=True)
    for cat, c in sorted(res['per_category'].items()):
        print(f"  {cat}: covered={c['covered']} missed={c['missed']}",
              flush=True)
    print(f"[DSO] missed_total={res['missed_total']} "
          f"evaluated={res['n_missed_evaluated']}", flush=True)
    print(f"[DSO] own_cat proposable={res['own_cat']['proposable']}/"
          f"{res['n_missed_evaluated']} rate={res['own_cat']['rate']:.4f} "
          f"loose={res['own_cat']['loose_z_only']} "
          f"({res['own_cat']['loose_rate']:.4f})", flush=True)
    print(f"[DSO] maxcat  proposable={res['max_over_cats']['proposable']}/"
          f"{res['n_missed_evaluated']} rate={res['max_over_cats']['rate']:.4f} "
          f"loose={res['max_over_cats']['loose_z_only']} "
          f"({res['max_over_cats']['loose_rate']:.4f})", flush=True)
    for r in res['missed_rows']:
        if r.get('own'):
            print(f"  row {r['video']} {r['category']} "
                  f"gt={r['gt']} det_iou={r['best_det_iou']:.3f} "
                  f"own(z={r['own']['z']}, riou={r['own']['region_iou']:.3f}, "
                  f"prop={r['own']['proposable']}) "
                  f"max(z={r['maxcat']['z']}, riou={r['maxcat']['region_iou']:.3f}, "
                  f"prop={r['maxcat']['proposable']}, cat={r['maxcat'].get('peak_cat')})",
                  flush=True)
    print(f"[DSO-VERDICT] go={res['go']} (need own>={GO_OWN} and "
          f"maxcat>={GO_MAXCAT})", flush=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[DSO] result written to {OUT}', flush=True)


if __name__ == '__main__':
    main()
