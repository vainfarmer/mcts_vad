"""v190 VMC — VideoMAE class-probability-curve oracle precheck
(OFFLINE, zero VLM, zero eval spend; analysis only, no batch_evaluate).

QUESTION (the last untested REPRESENTATION FAMILY for the recall residual):
v185/v188/v189 proved the 54 pipeline-missed GT intervals are invisible to
zero-shot image-encoder text-similarity curves (ViT-L/14 native+dense,
motion-zoom, SigLIP-SO400M-384). VMC tests VIDEO-NATIVE SUPERVISED features:
the VideoMAE classifier the pipeline already uses
(OPear/videomae-large-finetuned-UCF-Crime, 14 classes = 13 anomaly + Normal;
weights cached locally -> HF_HUB_OFFLINE=1, zero download) used as a
sliding-window CLASS-PROBABILITY CURVE substrate. The missed events are
motion events (crashes, shoplifting gestures); VideoMAE models motion and
was fine-tuned on UCF-Crime train — it may see what image CLIP cannot.

CURVE RECIPE (pre-registered): slide a 16-kept-frame window over the
pipeline's kept frames, stride 4 kept frames (=2s at fps=2); per window run
VideoMAE -> 14-class softmax probabilities; assign to the window CENTER
time. Per video this gives 14 class-probability curves over time. If a
video has <16 kept frames, evaluate the single available window
(documented per video).

ORACLE per missed GT (both variants, FM13 = the 54 missed GTs from v185):
  (i)  GT-aware: z = (max of the GT-own-category probability curve over the
       GT interval - median(curve)) / (1.4826*MAD(curve), eps 1e-8)
  (ii) verdict-independent: same using the max-over-the-13-anomaly-class-
       probabilities curve.
GUARD: non-GT p90 max-z of the same curves over each video's GT-complement
(all GT intervals of that video from the parquet), p90 across videos.

GO GATE (pre-registered):
  (a) GT-aware: >= 40% of the 54 missed GTs reach z >= 2.0; AND
  (b) median GT-aware z >= +1.0; AND
  (c) guard non-GT p90 < 4.0 (both curve variants; gate on the worse).
No z_full uplift column (different instrument family); the v185 CLIP
z_full (own-cat dense-substrate z) is reported alongside for color.

PIPELINE-EXACT kept frames: VideoPreprocessor(fps=2, segment_duration=6,
motion_threshold=1.0, min_motion_interval=2.0, segment_overlap=0.25);
recomputed kept-frame timestamps validity-checked against the committed
run (results/batch_eval_v182_ref88/*.json), max_abs_ts_diff must be 0.0.

Resumable: per-video rows (curves, window-center times, checks) are cached
inside the single output JSON results/v190_vmc_precheck.json.

Usage:
  python3.14 iteration_logs/analysis_v190_vmc_precheck.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Weights are cached locally; huggingface.co is unreachable from this
# network. Offline mode makes hf_hub_download use the cache and fail fast
# instead of hanging on SYN_SENT. Caller may override.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from video_preprocessor import VideoPreprocessor            # noqa: E402
from videomae_classifier import VideoMAEClassifier          # noqa: E402

RES_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v182_ref88')
V185 = Path('/Users/zzl/PycharmProjects/vad/results/v185_dso_oracle.json')
PARQ = Path('/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet')
DATASET = Path('/Users/zzl/Movies/ucf-crime')
OUT = Path('/Users/zzl/PycharmProjects/vad/results/v190_vmc_precheck.json')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]
# VideoMAE ID2LABEL (videomae_classifier.py): 0..13 with Normal at 7.
ANOM_IDS = [i for i in range(14) if i != 7]
LABEL2ID = {v: k for k, v in VideoMAEClassifier.ID2LABEL.items()}

DEV = 'mps'
GT_FPS = 30.0
WIN = 16            # kept frames per window (pre-registered)
STRIDE = 4          # kept frames (pre-registered; = 2s at fps=2)
Z_MIN = 2.0
MAD_EPS = 1e-8
GO_RATE = 0.40      # gate (a)
GO_MEDIAN = 1.0     # gate (b)
GO_GUARD = 4.0      # gate (c)
BATCH = 8           # VideoMAE-large windows per forward on MPS


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
    """stem -> {timestamps, duration, video_path} from the committed run
    (validity cross-check + path resolution)."""
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
    """The authoritative 54 missed GTs (+ CLIP z_full for color)."""
    d = json.load(open(V185))
    rows = d['missed_rows']
    by_vid = {}
    for r in rows:
        by_vid.setdefault(r['video'], []).append(r)
    return rows, by_vid


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


def peak_z_in(curve, cts, gs, ge):
    """z of curve max over window centers inside [gs, ge]
    (nearest center to GT midpoint if none — v185/v189 convention)."""
    med, mad = curve_med_mad(curve)
    idx = np.where((cts >= gs) & (cts <= ge))[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(cts - 0.5 * (gs + ge))))])
    pk = int(idx[int(np.argmax(curve[idx]))])
    return z_of(float(curve[pk]), med, mad), float(cts[pk])


def nongt_max_z(curve, cts, all_segs):
    med, mad = curve_med_mad(curve)
    mask = np.ones(len(cts), dtype=bool)
    for gs, ge in all_segs:
        mask &= ~((cts >= gs) & (cts <= ge))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(cts))
    return z_of(float(np.max(curve[idx])), med, mad)


# ---------------------------------------------------------------- windows

def make_windows(n):
    """Pre-registered: 16-kept-frame windows, stride 4. n<16 -> single
    window over all kept frames (VideoMAE _preprocess pads by repeating
    the last frame — videomae_classifier.py idiom)."""
    if n <= WIN:
        return [(0, n)], bool(n < WIN)
    ws = [(s, s + WIN) for s in range(0, n - WIN + 1, STRIDE)]
    return ws, False


def preprocess_batch(frames, windows):
    """VideoMAE _preprocess idiom (videomae_classifier.py L82-98), vectorized
    over windows: per window sample 16 frames (uniform linspace; pad-last if
    short), PIL resize 224x224 BILINEAR, /255, (W,16,3,224,224) on CPU."""
    out = []
    for s, e in windows:
        clip = frames[s:e]
        n = len(clip)
        if n >= WIN:
            idx = np.linspace(0, n - 1, WIN, dtype=int)
        else:
            idx = np.array(list(range(n)) + [n - 1] * (WIN - n))
        sampled = clip[idx]
        resized = [np.asarray(Image.fromarray(f).resize((224, 224),
                                                          Image.BILINEAR))
                   for f in sampled]
        out.append(np.stack(resized))
    t = torch.from_numpy(np.stack(out)).float() / 255.0
    return t.permute(0, 1, 4, 2, 3).contiguous()


@torch.no_grad()
def run_windows(clf, frames, windows):
    """-> (W, 14) softmax probabilities, batched on DEV."""
    probs = np.zeros((len(windows), 14), dtype=np.float64)
    i = 0
    while i < len(windows):
        for bs in (BATCH, 1):  # MPS-OOM fallback to single-window forwards
            try:
                chunk = windows[i:i + bs]
                t = preprocess_batch(frames, chunk).to(DEV)
                t = (t - clf.mean) / clf.std
                logits = clf.model(t).logits
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs[i:i + len(chunk)] = p
                i += len(chunk)
                break
            except RuntimeError:
                if bs == 1:
                    raise
                if DEV == 'mps':
                    torch.mps.empty_cache()
                print(f'[VMC] batch={bs} failed ({windows[i]}), retry bs=1',
                      flush=True)
    return probs


# ---------------------------------------------------------------- extraction

def extract(todo, missed_by_vid, committed, gt, cache, clf):
    pre = VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.25)
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

            windows, short = make_windows(len(frames))
            cts = np.array([ts[(s + e) // 2] for s, e in windows],
                           dtype=np.float64)  # window CENTER times
            probs = run_windows(clf, frames, windows)

            cats_here = {r['category'] for r in missed_by_vid[stem]}
            assert len(cats_here) == 1, f'{stem}: multiple cats {cats_here}'
            cat = next(iter(cats_here))
            ci = LABEL2ID[cat]
            own = probs[:, ci]
            mx = np.max(probs[:, ANOM_IDS], axis=1)

            # validity cross-check: recomputed kept-frame sampling vs the
            # committed run's per-video frame timestamps
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
                           'n_frames': int(len(ts)),
                           'n_windows': int(len(windows)),
                           'short_video': bool(short),
                           'centers': cts.tolist(),
                           'curve_own': own.tolist(),
                           'curve_max': mx.tolist()}
            save(cache)
            del frames, probs
            if DEV == 'mps':
                torch.mps.empty_cache()
            print(f'[VMC] {i + 1}/{len(todo)} {stem} cat={cat} '
                  f'frames={len(ts)} windows={len(windows)} '
                  f'short={short} ts_diff={chk["max_abs_ts_diff"]} '
                  f'in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            cache[stem] = {'error': str(e)}
            save(cache)
            print(f'[VMC-ERROR] {stem}: {e}', flush=True)
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

def analyze(v185_rows, gt, cache):
    per_gt = []
    per_vid_guard = {}
    for r in v185_rows:
        stem = r['video']
        cat = r['category']
        gs, ge = r['gt']
        c = cache.get(stem, {})
        clip_z = r['own']['z'] if r.get('own') else None
        clip_inf = bool(r.get('own', {}).get('z_inf')) if r.get('own') else None
        base = {'video': stem, 'cat': cat, 'gt': [gs, ge],
                'clip_z_full': clip_z, 'clip_z_full_inf': clip_inf}
        if 'curve_own' not in c:
            base.update({'error': c.get('error', 'no cache')})
            per_gt.append(base)
            continue
        cts = np.asarray(c['centers'], dtype=np.float64)
        own = np.asarray(c['curve_own'], dtype=np.float64)
        mx = np.asarray(c['curve_max'], dtype=np.float64)

        zo, pko = peak_z_in(own, cts, gs, ge)
        zm, pkm = peak_z_in(mx, cts, gs, ge)
        base.update({
            'vmae_z_owncat': zo if np.isfinite(zo) else None,
            'vmae_z_owncat_inf': bool(not np.isfinite(zo)),
            'vmae_z_owncat_peak_t': round(pko, 3),
            'vmae_z_maxcat': zm if np.isfinite(zm) else None,
            'vmae_z_maxcat_inf': bool(not np.isfinite(zm)),
            'vmae_z_maxcat_peak_t': round(pkm, 3),
        })
        per_gt.append(base)

        if stem not in per_vid_guard:
            per_vid_guard[stem] = {
                'owncat': nongt_max_z(own, cts, gt[stem]['segs']),
                'maxcat': nongt_max_z(mx, cts, gt[stem]['segs'])}

    done = [r for r in per_gt if 'vmae_z_owncat' in r]

    def zget(row, k):
        v = row[k]
        return float('inf') if row[k + '_inf'] else v

    def rate_ge2(key):
        if not done:
            return 0.0
        return sum(1 for r in done if zget(r, key) >= Z_MIN) / len(done)

    def med(key):
        zs = [zget(r, key) for r in done]
        fin = [z for z in zs if np.isfinite(z)]
        return (float(np.median(fin)) if fin else None,
                len(zs) - len(fin))

    med_own, ninf_own = med('vmae_z_owncat')
    med_max, ninf_max = med('vmae_z_maxcat')

    g_own = [v['owncat'] for v in per_vid_guard.values()]
    g_max = [v['maxcat'] for v in per_vid_guard.values()]
    g_own_fin = [z for z in g_own if np.isfinite(z)]
    g_max_fin = [z for z in g_max if np.isfinite(z)]
    p90_own = float(np.percentile(g_own_fin, 90)) if g_own_fin else None
    p90_max = float(np.percentile(g_max_fin, 90)) if g_max_fin else None
    n_guard_inf = (len(g_own) - len(g_own_fin)) + (len(g_max) - len(g_max_fin))

    gate_a = {'n_z_ge_2': sum(1 for r in done
                              if zget(r, 'vmae_z_owncat') >= Z_MIN),
              'n_eval': len(done), 'rate': rate_ge2('vmae_z_owncat'),
              'min': GO_RATE}
    gate_a['pass'] = bool(gate_a['rate'] >= GO_RATE
                          and len(done) == len(v185_rows))
    gate_b = {'median_z_owncat': med_own, 'n_inf': ninf_own, 'min': GO_MEDIAN,
              'pass': bool(med_own is not None and med_own >= GO_MEDIAN)}
    gate_c = {'guard_p90_owncat': p90_own, 'guard_p90_maxcat': p90_max,
              'n_guard_inf': n_guard_inf, 'max': GO_GUARD,
              'n_videos': len(per_vid_guard),
              'per_video': {k: {kk: (vv if np.isfinite(vv) else 'inf')
                                for kk, vv in v.items()}
                            for k, v in sorted(per_vid_guard.items())},
              'pass': bool(p90_own is not None and p90_own < GO_GUARD
                           and p90_max is not None and p90_max < GO_GUARD)}

    res = {
        'go': bool(gate_a['pass'] and gate_b['pass'] and gate_c['pass']),
        'gates': {'a_rate_min': GO_RATE, 'b_median_min': GO_MEDIAN,
                  'c_guard_p90_max': GO_GUARD,
                  'a': gate_a, 'b': gate_b, 'c': gate_c},
        'rates': {'owncat_z_ge_2': rate_ge2('vmae_z_owncat'),
                  'maxcat_z_ge_2': rate_ge2('vmae_z_maxcat')},
        'medians': {'owncat': med_own, 'owncat_n_inf': ninf_own,
                    'maxcat': med_max, 'maxcat_n_inf': ninf_max},
        'n_missed_total': len(v185_rows),
        'n_missed_evaluated': len(done),
        'per_gt': per_gt,
        'per_video_windows': {s: {'n_frames': c.get('n_frames'),
                                  'n_windows': c.get('n_windows'),
                                  'short_video': c.get('short_video')}
                              for s, c in sorted(cache.items())
                              if 'curve_own' in c},
        'validity': {
            'ts_mismatch_videos': sorted(
                s for s, c in cache.items()
                if c.get('check') and (
                    c['check'].get('max_abs_ts_diff') is None
                    or c['check']['max_abs_ts_diff'] > 1e-6
                    or c['check']['n_frames'] != c['check']['eval_n_frames'])),
            'errors': sorted(s for s, c in cache.items() if 'error' in c),
        },
        'recipe': {'model': 'OPear/videomae-large-finetuned-UCF-Crime',
                   'window_kept_frames': WIN, 'stride_kept_frames': STRIDE,
                   'curve': '14-class softmax probs at window-center time; '
                            'owncat = GT-own-class prob, maxcat = max over '
                            '13 anomaly classes (Normal excluded)',
                   'z': '(peak-in-GT - median) / (1.4826*MAD), eps 1e-8',
                   'guard': 'per-video non-GT max-z over GT-complement, '
                            'p90 across videos'},
    }
    return res


def main():
    gt = load_gt()
    committed = load_committed()
    v185_rows, missed_by_vid = load_v185()
    print(f'[VMC] missed_gts={len(v185_rows)} '
          f'videos={len(missed_by_vid)}', flush=True)

    cache = {}
    if OUT.exists():
        try:
            cache = json.load(open(OUT)).get('per_video', {})
        except Exception:
            cache = {}
    todo = [v for v in sorted(missed_by_vid)
            if v not in cache or 'curve_own' not in cache[v]]
    print(f'[VMC] cached={len(cache)} todo={len(todo)}', flush=True)

    if todo:
        clf = VideoMAEClassifier(device=DEV)
        assert clf.model is not None, 'VideoMAE failed to load'
        print('[VMC] VideoMAE OPear/videomae-large-finetuned-UCF-Crime '
              f'loaded on {DEV} (HF_HUB_OFFLINE='
              f'{os.environ.get("HF_HUB_OFFLINE")})', flush=True)
        extract(todo, missed_by_vid, committed, gt, cache, clf)

    cache = json.load(open(OUT)).get('per_video', {}) if OUT.exists() else cache
    res = analyze(v185_rows, gt, cache)
    res['per_video'] = cache

    print('\n[VMC-ANALYSIS] ' + '=' * 60, flush=True)
    a, b, c = res['gates']['a'], res['gates']['b'], res['gates']['c']
    print(f"[VMC] gate_a: own z>=2: {a['n_z_ge_2']}/{a['n_eval']} "
          f"rate={a['rate']:.4f} (need>={GO_RATE}) pass={a['pass']}",
          flush=True)
    print(f"[VMC] gate_b: median own z={b['median_z_owncat']} "
          f"(need>=+{GO_MEDIAN}) n_inf={b['n_inf']} pass={b['pass']}",
          flush=True)
    print(f"[VMC] gate_c: guard p90 own={c['guard_p90_owncat']} "
          f"max={c['guard_p90_maxcat']} (need<{GO_GUARD}) "
          f"n_inf={c['n_guard_inf']} pass={c['pass']}", flush=True)
    print(f"[VMC] maxcat rate z>=2: {res['rates']['maxcat_z_ge_2']:.4f} "
          f"median={res['medians']['maxcat']}", flush=True)
    def zfmt(row, key):
        v = row.get(key)
        if v is not None:
            return f'{v:.3f}'
        return 'inf' if row.get(key + '_inf') else 'ERR'

    def zsort(row):
        v = row.get('vmae_z_owncat')
        if v is not None:
            return -v
        return -1e9 if row.get('vmae_z_owncat_inf') else 1e9

    for r in sorted(res['per_gt'], key=zsort):
        print(f"  row {r['video']} {r['cat']} gt={r['gt']} "
              f"clip_z_full={r['clip_z_full']} "
              f"vmae_own={zfmt(r, 'vmae_z_owncat')} "
              f"vmae_max={zfmt(r, 'vmae_z_maxcat')}",
              flush=True)
    print(f"[VMC-VERDICT] go={res['go']}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[VMC] result written to {OUT}', flush=True)


if __name__ == '__main__':
    main()
