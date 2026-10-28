"""v189 BBS — backbone-swap oracle precheck (OFFLINE, zero VLM, zero eval spend).

QUESTION (the last remaining labels-free recall hypothesis): v185 proved the
54 pipeline-missed GT intervals are invisible to CLIP ViT-L/14 (openai) curves
at native scale (strict proposable 1/54); v188 proved motion-guided spatial
zoom does not help. BBS tests whether the failure is BACKBONE-SPECIFIC:
recompute the curves with a DIFFERENT pretrained image encoder and measure
whether the missed GTs become visible.

BACKBONE (pre-registered order):
  primary  = open_clip ViT-SO400M-14-SigLIP-384 pretrained webli
             (strong zero-shot, 384px input — better for small objects)
  fallback = ViT-H-14 pretrained laion2b_s32b_b79k (only if the primary's
             download/load fails: network, MPS OOM, ...)
  If both fail: report infrastructure failure loudly and stop (no silent
  substitution).

COMPARABILITY (pre-registered): the committed text-prompt TABLE
(13 cats x 6 prompts + 6 normal from anomaly_expert.CATEGORY_PROMPTS /
NORMAL_PROMPTS) is REUSED VERBATIM with the NEW backbone's own tokenizer;
the learned prompt_weights.json combination applies as-is (weights are
per-prompt, backbone-agnostic). All of this is achieved by instantiating the
COMMITTED VideoAnomalyExpert class with clip_model_name/clip_pretrained
pointed at the new backbone — its _precompute_text_features path (prompt
table -> new backbone's tokenizer -> encode_text -> L2 -> learned-weight
combo -> L2) is reused, not reimplemented. The curve recipe stays identical
(v185 / FrameScorer.get_category_scores): per-frame features -> L2-norm ->
logits = feats @ text.T * clamp(exp(logit_scale), <=100) -> softmax over the
14 columns -> cat/(cat+normal+1e-8). z-scores are in-video relative so
backbone calibration cancels.

BASELINE (not recomputed): results/v185_dso_oracle.json missed_rows carry the
full-frame ViT-L/14 peak z per missed GT (own-cat). 51 videos / 54 missed
GTs. Baseline loose rate = 8/54 (14.8%).

PIPELINE-EXACT sampling: VideoPreprocessor(fps=2, segment_duration=6,
motion_threshold=1.0, min_motion_interval=2.0, segment_overlap=0.25); the
recomputed kept-frame timestamps are validity-checked against the committed
run's per-video frame arrays (max_abs_ts_diff must be 0.0).

COMPUTE: per missed GT, peak z_new = (max-in-GT of the new-backbone
GT-own-cat curve - median) / (1.4826*MAD), MAD-scale eps 1e-8 (if scale <
eps: z = inf when peak > median else 0.0, same convention as v185/v188).
GUARD: per video, max-z of the same new curve over NON-GT frames (complement
of ALL GT intervals of that video, from the parquet), then p90 across the 51
videos.

GO GATE (pre-registered, identical shape to v185/v188): GO iff
  (a) >= 40% of the 54 missed GTs reach z_new >= 2.0 (GT-own-cat), AND
  (b) median uplift (z_new - z_full) over the 54 >= +1.0, AND
  (c) guard: non-GT p90 max-z < 4.0.

Resumable: per-video rows (curve, timestamps, check) are cached inside the
single output JSON results/v189_bbs_precheck.json.

Usage:
  python3.14 iteration_logs/analysis_v189_bbs_precheck.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# huggingface.co is unreachable from this network (curl timeout, SYN_SENT
# hangs); the public mirror hf-mirror.com works. Must be set BEFORE
# open_clip / huggingface_hub are imported (anomaly_expert imports open_clip
# at module level). Respect a caller-provided HF_ENDPOINT.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from video_preprocessor import VideoPreprocessor          # noqa: E402
from anomaly_expert import VideoAnomalyExpert             # noqa: E402

RES_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v182_ref88')
V185 = Path('/Users/zzl/PycharmProjects/vad/results/v185_dso_oracle.json')
PARQ = Path('/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet')
DATASET = Path('/Users/zzl/Movies/ucf-crime')
OUT = Path('/Users/zzl/PycharmProjects/vad/results/v189_bbs_precheck.json')

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
BATCH_SIZE = 32       # per pre-registration: keeps MPS happy at 384px

# pre-registered backbone order
BACKBONES = [
    {'name': 'ViT-SO400M-14-SigLIP-384', 'pretrained': 'webli'},
    {'name': 'ViT-H-14', 'pretrained': 'laion2b_s32b_b79k'},
]


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


# ---------------------------------------------------------------- backbone

def _find_weight_file(hf_repo):
    """Locate the actual weight file open_clip downloaded into the HF hub
    cache for the given repo id (reporting only)."""
    try:
        from huggingface_hub import scan_cache_dir
        rid = hf_repo.strip('/').split('/')[-1]
        for repo in scan_cache_dir().repos:
            if repo.repo_type != 'model':
                continue
            if repo.repo_id.strip('/').split('/')[-1] != rid:
                continue
            for rev in sorted(repo.revisions, key=lambda r: -r.size_on_disk):
                for f in rev.files:
                    if f.file_name.endswith(('.safetensors', '.bin')):
                        return {'hf_repo_id': repo.repo_id,
                                'commit': rev.commit_hash,
                                'weight_file': str(f.file_path),
                                'size_bytes': f.size_on_disk}
    except Exception:
        pass
    return {'hf_repo_id': hf_repo, 'commit': None,
            'weight_file': None, 'size_bytes': None}


def load_backbone():
    """Try the pre-registered backbone order. Returns (expert, info).
    Raises RuntimeError if every backbone fails (infrastructure failure)."""
    from open_clip import get_pretrained_cfg
    from PIL import Image
    errors = []
    for spec in BACKBONES:
        name, tag = spec['name'], spec['pretrained']
        t0 = time.time()
        try:
            print(f'[BBS] trying backbone {name}/{tag} ...', flush=True)
            expert = VideoAnomalyExpert(CATS, clip_model_name=name,
                                        clip_pretrained=tag, device=DEV)
            if expert.model is None or expert.text_features is None:
                raise RuntimeError('VideoAnomalyExpert fell back to '
                                   'model=None (load failed inside)')
            # smoke forward pass at native input size: catches MPS OOM /
            # broken weights before the 51-video run starts
            cfg = get_pretrained_cfg(name, tag)
            probe = expert.preprocess(
                Image.new('RGB', (400, 300), (128, 128, 128)))
            probe = probe.unsqueeze(0).to(DEV)
            with torch.no_grad():
                f = expert.model.encode_image(probe)
                assert torch.isfinite(f).all(), 'non-finite smoke features'
            wf = _find_weight_file(cfg.get('hf_hub', ''))
            info = {'name': name, 'pretrained': tag,
                    'hf_hub_cfg': cfg.get('hf_hub'),
                    'load_seconds': round(time.time() - t0, 1),
                    'text_features_shape': list(expert.text_features.shape),
                    'image_feature_dim': int(f.shape[-1]),
                    'input_size': list(probe.shape[-2:]),
                    **wf}
            print(f'[BBS] backbone loaded: {name}/{tag} in '
                  f'{info["load_seconds"]}s, text_features='
                  f'{info["text_features_shape"]}, feat_dim='
                  f'{info["image_feature_dim"]}, input='
                  f'{info["input_size"]}, weight_file='
                  f'{info["weight_file"]}', flush=True)
            return expert, info
        except Exception as e:
            print(f'[BBS-FAIL] backbone {name}/{tag} failed: {e}', flush=True)
            traceback.print_exc()
            errors.append(f'{name}/{tag}: {e}')
            try:
                del expert
            except Exception:
                pass
            if DEV == 'mps':
                torch.mps.empty_cache()
    raise RuntimeError('INFRASTRUCTURE FAILURE: all pre-registered backbones '
                       'failed to load -> ' + ' | '.join(errors))


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

def extract(todo, missed_by_vid, committed, gt, cache, expert):
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
            feats = expert.precompute_video_features(frames,
                                                     batch_size=BATCH_SIZE)

            # FrameScorer.get_category_scores verbatim (as v185/v188)
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
                           'n_frames': int(len(ts))}
            save(cache)
            del feats, sim, probs
            if DEV == 'mps':
                torch.mps.empty_cache()
            print(f'[BBS] {i + 1}/{len(todo)} {stem} cat={cat} '
                  f'frames={len(ts)} '
                  f'ts_diff={chk["max_abs_ts_diff"]} '
                  f'in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            cache[stem] = {'error': str(e)}
            save(cache)
            print(f'[BBS-ERROR] {stem}: {e}', flush=True)
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

def analyze(v185_rows, gt, cache, backbone_info):
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
                'z_full': z_full}
        if 'curve' not in entry:
            rows.append({**base, 'z_new': None, 'z_new_inf': False,
                         'uplift': None, 'error': True})
            continue
        curve = np.asarray(entry['curve'], dtype=np.float64)
        ts = np.asarray(entry['timestamps'], dtype=np.float64)
        zn = gt_peak_z(curve, ts, gseg)
        zn_inf = bool(not np.isfinite(zn))
        uplift = (None if zn_inf or z_full is None
                  else float(zn - z_full))
        rows.append({**base, 'z_new': None if zn_inf else float(zn),
                     'z_new_inf': zn_inf, 'uplift': uplift})
        if stem not in per_vid_guard:
            per_vid_guard[stem] = nongt_max_z(curve, ts, gt[stem]['segs'])

    done = [r for r in rows if not r.get('error')]
    reach = sum(1 for r in done
                if r['z_new_inf'] or (r['z_new'] is not None
                                      and r['z_new'] >= Z_MIN))
    rate = reach / len(done) if done else 0.0

    upl = [r['uplift'] for r in done if r['uplift'] is not None]
    med_uplift = float(np.median(upl)) if upl else None

    gz = [z for z in per_vid_guard.values() if np.isfinite(z)]
    n_guard_inf = sum(1 for z in per_vid_guard.values() if not np.isfinite(z))
    guard_p90 = float(np.percentile(gz, 90)) if gz else None

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
        'z_new_inf_rows': [f"{r['video']}|{r['gt']}" for r in done
                           if r['z_new_inf']],
    }

    return {
        'experiment': 'v189_bbs_precheck',
        'hypothesis': 'backbone-specific: the 54 pipeline-missed GTs are '
                      'invisible to ViT-L/14-openai specifically; a different '
                      'pretrained image encoder makes them visible',
        'backbone': backbone_info,
        'baseline': {'source': str(V185), 'backbone': 'ViT-L-14/openai',
                     'full_frame_loose_rate': 8 / 54,
                     'full_frame_loose': 8, 'n_missed': 54},
        'comparability': {
            'prompt_table': 'anomaly_expert CATEGORY_PROMPTS (13x6) + '
                            'NORMAL_PROMPTS (6) verbatim, tokenized with the '
                            'NEW backbone own tokenizer (open_clip.'
                            'get_tokenizer(backbone))',
            'prompt_combination': 'committed prompt_weights.json learned '
                                  'per-prompt weights, applied as-is '
                                  '(backbone-agnostic)',
            'curve_recipe': 'L2(feats) @ L2(text).T * clamp(exp(logit_scale)'
                            ',<=100) -> softmax over 14 -> cat/(cat+normal)',
        },
        'gates': {'z_min': Z_MIN, 'a_rate_min': GO_RATE,
                  'b_median_uplift_min': GO_UPLIFT, 'c_guard_p90_max': GO_GUARD},
        'n_missed': len(v185_rows),
        'n_videos': len({r['video'] for r in v185_rows}),
        'gate_a': gate_a,
        'gate_b': gate_b,
        'gate_c': gate_c,
        'go': go,
        'rows': rows,
        'validity': validity,
    }


def main():
    v185_rows, missed_by_vid = load_v185()
    gt = load_gt()
    committed = load_committed()
    stems = sorted(missed_by_vid)
    print(f'[BBS] videos={len(stems)} missed_gts={len(v185_rows)} '
          f'(baseline full-frame loose=8/54=0.1481)', flush=True)

    backbone_info = None
    if OUT.exists():
        try:
            old = json.load(open(OUT))
            backbone_info = old.get('backbone')
        except Exception:
            pass

    cache = {}
    if OUT.exists():
        try:
            cache = json.load(open(OUT)).get('per_video', {})
        except Exception:
            cache = {}
    todo = [v for v in stems if v not in cache or 'curve' not in cache[v]]
    print(f'[BBS] cached={len(stems) - len(todo)} todo={len(todo)}',
          flush=True)
    if todo:
        expert, backbone_info = load_backbone()  # raises if ALL fail
        extract(todo, missed_by_vid, committed, gt, cache, expert)
    elif backbone_info is None:
        # all videos cached but no backbone record (should not happen);
        # load anyway for an honest report
        _expert, backbone_info = load_backbone()
        del _expert
        if DEV == 'mps':
            torch.mps.empty_cache()

    cache = json.load(open(OUT)).get('per_video', {}) if OUT.exists() else cache
    res = analyze(v185_rows, gt, cache, backbone_info)
    res['per_video'] = cache

    a, b, c = res['gate_a'], res['gate_b'], res['gate_c']
    bb = res['backbone'] or {}
    print('\n[BBS-ANALYSIS] ' + '=' * 60, flush=True)
    print(f"[BBS] backbone: {bb.get('name')}/{bb.get('pretrained')} "
          f"weight={bb.get('weight_file')}", flush=True)
    print(f"[BBS] gate_a: {a['n_ge2']}/{a['n_evaluated']} z_new>=2.0 "
          f"rate={a['rate']:.4f} (need>={GO_RATE}) pass={a['pass']}", flush=True)
    print(f"[BBS] gate_b: median_uplift={b['median_uplift']} "
          f"(need>={GO_UPLIFT}) pass={b['pass']}", flush=True)
    print(f"[BBS] gate_c: guard_p90={c['guard_p90']} "
          f"(need<{GO_GUARD}) pass={c['pass']}", flush=True)
    fin = [r for r in res['rows'] if r.get('uplift') is not None]
    for r in sorted(fin, key=lambda r: -r['uplift'])[:5]:
        print(f"  top-uplift {r['video']} {r['cat']} gt={r['gt']} "
              f"z_full={r['z_full']:.3f} z_new={r['z_new']:.3f} "
              f"uplift=+{r['uplift']:.3f}", flush=True)
    print(f"[BBS] validity: ts_mismatch={res['validity']['ts_mismatch_videos']} "
          f"errors={res['validity']['errors']} "
          f"z_inf={len(res['validity']['z_new_inf_rows'])}", flush=True)
    print(f"[BBS-VERDICT] go={res['go']}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[BBS] result written to {OUT}', flush=True)


if __name__ == '__main__':
    main()
