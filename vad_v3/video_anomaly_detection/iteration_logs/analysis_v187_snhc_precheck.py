"""v187 SNH-C — window-level CLIP category-curve protection precheck (OFFLINE, zero VLM, zero eval spend).

CONTEXT. v179 SNH (video-level P(normal)>=tau=0.7894 veto) achieved Normal-video
FA -67.1% and held abs mAP but failed GT-harm (5 GT-overlapping detections
vetoed, limit <=1). MIL-based protection is PERMANENTLY CLOSED (v180 argmax:
peaks mislocalized; v186 window-threshold: MIL signal ABSENT on misfire videos
— video-level MIL max 0.039/0.00025). v187 SNH-C tests the LAST non-MIL
protection signal: the WINDOW-LEVEL CLIP category curve.

RULE (pre-registered): when the SNH video-level gate fires, a detection
SURVIVES iff the curve of ITS OWN anomaly_type category has peak z >= 2.0
inside the detection window,
    z = (max-in-window - median(curve)) / (1.4826 * MAD(curve) + 1e-8)
(v18 recipe, MAD = median absolute deviation, eps 1e-8 in the denominator).
No abstain clause: curves exist for all videos (short videos still have
~20 kept frames).

KNOWN PRIOR (documented): SNH's own video-level features already include
curve statistics (top-1 cat maxz, prominence), and it still fired on these
videos — so the window-level measurement is genuinely new information but
the prior is not high. This precheck exists to answer it exactly.

GO GATE (pre-registered):
  (a) >= 4/5 GT-harm detections survive (window z >= 2.0); AND
  (b) estimated Normal-FA relative reduction >= 25%:
      v179 fully vetoed 51 Normal videos (76 -> 25); a Normal video
      resurrects if ANY of its detections survive;
      est_fa = 25 + #resurrected; rel = (76 - est_fa)/76 >= 0.25.

INPUTS:
  - 92 [SNH-FIRE] lines parsed from results/batch_eval_v179_snh290.log
    (format verified: "[SNH-FIRE] vid=<stem> idx=<i> cat=<cat>
     t=<ts>-<te> conf=<old>->0.0000 p_normal=<p>"; the per-video eval JSONs
    results/batch_eval_v179_snh290/<stem>.json anomalies[idx] carry
    anomaly_type/t_start/t_end as ground truth — cross-checked in parse).
  - 60 distinct fired videos; pipeline-exact local recompute per video
    (machinery mirrors analysis_v185_dso_oracle.py):
      VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
        min_motion_interval=2.0, segment_overlap=0.25)
      VideoAnomalyExpert CLIP ViT-L-14/openai on mps
        -> precompute_video_features
      category curves replicate FrameScorer.get_category_scores
        (frame_scorer.py L41): logits = clip_features @ text_features.T *
        clamp(exp(logit_scale), <=100); softmax over 14 columns
        (13 cats + normal); curve_c = p_c / (p_c + p_normal + 1e-8).
      Text features: committed embeddings via the expert's own
      _precompute_text_features (13 cats x 6 prompts + committed
      prompt_weights.json) — instantiated, not reimplemented.

VALIDITY: recomputed kept-frame timestamps are compared against the eval
JSON's stored `timestamps` array per video (count + max abs diff);
mismatches are reported in the output JSON `validity`.

Resumable: per-video results are cached inside the single output JSON
results/v187_snhc_precheck.json (no other files are written).

Usage:
  python3.14 iteration_logs/analysis_v187_snhc_precheck.py
"""

import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from video_preprocessor import VideoPreprocessor          # noqa: E402
from anomaly_expert import VideoAnomalyExpert             # noqa: E402

LOG = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v179_snh290.log')
RES_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v179_snh290')
OUT = Path('/Users/zzl/PycharmProjects/vad/results/v187_snhc_precheck.json')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]

DEV = 'mps'
Z_MIN = 2.0
MAD_EPS = 1e-8

# The 5 GT-harm detections (v21 adjudication): (stem, t_start, t_end, cat)
HARM = [
    ('RoadAccidents128_x264', 3.00, 8.00, 'RoadAccidents'),   # IoU 0.7333
    ('Explosion025_x264', 10.00, 16.50, 'Explosion'),         # IoU 0.5106
    ('Shooting048_x264', 46.00, 52.00, 'Shooting'),           # IoU 0.4286
    ('Shoplifting022_x264', 45.50, 55.00, 'Shoplifting'),     # IoU 0.4211
    ('RoadAccidents002_x264', 9.00, 11.50, 'RoadAccidents'),  # IoU 0.2857
]

V179_FA_BEFORE = 76   # Normal-video FA without SNH
V179_FA_AFTER = 25    # Normal-video FA with v179 SNH (51 fully vetoed)
GO_HARM_MIN = 4
GO_FA_REL_MIN = 0.25

FIRE_RE = re.compile(
    r'\[SNH-FIRE\] vid=(\S+) idx=(\d+) cat=(\S+) '
    r't=([\d.]+)-([\d.]+) conf=([\d.]+)->0\.0000 p_normal=([\d.]+)')


# ---------------------------------------------------------------- inputs

def parse_fires():
    """Parse the 92 [SNH-FIRE] lines; cross-check vs eval JSON anomalies[]."""
    fires = []
    for m in FIRE_RE.finditer(LOG.read_text()):
        vid, idx, cat, ts, te, conf, pn = m.groups()
        fires.append({'video': vid, 'idx': int(idx), 'cat': cat,
                      't_start': float(ts), 't_end': float(te),
                      'old_conf': float(conf), 'p_normal': float(pn)})
    for f in fires:  # ground-truth cross-check against eval JSON anomalies[]
        d = json.load(open(RES_DIR / f"{f['video']}.json"))
        a = d['anomalies'][f['idx']]
        assert abs(a['t_start'] - f['t_start']) < 0.01, (f, a)
        assert abs(a['t_end'] - f['t_end']) < 0.01, (f, a)
        assert a['anomaly_type'] == f['cat'], (f, a)
    return fires


def load_eval(vids):
    """stem -> {video_path, duration, timestamps, is_normal_stem}."""
    out = {}
    for v in vids:
        d = json.load(open(RES_DIR / f'{v}.json'))
        out[v] = {
            'video_path': d.get('video_path'),
            'duration': float(d['duration']),
            'timestamps': d.get('timestamps'),
            'frame_scores_len': len(d.get('frame_scores') or []),
            # NB: the eval JSON `is_normal` field is unreliable (False for
            # Normal_Videos_*); the stem prefix is the committed convention.
            'is_normal': v.startswith('Normal'),
        }
    return out


# ---------------------------------------------------------------- rule

def window_z(curve, ts, t0, t1):
    """Peak z of `curve` inside [t0, t1] (v18 recipe, eps in denominator)."""
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med)))
    idx = np.where((ts >= t0) & (ts <= t1))[0]
    if len(idx) == 0:  # window shorter than frame spacing: nearest to center
        idx = np.array([int(np.argmin(np.abs(ts - 0.5 * (t0 + t1))))])
    peak = float(np.max(curve[idx]))
    z = (peak - med) / (1.4826 * mad + MAD_EPS)
    return {'z': float(z), 'peak': peak, 'median': med, 'mad': mad,
            'n_frames_in_window': int(len(idx)),
            'window_fallback': bool(len(idx) == 1 and not np.any(
                (ts >= t0) & (ts <= t1))),
            'survives': bool(z >= Z_MIN)}


# ---------------------------------------------------------------- recompute

def extract(todo, fires_by_vid, ev, cache):
    pre = VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.25)
    expert = VideoAnomalyExpert(CATS, clip_model_name='ViT-L-14',
                                clip_pretrained='openai', device=DEV)
    assert expert.model is not None, 'CLIP failed to load'
    assert expert.text_features is not None
    print(f'[SNHC] CLIP ViT-L-14/openai loaded on {DEV}, '
          f'text_features={tuple(expert.text_features.shape)} '
          f'(13 cats x 6 prompts + prompt_weights.json, committed path)',
          flush=True)

    for i, stem in enumerate(todo):
        t0 = time.time()
        try:
            vd = pre.process(ev[stem]['video_path'])
            frames = vd['all_frames']
            ts = np.asarray(vd['timestamps'], dtype=np.float64)
            feats = expert.precompute_video_features(frames)

            # FrameScorer.get_category_scores verbatim (frame_scorer.py L41)
            with torch.no_grad():
                logit_scale = expert.model.logit_scale.exp().detach().clamp(max=100.0)
                sim = feats @ expert.text_features.T * logit_scale
                probs = torch.softmax(sim, dim=-1).cpu().numpy()  # (N, 14)
            p_norm = probs[:, -1]
            cat_curves = {c: probs[:, ci] / (probs[:, ci] + p_norm + 1e-8)
                          for ci, c in enumerate(CATS)}

            # validity cross-check: recomputed sampling vs v179 eval run
            ev_ts = ev[stem]['timestamps']
            chk = {'n_frames': int(len(ts)),
                   'eval_n_frames': len(ev_ts) if ev_ts else None,
                   'eval_frame_scores_len': ev[stem]['frame_scores_len']}
            if ev_ts and len(ev_ts) == len(ts):
                chk['max_abs_ts_diff'] = float(np.max(np.abs(
                    np.asarray(ev_ts, dtype=np.float64) - ts)))
            else:
                chk['max_abs_ts_diff'] = None

            rows = []
            for f in fires_by_vid[stem]:
                r = dict(f)
                r.update(window_z(cat_curves[f['cat']], ts,
                                  f['t_start'], f['t_end']))
                rows.append(r)
            cache[stem] = {'check': chk, 'fires': rows}
            save(cache)
            print(f'[SNHC] {i + 1}/{len(todo)} {stem} frames={len(ts)} '
                  f'fires={len(rows)} '
                  f'survive={sum(r["survives"] for r in rows)} '
                  f'ts_diff={chk["max_abs_ts_diff"]} '
                  f'in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            cache[stem] = {'error': str(e), 'fires': []}
            save(cache)
            print(f'[SNHC-ERROR] {stem}: {e}', flush=True)
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

def analyze(fires, ev, cache):
    all_rows = []
    for stem, c in cache.items():
        all_rows.extend(c.get('fires', []))
    n_done_videos = sum(1 for c in cache.values() if 'fires' in c
                        and 'error' not in c)

    surviving = [r for r in all_rows if r.get('survives')]

    # gate (a): the 5 GT-harm detections
    harm_rows = []
    for stem, hs, he, cat in HARM:
        r = next((r for r in all_rows
                  if r['video'] == stem
                  and abs(r['t_start'] - hs) < 0.01
                  and abs(r['t_end'] - he) < 0.01), None)
        if r is None:
            harm_rows.append({'video': stem, 'window': [hs, he], 'cat': cat,
                              'error': 'fire row missing'})
        else:
            harm_rows.append({'video': stem, 'window': [hs, he], 'cat': cat,
                              'z': round(r['z'], 4), 'peak': round(r['peak'], 6),
                              'median': round(r['median'], 6),
                              'mad': round(r['mad'], 6),
                              'n_frames_in_window': r['n_frames_in_window'],
                              'survives': bool(r['survives'])})
    n_harm_survive = sum(1 for r in harm_rows if r.get('survives'))

    # gate (b): Normal-video resurrection among the 51 fully vetoed
    normal_vids = sorted(v for v in {f['video'] for f in fires}
                         if ev[v]['is_normal'])
    resurrected = sorted({r['video'] for r in surviving
                          if ev[r['video']]['is_normal']})
    est_fa = V179_FA_AFTER + len(resurrected)
    fa_rel = (V179_FA_BEFORE - est_fa) / V179_FA_BEFORE

    validity = {
        'n_videos_expected': len({f['video'] for f in fires}),
        'n_videos_recomputed': n_done_videos,
        'ts_mismatch_videos': sorted(
            s for s, c in cache.items()
            if c.get('check') and (
                c['check'].get('max_abs_ts_diff') is None
                or c['check']['max_abs_ts_diff'] > 1e-6
                or c['check']['n_frames'] != c['check']['eval_n_frames'])),
        'errors': sorted(s for s, c in cache.items() if 'error' in c),
    }

    go = bool(n_harm_survive >= GO_HARM_MIN
              and fa_rel >= GO_FA_REL_MIN
              and n_done_videos == validity['n_videos_expected']
              and not validity['errors'])

    return {
        'go': go,
        'rule': (f'detection survives SNH veto iff own-category curve peak '
                 f'z >= {Z_MIN} in detection window; '
                 f'z = (max_in_window - median) / (1.4826*MAD + {MAD_EPS})'),
        'harm_rows': harm_rows,
        'n_harm_survive': n_harm_survive,
        'harm_gate': f'{n_harm_survive}/5 survive (need >= {GO_HARM_MIN})',
        'fires_surviving': len(surviving),
        'fires_total': len(all_rows),
        'n_normal_fully_vetoed': len(normal_vids),
        'n_normal_resurrected': len(resurrected),
        'resurrected_videos': resurrected,
        'est_fa': est_fa,
        'fa_rel_est': round(fa_rel, 4),
        'fa_gate': (f'est_fa={est_fa} (25 + {len(resurrected)} resurrected), '
                    f'rel=(76-{est_fa})/76={fa_rel:.4f} '
                    f'(need >= {GO_FA_REL_MIN})'),
        'validity': validity,
        'fires': sorted(all_rows, key=lambda r: (r['video'], r['idx'])),
    }


def main():
    fires = parse_fires()
    vids = sorted({f['video'] for f in fires})
    ev = load_eval(vids)
    fires_by_vid = {}
    for f in fires:
        fires_by_vid.setdefault(f['video'], []).append(f)
    print(f'[SNHC] fires={len(fires)} videos={len(vids)} '
          f'normal={sum(ev[v]["is_normal"] for v in vids)}', flush=True)

    cache = {}
    if OUT.exists():
        try:
            cache = json.load(open(OUT)).get('per_video', {})
        except Exception:
            cache = {}
    todo = [v for v in vids if v not in cache]
    print(f'[SNHC] cached={len(cache)} todo={len(todo)}', flush=True)
    if todo:
        extract(todo, fires_by_vid, ev, cache)

    cache = json.load(open(OUT)).get('per_video', {}) if OUT.exists() else cache
    res = analyze(fires, ev, cache)
    res['per_video'] = cache

    print('\n[SNHC-ANALYSIS] ' + '=' * 60, flush=True)
    for r in res['harm_rows']:
        print(f"  HARM {r['video']} {r['window']} {r['cat']} "
              f"z={r.get('z')} survives={r.get('survives')}", flush=True)
    print(f"[SNHC] harm_survive={res['n_harm_survive']}/5 | "
          f"fires_surviving={res['fires_surviving']}/{res['fires_total']} | "
          f"resurrected={res['n_normal_resurrected']}/"
          f"{res['n_normal_fully_vetoed']} | {res['fa_gate']}", flush=True)
    print(f"[SNHC-VERDICT] go={res['go']}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[SNHC] result written to {OUT}', flush=True)


if __name__ == '__main__':
    main()
