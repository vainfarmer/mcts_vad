"""
v119 offline pre-check — MELC (Motion-Energy Localization Contrast) re-rank bake-off.

HARNESS-FIRST hard gate (v117/v118 lesson, C mandate #3): this JSON MUST exist and
PASS the fidelity abort-gate BEFORE any live A/B is permitted. Clones the trusted
analysis_v118_fsl_precheck.py structure (which itself reproduces v114 Q-on).

WHAT IS NEW vs v118: the re-rank signal is the appearance-ORTHOGONAL raw pixel-motion
channel. For each v114 REAL final detection window [t_s,t_e], using the IDENTICAL
in/out time-masking as _qscore_rerank, compute the AutoLoc Outer-Inner-Contrast motion
margin  melc_raw = mean(motion[in]) - mean(motion[out])  on the recomputed
motion_energy curve (pipeline L1667: cv2 64x64 per-frame mean-abs frame diff). Then the
EXACT v111 protocol: within-video min-max -> n in [0,1]; <=1 det or spread<1e-9 -> n=1
(inert); conf' = min(1, conf*(base+(1-base)*n)), base=0.5.

Faithful in-vivo protocol (FM6-safe): a re-rank cannot change detections/boundaries/
count, so re-scoring v114's OWN real final detections == the live output; mAP depends
only on the global confidence ordering. The one novel unproven step is offline motion
re-decode byte-matching live L1667 — caught by the fidelity abort-gate (anchor mAP +
per-video motion length == cached ts length).

Anchors (v114 JSONs are RAW, qscore NOT baked in — 0 [QSCORE-GATE] markers in v114 log):
  raw-baseline  = v114 as-is
  Q-only anchor = raw + committed v111 Q-score      (fidelity target 0.15632)
  arm stack     = raw + Q-score + MELC(arm)          (the thing under test)

THREE arms adjudicated OFFLINE ONLY (one controlled live change; only the winner flag
goes live):
  oic     = ungated plain-mean OIC              [PRIMARY]
  logoic  = ungated LogOIC distance-weighted outer mean (1/log(2+dist), FreeZAD arm)
  catgate = plain OIC applied ONLY to motion-salient-category detections (diagnostic:
            quantifies the double-counting A's category gate would incur)

PRE-REGISTERED VERDICT (C's binding launch gate):
  promote EXACTLY ONE arm iff its deterministic delta over the 0.15632 Q-on anchor
  >= +0.005 AND it does NOT net-regress the flat-CLIP subset {Shoplifting,Stealing,
  Arson,Burglary} AND broad-based across AP@0.3/0.5/0.7 AND (C mandate #1, HARD ABORT)
  |Pearson(MELC_n, Qscore_n)| < 0.6 AND (variance kill-gate) nonzero within-video melc
  spread on the majority of multi-detection videos. Else -> ship ALL flags OFF (revert).

Mandate #2: also reports Q-score's own within-video n-variance on the flat-CLIP subset,
to CONFIRM Q is actually near-inert there before crediting MELC with a complementary
subset effect (Q reads the FLOODED fused curve, not raw CLIP, so this must be measured
not assumed).

Run (motion re-decode is GPU-free CPU cv2):  python3 iteration_logs/analysis_v119_melc_precheck.py
"""
import os
import re
import sys
import json
import glob
import shutil
import pickle
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa
from config import VADConfig  # noqa
from video_preprocessor import VideoPreprocessor  # noqa

V114_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
FEAT_CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v119_melc_feats.pkl'
OUT_JSON = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v119_melc_precheck.json'
# optional: reuse the v118 CLIP feature cache (if present) to compute the SECONDARY
# TSM orthogonality check without any GPU CLIP re-extraction here.
V118_FEAT_CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v118_fsl_feats.pkl'

Q_ON_ANCHOR = 0.15632   # committed v114 Q-on baseline
FIDELITY_TOL = 0.0005
DELTA_GATE = 0.005
CORR_GATE = 0.6
MELC_BASE = 0.5
QS_BASE = 0.5

FLAT_CLIP = {'Shoplifting', 'Stealing', 'Arson', 'Burglary'}
MOTION_SALIENT = {'Fighting', 'Assault', 'Shooting', 'RoadAccidents',
                  'Explosion', 'Abuse', 'Vandalism'}
KNOWN_CATS = (FLAT_CLIP | MOTION_SALIENT |
              {'Arrest', 'RoadAccidents', 'Robbery', 'Fighting', 'Normal'})


def video_category(name):
    """UCF-Crime video id -> category via the leading-alpha prefix."""
    if name.startswith('Normal'):
        return 'Normal'
    m = re.match(r'([A-Za-z]+?)(\d+)', name)
    return m.group(1) if m else name


# ----------------------------------------------------------------------------
# Committed v111 Q-score (equal weights, base 0.5) — byte-matches _qscore_rerank
# ----------------------------------------------------------------------------
def qscore_raw(fs, ts, t_s, t_e):
    fs = np.asarray(fs, dtype=float); ts = np.asarray(ts, dtype=float)
    if len(fs) < 3:
        return 0.0
    in_mask = (ts >= t_s) & (ts <= t_e)
    if in_mask.sum() < 1:
        return 0.0
    peak = fs[in_mask].max()
    peak_idx = np.where(in_mask)[0][np.argmax(fs[in_mask])]
    out_mask = ~in_mask
    out_mean = fs[out_mask].mean() if out_mask.any() else fs.min()
    vid_range = fs.max() - fs.min() + 1e-9
    contrast = float(np.clip((peak - out_mean) / vid_range, 0, 1))
    grad = np.abs(np.gradient(fs))
    i_s = int(np.argmin(np.abs(ts - t_s))); i_e = int(np.argmin(np.abs(ts - t_e)))
    sharp = float(np.clip((grad[i_s] + grad[i_e]) / (2 * (grad.max() + 1e-9)), 0, 1))
    center_t = 0.5 * (t_s + t_e); dur = max(t_e - t_s, 1e-6)
    center = float(np.clip(1 - 2 * abs(ts[peak_idx] - center_t) / dur, 0, 1))
    return (sharp + contrast + center) / 3.0


def within_norm(x):
    x = np.asarray(x, dtype=float)
    if len(x) > 1 and (x.max() - x.min()) > 1e-9:
        return (x - x.min()) / (x.max() - x.min())
    return np.ones_like(x)


# ----------------------------------------------------------------------------
# MELC raw OIC margin — byte-matches VADPipeline._melc_rerank
# ----------------------------------------------------------------------------
def melc_raw(motion, ts, anoms, weighted=False):
    """Returns per-detection OIC motion margin m_in - m_out (weighted outer mean
    optionally LogOIC 1/log(2+|idx-center|))."""
    motion = np.asarray(motion, dtype=float); ts = np.asarray(ts, dtype=float)
    ms = []
    for a in anoms:
        t_s, t_e = a['t_start'], a['t_end']
        in_mask = (ts >= t_s) & (ts <= t_e)
        if in_mask.sum() < 1:
            ms.append(0.0)
            continue
        out_mask = ~in_mask
        m_in = float(motion[in_mask].mean())
        if out_mask.any():
            if weighted:
                center_t = 0.5 * (t_s + t_e)
                center_idx = int(np.argmin(np.abs(ts - center_t)))
                out_idx = np.where(out_mask)[0]
                w = 1.0 / np.log(2.0 + np.abs(out_idx - center_idx))
                m_out = float(np.average(motion[out_mask], weights=w))
            else:
                m_out = float(motion[out_mask].mean())
        else:
            m_out = float(motion.min())
        ms.append(m_in - m_out)
    return np.asarray(ms, dtype=float)


# ----------------------------------------------------------------------------
# Motion-energy re-decode (byte-matches pipeline.py L1667), cached to disk
# ----------------------------------------------------------------------------
def build_motion_cache():
    if FEAT_CACHE.exists():
        with open(FEAT_CACHE, 'rb') as f:
            return pickle.load(f)
    import cv2
    cfg = VADConfig()
    pp = cfg.preprocess
    pre = VideoPreprocessor(fps=pp.fps, segment_duration=pp.segment_duration,
                            motion_threshold=pp.motion_threshold,
                            min_motion_interval=pp.min_motion_interval,
                            segment_overlap=pp.segment_overlap)
    cache = {}
    jfs = sorted(glob.glob(str(V114_DIR / '*.json')))
    for k, jf in enumerate(jfs):
        name = Path(jf).stem
        d = json.load(open(jf))
        vp = d.get('video_path')
        if not vp or not Path(vp).exists():
            print(f"[MOTION] {k+1}/{len(jfs)} {name}: video missing -> skip", flush=True)
            cache[name] = None
            continue
        vd = pre.process(vp)
        frames = vd['all_frames']
        if len(frames) < 2:
            motion = np.zeros(len(frames))
        else:
            small = np.stack([cv2.resize(f, (64, 64)) for f in frames]).astype(np.float32)
            motion = np.zeros(len(frames))
            motion[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
        cache[name] = {'motion': np.asarray(motion, dtype=np.float32),
                       'ts': np.asarray(vd['timestamps'], dtype=float)}
        print(f"[MOTION] {k+1}/{len(jfs)} {name}: motion={motion.shape} "
              f"range[{motion.min():.3f},{motion.max():.3f}]", flush=True)
    with open(FEAT_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    return cache


def load_tsm_pd(name, ts_ref, anoms):
    """Optional secondary TSM per-detection signal, computed from the v118 CLIP
    cache if present; returns None when unavailable (GPU-free here)."""
    global _V118_CACHE, _TSM_FN
    if '_V118_CACHE' not in globals():
        if V118_FEAT_CACHE.exists():
            with open(V118_FEAT_CACHE, 'rb') as f:
                _V118_CACHE = pickle.load(f)
            from pipeline import _compute_tsm_novelty as _fn
            _TSM_FN = _fn
        else:
            _V118_CACHE = None
            _TSM_FN = None
    if _V118_CACHE is None:
        return None
    fc = _V118_CACHE.get(name)
    if fc is None or fc.get('F') is None or len(fc['F']) < 3:
        return None
    nov = _TSM_FN(fc['F'])
    Fts = fc['ts']
    out = []
    for a in anoms:
        m = (Fts >= a['t_start']) & (Fts <= a['t_end'])
        out.append(float(nov[m].mean()) if m.any() else 0.0)
    return np.asarray(out)


# ----------------------------------------------------------------------------
# Build re-ranked result dirs
# ----------------------------------------------------------------------------
def build_dir(out_dir, apply_qscore, melc_mode, motion_cache, cat_filter=None):
    """cat_filter: if not None, only copy videos whose category is in this set."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        if cat_filter is not None and video_category(name) not in cat_filter \
                and video_category(name) != 'Normal':
            continue
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if apply_qscore and fs and ts and anoms:
            qs = np.array([qscore_raw(fs, ts, a['t_start'], a['t_end']) for a in anoms])
            qn = within_norm(qs)
            for a, q in zip(anoms, qn):
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * (QS_BASE + (1 - QS_BASE) * float(q)))
        if melc_mode and anoms:
            mc = motion_cache.get(name)
            if mc is not None and mc['motion'] is not None and len(mc['motion']) >= 3:
                weighted = (melc_mode == 'logoic')
                raw = melc_raw(mc['motion'], mc['ts'], anoms, weighted=weighted)
                n = within_norm(raw)
                for a, nv in zip(anoms, n):
                    if melc_mode == 'catgate' and a.get('anomaly_type') not in MOTION_SALIENT:
                        continue  # gated out: confidence unchanged
                    a['confidence'] = min(1.0, a['confidence'] * (MELC_BASE + (1 - MELC_BASE) * float(nv)))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


# ----------------------------------------------------------------------------
# Orthogonality + variance diagnostics
# ----------------------------------------------------------------------------
def diagnostics(motion_cache):
    pooled = {'q': [], 'tsm': [], 'oic': [], 'logoic': []}
    tsm_available = False
    var_nonzero = {'oic': 0, 'logoic': 0}
    var_total = 0
    # flat-CLIP-subset specific diagnostics (mandate #2 + variance kill-gate split)
    flat_q_var, flat_melc_var, flat_n = [], [], 0
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        mc = motion_cache.get(name)
        if not (fs and ts and anoms) or len(anoms) < 2:
            continue
        if mc is None or mc['motion'] is None or len(mc['motion']) < 3:
            continue
        var_total += 1
        qs = np.array([qscore_raw(fs, ts, a['t_start'], a['t_end']) for a in anoms])
        qn = within_norm(qs)
        oic = melc_raw(mc['motion'], mc['ts'], anoms, weighted=False)
        logoic = melc_raw(mc['motion'], mc['ts'], anoms, weighted=True)
        if float(np.var(oic)) > 1e-12:
            var_nonzero['oic'] += 1
        if float(np.var(logoic)) > 1e-12:
            var_nonzero['logoic'] += 1
        pooled['q'].extend(qn.tolist())
        pooled['oic'].extend(within_norm(oic).tolist())
        pooled['logoic'].extend(within_norm(logoic).tolist())
        tsm_pd = load_tsm_pd(name, ts, anoms)
        if tsm_pd is not None and len(tsm_pd) == len(anoms):
            tsm_available = True
            pooled['tsm'].extend(within_norm(tsm_pd).tolist())
        else:
            pooled['tsm'].extend([np.nan] * len(anoms))
        if video_category(name) in FLAT_CLIP:
            flat_n += 1
            flat_q_var.append(float(np.var(within_norm(qs))))
            flat_melc_var.append(float(np.var(within_norm(oic))))

    def pear(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    out = {'multi_det_videos': var_total, 'tsm_available': tsm_available, 'per_arm': {}}
    for mode in ('oic', 'logoic'):
        out['per_arm'][mode] = {
            'corr_vs_q': abs(pear(pooled[mode], pooled['q'])),
            'corr_vs_tsm': abs(pear(pooled[mode], pooled['tsm'])) if tsm_available else None,
            'var_nonzero_videos': var_nonzero[mode],
        }
    out['flat_clip'] = {
        'n_multidet_videos': flat_n,
        'q_n_var_mean': float(np.mean(flat_q_var)) if flat_q_var else None,
        'melc_n_var_mean': float(np.mean(flat_melc_var)) if flat_melc_var else None,
        'q_near_inert': (bool(np.mean(flat_q_var) < 0.02) if flat_q_var else None),
    }
    return out


def stratified_delta(motion_cache, melc_mode):
    """Per-category stratified mAP delta (arm - Q-anchor) on flat-CLIP vs
    motion-salient subsets. Same video set in both dirs -> delta is faithful."""
    res = {}
    for label, catset in (('flat_clip', FLAT_CLIP), ('motion_salient', MOTION_SALIENT)):
        a_dir = ROOT / f'results/batch_eval_v119_{melc_mode}_{label}_anchor'
        m_dir = ROOT / f'results/batch_eval_v119_{melc_mode}_{label}_arm'
        build_dir(a_dir, apply_qscore=True, melc_mode=None, motion_cache=motion_cache, cat_filter=catset)
        build_dir(m_dir, apply_qscore=True, melc_mode=melc_mode, motion_cache=motion_cache, cat_filter=catset)
        a_tm = run_eval(a_dir); m_tm = run_eval(m_dir)
        res[label] = {'anchor_mAP': a_tm['mAP'], 'arm_mAP': m_tm['mAP'],
                      'delta': m_tm['mAP'] - a_tm['mAP']}
    return res


def main():
    print("[STEP] building motion-energy cache (GPU-free cv2 re-decode)...", flush=True)
    motion_cache = build_motion_cache()

    # --- FIDELITY ABORT-GATE part 1: motion length aligned to cached timestamps ---
    print("[STEP] fidelity: per-video len(motion)==len(ts) assertion", flush=True)
    mismatches = []
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        ts = d.get('timestamps')
        mc = motion_cache.get(name)
        if mc is None or mc['motion'] is None or ts is None:
            continue
        if len(mc['motion']) != len(ts):
            mismatches.append((name, int(len(mc['motion'])), int(len(ts))))
    fidelity_len_ok = (len(mismatches) == 0)
    if mismatches:
        print(f"[FIDELITY] LEN MISMATCH on {len(mismatches)} videos: {mismatches[:5]}")

    raw_tm = run_eval(V114_DIR)
    print(f"[SANITY] v114 raw-baseline mAP={raw_tm['mAP']:.5f}", flush=True)

    q_dir = ROOT / 'results/batch_eval_v119_qonly'
    build_dir(q_dir, apply_qscore=True, melc_mode=None, motion_cache=motion_cache)
    q_tm = run_eval(q_dir)
    print(f"[ANCHOR] Q-only mAP={q_tm['mAP']:.5f} (target {Q_ON_ANCHOR}, "
          f"|d|={abs(q_tm['mAP']-Q_ON_ANCHOR):.5f})", flush=True)
    fidelity_anchor_ok = abs(q_tm['mAP'] - Q_ON_ANCHOR) <= FIDELITY_TOL
    fidelity_ok = fidelity_len_ok and fidelity_anchor_ok

    diag = diagnostics(motion_cache)
    print(f"[FLAT-CLIP] Q n-var mean={diag['flat_clip']['q_n_var_mean']} "
          f"(near-inert={diag['flat_clip']['q_near_inert']}), "
          f"MELC n-var mean={diag['flat_clip']['melc_n_var_mean']}", flush=True)

    results = {
        'raw_baseline_mAP': raw_tm['mAP'],
        'q_only_anchor_mAP': q_tm['mAP'],
        'q_only_AP': {k: q_tm[k] for k in ('AP@0.3', 'AP@0.5', 'AP@0.7')},
        'fidelity_len_ok': fidelity_len_ok,
        'fidelity_anchor_ok': fidelity_anchor_ok,
        'fidelity_ok': fidelity_ok,
        'len_mismatches': mismatches,
        'multi_det_videos': diag['multi_det_videos'],
        'tsm_available': diag['tsm_available'],
        'flat_clip_diag': diag['flat_clip'],
        'arms': {},
    }
    print(f"\n{'arm(stack Q+MELC)':<22} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} "
          f"{'AP@0.7':>9} {'d_vs_Q':>9} {'|r|Q':>6} {'flatD':>8} {'msalD':>8} {'var>0':>7}")
    for mode in ('oic', 'logoic', 'catgate'):
        outd = ROOT / f'results/batch_eval_v119_{mode}'
        build_dir(outd, apply_qscore=True, melc_mode=mode, motion_cache=motion_cache)
        tm = run_eval(outd)
        delta = tm['mAP'] - q_tm['mAP']
        broad = (tm['AP@0.3'] >= q_tm['AP@0.3'] and tm['AP@0.5'] >= q_tm['AP@0.5']
                 and tm['AP@0.7'] >= q_tm['AP@0.7'])
        strat = stratified_delta(motion_cache, mode)
        flat_ok = strat['flat_clip']['delta'] >= -1e-9  # no net flat-CLIP regression
        # diagnostics: catgate reuses the plain-oic corr/variance (same raw signal)
        dg = diag['per_arm'].get('oic' if mode == 'catgate' else mode)
        corr_q = dg['corr_vs_q']
        corr_ok = corr_q < CORR_GATE  # C mandate #1 HARD ABORT
        var_ok = dg['var_nonzero_videos'] > (diag['multi_det_videos'] / 2.0)
        passes = (fidelity_ok and delta >= DELTA_GATE and broad and corr_ok
                  and var_ok and flat_ok)
        results['arms'][mode] = {
            'mAP': tm['mAP'], 'AP@0.3': tm['AP@0.3'], 'AP@0.5': tm['AP@0.5'],
            'AP@0.7': tm['AP@0.7'], 'delta_vs_Q': delta,
            'corr_vs_q': corr_q, 'corr_vs_tsm': dg['corr_vs_tsm'],
            'var_nonzero_videos': dg['var_nonzero_videos'],
            'stratified': strat,
            'broad_based': broad, 'corr_ok': corr_ok, 'var_ok': var_ok,
            'flat_no_regress': flat_ok, 'passes': passes,
        }
        print(f"{mode:<22} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f} {delta:>+9.5f} {corr_q:>6.3f} "
              f"{strat['flat_clip']['delta']:>+8.4f} {strat['motion_salient']['delta']:>+8.4f} "
              f"{dg['var_nonzero_videos']:>3d}/{diag['multi_det_videos']:<2d} pass={passes}",
              flush=True)

    # --- PRE-REGISTERED VERDICT ---
    # catgate is DIAGNOSTIC ONLY (no live gating flag was implemented — A's category
    # gate was deliberately dropped); only ungated oic / logoic are promotable live.
    winners = [(m, results['arms'][m]['delta_vs_Q']) for m in ('oic', 'logoic')
               if results['arms'][m]['passes']]
    winner = max(winners, key=lambda x: x[1])[0] if winners else None
    results['winner'] = winner
    results['launch_permitted'] = (winner is not None) and fidelity_ok
    # exact live flags for the winner (control keeps --qscore_rerank only)
    flag_map = {'oic': '--qscore_rerank --melc_rerank',
                'logoic': '--qscore_rerank --melc_rerank --melc_weighted'}
    results['winner_eval_flags'] = flag_map.get(winner) if winner else None
    print(f"\n[VERDICT] winner={winner}  launch_permitted={results['launch_permitted']}")
    if not results['launch_permitted']:
        print("[VERDICT] no arm clears every sub-gate (or fidelity failed) -> "
              "ship flags OFF, REVERT to v114.")

    json.dump(results, open(OUT_JSON, 'w'), indent=2)
    print(f"[WROTE] {OUT_JSON}")


if __name__ == '__main__':
    main()
