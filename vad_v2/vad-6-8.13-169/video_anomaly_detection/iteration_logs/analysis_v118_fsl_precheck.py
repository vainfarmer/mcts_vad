"""
v118 offline pre-check — FSL (Feature-space Structural Localization) re-rank bake-off.

HARNESS-FIRST gate (loop-17 root-cause fix): this JSON MUST exist and PASS before
any live A/B is permitted. It clones analysis_v111_qscore_precheck.py (the trusted
v108-reproducing tool) and extends to v114 REAL detections.

Faithful in-vivo protocol (FM6-safe): a re-rank cannot change detections / boundaries
/ count, so re-scoring v114's OWN real final detections == the live output. mAP depends
only on the global confidence ordering. The one novel unproven step is offline CLIP
re-extraction byte-matching live pipeline.py L917 — caught by the fidelity abort-gate
(per-video assert len(re-extracted F) == len(cached ts)).

Pipeline of anchors (mirrors v111: v114 JSONs are RAW = qscore NOT baked in, verified
by 0 [QSCORE-GATE] markers in the v114 log and v114 as-is mAP=0.15013):
  raw-baseline  = v114 as-is                        (~0.15013)
  Q-only anchor = raw + committed v111 Q-score      (fidelity target ~0.15632)
  arm stack     = raw + Q-score + FSL(arm)           (the thing under test)

Three arms scored (zero live budget):
  fbc  = (sbd + ebd + cohesion)/3
  sep  = cohesion - cross
  comp = mean( within_norm((sbd+ebd)/2), within_norm(sep) )
Each then follows the EXACT v111 protocol: within-video min-max -> n; single-detection
-> n=1 (inert); conf' = min(1, conf*(base+(1-base)*n)), base=0.5.

PRE-REGISTERED VERDICT (C's binding launch gate): pick the SINGLE arm with the largest
offline delta that ALSO satisfies delta >= +0.005 over the 0.15632 Q-on anchor AND
broad-based across AP@0.3/0.5/0.7 AND |corr| < 0.6 vs BOTH Q and TSM AND beats BOTH
Q-only and that-arm-only, AND nonzero within-video raw variance on the majority of
multi-detection videos. If NO arm clears every sub-gate -> ship all flags OFF (revert).

Run:  VAD_DEVICE=mps python3 iteration_logs/analysis_v118_fsl_precheck.py
"""
import os
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
from anomaly_expert import VideoAnomalyExpert  # noqa
from pipeline import _to_np, _compute_tsm_novelty  # noqa: reuse live helpers verbatim

V114_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
FEAT_CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v118_fsl_feats.pkl'
OUT_JSON = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v118_fsl_precheck.json'

Q_ON_ANCHOR = 0.15632   # committed v114 Q-on baseline
FIDELITY_TOL = 0.0005
DELTA_GATE = 0.005
CORR_GATE = 0.6
FSL_BASE = 0.5
QS_BASE = 0.5


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
# FSL raw axes — byte-matches VADPipeline._fsl_rerank
# ----------------------------------------------------------------------------
def fsl_axes(F, ts, anoms):
    """Returns (bd_raw, coh_raw, sep_raw) per detection. F: (N,D) unit-norm rows."""
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    N = len(Fn)
    bd_raw, coh_raw, sep_raw = [], [], []
    for a in anoms:
        t_s, t_e = a['t_start'], a['t_end']
        in_mask = (ts >= t_s) & (ts <= t_e)
        n_in = int(in_mask.sum())
        out_mask = ~in_mask
        if n_in < 2 or not out_mask.any():
            bd_raw.append(0.0); coh_raw.append(0.0); sep_raw.append(0.0)
            continue
        I_idx = np.where(in_mask)[0]
        i_s, i_e = int(I_idx[0]), int(I_idx[-1])
        c_in = Fn[in_mask].mean(axis=0)
        c_out = Fn[out_mask].mean(axis=0)
        cohesion = float(np.dot(c_in, c_in))
        cross = float(np.dot(c_in, c_out))
        sep = cohesion - cross
        m_s = min(n_in // 2, i_s)
        if m_s >= 1:
            inner_s = Fn[i_s:i_s + m_s].mean(axis=0)
            outer_s = Fn[i_s - m_s:i_s].mean(axis=0)
            sbd = float(np.clip(1.0 - float(np.dot(inner_s, outer_s)), 0.0, 2.0)) / 2.0
        else:
            sbd = 0.0
        m_e = min(n_in // 2, N - 1 - i_e)
        if m_e >= 1:
            inner_e = Fn[i_e - m_e + 1:i_e + 1].mean(axis=0)
            outer_e = Fn[i_e + 1:i_e + 1 + m_e].mean(axis=0)
            ebd = float(np.clip(1.0 - float(np.dot(inner_e, outer_e)), 0.0, 2.0)) / 2.0
        else:
            ebd = 0.0
        bd_raw.append((sbd + ebd) / 2.0)
        coh_raw.append(cohesion)
        sep_raw.append(sep)
    return np.asarray(bd_raw), np.asarray(coh_raw), np.asarray(sep_raw)


def arm_raw(mode, bd, coh, sep):
    if mode == 'fbc':
        return (2.0 * bd + coh) / 3.0
    if mode == 'sep':
        return sep
    if mode == 'comp':
        return 0.5 * (within_norm(bd) + within_norm(sep))
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# Feature re-extraction (same code path as pipeline.py L917), cached to disk
# ----------------------------------------------------------------------------
def build_feature_cache():
    if FEAT_CACHE.exists():
        with open(FEAT_CACHE, 'rb') as f:
            return pickle.load(f)
    device = os.environ.get('VAD_DEVICE', 'mps')
    cfg = VADConfig()
    pp = cfg.preprocess; cl = cfg.clip
    pre = VideoPreprocessor(fps=pp.fps, segment_duration=pp.segment_duration,
                            motion_threshold=pp.motion_threshold,
                            min_motion_interval=pp.min_motion_interval,
                            segment_overlap=pp.segment_overlap)
    expert = VideoAnomalyExpert(anomaly_categories=cfg.anomaly_categories or ['anomaly'],
                                clip_model_name=cl.model_name,
                                clip_pretrained=cl.pretrained, device=device)
    cache = {}
    jfs = sorted(glob.glob(str(V114_DIR / '*.json')))
    for k, jf in enumerate(jfs):
        name = Path(jf).stem
        d = json.load(open(jf))
        vp = d.get('video_path')
        if not vp or not Path(vp).exists():
            print(f"[FEAT] {k+1}/{len(jfs)} {name}: video missing -> skip")
            cache[name] = None
            continue
        vd = pre.process(vp)
        feats = expert.precompute_video_features(vd['all_frames'])
        F = _to_np(feats)
        cache[name] = {'F': None if F is None else np.asarray(F, dtype=np.float32),
                       'ts': np.asarray(vd['timestamps'], dtype=float)}
        print(f"[FEAT] {k+1}/{len(jfs)} {name}: F={None if F is None else F.shape}", flush=True)
    with open(FEAT_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    return cache


# ----------------------------------------------------------------------------
# Build re-ranked result dirs
# ----------------------------------------------------------------------------
def build_dir(out_dir, apply_qscore, fsl_mode, feats):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if apply_qscore and fs and ts and anoms:
            qs = np.array([qscore_raw(fs, ts, a['t_start'], a['t_end']) for a in anoms])
            qn = within_norm(qs)
            for a, q in zip(anoms, qn):
                a['confidence'] = min(1.0, a.get('confidence', 0.0) * (QS_BASE + (1 - QS_BASE) * float(q)))
        if fsl_mode and anoms:
            fc = feats.get(name)
            if fc is not None and fc['F'] is not None and len(fc['F']) >= 3:
                Fts = fc['ts']
                bd, coh, sep = fsl_axes(fc['F'], Fts, anoms)
                raw = arm_raw(fsl_mode, bd, coh, sep)
                n = within_norm(raw)
                for a, nv in zip(anoms, n):
                    a['confidence'] = min(1.0, a['confidence'] * (FSL_BASE + (1 - FSL_BASE) * float(nv)))
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


# ----------------------------------------------------------------------------
# Orthogonality + variance diagnostics (pooled per-detection, within-video normed)
# ----------------------------------------------------------------------------
def diagnostics(feats):
    """Returns per-arm pooled Pearson |corr| vs Q and vs TSM, and within-video
    raw-variance stats, computed on multi-detection videos only."""
    pooled = {'q': [], 'tsm': [],
              'fbc': [], 'sep': [], 'comp': []}
    var_nonzero = {'fbc': 0, 'sep': 0, 'comp': 0}
    var_total = 0
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        fc = feats.get(name)
        if not (fs and ts and anoms) or len(anoms) < 2:
            continue
        if fc is None or fc['F'] is None or len(fc['F']) < 3:
            continue
        var_total += 1
        Fts = fc['ts']
        # Q per detection (within-video normed)
        qs = np.array([qscore_raw(fs, ts, a['t_start'], a['t_end']) for a in anoms])
        qn = within_norm(qs)
        # TSM novelty per detection (mean novelty inside window), within-normed
        nov = _compute_tsm_novelty(fc['F'])
        tsm_pd = []
        for a in anoms:
            m = (Fts >= a['t_start']) & (Fts <= a['t_end'])
            tsm_pd.append(float(nov[m].mean()) if m.any() else 0.0)
        tsm_n = within_norm(np.array(tsm_pd))
        bd, coh, sep = fsl_axes(fc['F'], Fts, anoms)
        for mode in ('fbc', 'sep', 'comp'):
            raw = arm_raw(mode, bd, coh, sep)
            if float(np.var(raw)) > 1e-12:
                var_nonzero[mode] += 1
            pooled[mode].extend(within_norm(raw).tolist())
        pooled['q'].extend(qn.tolist())
        pooled['tsm'].extend(tsm_n.tolist())

    def pear(a, b):
        a = np.asarray(a); b = np.asarray(b)
        if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    out = {'multi_det_videos': var_total, 'per_arm': {}}
    for mode in ('fbc', 'sep', 'comp'):
        out['per_arm'][mode] = {
            'corr_vs_q': abs(pear(pooled[mode], pooled['q'])),
            'corr_vs_tsm': abs(pear(pooled[mode], pooled['tsm'])),
            'var_nonzero_videos': var_nonzero[mode],
        }
    return out


def main():
    print("[STEP] building CLIP feature cache (one-time; VAD_DEVICE=%s)..."
          % os.environ.get('VAD_DEVICE', 'mps'), flush=True)
    feats = build_feature_cache()

    # --- FIDELITY ABORT-GATE ---
    print("[STEP] fidelity: per-video len(F)==len(ts) assertion", flush=True)
    mismatches = []
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        ts = d.get('timestamps')
        fc = feats.get(name)
        if fc is None or fc['F'] is None or ts is None:
            continue
        if len(fc['F']) != len(ts):
            mismatches.append((name, len(fc['F']), len(ts)))
    fidelity_len_ok = (len(mismatches) == 0)
    if mismatches:
        print(f"[FIDELITY] LEN MISMATCH on {len(mismatches)} videos: {mismatches[:5]}")

    raw_tm = run_eval(V114_DIR)
    print(f"[SANITY] v114 raw-baseline mAP={raw_tm['mAP']:.5f}", flush=True)

    q_dir = ROOT / 'results/batch_eval_v118_qonly'
    build_dir(q_dir, apply_qscore=True, fsl_mode=None, feats=feats)
    q_tm = run_eval(q_dir)
    print(f"[ANCHOR] Q-only mAP={q_tm['mAP']:.5f} (target {Q_ON_ANCHOR}, "
          f"|d|={abs(q_tm['mAP']-Q_ON_ANCHOR):.5f})", flush=True)
    fidelity_anchor_ok = abs(q_tm['mAP'] - Q_ON_ANCHOR) <= FIDELITY_TOL
    fidelity_ok = fidelity_len_ok and fidelity_anchor_ok

    diag = diagnostics(feats)

    results = {
        'raw_baseline_mAP': raw_tm['mAP'],
        'q_only_anchor_mAP': q_tm['mAP'],
        'q_only_AP': {k: q_tm[k] for k in ('AP@0.3', 'AP@0.5', 'AP@0.7')},
        'fidelity_len_ok': fidelity_len_ok,
        'fidelity_anchor_ok': fidelity_anchor_ok,
        'fidelity_ok': fidelity_ok,
        'len_mismatches': mismatches,
        'multi_det_videos': diag['multi_det_videos'],
        'arms': {},
    }
    print(f"\n{'arm(stack Q+FSL)':<22} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} "
          f"{'AP@0.7':>9} {'d_vs_Q':>9} {'|r|Q':>6} {'|r|TSM':>7} {'var>0':>7}")
    for mode in ('fbc', 'sep', 'comp'):
        outd = ROOT / f'results/batch_eval_v118_{mode}'
        build_dir(outd, apply_qscore=True, fsl_mode=mode, feats=feats)
        tm = run_eval(outd)
        delta = tm['mAP'] - q_tm['mAP']
        dg = diag['per_arm'][mode]
        broad = (tm['AP@0.3'] >= q_tm['AP@0.3'] and tm['AP@0.5'] >= q_tm['AP@0.5']
                 and tm['AP@0.7'] >= q_tm['AP@0.7'])
        # arm-only (FSL without Q) for the beats-both gate
        aod = ROOT / f'results/batch_eval_v118_{mode}_armonly'
        build_dir(aod, apply_qscore=False, fsl_mode=mode, feats=feats)
        arm_only_tm = run_eval(aod)
        beats_both = (tm['mAP'] > q_tm['mAP'] and tm['mAP'] > arm_only_tm['mAP'])
        corr_ok = (dg['corr_vs_q'] < CORR_GATE and dg['corr_vs_tsm'] < CORR_GATE)
        var_ok = dg['var_nonzero_videos'] > (diag['multi_det_videos'] / 2.0)
        passes = (fidelity_ok and delta >= DELTA_GATE and broad and corr_ok
                  and beats_both and var_ok)
        results['arms'][mode] = {
            'mAP': tm['mAP'], 'AP@0.3': tm['AP@0.3'], 'AP@0.5': tm['AP@0.5'],
            'AP@0.7': tm['AP@0.7'], 'delta_vs_Q': delta,
            'arm_only_mAP': arm_only_tm['mAP'],
            'corr_vs_q': dg['corr_vs_q'], 'corr_vs_tsm': dg['corr_vs_tsm'],
            'var_nonzero_videos': dg['var_nonzero_videos'],
            'broad_based': broad, 'beats_both': beats_both,
            'corr_ok': corr_ok, 'var_ok': var_ok, 'passes': passes,
        }
        print(f"{mode:<22} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f} {delta:>+9.5f} {dg['corr_vs_q']:>6.3f} "
              f"{dg['corr_vs_tsm']:>7.3f} {dg['var_nonzero_videos']:>4d}/"
              f"{diag['multi_det_videos']:<2d}  pass={passes}", flush=True)

    # --- PRE-REGISTERED VERDICT ---
    winners = [(m, results['arms'][m]['delta_vs_Q']) for m in ('fbc', 'sep', 'comp')
               if results['arms'][m]['passes']]
    winner = max(winners, key=lambda x: x[1])[0] if winners else None
    results['winner'] = winner
    results['launch_permitted'] = winner is not None
    print(f"\n[VERDICT] winner={winner}  launch_permitted={winner is not None}")
    if winner is None:
        print("[VERDICT] no arm clears every sub-gate -> ship flags OFF, REVERT to v114.")

    json.dump(results, open(OUT_JSON, 'w'), indent=2)
    print(f"[WROTE] {OUT_JSON}")


if __name__ == '__main__':
    main()
