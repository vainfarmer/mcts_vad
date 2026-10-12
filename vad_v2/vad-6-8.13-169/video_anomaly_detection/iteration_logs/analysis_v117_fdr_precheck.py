"""
v117 offline bake-off — FDR: two orthogonal per-detection CLIP-FEATURE re-rank arms.

Deterministically re-scores v114's OWN real cached detections
(results/batch_eval_v114_full) through TWO independent post-hoc re-rank arms and
promotes EXACTLY ONE to the single live A/B. Faithful in-vivo (mAP depends ONLY on
global confidence ordering; re-ranking cannot change boundaries/count/order -> NOT a
candidate-pool simulation, FM6 does not apply). Both arms read the UNCONTAMINATED
clip_features (unit-norm, timestamp-aligned) that NO committed re-rank touches
(frame_scorer L106-110 floods the fused curve with np.maximum(base,conf), so Q-score
v111 reads a confidence artifact; clip_features is the orthogonal axis).

  ARM-SLM (category-SEMANTIC text margin):
    margin(t) = F[t]·T[c_idx] - F[t]·T[-1]   (Normal-subtracted; T[-1]=Normal proto)
    slm_raw   = mean(margin[in]) - mean(margin[out])          (AutoLoc OIC on margin)
    c unknown/Normal/absent or empty in_mask -> worst-value sentinel (benign demotion)

  ARM-TFO (category-AGNOSTIC angular visual self-deviance):
    bg  = L2norm(per-dim MEDIAN of all frame features)   (median => anomaly minority
                                                           cannot pull the prototype)
    d_t = 1 - cos(F[t], bg)                               (angular; magnitude is dead
                                                           on unit-norm CLIP -> sidesteps
                                                           the v110 VideoMAE saturation)
    tfo_raw = mean(d[in])   (ABSOLUTE within-window deviance, NOT a contrast form, to
                             stay orthogonal to Q-score peak_contrast)
    empty/misaligned in_mask -> worst-value sentinel

Each arm: within-video min-max normalize -> n in [0,1]; conf' = min(1, conf*(base+(1-base)*n)),
base=0.5. Byte-identical structure to pipeline._slm_rerank / _tfo_rerank.

C's THREE MANDATED TIGHTENINGS (pre-registered, before any live smoke):
  (1) HARNESS-FIDELITY ABORT-GATE (hard): reproduce BOTH the raw-baseline mAP (cached
      conf as-is) AND the Q-only round-trip anchor to |Δ| <= 0.0005 of 0.15632. Because
      the harness RE-EXTRACTS clip_features, a silent frame-sampling/feature-drift bug
      corrupts both arms while leaving the anchor (reads CACHED frame_scores) intact ->
      ALSO assert re-extracted len(F) == len(ts_cached) per video and log divergences.
      If anchor or raw baseline does not reproduce -> ABORT (do not report arm deltas).
  (2) QUANTIFIED ORTHOGONALITY KILL-GATE: per-detection Pearson corr between each arm's
      within-video-normalized signal and the Q-score signal, pooled over multi-detection
      videos. REJECT an arm as Q-redundant if |corr| > 0.6, OR if the stack does not
      strictly beat BOTH Q-only AND arm-only. Correlation reported for both arms.
  (3) (post-activation invariant re-confirm is a LIVE step, done by the eval agent, not here.)

PRE-REGISTERED ADJUDICATION (no post-hoc discretion): the LIVE candidate = the single arm
that (a) clears offline stack delta >= +0.005 over the 0.15632 Q-on anchor AND (b) rises
broad-based across AP@0.3/0.5/0.7 AND (c) is the least Q-correlated of the qualifying arms.
If BOTH qualify -> take higher & more Q-orthogonal; the other ships inert. If NEITHER ->
both ship flag-off, loop REVERTS to v114.

NOTE: instantiates real CLIP (heavy). Run it; do not edit-and-guess. VAD_DEVICE=cuda|mps|cpu.
"""
import os, sys, json, glob, shutil
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))

from evaluate import evaluate            # noqa: E402
from config import VADConfig             # noqa: E402
from factory import build_vad            # noqa: E402
from llm_client import DashScopeLLMClient  # noqa: E402

V114_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
V114_ANCHOR_MAP = 0.15632
FID_TOL = 0.0005
QBASE = 0.5
SLM_BASE = 0.5
TFO_BASE = 0.5
CORR_MAX = 0.6
COMMIT_BAR = 0.005
UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def _to_np(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, dtype=float)


# --------- frozen Q-score (byte-identical to pipeline._qscore_rerank) ---------
def qn_from_curve(fs, ts, anoms):
    fs = np.asarray(fs, dtype=float); ts = np.asarray(ts, dtype=float)
    if len(fs) < 3 or len(ts) != len(fs):
        return np.ones(len(anoms))
    grad = np.abs(np.gradient(fs))
    vid_range = fs.max() - fs.min() + 1e-9
    qs = []
    for a in anoms:
        t_s, t_e = a['t_start'], a['t_end']
        in_mask = (ts >= t_s) & (ts <= t_e)
        if in_mask.sum() < 1:
            qs.append(0.0); continue
        peak = fs[in_mask].max()
        peak_idx = np.where(in_mask)[0][np.argmax(fs[in_mask])]
        out_mask = ~in_mask
        out_mean = fs[out_mask].mean() if out_mask.any() else fs.min()
        contrast = float(np.clip((peak - out_mean) / vid_range, 0, 1))
        i_s = int(np.argmin(np.abs(ts - t_s))); i_e = int(np.argmin(np.abs(ts - t_e)))
        sharp = float(np.clip((grad[i_s] + grad[i_e]) / (2 * (grad.max() + 1e-9)), 0, 1))
        center_t = 0.5 * (t_s + t_e); dur = max(t_e - t_s, 1e-6)
        center = float(np.clip(1 - 2 * abs(ts[peak_idx] - center_t) / dur, 0, 1))
        qs.append((sharp + contrast + center) / 3.0)
    qs = np.array(qs)
    if len(qs) > 1 and qs.max() - qs.min() > 1e-9:
        return (qs - qs.min()) / (qs.max() - qs.min())
    return np.ones_like(qs)


# --------- SLM raw margin (mirrors pipeline._slm_rerank) ---------
def slm_raw(F, T, ts, anoms):
    raw, sent = [], []
    for a in anoms:
        c = a.get('anomaly_type')
        if c is None or c == 'Normal' or c not in UCF_CRIME_CATEGORIES:
            raw.append(None); sent.append(True); continue
        c_idx = UCF_CRIME_CATEGORIES.index(c)
        margin = F @ T[c_idx] - F @ T[-1]
        in_mask = (ts >= a['t_start']) & (ts <= a['t_end'])
        if in_mask.sum() < 1:
            raw.append(None); sent.append(True); continue
        out_mask = ~in_mask
        s_out = float(margin[out_mask].mean()) if out_mask.any() else float(margin.min())
        raw.append(float(margin[in_mask].mean()) - s_out); sent.append(False)
    return raw, sent


# --------- TFO raw angular deviance (mirrors pipeline._tfo_rerank) ---------
def tfo_raw(F, ts, anoms):
    bg = np.median(F, axis=0)
    bg = bg / (np.linalg.norm(bg) + 1e-8)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    d = 1.0 - (Fn @ bg)
    raw, sent = [], []
    for a in anoms:
        in_mask = (ts >= a['t_start']) & (ts <= a['t_end'])
        if in_mask.sum() < 1:
            raw.append(None); sent.append(True); continue
        raw.append(float(d[in_mask].mean())); sent.append(False)
    return raw, sent


def normalize_with_sentinel(raw):
    real = [r for r in raw if r is not None]
    worst = min(real) if real else 0.0
    arr = np.array([(worst if r is None else r) for r in raw], dtype=float)
    if len(arr) > 1 and arr.max() - arr.min() > 1e-9:
        n = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        n = np.ones_like(arr)
    return arr, n


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def write_dir(out_dir, per_video, conf_key):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, pv in per_video.items():
        d = pv['raw_json']
        for a, cnf in zip(d.get('anomalies', []), pv[conf_key]):
            a['confidence'] = float(cnf)
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def main():
    device = os.environ.get('VAD_DEVICE', 'cpu')
    cfg = VADConfig.from_args(type('A', (), {})())
    cfg.anomaly_categories = UCF_CRIME_CATEGORIES
    cfg.device = device
    vad = build_vad(cfg, DashScopeLLMClient(model="qwen-vl-max"))
    expert = vad.anomaly_expert
    assert expert.model is not None, "CLIP failed to load; cannot run faithful FDR precheck"
    T = _to_np(expert.text_features)
    print(f"[SETUP] device={device} T.shape={T.shape} (last row = Normal prototype)")

    files = sorted(glob.glob(str(V114_DIR / '*.json')))
    per_video = {}
    fidelity_fail = 0
    len_divergence = 0
    cat_counter = Counter()
    n_det_total = 0
    slm_sent_total = 0; tfo_sent_total = 0
    slm_spread_zero = 0; tfo_spread_zero = 0; multi_det_videos = 0
    # pooled per-detection signals for orthogonality correlation (multi-det videos only)
    pool_q, pool_slm, pool_tfo = [], [], []

    for i, jf in enumerate(files):
        name = Path(jf).stem
        d = json.load(open(jf))
        anoms = d.get('anomalies', [])
        fs = d.get('frame_scores'); ts_cached = d.get('timestamps')
        cached_conf = [a.get('confidence', 0.0) for a in anoms]
        pv = {'raw_json': d, 'cached': list(cached_conf), 'q_only': list(cached_conf),
              'q_slm': list(cached_conf), 'q_tfo': list(cached_conf),
              'slm_only': list(cached_conf), 'tfo_only': list(cached_conf)}

        if not anoms:
            per_video[name] = pv
            print(f"[{i+1}/{len(files)}] {name}: no detections (skip)")
            continue
        for a in anoms:
            cat_counter[a.get('anomaly_type')] += 1
        n_det_total += len(anoms)

        try:
            vdata = vad.preprocessor.process(d['video_path'])
            F = expert.precompute_video_features(vdata['all_frames'])
            ts_ext = np.asarray(vdata['timestamps'], dtype=float)
        except Exception as e:
            fidelity_fail += 1
            per_video[name] = pv
            print(f"[{i+1}/{len(files)}] {name}: EXTRACT-FAIL {e} -> pass-through")
            continue
        F = None if F is None else _to_np(F)
        lf = 0 if F is None else len(F)
        lc = len(ts_cached) if ts_cached else 0
        print(f"[{i+1}/{len(files)}] {name}: len(F)={lf} len(ts_ext)={len(ts_ext)} "
              f"len(ts_cached)={lc} n_anom={len(anoms)}")
        if lc and lf and lf != lc:
            len_divergence += 1
            print(f"    -> LEN-DIVERGENCE: re-extracted len(F)={lf} != len(ts_cached)={lc}")
        if F is None or F.ndim != 2 or len(ts_ext) != len(F) or T.shape[0] < 2:
            fidelity_fail += 1
            per_video[name] = pv
            print(f"    -> FIDELITY GUARD hit (F/ts mismatch or None) -> pass-through")
            continue

        rm, s_sent = slm_raw(F, T, ts_ext, anoms)
        rt, t_sent = tfo_raw(F, ts_ext, anoms)
        slm_sent_total += sum(s_sent); tfo_sent_total += sum(t_sent)
        slm_arr, slm_n = normalize_with_sentinel(rm)
        tfo_arr, tfo_n = normalize_with_sentinel(rt)

        qn = qn_from_curve(fs, ts_cached, anoms) if (fs and ts_cached) else np.ones(len(anoms))

        if len(anoms) > 1:
            multi_det_videos += 1
            if float(slm_arr.max() - slm_arr.min()) < 1e-6:
                slm_spread_zero += 1
                print(f"    -> SLM KILL-GATE: spread ~0")
            if float(tfo_arr.max() - tfo_arr.min()) < 1e-6:
                tfo_spread_zero += 1
                print(f"    -> TFO KILL-GATE: spread ~0")
            pool_q.extend(list(qn)); pool_slm.extend(list(slm_n)); pool_tfo.extend(list(tfo_n))

        q_mult = QBASE + (1 - QBASE) * qn
        raw_conf = np.array(cached_conf) / q_mult

        pv['q_only'] = list(np.minimum(1.0, raw_conf * q_mult))
        pv['q_slm'] = list(np.minimum(1.0, np.array(cached_conf) * (SLM_BASE + (1 - SLM_BASE) * slm_n)))
        pv['q_tfo'] = list(np.minimum(1.0, np.array(cached_conf) * (TFO_BASE + (1 - TFO_BASE) * tfo_n)))
        pv['slm_only'] = list(np.minimum(1.0, raw_conf * (SLM_BASE + (1 - SLM_BASE) * slm_n)))
        pv['tfo_only'] = list(np.minimum(1.0, raw_conf * (TFO_BASE + (1 - TFO_BASE) * tfo_n)))
        per_video[name] = pv

    # ---------------- reports ----------------
    print("\n================ C-MANDATED CHECKS ================")
    print(f"[1 FEATURE-FIDELITY] videos hitting extract/len-guard: {fidelity_fail}/{len(files)} "
          f"| len(F)!=len(ts_cached) divergences: {len_divergence}")
    print(f"[2 CATEGORY-RESOLUTION] distribution across {n_det_total} detections:")
    for c, n in cat_counter.most_common():
        print(f"    {str(c):<18} {n}")
    print(f"[3 SENTINEL] SLM sentinel: {slm_sent_total}/{n_det_total} "
          f"({slm_sent_total/max(n_det_total,1):.1%}) | TFO sentinel: {tfo_sent_total}/{n_det_total} "
          f"({tfo_sent_total/max(n_det_total,1):.1%})")
    print(f"[KILL-GATE variance] multi-det videos={multi_det_videos} | "
          f"SLM spread~0: {slm_spread_zero} | TFO spread~0: {tfo_spread_zero}")

    def pearson(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    corr_slm = pearson(pool_slm, pool_q)
    corr_tfo = pearson(pool_tfo, pool_q)
    print(f"[ORTHOGONALITY] per-detection Pearson vs Q-score (pooled multi-det, "
          f"n={len(pool_q)}): SLM={corr_slm:+.3f} TFO={corr_tfo:+.3f} (reject if |corr|>{CORR_MAX})")

    # ---------------- fidelity abort-gate ----------------
    print("\n================ FIDELITY ABORT-GATE ================")
    outc = ROOT / "results/batch_eval_v117_precheck_cached"
    outq = ROOT / "results/batch_eval_v117_precheck_qonly"
    write_dir(outc, per_video, 'cached')
    write_dir(outq, per_video, 'q_only')
    m_cached = run_eval(outc)['mAP']
    m_qonly = run_eval(outq)['mAP']
    print(f"raw-baseline (cached as-is) mAP = {m_cached:.5f} (expect {V114_ANCHOR_MAP})")
    print(f"Q-only round-trip anchor  mAP = {m_qonly:.5f} (expect {V114_ANCHOR_MAP})")
    abort = (abs(m_cached - V114_ANCHOR_MAP) > FID_TOL) or (abs(m_qonly - V114_ANCHOR_MAP) > FID_TOL)
    if abort:
        print(f"!! FIDELITY ABORT: anchor/raw did not reproduce within |Δ|<={FID_TOL}. "
              f"Do NOT trust arm deltas. Aborting.")
        json.dump({'aborted': True, 'm_cached': m_cached, 'm_qonly': m_qonly,
                   'fidelity_fail': fidelity_fail, 'len_divergence': len_divergence},
                  open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v117_fdr_precheck.json', 'w'),
                  indent=2)
        return

    # ---------------- A/B eval ----------------
    print("\n================ DETERMINISTIC OFFLINE A/B ================")
    arms = {
        'Q-only (anchor)': 'q_only',
        'SLM-only':        'slm_only',
        'TFO-only':        'tfo_only',
        'Q+SLM':           'q_slm',
        'Q+TFO':           'q_tfo',
    }
    results = {}
    print(f"{'arm':<18} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  {'delta':>9}")
    base_map = m_qonly
    for label, key in arms.items():
        outd = ROOT / f"results/batch_eval_v117_precheck_{key}"
        write_dir(outd, per_video, key)
        tm = run_eval(outd)
        results[label] = {k: tm[k] for k in ('mAP', 'AP@0.3', 'AP@0.5', 'AP@0.7')}
        delta = tm['mAP'] - base_map
        print(f"{label:<18} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f}  {delta:>+9.5f}")

    q_only = results['Q-only (anchor)']['mAP']

    def evaluate_arm(name, stack_label, only_label, corr):
        stack = results[stack_label]; only = results[only_label]['mAP']
        delta = stack['mAP'] - q_only
        beats_both = (stack['mAP'] > q_only) and (stack['mAP'] > only)
        broad = (stack['AP@0.3'] >= results['Q-only (anchor)']['AP@0.3'] and
                 stack['AP@0.5'] >= results['Q-only (anchor)']['AP@0.5'] and
                 stack['AP@0.7'] >= results['Q-only (anchor)']['AP@0.7'])
        ortho = abs(corr) <= CORR_MAX
        qualifies = (delta >= COMMIT_BAR) and broad and beats_both and ortho
        print(f"\n[{name}] delta={delta:+.5f} beats_both(Q-only & arm-only)={beats_both} "
              f"broad(AP@0.3/0.5/0.7 up)={broad} |corr|<= {CORR_MAX}={ortho} "
              f"-> {'QUALIFIES' if qualifies else 'does NOT qualify'}")
        return {'delta': delta, 'beats_both': beats_both, 'broad': broad,
                'ortho': ortho, 'corr': corr, 'qualifies': qualifies, 'mAP': stack['mAP']}

    print("\n================ VERDICT (pre-registered adjudication) ================")
    v_slm = evaluate_arm('SLM', 'Q+SLM', 'SLM-only', corr_slm)
    v_tfo = evaluate_arm('TFO', 'Q+TFO', 'TFO-only', corr_tfo)

    qualifiers = [(n, v) for n, v in (('SLM', v_slm), ('TFO', v_tfo)) if v['qualifies']]
    if not qualifiers:
        winner = None
        print("\nNEITHER arm qualifies -> both ship flag-off; loop REVERTS to v114 (pre-registered).")
    elif len(qualifiers) == 1:
        winner = qualifiers[0][0]
        print(f"\nWINNER = {winner} (only qualifier). Live A/B: --qscore_rerank --{winner.lower()}_rerank")
    else:
        # both qualify -> higher mAP; tie -> more Q-orthogonal (lower |corr|)
        (n0, v0), (n1, v1) = qualifiers
        if abs(v0['mAP'] - v1['mAP']) < 1e-6:
            winner = n0 if abs(v0['corr']) <= abs(v1['corr']) else n1
        else:
            winner = n0 if v0['mAP'] > v1['mAP'] else n1
        print(f"\nBOTH qualify -> WINNER = {winner} (higher/more-orthogonal). "
              f"Loser ships inert. Live A/B: --qscore_rerank --{winner.lower()}_rerank")

    json.dump({'results': results, 'q_only': q_only, 'm_cached': m_cached,
               'corr_slm': corr_slm, 'corr_tfo': corr_tfo,
               'verdict_slm': v_slm, 'verdict_tfo': v_tfo, 'winner': winner,
               'fidelity_fail': fidelity_fail, 'len_divergence': len_divergence,
               'slm_sentinel': slm_sent_total, 'tfo_sentinel': tfo_sent_total,
               'n_det_total': n_det_total, 'multi_det_videos': multi_det_videos},
              open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v117_fdr_precheck.json', 'w'),
              indent=2)


if __name__ == '__main__':
    main()
