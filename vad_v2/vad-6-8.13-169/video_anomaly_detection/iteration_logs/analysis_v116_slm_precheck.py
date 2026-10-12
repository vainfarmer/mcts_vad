"""
v116 offline pre-check — SLM (Semantic Localization Margin) confidence re-rank.

Targets the FM7 intra-video-mislocalization bottleneck with a PER-DETECTION signal
ORTHOGONAL to the v111/v114 Q-score curve geometry. SLM reads the HIGH-DIM CLIP
feature space projected onto the anomaly(c)-minus-Normal text direction (the axis the
scalar fused curve throws away and then overwrites via np.maximum(base,conf)):

  margin = F @ T[c_idx] - F @ T[-1]         # per-frame, Normal-subtracted
  slm_raw = mean(margin[in_window]) - mean(margin[out])   # plain-mean PRIMARY arm
            OR  mean(top-k inner) - mean(out)             # AutoLoc-OIC ablation arm
  within-video min-max normalize -> slm_n
  conf' = min(1, conf * (base + (1-base) * slm_n))

FAITHFUL offline protocol (mAP depends ONLY on global confidence ordering; re-ranking
cannot change detections/boundaries/count):
  * reads v114's OWN real cached detections (results/batch_eval_v114_full) — boundaries,
    count, and post-Q confidences FROZEN.
  * JSONs cache frame_scores+timestamps but NOT raw clip_features, so this harness
    DETERMINISTICALLY RE-EXTRACTS clip_features per video via the real
    AnomalyExpert.precompute_video_features (frozen CLIP + fixed frame sampling =>
    byte-reproducible, timestamp-aligned, VLM-noise-free) and rebuilds text_features by
    instantiating the real pipeline. This is A's fidelity fix over a coarse uniform-time
    proxy (which would be the FM8 config-specific trap).

The cached v114 confidences are already POST-Q (qscore_rerank committed default-ON), so:
  Q-only baseline  = cached conf                                  (== v114 mAP 0.15632)
  Q+SLM            = cached conf * SLM-multiplier                 (the committed live path)
  SLM-only         = raw conf   * SLM-multiplier   where raw = cached / Q-multiplier
                     (Q inverted from the frozen curve to isolate SLM's own contribution)

C's mandated checks (silent-failure guards the SLM-disabled anchor cannot catch):
  1. FEATURE-FIDELITY: per-video len(F) vs len(ts_extracted) vs len(ts_cached); assert
     len(F)==len(ts_extracted); count videos hitting the len-mismatch/None guard.
  2. CATEGORY-RESOLUTION: resolved-category distribution from cached 'anomaly_type'.
  3. SENTINEL-vs-MARGIN: fraction of detections whose slm_raw is the worst-value sentinel
     (unknown/Normal/empty in_mask) vs a real computed margin.
  4. OIC-vs-plain-mean reported offline only; the LIVE mechanism ships EXACTLY ONE
     aggregation, locked here before any live run (no second CLI flag).

SANITY ANCHOR: SLM disabled (round-trip raw->Q) reproduces v114 mAP=0.15632 EXACTLY.
KILL-GATE: within-video SLM spread on every multi-detection video; if the large majority
are ~0, REJECT offline (v110/v112 zero-variance trap).
COMMIT BAR: deterministic A/B delta of the winning single arm over the Q-on baseline
>= +0.005; else ship flag-off byte-identical.

NOTE: this harness instantiates real CLIP (heavy). Run it, do not edit-and-guess. Set
env VAD_DEVICE=cuda|mps|cpu (default cpu).
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
QBASE = 0.5     # committed qscore_base (config.ScoringConfig.qscore_base)
SLM_BASE = 0.5  # config.ScoringConfig.slm_base
OIC_K = 3       # AutoLoc-canonical top-k inner
UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


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


# --------- SLM raw margin per detection (mirrors pipeline._slm_rerank) ---------
def slm_raw_arms(F, T, ts, anoms):
    """Returns (raw_mean, raw_oic, sentinel_mask). raw_* pre-normalization, with
    worst-value sentinel resolved by caller (needs the video's min of reals)."""
    raw_mean, raw_oic, sent = [], [], []
    for a in anoms:
        c = a.get('anomaly_type')
        if c is None or c == 'Normal' or c not in UCF_CRIME_CATEGORIES:
            raw_mean.append(None); raw_oic.append(None); sent.append(True); continue
        c_idx = UCF_CRIME_CATEGORIES.index(c)
        margin = F @ T[c_idx] - F @ T[-1]
        t_s, t_e = a['t_start'], a['t_end']
        in_mask = (ts >= t_s) & (ts <= t_e)
        in_count = int(in_mask.sum())
        if in_count < 1:
            raw_mean.append(None); raw_oic.append(None); sent.append(True); continue
        out_mask = ~in_mask
        s_out = float(margin[out_mask].mean()) if out_mask.any() else float(margin.min())
        s_in_mean = float(margin[in_mask].mean())
        k = min(OIC_K, in_count)
        s_in_oic = float(np.sort(margin[in_mask])[-k:].mean())
        raw_mean.append(s_in_mean - s_out)
        raw_oic.append(s_in_oic - s_out)
        sent.append(False)
    return raw_mean, raw_oic, sent


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
    """conf_key selects which precomputed confidence list to write per video."""
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
    cfg = VADConfig.from_args(type('A', (), {})())  # empty ns -> committed defaults
    cfg.anomaly_categories = UCF_CRIME_CATEGORIES
    cfg.device = device
    vad = build_vad(cfg, DashScopeLLMClient(model="qwen-vl-max"))
    expert = vad.anomaly_expert
    assert expert.model is not None, "CLIP failed to load; cannot run faithful SLM precheck"
    T = expert.text_features.detach().cpu().numpy().astype(float)
    print(f"[SETUP] device={device} T.shape={T.shape} (last row = Normal prototype)")

    files = sorted(glob.glob(str(V114_DIR / '*.json')))
    per_video = {}
    fidelity_fail = 0
    cat_counter = Counter()
    n_det_total = 0; n_sentinel_total = 0
    spread_zero = 0; multi_det_videos = 0

    for i, jf in enumerate(files):
        name = Path(jf).stem
        d = json.load(open(jf))
        anoms = d.get('anomalies', [])
        fs = d.get('frame_scores'); ts_cached = d.get('timestamps')
        cached_conf = [a.get('confidence', 0.0) for a in anoms]
        # default: pass-through (used if this video is guarded out)
        pv = {'raw_json': d, 'q_only': list(cached_conf),
              'q_slm_mean': list(cached_conf), 'q_slm_oic': list(cached_conf),
              'slm_only_mean': list(cached_conf)}

        if not anoms:
            per_video[name] = pv
            print(f"[{i+1}/{len(files)}] {name}: no detections (skip)")
            continue
        for a in anoms:
            cat_counter[a.get('anomaly_type')] += 1
        n_det_total += len(anoms)

        # deterministic re-extraction of clip_features (faithful, VLM-noise-free)
        try:
            vdata = vad.preprocessor.process(d['video_path'])
            F = expert.precompute_video_features(vdata['all_frames'])
            ts_ext = np.asarray(vdata['timestamps'], dtype=float)
        except Exception as e:
            fidelity_fail += 1
            per_video[name] = pv
            print(f"[{i+1}/{len(files)}] {name}: EXTRACT-FAIL {e} -> pass-through")
            continue
        F = None if F is None else np.asarray(F, dtype=float)
        lf = 0 if F is None else len(F)
        print(f"[{i+1}/{len(files)}] {name}: len(F)={lf} len(ts_ext)={len(ts_ext)} "
              f"len(ts_cached)={len(ts_cached) if ts_cached else 0} n_anom={len(anoms)}")
        if F is None or F.ndim != 2 or len(ts_ext) != len(F) or T.shape[0] < 2:
            fidelity_fail += 1
            per_video[name] = pv
            print(f"    -> FIDELITY GUARD hit (F/ts mismatch or None) -> pass-through")
            continue

        # SLM raw margins (both arms) using RE-EXTRACTED timestamps (matches live)
        rm, ro, sent = slm_raw_arms(F, T, ts_ext, anoms)
        n_sentinel_total += sum(sent)
        raw_mean_arr, slm_n_mean = normalize_with_sentinel(rm)
        _, slm_n_oic = normalize_with_sentinel(ro)

        # kill-gate spread (plain-mean arm) on multi-detection videos
        if len(anoms) > 1:
            multi_det_videos += 1
            spread = float(raw_mean_arr.max() - raw_mean_arr.min())
            if spread < 1e-6:
                spread_zero += 1
                print(f"    -> KILL-GATE: SLM spread ~0 ({spread:.2e})")
            else:
                print(f"    -> SLM raw spread={spread:.4f} sentinel={sum(sent)}/{len(anoms)}")

        # invert committed Q to isolate raw conf (Q multiplier in [QBASE,1] -> safe)
        qn = qn_from_curve(fs, ts_cached, anoms) if (fs and ts_cached) else np.ones(len(anoms))
        q_mult = QBASE + (1 - QBASE) * qn
        raw_conf = np.array(cached_conf) / q_mult

        pv['q_only'] = list(np.minimum(1.0, raw_conf * q_mult))          # round-trip anchor == cached
        pv['q_slm_mean'] = list(np.minimum(1.0, np.array(cached_conf) * (SLM_BASE + (1 - SLM_BASE) * slm_n_mean)))
        pv['q_slm_oic'] = list(np.minimum(1.0, np.array(cached_conf) * (SLM_BASE + (1 - SLM_BASE) * slm_n_oic)))
        pv['slm_only_mean'] = list(np.minimum(1.0, raw_conf * (SLM_BASE + (1 - SLM_BASE) * slm_n_mean)))
        per_video[name] = pv

    # ---------------- reports ----------------
    print("\n================ C-MANDATED CHECKS ================")
    print(f"[1 FEATURE-FIDELITY] videos hitting extract/len-guard: {fidelity_fail}/{len(files)}")
    if fidelity_fail > len(files) * 0.15:
        print("    !! WARNING: non-trivial fraction skipped -> offline number is SUSPECT (dead-signal masquerade)")
    print(f"[2 CATEGORY-RESOLUTION] resolved-category distribution across {n_det_total} detections:")
    for c, n in cat_counter.most_common():
        print(f"    {str(c):<18} {n}")
    unknown = sum(n for c, n in cat_counter.items()
                  if c is None or c == 'Normal' or c not in UCF_CRIME_CATEGORIES)
    print(f"    unknown/Normal/not-in-cats: {unknown}")
    frac_sent = n_sentinel_total / max(n_det_total, 1)
    print(f"[3 SENTINEL-vs-MARGIN] sentinel detections: {n_sentinel_total}/{n_det_total} ({frac_sent:.1%}) "
          f"| real-margin: {n_det_total - n_sentinel_total}")
    if frac_sent > 0.5:
        print("    !! WARNING: majority sentinel -> 'gain' would be demoting hard-to-categorize dets, NOT semantic localization")
    print(f"[KILL-GATE] multi-detection videos with SLM spread~0: {spread_zero}/{multi_det_videos}")
    if multi_det_videos and spread_zero > multi_det_videos * 0.5:
        print("    !! KILL-GATE TRIPPED: majority zero within-video spread -> REJECT (v110/v112 zero-variance trap)")

    # ---------------- A/B eval ----------------
    print("\n================ DETERMINISTIC OFFLINE A/B ================")
    arms = {
        'Q-only (anchor)': ('q_only', 'q_only'),
        'SLM-only (mean)': ('slm_only', 'slm_only_mean'),
        'Q+SLM (mean)':    ('q_slm_mean', 'q_slm_mean'),
        'Q+SLM (oic)':     ('q_slm_oic', 'q_slm_oic'),
    }
    results = {}
    base_map = None
    print(f"{'arm':<20} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  {'delta':>9}")
    for label, (tag, key) in arms.items():
        outd = ROOT / f"results/batch_eval_v116_precheck_{tag}"
        write_dir(outd, per_video, key)
        tm = run_eval(outd)
        results[label] = {k: tm[k] for k in ('mAP', 'AP@0.3', 'AP@0.5', 'AP@0.7')}
        if base_map is None:
            base_map = tm['mAP']  # Q-only anchor is the A/B baseline
        delta = tm['mAP'] - base_map
        print(f"{label:<20} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f}  {delta:>+9.5f}")

    q_only = results['Q-only (anchor)']['mAP']
    slm_only = results['SLM-only (mean)']['mAP']
    best_stack = max(results['Q+SLM (mean)']['mAP'], results['Q+SLM (oic)']['mAP'])
    print("\n================ VERDICT ================")
    print(f"SANITY ANCHOR: Q-only={q_only:.5f} (expect ~0.15632; round-trip through Q inversion)")
    print(f"ORTHOGONALITY: require stack > max(Q-only={q_only:.5f}, SLM-only={slm_only:.5f}); "
          f"best stack={best_stack:.5f} -> {'PASS' if best_stack > max(q_only, slm_only) else 'FAIL'}")
    best_delta = best_stack - q_only
    print(f"COMMIT BAR (>=+0.005 over Q-on baseline): best delta={best_delta:+.5f} -> "
          f"{'COMMIT (lock winning agg)' if best_delta >= 0.005 else 'REJECT -> ship flag-off byte-identical'}")

    json.dump({'results': results, 'q_only': q_only, 'slm_only': slm_only,
               'best_stack': best_stack, 'best_delta': best_delta,
               'fidelity_fail': fidelity_fail, 'sentinel_frac': frac_sent,
               'killgate_zero': spread_zero, 'multi_det_videos': multi_det_videos},
              open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v116_slm_precheck.json', 'w'),
              indent=2)


if __name__ == '__main__':
    main()
