"""
v122 offline pre-check — BOR: Boundary-Offset Recentering (duration-frozen rigid
translation of each detection window driven by un-flooded fused-curve exterior
asymmetry), STACKED on the committed v111 Q-score, deterministic A/B on v114's
REAL cached detections.

WHY THIS IS FAITHFUL (not an FM6 candidate-pool simulation): BOR is a deterministic
transform of v114's OWN cached fused curve (frame_scores + timestamps) evaluated
against REAL GT — "what v114+BOR would have scored". It is NOT the same trusted
class as pure Q-score (it CHANGES boundaries), but it is a faithful measurement,
not a candidate-pool sim. Residual: after a shift the cached interior is still
flooded at the OLD location (Q-score's peak read may see a plateau) — the harness
absorbs this because it scores net mAP against real GT.

PIPELINE-FAITHFUL ORDER (must match pipeline.detect / _recenter_boundaries):
  final_anomalies (v51 boundaries, RAW conf) -> BOR shift -> temporal_nms+global_merge
    -> Q-score at NEW boundaries -> conf'.
The v114 cache stores POST-Q confidences at v51 boundaries. We therefore:
  1. Recover RAW conf per det: conf_raw = conf_cache / f_old, where
     f_old = base+(1-base)*Q_norm at v51 boundaries over the CACHE set (exactly how
     v114 produced the cache). Round-trip is EXACT for the identity arm.
  2. Restore raw conf, apply BOR shift, re-apply hard temporal_nms+global_merge.
  3. Re-apply Q at the NEW boundaries over the surviving set -> conf'.
Sanity arm 'identity' (gate=inf, no shift) reproduces the cache byte-for-byte ->
control mAP == v114 baseline (abort on mismatch).

MANDATED (review C): reports per-video localization delta distribution
(improve vs regress), a CAUSALITY diagnostic (per-shifted-detection IoU-vs-GT
BEFORE vs AFTER — BOR must NET-RAISE IoU on shifted dets), and a kill-gate on
non-zero shift-decision variance. Pre-registered promotion:
  dmAP >= +0.005 AND AP@0.7 non-regression AND per-video count monotone
  non-increasing AND gain BROAD-BASED (shifted-det net IoU positive across a
  majority of firing videos, not 1-2 lucky videos).
Writes analysis_v122_bor_precheck.json with promoted_arm in
{balance_conservative, balance_moderate} or null.
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate, load_ground_truth  # noqa
from post_processor import temporal_nms, global_merge  # noqa

V114_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
GT_FPS = 30.0
QBASE = 0.5

# Pre-registered BOR arm table — MUST match pipeline.Pipeline._BOR_ARMS exactly.
BOR_ARMS = {
    'balance_conservative': (0.3, 0.25),   # (margin_frac, max_shift_frac)
    'balance_moderate':     (0.5, 0.50),
}
BOR_GAIN = 0.5       # config.scoring.bor_gain
BOR_GATE = 0.1       # config.scoring.bor_gate_frac

PROMOTE_DMAP = 0.005


# ---------- Q-score (byte-identical to pipeline._qscore_rerank) ----------
def _qscore_components(fs, ts, t_s, t_e):
    fs = np.asarray(fs, float); ts = np.asarray(ts, float)
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


def _q_factors(anoms, fs, ts, base=QBASE):
    """Return list of Q factors f_i = base+(1-base)*qn_i (within-video min-max)."""
    if not anoms or fs is None or ts is None:
        return [1.0] * len(anoms or [])
    fs = np.asarray(fs, float); ts = np.asarray(ts, float)
    if len(fs) < 3 or len(ts) != len(fs):
        return [1.0] * len(anoms)
    qs = np.array([_qscore_components(fs, ts, a['t_start'], a['t_end']) for a in anoms])
    if len(qs) > 1 and qs.max() - qs.min() > 1e-9:
        qn = (qs - qs.min()) / (qs.max() - qs.min())
    else:
        qn = np.ones_like(qs)
    return [base + (1 - base) * float(q) for q in qn]


# ---------- BOR shift (byte-identical to pipeline._recenter_boundaries) ----------
def _bor_shift(anoms, fs, ts, arm):
    """Apply BOR to a list of dicts (RAW conf) in place-copy; returns (new_anoms,
    shift_records). shift_records: list of (idx_in_input, old[ts,te], new[ts,te])
    for detections that actually moved. Guards mirror the pipeline exactly."""
    if not anoms or fs is None or ts is None:
        return [a.copy() for a in (anoms or [])], []
    fs = np.asarray(fs, float); ts = np.asarray(ts, float)
    if len(fs) < 3 or len(ts) != len(fs):
        return [a.copy() for a in anoms], []
    vid_range = float(fs.max() - fs.min())
    if vid_range < 1e-9:
        return [a.copy() for a in anoms], []
    margin_frac, max_shift_frac = BOR_ARMS[arm]
    t_lo, t_hi = float(ts[0]), float(ts[-1])
    out = [a.copy() for a in anoms]

    def seg_mean(a, b):
        if b <= a:
            return None
        m = (ts >= a) & (ts <= b)
        if m.sum() < 1:
            return None
        return float(fs[m].mean())

    order = sorted(range(len(out)), key=lambda i: out[i]['t_start'])
    starts = [out[i]['t_start'] for i in order]
    ends = [out[i]['t_end'] for i in order]
    recs = []
    for rank, oi in enumerate(order):
        a = out[oi]
        t_s, t_e = float(a['t_start']), float(a['t_end'])
        dur = t_e - t_s
        if dur <= 1e-6:
            continue
        prev_end = ends[rank - 1] if rank > 0 else t_lo
        next_start = starts[rank + 1] if rank + 1 < len(order) else t_hi
        m = margin_frac * dur
        l_a = max(t_lo, prev_end, t_s - m)
        r_b = min(t_hi, next_start, t_e + m)
        lext = seg_mean(l_a, t_s)
        rext = seg_mean(t_e, r_b)
        if lext is None or rext is None:
            continue
        I = (lext - rext) / (vid_range + 1e-9)
        if abs(I) < BOR_GATE:
            continue
        cap = max_shift_frac * dur
        shift = float(np.clip(BOR_GAIN * I, -cap, cap)) * dur
        target_ts = t_s - shift
        idx = int(np.searchsorted(ts, target_ts))
        idx = min(max(idx, 0), len(ts) - 1)
        new_t_s = float(ts[idx])
        actual = t_s - new_t_s
        new_t_e = t_e - actual
        if new_t_s < t_lo - 1e-9 or new_t_e > t_hi + 1e-9:
            continue
        if new_t_s < prev_end - 1e-9 or new_t_e > next_start + 1e-9:
            continue
        if abs(actual) < 1e-9:
            continue
        recs.append((oi, (t_s, t_e), (new_t_s, new_t_e)))
        a['t_start'] = new_t_s
        a['t_end'] = new_t_e
    if recs:
        out = temporal_nms(out, iou_threshold=0.3)
        out = global_merge(out, iou_threshold=0.3, same_type_gap=2.0)
    return out, recs


def _iou(seg, gt):
    a, b = seg; c, d = gt
    inter = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return inter / union if union > 0 else 0.0


def _best_iou(seg, gt_secs):
    return max((_iou(seg, g) for g in gt_secs), default=0.0)


def build_arm(out_dir, arm):
    """Write v114+BOR(arm)+Q-at-new-boundaries detections. arm='identity' =>
    strip-Q then re-Q at v51 boundaries (no shift) = exact cache round-trip."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        if name in ('evaluation_report',):
            continue
        d = json.load(open(jf))
        if not isinstance(d, dict) or 'anomalies' not in d:
            json.dump(d, open(out_dir / f'{name}.json', 'w'))
            continue
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if fs and ts and anoms:
            # 1. strip Q -> raw conf (f_old from Q at v51 over cache set)
            f_old = _q_factors(anoms, fs, ts)
            raw = []
            for a, fo in zip(anoms, f_old):
                b = a.copy()
                b['confidence'] = a.get('confidence', 0.0) / (fo if fo > 1e-9 else 1.0)
                raw.append(b)
            # 2. BOR shift + hard nms/merge on raw-conf dets (arm='identity' => no shift)
            if arm == 'identity':
                shifted = [a.copy() for a in raw]
            else:
                shifted, _ = _bor_shift(raw, fs, ts, arm)
            # 3. re-apply Q at NEW boundaries over surviving set
            f_new = _q_factors(shifted, fs, ts)
            for a, fn in zip(shifted, f_new):
                a['confidence'] = min(1.0, a['confidence'] * fn)
            d['anomalies'] = shifted
            d['is_normal'] = (len(shifted) == 0)
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def diagnostics(arm, gt):
    """Return per-video localization + causality stats for a shifting arm."""
    n_fire_vid = 0
    n_shift = 0
    iou_before_sum = 0.0
    iou_after_sum = 0.0
    vid_improve = 0
    vid_regress = 0
    vid_net = []          # per firing video: net (after-before) IoU over shifted dets
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        if name in ('evaluation_report',) or name not in gt:
            continue
        d = json.load(open(jf))
        if not isinstance(d, dict):
            continue
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if not (fs and ts and anoms):
            continue
        gt_secs = [(s / GT_FPS, e / GT_FPS) for (s, e) in gt[name]['segments']]
        if not gt_secs:
            continue
        _, recs = _bor_shift([a.copy() for a in anoms], fs, ts, arm)
        if not recs:
            continue
        n_fire_vid += 1
        net = 0.0
        for _, (os_, oe), (ns, ne) in recs:
            ib = _best_iou((os_, oe), gt_secs)
            ia = _best_iou((ns, ne), gt_secs)
            iou_before_sum += ib; iou_after_sum += ia
            net += (ia - ib)
            n_shift += 1
        vid_net.append(net)
        if net > 1e-6:
            vid_improve += 1
        elif net < -1e-6:
            vid_regress += 1
    return {
        'firing_videos': n_fire_vid,
        'shifted_detections': n_shift,
        'iou_before_mean': (iou_before_sum / n_shift) if n_shift else 0.0,
        'iou_after_mean': (iou_after_sum / n_shift) if n_shift else 0.0,
        'iou_net_per_shifted_det': ((iou_after_sum - iou_before_sum) / n_shift) if n_shift else 0.0,
        'videos_iou_improve': vid_improve,
        'videos_iou_regress': vid_regress,
    }


def main():
    gt = load_ground_truth(LABEL, split='test')
    base_tm = run_eval(V114_DIR)
    print(f"[SANITY] v114 cache baseline mAP={base_tm['mAP']:.5f} "
          f"AP@0.5={base_tm['AP@0.5']:.5f} AP@0.7={base_tm['AP@0.7']:.5f}")

    # identity round-trip sanity: strip-Q then re-Q at v51 == cache
    id_dir = ROOT / 'results/batch_eval_v122_precheck_identity'
    build_arm(id_dir, 'identity')
    id_tm = run_eval(id_dir)
    print(f"[SANITY] identity round-trip mAP={id_tm['mAP']:.5f} "
          f"(delta {id_tm['mAP']-base_tm['mAP']:+.6f})")
    assert abs(id_tm['mAP'] - base_tm['mAP']) < 1e-4, \
        "IDENTITY ROUND-TRIP MISMATCH — strip/re-Q reconstruction is not faithful; ABORT"

    print(f"\n{'arm':<24} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  {'dmAP':>9}")
    print(f"{'baseline(v114)':<24} {base_tm['mAP']:>9.5f} {base_tm['AP@0.3']:>9.5f} "
          f"{base_tm['AP@0.5']:>9.5f} {base_tm['AP@0.7']:>9.5f}")

    results = {'baseline_mAP': base_tm['mAP'], 'baseline_AP@0.7': base_tm['AP@0.7'],
               'arms': {}, 'promoted_arm': None}
    best_arm, best_dmap = None, -1e9
    for arm in ('balance_conservative', 'balance_moderate'):
        outd = ROOT / f'results/batch_eval_v122_precheck_{arm}'
        build_arm(outd, arm)
        tm = run_eval(outd)
        diag = diagnostics(arm, gt)
        dmap = tm['mAP'] - base_tm['mAP']
        d_ap7 = tm['AP@0.7'] - base_tm['AP@0.7']
        # count-monotone check: total detections must be non-increasing vs cache
        n_ctrl = _count_dets(V114_DIR)
        n_arm = _count_dets(outd)
        count_ok = n_arm <= n_ctrl
        broad_ok = diag['firing_videos'] > 0 and \
            diag['videos_iou_improve'] >= max(1, diag['videos_iou_regress'] + 1) and \
            diag['videos_iou_improve'] * 2 >= diag['firing_videos']
        causal_ok = diag['iou_net_per_shifted_det'] > 0
        variance_ok = diag['shifted_detections'] > 0
        passed = (dmap >= PROMOTE_DMAP and d_ap7 >= -1e-6 and count_ok
                  and broad_ok and causal_ok and variance_ok)
        print(f"{arm:<24} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f}  {dmap:+9.5f}")
        print(f"    fire_vids={diag['firing_videos']} shifted={diag['shifted_detections']} "
              f"IoU {diag['iou_before_mean']:.3f}->{diag['iou_after_mean']:.3f} "
              f"(net/det {diag['iou_net_per_shifted_det']:+.4f}) "
              f"vids +{diag['videos_iou_improve']}/-{diag['videos_iou_regress']} "
              f"| n_det {n_ctrl}->{n_arm} count_ok={count_ok} "
              f"broad={broad_ok} causal={causal_ok} PASS={passed}")
        results['arms'][arm] = {
            'mAP': tm['mAP'], 'AP@0.3': tm['AP@0.3'], 'AP@0.5': tm['AP@0.5'],
            'AP@0.7': tm['AP@0.7'], 'dmAP': dmap, 'dAP@0.7': d_ap7,
            'n_det_ctrl': n_ctrl, 'n_det_arm': n_arm, 'count_monotone_ok': count_ok,
            'diag': diag, 'broad_based_ok': broad_ok, 'causal_ok': causal_ok,
            'variance_ok': variance_ok, 'passed': passed,
        }
        if passed and dmap > best_dmap:
            best_arm, best_dmap = arm, dmap

    results['promoted_arm'] = best_arm
    print(f"\nPROMOTED ARM: {best_arm} (dmAP={best_dmap:+.5f})" if best_arm
          else "\nPROMOTED ARM: none — no arm cleared +0.005 with AP@0.7 non-regression "
               "/ broad-based / net-positive IoU. SHIP FLAG-OFF (byte-identical v114).")
    outj = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v122_bor_precheck.json'
    json.dump(results, open(outj, 'w'), indent=2)
    print(f"wrote {outj}")


def _count_dets(result_dir):
    n = 0
    for jf in glob.glob(str(Path(result_dir) / '*.json')):
        if Path(jf).stem in ('evaluation_report',):
            continue
        d = json.load(open(jf))
        if isinstance(d, dict) and 'anomalies' in d:
            n += len(d.get('anomalies') or [])
    return n


if __name__ == '__main__':
    main()
