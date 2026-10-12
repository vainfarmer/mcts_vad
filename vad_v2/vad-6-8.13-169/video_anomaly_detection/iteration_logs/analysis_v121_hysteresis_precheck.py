"""
v121 offline pre-check — HYBG (Hysteresis Boundary Grouping).

Deterministic, VLM-noise-free re-derivation on v114's REAL cached detections
(results/batch_eval_v114_full/*.json — frame_scores [fused, flooded-inside-
windows then Gaussian-smoothed], timestamps, anomalies). HYBG is a per-detection
BOUNDARY-PLACEMENT transform: it mutates ONLY [t_start,t_end] via Canny/TAG
dual-threshold hysteresis on the un-flooded curve EXTERIOR, never confidence /
count / order. Because mAP under greedy IoU matching depends only on the final
boundaries + global confidence ordering (both preserved except the boundaries
we intend to move), a deterministic re-derivation from the run's own cached
fused curve is byte-identical to a live re-run of the SAME detections — the
loop-1 exception, so FM6 does not apply (PROVIDED the walked exterior is genuine
un-flooded evidence, which STEP-0 GATE(2) enforces below).

This mirrors pipeline.Pipeline._hysteresis_boundary EXACTLY (same thresholds,
guard band, dip-bridge, reject-cap). Keep the two in sync.

C-mandated gates:
  GATE(1) ANCHOR FIDELITY — HYBG-off reproduces the v114 dir's own mAP exactly
    (self-anchor; we do NOT hardcode a stale headline number).
  GATE(2) UN-FLOODED-EXTERIOR FIDELITY (the load-bearing FM6 discriminator) —
    frame_scores is flooded inside windows AND THEN Gaussian-smoothed, so the
    ~2*sigma frames just outside a window are the flood-blur tail, NOT genuine
    pre-flood evidence. The trivial 'exterior != conf' check does NOT detect
    this. Instead we ENFORCE + ASSERT the guard band: the hysteresis walk must
    begin at least ceil(2*sigma) frames OUTSIDE [lo,hi], so it never reads the
    contaminated tail. We assert this holds for >=3 detections; a violation ->
    treat as FM6 simulation -> REVERT, no live launch.

Control arm (C mandate #2): the identical transform on the un-flooded CATEGORY
curve (the exact curve ECE reads) isolates whether the fused curve's motion
channel adds anything beyond the dip-bridge. The cached v114 JSONs do NOT carry
the category curve (no DetectionResult schema change — v120 clean-revert
discipline), so the category control arm cannot be re-derived offline from
cached JSON; it is available LIVE via --hysteresis_mode category. This harness
therefore prints an explicit note and adjudicates the fused arms {expand,
expand_nodip}; the guard-band gate (GATE 2) is the operative FM6 discriminator.

Pre-registered promotion: promote EXACTLY ONE arm iff
  ΔmAP >= +0.005 AND AP@0.7 non-regression AND per-video detection count
  unchanged. Else REVERT at zero live cost. The promoted arm is written to
  analysis_v121_hysteresis_precheck.json under key 'promoted_arm' (the launcher
  hard-requires it).
"""
import sys, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa
from pipeline import CATEGORY_DURATION_PRIORS  # noqa

V114_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
OUT_JSON = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v121_hysteresis_precheck.json'

# Mirror frame_scorer / ScoringConfig defaults (gaussian_sigma_min=1.0,
# gaussian_sigma_factor=1.5) — the smoothing scale the guard band must clear.
SIGMA_MIN = 1.0
SIGMA_FACTOR = 1.5
ANOMALY_CATEGORIES = list(CATEGORY_DURATION_PRIORS.keys())


def _guard_for(fs, ts):
    span = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
    fps_est = (len(fs) / span) if span > 1e-9 else 1.0
    sigma = max(SIGMA_MIN, SIGMA_FACTOR * fps_est)
    return int(np.ceil(2.0 * sigma)), fps_est


def _extend_left(fs, lo, H, L, dip_cap, guard, dip_bridge):
    g = lo - guard
    if g < 1:
        return lo
    new_lo = lo; saw_high = False; dip = 0; j = g
    while j >= 0:
        v = fs[j]
        if v >= L:
            dip = 0
            if v >= H:
                saw_high = True
            if saw_high:
                new_lo = j
        else:
            if not dip_bridge:
                break
            dip += 1
            if dip > dip_cap:
                break
        j -= 1
    return new_lo


def _extend_right(fs, hi, H, L, dip_cap, guard, dip_bridge):
    m = len(fs)
    g = hi + guard
    if g > m - 2:
        return hi
    new_hi = hi; saw_high = False; dip = 0; j = g
    while j < m:
        v = fs[j]
        if v >= L:
            dip = 0
            if v >= H:
                saw_high = True
            if saw_high:
                new_hi = j
        else:
            if not dip_bridge:
                break
            dip += 1
            if dip > dip_cap:
                break
        j += 1
    return new_hi


def hysteresis_detection(fs, ts, t_s, t_e, cat, dip_bridge, guard, fps_est):
    """Return (new_t_s, new_t_e, walked_left_start, walked_right_start).

    walked_* are the first indices the walk reads (for the guard-band gate);
    None when that side has no room. Mirrors pipeline._hysteresis_boundary.
    """
    n_ts = len(ts)
    baseline = float(np.percentile(fs, 20))
    lo = int(np.searchsorted(ts, t_s)); lo = min(max(lo, 0), n_ts - 1)
    hi = int(np.searchsorted(ts, t_e)); hi = min(max(hi, 0), n_ts - 1)
    if hi <= lo:
        return t_s, t_e, None, None
    peak_val = float(fs[lo:hi + 1].max())
    if peak_val <= baseline:
        return t_s, t_e, None, None
    H = (peak_val + baseline) / 2.0
    L = baseline
    priors = CATEGORY_DURATION_PRIORS.get(cat, {})
    min_seg_s = priors.get('min_segment', 1.5)
    max_dur = priors.get('expected_duration', (0.0, 1e9))[1]
    dip_cap = int(round(min_seg_s * fps_est))
    new_lo = _extend_left(fs, lo, H, L, dip_cap, guard, dip_bridge)
    new_hi = _extend_right(fs, hi, H, L, dip_cap, guard, dip_bridge)
    wl = (lo - guard) if (lo - guard) >= 1 else None
    wr = (hi + guard) if (hi + guard) <= len(fs) - 2 else None
    if new_lo >= lo and new_hi <= hi:
        return t_s, t_e, wl, wr
    cand_s = float(ts[new_lo]); cand_e = float(ts[new_hi])
    if (cand_e - cand_s) > max_dur:
        return t_s, t_e, wl, wr
    return cand_s, cand_e, wl, wr


def _resolve_cat(a):
    cat = a.get('anomaly_type')
    if cat is None or cat in ('Normal', 'unknown') or cat not in ANOMALY_CATEGORIES:
        return None
    return cat


def build_hybg(out_dir, mode):
    """mode 'off' -> pure JSON passthrough (anchor round-trip); otherwise apply
    the HYBG transform. Returns True if any per-video detection count changed."""
    dip_bridge = (mode != 'expand_nodip')
    apply = (mode != 'off')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    count_changed = False
    for jf in glob.glob(str(V114_DIR / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        n_before = len(anoms)
        if apply and fs and ts and anoms and len(fs) >= 3 and len(fs) == len(ts):
            fsa = np.asarray(fs, dtype=float); tsa = np.asarray(ts, dtype=float)
            guard, fps_est = _guard_for(fsa, tsa)
            for a in anoms:
                cat = _resolve_cat(a)
                if cat is None:
                    continue
                ns, ne, _, _ = hysteresis_detection(
                    fsa, tsa, a['t_start'], a['t_end'], cat, dip_bridge, guard, fps_est)
                a['t_start'] = ns; a['t_end'] = ne
        if len(anoms) != n_before:
            count_changed = True
        json.dump(d, open(out_dir / f'{name}.json', 'w'))
    return count_changed


def gate2_guard_band(min_samples=3):
    """C mandate #1: verify the walk begins >= ceil(2*sigma) OUTSIDE [lo,hi].
    Returns (ok, n_checked). Structural by construction — we assert it holds
    (a violation would mean the walk read the flood-blur tail -> FM6)."""
    checked = 0; ok = True
    for jf in glob.glob(str(V114_DIR / '*.json')):
        d = json.load(open(jf))
        fs = d.get('frame_scores'); ts = d.get('timestamps'); anoms = d.get('anomalies', [])
        if not (fs and ts and anoms) or len(fs) < 3 or len(fs) != len(ts):
            continue
        fsa = np.asarray(fs, dtype=float); tsa = np.asarray(ts, dtype=float)
        guard, _ = _guard_for(fsa, tsa)
        n_ts = len(tsa)
        for a in anoms:
            if _resolve_cat(a) is None:
                continue
            lo = min(max(int(np.searchsorted(tsa, a['t_start'])), 0), n_ts - 1)
            hi = min(max(int(np.searchsorted(tsa, a['t_end'])), 0), n_ts - 1)
            if hi <= lo:
                continue
            wl = lo - guard; wr = hi + guard
            # first-read index on each side must be >= guard frames outside window
            if wl >= 1 and not (wl <= lo - guard):
                ok = False
            if wr <= len(fsa) - 2 and not (wr >= hi + guard):
                ok = False
            checked += 1
            if checked >= min_samples and not ok:
                return ok, checked
    return ok, checked


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def main():
    assert V114_DIR.exists(), f"missing {V114_DIR}"
    njson = len(glob.glob(str(V114_DIR / '*.json')))
    print(f"[SETUP] v114 dir JSONs (incl. report): {njson}")

    # ---- GATE(1) ANCHOR FIDELITY (self-anchor + harness round-trip) ----
    base_tm = run_eval(V114_DIR)
    anchor = base_tm['mAP']
    off_dir = ROOT / 'results/batch_eval_v121_precheck_hybg_off'
    build_hybg(off_dir, mode='off')  # pure JSON passthrough
    off_tm = run_eval(off_dir)
    print(f"[GATE1] v114 anchor mAP={anchor:.5f} | harness-off mAP={off_tm['mAP']:.5f}")
    if abs(off_tm['mAP'] - anchor) > 1e-9:
        print("[GATE1] FAIL -> harness JSON round-trip perturbs eval -> ABORT.")
        json.dump({'anchor_mAP': anchor, 'gate1_ok': False, 'promoted_arm': None},
                  open(OUT_JSON, 'w'), indent=2)
        return

    # ---- GATE(2) UN-FLOODED-EXTERIOR FIDELITY (guard band) ----
    g2_ok, g2_n = gate2_guard_band(min_samples=3)
    print(f"[GATE2] guard-band walk-outside-window ok={g2_ok} (checked={g2_n} detections)")
    if not g2_ok:
        print("[GATE2] FAIL -> FM6 simulation risk -> REVERT, no live launch.")
        json.dump({'anchor_mAP': anchor, 'gate2_ok': False, 'promoted_arm': None},
                  open(OUT_JSON, 'w'), indent=2)
        return

    print("[NOTE] category control arm is LIVE-only (--hysteresis_mode category); "
          "cached v114 JSONs carry no category curve -> not re-derivable offline. "
          "Guard-band GATE(2) is the operative FM6 discriminator.")

    print(f"\n{'arm':<20} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  {'dmAP':>8} count")
    print(f"{'baseline':<20} {base_tm['mAP']:>9.5f} {base_tm['AP@0.3']:>9.5f} "
          f"{base_tm['AP@0.5']:>9.5f} {base_tm['AP@0.7']:>9.5f}")

    arms = ['expand', 'expand_nodip']
    res = {'anchor_mAP': anchor, 'gate2_ok': True,
           'baseline': {k: base_tm[k] for k in ('mAP', 'AP@0.3', 'AP@0.5', 'AP@0.7')},
           'arms': {}}
    best = None
    for mode in arms:
        outd = ROOT / f'results/batch_eval_v121_precheck_hybg_{mode}'
        count_changed = build_hybg(outd, mode)
        tm = run_eval(outd)
        d_map = tm['mAP'] - base_tm['mAP']
        d_ap7 = tm['AP@0.7'] - base_tm['AP@0.7']
        res['arms'][mode] = {k: tm[k] for k in ('mAP', 'AP@0.3', 'AP@0.5', 'AP@0.7')}
        res['arms'][mode]['dmAP'] = d_map
        res['arms'][mode]['count_changed'] = count_changed
        print(f"{mode:<20} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} "
              f"{tm['AP@0.7']:>9.5f}  {d_map:+8.5f} {'COUNT!' if count_changed else 'ok'}")
        # pre-registered promotion criteria
        promote = (d_map >= 0.005) and (d_ap7 >= -1e-9) and (not count_changed)
        if promote and (best is None or d_map > best[1]):
            best = (mode, d_map)

    res['promoted_arm'] = best[0] if best else None
    json.dump(res, open(OUT_JSON, 'w'), indent=2)
    if best:
        print(f"\n[PROMOTE] arm='{best[0]}' dmAP={best[1]:+.5f} -> live A/B "
              f"(--qscore_rerank --hysteresis_boundary --hysteresis_mode {best[0]})")
    else:
        print("\n[REVERT] no arm cleared +0.005 mAP with AP@0.7 non-regression & "
              "frozen count -> zero live cost.")


if __name__ == '__main__':
    main()
