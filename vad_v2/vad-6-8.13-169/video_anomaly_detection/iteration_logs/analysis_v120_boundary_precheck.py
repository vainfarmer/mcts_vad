"""
v120 offline pre-check — BREF Boundary-REFinement bake-off (HARNESS-FIRST).

Adjudicates TWO orthogonal per-detection boundary-PLACEMENT anchors offline on
v114's REAL detections, deterministically (confidence/count/order FROZEN — only
the [t_s,t_e] offsets move). mAP then depends only on the IoU of the existing
greedy matches, so re-deriving new boundaries for the run's OWN real detections
is FAITHFUL in-vivo (v107 protocol; NOT an FM6 candidate-pool simulation).

Arms (mirror pipeline._boundary_refine EXACTLY, W = (t_e - t_s) * 0.3):
  Arm-1 'fused_deriv' : grad = np.gradient(frame_scores) on the cached fused
                        curve; ±W snap start->argmax(+grad), end->argmin(grad).
                        Fully cached, zero-decode.
  Arm-2 'motion_edge' : motion_energy is NOT in the JSON, so recompute it
                        deterministically by decoding each video with the SAME
                        cv2 64x64 mean-abs frame-diff as pipeline
                        ._compute_motion_energy, sampled to len(timestamps);
                        smooth with the SAME 1s kernel _motion_temporal_
                        localization uses; half-max LEVEL on |grad|; ±W snap
                        start->dominant rising edge, end->dominant falling edge.
                        Length must align to timestamps else the video is
                        SKIPPED (kept at v114 boundaries) with a logged warning.
  Arm-3 'agree'       : (report only) snap only where Arm-1 and Arm-2 agree
                        within one grid step.

PRE-REGISTERED GATE (C mandate #3/#4): full mAP recomputed via
evaluate.compute_temporal_map (mAP + AP@0.3/0.5/0.7). Promote EXACTLY the ONE
arm whose mAP delta (i) clears +0.005 AND (ii) does NOT regress AP@0.7 (HARD
ABORT). If NO arm clears both -> REVERT at zero live cost (do not launch).

Sanity: an all-no-edge / all-skip pass reproduces v114 mAP exactly.
"""
import sys, json, glob, shutil, math
from pathlib import Path
import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa
# duration priors live in pipeline; import without running the pipeline
from pipeline import CATEGORY_DURATION_PRIORS  # noqa

# C mandate #2: repoint to v114's REAL per-video detections (NOT v108).
V114_DIR = ROOT / 'results/batch_eval_v114_full'
LABEL = '/Users/zzl/Movies/ucf-crime/data'
OUT_JSON = ROOT / 'video_anomaly_detection/iteration_logs/analysis_v120_boundary_precheck.json'

UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def _resolve_cat(a):
    cat = a.get('anomaly_type')
    if cat is None or cat in ('Normal', 'unknown') or cat not in UCF_CRIME_CATEGORIES:
        return None
    return cat


def _max_dur(cat):
    return CATEGORY_DURATION_PRIORS.get(cat, {}).get('expected_duration', (3.0, 30.0))[1]


def _snap_fused(sig, ts, t, w, is_start, n_ts):
    lo = max(0, min(int(np.searchsorted(ts, t - w, side='left')), n_ts - 1))
    hi = max(lo + 1, min(int(np.searchsorted(ts, t + w, side='right')), n_ts))
    seg = sig[lo:hi]
    if len(seg) < 1:
        return None
    if is_start:
        j = int(np.argmax(seg))
        if seg[j] <= 0:
            return None
    else:
        j = int(np.argmin(seg))
        if seg[j] >= 0:
            return None
    return lo + j


def _snap_motion(mg, level, ts, t, w, is_start, n_ts):
    lo = max(0, min(int(np.searchsorted(ts, t - w, side='left')), n_ts - 1))
    hi = max(lo + 1, min(int(np.searchsorted(ts, t + w, side='right')), n_ts))
    seg_g = mg[lo:hi]
    seg_a = np.abs(seg_g)
    if len(seg_g) < 1:
        return None
    mask = (seg_g > 0) & (seg_a >= level) if is_start else (seg_g < 0) & (seg_a >= level)
    if not mask.any():
        return None
    idxs = np.where(mask)[0]
    j = idxs[int(np.argmax(seg_a[idxs]))]
    return lo + j


def _recompute_motion_energy(video_path, n_ts):
    """Deterministic cv2 64x64 mean-abs frame-diff sampled to n_ts frames.

    Uniformly samples n_ts frame indices across the decoded stream (matching the
    per-frame all_frames grid length that timestamps aligns to), then applies
    the SAME motion formula as pipeline._compute_motion_energy. Returns an array
    of length n_ts, or None on any decode failure / length mismatch (caller then
    SKIPS that video, keeping v114 boundaries).
    """
    try:
        import cv2
    except Exception as e:
        print(f"[WARN] cv2 unavailable: {e}")
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2 or n_ts < 2:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, n_ts).astype(int)
    small = []
    want = set(int(i) for i in idxs)
    grabbed = {}
    fi = 0
    ok, frame = cap.read()
    while ok:
        if fi in want:
            f = cv2.resize(frame, (64, 64)).astype(np.float32)
            grabbed[fi] = f
        fi += 1
        if fi > total:
            break
        ok, frame = cap.read()
    cap.release()
    for i in idxs:
        i = int(i)
        if i in grabbed:
            small.append(grabbed[i])
        elif grabbed:
            # nearest available (deterministic fallback for dropped decodes)
            k = min(grabbed.keys(), key=lambda x: abs(x - i))
            small.append(grabbed[k])
        else:
            return None
    small = np.stack(small).astype(np.float32)
    if small.shape[0] != n_ts:
        return None
    motion = np.zeros(n_ts)
    motion[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    return motion


def refine_video(d, mode):
    """Return (refined_detection_dicts, n_refined). Mutates a copy."""
    anoms = [dict(a) for a in d.get('anomalies', [])]
    fs = d.get('frame_scores')
    ts = d.get('timestamps')
    if not anoms or not fs or not ts:
        return anoms, 0
    ts = np.asarray(ts, dtype=np.float64)
    n_ts = len(ts)
    fsa = np.asarray(fs, dtype=np.float64)
    if n_ts < 3 or len(fsa) != n_ts:
        return anoms, 0

    sig = np.gradient(fsa) if mode in ('fused_deriv', 'agree') else None

    mg = level = None
    if mode in ('motion_edge', 'agree'):
        motion = _recompute_motion_energy(d.get('video_path'), n_ts)
        if motion is None or len(motion) != n_ts:
            # cannot build motion arm for this video -> keep v114 boundaries
            if mode == 'motion_edge':
                return anoms, 0
        else:
            duration = d.get('duration', float(ts[-1]))
            fps = max(1, len(ts) / max(1, duration))
            window = max(2, int(1.0 * fps))
            kernel = np.ones(window) / window
            smoothed = np.convolve(motion, kernel, mode='same')
            mg = np.gradient(smoothed)
            amg = np.abs(mg)
            level = (float(np.max(amg)) + float(np.percentile(amg, 20))) / 2.0

    n_refined = 0
    for a in anoms:
        cat = _resolve_cat(a)
        if cat is None:
            continue
        t_s, t_e = a['t_start'], a['t_end']
        w = (t_e - t_s) * 0.3
        if w <= 0:
            continue

        def snap(t, is_start):
            if mode == 'fused_deriv':
                return _snap_fused(sig, ts, t, w, is_start, n_ts)
            if mode == 'motion_edge':
                if mg is None:
                    return None
                return _snap_motion(mg, level, ts, t, w, is_start, n_ts)
            # agree: both must agree within one grid step
            i_f = _snap_fused(sig, ts, t, w, is_start, n_ts)
            i_m = _snap_motion(mg, level, ts, t, w, is_start, n_ts) if mg is not None else None
            if i_f is None or i_m is None or abs(i_f - i_m) > 1:
                return None
            return i_f

        i_s = snap(t_s, True)
        i_e = snap(t_e, False)
        ns = float(ts[i_s]) if i_s is not None else t_s
        ne = float(ts[i_e]) if i_e is not None else t_e
        if ne <= ns:
            continue
        if (ne - ns) > _max_dur(cat):
            continue
        moved = (abs(ns - t_s) > 1e-9) or (abs(ne - t_e) > 1e-9)
        if moved:
            a['t_start'] = ns
            a['t_end'] = ne
            n_refined += 1
    return anoms, n_refined


def build(out_dir, mode):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    total_ref = 0
    for jf in glob.glob(str(V114_DIR / '*.json')):
        if Path(jf).name == 'evaluation_report.json':
            continue
        d = json.load(open(jf))
        anoms, nref = refine_video(d, mode)
        d['anomalies'] = anoms
        total_ref += nref
        json.dump(d, open(out_dir / Path(jf).name, 'w'))
    return total_ref


def run_eval(result_dir):
    return evaluate(LABEL, str(result_dir), fps=30, split='test')['temporal_map']


def main():
    # C mandate #2: assert input dir + count == 65 BEFORE adjudicating.
    jsons = [p for p in glob.glob(str(V114_DIR / '*.json'))
             if Path(p).name != 'evaluation_report.json']
    assert V114_DIR.exists(), f"missing input dir {V114_DIR}"
    print(f"[INPUT] {V114_DIR}: {len(jsons)} detection JSONs")
    assert len(jsons) == 65, f"expected 65 v114 detection JSONs, got {len(jsons)}"

    base = run_eval(V114_DIR)
    print(f"[SANITY] v114 baseline mAP={base['mAP']:.5f} "
          f"AP@0.5={base['AP@0.5']:.5f} AP@0.7={base['AP@0.7']:.5f}")
    print(f"\n{'arm':<16} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9} "
          f"{'refined':>8}  d_mAP    d_AP07")
    print(f"{'baseline':<16} {base['mAP']:>9.5f} {base['AP@0.3']:>9.5f} "
          f"{base['AP@0.5']:>9.5f} {base['AP@0.7']:>9.5f}")

    res = {'baseline': {k: base[k] for k in ('mAP', 'AP@0.3', 'AP@0.5', 'AP@0.7')}}
    arms = ['fused_deriv', 'motion_edge', 'agree']  # agree = report-only
    for mode in arms:
        outd = ROOT / f"results/batch_eval_v120_precheck_{mode}"
        nref = build(outd, mode)
        tm = run_eval(outd)
        res[mode] = {k: tm[k] for k in ('mAP', 'AP@0.3', 'AP@0.5', 'AP@0.7')}
        res[mode]['refined'] = nref
        res[mode]['d_mAP'] = tm['mAP'] - base['mAP']
        res[mode]['d_AP07'] = tm['AP@0.7'] - base['AP@0.7']
        print(f"{mode:<16} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} "
              f"{tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f} {nref:>8} "
              f"{tm['mAP']-base['mAP']:+.5f} {tm['AP@0.7']-base['AP@0.7']:+.5f}")

    # PRE-REGISTERED GATE: promote exactly ONE candidate arm (agree is report-only)
    candidates = ['fused_deriv', 'motion_edge']
    passing = [m for m in candidates
               if res[m]['d_mAP'] >= 0.005 and res[m]['d_AP07'] >= -1e-9]
    promoted = None
    if passing:
        promoted = max(passing, key=lambda m: res[m]['d_mAP'])
    res['gate'] = {
        'bar_mAP': 0.005,
        'ap07_non_regression': True,
        'passing_arms': passing,
        'promoted_arm': promoted,
        'verdict': ('PROMOTE ' + promoted) if promoted else 'REVERT (no arm clears +0.005 with AP@0.7 non-regression)',
    }
    print(f"\n[GATE] passing={passing} promoted={promoted}")
    print(f"[GATE] {res['gate']['verdict']}")
    if promoted:
        print(f"[GATE] set config.scoring.boundary_refine_mode = '{promoted}' and "
              f"launch ONE live A/B: control=['--qscore_rerank'], "
              f"treatment=['--qscore_rerank','--boundary_refine','--boundary_refine_mode',"
              f"'{promoted}']")

    json.dump(res, open(OUT_JSON, 'w'), indent=2)
    print(f"\n[WROTE] {OUT_JSON}")


if __name__ == '__main__':
    main()
