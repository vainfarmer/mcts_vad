"""v109 pre-proposal evidence: VideoMAE-window-guided targeting (VTG).

Question: does the TRAINED VideoMAE sliding-window anomaly signal (Stage 2b,
computed for EVERY video, currently consumed ONLY as the 0.15-weight additive
channel of the post-hoc frame-AUC curve) align with GT anomaly location better
than the incumbent scan targeting — in BOTH the directed branch (Source-2
generic CLIP top-up ranking) and the fallback branch (near-random pre-screen)?

Context: 28.8% wrong-location bucket. FCCS (scan everything comparatively)
failed in vivo (-0.11); TSG (EventVAD structural peaks in the fallback branch)
failed in vivo (coverage -15.0s, -2/5 GT segments). Both used ZERO-SHOT or
STRUCTURAL signals. The VideoMAE window curve is the pipeline's ONLY trained
temporal localizer (VideoMAE-large fine-tuned on UCF-Crime; its full-video
classification already drives the pipeline category) and has NEVER been used
for candidate placement or scan targeting.

Method (deterministic, VLM-free):
  For each test video below:
    1. vd = REAL VideoPreprocessor(...).process(path) (v108 run config).
    2. clip_features: cached (analysis_keyframe_feats.pkl) when frame counts
       match, else encoded fresh with the real VideoAnomalyExpert.
    3. videomae_category: parsed from the v108 full log [VideoMAE] line.
    4. clip_regions: REAL pipeline._clip_temporal_localization + dedup
       (categories incl. VIDEOMAE_CONFUSION_MAP, exactly like detect()).
    5a. DIRECTED branch (clip_regions non-empty): baseline = exact replica of
        _directed_llm_scan SELECTION (Source1 region overlap + Source2 generic
        CLIP top-up min_extra=max(2,n//5), cap max_scan=min(max(10,n//5),18)).
        VTG = identical except Source-2 picks are ranked by the VideoMAE
        window curve (np.interp at segment time centers) instead of the
        generic CLIP score. Same count, same cap -> exact call parity.
    5b. FALLBACK branch: baseline = REAL
        pipeline._tsg_incumbent_prescreen_indices(vd); VTG = top-N segments
        by VideoMAE window score at segment centers, N = len(baseline).
    6. Coverage A/B vs GT (exact v108 adjudication functions).

FM6 caveat: this is an OFFLINE coverage measurement (the necessary-condition
gate, same protocol as v108's pre-registered adjudication) — NOT a live mAP
simulation. A positive result only justifies a live loop with the same
pre-registered coverage adjudication; it does not predict the live delta.
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

VAD_ROOT = Path("/Users/zzl/PycharmProjects/vad")
sys.path.insert(0, str(VAD_ROOT / "video_anomaly_detection"))

from evaluate import load_ground_truth  # noqa: E402
from config import VADConfig  # noqa: E402
from pipeline import MultiAnomalyVAD, VIDEOMAE_CONFUSION_MAP  # noqa: E402

DATASET_DIR = Path("/Users/zzl/Movies/ucf-crime")
LOG = VAD_ROOT / "results" / "batch_eval_v108_full.log"
FEAT_CACHE = VAD_ROOT / "video_anomaly_detection/iteration_logs/analysis_keyframe_feats.pkl"
OUT_CACHE = VAD_ROOT / "video_anomaly_detection/iteration_logs/analysis_v109_vtg_cache.pkl"
GT_FPS = 30.0

# 17 zero-IoU wrong-location videos (v97) + 4 TSG-fired fallback videos
WRONG = ['Abuse028_x264', 'Burglary005_x264', 'Burglary079_x264',
         'Fighting003_x264', 'Fighting018_x264', 'Fighting033_x264',
         'RoadAccidents001_x264', 'RoadAccidents019_x264', 'Robbery050_x264',
         'Robbery102_x264', 'Shooting018_x264', 'Shooting032_x264',
         'Shoplifting004_x264', 'Shoplifting037_x264', 'Stealing036_x264',
         'Stealing062_x264', 'Vandalism015_x264']
TSG_FIRED = ['Shoplifting001_x264', 'Shoplifting020_x264',
             'Vandalism007_x264', 'Vandalism028_x264']
VIDEOS = WRONG + TSG_FIRED


class _DummyLLM:
    async def generate(self, prompt, images=None):
        return '{}'


def overlap_secs(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def union_ranges(ranges):
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def coverage(index_set, seg_ranges, gt_secs):
    sel = [seg_ranges[i] for i in index_set]
    uni = union_ranges(sel)
    total = 0.0
    hits = 0
    for gs, ge in gt_secs:
        ov = sum(overlap_secs(us, ue, gs, ge) for us, ue in uni)
        total += ov
        if ov > 0:
            hits += 1
    return total, hits


def parse_videomae_categories():
    cats = {}
    cur = None
    start_re = re.compile(r"\[(\d+)/65\] START (\S+)")
    vm_re = re.compile(r"\[VideoMAE\] .* -> (\w+)@")
    with open(LOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = start_re.search(line)
            if m:
                cur = m.group(2)
                continue
            m = vm_re.search(line)
            if m and cur is not None:
                cats[cur] = m.group(1)
    return cats


def directed_baseline_selection(pipe, vd, clip_regions):
    """EXACT replica of _directed_llm_scan's selection logic (no VLM calls)."""
    segments = vd['segments']
    precomputed = vd.get('clip_features')

    targeted_idx = []
    targeted_set = set()

    # Source 1: segments overlapping CLIP-loc regions (time order)
    for si, seg in enumerate(segments):
        for region in clip_regions:
            if seg['t_start'] < region['t_end'] and seg['t_end'] > region['t_start']:
                if si not in targeted_set:
                    targeted_idx.append(si)
                    targeted_set.add(si)
                break

    # Source 2: generic CLIP anomaly pre-screen top-up
    n = len(segments)
    min_extra = max(2, int(n * 0.2))
    if pipe.anomaly_expert.model is not None and precomputed is not None:
        clip_results = pipe.anomaly_expert.batch_score_segments(segments, precomputed)
        all_scores = [r['anomaly_score'] for r in clip_results]
        scored_indices = sorted(range(n), key=lambda i: all_scores[i], reverse=True)
        added = 0
        for i in scored_indices:
            if added >= min_extra:
                break
            if i not in targeted_set:
                targeted_idx.append(i)
                targeted_set.add(i)
                added += 1

    max_scan = min(max(10, n // 5), 18)
    if len(targeted_idx) > max_scan:
        targeted_idx = targeted_idx[:max_scan]
    return sorted(targeted_idx)


def directed_vtg_selection(pipe, vd, clip_regions, seg_mae_scores):
    """VTG: identical to baseline except Source-2 top-up ranked by the
    trained VideoMAE window curve instead of the generic CLIP score."""
    segments = vd['segments']
    targeted_idx = []
    targeted_set = set()

    # Source 1: unchanged (region overlap)
    for si, seg in enumerate(segments):
        for region in clip_regions:
            if seg['t_start'] < region['t_end'] and seg['t_end'] > region['t_start']:
                if si not in targeted_set:
                    targeted_idx.append(si)
                    targeted_set.add(si)
                break

    # Source 2': VideoMAE-window-ranked top-up (same count, same cap)
    n = len(segments)
    min_extra = max(2, int(n * 0.2))
    ranked = sorted(range(n), key=lambda i: (-seg_mae_scores[i], i))
    added = 0
    for i in ranked:
        if added >= min_extra:
            break
        if i not in targeted_set:
            targeted_idx.append(i)
            targeted_set.add(i)
            added += 1

    max_scan = min(max(10, n // 5), 18)
    if len(targeted_idx) > max_scan:
        targeted_idx = targeted_idx[:max_scan]
    return sorted(targeted_idx)


def main():
    gt = load_ground_truth(str(DATASET_DIR / "data"), split="test")
    vm_cats = parse_videomae_categories()
    feat_cache = pickle.load(open(FEAT_CACHE, "rb")) if FEAT_CACHE.exists() else {}
    run_cache = pickle.load(open(OUT_CACHE, "rb")) if OUT_CACHE.exists() else {}

    cfg = VADConfig(device="mps")
    pipe = MultiAnomalyVAD(llm_client=_DummyLLM(), config=cfg)

    import pandas as pd
    df = pd.read_parquet(DATASET_DIR / "data" / "test-00000-of-00001.parquet")
    path_by_id = {Path(r["video_path"]).stem: r["video_path"]
                  for _, r in df.iterrows()}

    rows = []
    for vid in VIDEOS:
        if vid not in gt or vid not in path_by_id:
            print("SKIP missing", vid)
            continue
        video_path = str(DATASET_DIR / path_by_id[vid])
        gt_secs = [(s / GT_FPS, e / GT_FPS) for s, e in gt[vid]["segments"]]

        if vid in run_cache:
            ent = run_cache[vid]
        else:
            vd = pipe.preprocessor.process(video_path)
            n_frames = len(vd["all_frames"])
            feats = None
            if vid in feat_cache and feat_cache[vid]["feats"].shape[0] == n_frames:
                import torch as _t
                feats = _t.tensor(feat_cache[vid]["feats"]).to(
                    pipe.anomaly_expert.text_features.device)
                print(f"{vid}: cached CLIP feats ({n_frames} frames)")
            else:
                feats = pipe.anomaly_expert.precompute_video_features(vd["all_frames"])
                print(f"{vid}: encoded CLIP feats fresh ({n_frames} frames)")
            vd["clip_features"] = feats
            vd["motion_energy"] = pipe._compute_motion_energy(vd)
            centers, mae_scores = pipe.videomae.sliding_window_scores(
                vd["all_frames"], window_size=32, stride=16)
            vd["videomae_window_centers"] = centers
            vd["videomae_window_scores"] = mae_scores
            cat = vm_cats.get(vid)
            if cat is None:
                r = pipe.videomae.classify(vd["all_frames"], top_k=3, tta=True)
                cat = r["category"]
            cats = [cat] if cat else []
            if cat in VIDEOMAE_CONFUSION_MAP:
                for alt in VIDEOMAE_CONFUSION_MAP[cat]:
                    if alt not in cats:
                        cats.append(alt)
            all_regions = []
            for c in cats:
                all_regions.extend(pipe._clip_temporal_localization(vd, c))
            clip_regions = pipe._deduplicate_clip_regions(all_regions)
            # per-segment VideoMAE window score (interp at segment time centers)
            ts = np.asarray(vd["timestamps"], dtype=float)
            c_times = ts[np.asarray(centers, dtype=int)] if len(centers) else np.array([0.0])
            m_scores = np.asarray(mae_scores, dtype=float)
            seg_centers = np.array([(s["t_start"] + s["t_end"]) / 2.0
                                    for s in vd["segments"]])
            seg_mae = np.interp(seg_centers, c_times, m_scores) if len(c_times) >= 2 \
                else np.full(len(seg_centers), float(m_scores[0]) if len(m_scores) else 0.0)
            if clip_regions:
                base = directed_baseline_selection(pipe, vd, clip_regions)
                vtg = directed_vtg_selection(pipe, vd, clip_regions, seg_mae)
                branch = "directed"
            else:
                base = pipe._tsg_incumbent_prescreen_indices(vd)
                if base is None:
                    base = list(range(min(len(vd["segments"]), 6)))
                n = len(base)
                ranked = sorted(range(len(vd["segments"])),
                                key=lambda i: (-seg_mae[i], i))
                vtg = sorted(ranked[:n])
                branch = "fallback"
            ent = {"branch": branch, "base": base, "vtg": vtg,
                   "seg_ranges": [(s["t_start"], s["t_end"]) for s in vd["segments"]],
                   "cat": cat, "n_regions": len(clip_regions),
                   "seg_mae": seg_mae.tolist(), "duration": vd["duration"]}
            run_cache[vid] = ent
            pickle.dump(run_cache, open(OUT_CACHE, "wb"))

        seg_ranges = ent["seg_ranges"]
        b_sec, b_hit = coverage(ent["base"], seg_ranges, gt_secs)
        v_sec, v_hit = coverage(ent["vtg"], seg_ranges, gt_secs)
        rows.append({"vid": vid, "branch": ent["branch"], "cat": ent["cat"],
                     "n_regions": ent["n_regions"], "k": len(ent["base"]),
                     "n_seg": len(seg_ranges),
                     "b_sec": b_sec, "b_hit": b_hit,
                     "v_sec": v_sec, "v_hit": v_hit,
                     "gt_tot": sum(ge - gs for gs, ge in gt_secs),
                     "gt_n": len(gt_secs)})
        r = rows[-1]
        print(f"{vid:24s} {r['branch']:8s} cat={r['cat']:14s} reg={r['n_regions']} "
              f"k={r['k']}/{r['n_seg']}  base={b_sec:6.2f}s/{b_hit}hit  "
              f"vtg={v_sec:6.2f}s/{v_hit}hit  delta={v_sec-b_sec:+6.2f}s/{v_hit-b_hit:+d}")

    print("=" * 78)
    for branch in ("directed", "fallback", "all"):
        rs = rows if branch == "all" else [r for r in rows if r["branch"] == branch]
        if not rs:
            continue
        b = sum(r["b_sec"] for r in rs)
        v = sum(r["v_sec"] for r in rs)
        bh = sum(r["b_hit"] for r in rs)
        vh = sum(r["v_hit"] for r in rs)
        gn = sum(r["gt_n"] for r in rs)
        print(f"AGG {branch:8s} n={len(rs):2d}  base={b:7.2f}s {bh}/{gn} hits  "
              f"vtg={v:7.2f}s {vh}/{gn} hits  delta={v-b:+7.2f}s {vh-bh:+d} hits")
    json.dump(rows, open(str(OUT_CACHE).replace(".pkl", "_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
