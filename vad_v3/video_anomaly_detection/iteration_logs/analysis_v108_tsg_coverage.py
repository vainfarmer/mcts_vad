"""v108 TSG adjudication — deterministic GT-overlap coverage A/B.

Pre-registered protocol (pipeline.py v104 header): for every fired video,
compare the TSG-selected segment index set vs the baseline (incumbent CLIP
pre-screen) index set by temporal overlap with UCF-Crime GT anomaly segments.

Segment time mapping: the index sets are indices into video_data['segments'],
built by VideoPreprocessor(fps=2, segment_duration=6.0, segment_overlap=0.25,
motion_threshold=1.0, min_motion_interval=2.0) — the exact config used by the
v108 batch run (batch_evaluate.py defaults -> VADConfig.preprocess). We
reconstruct the segments deterministically from the source videos rather than
assuming a nominal grid. A sensitivity pass with the nominal [6i, 6i+6) grid
is also reported.

GT: evaluate.load_ground_truth on the HF parquet dir, frames @ 30fps (the
same fps batch_evaluate.py passes to evaluate()).
"""

import ast
import json
import re
import sys
from pathlib import Path

VAD_ROOT = Path("/Users/zzl/PycharmProjects/vad")
sys.path.insert(0, str(VAD_ROOT / "video_anomaly_detection"))

from evaluate import load_ground_truth  # noqa: E402
from video_preprocessor import VideoPreprocessor  # noqa: E402

LOG = VAD_ROOT / "results" / "batch_eval_v108_full.log"
DATASET_DIR = Path("/Users/zzl/Movies/ucf-crime")
GT_FPS = 30.0  # batch_evaluate.py: evaluate(..., fps=30)

START_RE = re.compile(r"\[(\d+)/65\] START (\S+)")
TSG_RE = re.compile(
    r"\[TSG\] summary: fired selected=(\[[^\]]*\]) baseline=(\[[^\]]*\]) peaks=(\d+)")
GATE_RE = re.compile(r"\[TSG-GATE\].*")


def parse_log():
    fired = {}          # video_id -> (selected, baseline, peaks)
    n_markers = 0
    current = None
    gate_line = None
    with open(LOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = START_RE.search(line)
            if m:
                current = m.group(2)
                continue
            if "[TSG] summary:" in line:
                n_markers += 1
                m = TSG_RE.search(line)
                if m:
                    assert current is not None
                    fired[current] = (ast.literal_eval(m.group(1)),
                                      ast.literal_eval(m.group(2)),
                                      int(m.group(3)))
            g = GATE_RE.search(line)
            if g:
                gate_line = g.group(0).strip()
    return fired, n_markers, gate_line


def overlap_secs(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def union_ranges(ranges):
    """Merge [start, end) ranges into a disjoint union list."""
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def coverage(index_set, seg_ranges, gt_secs):
    """Return (gt_seconds_covered, gt_segments_hit, union_seconds)."""
    sel = [seg_ranges[i] for i in index_set]
    uni = union_ranges(sel)
    total = 0.0
    hits = 0
    for gs, ge in gt_secs:
        ov = sum(overlap_secs(us, ue, gs, ge) for us, ue in uni)
        total += ov
        if ov > 0:
            hits += 1
    union_seconds = sum(ue - us for us, ue in uni)
    return total, hits, union_seconds


def main():
    fired, n_markers, gate_line = parse_log()
    print(f"[TSG-GATE] line from log: {gate_line}")
    print(f"[TSG] summary markers={n_markers} fired={len(fired)} "
          f"not_applicable={n_markers - len(fired)}")
    print()

    gt = load_ground_truth(str(DATASET_DIR / "data"), split="test")

    import pandas as pd
    df = pd.read_parquet(DATASET_DIR / "data" / "test-00000-of-00001.parquet")
    path_by_id = {Path(r["video_path"]).stem: r["video_path"]
                  for _, r in df.iterrows()}

    pre = VideoPreprocessor(fps=2, segment_duration=6.0, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.25)

    agg = {"sel_sec": 0.0, "base_sec": 0.0, "sel_hit": 0, "base_hit": 0,
           "gt_total_sec": 0.0, "gt_total_segs": 0}
    agg6 = dict(agg)  # nominal [6i, 6i+6) sensitivity grid

    for vid, (sel, base, peaks) in sorted(fired.items()):
        entry = gt[vid]
        gt_secs = [(s / GT_FPS, e / GT_FPS) for s, e in entry["segments"]]
        video_path = str(DATASET_DIR / path_by_id[vid])
        vd = pre.process(video_path)
        seg_ranges = [(s["t_start"], s["t_end"]) for s in vd["segments"]]
        n_seg = len(seg_ranges)
        assert max(sel + base) < n_seg, f"{vid}: index out of range"

        s_sec, s_hit, s_uni = coverage(sel, seg_ranges, gt_secs)
        b_sec, b_hit, b_uni = coverage(base, seg_ranges, gt_secs)

        # Nominal-grid sensitivity: segment i covers [6i, 6i+6)
        dur = vd["duration"]
        grid6 = [(6.0 * i, min(6.0 * i + 6.0, dur)) for i in range(1000)]
        s6_sec, s6_hit, _ = coverage(sel, grid6, gt_secs)
        b6_sec, b6_hit, _ = coverage(base, grid6, gt_secs)

        gt_tot = sum(ge - gs for gs, ge in gt_secs)
        agg["sel_sec"] += s_sec; agg["base_sec"] += b_sec
        agg["sel_hit"] += s_hit; agg["base_hit"] += b_hit
        agg["gt_total_sec"] += gt_tot; agg["gt_total_segs"] += len(gt_secs)
        agg6["sel_sec"] += s6_sec; agg6["base_sec"] += b6_sec
        agg6["sel_hit"] += s6_hit; agg6["base_hit"] += b6_hit
        agg6["gt_total_sec"] += gt_tot; agg6["gt_total_segs"] += len(gt_secs)

        print(f"{vid} (peaks={peaks}, n_segments={n_seg}, duration={dur:.1f}s)")
        print(f"  GT (s): {[(round(a,1), round(b,1)) for a, b in gt_secs]} "
              f"total={gt_tot:.1f}s")
        print(f"  selected={sel}")
        print(f"    ranges={[(round(seg_ranges[i][0],1), round(seg_ranges[i][1],1)) for i in sel]}")
        print(f"    GT covered={s_sec:.2f}s hits={s_hit}/{len(gt_secs)} "
              f"(union span {s_uni:.1f}s)")
        print(f"  baseline={base}")
        print(f"    ranges={[(round(seg_ranges[i][0],1), round(seg_ranges[i][1],1)) for i in base]}")
        print(f"    GT covered={b_sec:.2f}s hits={b_hit}/{len(gt_secs)} "
              f"(union span {b_uni:.1f}s)")
        print(f"  delta (selected-baseline): {s_sec - b_sec:+.2f}s "
              f"hits {s_hit - b_hit:+d}")
        print(f"  [nominal 6s grid] selected={s6_sec:.2f}s/{s6_hit} hit(s) "
              f"baseline={b6_sec:.2f}s/{b6_hit} hit(s) "
              f"delta={s6_sec - b6_sec:+.2f}s/{s6_hit - b6_hit:+d}")
        print()

    print("=" * 70)
    print("AGGREGATE over fired videos (exact reconstructed segments):")
    print(f"  GT seconds covered: selected={agg['sel_sec']:.2f}s "
          f"baseline={agg['base_sec']:.2f}s "
          f"delta={agg['sel_sec'] - agg['base_sec']:+.2f}s "
          f"(GT total {agg['gt_total_sec']:.1f}s)")
    print(f"  GT segments hit:    selected={agg['sel_hit']}/{agg['gt_total_segs']} "
          f"baseline={agg['base_hit']}/{agg['gt_total_segs']} "
          f"delta={agg['sel_hit'] - agg['base_hit']:+d}")
    print("AGGREGATE (nominal [6i,6i+6) grid, sensitivity):")
    print(f"  GT seconds covered: selected={agg6['sel_sec']:.2f}s "
          f"baseline={agg6['base_sec']:.2f}s "
          f"delta={agg6['sel_sec'] - agg6['base_sec']:+.2f}s")
    print(f"  GT segments hit:    selected={agg6['sel_hit']}/{agg6['gt_total_segs']} "
          f"baseline={agg6['base_hit']}/{agg6['gt_total_segs']} "
          f"delta={agg6['sel_hit'] - agg6['base_hit']:+d}")

    improved = (agg["sel_sec"] > agg["base_sec"]) or \
               (agg["sel_hit"] > agg["base_hit"])
    print()
    print(f"VERDICT coverage_improved={improved}")

    out = {
        "gate_line": gate_line, "markers": n_markers, "fired": len(fired),
        "aggregate_exact": agg, "aggregate_nominal6": agg6,
        "improved_exact": improved,
    }
    with open(Path(__file__).with_suffix(".json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
