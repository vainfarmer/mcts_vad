"""v106 ECE deterministic intra-run pre/post-boundary A/B adjudication.

For each per-video output JSON of a flag-ON (--event_continuation_extension)
batch run, build:
  set A: detections with pre_ece_boundaries (pre-ECE t_start/t_end, same
         confidence / anomaly_type / count / order);
  set B: final anomalies (post-ECE boundaries).
Score BOTH through the official evaluate.compute_temporal_map greedy matcher
and print the deterministic pre/post mAP delta (immune to VLM noise).

Per C's mandate: JSONs lacking pre_ece_boundaries are skipped AND counted;
if any such video has >=1 detection on a flag-ON run, that is an activation
anomaly (investigate before accepting the delta).

Usage:
  python3 iteration_logs/analysis_v106_ece_ab.py \
      --results_dir results/batch_eval_v106 \
      --dataset_dir /path/to/ucf_crime [--fps 30]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import load_ground_truth, compute_temporal_map, _extract_video_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="v106 ECE pre/post deterministic A/B")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    gt = load_ground_truth(str(Path(args.dataset_dir) / "data"), split=args.split)

    pred_pre, pred_post = {}, {}
    n_videos = n_missing = n_missing_with_dets = 0
    for jf in sorted(Path(args.results_dir).glob("*.json")):
        if jf.name == "evaluation_report.json":
            continue
        with open(jf) as f:
            data = json.load(f)
        if "anomalies" not in data:
            continue
        n_videos += 1
        video_id = _extract_video_id(data.get("video_path", jf.stem))
        anomalies = data.get("anomalies", [])
        pre = data.get("pre_ece_boundaries")
        if pre is None:
            n_missing += 1
            if len(anomalies) >= 1:
                n_missing_with_dets += 1
            continue
        if len(pre) != len(anomalies):
            print(f"WARNING {video_id}: pre_ece_boundaries len {len(pre)} != "
                  f"anomalies len {len(anomalies)} — skipped (anomaly)")
            n_missing += 1
            n_missing_with_dets += 1
            continue
        pre_anoms = [
            {**a, "t_start": float(p[0]), "t_end": float(p[1])}
            for a, p in zip(anomalies, pre)
        ]
        pred_pre[video_id] = {"anomalies": pre_anoms, "duration": data.get("duration", 0)}
        pred_post[video_id] = {"anomalies": anomalies, "duration": data.get("duration", 0)}

    print(f"videos scanned: {n_videos}")
    print(f"videos lacking pre_ece_boundaries (skipped): {n_missing}")
    print(f"  of which with >=1 detection (ACTIVATION ANOMALY if >0): {n_missing_with_dets}")
    print(f"videos in A/B: {len(pred_post)}")
    if not pred_post:
        print("No videos with pre_ece_boundaries — was the run flag-ON?")
        sys.exit(1)

    res_pre = compute_temporal_map(gt, pred_pre, fps=args.fps)
    res_post = compute_temporal_map(gt, pred_post, fps=args.fps)

    print("\n--- deterministic intra-run A/B (official greedy matcher) ---")
    print(f"A (pre-ECE):  {json.dumps(res_pre, default=float)}")
    print(f"B (post-ECE): {json.dumps(res_post, default=float)}")
    delta = res_post["mAP"] - res_pre["mAP"]
    print(f"\npre mAP  = {res_pre['mAP']:.5f}")
    print(f"post mAP = {res_post['mAP']:.5f}")
    print(f"DELTA    = {delta:+.5f}")
    print("rule: keep iff delta >= +0.015 (and headline sanity); "
          "revert if delta < +0.005; else inconclusive/hold")


if __name__ == "__main__":
    main()
