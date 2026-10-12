"""
UCF-Crime 批量评测脚本 (v5)

v5: 双 VLM 模型 + 优化参数
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import DashScopeLLMClient
from run import UCF_CRIME_CATEGORIES
from config import VADConfig
from factory import build_vad
from evaluate import load_ground_truth, evaluate
import vad_logger

logger = vad_logger.get("vad.batch")


def select_videos(dataset_dir: str, max_per_category: int, split: str = "test"):
    data_dir = Path(dataset_dir) / "data"
    parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No {split} parquet files in {data_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    selected = []
    summary = {}
    for cat, group in df.groupby("category"):
        n = min(max_per_category, len(group))
        sampled = group.sample(n=n, random_state=42).sort_values("video_path")
        selected.append(sampled)
        summary[cat] = n
    result = pd.concat(selected, ignore_index=True)
    logger.info(f"Selected {len(result)} videos from {len(summary)} categories:")
    for cat, n in sorted(summary.items()):
        total = len(df[df["category"] == cat])
        logger.info(f"  {cat:20s}: {n:3d} / {total}")
    return result


async def process_one_video(vad, video_path, output_file, video_id, category,
                             idx, total, semaphore):
    async with semaphore:
        if os.path.exists(output_file):
            logger.info(f"[{idx}/{total}] SKIP {video_id} (result exists)")
            return "skipped", 0
        logger.info(f"[{idx}/{total}] START {video_id} ({category})")
        t0 = time.time()
        try:
            result = await vad.detect(video_path=video_path)
            elapsed = time.time() - t0
            serializable = result.to_dict()
            serializable["processing_time"] = round(elapsed, 2)
            with open(output_file, "w") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            n_anomalies = len(result.anomalies)
            logger.info(f"[{idx}/{total}] DONE  {video_id}: "
                        f"{result.duration:.0f}s video, {n_anomalies} anomalies, "
                        f"{elapsed:.1f}s processing")
            return "succeeded", elapsed
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{idx}/{total}] FAIL  {video_id}: {e} ({elapsed:.1f}s)", exc_info=True)
            return "failed", elapsed


def check_ece_activation_gate(log_file, succeeded, enabled):
    """v105 ECE FM5 hard gate — active ONLY when --event_continuation_extension
    is passed (flag-off runs are byte-identical to prior behavior).

    Counts '[ECE] summary:' INFO markers in the run log; the pipeline emits
    exactly one per processed video (including both early-return paths and
    all-skip videos), so count < succeeded means the mechanism never
    activated on some video (FM5). Returns (ok, count).
    """
    if not enabled:
        return True, 0
    count = 0
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            count = sum(1 for line in f if "[ECE] summary:" in line)
    except OSError:
        count = 0
    return count >= succeeded, count


async def run_batch(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging — key events go to run.log with timestamps
    log_file = output_dir / "run.log"
    vad_logger.setup(log_file=log_file)
    logger.info(f"VAD batch evaluation started | output_dir={output_dir} | log={log_file}")

    videos = select_videos(str(dataset_dir), args.max_per_category)
    total = len(videos)

    # 双模型：轻量模型做检测/MCTS，强模型做分类
    detect_client = DashScopeLLMClient(
        model=args.detect_model,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
    )
    classify_client = DashScopeLLMClient(
        model=args.classify_model,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
    )

    cfg = VADConfig.from_args(args)
    cfg.anomaly_categories = UCF_CRIME_CATEGORIES
    vad = build_vad(cfg, detect_client, classify_client)

    semaphore = asyncio.Semaphore(args.concurrency)
    batch_start = time.time()

    tasks = []
    for idx, (_, row) in enumerate(videos.iterrows()):
        video_path = str(dataset_dir / row["video_path"])
        video_id = Path(row["video_path"]).stem
        output_file = str(output_dir / f"{video_id}.json")
        tasks.append(process_one_video(
            vad, video_path, output_file, video_id, row["category"],
            idx + 1, total, semaphore))

    logger.info(f"Starting batch processing: {total} videos, concurrency={args.concurrency}")
    logger.info(f"Detect model: {args.detect_model}, Classify model: {args.classify_model}")
    logger.info("=" * 60)

    results = await asyncio.gather(*tasks)
    batch_elapsed = time.time() - batch_start

    succeeded = sum(1 for s, _ in results if s == "succeeded")
    failed = sum(1 for s, _ in results if s == "failed")
    skipped = sum(1 for s, _ in results if s == "skipped")
    total_proc_time = sum(t for _, t in results)

    logger.info("=" * 60)
    logger.info("Batch complete:")
    logger.info(f"  Succeeded: {succeeded}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"  Total wall-clock time:   {batch_elapsed:.1f}s ({batch_elapsed/60:.1f}min)")
    logger.info(f"  Total processing time:   {total_proc_time:.1f}s ({total_proc_time/60:.1f}min)")
    if succeeded > 0:
        logger.info(f"  Avg per video:           {total_proc_time/max(succeeded,1):.1f}s")
        speedup = total_proc_time / batch_elapsed if batch_elapsed > 0 else 1
        logger.info(f"  Parallelism speedup:     {speedup:.2f}x")

    logger.info("=" * 60)

    # v106 FM5 hard gate: with --event_continuation_extension, every processed
    # video must have emitted one '[ECE] summary:' INFO marker into run.log.
    ok, cnt = check_ece_activation_gate(
        log_file, succeeded, getattr(args, "event_continuation_extension", False))
    logger.info(f"[ECE-GATE] markers={cnt} succeeded={succeeded} ok={ok}")
    if not ok:
        logger.error("[ECE-GATE] FM5: '[ECE] summary:' count < succeeded — "
                     "refusing to write evaluation_report.json")
        sys.exit(2)

    logger.info("Running evaluation...")
    label_path = str(dataset_dir / "data")
    eval_results = evaluate(label_path, str(output_dir), fps=30, split="test")

    report = {
        "timing": {
            "wall_clock_seconds": round(batch_elapsed, 2),
            "total_processing_seconds": round(total_proc_time, 2),
            "avg_per_video_seconds": round(total_proc_time / max(succeeded, 1), 2),
            "concurrency": args.concurrency,
            "num_videos": total,
            "succeeded": succeeded, "failed": failed, "skipped": skipped,
        },
        "evaluation": eval_results,
    }
    report_file = str(output_dir / "evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Full report saved to: {report_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime Batch Evaluation (v5)")

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/batch_eval")
    parser.add_argument("--max_per_category", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=3)

    parser.add_argument("--detect_model", type=str, default="qwen-vl-plus",
                        help="Light VLM for detection/MCTS")
    parser.add_argument("--classify_model", type=str, default="qwen3-vl-plus",
                        help="Strong VLM for classification")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--min_frames", type=int, default=3)
    parser.add_argument("--fps_sample", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--segment_duration", type=float, default=6.0)
    parser.add_argument("--segment_overlap", type=float, default=0.25)
    parser.add_argument("--mcts_simulations", type=int, default=12)
    parser.add_argument("--mcts_max_depth", type=int, default=4)
    parser.add_argument("--motion_threshold", type=float, default=1.0)
    parser.add_argument("--min_motion_interval", type=float, default=2.0)
    parser.add_argument("--clip_model", type=str, default="ViT-L-14")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--full_coverage_scan", action="store_true",
                        help="v97 FCCS: full-coverage comparative VLM scan "
                             "(default off = prior partial-coverage behavior)")
    parser.add_argument("--event_unit_boundary_contraction", action="store_true",
                        help="v100 EUBC: event-unit boundary contraction of "
                             "over-wide final detections (default off = prior "
                             "behavior)")
    parser.add_argument("--event_continuation_extension", action="store_true",
                        help="v106 ECE: whole-video half-max continuation of "
                             "under-extended final detections (default off = "
                             "prior behavior)")
    parser.add_argument("--qscore_rerank", action="store_true", default=True,
                        help="v111 QSCORE: pure post-hoc localization-quality "
                             "confidence re-rank of final detections. v124: now "
                             "DEFAULT-ON (committed v114 baseline, map=0.15632).")
    parser.add_argument("--no_qscore_rerank", action="store_false",
                        dest="qscore_rerank",
                        help="v124: disable the default-ON Q-score re-rank, "
                             "restoring the pre-v114 baseline (map~0.15013).")
    parser.add_argument("--classify_skip_class_agnostic", action="store_true",
                        default=True,
                        help="v137 NOCLS-CA: skip the Stage-7 classify VLM "
                             "sub-tree (~-7.5%% VLM calls). DEFAULT-ON, "
                             "mAP-neutral by construction (anomaly_type unused "
                             "by temporal mAP); degrades only category accuracy.")
    parser.add_argument("--no_classify_skip_class_agnostic",
                        action="store_false",
                        dest="classify_skip_class_agnostic",
                        help="v137: disable the default-ON classify-path DROP, "
                             "restoring full VLM category classification.")
    # v141 CEX (default OFF = byte-identical prior behavior)
    parser.add_argument("--cex_extend", action="store_true",
                        help="v141 CEX: continuation-evidence extension — grow "
                             "each FINAL detection outward in cex_step_sec steps "
                             "on a confident binary VLM 'same event still "
                             "ongoing?' vote (count-preserving, "
                             "confidence-frozen; default off = prior behavior)")
    parser.add_argument("--cex_min_conf", type=float, default=0.7,
                        help="v141 CEX: accept threshold on the probe's "
                             "self-reported confidence (sets the "
                             "false-extension rate q; pre-registered 0.7)")
    parser.add_argument("--cex_max_steps", type=int, default=3,
                        help="v141 CEX: max extension rounds per side "
                             "(pre-registered 3)")
    parser.add_argument("--cex_step_sec", type=float, default=6.0,
                        help="v141 CEX: seconds grown per accepted step "
                             "(pre-registered 6.0 = the scan segment length)")
    parser.add_argument("--cex_max_total_sec", type=float, default=60.0,
                        help="v141 CEX: absolute per-detection growth cap in "
                             "seconds (pre-registered 60.0)")
    # v142 BUE (default OFF = byte-identical prior behavior)
    parser.add_argument("--bue_extension", action="store_true",
                        dest="bue_enabled",
                        help="v142 BUE: boundary-union extension — grow each "
                             "FINAL detection outward to the outer-inner-contrast "
                             "(OIC) argmax window on the read-only category "
                             "curve, UNION-ONLY, length-capped at bue_lmax_sec "
                             "(count-preserving, confidence-frozen, zero new "
                             "VLM/CLIP calls; default off = prior behavior)")
    parser.add_argument("--no_bue_extension", action="store_false",
                        dest="bue_enabled",
                        help="v142 BUE: explicitly disable BUE (byte-identical "
                             "prior behavior; this is already the default)")
    parser.add_argument("--bue_alpha", type=float, default=1.0,
                        help="v142 BUE: outer ring width as a multiple of the "
                             "candidate window length (pre-registered 1.0)")
    parser.add_argument("--bue_lmax_sec", type=float, default=40.0,
                        help="v142 BUE: max candidate window seconds "
                             "(pre-registered 40.0; M6 — 40 and 50 are "
                             "bit-identical because 0.8*duration binds above 40)")
    parser.add_argument("--no_bue_closure_gate", action="store_false",
                        dest="bue_closure_gate",
                        help="v142 BUE: disable the v51 half-max closure test "
                             "(measured mAP no-op; the gate only reduces blast "
                             "radius, 67 fires instead of 78)")
    parser.set_defaults(bue_enabled=True, bue_closure_gate=True)

    # Backward compatibility
    parser.add_argument("--model", type=str, default=None,
                        help="(deprecated) Use --detect_model and --classify_model instead")

    args = parser.parse_args()
    if args.model:
        args.detect_model = args.model
        args.classify_model = args.model

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY environment variable not set")
        sys.exit(1)

    asyncio.run(run_batch(args))
