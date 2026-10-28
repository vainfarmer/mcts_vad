"""
UCF-Crime 批量评测脚本 (v5)

v5: 双 VLM 模型 + 优化参数
"""

import argparse
try:
    from .llm_usage import track_batch_usage, video_usage_all_attempts
except ImportError:
    from llm_usage import track_batch_usage, video_usage_all_attempts

import asyncio
import json
import os
import sys
import time
import uuid
from api_recovery import (add_api_arguments, client_retry_options, is_retryable,
                          run_video_jobs, validate_delay)
from pathlib import Path
from dataset_index import parquet_directory, resolve_video

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import DashScopeLLMClient
from run import UCF_CRIME_CATEGORIES
from config import VADConfig
from mcts_config import add_mcts_arguments
from experiment import prepare_manifest
from factory import build_vad
from evaluate import load_ground_truth, evaluate
import vad_logger

logger = vad_logger.get("vad.batch")


def select_videos(dataset_dir: str, max_per_category: int, split: str = "test",
                  categories=None):
    data_dir = parquet_directory(dataset_dir)
    parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No {split} parquet files in {data_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    if categories:  # optional category filter (eval harness only; pipeline untouched)
        keep = {c.strip() for c in categories}
        df = df[df["category"].isin(keep)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No videos match --categories {sorted(keep)}")
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
    result["video_path"] = result["video_path"].map(lambda p: resolve_video(dataset_dir, p))
    return result


async def process_one_video(vad, video_path, output_file, video_id, category,
                             idx, total, semaphore, fingerprint=None, attempt=1):
    async with semaphore:
        if os.path.exists(output_file):
            existing = json.loads(Path(output_file).read_text(encoding='utf-8'))
            if existing.get('experiment_fingerprint') != fingerprint:
                raise ValueError(f'Result fingerprint mismatch: {output_file}')
            logger.info(f"[{idx}/{total}] SKIP {video_id} (result exists)")
            return dict(status='skipped',elapsed=0.,retryable=False,error_type=None)
        logger.info(f"[{idx}/{total}] START {video_id} ({category}) attempt={attempt}")
        t0 = time.time()
        try:
            result = await vad.detect(video_path=video_path)
            elapsed = time.time() - t0
            serializable = result.to_dict()
            serializable['experiment_fingerprint'] = fingerprint
            serializable['diagnosis_attempt'] = attempt
            serializable['token_usage_all_attempts'] = video_usage_all_attempts(video_path)
            serializable["processing_time"] = round(elapsed, 2)
            temp = Path(output_file).with_suffix('.json.tmp')
            temp.write_text(json.dumps(serializable, indent=2, ensure_ascii=False),encoding='utf-8')
            temp.replace(output_file)
            n_anomalies = len(result.anomalies)
            logger.info(f"[{idx}/{total}] DONE  {video_id}: "
                        f"{result.duration:.0f}s video, {n_anomalies} anomalies, "
                        f"{elapsed:.1f}s processing")
            return dict(status='succeeded',elapsed=elapsed,retryable=False,error_type=None)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{idx}/{total}] FAIL  {video_id}: {e} ({elapsed:.1f}s)", exc_info=True)
            return dict(status='failed',elapsed=elapsed,retryable=is_retryable(e),error_type=type(e).__name__)




@track_batch_usage
async def run_batch(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging — key events go to run.log with timestamps
    log_file = output_dir / "run.log"
    vad_logger.setup(log_file=log_file)
    logger.info(f"VAD batch evaluation started | output_dir={output_dir} | log={log_file}")

    _cats = getattr(args, "categories", None)
    _cats = _cats.split(",") if _cats else None
    videos = select_videos(str(dataset_dir), args.max_per_category, categories=_cats)
    total = len(videos)

    retry_rounds=getattr(args,'video_retry_rounds',1)
    retry_low=getattr(args,'video_retry_min_seconds',5.)
    retry_high=getattr(args,'video_retry_max_seconds',15.)
    validate_delay(retry_low,retry_high)
    if retry_rounds<0 or args.concurrency<1:
        raise ValueError('Video retries must be nonnegative; concurrency must be positive')
    api_options=client_retry_options(args)
    if api_options['api_concurrency']<1:
        raise ValueError('api_concurrency must be positive')
    api_semaphore=asyncio.Semaphore(api_options['api_concurrency'])
    # 两个客户端共享一个API并发池，等待重试不占槽位。
    # 双模型：轻量模型做检测/MCTS，强模型做分类
    detect_client = DashScopeLLMClient(
        model=args.detect_model,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
        api_semaphore=api_semaphore, **api_options,
    )
    classify_client = DashScopeLLMClient(
        model=args.classify_model,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
        api_semaphore=api_semaphore, **api_options,
    )

    cfg = VADConfig.from_args(args)
    cfg.anomaly_categories = UCF_CRIME_CATEGORIES
    manifest_path = output_dir / 'metadata' / 'manifest.json'
    if not manifest_path.exists() and list(output_dir.glob('*.json')):
        raise ValueError('Existing results have no v3 manifest; choose a fresh directory')
    ids = [Path(v).stem for v in videos['video_path']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate video stems would overwrite results')
    manifest = prepare_manifest(manifest_path,cfg,[detect_client,classify_client],
                                [str(dataset_dir / v) for v in videos['video_path']],
                                runtime_settings={'video_concurrency':args.concurrency,'video_retry_rounds':retry_rounds,
                                                  'video_retry_min_seconds':retry_low,'video_retry_max_seconds':retry_high})
    vad = build_vad(cfg, detect_client, classify_client)

    semaphore = asyncio.Semaphore(args.concurrency)
    batch_start = time.time()

    jobs = []
    for idx, (_, row) in enumerate(videos.iterrows()):
        video_path = str(dataset_dir / row["video_path"])
        video_id = Path(row["video_path"]).stem
        output_file = str(output_dir / f"{video_id}.json")
        jobs.append(dict(video_path=video_path,output_file=output_file,video_id=video_id,
                         category=row['category'],idx=idx+1))

    logger.info(f"Starting batch processing: {total} videos, concurrency={args.concurrency}")
    logger.info(f"Detect model: {args.detect_model}, Classify model: {args.classify_model}")
    logger.info("=" * 60)

    attempt_file=output_dir/'metadata'/('video_attempts_'+uuid.uuid4().hex+'.jsonl')
    def record(entry):
        with attempt_file.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
    async def execute(job,attempt):
        return await process_one_video(vad,job['video_path'],job['output_file'],job['video_id'],
            job['category'],job['idx'],total,semaphore,manifest['fingerprint'],attempt)
    results,history = await run_video_jobs(jobs,execute,retry_rounds,retry_low,retry_high,record)
    batch_elapsed = time.time() - batch_start

    succeeded = sum(r['status']=='succeeded' for r in results)
    failed = sum(r['status']=='failed' for r in results)
    skipped = sum(r['status']=='skipped' for r in results)
    total_proc_time = sum(r['elapsed'] for r in history)

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


    (output_dir / 'metadata' / 'run_status.json').write_text(json.dumps({
        'succeeded':succeeded,'failed':failed,'skipped':skipped,'total':total,
        'attempts':len(history),'attempt_log':attempt_file.name,'recovered_videos':sum(r['status']=='succeeded' and r['attempt']>1 for r in history)}),encoding='utf-8')
    if failed:
        raise RuntimeError(f'{failed} videos failed; incomplete results are not evaluated')
    logger.info("Running evaluation...")
    label_path = str(parquet_directory(dataset_dir))
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
        "recovery": {"attempt_log":attempt_file.name,"attempts":len(history),
                     "recovered_videos":sum(r["status"]=="succeeded" and r["attempt"]>1 for r in history)},
    }
    report_file = str(output_dir / "evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Full report saved to: {report_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime Batch Evaluation (v5)")

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/batch_eval")
    parser.add_argument("--max_per_category", type=int, default=5)
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category filter (eval harness only), "
                             "e.g. Shooting,RoadAccidents,Shoplifting,Explosion")
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
    parser.add_argument("--cbt_tighten", action="store_true", dest="cbt_enabled",
                        help="v155 CBT: TERMINAL (post-BUE) inward tighten of "
                             "over-wide hard-category detections "
                             "(Shooting/RoadAccidents/Shoplifting/Explosion) "
                             "toward per-category GT-p75 caps via AutoLoc "
                             "outer-inner contrast window selection "
                             "(count-neutral, confidence-frozen, monotone-inward; "
                             "default ON since v164 promotion)")
    parser.add_argument("--no_cbt_tighten", action="store_false", dest="cbt_enabled",
                        help="disable v155 CBT tightening (opt-out of default-ON)")
    parser.set_defaults(cbt_enabled=True)
    parser.add_argument("--cbt_recal", action="store_true", dest="cbt_recal",
                        help="v160 CBT-RECAL: GT-p75 recal caps + trigger 1.1 "
                             "(use with --cbt_tighten; default ON since v164)")
    parser.add_argument("--no_cbt_recal", action="store_false", dest="cbt_recal",
                        help="disable CBT-RECAL, revert to v155 caps/trigger")
    parser.set_defaults(cbt_recal=True)

    # Backward compatibility
    parser.add_argument("--model", type=str, default=None,
                        help="(deprecated) Use --detect_model and --classify_model instead")

    add_api_arguments(parser)
    parser.add_argument("--video_retry_rounds",type=int,default=1)
    parser.add_argument("--video_retry_min_seconds",type=float,default=5.)
    parser.add_argument("--video_retry_max_seconds",type=float,default=15.)
    add_mcts_arguments(parser)
    args = parser.parse_args()
    if args.model:
        args.detect_model = args.model
        args.classify_model = args.model

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY environment variable not set")
        sys.exit(1)

    asyncio.run(run_batch(args))
