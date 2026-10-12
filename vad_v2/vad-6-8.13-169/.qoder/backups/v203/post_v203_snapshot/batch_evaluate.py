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
from config import VADConfig, AB_ABLATE_CHOICES
from factory import build_vad
from evaluate import load_ground_truth, evaluate
import vad_logger

logger = vad_logger.get("vad.batch")


def select_videos(dataset_dir: str, max_per_category: int, split: str = "test",
                  categories=None):
    data_dir = Path(dataset_dir) / "data"
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

    _cats = getattr(args, "categories", None)
    _cats = _cats.split(",") if _cats else None
    videos = select_videos(str(dataset_dir), args.max_per_category, categories=_cats)
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
    parser.add_argument("--cbt_all_cats", action="store_true", dest="cbt_all_cats",
                        default=False,
                        help="v203 CBT-ALL: extend the CBT caps dict KEY SET from "
                             "the 4 hard categories to all 13, using caps derived "
                             "from CATEGORY_DURATION_PRIORS expected_duration max "
                             "(no test GT). Method/trigger/min_segment floor and the "
                             "4 committed cap VALUES unchanged. Fires only where a "
                             "detection's RESOLVED anomaly_type is one of the 9 "
                             "extended cats: on the committed 290 run 31 dets are "
                             "addressable, only 4 of them on a hard-88 video, so "
                             "hard-88 is ~5x under-powered — judge at full-290 "
                             "(--max_per_category 1000) with --ab_ablate cbt_all. "
                             "Use with --cbt_tighten; default OFF")
    parser.add_argument("--no_cbt_all_cats", action="store_false", dest="cbt_all_cats",
                        help="disable v203 CBT-ALL (explicit default)")
    parser.add_argument("--cgr_relocate", action="store_true", dest="cgr_enabled",
                        help="v167 CGR-M: PRE-BUE count-neutral motion-energy-peak "
                             "RELOCATION of the lowest-confidence detection onto an "
                             "uncovered motion peak, gated to the 4 hard categories "
                             "(Shooting/RoadAccidents/Shoplifting/Explosion); "
                             "width-neutral, confidence-frozen, count-neutral "
                             "(default ON since v167)")
    parser.add_argument("--no_cgr_relocate", action="store_false", dest="cgr_enabled",
                        help="disable CGR-M motion-peak relocation (revert to pre-v167 behavior)")
    parser.set_defaults(cgr_enabled=False)
    parser.add_argument("--ccr_relocate", action="store_true", dest="ccr_enabled",
                        help="v168 CCR-SE: PRE-BUE (post-CGR) count-neutral CLIP-"
                             "category-curve semantic RELOCATION of the lowest-"
                             "confidence detection onto an uncovered curve peak, "
                             "gated to Shoplifting/Shooting; width-neutral, "
                             "confidence-frozen, count-neutral (default OFF)")
    parser.add_argument("--no_ccr_relocate", action="store_false", dest="ccr_enabled",
                        help="disable CCR-SE category-curve relocation (default)")
    parser.set_defaults(ccr_enabled=False)
    parser.add_argument("--cgr_ab_diag", action="store_true", dest="cgr_ab_diag",
                        help="v169 CGR-AB: DUAL-ARM in-run A/B instrument for CGR-M "
                             "(requires --cgr_relocate; pure diagnostic, the run's "
                             "real output stays byte-identical). Replays the exact "
                             "downstream chain (CCR-if-on -> BUE -> CBT) on a DEEP "
                             "COPY of the pre-CGR detection list and persists the "
                             "result as ab_arm_off_boundaries, so the run's own "
                             "final output is arm-ON and that field is arm-OFF -> "
                             "an EXACT paired A/B with zero cross-run VLM noise. "
                             "Adjudicate with analysis_bootstrap_ab.py "
                             "--paired-json ab_arm_off_boundaries (default OFF)")
    parser.add_argument("--no_cgr_ab_diag", action="store_false", dest="cgr_ab_diag",
                        help="disable the CGR-AB dual-arm instrument (default)")
    parser.set_defaults(cgr_ab_diag=False)
    parser.add_argument("--ab_ablate", dest="ab_ablate", default="",
                        choices=list(AB_ABLATE_CHOICES), metavar="MECH",
                        help="v27-RIG AB-ABLATE: OPERATOR-AGNOSTIC dual-arm in-run "
                             "A/B instrument (generalization of --cgr_ab_diag from "
                             "CGR-only to any terminal-slot operator). PURE "
                             "DIAGNOSTIC: the run's real output stays byte-identical "
                             "(default '' = OFF: no snapshot, no replay, no field, no "
                             "marker). MECH in {" + "|".join(
                                 m for m in AB_ABLATE_CHOICES if m) + "}. arm-ON = "
                             "this run's real output; arm-OFF = the SAME deterministic "
                             "terminal chain (ECE->EUBC->proj->frame_scores->qscore->"
                             "CGR->CCR->BUE->CBT->CNV->SNH->SVA) replayed on a DEEP "
                             "COPY of the pre-ECE detection list with MECH's whole "
                             "config family reset to committed ScoringConfig() "
                             "defaults. Persists ab_arm_off_boundaries AND "
                             "ab_arm_off_confidences (index-aligned, len == "
                             "len(anomalies)) -> an EXACT paired A/B with zero "
                             "cross-run VLM noise. Adjudicate boundary operators with "
                             "analysis_bootstrap_ab.py --paired-json "
                             "ab_arm_off_boundaries; adjudicate confidence-veto "
                             "operators (snh/cnv, which never move a window, so a "
                             "boundary-only compare is an EXACT-ZERO FALSE NULL) with "
                             "--paired-log <dir>/run.log --marker AB-FIRE "
                             "--paired-mode conf. NOT usable for CEX/VLC/DSV (they "
                             "issue VLM calls) or FSD (upstream of the VLM)")
    parser.add_argument("--cnv_veto", action="store_true", dest="cnv_veto",
                        help="v172 CNV variant TIGHTEN: Cascade Null-Hypothesis "
                             "Veto — TERMINAL-slot (post-BUE/CBT/CEX) ONE-SIDED "
                             "confidence veto (confidence := 0.0, detection KEPT "
                             "count-frozen, never moves/adds/re-ranks). Stage 1 "
                             "vetoes ALL detections of a video iff MIL top-k "
                             "mean < 0.20 AND fused-curve in-video z-max < 1.5 "
                             "AND motion outlier fraction < 0.01 AND the video's "
                             "videomae_category is 'Normal' (v172 4th conjunct); "
                             "stage 2 vetoes detections whose window curve mass "
                             "is below the video's own median (v172: z_window < "
                             "0, kept with the v171 z_window < 0.5 bound) and "
                             "not covering the curve argmax. Zero new "
                             "VLM/CLIP/MIL calls; arm-OFF restored from "
                             "[CNV-FIRE] log lines via analysis_bootstrap_ab.py "
                             "--paired-mode conf (default OFF)")
    parser.add_argument("--no_cnv_veto", action="store_false", dest="cnv_veto",
                        help="disable v172 CNV veto (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(cnv_veto=False)
    parser.add_argument("--snh_veto", action="store_true", dest="snh_veto",
                        help="v179 SNH (Supervised Normality Head): VIDEO-LEVEL-ONLY "
                             "P(normal) gate at the TERMINAL slot (post-BUE/CBT/CEX, "
                             "on the genuinely-final list; NO per-segment stage). "
                             "Loads the v19-trained model_snh.json (CV AUC 0.996, "
                             "tau=0.7894), computes the 16 cached-signal features "
                             "via the SHARED snh_features.compute_snh_features "
                             "(train/eval skew-free: MIL snippet scores, 13 "
                             "cat/(cat+normal) CLIP curves, Normal-prompt softmax "
                             "curve, motion energy — zero new VLM/CLIP/MIL calls), "
                             "and iff P(normal) >= tau vetoes ALL of the video's "
                             "detections (confidence := 0.0, detection KEPT "
                             "count-frozen, never moves/adds/re-ranks). Missing/"
                             "corrupt model -> [SNH-ERR] and the video is treated "
                             "as flag-off (never crashes). Arm-OFF restored from "
                             "[SNH-FIRE] log lines via analysis_bootstrap_ab.py "
                             "--paired-log --marker SNH-FIRE --paired-mode conf; "
                             "one unconditional [SNH-VID] per video on every path "
                             "(default OFF = byte-identical: model file and signals "
                             "never read)")
    parser.add_argument("--no_snh_veto", action="store_false", dest="snh_veto",
                        help="disable v179 SNH veto (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(snh_veto=False)
    # v196 VLC (default OFF = byte-identical prior behavior)
    parser.add_argument("--vlc_calib", action="store_true", dest="vlc_calib",
                        help="v196 VLC (VLM video-Level Calibration): TERMINAL-slot "
                             "(post-CNV/SNH, on the genuinely-final list) VIDEO-LEVEL "
                             "confidence calibration. For each video with >=1 final "
                             "detection, samples 16 frames UNIFORMLY across the whole "
                             "duration and issues ONE extra detect-client (qwen-vl-plus) "
                             "video-level call ('Does this video contain any abnormal "
                             "activity...') -> v in [0,1] (any failure -> v=0.5 + "
                             "[VLC-PARSE-FALLBACK]); then every detection: "
                             "conf *= (0.5 + 0.5*v). Count-frozen, windows-frozen, "
                             "order-frozen (confidence only). 0-detection videos SKIP "
                             "the call (cost) but still log [VLC-VID]. Arm-OFF restored "
                             "from [VLC-FIRE] log lines via analysis_bootstrap_ab.py "
                             "--paired-log --marker VLC-FIRE --paired-mode conf; one "
                             "unconditional [VLC-VID] per video on every path "
                             "(default OFF = byte-identical: no extra VLM call)")
    parser.add_argument("--no_vlc_calib", action="store_false", dest="vlc_calib",
                        help="disable v196 VLC calibration (byte-identical prior "
                             "behavior; this is already the default)")
    parser.set_defaults(vlc_calib=False)
    # v197 DSV (default OFF = byte-identical prior behavior)
    parser.add_argument("--dsv_verify", action="store_true", dest="dsv_verify",
                        help="v197 DSV (Dual-model Second-opinion Verification): "
                             "TERMINAL-slot (post-CNV/SNH/VLC, on the genuinely-final "
                             "list) DETECTION-LEVEL confidence re-scoring. For each "
                             "FINAL detection (COST CAP: top-8 by confidence per "
                             "video), <=8 frames sampled uniformly inside "
                             "[t_start, t_end] are re-read by the INDEPENDENT "
                             "classify client (qwen3-vl-plus — a different model "
                             "from the qwen-vl-plus scan VLM): 'Rate whether this "
                             "clip genuinely shows <category>-type abnormal "
                             "activity...' -> a in [0,1] (any failure -> a=0.5 + "
                             "[DSV-PARSE-FALLBACK]); then conf *= (0.6 + 0.4*a). "
                             "Count-frozen, windows-frozen, order-frozen "
                             "(confidence only). 0-detection videos make NO call "
                             "but still log [DSV-VID]. Arm-OFF restored from "
                             "[DSV-FIRE] log lines via analysis_bootstrap_ab.py "
                             "--paired-log --marker DSV-FIRE --paired-mode conf; "
                             "one unconditional [DSV-VID] per video on every path "
                             "(default OFF = byte-identical: no extra VLM call)")
    parser.add_argument("--no_dsv_verify", action="store_false", dest="dsv_verify",
                        help="disable v197 DSV verification (byte-identical prior "
                             "behavior; this is already the default)")
    parser.set_defaults(dsv_verify=False)
    # v198 MCS (default OFF = byte-identical prior behavior)
    parser.add_argument("--mcs_curves", action="store_true", dest="mcs_curves",
                        help="v198 MCS (MaxCat curve Selection for BUE/CBT): feed "
                             "BUE and CBT the POINTWISE MAX over all 13 category "
                             "curves (maxcat(t) = max_c curve_c(t)) instead of the "
                             "resolved-category curve. Deterministic post-VLM "
                             "(cached read-only CLIP curves only — zero new "
                             "VLM/CLIP calls), count-frozen, conf-frozen; CBT-recal "
                             "caps stay keyed on the RESOLVED category in both "
                             "arms. Dual-arm: the committed resolved-curve "
                             "BUE+CBT(+recal) replay on a deep copy of the pre-BUE "
                             "list is written as the top-level per-video JSON "
                             "field pre_mcs_boundaries (index-aligned with "
                             "anomalies; arm-OFF rebuilt EXACTLY via "
                             "analysis_bootstrap_ab.py --paired-json "
                             "pre_mcs_boundaries, length-mismatch HARD-VOID "
                             "inherited); the real output is the maxcat arm. One "
                             "unconditional [MCS-VID] per video on every path, "
                             "[MCS-FIRE] per changed window (default OFF = "
                             "byte-identical: no maxcat curve is even computed)")
    parser.add_argument("--no_mcs_curves", action="store_false", dest="mcs_curves",
                        help="disable v198 MCS (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(mcs_curves=False)
    parser.add_argument("--kfd_uniform_keyframes", action="store_true",
                        dest="kfd_uniform_keyframes",
                        help="v173 KFD-U: replace the comparative scan's first-k "
                             "keyframe truncation (VLM sees only the leading "
                             "~0.5s of each 6s segment) with a deterministic "
                             "uniform spread — k=2: frames nearest "
                             "{t_start, t_start+3.0s}; the >60-segment "
                             "1-keyframe rule: the frame nearest the segment "
                             "midpoint. Same segment grid / batch geometry "
                             "(8x2, 16x1) / 16 frames-per-call / call count / "
                             "prompts / thresholds / merge / top-k — zero "
                             "latency cost. The truncation exists only on the "
                             "comparative scan path, so this flag runs Stage 4 "
                             "through that path. Emits [KFD-FIRE] per affected "
                             "batch and one unconditional [KFD-VID] per video "
                             "(default OFF = byte-identical committed pipeline)")
    parser.add_argument("--no_kfd_uniform_keyframes", action="store_false",
                        dest="kfd_uniform_keyframes",
                        help="disable v173 KFD-U (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(kfd_uniform_keyframes=False)
    parser.add_argument("--kfd_k", type=int, default=2, choices=[1, 2, 3],
                        dest="kfd_k",
                        help="v173 KFD-U spread arm: 2 = {t, t+3.0s} "
                             "(pre-registered default); 3 = {t, t+2.5s, t+5.0s} "
                             "(fallback arm, batch geometry 8x3, +50%% scan "
                             "tokens); 1 = segment midpoint")
    parser.add_argument("--tcp_prompt", action="store_true",
                        dest="tcp_prompt",
                        help="v177 TCP (Temporal-Contrast Prompt): append the "
                             "pre-registered 'Temporal reasoning instructions' "
                             "block to the scan_segment prompt immediately "
                             "before 'Respond ONLY in JSON format' — judge by "
                             "the single most anomalous moment, do NOT average "
                             "the clip, compare early vs late frames for sudden "
                             "change. Prompt-side ONLY: same frames, same call "
                             "count, same directed path, same JSON schema "
                             "(is_anomalous/confidence/description) — zero "
                             "latency cost. Emits one unconditional [TCP-VID] "
                             "per video on every path (+ [TCP-BLOCK] / "
                             "[TCP-PROMPT] once per video) (default OFF = "
                             "prompt bit-identical to committed)")
    parser.add_argument("--no_tcp_prompt", action="store_false",
                        dest="tcp_prompt",
                        help="disable v177 TCP (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(tcp_prompt=False)
    parser.add_argument("--fsd_cats", action="store_true",
                        dest="fsd_cats",
                        help="v182 FSD (Fine Segment Division): when the "
                             "pipeline-resolved category is Shoplifting or "
                             "RoadAccidents, the DIRECTED Stage-4 scan "
                             "(scan_segment path) iterates 3s HALF-segments "
                             "(each SELECTED 6s segment split into its 2 "
                             "chronological halves, ~6 frames at 2fps; "
                             ">=3-frame halves kept) instead of the 6s grid. "
                             "Segment selection (candidate build, pre-screen, "
                             "caps) is computed on the ORIGINAL 6s grid "
                             "BEFORE splitting; same scan_segment call/prompt/"
                             "schema per sub-segment; merge/NMS/classify/"
                             "BUE/CBT untouched; the >60-segment long-video "
                             "rule stays on the original 6s grid; FCCS "
                             "unaffected (stays OFF). Emits [FSD-FIRE] once "
                             "per gated video (n_segments_6s -> n_segments_3s) "
                             "and one unconditional [FSD-VID] per video on "
                             "every path (default OFF = byte-identical "
                             "committed behavior)")
    parser.add_argument("--no_fsd_cats", action="store_false",
                        dest="fsd_cats",
                        help="disable v182 FSD (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(fsd_cats=False)
    parser.add_argument("--fds_scan", action="store_true",
                        dest="fds_scan",
                        help="v199 FDS (Frame-Density Scan): when the "
                             "pipeline-resolved category is Shooting, "
                             "RoadAccidents, Shoplifting or Explosion "
                             "(pipeline.FDS_CATEGORIES), the DIRECTED Stage-4 "
                             "scan (scan_segment path) shows the VLM ~24 frames "
                             "per 6s segment (4fps, same uniform placement, "
                             "double density) instead of ~12 (2fps), via a "
                             "once-per-video 2x-fps re-decode threaded as a "
                             "dense frame pool. Same scan_segment call/prompt/"
                             "JSON schema; same 6s segment grid; candidate "
                             "build stays on the ORIGINAL 2fps grid; "
                             "merge/NMS/classify/BUE/CBT untouched; the "
                             ">60-segment long-video rule is FCCS-only and "
                             "keeps its structure; FCCS unaffected (stays "
                             "OFF). Emits [FDS-FIRE] once per gated video "
                             "(frames_per_segment base -> dense) and one "
                             "unconditional [FDS-VID] per video on every path "
                             "(default OFF = byte-identical committed "
                             "behavior)")
    parser.add_argument("--no_fds_scan", action="store_false",
                        dest="fds_scan",
                        help="disable v199 FDS (byte-identical prior behavior; "
                             "this is already the default)")
    parser.set_defaults(fds_scan=False)
    # v200 SVA (default OFF = byte-identical prior behavior)
    parser.add_argument("--sva_calib", action="store_true", dest="sva_calib",
                        help="v200 SVA (Scan-Verdict Aggregate video-level "
                             "calibration): TERMINAL-slot (post-CNV/SNH/VLC/DSV, "
                             "on the genuinely-final list) VIDEO-LEVEL "
                             "confidence calibration using the scan VLM's OWN "
                             "aggregate — ZERO new VLM/CLIP/MIL calls ([COST] "
                             "delta exactly 0). The directed Stage-4 scan "
                             "already produces a per-segment verdict confidence "
                             "for every scanned 6s segment; when ON, those are "
                             "retained per-video (video_data['_sva_scan_confs']) "
                             "and level = mean of the top-3 (mean of ALL if <3; "
                             "clamped to [0,1]); then every detection: "
                             "conf *= (0.5 + 0.5*level). Count-frozen, "
                             "windows-frozen, order-frozen (confidence only). "
                             "Arm-OFF restored from [SVA-FIRE] log lines via "
                             "analysis_bootstrap_ab.py --paired-log --marker "
                             "SVA-FIRE --paired-mode conf; one unconditional "
                             "[SVA-VID] per video on every path (level=NA when "
                             "no segments) (default OFF = byte-identical: no "
                             "retention, no marker)")
    parser.add_argument("--no_sva_calib", action="store_false", dest="sva_calib",
                        help="disable v200 SVA calibration (byte-identical prior "
                             "behavior; this is already the default)")
    parser.set_defaults(sva_calib=False)

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
