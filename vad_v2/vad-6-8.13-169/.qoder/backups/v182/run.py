"""
运行脚本 — 视频异常检测与时间定位

入口脚本，负责：
  1. 解析命令行参数（视频路径、模型配置、搜索超参数等）
  2. 初始化 DashScope VLM 客户端（通过 OpenAI 兼容接口）
  3. 构建 MultiAnomalyVAD Pipeline 并执行检测
  4. 输出结果到终端和 JSON 文件

使用示例：
  export DASHSCOPE_API_KEY=sk-xxx
  python run.py --video_path /path/to/video.mp4 --output_file result.json

  # 使用 GPU 加速 CLIP 编码
  python run.py --video_path video.mp4 --device cuda

  # 调整搜索精度（更多模拟 = 更精确但更慢）
  python run.py --video_path video.mp4 --mcts_simulations 16 --mcts_max_depth 5

依赖：
  pip install opencv-python-headless numpy openai open_clip_torch pillow
"""

import argparse
import asyncio
import json
import os
import time

import vad_logger

import numpy as np

from config import VADConfig
from factory import build_vad
# Re-exported for backward compat (batch_evaluate imports it from here)
from llm_client import DashScopeLLMClient  # noqa: F401

UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]




async def main(args):
    vad_logger.setup()
    llm_client = DashScopeLLMClient(
        api_key=args.api_key,
        base_url=args.api_base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
    )

    cfg = VADConfig.from_args(args)
    cfg.anomaly_categories = UCF_CRIME_CATEGORIES
    vad = build_vad(cfg, llm_client)

    print(f"Processing video: {args.video_path}")
    print(f"Model: {args.model}")
    print(f"Anomaly categories: {', '.join(UCF_CRIME_CATEGORIES)}")
    start_time = time.time()

    result = await vad.detect(
        video_path=args.video_path,
        context=args.context,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Video duration:  {result.duration:.1f}s")
    print(f"Is normal:       {result.is_normal}")
    print(f"Anomalies found: {len(result.anomalies)}")

    for i, anomaly in enumerate(result.anomalies):
        print(f"\n  Anomaly {i+1}:")
        print(f"    Time:       {anomaly.t_start:.1f}s - {anomaly.t_end:.1f}s")
        print(f"    Confidence: {anomaly.confidence:.3f}")
        print(f"    Type:       {anomaly.anomaly_type}")

    if args.output_file:
        serializable = result.to_dict()
        serializable['processing_time'] = round(elapsed, 2)
        with open(args.output_file, 'w') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Anomaly Detection with LLM + Temporal MCTS"
    )

    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--context", type=str, default="surveillance camera",
                        help="Scene context description")

    parser.add_argument("--model", type=str, default="qwen-vl-max",
                        help="DashScope model name")
    parser.add_argument("--api_key", type=str, default=None,
                        help="DashScope API key (or set DASHSCOPE_API_KEY env)")
    parser.add_argument("--api_base_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="DashScope API base URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)

    parser.add_argument("--max_frames", type=int, default=200,
                        help="Max frames per LLM request (hard upper bound)")
    parser.add_argument("--min_frames", type=int, default=3,
                        help="Min frames per segment (ensure LLM can recognize content)")
    parser.add_argument("--fps_sample", type=float, default=2.0,
                        help="Target frames per second for dynamic frame count")

    parser.add_argument("--fps", type=int, default=4,
                        help="Frame sampling rate")
    parser.add_argument("--segment_duration", type=float, default=8.0,
                        help="Initial segment duration in seconds")
    parser.add_argument("--mcts_simulations", type=int, default=8,
                        help="Number of MCTS simulations per candidate")
    parser.add_argument("--mcts_max_depth", type=int, default=4,
                        help="Maximum MCTS search depth")

    parser.add_argument("--motion_threshold", type=float, default=1.0,
                        help="Frame diff threshold for motion filter (0 to disable)")
    parser.add_argument("--min_motion_interval", type=float, default=2.0,
                        help="Min seconds between kept frames in static regions")

    parser.add_argument("--clip_model", type=str, default="ViT-B-32",
                        help="CLIP model name for anomaly expert")
    parser.add_argument("--clip_pretrained", type=str, default="openai",
                        help="Pretrained weights (openai for CLIP, webli for SigLIP)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for CLIP model (cpu / cuda / mps)")
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
                             "mAP-neutral by construction; degrades only "
                             "category accuracy.")
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

    args = parser.parse_args()
    asyncio.run(main(args))
