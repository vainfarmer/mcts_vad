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

    args = parser.parse_args()
    asyncio.run(main(args))
