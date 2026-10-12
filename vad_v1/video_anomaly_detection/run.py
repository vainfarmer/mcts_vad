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

import numpy as np
from openai import AsyncOpenAI

from pipeline import MultiAnomalyVAD

UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


class DashScopeLLMClient:
    """
    DashScope VLM 客户端（实现 Pipeline 要求的 LLM 接口）

    通过 OpenAI 兼容接口调用阿里云 DashScope 的视觉语言模型。
    Pipeline 要求 LLM 客户端实现 async generate(prompt, images) → str 接口，
    本类将 numpy 帧数组编码为 base64 JPEG 后通过 OpenAI SDK 发送。

    支持的模型：qwen-vl-max（默认，最强）、qwen-vl-plus 等 DashScope VLM 系列。
    也可指向其他兼容 OpenAI API 的服务（如 vLLM、Ollama）通过 base_url 配置。

    temperature=0.0 确保可复现性——异常检测需要确定性判断，不需要创造性回答。
    """

    def __init__(self, api_key=None, base_url=None, model="qwen-vl-max",
                 temperature=0.0, max_tokens=512, max_frames=200):
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_frames = max_frames

    async def generate(self, prompt, images=None, as_video=False, video_fps=2.0):
        """
        调用 VLM 生成回复（Pipeline 的统一 LLM 接口）

        被 AnomalyDiscovery（Stage 2 异常判断）和 TemporalBoundaryMCTS
        （Stage 3 异常确认）调用。每次调用传入一段视频的关键帧和判断 prompt。

        图片编码流程：
          numpy RGB → PIL Image → JPEG (quality=80) → base64 → data URI
          使用 JPEG 而非 PNG 因为体积更小（通常 5-10x），加速网络传输。
          quality=80 在文件大小和视觉质量间取得平衡。

        Args:
            prompt: 文本提示（如 Stage 2 的异常判断 prompt 或 Stage 3 的确认 prompt）
            images: numpy 数组 [N, H, W, 3] (RGB uint8) 或 None

        Returns:
            str: LLM 生成的文本回复
        """
        import base64
        import io
        from PIL import Image

        content = []

        if images is not None and len(images) > 0:
            selected = self._sample_frames(images, self.max_frames)

            if as_video and len(selected) > 1:
                video_frames = []
                for img_array in selected:
                    img = Image.fromarray(img_array)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=80)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    video_frames.append(f"data:image/jpeg;base64,{img_b64}")
                content.append({
                    "type": "video",
                    "video": video_frames,
                    "fps": video_fps,
                })
            else:
                for img_array in selected:
                    img = Image.fromarray(img_array)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=80)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

        content.append({"type": "text", "text": prompt})

        import asyncio as _asyncio
        for _attempt in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as _e:
                if _attempt < 4 and ('429' in str(_e) or 'rate' in str(_e).lower() or 'limit' in str(_e).lower()):
                    wait = 2 ** _attempt + 1
                    print(f"    [LLM] Rate limited, retry in {wait}s (attempt {_attempt+1}/5)")
                    await _asyncio.sleep(wait)
                else:
                    raise

    def _sample_frames(self, frames, n):
        """
        从帧数组中均匀采样 n 帧

        np.linspace 保证首尾帧都被包含，中间帧等距分布。
        这是 Pipeline 中 SemanticFrameSampler 之后的最后一道保险，
        确保发送给 API 的帧数不超过限制。
        """
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return frames[indices]


async def main(args):
    llm_client = DashScopeLLMClient(
        api_key=args.api_key,
        base_url=args.api_base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
    )

    vad = MultiAnomalyVAD(
        llm_client=llm_client,
        fps=args.fps,
        segment_duration=args.segment_duration,
        mcts_simulations=args.mcts_simulations,
        mcts_max_depth=args.mcts_max_depth,
        anomaly_categories=UCF_CRIME_CATEGORIES,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        fps_sample=args.fps_sample,
        motion_threshold=args.motion_threshold,
        min_motion_interval=args.min_motion_interval,
    )

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
    print(f"Video duration:  {result['duration']:.1f}s")
    print(f"Is normal:       {result['is_normal']}")
    print(f"Anomalies found: {len(result['anomalies'])}")

    for i, anomaly in enumerate(result['anomalies']):
        print(f"\n  Anomaly {i+1}:")
        print(f"    Time:       {anomaly['t_start']:.1f}s - {anomaly['t_end']:.1f}s")
        print(f"    Confidence: {anomaly['confidence']:.3f}")
        print(f"    Type:       {anomaly['anomaly_type']}")

    if args.output_file:
        serializable = {
            'video_path': result['video_path'],
            'duration': result['duration'],
            'processing_time': round(elapsed, 2),
            'is_normal': result['is_normal'],
            'anomalies': result['anomalies'],
        }
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

    args = parser.parse_args()
    asyncio.run(main(args))
