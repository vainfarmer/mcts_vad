"""
DashScope VLM 客户端 — 从 run.py 抽离，实现 llm_protocol.LLMClient。

通过 OpenAI 兼容接口调用阿里云 DashScope 视觉语言模型（qwen-vl-plus / qwen3-vl-plus）。
将 numpy 帧编码为 base64 JPEG 后发送。temperature=0 保证可复现。

安全：API key 仅从环境变量 DASHSCOPE_API_KEY 读取，永不记录到日志。
"""

from __future__ import annotations
try:
    from .llm_usage import tracked_completion
except ImportError:
    from llm_usage import tracked_completion

import os
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

try:
    import vad_logger as _vl
    _log = _vl.get("vad.llm")
except Exception:
    import logging
    _log = logging.getLogger("vad.llm")


class DashScopeLLMClient:
    """DashScope VLM 客户端（实现 LLMClient 协议）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen-vl-plus",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_frames: int = 200,
    ):
        # API key 仅从环境读取，禁止打印
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_frames = max_frames
        # v124 [COST]: drift-free per-client VLM call counter (FM10). Pure
        # instrumentation — never read inside generate(), so it cannot alter
        # any request or detection. Reset/read by the pipeline per video.
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        images: Optional[np.ndarray] = None,
        as_video: bool = False,
        video_fps: float = 2.0,
        temperature_override: Optional[float] = None,
    ) -> str:
        """调用 VLM 生成回复（pipeline 统一 LLM 接口）。

        图片编码：numpy RGB → PIL → JPEG (quality=80) → base64 → data URI。
        用 JPEG 而非 PNG，体积小 5-10x，加速网络传输。
        """
        # v124 [COST]: count every VLM API call (FM10 drift-free latency proxy).
        self.call_count += 1

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
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })

        content.append({"type": "text", "text": prompt})

        import asyncio as _asyncio
        for _attempt in range(5):
            try:
                response = await tracked_completion(self.client,
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.max_tokens,
                    temperature=temperature_override if temperature_override is not None else self.temperature,
                )
                return response.choices[0].message.content
            except Exception as _e:
                # 仅对限流类错误重试，其它直接抛出
                if _attempt < 4 and (
                    "429" in str(_e)
                    or "rate" in str(_e).lower()
                    or "limit" in str(_e).lower()
                ):
                    wait = 2 ** _attempt + 1
                    _log.warning(f"    [LLM] Rate limited, retry in {wait}s "
                                 f"(attempt {_attempt+1}/5)")
                    await _asyncio.sleep(wait)
                else:
                    raise

    def _sample_frames(self, frames: np.ndarray, n: int) -> np.ndarray:
        """均匀采样 n 帧（首尾必含），作为发送 API 前的帧数保险。"""
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return frames[indices]


__all__ = ["DashScopeLLMClient"]
