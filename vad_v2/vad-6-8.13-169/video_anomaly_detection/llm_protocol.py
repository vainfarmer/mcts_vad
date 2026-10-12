"""
LLM 客户端抽象接口 (Protocol)

解决的核心问题：pipeline 直接依赖 DashScopeLLMClient 的具体实现（鸭子类型），
无法轻易替换为本地模型（Ollama/vLLM）、其他 API、或 mock 做单测。

定义 LLMClient Protocol 后：
  - pipeline 只依赖协议，与具体 VLM 实现解耦
  - 测试可用 MockLLMClient 返回固定 JSON，无需真实 API 调用
  - 换模型/换 provider 只需新增一个实现该协议的类
"""

from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class LLMClient(Protocol):
    """VLM 客户端协议 — pipeline 要求的最小接口。

    实现方只需提供 async generate(...)。被 AnomalyDiscovery（Stage 2 异常判断）、
    TemporalBoundaryMCTS（Stage 5 边界确认）、ensemble 分类（Stage 7）调用。

    Args:
        prompt: 文本提示
        images: 帧数组 [N,H,W,3] RGB uint8 或 None
        as_video: 是否以「视频」形式（多帧 + fps）发送，而非独立图片
        video_fps: as_video=True 时的视频帧率

    Returns:
        str: 模型生成的文本回复（通常为 JSON）
    """

    async def generate(
        self,
        prompt: str,
        images: Optional[np.ndarray] = None,
        as_video: bool = False,
        video_fps: float = 2.0,
        temperature_override: Optional[float] = None,
    ) -> str:
        ...


__all__ = ["LLMClient"]
