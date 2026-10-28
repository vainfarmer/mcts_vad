"""
VAD pipeline (build_vad)

VADConfig 作为单一配置源传入 MultiAnomalyVAD，避免构造参数散落、
默认值不一致。pipeline 内部所有超参数（预处理/CLIP/MCTS/评分/发现）
均从 config 读取，换实验只改 config 不改代码。

用法:
    from config import VADConfig
    from llm_client import DashScopeLLMClient
    from factory import build_vad

    cfg = VADConfig.from_args(args)
    vad = build_vad(cfg, DashScopeLLMClient(model=...))
    result = await vad.detect(video_path)
"""

from __future__ import annotations
from typing import Optional

from pipeline import MultiAnomalyVAD
from config import VADConfig
from llm_protocol import LLMClient


def build_vad(
    config: VADConfig,
    detect_client: LLMClient,
    classify_client: Optional[LLMClient] = None,
    image_processor=None,
) -> MultiAnomalyVAD:
    """从 VADConfig + LLM 客户端构造 pipeline。

    Args:
        config: 完整 VAD 配置
        detect_client: 检测/MCTS 用的（轻量）VLM 客户端
        classify_client: 分类用的（强）VLM 客户端，None 时复用 detect_client
        image_processor: 可选图像预处理器
    """
    return MultiAnomalyVAD(
        llm_client=detect_client,
        config=config,
        classify_llm_client=classify_client or detect_client,
        image_processor=image_processor,
    )


__all__ = ["build_vad"]
