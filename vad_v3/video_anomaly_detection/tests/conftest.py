"""pytest 配置 — 把包目录加入 sys.path 并初始化日志。"""
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import vad_logger  # noqa: E402

# 测试时输出 INFO 级日志到 stdout（模块独立运行场景，setup 兜底已处理）
vad_logger.setup()
