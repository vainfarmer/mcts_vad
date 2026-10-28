"""集中日志配置 — 供所有 VAD 模块使用."""
import logging
import sys
from pathlib import Path

_configured = False
# Format: timestamp [LEVEL] message  — logger name (e.g. vad.batch) omitted for readability.
LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """初始化 vad 根日志器，同时输出到 stdout 和指定文件（若提供）."""
    global _configured
    root = logging.getLogger("vad")
    if _configured:
        return root
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.propagate = False
    _configured = True
    return root


def get(name: str = "vad") -> logging.Logger:
    """获取子日志器，如 get('vad.pipeline')。

    若 setup() 尚未调用（模块独立运行 / 单测），自动为 vad 根附加一个
    stdout handler，保证日志记录不会被静默丢弃；setup() 后会被其 clear() 替换。
    """
    logger = logging.getLogger(name)
    root = logging.getLogger("vad")
    if not _configured and not root.handlers:
        fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        root.setLevel(logging.INFO)
    return logger
