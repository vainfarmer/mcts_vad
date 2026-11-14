"""Shared, lightweight configuration for both HTTP and worker entry points."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

WEB_DIR = Path(__file__).resolve().parent
V3_DIR = WEB_DIR.parent
ROOT = V3_DIR.parent
ENV_FILE = WEB_DIR / '.env'
DEFAULT_MODEL = 'qwen3-vl-plus'
DEFAULT_MCTS_MODE = 'refine'
DEFAULT_MCTS_ACCEPTANCE_POLICY = 'force_best'
DATASET_ROOT = Path(os.getenv('VAD_DATASET_DIR', 'F:/dataset/ucf_crime'))
CATEGORIES = ('Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
              'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
              'Stealing', 'Vandalism')
ACTIVE = ('queued', 'running', 'cancelling')


def load_environment(path=None):
    load_dotenv(path or ENV_FILE, override=False, encoding='utf-8-sig')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    # Includes uncommitted changes; a Git HEAD alone is not an execution identity.
    files = sorted((V3_DIR / 'video_anomaly_detection').glob('*.py'))
    return digest({p.name: file_hash(p) for p in files})


def runtime_identity():
    packages={}
    for name in ('torch','torchvision','transformers','open_clip_torch','numpy','opencv-python','safetensors','openai'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    assets={}
    try:
        from huggingface_hub import try_to_load_from_cache
        for repo,name in (
            ('OPear/videomae-large-finetuned-UCF-Crime','config.json'),
            ('OPear/videomae-large-finetuned-UCF-Crime','model.safetensors'),
            ('timm/vit_large_patch14_clip_224.openai','open_clip_model.safetensors')):
            cached=try_to_load_from_cache(repo,name)
            path=Path(cached) if isinstance(cached,str) else None
            assets[repo+'/'+name]=(dict(revision=path.parent.name,bytes=path.stat().st_size)
                if path and path.is_file() else None)
    except ImportError:
        pass
    return dict(python=sys.version.split()[0],packages=packages,model_assets=assets)


def redact(text):
    key = os.environ.get('DASHSCOPE_API_KEY')
    return text.replace(key, '[REDACTED]') if key else text


def safe_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('文件路径超出数据目录。')
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(dumps(value), encoding='utf-8')
    temp.replace(path)


def effective_config(options=None, timeout=3600, model=None):
    if options is None:
        options={}
    if not isinstance(options,dict):
        raise ValueError('配置必须是 JSON 对象。')
    allowed = {'device', 'mcts_mode', 'context', 'full_classification', 'mcts',
               'postprocess', 'timeout', 'model', 'profile_id', 'evidence_level'}
    if set(options) - allowed:
        raise ValueError('不支持的配置字段：' + ', '.join(sorted(set(options) - allowed)))
    # Load the pure dataclasses without importing the algorithm package / CUDA.
    algorithm_dir = str(V3_DIR / 'video_anomaly_detection')
    if algorithm_dir not in sys.path:
        sys.path.insert(0, algorithm_dir)
    from config import VADConfig
    cfg = VADConfig()
    cfg.device = options.get('device', 'cuda')
    if cfg.device not in ('cuda', 'cpu', 'mps'):
        raise ValueError('无效计算设备。')
    cfg.anomaly_categories = list(CATEGORIES)
    cfg.mcts.mode = options.get('mcts_mode', DEFAULT_MCTS_MODE)
    # Web 与算法新任务均默认 force_best；显式配置和实验分支仍按各自策略执行。
    cfg.mcts.acceptance_policy = DEFAULT_MCTS_ACCEPTANCE_POLICY
    cfg.mcts.simulations = 8  # Match run.py's existing CLI default.
    cfg.mcts.trace_enabled = True
    mcts = options.get('mcts', {})
    if not isinstance(mcts, dict):
        raise ValueError('mcts 必须为对象。')
    editable = {'mode', 'search_strategy', 'simulations', 'max_depth',
                'max_candidates_per_video', 'max_vlm_evaluations_per_candidate',
                'max_vlm_evaluations_per_video', 'timeout_seconds_per_candidate',
                'action_set', 'acceptance_policy', 'trigger_policy', 'random_seed'}
    if set(mcts) - editable:
        raise ValueError('存在不支持的 MCTS 参数。')
    for key, value in mcts.items():
        setattr(cfg.mcts, key, value)
    try:
        cfg.mcts.__post_init__()
    except (TypeError,AttributeError) as exc:
        raise ValueError('MCTS 参数类型无效。') from exc
    if (cfg.mcts.simulations > 200 or cfg.mcts.max_vlm_evaluations_per_video > 500
            or cfg.mcts.max_candidates_per_video > 50):
        raise ValueError('搜索预算超过单机系统上限。')
    post = options.get('postprocess', {})
    mapping = {'qscore_enabled': 'qscore_rerank', 'bue_enabled': 'bue_enabled',
               'cbt_enabled': 'cbt_enabled'}
    if not isinstance(post, dict) or set(post) - (set(mapping) | {'final_nms_enabled'}):
        raise ValueError('无效后处理配置。')
    for key, value in post.items():
        if type(value) is not bool:
            raise ValueError('后处理开关必须是布尔值。')
        if key == 'final_nms_enabled':
            cfg.mcts.final_nms_enabled = value
        else:
            setattr(cfg.scoring, mapping[key], value)
    full = options.get('full_classification', False)
    if type(full) is not bool:
        raise ValueError('类别复核开关必须是布尔值。')
    cfg.scoring.classify_skip_class_agnostic = not full
    # CLI default is True, while some historical dataclass versions differed.
    cfg.scoring.cbt_recal = True
    context = options.get('context', 'surveillance camera')
    if not isinstance(context, str) or len(context) > 1000:
        raise ValueError('场景描述最多 1000 字符。')
    limit = options.get('timeout', timeout)
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 1 <= limit <= 86400:
        raise ValueError('任务超时应在 1—86400 秒之间。')
    selected_model = options.get('model') or model or os.getenv('DASHSCOPE_MODEL') or DEFAULT_MODEL
    if not isinstance(selected_model, str) or len(selected_model) > 120 or not selected_model.strip():
        raise ValueError('模型名称无效。')
    return dict(device=cfg.device, mcts_mode=cfg.mcts.mode, context=context,
                full_classification=full, model=selected_model.strip(), timeout=limit,
                client=dict(temperature=0.,max_tokens=512,max_frames=200,
                    api_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                    api_retry_attempts=3,api_retry_min_seconds=2.,api_retry_max_seconds=8.,
                    api_concurrency=4,api_timeout_seconds=60.,api_connect_timeout_seconds=15.),
                environment=runtime_identity(),
                detector_config=cfg.to_dict(), schema_version=1,
                response_policy='strict_discovery_v1', description_policy='original',
                evidence_level='research')
