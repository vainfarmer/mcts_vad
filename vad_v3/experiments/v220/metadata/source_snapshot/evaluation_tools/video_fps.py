"""读取原视频时间轴；清单接口仅供旧版本和显式历史复现使用。"""
from collections.abc import Mapping
import json
import math
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'data' / 'ucf_crime_test_video_fps.json'


def probe_video_metadata(path):
    """直接打开视频读取源 FPS/帧数，不取整 FPS，也不读取预先生成的清单。"""
    import cv2

    path = Path(path).resolve(strict=True)
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, _ = cap.read()
        if (not cap.isOpened() or not readable or not math.isfinite(fps) or fps <= 0
                or not math.isfinite(count) or count < 1):
            raise ValueError(f'无法读取原视频的有效 FPS/帧数：{path}')
        return dict(video_path=str(path), fps=fps, frame_count=int(count),
                    duration=int(count) / fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    timebase='opencv_source_fps', vfr_verified=False)
    finally:
        cap.release()


def source_video_metadata(video_paths, *, video_root=None, relative_to=None):
    """按结果中的原视频路径探测；搬迁数据可指定根目录，重名文件拒绝猜测。

    返回值是本次探测产生的内存字典，并非从 FPS map 文件读取。
    一批只建立一次文件名索引，避免每个视频都递归遍历数据集。
    """
    root = Path(video_root).resolve() if video_root is not None else None
    base = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
    index = None
    result = {}
    for video_id, saved_path in video_paths.items():
        if not saved_path:
            raise ValueError(f'{video_id}: 结果缺少原视频路径')
        original = Path(str(saved_path).replace('\\', '/'))
        candidates = [original] if original.is_absolute() else [base / original, Path.cwd() / original]
        if root is not None and not original.is_absolute():
            candidates.append(root / original)
        existing = {p.resolve() for p in candidates if p.is_file()}
        if len(existing) > 1:
            raise ValueError(f'{video_id}: 相对路径对应多个原视频，请提供绝对路径：{existing}')
        path = next(iter(existing), None)
        if path is None and root is not None:
            if not root.is_dir():
                raise FileNotFoundError(f'原视频目录不存在：{root}')
            if index is None:
                index = {}
                for item in root.rglob('*'):
                    if item.is_file() and item.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm'}:
                        index.setdefault(item.name.casefold(), []).append(item)
            matches = index.get(original.name.casefold(), [])
            if len(matches) > 1:
                raise ValueError(f'{video_id}: 找到多个同名原视频，请提供明确路径：{matches}')
            if matches:
                path = matches[0]
        if path is None:
            raise FileNotFoundError(f'{video_id}: 原视频不可用：{saved_path}；可用 --video_root 指定搬迁后的目录')
        result[video_id] = probe_video_metadata(path)
    return result


def load_fps_manifest(path=DEFAULT_MANIFEST, required_ids=()):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('schema') != 'video-fps-v1':
        raise ValueError('Unsupported video FPS manifest schema')
    result = {}
    for name, item in obj['videos'].items():
        value = float(item['fps'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid FPS for {name}: {value}')
        result[name] = value
    missing = set(required_ids) - result.keys()
    if missing:
        raise ValueError(f'FPS manifest missing videos: {sorted(missing)}')
    return result


def fps_for_video(fps, video_id):
    value = float(fps[video_id] if isinstance(fps, Mapping) else fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid FPS for {video_id}: {value}')
    return value


def evaluation_fps(fallback, manifest, required_ids):
    """None自动使用项目索引；False仅供显式复现旧固定FPS指标。"""
    if manifest is False:
        return fallback
    target = Path(manifest) if manifest is not None else DEFAULT_MANIFEST
    # 默认索引丢失也不回退固定30fps；历史复现必须显式选择False。
    return load_fps_manifest(target, required_ids)
