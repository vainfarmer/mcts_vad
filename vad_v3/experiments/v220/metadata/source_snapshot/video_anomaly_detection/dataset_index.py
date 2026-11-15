"""Read existing dataset indexes and resolve extracted video paths without rewriting parquet."""
from pathlib import Path


# 优先使用数据集已有data_parquet，兼容旧data目录；不生成或改写索引。
def parquet_directory(root):
    root = Path(root)
    for directory in (root / 'data_parquet', root / 'data', root):
        if list(directory.glob('test-*.parquet')):
            return directory
    raise FileNotFoundError(f'No test parquet index under {root}')


# 索引可能保留videos/...布局；本机按解压目录查找唯一同名视频，歧义时报错。
def resolve_video(root, indexed_path):
    root = Path(root)
    path = root / indexed_path
    if path.is_file():
        return str(path.resolve())
    # Normal测试视频在事件识别目录也可能有副本，按索引所属测试集优先定位。
    if Path(indexed_path).parent.name == 'Testing_Normal':
        testing_path = root / 'Testing_Normal_Videos_Anomaly' / Path(indexed_path).name
        if testing_path.is_file():
            return str(testing_path.resolve())
    name = Path(indexed_path).name
    matches = sorted(p for p in root.rglob(name) if p.is_file() and 'test' not in p.relative_to(root).parts)
    if len(matches) != 1:
        raise ValueError(f'Expected one original video for {indexed_path}, found {len(matches)}')
    return str(matches[0].resolve())
