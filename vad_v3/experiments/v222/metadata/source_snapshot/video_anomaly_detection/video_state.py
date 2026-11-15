"""配对尾流程的数据隔离：共享已完成的只读大数组，深拷贝可变容器。"""
import copy

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# 这些数据在Stage5之前构建，尾流程只读取。新增字段默认深拷贝，不能自动
# 将所有ndarray/Tensor都共享，以免未来的分支工作缓冲区意外串用。
_READ_ONLY_KEYS = frozenset({
    'all_frames', 'timestamps', 'clip_features', 'motion_energy',
    'mil_scores', 'tsm_novelty',
})


def fork_video_state(video_data):
    """返回独立分支字典；不复制RGB帧或CLIP特征，不修改原数组可写标志。

    缓存字典/新增缓存条目及其他未知字段保持独立。已生成的类别曲线是只读
    数值，也可共享。numpy使用只读view防止常规原地写入；PyTorch没有只读
    Tensor标志，clip_features按只读契约共享，调用方若需修改须先clone。
    """
    memo = {}
    for key, value in video_data.items():
        if key not in _READ_ONLY_KEYS and not (isinstance(key, str) and key.startswith('_cat_scores_')):
            continue
        if isinstance(value, np.ndarray) and not value.dtype.hasobject:
            view = value.view()
            view.flags.writeable = False
            memo[id(value)] = view
        elif key == 'clip_features' and torch is not None and isinstance(value, torch.Tensor):
            memo[id(value)] = value
    # memo也保持同一分支内部原有的别名关系；其余列表、dict、数组沿用旧深拷贝。
    return copy.deepcopy(video_data, memo)
