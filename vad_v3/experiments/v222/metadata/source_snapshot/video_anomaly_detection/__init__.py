"""Load GPU modules only when their public classes are requested.

Configuration, response validation and offline evaluation remain usable without
initializing CUDA. Requesting the detector still propagates dependency errors.
"""
from importlib import import_module

_EXPORTS = {
    'MultiAnomalyVAD': 'pipeline',
    'VideoPreprocessor': 'video_preprocessor',
    'AnomalyDiscovery': 'anomaly_discovery',
    'TemporalMCTSNode': 'temporal_mcts',
    'TemporalBoundaryMCTS': 'temporal_mcts',
    'VideoAnomalyExpert': 'anomaly_expert',
    'SemanticFrameSampler': 'token_compressor',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
