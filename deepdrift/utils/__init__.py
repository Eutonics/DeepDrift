"""
Utility modules for DeepDrift: pooling, hooks, statistics.
"""

from .pooling import (
    cls_pool,
    mean_pool,
    flatten_pool,
    sparse_sample,
    get_pooling_fn
)

from .hooks import (
    find_target_layers,
    register_hooks
)

from .stats import (
    compute_iqr_threshold,
    bootstrap_ci
)

__all__ = [
    'cls_pool',
    'mean_pool',
    'flatten_pool',
    'sparse_sample',
    'get_pooling_fn',
    'find_target_layers',
    'register_hooks',
    'compute_iqr_threshold',
    'bootstrap_ci',
]
