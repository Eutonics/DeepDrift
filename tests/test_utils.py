"""
Unit tests for DeepDrift utility functions.
"""

import pytest
import torch
import numpy as np
from deepdrift.utils.pooling import (
    cls_pool,
    mean_pool,
    flatten_pool,
    sparse_sample,
    get_pooling_fn
)
from deepdrift.utils.stats import compute_iqr_threshold, bootstrap_ci
from deepdrift.utils.hooks import find_target_layers, register_hooks


# ========== POOLING TESTS ==========

def test_cls_pool():
    """Test CLS token extraction."""
    x = torch.randn(2, 10, 768)
    pooled = cls_pool(x)
    assert pooled.shape == (2, 768)
    
    # Fallback for 2D input
    x2 = torch.randn(2, 768)
    pooled2 = cls_pool(x2)
    assert pooled2.shape == (2, 768)


def test_mean_pool():
    """Test mean pooling for CNN and sequence formats."""
    # CNN format [B, C, H, W]
    x_cnn = torch.randn(2, 64, 16, 16)
    pooled_cnn = mean_pool(x_cnn)
    assert pooled_cnn.shape == (2, 64)
    
    # Sequence format [B, Seq, Dim]
    x_seq = torch.randn(2, 10, 768)
    pooled_seq = mean_pool(x_seq)
    assert pooled_seq.shape == (2, 768)
    
    # 2D input [B, Dim]
    x_2d = torch.randn(2, 768)
    pooled_2d = mean_pool(x_2d)
    assert pooled_2d.shape == (2, 768)


def test_flatten_pool():
    """Test flatten pooling."""
    x = torch.randn(2, 64, 4, 4)
    pooled = flatten_pool(x)
    assert pooled.shape == (2, 64 * 4 * 4)


def test_sparse_sample():
    """Test sparse channel sampling."""
    x = torch.randn(2, 100)
    sampled = sparse_sample(x, n_channels=10)
    assert sampled.shape == (2, 10)
    
    # With pre-computed indices
    indices = torch.randperm(100)[:20]
    sampled2 = sparse_sample(x, n_channels=20, indices=indices)
    assert sampled2.shape == (2, 20)
    
    # 3D case
    x3 = torch.randn(2, 10, 100)
    sampled3 = sparse_sample(x3, n_channels=30)
    assert sampled3.shape == (2, 10, 30)


def test_get_pooling_fn():
    """Test pooling function resolution."""
    assert callable(get_pooling_fn('cls'))
    assert callable(get_pooling_fn('mean'))
    assert callable(get_pooling_fn('flatten'))
    
    # Custom callable
    custom_fn = lambda x: x.mean(dim=1)
    assert get_pooling_fn(custom_fn) == custom_fn
    
    with pytest.raises(ValueError):
        get_pooling_fn('invalid')


# ========== STATS TESTS ==========

def test_compute_iqr_threshold():
    """Test IQR threshold computation."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    threshold = compute_iqr_threshold(data, factor=1.5)
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    expected = q75 + 1.5 * (q75 - q25)
    assert threshold == expected


def test_bootstrap_ci():
    """Test bootstrap confidence interval computation."""
    data = np.random.randn(100)
    ci_low, ci_high = bootstrap_ci(data, n_resamples=100)
    assert ci_low < ci_high
    assert isinstance(ci_low, float)
    assert isinstance(ci_high, float)


# ========== HOOKS TESTS (lightweight) ==========

def test_find_target_layers_simple_model():
    """Test layer detection on a simple MLP."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    layers = find_target_layers(model)
    assert isinstance(layers, list)
    # Should find at least the Linear layers
    assert any('0' in name or '2' in name for name in layers)


def test_register_hooks():
    """Test hook registration (basic smoke test)."""
    model = torch.nn.Linear(10, 10)
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = register_hooks(model, ['weight'], hook_fn)
    assert len(hooks) == 0  # 'weight' is a parameter, not a module
    
    # Clean up
    for h in hooks:
        h.remove()
