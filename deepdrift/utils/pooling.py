"""
Pooling utilities for extracting features from hidden states.
Supports CLS token, mean pooling, flatten, and sparse sampling.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Union

def cls_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Extract CLS token from ViT-like output [B, Seq, Dim] -> [B, Dim].
    For 2D input [B, Dim] — return as is.
    For other shapes — fallback to mean pooling.
    """
    if x.dim() == 3:
        return x[:, 0, :]
    elif x.dim() == 2:
        # Already [B, Dim] — just return
        return x
    else:
        # Fallback: mean over remaining dimensions
        return x.mean(dim=tuple(range(1, x.dim())))

def mean_pool(x: torch.Tensor) -> torch.Tensor:
    """Global Average Pooling or Mean pooling across sequence/spatial dimensions"""
    if x.dim() == 4:      # CNN: [B, C, H, W]
        return x.mean(dim=[2, 3])
    elif x.dim() == 3:    # Sequence: [B, Seq, Dim]
        return x.mean(dim=1)
    elif x.dim() == 2:    # [B, Dim]
        return x
    else:                 # Other
        return x.flatten(1)

def flatten_pool(x: torch.Tensor) -> torch.Tensor:
    """Flatten all but batch dimension"""
    return x.flatten(1)

def sparse_sample(x: torch.Tensor, n_channels: int, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Select N random channels from the last dimension.
    
    Args:
        x: Input tensor of shape [B, ..., D]
        n_channels: Number of channels to keep
        indices: Pre-computed channel indices (for reproducibility)
    
    Returns:
        Tensor with shape [B, ..., n_channels]
    """
    total_channels = x.shape[-1]
    if n_channels >= total_channels:
        return x

    if indices is None:
        indices = torch.randperm(total_channels, device=x.device)[:n_channels]

    # Handle different dimensionalities
    if x.dim() == 2:
        return x[:, indices]
    elif x.dim() == 3:
        return x[:, :, indices]
    else:
        return x[..., indices]

def get_pooling_fn(pooling: Union[str, Callable]) -> Callable:
    """
    Resolve pooling function by name or return callable.
    
    Args:
        pooling: 'cls', 'mean', 'flatten', or a custom callable
    
    Returns:
        Pooling function
    """
    if callable(pooling):
        return pooling

    mapping = {
        "cls": cls_pool,
        "mean": mean_pool,
        "flatten": flatten_pool
    }

    if pooling not in mapping:
        raise ValueError(f"Unknown pooling method: {pooling}. Use 'cls', 'mean', 'flatten' or a callable.")

    return mapping[pooling]
