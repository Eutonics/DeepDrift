import torch
import torch.nn as nn
from typing import Optional, Callable, Union

def cls_pool(x: torch.Tensor) -> torch.Tensor:
    """Extract CLS token from ViT-like output [B, Seq, Dim] -> [B, Dim]"""
    if x.dim() != 3:
        # Fallback if not 3D
        return x.mean(dim=1) if x.dim() > 1 else x
    return x[:, 0, :]

def mean_pool(x: torch.Tensor) -> torch.Tensor:
    """Global Average Pooling or Mean pooling across sequence/spatial dims"""
    if x.dim() == 4: # CNN [B, C, H, W]
        return x.mean(dim=[2, 3])
    elif x.dim() == 3: # Seq [B, Seq, Dim]
        return x.mean(dim=1)
    return x

def flatten_pool(x: torch.Tensor) -> torch.Tensor:
    """Flatten all but batch dimension"""
    return x.flatten(1)

def sparse_sample(x: torch.Tensor, n_channels: int, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Select N random channels from the tensor"""
    total_channels = x.shape[1]
    if n_channels >= total_channels:
        return x

    if indices is None:
        indices = torch.randperm(total_channels)[:n_channels]

    return x[:, indices]

def get_pooling_fn(pooling: Union[str, Callable]) -> Callable:
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
