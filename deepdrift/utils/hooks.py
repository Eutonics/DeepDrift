import torch.nn as nn
from typing import List, Optional

def find_target_layers(model: nn.Module, architecture: Optional[str] = None) -> List[str]:
    """
    Heuristic to find suitable layers for monitoring.
    - For ViT: returns names of encoder layers.
    - For CNN: returns names of last convolutional blocks.
    - Default: returns names of blocks that look like feature extractors.
    """
    layer_names = []

    # ViT Heuristic
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return [f'encoder.layers.{i}' for i in range(len(model.encoder.layers))]

    # General CNN/MLP Heuristic
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.TransformerEncoderLayer)):
            candidates.append(name)

    if not candidates:
        return []

    # Take a few layers from the second half of the network (semantic region)
    if len(candidates) > 5:
        start = int(len(candidates) * 0.6)
        end = len(candidates)
        # Select up to 4 layers
        idx = [int(i) for i in iter(range(start, end, max(1, (end - start) // 4)))]
        layer_names = [candidates[i] for i in idx[:4]]
    else:
        layer_names = candidates

    return layer_names

def register_hooks(model: nn.Module, layer_names: List[str], hook_fn):
    hooks = []
    modules = dict(model.named_modules())
    for name in layer_names:
        if name in modules:
            hooks.append(modules[name].register_forward_hook(hook_fn(name)))
    return hooks
