import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Callable, Dict, Any
from .utils.pooling import get_pooling_fn, sparse_sample
from .utils.hooks import register_hooks, find_target_layers
from .utils.stats import compute_iqr_threshold

class DeepDriftMonitor:
    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        pooling: Union[str, Callable] = "cls",
        n_channels: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu"
    ):
        """
        Unified monitor for Semantic Velocity.
        
        Args:
            model: PyTorch model to monitor
            layer_names: List of layer names to attach hooks to. If None, auto-detects.
            layer_indices: Alternative to layer_names (not implemented yet)
            pooling: Pooling strategy: 'cls', 'mean', 'flatten', or callable
            n_channels: Number of channels for sparse sampling (None = use all)
            seed: Random seed for reproducibility
            device: Device to run calibration on
        """
        self.model = model
        self.device = device
        self.pooling_mode = pooling
        self.pooling_fn = get_pooling_fn(pooling)
        self.n_channels = n_channels
        self.seed = seed

        # Set seed for reproducible sparse sampling
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Auto-detect layers if none provided
        if layer_names is None:
            layer_names = find_target_layers(model)
        self.layer_names = layer_names if layer_names else []

        self.activations = {}
        self.channel_indices = {}
        self.hooks = []
        self.prev_state = None
        self.threshold = None
        self.calibration_stats = {}

        self._register_all_hooks()

    def _register_all_hooks(self):
        """Register forward hooks on all target layers."""
        def make_hook(name):
            def hook(module, input, output):
                # Apply pooling
                try:
                    pooled = self.pooling_fn(output)
                except Exception as e:
                    raise RuntimeError(f"Pooling failed for layer {name}: {e}")

                # Apply sparse sampling if requested
                if self.n_channels is not None:
                    if name not in self.channel_indices:
                        # Initialize random indices once
                        total_ch = pooled.shape[-1]
                        n = min(total_ch, self.n_channels)
                        self.channel_indices[name] = torch.randperm(total_ch, device=pooled.device)[:n]

                    pooled = pooled[..., self.channel_indices[name]]

                self.activations[name] = pooled.detach()
            return hook

        if self.layer_names:
            self.hooks = register_hooks(self.model, self.layer_names, make_hook)

def get_spatial_velocity(self) -> List[float]:
    """
    Returns L2-norms between successive layers for the current batch.
    Handles layers with different dimensions by computing difference in a common space
    or falling back to norm difference when shapes mismatch.
    """
    if len(self.activations) < 2:
        return []

    available_names = sorted(self.activations.keys())
    acts = [self.activations[name] for name in available_names]

    velocities = []
    for i in range(len(acts) - 1):
        a, b = acts[i], acts[i + 1]
        
        # Если размерности совпадают — считаем разность
        if a.shape[-1] == b.shape[-1]:
            diff = b - a
            vel = torch.norm(diff, p=2, dim=-1).mean().item()
        else:
            # Разные размерности: считаем разницу норм (fallback)
            # ||b|| - ||a|| (по модулю)
            norm_a = torch.norm(a, p=2, dim=-1)
            norm_b = torch.norm(b, p=2, dim=-1)
            vel = torch.abs(norm_b - norm_a).mean().item()
        
        velocities.append(vel)

    return velocities

    def get_temporal_velocity(self, step: Optional[int] = None) -> float:
        """
        Returns L2-norm between current and previous state of the first monitored layer.
        
        Args:
            step: Optional step number; if 0 or None resets state on first call
            
        Returns:
            Velocity between current and previous state, or 0.0 if not enough states.
        """
        if not self.activations:
            return 0.0

        # Use first available activation key, not self.layer_names[0]
        available_names = sorted(self.activations.keys())
        if not available_names:
            return 0.0

        current_layer_name = available_names[0]
        current_state = self.activations[current_layer_name]

        if self.prev_state is None or step == 0:
            self.prev_state = current_state
            return 0.0

        diff = current_state - self.prev_state
        velocity = torch.norm(diff, p=2, dim=-1).mean().item()
        self.prev_state = current_state
        return velocity

    def calibrate(
        self,
        dataloader: Any,
        method: str = "iqr",
        device: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calibrates the threshold on normal data.
        
        Args:
            dataloader: DataLoader with in-distribution samples
            method: Calibration method ('iqr' only currently)
            device: Device to run on (overrides init device)
            
        Returns:
            Dictionary with calibration statistics
        """
        target_device = device or self.device
        self.model.to(target_device)
        self.model.eval()

        all_velocities = []

        with torch.no_grad():
            for batch in dataloader:
                # Handle both tuple/list and direct tensor
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(target_device)
                else:
                    x = batch.to(target_device)

                self.model(x)
                spatial_v = self.get_spatial_velocity()
                if spatial_v:
                    all_velocities.append(max(spatial_v))

        if not all_velocities:
            return {}

        all_velocities = np.array(all_velocities)
        
        if method == "iqr":
            self.threshold = compute_iqr_threshold(all_velocities)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibration_stats = {
            "mean": float(np.mean(all_velocities)),
            "std": float(np.std(all_velocities)),
            "q25": float(np.percentile(all_velocities, 25)),
            "q75": float(np.percentile(all_velocities, 75)),
            "iqr": float(np.percentile(all_velocities, 75) - np.percentile(all_velocities, 25)),
            "threshold": self.threshold,
            "n_samples": len(all_velocities)
        }

        return self.calibration_stats

    def detect_anomaly(
        self, 
        x: torch.Tensor, 
        use_two_sided: bool = False,
        lower_factor: float = 1.5,
        upper_factor: float = 1.5
    ) -> Union[bool, Dict[str, Any]]:
        """
        Performs forward pass and checks for anomaly.
        
        Args:
            x: Input tensor
            use_two_sided: If True, check both low and high velocity anomalies
            lower_factor: IQR multiplier for lower threshold (default: 1.5)
            upper_factor: IQR multiplier for upper threshold (default: 1.5)
            
        Returns:
            If use_two_sided is False: bool indicating anomaly
            If use_two_sided is True: dict with 'is_anomaly', 'direction', 'peak_velocity'
        """
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before detect_anomaly()")

        self.model.eval()
        self.clear()
        
        with torch.no_grad():
            self.model(x)

        velocities = self.get_spatial_velocity()
        if not velocities:
            return False if not use_two_sided else {
                'is_anomaly': False, 
                'direction': 'none',
                'peak_velocity': 0.0,
                'lower_threshold': None,
                'upper_threshold': None
            }

        peak_v = max(velocities)

        if not use_two_sided:
            return peak_v > self.threshold

        # Two-sided detection
        lower_threshold = self.calibration_stats['q25'] - lower_factor * self.calibration_stats['iqr']
        upper_threshold = self.calibration_stats['q75'] + upper_factor * self.calibration_stats['iqr']

        is_low = peak_v < lower_threshold
        is_high = peak_v > upper_threshold

        direction = 'low' if is_low else 'high' if is_high else 'normal'

        return {
            'is_anomaly': is_low or is_high,
            'direction': direction,
            'peak_velocity': peak_v,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold
        }

    def clear(self):
        """Reset stored activations and temporal state."""
        self.activations = {}
        self.prev_state = None

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_layer_velocities(self) -> Dict[str, float]:
        """
        Returns velocity per layer pair for interpretability.
        
        Returns:
            Dictionary mapping layer pairs to velocities
        """
        if len(self.activations) < 2:
            return {}

        available_names = sorted(self.activations.keys())
        velocities = {}
        
        for i in range(len(available_names) - 1):
            name1 = available_names[i]
            name2 = available_names[i + 1]
            diff = self.activations[name2] - self.activations[name1]
            vel = torch.norm(diff, p=2, dim=-1).mean().item()
            velocities[f"{name1}→{name2}"] = vel

        return velocities

    def reset_temporal(self):
        """Reset temporal velocity state."""
        self.prev_state = None
