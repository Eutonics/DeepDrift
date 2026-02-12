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

        if layer_names is None:
            layer_names = find_target_layers(model)

        self.layer_names = layer_names
        self.activations = {}
        self.channel_indices = {}
        self.hooks = []

        self._register_all_hooks()

        self.prev_state = None
        self.threshold = None
        self.calibration_stats = {}

    def _register_all_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                # Apply pooling
                pooled = self.pooling_fn(output)

                # Apply sparse sampling if requested
                if self.n_channels is not None:
                    if name not in self.channel_indices:
                        # Initialize random indices once
                        total_ch = pooled.shape[-1]
                        n = min(total_ch, self.n_channels)
                        self.channel_indices[name] = torch.randperm(total_ch)[:n].to(pooled.device)

                    pooled = pooled[..., self.channel_indices[name]]

                self.activations[name] = pooled.detach()
            return hook

        self.hooks = register_hooks(self.model, self.layer_names, make_hook)

    def get_spatial_velocity(self) -> List[float]:
        """
        Returns L2-norms between successive layers for the current batch.
        """
        if not self.activations:
            return []

        velocities = []
        # Ensure we follow the order of layers
        acts = [self.activations[name] for name in self.layer_names if name in self.activations]

        for i in range(len(acts) - 1):
            # L2 norm of difference (normalized by dimension if needed, but paper uses raw L2)
            # Actually, standard velocity is norm(x_l - x_{l-1})
            # We assume shapes are compatible or pooling made them so
            diff = acts[i+1] - acts[i]
            # Mean over batch, then norm
            vel = torch.norm(diff, p=2, dim=-1).mean().item()
            velocities.append(vel)

        return velocities

    def get_temporal_velocity(self, step: Optional[int] = None) -> float:
        """
        Returns L2-norm between current and previous state of the first monitored layer.
        """
        if not self.activations or not self.layer_names:
            return 0.0

        current_layer_name = self.layer_names[0]
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
        """
        target_device = device or self.device
        self.model.to(target_device)
        self.model.eval()

        all_velocities = []

        with torch.no_grad():
            for batch in dataloader:
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
        self.threshold = compute_iqr_threshold(all_velocities)

        self.calibration_stats = {
            "mean": float(np.mean(all_velocities)),
            "std": float(np.std(all_velocities)),
            "q75": float(np.percentile(all_velocities, 75)),
            "threshold": self.threshold
        }

        return self.calibration_stats

    def detect_anomaly(self, x: torch.Tensor) -> bool:
        """
        Performs forward pass and checks for anomaly.
        """
        self.model.eval()
        with torch.no_grad():
            self.model(x)

        velocities = self.get_spatial_velocity()
        if not velocities:
            return False

        peak_v = max(velocities)

        if self.threshold is None:
            return False

        return peak_v > self.threshold

    def clear(self):
        self.activations = {}
        self.prev_state = None

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
