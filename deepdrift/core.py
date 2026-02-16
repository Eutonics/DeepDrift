import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Callable, Dict, Any, Tuple
from .utils.pooling import get_pooling_fn
from .utils.hooks import register_hooks, find_target_layers
from .utils.stats import compute_iqr_threshold

class DeepDriftMonitor:
    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        pooling: Union[str, Callable] = "cls",
        relative_norm: bool = False,
        epsilon: float = 1e-6,
        n_channels: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
        debug: bool = False
    ):
        """
        Unified monitor for Semantic Velocity.
        
        Args:
            model: PyTorch model to monitor
            layer_names: List of layer names to attach hooks to. If None, auto-detects.
            layer_indices: Not implemented (reserved for future use).
            pooling: Pooling strategy: 'auto', 'cls', 'mean', 'flatten', 'last_token', or callable.
            relative_norm: If True, normalize velocity by previous-state magnitude.
            epsilon: Numerical stability constant for relative normalization.
            n_channels: Number of channels for sparse sampling (None = use all).
            seed: Random seed for reproducibility.
            device: Device to run calibration on.
            debug: Print debug information.
        """
        self.model = model
        self.device = device
        self.pooling_mode = pooling
        self.pooling_fn = get_pooling_fn(pooling)
        self.relative_norm = relative_norm
        self.epsilon = epsilon
        self.n_channels = n_channels
        self.seed = seed
        self.debug = debug

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
        if self.debug:
            print(f"[DeepDrift] Initialized with layers: {self.layer_names}")

    @staticmethod
    def _normalize_output(output: Any) -> torch.Tensor:
        """Extract activation tensor from common model output structures."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first
        if isinstance(output, dict):
            for key in ("last_hidden_state", "hidden_states", "logits"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                    return value[0]
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
        if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
            return output.last_hidden_state
        if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
            return output.logits

        raise TypeError(f"Unsupported hook output type: {type(output)}")

    def _ordered_activation_names(self) -> List[str]:
        """Return activation names aligned to configured layer order."""
        return [name for name in self.layer_names if name in self.activations]

    def _pair_velocity(self, prev: torch.Tensor, cur: torch.Tensor) -> Tuple[float, str]:
        """Compute robust pairwise velocity between two pooled activations."""
        if prev.device != cur.device:
            cur = cur.to(prev.device)

        if prev.shape == cur.shape:
            velocity_vector = torch.norm(cur - prev, p=2, dim=-1)
            method = "direct"
        elif prev.shape[:-1] == cur.shape[:-1]:
            prev_norm = torch.norm(prev, p=2, dim=-1)
            cur_norm = torch.norm(cur, p=2, dim=-1)
            velocity_vector = torch.abs(cur_norm - prev_norm)
            method = "norm_diff"
        else:
            raise ValueError(
                f"Incompatible activation shapes for velocity: {prev.shape} vs {cur.shape}. "
                "Use compatible layers or choose a pooling strategy that aligns hidden dims."
            )

        if self.relative_norm:
            prev_mag = torch.norm(prev, p=2, dim=-1)
            velocity_vector = velocity_vector / (prev_mag + self.epsilon)
            method = f"{method}+relative"

        return velocity_vector.mean().item(), method

    def _register_all_hooks(self):
        """Register forward hooks on all target layers."""
        def make_hook(name):
            def hook(module, input, output):
                output_tensor = self._normalize_output(output)
                # Apply pooling
                try:
                    pooled = self.pooling_fn(output_tensor)
                except Exception as e:
                    raise RuntimeError(f"Pooling failed for layer {name}: {e}")

                # Apply sparse sampling if requested
                if self.n_channels is not None:
                    if name not in self.channel_indices:
                        total_ch = pooled.shape[-1]
                        n = min(total_ch, self.n_channels)
                        self.channel_indices[name] = torch.randperm(total_ch, device=pooled.device)[:n]

                    pooled = pooled[..., self.channel_indices[name]]

                self.activations[name] = pooled.detach()
                
                if self.debug:
                    print(f"[DeepDrift] Hook {name}: pooled shape {pooled.shape}")
            return hook

        if self.layer_names:
            modules = dict(self.model.named_modules())
            missing_layers = [name for name in self.layer_names if name not in modules]
            if missing_layers:
                raise ValueError(f"Layer(s) not found in model: {missing_layers}")
            self.hooks = register_hooks(self.model, self.layer_names, make_hook)
            if self.debug:
                print(f"[DeepDrift] Registered hooks on: {self.layer_names}")

    def get_spatial_velocity(self) -> List[float]:
        """
        Returns L2-norms between successive layers for the current batch.
        Handles layers with different dimensions safely.
        """
        if len(self.activations) < 2:
            if self.debug:
                print(f"[DeepDrift] Not enough activations: {len(self.activations)}")
            return []

        available_names = self._ordered_activation_names()
        acts = [self.activations[name] for name in available_names]

        if self.debug:
            print(f"[DeepDrift] Computing spatial velocity for layers: {available_names}")
            for name, act in zip(available_names, acts):
                print(f"  {name}: shape {act.shape}")

        velocities = []
        for i in range(len(acts) - 1):
            a, b = acts[i], acts[i + 1]
            name_a = available_names[i]
            name_b = available_names[i + 1]

            vel, method = self._pair_velocity(a, b)

            velocities.append(vel)
            if self.debug:
                print(f"  {name_a} → {name_b}: vel={vel:.6f} ({method})")

        return velocities

    def get_temporal_velocity(self, step: Optional[int] = None) -> float:
        """
        Returns L2-norm between current and previous state of the first monitored layer.
        """
        if not self.activations:
            if self.debug:
                print("[DeepDrift] No activations for temporal velocity")
            return 0.0

        available_names = self._ordered_activation_names()
        if not available_names:
            return 0.0

        current_layer_name = available_names[0]
        current_state = self.activations[current_layer_name]

        if self.prev_state is None or step == 0:
            self.prev_state = current_state.clone()  # clone to avoid in-place modification
            if self.debug:
                print(f"[DeepDrift] Temporal: initialized prev_state from {current_layer_name}, shape {current_state.shape}")
            return 0.0

        velocity, _ = self._pair_velocity(self.prev_state, current_state)
        
        # Update previous state for next call
        self.prev_state = current_state.clone()
        
        if self.debug:
            print(f"[DeepDrift] Temporal velocity: {velocity:.6f}")
            if velocity == 0.0:
                print(f"[DeepDrift] WARNING: velocity is zero. prev and current may be identical.")
        
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
@@ -261,69 +305,108 @@ class DeepDriftMonitor:
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

    def get_layer_velocities(self) -> Dict[str, float]:
        """
        Returns velocity per layer pair for interpretability.
        """
        if len(self.activations) < 2:
            return {}

        available_names = self._ordered_activation_names()
        velocities = {}

        for i in range(len(available_names) - 1):
            name1 = available_names[i]
            name2 = available_names[i + 1]
            a = self.activations[name1]
            b = self.activations[name2]

            vel, _ = self._pair_velocity(a, b)

            velocities[f"{name1}→{name2}"] = vel

        return velocities

    def clear(self):
        """Reset stored activations and temporal state."""
        self.activations = {}
        self.prev_state = None
        if self.debug:
            print("[DeepDrift] Cleared activations and temporal state")

    def compute_velocity(self) -> torch.Tensor:
        """
        Returns a batch-wise velocity profile with shape [batch, transitions].
        """
        ordered_names = self._ordered_activation_names()
        ordered_acts = [self.activations[name] for name in ordered_names]

        if len(ordered_acts) < 2:
            raise RuntimeError("Need at least 2 layers with activations to compute velocity.")

        velocities = []
        for i in range(1, len(ordered_acts)):
            prev = ordered_acts[i - 1]
            cur = ordered_acts[i]

            if prev.device != cur.device:
                cur = cur.to(prev.device)

            if prev.shape == cur.shape:
                v = torch.norm(cur - prev, p=2, dim=-1)
            elif prev.shape[:-1] == cur.shape[:-1]:
                v = torch.abs(torch.norm(cur, p=2, dim=-1) - torch.norm(prev, p=2, dim=-1))
            else:
                raise ValueError(
                    f"Incompatible activation shapes for velocity profile: {prev.shape} vs {cur.shape}."
                )

            if self.relative_norm:
                prev_mag = torch.norm(prev, p=2, dim=-1)
                v = v / (prev_mag + self.epsilon)

            velocities.append(v)

        return torch.stack(velocities, dim=1)

    def get_drift_score(self, aggregate: bool = True) -> Union[torch.Tensor, float]:
        """
        Main drift metric based on velocity profile.

        Args:
            aggregate: If True, returns mean batch score as float.
                       If False, returns full [batch, transitions] tensor.
        """
        vel_profile = self.compute_velocity()
        if not aggregate:
            return vel_profile
        return vel_profile.sum(dim=1).mean().item()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        if self.debug:
            print("[DeepDrift] Removed all hooks")

    def reset_temporal(self):
        """Reset temporal velocity state."""
        self.prev_state = None
        if self.debug:
            print("[DeepDrift] Reset temporal state")
