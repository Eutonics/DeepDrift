import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union

class DeepDriftMonitor:
    """
    Production-grade monitor for Neural Network Kinetic Dynamics (Semantic Velocity).
    
    Key Features:
    - Sparse Channel Sampling: Monitors only N random channels (drastically reduces overhead).
    - Global Average Pooling: Makes drift calculation spatial-invariant and fast.
    - EMA Smoothing: Reduces noise in the velocity signal.
    - Robust Thresholding: Uses IQR (Interquartile Range) instead of StdDev.
    """
    def __init__(
        self, 
        model: nn.Module, 
        layers_map: Optional[Dict[str, nn.Module]] = None,
        n_channels: int = 50,
        ema_alpha: float = 0.1
    ):
        self.model = model
        self.n_channels = n_channels
        self.ema_alpha = ema_alpha
        
        self.hooks = []
        self.activations = {}
        self.layer_indices = {} # Cache for stratified sampling indices
        
        # Statistics
        self.baseline_mean = {}
        self.baseline_std = {}
        self.drift_ema = None
        
        # Thresholds
        self.threshold_warning = float('inf')
        self.threshold_critical = float('inf')
        self.is_calibrated = False

        # Auto-detect layers if not provided
        if layers_map is None:
            self.target_layers = self._auto_detect_layers()
        else:
            self.target_layers = layers_map
            
        self._register_hooks()

    def _auto_detect_layers(self) -> Dict[str, nn.Module]:
        """Heuristic to find impactful layers in the middle/end of the network."""
        candidates = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Skip very small layers (e.g. initial projections)
                if hasattr(module, 'weight') and module.weight.numel() > 10000:
                    candidates.append((name, module))
        
        # Pick up to 4 layers evenly spaced
        if len(candidates) > 4:
            indices = np.linspace(0, len(candidates)-1, 4, dtype=int)
            return {candidates[i][0]: candidates[i][1] for i in indices}
        return {name: module for name, module in candidates}

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                # 1. Global Average Pooling (B, C, H, W -> B, C)
                if output.dim() == 4:
                    flat = output.mean(dim=[2, 3])
                elif output.dim() == 3: # Transformer (B, S, D) -> Take mean over sequence
                    flat = output.mean(dim=1) 
                else:
                    flat = output
                
                # 2. Sparse Sampling (Deterministically random channels)
                C = flat.shape[1]
                if name not in self.layer_indices:
                    limit = min(C, self.n_channels)
                    g = torch.Generator().manual_seed(42) # Fixed seed for consistency
                    self.layer_indices[name] = torch.randperm(C, generator=g)[:limit]
                
                indices = self.layer_indices[name].to(flat.device)
                self.activations[name] = flat[:, indices].detach()
            return hook

        for name, layer in self.target_layers.items():
            self.hooks.append(layer.register_forward_hook(get_hook(name)))

    def calibrate(self, dataloader, device=None, max_batches=50):
        """
        Calibrates baseline statistics (Mean/Std) and Thresholds (IQR) on normal data.
        """
        if device is None:
            # Try to guess device from model parameters
            try:
                device = next(self.model.parameters()).device
            except:
                device = 'cpu'

        print(f"⚙️ DeepDrift: Calibrating on {min(len(dataloader), max_batches)} batches...")
        stats_collector = {name: [] for name in self.target_layers}
        self.model.eval()
        
        # 1. Collect Activations
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches: break
                
                # Handle tuple (x, y) or just x
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(device)
                _ = self.model(x) # Forward pass triggers hooks
                
                for name in self.target_layers:
                    stats_collector[name].append(self.activations[name])

        # 2. Compute Mean/Std
        for name, data in stats_collector.items():
            if not data: continue
            data_cat = torch.cat(data, dim=0)
            self.baseline_mean[name] = data_cat.mean(dim=0)
            self.baseline_std[name] = data_cat.std(dim=0) + 1e-6

        # 3. Compute Thresholds (Re-run to calculate drift scores)
        drifts = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches: break
                if isinstance(batch, (list, tuple)): x = batch[0]
                else: x = batch
                x = x.to(device)
                _ = self.model(x)
                drifts.append(self._compute_instant_drift())
        
        drifts = np.array(drifts)
        q75, q25 = np.percentile(drifts, [75, 25])
        iqr = q75 - q25
        
        self.threshold_warning = q75 + 1.5 * iqr
        self.threshold_critical = q75 + 3.0 * iqr
        self.is_calibrated = True
        
        print(f"✅ Calibration Complete. Thresholds: Warning={self.threshold_warning:.2f}, Critical={self.threshold_critical:.2f}")

    def _compute_instant_drift(self) -> float:
        total_drift = 0
        count = 0
        for name in self.target_layers:
            if name not in self.activations: continue
            
            act = self.activations[name]
            # Z-score distance: || (x - mu) / sigma ||
            mean = self.baseline_mean[name].to(act.device)
            std = self.baseline_std[name].to(act.device)
            
            z = (act - mean) / std
            drift = torch.norm(z, p=2, dim=1).mean().item()
            total_drift += drift
            count += 1
            
        return total_drift / max(count, 1)

    def step(self, x_input=None) -> tuple[float, dict]:
        """
        Calculates Kinetic Drift for the current forward pass.
        Returns: (drift_value, status_dict)
        """
        if x_input is not None:
            with torch.no_grad():
                _ = self.model(x_input)
        
        raw_drift = self._compute_instant_drift()
        
        # EMA Update
        if self.drift_ema is None:
            self.drift_ema = raw_drift
        else:
            self.drift_ema = (self.ema_alpha * raw_drift) + ((1 - self.ema_alpha) * self.drift_ema)
            
        # Status
        if self.drift_ema > self.threshold_critical:
            status = "CRITICAL"
        elif self.drift_ema > self.threshold_warning:
            status = "WARNING"
        else:
            status = "OK"
            
        info = {
            "drift_ema": self.drift_ema,
            "drift_raw": raw_drift,
            "status": status,
            "layers": list(self.target_layers.keys())
        }
            
        return self.drift_ema, info

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
