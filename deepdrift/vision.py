import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class VisionDiagnosis:
    drift_score: float
    status: str  # "STABLE", "WARNING", "CRITICAL"
    layers_drift: dict
    
    def __repr__(self):
        icon = "🟢" if self.status == "STABLE" else ("🟡" if self.status == "WARNING" else "🔴")
        return f"{icon} Vision Diagnosis: {self.status} | Max Drift: {self.drift_score:.2f}"

class DeepDriftVision:
    """
    Spatial ODD Monitor for Vision Models (CNN/ViT).
    Implements 'Burning Bottleneck' detection and Sparse Sampling.
    """
    def __init__(self, model: nn.Module, auto_hook=True, n_channels=50):
        self.model = model
        self.n_channels = n_channels
        self.activations = {}
        self.hooks = []
        self.baseline_mu = {}
        self.baseline_sigma = {}
        self.thresholds = {}
        
        if auto_hook:
            self._auto_hook()

    def _auto_hook(self):
        # 1. Find all convolutional or linear layers
        all_layers = [n for n, m in self.model.named_modules() 
                      if isinstance(m, (nn.Conv2d, nn.Linear))]
        
        # 2. Select layers in the middle (Burning Bottleneck region: 50%-80%)
        if len(all_layers) > 3:
            start = int(len(all_layers) * 0.5)
            end = int(len(all_layers) * 0.8)
            step = max(1, (end - start) // 2)
            target_layers = all_layers[start:end:step][:3]
        else:
            target_layers = all_layers

        print(f"👁 DeepDrift Vision: Hooked layers {target_layers}")
        
        for name in target_layers:
            layer = dict(self.model.named_modules())[name]
            self.hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(model, input, output):
            # Spatial Pooling & Sparse Sampling
            if output.dim() == 4: # CNN [B, C, H, W]
                # Take first N channels only (Optimization from paper v4.1)
                ch = min(output.shape[1], self.n_channels)
                val = output[:, :ch].mean(dim=[2, 3]) # Global Avg Pooling
            elif output.dim() == 3: # ViT [B, Seq, Dim]
                # Take CLS token + Sparse channels
                ch = min(output.shape[2], self.n_channels)
                val = output[:, 0, :ch]
            else:
                ch = min(output.shape[1], self.n_channels)
                val = output[:, :ch]
                
            self.activations[name] = val.detach()
        return hook

    def fit(self, dataloader, device="cpu"):
        """Calibrate baseline statistics (Mu, Sigma) on normal data."""
        print("🛠 DeepDrift: Calibrating Vision Monitor...")
        self.model.eval()
        accumulated = defaultdict(list)
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i > 50: break # Calibration limit
                if isinstance(batch, (tuple, list)): batch = batch[0]
                
                self.model(batch.to(device))
                for name, val in self.activations.items():
                    accumulated[name].append(val.cpu())
        
        # Calculate Stats per layer
        for name, data in accumulated.items():
            data = torch.cat(data, dim=0)
            self.baseline_mu[name] = data.mean(dim=0)
            self.baseline_sigma[name] = data.std(dim=0) + 1e-6
            
            # Calculate threshold (99th percentile of training drift)
            z_scores = torch.abs((data - self.baseline_mu[name]) / self.baseline_sigma[name]).mean(dim=1)
            self.thresholds[name] = np.percentile(z_scores.numpy(), 99)
            
        print(f"✅ Calibration Done. Thresholds: {[f'{v:.2f}' for v in self.thresholds.values()]}")

    def predict(self, x) -> VisionDiagnosis:
        """Analyze a single batch for anomalies."""
        self.model.eval()
        with torch.no_grad():
            self.model(x)
        
        layer_drifts = {}
        max_drift_ratio = 0.0
        
        for name, val in self.activations.items():
            mu = self.baseline_mu[name].to(val.device)
            sigma = self.baseline_sigma[name].to(val.device)
            threshold = self.thresholds[name]
            
            # Calculate Drift
            drift = torch.abs((val - mu) / sigma).mean().item()
            
            # Normalize by threshold ( > 1.0 means anomaly)
            ratio = drift / threshold if threshold > 0 else 0
            layer_drifts[name] = ratio
            
            if ratio > max_drift_ratio:
                max_drift_ratio = ratio

        status = "STABLE"
        if max_drift_ratio > 1.0: status = "WARNING"
        if max_drift_ratio > 2.0: status = "CRITICAL"
        
        return VisionDiagnosis(max_drift_ratio, status, layer_drifts)
