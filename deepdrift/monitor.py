import torch
import numpy as np
from .observer import LayerObserver, ObserverConfig, MonitorState

class DeepDriftMonitor:
    """
    Monitors layer-wise representation drift with stateful alerting system.
    """
    def __init__(self, model, arch_name=None, layers_map=None, drift_config=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.mu = {}
        self.sigma = {}
        self.step_counter = 0
        
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = 'cpu'
        
        # 1. Setup Layers
        if layers_map is not None:
            self.layers = layers_map
        elif arch_name is not None:
            self.layers = self._auto_detect_layers(model, arch_name)
        else:
            self.layers = {name: module for name, module in model.named_children()}

        # 2. Setup Observers
        cfg = drift_config if drift_config else ObserverConfig()
        self.observers = {
            name: LayerObserver(name, cfg) for name in self.layers
        }

        self._register_hooks()

    def _auto_detect_layers(self, model, arch_name):
        layers = {}
        name_lower = arch_name.lower()
        if 'resnet' in name_lower:
            layers = {
                'UV': getattr(model, 'layer1', None) or getattr(model, 'features', None)[0],
                'Mid': getattr(model, 'layer2', None) or getattr(model, 'features', None)[4],
                'Deep': getattr(model, 'layer3', None) or getattr(model, 'features', None)[6],
                'IR': getattr(model, 'layer4', None) or getattr(model, 'features', None)[-1]
            }
        return {k: v for k, v in layers.items() if v is not None}

    def _hook_fn(self, name):
        def hook(model, input, output):
            if output.dim() == 4: 
                act = output.mean(dim=[2, 3])
            elif output.dim() == 3: 
                act = output[:, 0, :] 
            else:
                act = output.flatten(1)
            self.activations[name] = act.detach()
        return hook

    def _register_hooks(self):
        for name, module in self.layers.items():
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def calibrate(self, loader, max_batches=50):
        print("⚙️ DeepDrift: Calibrating baseline...")
        self.model.eval()
        acc = {k: [] for k in self.layers}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches: break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                _ = self.model(x)
                for k in self.layers: acc[k].append(self.activations[k])
        
        for k in self.layers:
            if len(acc[k]) > 0:
                d = torch.cat(acc[k], dim=0)
                self.mu[k] = d.mean(dim=0)
                dist = torch.norm(d - self.mu[k], dim=1)
                self.sigma[k] = dist.std().item() + 1e-9
        print("✅ Calibration complete.")

    def step(self, inputs):
        """
        Process one batch and return system status.
        """
        self.model.eval()
        self.step_counter += 1
        
        with torch.no_grad():
            _ = self.model(inputs)
        
        current_status = {}
        alerts = []
        
        for name in self.layers:
            if name in self.activations and name in self.mu:
                batch_mu = self.activations[name].mean(dim=0)
                dist = torch.norm(batch_mu - self.mu[name]).item()
                z_score = dist / self.sigma[name]
                
                state, event = self.observers[name].update(z_score, self.step_counter)
                
                current_status[name] = {
                    'drift': z_score,
                    'slope': self.observers[name].current_beta,
                    'state': state.value
                }
                if event: alerts.append(event)
            
        return current_status, alerts

    def close(self):
        for h in self.hooks: h.remove()
