import torch
import numpy as np
from tqdm.auto import tqdm

class DeepDriftMonitor:
    """
    Monitors layer-wise representation drift to detect OOD shifts and architectural failures.
    """
    def __init__(self, model, arch_name=None, layers_map=None):
        """
        Args:
            model: PyTorch model (nn.Module).
            arch_name: String ('ResNet-18', 'ViT-B/16', etc.) for auto-mapping.
            layers_map: Dictionary {'LayerName': module} mapping UV -> IR depth.
        """
        self.model = model
        self.activations = {}
        self.hooks = []
        self.mu = {}
        self.sigma = {}
        # Определяем устройство по первому параметру модели
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = 'cpu'
        
        if layers_map is not None:
            self.layers = layers_map
        elif arch_name is not None:
            self.layers = self._auto_detect_layers(model, arch_name)
        else:
            # Fallback: try to grab top-level children if nothing specified
            print("DeepDrift Warning: No architecture specified. Using top-level modules.")
            self.layers = {name: module for name, module in model.named_children()}

        self._register_hooks()

    def _auto_detect_layers(self, model, arch_name):
        layers = {}
        if 'ResNet' in arch_name or 'resnet' in arch_name.lower():
            layers = {
                'UV': getattr(model, 'layer1', None) or getattr(model, 'features', None)[0],
                'Mid': getattr(model, 'layer2', None) or getattr(model, 'features', None)[4],
                'Deep': getattr(model, 'layer3', None) or getattr(model, 'features', None)[6],
                'IR': getattr(model, 'layer4', None) or getattr(model, 'features', None)[-1]
            }
        # Упрощенная логика для старта. Пользователь может передать свой map.
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
                # Handle tuple (x, y) or just x
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

    def scan(self, inputs):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs)
        
        profile = []
        for k in self.layers:
            if k in self.activations and k in self.mu:
                batch_mu = self.activations[k].mean(dim=0)
                dist = torch.norm(batch_mu - self.mu[k]).item()
                z_score = dist / self.sigma[k]
                profile.append(z_score)
            
        return profile

    def close(self):
        for h in self.hooks: h.remove()
