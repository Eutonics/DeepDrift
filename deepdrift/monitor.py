import torch
import torch.nn.functional as F
import numpy as np
from .observer import LayerObserver, ObserverConfig, MonitorState

class DeepDriftMonitor:
    """
    Advanced monitor tracking Mean, Cosine, and Variance drift per layer.
    """
    def __init__(self, model, arch_name=None, layers_map=None, drift_config=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        
        # Calibration Stats
        self.baseline_mu = {}   # Mean vector
        self.baseline_var = {}  # Variance vector
        self.baseline_norm = {} # Average norm (for cosine)
        
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
        elif 'vit' in name_lower or 'bert' in name_lower or 'llama' in name_lower or 'gpt' in name_lower:
             # Generic transformer heuristic
             blocks = []
             if hasattr(model, 'layers'): blocks = model.layers
             elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'): blocks = model.encoder.layers
             elif hasattr(model, 'h'): blocks = model.h # GPT-2
             elif hasattr(model, 'model') and hasattr(model.model, 'layers'): blocks = model.model.layers # Llama
             
             if len(blocks) > 0:
                 total = len(blocks)
                 layers = {
                     'UV': blocks[0],
                     'Mid': blocks[total // 2],
                     'IR': blocks[-1]
                 }
                 
        return {k: v for k, v in layers.items() if v is not None}

    def _hook_fn(self, name):
        def hook(model, input, output):
            # Universal Adapter
            if isinstance(output, tuple): output = output[0]
            
            if hasattr(output, 'dim'):
                if output.dim() == 4: # CNN
                    act = output.mean(dim=[2, 3])
                elif output.dim() == 3: # Transformer (Batch, Seq, Dim)
                    # For robust stats, mean over sequence is safer than just CLS
                    act = output.mean(dim=1) 
                else:
                    act = output.flatten(1)
                self.activations[name] = act.detach()
        return hook

    def _register_hooks(self):
        for name, module in self.layers.items():
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def calibrate(self, loader, max_batches=50):
        print("⚙️ DeepDrift: Calibrating advanced statistics...")
        self.model.eval()
        
        acc_mu = {k: [] for k in self.layers}
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches: break
                # Handle different loader formats (dict vs tuple)
                if isinstance(batch, dict): x = batch['input_ids'] if 'input_ids' in batch else list(batch.values())[0]
                elif isinstance(batch, (list, tuple)): x = batch[0]
                else: x = batch
                
                if hasattr(x, 'to'): x = x.to(self.device)
                
                try:
                    _ = self.model(x)
                except:
                    # Fallback for HF models requiring kwargs
                    if isinstance(batch, dict): 
                        batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                        _ = self.model(**batch)

                for k in self.layers: acc_mu[k].append(self.activations[k])
        
        for k in self.layers:
            if len(acc_mu[k]) > 0:
                data = torch.cat(acc_mu[k], dim=0) # [N_total, Dim]
                
                # 1. Mean
                self.baseline_mu[k] = data.mean(dim=0)
                # 2. Variance (Diagonal)
                self.baseline_var[k] = data.var(dim=0) + 1e-6
                # 3. Norm (for Cosine)
                self.baseline_norm[k] = torch.norm(self.baseline_mu[k]) + 1e-9
                
        print("✅ Calibration complete (Mean + Variance + Geometry).")

    def step(self, inputs):
        self.model.eval()
        self.step_counter += 1
        
        with torch.no_grad():
            # Support both Tensor and Dict inputs
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                _ = self.model(**inputs)
            else:
                _ = self.model(inputs)
        
        current_status = {}
        alerts = []
        
        for name in self.layers:
            if name in self.activations and name in self.baseline_mu:
                batch = self.activations[name]
                batch_mu = batch.mean(dim=0)
                
                # --- METRIC 1: Euclidean Shift (Normalized by Variance) ---
                diff_sq = (batch_mu - self.baseline_mu[name]) ** 2
                z_euclid = torch.sqrt(torch.sum(diff_sq / self.baseline_var[name])).item()
                z_euclid /= np.sqrt(batch.shape[1]) 
                
                # --- METRIC 2: Cosine Drift (Angle) ---
                dot = torch.dot(batch_mu, self.baseline_mu[name])
                curr_norm = torch.norm(batch_mu) + 1e-9
                cosine_sim = dot / (curr_norm * self.baseline_norm[name])
                drift_cosine = 1.0 - cosine_sim.item()
                
                # --- METRIC 3: Variance Shift ---
                drift_energy = (curr_norm / self.baseline_norm[name]).item()
                
                # Main signal for Observer
                z_scaled = z_euclid * 10.0 
                
                state, event = self.observers[name].update(z_scaled, self.step_counter)
                
                current_status[name] = {
                    'drift': z_scaled,
                    'drift_cosine': drift_cosine,
                    'drift_energy': drift_energy,
                    'slope': self.observers[name].current_beta,
                    'state': state.value
                }
                
                if event: alerts.append(event)
            
        return current_status, alerts

    def close(self):
        for h in self.hooks: h.remove()
