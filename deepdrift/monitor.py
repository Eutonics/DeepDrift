import torch
import torch.nn.functional as F
import numpy as np
from .observer import LayerObserver, ObserverConfig, MonitorState
import warnings

class DeepDriftMonitor:
    """
    Advanced monitor tracking Mean, Cosine, Variance drift, and Semantic Velocity.
    Supports LLMs (Last Token physics) and Vision Models (Spatial Mean).
    """
    def __init__(self, model, arch_name=None, layers_map=None, drift_config=None, strategy='auto'):
        """
        strategy: 'auto', 'mean' (pooling), or 'last_token' (for generation dynamics)
        """
        self.model = model
        self.activations = {}
        self.hooks = []
        
        # Calibration Stats
        self.baseline_mu = {}   # Mean vector
        self.baseline_var = {}  # Variance vector
        self.baseline_norm = {} # Average norm (for cosine)
        
        # Dynamics Tracking (State t-1)
        self.prev_activations = {} 
        
        self.step_counter = 0
        
        # Strategy selection
        self.strategy = strategy
        if self.strategy == 'auto':
            # Heuristic: if it looks like a GPT/Llama, use last_token
            name_lower = (arch_name or "").lower()
            if any(x in name_lower for x in ['llama', 'gpt', 'mistral', 'qwen']):
                self.strategy = 'last_token'
            else:
                self.strategy = 'mean'
        
        print(f"🔧 DeepDrift Strategy: {self.strategy.upper()}")

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
        
        if 'resnet' in name_lower or 'convnext' in name_lower:
            # Vision Heuristics
            # ... (Existing logic for Vision) ...
            features = getattr(model, 'features', None)
            if features:
                 layers = {
                    'UV': features[0],
                    'Mid': features[len(features)//2],
                    'IR': features[-1]
                 }
        elif any(x in name_lower for x in ['vit', 'bert', 'llama', 'gpt', 'mistral']):
             # Transformer Heuristics
             blocks = []
             if hasattr(model, 'layers'): blocks = model.layers
             elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'): blocks = model.encoder.layers
             elif hasattr(model, 'h'): blocks = model.h
             elif hasattr(model, 'model') and hasattr(model.model, 'layers'): blocks = model.model.layers
             
             if len(blocks) > 0:
                 total = len(blocks)
                 # Map Top, Middle, Bottom
                 layers = {
                     'UV': blocks[0],
                     'Mid': blocks[total // 2],
                     'Deep': blocks[int(total * 0.75)],
                     'IR': getattr(model, 'norm', None) or getattr(model.model, 'norm', None) or blocks[-1]
                 }
                 
        return {k: v for k, v in layers.items() if v is not None}

    def _hook_fn(self, name):
        def hook(module, input, output):
            # 1. ROBUST UNWRAPPING (Fix for Llama/HF)
            data = output
            if isinstance(output, tuple):
                data = output[0]
            elif hasattr(output, 'last_hidden_state'):
                data = output.last_hidden_state
            elif hasattr(output, 'hidden_states'):
                data = output.hidden_states[-1]
            
            # 2. AGGREGATION STRATEGY
            # Ensure we have a tensor
            if not isinstance(data, torch.Tensor):
                return # Fail silently or log
            
            # [Batch, Seq, Dim] -> [Batch, Dim]
            if data.dim() == 3:
                if self.strategy == 'last_token':
                    # Physics of Generation: The "Tip of the spear"
                    act = data[:, -1, :]
                else:
                    # Physics of Classification: Context Mean
                    act = data.mean(dim=1)
            elif data.dim() == 4: # CNN [B, C, H, W]
                act = data.mean(dim=[2, 3])
            else:
                act = data.flatten(1)
                
            self.activations[name] = act.detach().float() # Force float32 for stats
        return hook

    def _register_hooks(self):
        # Clear existing hooks first if re-registering
        for h in self.hooks: h.remove()
        self.hooks = []
        
        for name, module in self.layers.items():
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def calibrate(self, loader, max_batches=50):
        print("⚙️ DeepDrift: Calibrating advanced statistics...")
        self.model.eval()
        
        acc_mu = {k: [] for k in self.layers}
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches: break
                
                # Input Normalization
                if isinstance(batch, dict): 
                    # HF Tokenizer output
                    if 'input_ids' in batch:
                        x = batch['input_ids'].to(self.device)
                        mask = batch.get('attention_mask', None)
                        if mask is not None: mask = mask.to(self.device)
                        # We pass kwargs to support models that need attention_mask
                        try:
                            _ = self.model(input_ids=x, attention_mask=mask)
                        except:
                            _ = self.model(x)
                    else:
                        # Generic dict
                        x = list(batch.values())[0].to(self.device)
                        _ = self.model(x)
                else:
                    # Tensor or List
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    if hasattr(x, 'to'): x = x.to(self.device)
                    _ = self.model(x)

                for k in self.layers: 
                    if k in self.activations:
                        acc_mu[k].append(self.activations[k])
        
        # Calculate Stats
        for k in self.layers:
            if len(acc_mu[k]) > 0:
                data = torch.cat(acc_mu[k], dim=0) # [N_total, Dim]
                
                self.baseline_mu[k] = data.mean(dim=0)
                # Robust variance (prevent division by zero)
                self.baseline_var[k] = data.var(dim=0) + 1e-5
                self.baseline_norm[k] = torch.norm(self.baseline_mu[k]) + 1e-9
            else:
                warnings.warn(f"Layer {k} captured no data during calibration!")
                
        print("✅ Calibration complete.")

    def step(self, inputs):
        self.model.eval()
        self.step_counter += 1
        
        # Forward Pass
        with torch.no_grad():
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                _ = self.model(**inputs)
            elif torch.is_tensor(inputs):
                _ = self.model(inputs.to(self.device))
            else:
                 # Fallback
                 _ = self.model(inputs)
        
        current_status = {}
        alerts = []
        
        for name in self.layers:
            # Check if hook fired
            if name not in self.activations:
                continue
                
            if name not in self.baseline_mu:
                # If not calibrated, return raw data or skip
                continue

            batch = self.activations[name] # [Batch, Dim]
            # Handle Batch>1 by averaging for the metric
            batch_mu = batch.mean(dim=0) 
            
            # --- METRIC 1: Euclidean Shift (Normalized) ---
            diff_sq = (batch_mu - self.baseline_mu[name]) ** 2
            # Mean of Z-scores per neuron
            z_euclid = torch.mean(torch.sqrt(diff_sq / self.baseline_var[name])).item()
            
            # --- METRIC 2: Cosine Drift (Angle) ---
            dot = torch.dot(batch_mu, self.baseline_mu[name])
            curr_norm = torch.norm(batch_mu) + 1e-9
            cosine_sim = dot / (curr_norm * self.baseline_norm[name])
            drift_cosine = 1.0 - cosine_sim.item()
            
            # --- METRIC 3: Semantic Velocity (New!) ---
            velocity = 0.0
            if name in self.prev_activations:
                prev = self.prev_activations[name]
                # L2 Distance between current and prev state
                # Note: Assuming batch size 1 for simplicity in generation, or mean otherwise
                velocity = torch.norm(batch_mu - prev.mean(dim=0)).item()
            
            # Store current as prev for next step
            self.prev_activations[name] = batch.detach().clone()
            
            # --- Observer Logic ---
            state, event = self.observers[name].update(z_euclid, self.step_counter)
            
            current_status[name] = {
                'drift': z_euclid,
                'velocity': velocity,     # <--- ADDED
                'drift_cosine': drift_cosine,
                'state': state.value
            }
            
            if event: alerts.append(event)
            
        return current_status, alerts

    def close(self):
        for h in self.hooks: h.remove()
        self.hooks = []
