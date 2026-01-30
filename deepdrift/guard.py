import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from transformers import PreTrainedModel, StoppingCriteria

class DeepDriftGuard(StoppingCriteria):
    """
    Real-time kinetic monitor for LLMs.
    Detects hallucinations by measuring 'Semantic Velocity' (hidden state tremor).
    Stops generation if velocity exceeds the dynamic threshold.
    """
    def __init__(self, model: PreTrainedModel, threshold="auto", warm_up=2):
        self.model = model
        self.warm_up = warm_up
        self.threshold = float('inf') if threshold == "auto" else threshold
        
        # Internal state
        self.hidden_states_history = []
        self.velocities = []
        self.stopped_reason = None
        self.is_calibrating = False
        self.hook_handle = None
        
        # 1. Auto-Discovery: Find the "Burning Bottleneck" layer
        self.target_layer = self._find_layer()
        # 2. Register Hook
        self._register_hooks()
        
        if threshold == "auto":
            print(f"📡 DeepDrift: Sensor attached to {self.target_layer}")

    def _find_layer(self):
        # Heuristic: MLP blocks in the middle of the network are best for hallucination detection
        candidates = [n for n, m in self.model.named_modules() 
                      if n.endswith('.mlp') or n.endswith('.output')]
        if not candidates:
            # Fallback for non-standard architectures
            candidates = [n for n, m in self.model.named_modules() if 'layer' in n]
        
        if candidates:
            return candidates[len(candidates) // 2] # Middle layer
        return list(self.model.named_modules())[-2][0]

    def _register_hooks(self):
        target_module = dict(self.model.named_modules())[self.target_layer]
        
        def hook(module, input, output):
            # Extract last token hidden state
            if isinstance(output, tuple): hidden = output[0]
            else: hidden = output
            
            # Detach to save memory
            self.hidden_states_history.append(hidden[:, -1, :].detach().cpu())
            self._compute_velocity_step()
            
        self.hook_handle = target_module.register_forward_hook(hook)

    def _compute_velocity_step(self):
        if len(self.hidden_states_history) < 2:
            self.velocities.append(0.0)
            return
        
        # Semantic Velocity Formula: v_t = ||h_t - h_{t-1}||
        h_t = self.hidden_states_history[-1]
        h_prev = self.hidden_states_history[-2]
        velocity = torch.norm(h_t - h_prev, dim=-1).item()
        self.velocities.append(velocity)

    def __call__(self, input_ids, scores, **kwargs):
        """Called by Hugging Face .generate() on every step"""
        if not self.velocities or self.is_calibrating: return False
        
        # Ignore warm-up phase (prompt processing tremor)
        if len(self.velocities) <= self.warm_up: return False

        current_velocity = self.velocities[-1]
        
        # Fail-Fast Logic
        if current_velocity > self.threshold:
            self.stopped_reason = f"STOP: Velocity {current_velocity:.2f} > {self.threshold:.2f}"
            return True # Stop generation
        return False

    def fit(self, tokenizer, calibration_prompts=["The capital of France is Paris."]):
        """
        Auto-calibrates the threshold using robust statistics (IQR).
        """
        print("🛠 DeepDrift: Auto-Calibrating...")
        self.is_calibrating = True
        max_velocities = []
        
        for prompt in calibration_prompts:
            self.hidden_states_history = []
            self.velocities = []
            inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
            self.model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
            
            if len(self.velocities) > self.warm_up:
                max_velocities.extend(self.velocities[self.warm_up:])
        
        if max_velocities:
            # IQR Rule: Q75 + 3.0 * IQR (Conservative threshold)
            q75, q25 = np.percentile(max_velocities, [75, 25])
            self.threshold = q75 + 3.0 * (q75 - q25)
            print(f"✅ DeepDrift: Threshold set to {self.threshold:.2f}")
        else:
            print("⚠️ Calibration failed (generation too short). Using default.")
            self.threshold = 15.0
        
        self.is_calibrating = False
        self.reset()

    def reset(self):
        self.hidden_states_history = []
        self.velocities = []
        self.stopped_reason = None

    def save_profile(self, path="deepdrift_profile.json"):
        data = {
            "target_layer": self.target_layer,
            "threshold": self.threshold,
            "warm_up": self.warm_up
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"💾 Profile saved to {path}")

    @classmethod
    def load_profile(cls, model, path="deepdrift_profile.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile {path} not found")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        guard = cls(model, threshold=data["threshold"], warm_up=data["warm_up"])
        
        # Ensure consistency
        if guard.target_layer != data["target_layer"]:
            # Re-register hook to the correct layer from profile
            guard.hook_handle.remove()
            guard.target_layer = data["target_layer"]
            guard._register_hooks()
            
        print(f"📂 DeepDrift Profile loaded. Threshold: {guard.threshold:.2f}")
        return guard

    def plot(self):
        """Visualizes the Seismograph of the last generation"""
        plt.figure(figsize=(10, 3))
        plt.plot(self.velocities, label='Semantic Velocity', color='#2c3e50', linewidth=2)
        plt.axhline(y=self.threshold, color='red', linestyle='--', label='Fail-Fast Threshold')
        
        if self.warm_up > 0:
            plt.axvspan(0, self.warm_up, color='gray', alpha=0.1, label='Warm-up')

        if self.stopped_reason:
            plt.scatter(len(self.velocities)-1, self.velocities[-1], color='red', s=100, zorder=5)
            plt.title(f"⚠️ {self.stopped_reason}", color='red', fontweight='bold')
        else:
            plt.title("✅ Laminar Flow (Stable)", color='green', fontweight='bold')
            
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
