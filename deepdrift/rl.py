import torch
from .core import DeepDriftMonitor
from .diagnostics import VelocityDiagnosis

class DeepDriftRL:
    def __init__(self, agent_model, threshold=None, n_channels=32):
        """
        Monitor for RL Agents.
        Detects sudden changes in policy/value function activations (Early Warning systems).
        """
        self.model = agent_model
        # RL models often use flatten pooling for feature extraction
        self.monitor = DeepDriftMonitor(
            agent_model,
            pooling='flatten',
            n_channels=n_channels
        )
        if threshold:
            self.monitor.threshold = threshold

    def step(self, obs) -> VelocityDiagnosis:
        """
        Call this during the agent's step.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float().unsqueeze(0)

        # Forward pass (ensure it triggers the internal model)
        _ = self.model(obs)

        # RL can use both temporal (sudden change in state) or spatial (unusual activation patterns)
        # We focus on temporal velocity for crash prediction
        vel = self.monitor.get_temporal_velocity()

        is_anomaly = False
        if self.monitor.threshold is not None:
            is_anomaly = vel > self.monitor.threshold

        return VelocityDiagnosis(
            peak_velocity=vel,
            layer_velocities=[vel],
            is_anomaly=is_anomaly,
            threshold=self.monitor.threshold,
            status="CRITICAL" if is_anomaly else "NORMAL"
        )
