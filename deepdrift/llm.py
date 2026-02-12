import torch
from .core import DeepDriftMonitor
from .diagnostics import LLMDiagnosis

class DeepDriftGuard:
    def __init__(self, model, threshold=None, n_channels=50, pooling='cls'):
        """
        Monitor for LLM generation tasks.
        Uses temporal velocity of the CLS/last token.
        """
        self.model = model
        self.monitor = DeepDriftMonitor(
            model,
            pooling=pooling,
            n_channels=n_channels
        )
        if threshold:
            self.monitor.threshold = threshold

    def reset(self):
        """Reset sequence history."""
        self.monitor.clear()

    def __call__(self, input_ids, **kwargs) -> LLMDiagnosis:
        """
        To be used during generation loop.
        """
        # Forward pass to trigger hooks
        _ = self.model(input_ids, **kwargs)

        velocity = self.monitor.get_temporal_velocity()
        is_anomaly = False
        if self.monitor.threshold is not None:
            is_anomaly = velocity > self.monitor.threshold

        return LLMDiagnosis(
            velocity=velocity,
            is_anomaly=is_anomaly,
            threshold=self.monitor.threshold,
            status="HALUCINATION" if is_anomaly else "NORMAL"
        )
