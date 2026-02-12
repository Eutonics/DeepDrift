import torch
from .core import DeepDriftMonitor
from .diagnostics import VisionDiagnosis

class DeepDriftVision:
    def __init__(self, model, auto_hook=True, n_channels=50, pooling='cls'):
        """
        Backward compatible wrapper for Vision tasks.
        Uses DeepDriftMonitor under the hood.
        """
        self.model = model
        self.monitor = DeepDriftMonitor(
            model,
            pooling=pooling,
            n_channels=n_channels
        )
        self.calibrated = False

    def fit(self, dataloader, device='cpu'):
        """Calibrate baseline statistics."""
        stats = self.monitor.calibrate(dataloader, device=device)
        self.calibrated = True
        return stats

    def predict(self, x) -> VisionDiagnosis:
        """Analyze a batch for anomalies."""
        _ = self.monitor.model(x)
        velocities = self.monitor.get_spatial_velocity()
        peak_vel = max(velocities) if velocities else 0.0

        is_anomaly = False
        if self.monitor.threshold is not None:
            is_anomaly = peak_vel > self.monitor.threshold

        return VisionDiagnosis(
            peak_velocity=peak_vel,
            layer_velocities=velocities,
            is_anomaly=is_anomaly,
            drift_score=peak_vel, # Backward compatibility
            status="ANOMALY" if is_anomaly else "NORMAL",
            threshold=self.monitor.threshold
        )

    def remove_hooks(self):
        self.monitor.remove_hooks()
