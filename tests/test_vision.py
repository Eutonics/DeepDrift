import torch
import torch.nn as nn
from deepdrift import DeepDriftVision
from deepdrift.diagnostics import VisionDiagnosis

class MockVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))

def test_vision_predict():
    model = MockVision()
    # Mock some layer names for the heuristic
    for i, m in enumerate(model.encoder):
        setattr(model.encoder, f'layer_{i}', m)

    # We'll explicitly set layer names for test stability
    monitor = DeepDriftVision(model)
    monitor.monitor.layer_names = ['encoder.0', 'fc']
    monitor.monitor._register_all_hooks()

    # Calibrate
    calib_data = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    monitor.fit(calib_data)

    # Predict
    x = torch.randn(1, 3, 32, 32)
    diag = monitor.predict(x)

    assert isinstance(diag, VisionDiagnosis)
    assert hasattr(diag, 'peak_velocity')
    assert hasattr(diag, 'is_anomaly')
