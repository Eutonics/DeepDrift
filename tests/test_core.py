import torch
import torch.nn as nn
import pytest
from deepdrift import DeepDriftMonitor

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))

def test_spatial_velocity():
    model = SimpleModel()
    monitor = DeepDriftMonitor(model, layer_names=['fc1', 'fc2'], pooling='flatten')

    x = torch.randn(2, 10)
    _ = model(x)

    v = monitor.get_spatial_velocity()
    assert len(v) == 1
    assert v[0] >= 0

def test_temporal_velocity():
    model = SimpleModel()
    monitor = DeepDriftMonitor(model, layer_names=['fc1'], pooling='flatten')

    x1 = torch.randn(1, 10)
    _ = model(x1)
    v1 = monitor.get_temporal_velocity(step=0)
    assert v1 == 0

    x2 = torch.randn(1, 10)
    _ = model(x2)
    v2 = monitor.get_temporal_velocity(step=1)
    assert v2 > 0

def test_calibration():
    model = SimpleModel()
    monitor = DeepDriftMonitor(model, layer_names=['fc1', 'fc2'], pooling='flatten')

    data = [torch.randn(4, 10) for _ in range(5)]
    stats = monitor.calibrate(data)

    assert "threshold" in stats
    assert monitor.threshold is not None
    assert stats["threshold"] > 0
