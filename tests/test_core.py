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
        
class TupleLayer(nn.Module):
    def forward(self, x):
        return (x, x * 0)


class TupleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.tuple_layer = TupleLayer()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tuple_layer(x)[0]
        return self.fc2(x)

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


def test_auto_pooling_and_relative_velocity_profile():
    model = SimpleModel()
    monitor = DeepDriftMonitor(
        model,
        layer_names=['fc1', 'fc2'],
        pooling='auto',
        relative_norm=True,
    )

    x = torch.randn(3, 10)
    _ = model(x)

    profile = monitor.compute_velocity()
    assert profile.shape == (3, 1)
    assert torch.all(profile >= 0)


def test_tuple_outputs_supported_and_drift_score():
    model = TupleModel()
    monitor = DeepDriftMonitor(model, layer_names=['fc1', 'tuple_layer', 'fc2'], pooling='flatten')

    x = torch.randn(2, 10)
    _ = model(x)

    score = monitor.get_drift_score(aggregate=True)
    profile = monitor.get_drift_score(aggregate=False)

    assert isinstance(score, float)
    assert profile.shape == (2, 2)


def test_missing_layer_raises_value_error():
    model = SimpleModel()
    with pytest.raises(ValueError, match=r"Layer\(s\) not found"):
        DeepDriftMonitor(model, layer_names=['fc1', 'missing.layer'], pooling='flatten')
