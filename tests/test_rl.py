import torch
import torch.nn as nn

from deepdrift.rl import DeepDriftRL


class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_calibrate_baseline_global():
    model = SimplePolicy()
    rl = DeepDriftRL(model, use_ewma=True, ewma_alpha=0.3, z_threshold=2.0)

    baseline = [
        [0.10, 0.12, 0.11, 0.09],
        [0.11, 0.10, 0.12, 0.10],
    ]
    stats = rl.calibrate_baseline(baseline, per_step=False)

    assert stats["mode"] == "global"
    assert "mean" in stats and "std" in stats
    assert stats["n_points"] == 8


def test_calibrate_baseline_per_step():
    model = SimplePolicy()
    rl = DeepDriftRL(model, use_ewma=True, ewma_alpha=0.3, z_threshold=2.0)

    baseline = [
        [0.10, 0.20, 0.30],
        [0.11, 0.21, 0.31],
        [0.09, 0.19, 0.29],
    ]
    stats = rl.calibrate_baseline(baseline, per_step=True)

    assert stats["mode"] == "per_step"
    assert stats["n_steps"] == 3


def test_step_produces_extended_diagnosis_fields():
    model = SimplePolicy()
    rl = DeepDriftRL(model, use_ewma=True, ewma_alpha=0.5, z_threshold=None)

    vals = iter([0.2, 0.6])
    rl.monitor.get_temporal_velocity = lambda step=None: next(vals)

    x = torch.randn(1, 10)
    d1 = rl.step(x, step_idx=0)
    d2 = rl.step(x, step_idx=1)

    assert d1.raw_velocity == 0.2
    assert d1.smoothed_velocity == 0.2
    assert d1.z_score is None
    assert d1.episode_step == 0

    assert d2.raw_velocity == 0.6
    assert d2.smoothed_velocity is not None
    assert d2.smoothed_velocity != d2.raw_velocity
    assert d2.episode_step == 1


def test_zscore_trigger_and_latency_tracking():
    model = SimplePolicy()
    rl = DeepDriftRL(model, use_ewma=False, z_threshold=2.0)
    rl.calibrate_baseline([[0.1, 0.1, 0.1, 0.1]], per_step=False)

    vals = iter([0.1, 0.1, 1.0, 1.2])
    rl.monitor.get_temporal_velocity = lambda step=None: next(vals)

    x = torch.randn(1, 10)
    out0 = rl.step(x, step_idx=0)
    out1 = rl.step(x, step_idx=1)
    out2 = rl.step(x, step_idx=2)
    out3 = rl.step(x, step_idx=3)

    assert out0.is_anomaly is False
    assert out1.is_anomaly is False
    assert out2.is_anomaly is True
    assert out2.detected_at_step == 2
    assert out2.detection_latency == 2
    assert out3.detected_at_step == 2
    assert out3.detection_latency == 2


def test_reset_episode_resets_detection_state():
    model = SimplePolicy()
    rl = DeepDriftRL(model, use_ewma=False, z_threshold=0.0)
    rl.calibrate_baseline([[0.0, 0.0]], per_step=False)

    vals = iter([1.0, 1.0])
    rl.monitor.get_temporal_velocity = lambda step=None: next(vals)
    x = torch.randn(1, 10)

    d = rl.step(x, step_idx=0)
    assert d.is_anomaly is True
    assert d.detected_at_step == 0

    rl.reset_episode()
    assert rl._first_detection_step is None
    assert rl._ewma_value is None
