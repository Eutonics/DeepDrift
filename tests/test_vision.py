"""
Unit tests for DeepDriftVision wrapper.
"""

import pytest
import torch
import torch.nn as nn
from deepdrift.vision import DeepDriftVision

class MockVision(nn.Module):
    """Mock ViT-like model for testing."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(768, 768) for _ in range(12)
        ])
        self.head = nn.Linear(768, 100)
    
    def forward(self, x):
        # Simulate ViT forward: assume x is already [B, N, D]
        for layer in self.encoder:
            x = layer(x)
        x = x[:, 0, :]  # CLS token
        x = self.head(x)
        return x

def test_vision_predict():
    """Test that DeepDriftVision.predict runs without errors."""
    model = MockVision()
    monitor = DeepDriftVision(model, auto_hook=True, pooling='cls')
    
    # Create dummy input: [B, Seq, Dim]
    x = torch.randn(2, 197, 768)
    
    # Should not raise
    diagnosis = monitor.predict(x)
    
    assert hasattr(diagnosis, 'peak_velocity')
    assert hasattr(diagnosis, 'layer_velocities')
    assert isinstance(diagnosis.peak_velocity, float)
    assert isinstance(diagnosis.layer_velocities, list)
