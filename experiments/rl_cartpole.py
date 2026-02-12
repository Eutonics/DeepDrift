#!/usr/bin/env python3
"""
DeepDrift RL CartPole experiment (quick/smoke test).
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from deepdrift import DeepDriftMonitor

def run_quick():
    """Smoke test: synthetic MLP + random data."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )
    monitor = DeepDriftMonitor(model, layer_names=None, pooling='flatten')
    x = torch.randn(32, 4)
    _ = model(x)
    vel = monitor.get_spatial_velocity()
    print(f"[quick] Spatial velocity: {vel}")
    print("✅ Quick test passed")

def run_full():
    """Full experiment: train PPO/DQN on CartPole, evaluate, plot."""
    # (здесь был бы полный код, но для smoke-теста оставляем заглушку)
    print("Full experiment not implemented in smoke mode, use --quick for testing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    if args.quick:
        run_quick()
    else:
        run_full()
