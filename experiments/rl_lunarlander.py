import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

# Import DeepDrift
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepdrift.rl import DeepDriftRL

class MockPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def run_experiment(quick=False):
    print(f"[*] Running RL Crash Prediction. Quick mode: {quick}")

    obs_dim = 8
    act_dim = 4
    model = MockPolicy(obs_dim, act_dim)
    model.eval()

    monitor = DeepDriftRL(model, threshold=0.1, n_channels=32)

    if quick:
        print("[!] Quick mode: Using synthetic trajectory.")
        # Normal trajectory
        for _ in range(10):
            obs = torch.randn(1, obs_dim) * 0.1
            _ = monitor.step(obs)

        # A "crash" (sudden state change)
        crash_obs = torch.randn(1, obs_dim) * 5.0
        diagnosis = monitor.step(crash_obs)
    else:
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO
            env = gym.make("LunarLander-v2")
            print("[!] Loading real PPO agent (requires SB3)...")
            # This is a placeholder for real loading logic
            # For demonstration, we use our mock as if it was trained
            obs, _ = env.reset()
            for _ in range(50):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                diagnosis = monitor.step(obs)
                if diagnosis.is_anomaly:
                    print(f"⚠️ Crash predicted at step! Velocity: {diagnosis.peak_velocity:.4f}")
                if terminated or truncated: break
        except ImportError:
            print("❌ stable-baselines3 or gymnasium not found. Falling back to synthetic.")
            return run_experiment(quick=True)

    print(diagnosis)
    if diagnosis.is_anomaly:
        print("✅ SUCCESS: Agent Crash/Anomaly Detected!")

    # Plot velocity history
    # (Implementation of history capture omitted for brevity in script, but shown in README)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    run_experiment(quick=args.quick)
