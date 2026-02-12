#!/usr/bin/env python3
"""
DeepDrift RL CartPole Experiment (Gymnasium API)
Supports --quick mode for smoke testing.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

from deepdrift import DeepDriftMonitor

def run_quick():
    """Smoke test: synthetic MLP, no environment."""
    print("[!] Quick mode: Using synthetic MLP.")
    
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    monitor = DeepDriftMonitor(
        model,
        layer_names=None,
        pooling='flatten',
        debug=False
    )
    
    x = torch.randn(16, 4)
    _ = model(x)
    
    vel = monitor.get_spatial_velocity()
    print(f"  Spatial velocity: {[round(v, 4) for v in vel]}")
    
    v1 = monitor.get_temporal_velocity()
    _ = model(torch.randn(16, 4))
    v2 = monitor.get_temporal_velocity()
    print(f"  Temporal velocity: first={v1:.4f}, second={v2:.4f}")
    
    print("[✓] Quick test passed")
    return 0

def run_full(args):
    """Full experiment (CartPole training + evaluation)."""
    if not RL_AVAILABLE:
        print("[!] Install gymnasium and stable-baselines3 for full mode")
        return 1
    
    print(f"[*] Running CartPole experiment with {args.algo.upper()}, timesteps={args.timesteps}")
    
    # Environment
    env = gym.make("CartPole-v1", render_mode=None)
    env.reset(seed=args.seed)
    
    # Agent
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=0, seed=args.seed)
    else:
        model = DQN("MlpPolicy", env, verbose=0, seed=args.seed)
    
    model.learn(total_timesteps=args.timesteps)
    
    # Monitor
    monitor = DeepDriftMonitor(
        model.policy,
        layer_names=None,
        pooling='flatten'
    )
    
    # Evaluate
    velocities = []
    rewards = []
    
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_vel = []
        ep_rew = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs).float().unsqueeze(0)
            _ = model.policy(obs_t)
            vel = monitor.get_temporal_velocity()
            ep_vel.append(vel)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew += reward
        
        velocities.append(np.mean(ep_vel))
        rewards.append(ep_rew)
    
    # Summary
    success_rate = np.mean([1 if r >= 195 else 0 for r in rewards]) * 100
    print(f"[*] Success rate: {success_rate:.1f}%")
    print(f"[*] Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"[*] Mean velocity: {np.mean(velocities):.4f} ± {np.std(velocities):.4f}")
    
    if len(velocities) > 1:
        corr = np.corrcoef(velocities, rewards)[0,1]
        print(f"[*] Correlation reward-velocity: {corr:.3f}")
    
    env.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test")
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.quick:
        exit(run_quick())
    else:
        exit(run_full(args))
