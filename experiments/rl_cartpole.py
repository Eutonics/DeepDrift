#!/usr/bin/env python3
"""
DeepDrift RL CartPole Experiment

This script demonstrates Semantic Velocity for failure prediction in CartPole-v1.
Supports --quick smoke test mode for fast verification.
Uses Gymnasium (maintained fork of OpenAI Gym) with modern API.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

# For smoke test we don't need heavy dependencies
try:
    import gymnasium as gym  # Note: import gymnasium, not gym
    from stable_baselines3 import PPO, DQN
    RL_AVAILABLE = True
    print("✅ Gymnasium and Stable-Baselines3 loaded successfully")
except ImportError as e:
    RL_AVAILABLE = False
    print(f"⚠️ RL dependencies not available: {e}")
    print("   Install with: pip install gymnasium stable-baselines3")

from deepdrift import DeepDriftMonitor


def run_quick():
    """Smoke test: synthetic MLP + random data. No environment needed."""
    print("\n🧪 Running quick smoke test (no training, no environment)")
    print("   Testing DeepDriftMonitor on synthetic MLP...")
    
    # Simple MLP with 3 layers (to test spatial velocity)
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Initialize monitor with auto-detected layers
    monitor = DeepDriftMonitor(
        model,
        layer_names=None,      # auto-detect
        pooling='flatten',
        debug=False
    )
    
    # Dummy input (batch 16)
    x = torch.randn(16, 4)
    
    # Forward pass – hooks capture activations
    _ = model(x)
    
    # Get spatial velocity between consecutive layers
    vel = monitor.get_spatial_velocity()
    print(f"📊 Spatial velocity (across {len(vel)} layer transitions): {[round(v, 4) for v in vel]}")
    
    # Temporal velocity – two calls with different inputs
    v1 = monitor.get_temporal_velocity()
    _ = model(torch.randn(16, 4))
    v2 = monitor.get_temporal_velocity()
    print(f"⏱️ Temporal velocity: first={v1:.4f}, second={v2:.4f}")
    
    print("✅ Quick test passed – library is working correctly")
    return 0


def run_full(args):
    """
    Full experiment: train PPO/DQN on CartPole-v1, evaluate, plot.
    Uses modern Gymnasium API (terminated/truncated, reset(seed=...), etc.)
    """
    if not RL_AVAILABLE:
        print("❌ Gymnasium or Stable-Baselines3 not installed.")
        print("   Run: pip install gymnasium stable-baselines3")
        return 1
    
    print("\n🚀 Running full CartPole experiment with Gymnasium API")
    print("   This will train an agent and evaluate Semantic Velocity\n")
    
    # Create environment with modern API
    if args.render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1", render_mode=None)
    
    # Set seed for reproducibility (modern API: seed in reset, not env.seed())
    env.reset(seed=args.seed)
    
    # Initialize agent based on algorithm choice
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=0, seed=args.seed)
    elif args.algo == "dqn":
        model = DQN("MlpPolicy", env, verbose=0, seed=args.seed)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    print(f"🤖 Training {args.algo.upper()} for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    
    # Evaluate trained agent
    print("📊 Evaluating agent and measuring Semantic Velocity...")
    
    # We need to access the policy network's hidden layers
    # For Stable-Baselines3, the policy is model.policy
    policy = model.policy
    
    # Create monitor on the policy network
    # For MLP policies, we monitor the first hidden layer
    monitor = DeepDriftMonitor(
        policy,
        layer_names=None,  # auto-detect
        pooling='flatten',
        debug=False
    )
    
    velocities = []
    rewards = []
    
    for episode in range(args.n_episodes):
        # Modern reset: returns (obs, info)
        obs, info = env.reset(seed=args.seed + episode)
        episode_velocities = []
        episode_reward = 0
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Modern step: returns 5 values
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Forward pass through policy to capture activations
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            _ = policy(obs_tensor)
            
            # Get temporal velocity (difference between consecutive steps)
            vel = monitor.get_temporal_velocity()
            episode_velocities.append(vel)
            
            episode_reward += reward
            step_count += 1
        
        # Store episode-level metrics
        if episode_velocities:
            velocities.append(np.mean(episode_velocities))
        rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"   Episode {episode + 1}/{args.n_episodes} - Reward: {episode_reward:.1f}, Mean Velocity: {np.mean(episode_velocities):.4f}")
    
    # Calculate success rate (CartPole-v1 solved at 195+)
    successes = sum(1 for r in rewards if r >= 195)
    success_rate = successes / len(rewards) * 100
    
    print("\n📈 Results:")
    print(f"   Success rate: {success_rate:.1f}% ({successes}/{len(rewards)} episodes)")
    print(f"   Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Mean velocity: {np.mean(velocities):.4f} ± {np.std(velocities):.4f}")
    
    # Simple correlation analysis
    if len(velocities) > 1 and len(rewards) > 1:
        corr = np.corrcoef(velocities, rewards)[0, 1]
        print(f"   Correlation (reward vs velocity): {corr:.3f}")
    
    env.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDrift RL CartPole experiment with Gymnasium API")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test (no training, synthetic data)")
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo", help="RL algorithm (full mode)")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps (full mode)")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of evaluation episodes (full mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render environment (full mode)")
    
    args = parser.parse_args()
    
    if args.quick:
        exit(run_quick())
    else:
        exit(run_full(args))
