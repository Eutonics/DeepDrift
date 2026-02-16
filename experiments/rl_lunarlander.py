#!/usr/bin/env python3
"""
DeepDrift RL LunarLander experiment (clean script version).

Key goals:
- No notebook magics (no `%matplotlib inline`)
- Reproducible calibration/evaluation
- EWMA + z-score early-warning analysis
"""

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

from deepdrift import DeepDriftMonitor
from deepdrift.utils.stats import (
    compute_iqr_threshold,
    ewma_update,
    safe_zscore,
    compute_global_baseline,
)


def run_episode(
    env,
    model,
    monitor: DeepDriftMonitor,
    max_steps: int = 300,
    seed: int = 42,
    noise_std: float = 0.0,
    noise_start: int = 50,
) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    monitor.reset_temporal()

    velocities_raw: List[float] = []
    x_positions: List[float] = []
    actions: List[int] = []

    for t in range(max_steps):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)
            _ = model.policy.extract_features(t_obs)

        v = monitor.get_temporal_velocity(step=t)
        velocities_raw.append(float(v))

        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))

        if t >= noise_start and noise_std > 0:
            obs_step, reward, term, trunc, info = env.step(action)
            # simple observation perturbation for robustness stress-test
            obs = obs_step + np.random.normal(0, noise_std, size=np.asarray(obs_step).shape)
        else:
            obs, reward, term, trunc, info = env.step(action)

        # LunarLander obs[0] ~= x-position
        x_positions.append(float(obs[0]))

        if term or trunc:
            break

    return {
        "velocities_raw": velocities_raw,
        "x_positions": x_positions,
        "actions": actions,
        "n_steps": len(velocities_raw),
    }


def detect_first_step_over_threshold(values: List[float], threshold: float, start_step: int = 0):
    for t in range(start_step, len(values)):
        if values[t] > threshold:
            return t
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=30000)
    parser.add_argument("--calib_episodes", type=int, default=8)
    parser.add_argument("--eval_episodes", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_std", type=float, default=0.25)
    parser.add_argument("--noise_start", type=int, default=50)
    parser.add_argument("--ewma_alpha", type=float, default=0.3)
    parser.add_argument("--z_threshold", type=float, default=2.0)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 70)
    print("DeepDrift RL LunarLander (clean script)")
    print("=" * 70)

    print("[1] Training PPO...")
    env = gym.make("LunarLander-v3")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.train_steps, progress_bar=False)

    print("[2] Creating monitor...")
    monitor = DeepDriftMonitor(model.policy, pooling="flatten")

    print("[3] Calibration on clean runs...")
    calibration_sequences = []
    for ep in range(args.calib_episodes):
        r = run_episode(
            env, model, monitor,
            max_steps=args.max_steps,
            seed=args.seed + 100 + ep,
            noise_std=0.0,
            noise_start=args.noise_start,
        )
        # skip first 2 points to avoid temporal init bias
        calibration_sequences.append(r["velocities_raw"][2:])

    flat_calib = [v for seq in calibration_sequences for v in seq]
    baseline = compute_global_baseline(flat_calib)
    iqr_threshold = compute_iqr_threshold(np.asarray(flat_calib))
    print(f"  baseline mean={baseline['mean']:.6f}, std={baseline['std']:.6f}, iqr_thr={iqr_threshold:.6f}")

    print("[4] Evaluation with noise...")
    eval_rows = []
    lead_times = []
    detected_count = 0

    for ep in range(args.eval_episodes):
        rr = run_episode(
            env, model, monitor,
            max_steps=args.max_steps,
            seed=args.seed + 1000 + ep,
            noise_std=args.noise_std,
            noise_start=args.noise_start,
        )
        raw = rr["velocities_raw"]

        # EWMA transform
        smoothed = []
        ew = None
        for v in raw:
            ew = ewma_update(ew, v, alpha=args.ewma_alpha)
            smoothed.append(ew)

        # z-score against calibration baseline
        z = [safe_zscore(v, baseline["mean"], baseline["std"]) for v in smoothed]

        t_detect = detect_first_step_over_threshold(z, args.z_threshold, start_step=args.noise_start)
        t_traj = detect_first_step_over_threshold(
            [abs(x) for x in rr["x_positions"]],
            threshold=np.percentile(np.abs(rr["x_positions"]), 90),
            start_step=args.noise_start,
        )

        lead = None
        if t_detect is not None and t_traj is not None:
            lead = t_traj - t_detect
            lead_times.append(lead)

        if t_detect is not None:
            detected_count += 1

        eval_rows.append({
            "episode": ep,
            "n_steps": rr["n_steps"],
            "t_detect": t_detect,
            "t_traj": t_traj,
            "lead_time": lead,
            "max_raw_velocity": float(np.max(raw)) if raw else None,
            "max_smoothed_velocity": float(np.max(smoothed)) if smoothed else None,
            "max_z": float(np.max(z)) if z else None,
        })

    print("[5] Saving results...")
    summary = {
        "config": vars(args),
        "calibration": {
            "mean": baseline["mean"],
            "std": baseline["std"],
            "iqr_threshold": iqr_threshold,
            "n_points": baseline["count"],
        },
        "evaluation": {
            "episodes": args.eval_episodes,
            "detected_count": detected_count,
            "detection_rate": detected_count / max(args.eval_episodes, 1),
            "lead_time_mean": float(np.mean(lead_times)) if lead_times else None,
            "lead_time_std": float(np.std(lead_times)) if lead_times else None,
            "rows": eval_rows,
        },
    }

    out_json = os.path.join(args.out_dir, "rl_lunarlander_clean_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {out_json}")

    print("[6] Plot detection timeline...")
    plt.figure(figsize=(10, 4))
    det_vals = [r["t_detect"] if r["t_detect"] is not None else np.nan for r in eval_rows]
    traj_vals = [r["t_traj"] if r["t_traj"] is not None else np.nan for r in eval_rows]
    x = np.arange(len(eval_rows))
    plt.plot(x, det_vals, "ro-", label="ODD detect (z-threshold)")
    plt.plot(x, traj_vals, "bs-", label="Trajectory deviation")
    plt.axhline(args.noise_start, color="gray", linestyle="--", alpha=0.7, label="noise_start")
    plt.xlabel("Episode")
    plt.ylabel("Step")
    plt.title("Detection timing per episode")
    plt.grid(alpha=0.3)
    plt.legend()
    fig_path = "figures/rl_lunarlander_clean_timeline.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  saved: {fig_path}")

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
