import torch
import numpy as np
from typing import Dict, Any, Optional, List, Sequence

from .core import DeepDriftMonitor
from .diagnostics import VelocityDiagnosis


class DeepDriftRL:
    def __init__(
        self,
        agent_model,
        threshold: Optional[float] = None,
        n_channels: int = 32,
        use_ewma: bool = True,
        ewma_alpha: float = 0.3,
        z_threshold: Optional[float] = 2.0,
        eps: float = 1e-8,
    ):
        """
        Monitor for RL Agents with optional EWMA smoothing and z-score detection.
        """
        self.model = agent_model
        self.monitor = DeepDriftMonitor(
            agent_model,
            pooling="flatten",
            n_channels=n_channels,
        )
        if threshold is not None:
            self.monitor.threshold = threshold

        self.use_ewma = use_ewma
        self.ewma_alpha = ewma_alpha
        self.z_threshold = z_threshold
        self.eps = eps

        self._ewma_value: Optional[float] = None
        self._episode_step: int = 0
        self._first_detection_step: Optional[int] = None

        # Baseline stats
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._baseline_by_step: Dict[int, Dict[str, float]] = {}
        self._use_per_step_baseline: bool = False

    def reset_episode(self):
        """Reset per-episode state for EWMA / latency tracking."""
        self._ewma_value = None
        self._episode_step = 0
        self._first_detection_step = None
        self.monitor.reset_temporal()

    def _update_ewma(self, value: float) -> float:
        if (not self.use_ewma) or (self._ewma_value is None):
            self._ewma_value = value
            return value
        self._ewma_value = self.ewma_alpha * value + (1.0 - self.ewma_alpha) * self._ewma_value
        return self._ewma_value

    def calibrate_baseline(
        self,
        velocity_sequences: Sequence[Sequence[float]],
        per_step: bool = False,
    ) -> Dict[str, Any]:
        """
        Calibrate baseline from pre-collected velocity sequences.
        Each sequence is one episode/list of velocities.
        """
        if not velocity_sequences:
            raise ValueError("velocity_sequences must be non-empty")

        self._use_per_step_baseline = per_step
        self._baseline_by_step = {}

        if per_step:
            step_buckets: Dict[int, List[float]] = {}
            for seq in velocity_sequences:
                for t, v in enumerate(seq):
                    step_buckets.setdefault(t, []).append(float(v))

            for t, vals in step_buckets.items():
                arr = np.asarray(vals, dtype=float)
                self._baseline_by_step[t] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "count": int(arr.size),
                }

            return {
                "mode": "per_step",
                "n_steps": len(self._baseline_by_step),
                "total_points": int(sum(v["count"] for v in self._baseline_by_step.values())),
            }

        flat = np.asarray([float(v) for seq in velocity_sequences for v in seq], dtype=float)
        self._baseline_mean = float(flat.mean())
        self._baseline_std = float(flat.std(ddof=0))
        return {
            "mode": "global",
            "mean": self._baseline_mean,
            "std": self._baseline_std,
            "n_points": int(flat.size),
        }

    def _compute_z_score(self, value: float, step_idx: int) -> Optional[float]:
        if self._use_per_step_baseline:
            st = self._baseline_by_step.get(step_idx)
            if st is None:
                return None
            mu = st["mean"]
            sigma = st["std"]
            return float((value - mu) / (sigma + self.eps))

        if self._baseline_mean is None or self._baseline_std is None:
            return None
        return float((value - self._baseline_mean) / (self._baseline_std + self.eps))

    def step(
        self,
        obs,
        step_idx: Optional[int] = None,
        action: Optional[int] = None,
    ) -> VelocityDiagnosis:
        """
        Call this during the agent's step.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float().unsqueeze(0)

        _ = self.model(obs)

        raw_vel = float(self.monitor.get_temporal_velocity(step=step_idx))
        smoothed_vel = float(self._update_ewma(raw_vel))

        idx = self._episode_step if step_idx is None else int(step_idx)
        z_score = self._compute_z_score(smoothed_vel, idx)

        is_anomaly = False
        if self.monitor.threshold is not None:
            is_anomaly = is_anomaly or (smoothed_vel > float(self.monitor.threshold))
        if (self.z_threshold is not None) and (z_score is not None):
            is_anomaly = is_anomaly or (z_score > float(self.z_threshold))

        if is_anomaly and self._first_detection_step is None:
            self._first_detection_step = idx

        detection_latency = None
        if self._first_detection_step is not None:
            detection_latency = self._first_detection_step

        diagnosis = VelocityDiagnosis(
            peak_velocity=smoothed_vel,
            layer_velocities=[smoothed_vel],
            is_anomaly=is_anomaly,
            threshold=self.monitor.threshold,
            status="CRITICAL" if is_anomaly else "NORMAL",
            raw_velocity=raw_vel,
            smoothed_velocity=smoothed_vel,
            z_score=z_score,
            detected_at_step=self._first_detection_step,
            detection_latency=detection_latency,
            episode_step=idx,
            metadata={
                "use_ewma": self.use_ewma,
                "ewma_alpha": self.ewma_alpha,
                "z_threshold": self.z_threshold,
                "action": action,
            },
        )

        if step_idx is None:
            self._episode_step += 1
        return diagnosis
