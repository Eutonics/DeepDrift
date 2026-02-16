"""
Statistical utilities for DeepDrift.
Includes:
- IQR thresholding
- Bootstrap confidence interval
- Youden's J threshold
- EWMA update helper
- Safe z-score
- Baseline stats (global / per-step)
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def compute_iqr_threshold(data: np.ndarray, factor: float = 1.5) -> float:
    """
    Compute upper threshold using Tukey's fences.

    Args:
        data: 1D array of calibration velocities
        factor: IQR multiplier (default 1.5)

    Returns:
        Upper threshold = Q75 + factor * IQR
    """
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    iqr = q75 - q25
    return float(q75 + factor * iqr)


def bootstrap_ci(data: np.ndarray, n_resamples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data: 1D array of values
        n_resamples: Number of bootstrap resamples
        ci: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    means = []
    n = len(data)
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def youden_j(tpr: np.ndarray, fpr: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Find optimal threshold using Youden's J statistic.

    Args:
        tpr: True positive rates
        fpr: False positive rates
        thresholds: Corresponding thresholds

    Returns:
        Optimal threshold
    """
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


def ewma_update(previous: Optional[float], value: float, alpha: float = 0.3) -> float:
    """
    Single-step EWMA update.

    Args:
        previous: previous EWMA value, or None for first step
        value: current raw value
        alpha: smoothing factor in [0,1]

    Returns:
        updated EWMA value
    """
    if previous is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * previous)


def safe_zscore(value: float, mean: float, std: float, eps: float = 1e-8) -> float:
    """
    Numerically stable z-score.
    """
    return float((value - mean) / (std + eps))


def compute_global_baseline(values: Sequence[float]) -> Dict[str, float]:
    """
    Compute global baseline statistics for a 1D sequence.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "count": int(arr.size),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def compute_step_baseline(sequences: Sequence[Sequence[float]]) -> Dict[int, Dict[str, float]]:
    """
    Compute per-step baseline stats from episode-wise sequences.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty")

    buckets: Dict[int, List[float]] = {}
    for seq in sequences:
        for t, v in enumerate(seq):
            buckets.setdefault(t, []).append(float(v))

    out: Dict[int, Dict[str, float]] = {}
    for t, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[t] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "count": int(arr.size),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }
    return out
