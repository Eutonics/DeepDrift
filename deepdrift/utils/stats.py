import numpy as np
import torch
from typing import Dict, Tuple

def compute_iqr_threshold(velocities: np.ndarray, factor: float = 1.5) -> float:
    """Compute threshold using Q75 + factor * IQR"""
    if len(velocities) == 0:
        return 0.0
    q25, q75 = np.percentile(velocities, [25, 75])
    iqr = q75 - q25
    return q75 + factor * iqr

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean"""
    means = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(resample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

def youden_j(tps: np.ndarray, fps: np.ndarray, tns: np.ndarray, fns: np.ndarray):
    """Compute Youden's J statistic for optimal threshold search"""
    sensitivity = tps / (tps + fns)
    specificity = tns / (tns + fps)
    return sensitivity + specificity - 1
