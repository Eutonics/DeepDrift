"""
Statistical utilities for DeepDrift: IQR thresholding, bootstrap CI.
"""

import numpy as np
from typing import Tuple

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
    return thresholds[idx]
