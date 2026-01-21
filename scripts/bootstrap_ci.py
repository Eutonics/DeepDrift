#!/usr/bin/env python3
"""
Bootstrap Confidence Interval Computation for DeepDrift/ODD

This script provides functions for computing bootstrap confidence intervals
for AUC, Cohen's d, and other metrics used in the ODD framework validation.
It implements the statistical protocol described in the "Confidently Wrong" paper.

Usage:
    python bootstrap_ci.py --data_file episode_data.csv --metric full --n_boot 1000

Author: Alexey Evtushenko (alexey@eutonics.ru)
Repository: https://github.com/Eutonics/DeepDrift
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import argparse
import json
from pathlib import Path

# Requires scikit-learn and scipy
try:
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    from scipy import stats
except ImportError:
    raise ImportError("Please install scikit-learn and scipy: pip install scikit-learn scipy")


def pick_threshold_youden(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Select optimal threshold using Youden's J statistic.
    J = Sensitivity + Specificity - 1 = TPR - FPR

    Args:
        scores: Prediction scores (e.g., semantic velocity)
        labels: Binary labels (0 = success, 1 = failure/crash)

    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)

    return thresholds[optimal_idx]


def bootstrap_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute AUC with bootstrap confidence intervals.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        n_boot: Number of bootstrap samples
        confidence_level: Confidence level for CI (default 0.95)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, (lower_ci, upper_ci))
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    aucs = []

    # Point estimate
    point_auc = roc_auc_score(y_true, y_scores)

    # Bootstrap resampling
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Ensure both classes are present in bootstrap sample to avoid errors
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc_boot = roc_auc_score(y_true[idx], y_scores[idx])
            aucs.append(auc_boot)
        except ValueError:
            continue

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))

    return point_auc, (lower, upper)


def bootstrap_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute Cohen's d effect size with bootstrap confidence intervals.

    Args:
        group1: First group values (e.g., successful episodes velocity)
        group2: Second group values (e.g., crashed episodes velocity)
        n_boot: Number of bootstrap samples
        confidence_level: Confidence level for CI
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, (lower_ci, upper_ci))
    """
    rng = np.random.default_rng(random_state)

    def compute_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
        """Compute Cohen's d with pooled standard deviation."""
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2: return 0.0
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0: return 0.0
        return (np.mean(g2) - np.mean(g1)) / pooled_std

    # Point estimate
    point_d = compute_cohens_d(group1, group2)

    # Bootstrap resampling
    d_values = []
    n1, n2 = len(group1), len(group2)

    for _ in range(n_boot):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        d_boot = compute_cohens_d(group1[idx1], group2[idx2])
        d_values.append(d_boot)

    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(d_values, 100 * alpha / 2)
    upper = np.percentile(d_values, 100 * (1 - alpha / 2))

    return point_d, (lower, upper)


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42
) -> float:
    """
    Perform permutation test for difference in means.
    Used to validate statistical significance without assuming normality.

    Returns:
        p-value (two-tailed)
    """
    rng = np.random.default_rng(random_state)

    observed_diff = np.abs(np.mean(group2) - np.mean(group1))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations


def welch_t_test(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances).

    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value


def run_full_rl_analysis(
    velocities: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    n_boot: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run full RL analysis with proper train/val/test splits as defined in the paper.
    
    Protocol:
    1. Split data into Train (thresholding), Val (tuning), Test (evaluation).
    2. Derive optimal threshold from Train set using Youden's J.
    3. Evaluate performance on Test set (AUC, Accuracy, F1).
    4. Compute statistical significance and effect size.

    Args:
        velocities: Episode mean velocities
        labels: Episode labels (0 = success, 1 = crash)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        n_boot: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Dictionary with all analysis results
    """
    # Split data
    test_ratio = 1 - train_ratio - val_ratio

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        velocities, labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels
    )

    # Second split: train vs val
    # Adjust val_size to be relative to trainval set
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_trainval
    )

    # Derive threshold on TRAIN set only
    threshold = pick_threshold_youden(X_train, y_train)

    # Compute AUC on TEST set with bootstrap CI
    test_auc, test_auc_ci = bootstrap_auc(y_test, X_test, n_boot, random_state=random_state)

    # Group by outcome for effect size analysis
    success_vel = X_test[y_test == 0]
    crash_vel = X_test[y_test == 1]

    # Cohen's d with CI
    cohens_d, cohens_d_ci = bootstrap_cohens_d(
        success_vel, crash_vel, n_boot, random_state=random_state
    )

    # Welch's t-test
    t_stat, p_value = welch_t_test(success_vel, crash_vel)

    # Permutation test
    perm_p = permutation_test(success_vel, crash_vel, random_state=random_state)

    # Classification metrics on test set
    y_pred = (X_test >= threshold).astype(int)

    results = {
        'split_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'threshold': float(threshold),
        'test_metrics': {
            'auc': float(test_auc),
            'auc_ci_95': [float(test_auc_ci[0]), float(test_auc_ci[1])],
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0))
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'cohens_d_ci_95': [float(cohens_d_ci[0]), float(cohens_d_ci[1])]
        },
        'statistical_tests': {
            'welch_t': float(t_stat),
            'welch_p': float(p_value),
            'permutation_p': float(perm_p)
        },
        'group_stats': {
            'success_mean': float(np.mean(success_vel)) if len(success_vel) > 0 else 0.0,
            'success_std': float(np.std(success_vel)) if len(success_vel) > 0 else 0.0,
            'crash_mean': float(np.mean(crash_vel)) if len(crash_vel) > 0 else 0.0,
            'crash_std': float(np.std(crash_vel)) if len(crash_vel) > 0 else 0.0
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap CI computation for DeepDrift/ODD'
    )
    parser.add_argument(
        '--data_file', type=str, required=True,
        help='Path to data file (CSV with "velocity" and "label" columns)'
    )
    parser.add_argument(
        '--metric', type=str, default='full',
        choices=['auc', 'cohens_d', 'full'],
        help='Metric to compute or "full" for complete analysis'
    )
    parser.add_argument(
        '--n_boot', type=int, default=1000,
        help='Number of bootstrap samples'
    )
    parser.add_argument(
        '--output', type=str, default='bootstrap_results.json',
        help='Output file path'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    import pandas as pd

    try:
        df = pd.read_csv(args.data_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if 'velocity' not in df.columns or 'label' not in df.columns:
        print("Error: CSV must contain 'velocity' and 'label' columns.")
        return

    velocities = df['velocity'].values
    labels = df['label'].values

    if args.metric == 'full':
        results = run_full_rl_analysis(
            velocities, labels,
            n_boot=args.n_boot,
            random_state=args.seed
        )
    elif args.metric == 'auc':
        auc, ci = bootstrap_auc(labels, velocities, args.n_boot, random_state=args.seed)
        results = {'auc': auc, 'ci_95': list(ci)}
    elif args.metric == 'cohens_d':
        success = velocities[labels == 0]
        crash = velocities[labels == 1]
        d, ci = bootstrap_cohens_d(success, crash, args.n_boot, random_state=args.seed)
        results = {'cohens_d': d, 'ci_95': list(ci)}

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Analysis complete. Results saved to: {output_path}")
    # Print a brief summary for immediate verification
    if 'test_metrics' in results:
        print(f"Test AUC: {results['test_metrics']['auc']:.3f}")
    if 'effect_size' in results:
        print(f"Cohen's d: {results['effect_size']['cohens_d']:.3f}")


if __name__ == '__main__':
    main()

"""
Example Usage:
--------------
1. Create sample data file (episode_data.csv):
   velocity,label
   3.2,0
   3.5,0
   9.1,1
   10.2,1
   ...

2. Run full analysis:
   python bootstrap_ci.py --data_file episode_data.csv --metric full --n_boot 1000

3. From Python:
   >>> from bootstrap_ci import bootstrap_auc
   >>> velocities = np.array([3.2, 3.5, 9.1, 10.2])
   >>> labels = np.array([0, 0, 1, 1])
   >>> auc, ci = bootstrap_auc(labels, velocities)
   >>> print(f"AUC: {auc:.3f} [95% CI: {ci[0]:.3f}, {ci[1]:.3f}]")
"""
