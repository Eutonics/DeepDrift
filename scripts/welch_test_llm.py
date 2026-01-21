#!/usr/bin/env python3
"""
Statistical Validation for LLM Experiments in DeepDrift/ODD

This script performs statistical tests (Mann-Whitney U, Welch's t-test)
and effect size estimation (Cohen's d, Cliff's delta) to validate the
difference between Factual and Hallucinated generation velocities.

Usage:
    python welch_test_llm.py --factual_file facts.csv --hallucination_file hallucinations.csv

Author: Alexey Evtushenko (alexey@eutonics.ru)
Repository: https://github.com/Eutonics/DeepDrift
"""

import numpy as np
import argparse
import json
import sys
from pathlib import Path

try:
    from scipy import stats
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size (non-parametric).
    Values range from -1 to 1.
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0: return 0.0
    
    # Efficient calculation using broadcasting
    # Matrix of relations: 1 if g1 > g2, -1 if g1 < g2, 0 otherwise
    g1_matrix = group1[:, np.newaxis]
    g2_matrix = group2[np.newaxis, :]
    
    greater = np.sum(g1_matrix > g2_matrix)
    less = np.sum(g1_matrix < g2_matrix)
    
    return (greater - less) / (n1 * n2)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (parametric effect size)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0.0
    
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0: return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def benjamini_hochberg_correction(p_values: list, alpha: float = 0.05):
    """
    Apply Benjamini-Hochberg procedure for multiple testing correction.
    Returns list of booleans (True = significant).
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # BH critical values
    critical_values = (np.arange(1, n + 1) / n) * alpha
    
    # Find largest index where p < critical
    below_threshold = sorted_p <= critical_values
    if not np.any(below_threshold):
        max_idx = -1
    else:
        max_idx = np.max(np.where(below_threshold)[0])
        
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices[:max_idx+1]] = True
    return significant.tolist()


def run_analysis(factual_vel: np.ndarray, hall_vel: np.ndarray):
    """Run full statistical suite."""
    
    # 1. Normality Check (Shapiro-Wilk)
    # If N > 5000, Shapiro is too sensitive, but we check anyway
    _, p_norm_fact = stats.shapiro(factual_vel) if len(factual_vel) < 5000 else (0, 0.0)
    _, p_norm_hall = stats.shapiro(hall_vel) if len(hall_vel) < 5000 else (0, 0.0)
    
    # 2. Mann-Whitney U Test (Non-parametric)
    u_stat, mw_p = stats.mannwhitneyu(hall_vel, factual_vel, alternative='greater')
    
    # 3. Welch's t-test (Parametric, unequal variance)
    t_stat, t_p = stats.ttest_ind(hall_vel, factual_vel, equal_var=False, alternative='greater')
    
    # 4. Effect Sizes
    d_val = cohens_d(hall_vel, factual_vel) # Hallucination > Fact expected
    delta_val = cliffs_delta(hall_vel, factual_vel)
    
    return {
        "samples": {"factual": len(factual_vel), "hallucination": len(hall_vel)},
        "means": {"factual": float(np.mean(factual_vel)), "hallucination": float(np.mean(hall_vel))},
        "mann_whitney": {"u_stat": float(u_stat), "p_value": float(mw_p)},
        "welch_t": {"t_stat": float(t_stat), "p_value": float(t_p)},
        "effect_sizes": {"cohens_d": float(d_val), "cliffs_delta": float(delta_val)}
    }


def main():
    parser = argparse.ArgumentParser(description='LLM Velocity Statistical Validation')
    parser.add_argument('--factual_file', type=str, required=True, help='CSV with factual velocities')
    parser.add_argument('--hallucination_file', type=str, required=True, help='CSV with hallucination velocities')
    parser.add_argument('--column', type=str, default='velocity', help='Column name for velocity values')
    parser.add_argument('--output', type=str, default='llm_stats.json')
    args = parser.parse_args()

    try:
        df_fact = pd.read_csv(args.factual_file)
        df_hall = pd.read_csv(args.hallucination_file)
        
        # Extract numpy arrays
        # Handle cases where CSV might not have headers or different col names
        vel_f = df_fact[args.column].values if args.column in df_fact.columns else df_fact.iloc[:, 0].values
        vel_h = df_hall[args.column].values if args.column in df_hall.columns else df_hall.iloc[:, 0].values
        
        results = run_analysis(vel_f, vel_h)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"✅ LLM Analysis complete. Results saved to {args.output}")
        print(f"   Mann-Whitney P-Value: {results['mann_whitney']['p_value']:.4e}")
        print(f"   Cohen's d: {results['effect_sizes']['cohens_d']:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
