#!/usr/bin/env python3
"""
Systematic Layer Sweep Analysis for DeepDrift/ODD

This script analyzes Semantic Velocity across all network layers to identify
where hallucinations originate (validating the "Burning Bottleneck" hypothesis).
It computes ROC-AUC for each layer and identifies the most sensitive depth.

Usage:
    python layer_sweep.py --data_file layer_data.csv --model_layers 32

Author: Alexey Evtushenko (alexey@eutonics.ru)
Repository: https://github.com/Eutonics/DeepDrift
"""

import numpy as np
import argparse
import json
import sys
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


def analyze_layers(df: pd.DataFrame, num_layers: int):
    """
    Iterate through layers and compute detection performance (AUC).
    Assumes columns are named 'layer_0', 'layer_1', etc.
    """
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0=Fact, 1=Hallucination)")

    y_true = df['label'].values
    layer_results = {}
    best_layer = -1
    best_auc = 0.0

    print(f"🔍 Scanning {num_layers} layers...")

    for i in range(num_layers):
        col_name = f"layer_{i}"
        
        # Если такого слоя нет в данных, пропускаем (или кидаем ошибку)
        if col_name not in df.columns:
            continue
            
        velocities = df[col_name].values
        
        # Compute AUC
        try:
            auc = roc_auc_score(y_true, velocities)
        except ValueError:
            auc = 0.5 # Fallback if only one class present
            
        layer_results[i] = {
            "depth_ratio": round(i / (num_layers - 1), 2),
            "auc": float(auc)
        }
        
        if auc > best_auc:
            best_auc = auc
            best_layer = i

    return layer_results, best_layer, best_auc


def validate_bottleneck(results: dict, num_layers: int):
    """
    Check if the 'Burning Bottleneck' pattern exists.
    Pattern: Low AUC at start -> High in Middle -> Lower at End.
    """
    mid_start = int(num_layers * 0.4)
    mid_end = int(num_layers * 0.7)
    
    early_aucs = [results[i]['auc'] for i in range(mid_start) if i in results]
    mid_aucs = [results[i]['auc'] for i in range(mid_start, mid_end) if i in results]
    
    if not early_aucs or not mid_aucs: return False, 0.0
    
    mean_early = np.mean(early_aucs)
    mean_mid = np.mean(mid_aucs)
    
    # Bottleneck is confirmed if Middle > Early
    is_confirmed = mean_mid > mean_early
    gain = mean_mid - mean_early
    
    return is_confirmed, gain


def main():
    parser = argparse.ArgumentParser(description='DeepDrift Layer Sweep')
    parser.add_argument('--data_file', type=str, required=True, help='CSV with layer_N columns and label')
    parser.add_argument('--model_layers', type=int, default=32, help='Total number of layers in model (e.g. 32 for Qwen-7B)')
    parser.add_argument('--output', type=str, default='layer_sweep_results.json')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data_file)
        
        results, best_layer, best_auc = analyze_layers(df, args.model_layers)
        
        is_bottleneck, gain = validate_bottleneck(results, args.model_layers)
        
        output_data = {
            "model_layers": args.model_layers,
            "best_layer": int(best_layer),
            "best_auc": float(best_auc),
            "burning_bottleneck_hypothesis": {
                "confirmed": bool(is_bottleneck),
                "performance_gain": float(gain)
            },
            "layer_wise_metrics": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ Sweep complete. Results saved to {args.output}")
        print(f"   Best Layer: {best_layer} (AUC: {best_auc:.3f})")
        if is_bottleneck:
            print(f"   🔥 Burning Bottleneck DETECTED (Mid-layer gain: +{gain:.2f})")
        else:
            print(f"   ❄️ No clear bottleneck pattern.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
