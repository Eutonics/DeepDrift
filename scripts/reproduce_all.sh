#!/bin/bash
# DeepDrift v1.0.0 Global Reproduction Script
set -e

echo "===================================================="
echo "  DeepDrift: Starting Global Reproduction"
echo "===================================================="

# 1. Инсталляция в режиме редактирования
pip install -e .

# 2. Vision Experiment
echo -e "\n[*] Running Vision SOTA Benchmark (ViT)..."
python3 experiments/vit_svhn_ood.py

# 3. RL Experiment
echo -e "\n[*] Running RL Lead Time Benchmark (LunarLander)..."
# Мы используем v5.2 который у тебя уже в scripts
python3 experiments/rl_lunarlander.py

echo -e "\n===================================================="
echo "✅ ALL TESTS PASSED"
echo "===================================================="
