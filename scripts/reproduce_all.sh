#!/bin/bash
# DeepDrift v1.1.4 Reproduction Script
# -----------------------------------
# This script runs all experiments in --quick mode to verify installation.

set -e

echo "🚀 Starting DeepDrift Reproduction (Smoke Test Mode)..."

# 1. Install dependencies
echo "[*] Checking/Installing dependencies..."
pip install -r requirements.txt
pip install -e .

# 2. Run Experiments
echo "[*] 1/4: Vision OOD (ViT)..."
python experiments/vit_svhn_ood.py --quick

echo "[*] 2/4: RL Crash Prediction (LunarLander)..."
python experiments/rl_lunarlander.py --quick

echo "[*] 3/4: LLM Hallucination Detection..."
python experiments/llm_hallucination.py --quick

echo "[*] 4/4: Diffusion Memorization..."
python experiments/diffusion_memorization.py --quick

echo ""
echo "✅ All tests passed! DeepDrift is correctly installed and functional."
echo "To run full experiments, remove the --quick flag in individual scripts."
echo "Note: Full experiments require datasets (CIFAR, SVHN) and/or SB3/Gymnasium."
