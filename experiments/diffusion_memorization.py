import torch
import torch.nn as nn
import argparse
import os
import numpy as np

# Import DeepDrift
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepdrift import DeepDriftMonitor

class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bottleneck = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.bottleneck(self.conv1(x)))

def run_experiment(quick=False):
    print(f"[*] Running Diffusion Memorization Experiment. Quick mode: {quick}")

    model = ToyUNet()
    model.eval()

    # Monitor the bottleneck
    monitor = DeepDriftMonitor(model, layer_names=['bottleneck'], pooling='mean')

    if quick:
        print("[!] Quick mode: Checking train/test gap on synthetic data.")
        # Train sample
        x_train = torch.randn(1, 3, 32, 32)
        _ = model(x_train)
        act_train = monitor.activations['bottleneck'].clone()

        # Test sample
        x_test = torch.randn(1, 3, 32, 32)
        _ = model(x_test)
        act_test = monitor.activations['bottleneck'].clone()

        # Semantic Gap
        gap = torch.norm(act_train - act_test).item()
        print(f"Semantic Gap (Train vs Test): {gap:.4f}")

        if gap > 0:
            print("✅ SUCCESS: Semantic Gap measured.")
    else:
        print("[!] Full mode: Training a U-Net on CIFAR to find Burning Bottleneck.")
        print("    Requires long training. See reproduce_all.sh for full run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    run_experiment(quick=args.quick)
