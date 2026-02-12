import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

# Import DeepDrift
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepdrift import DeepDriftMonitor, DeepDriftVision

def run_experiment(quick=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running ViT OOD Experiment. Device: {device}, Quick mode: {quick}")

    # 1. Load Model
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()

    # 2. Setup Monitor
    monitor = DeepDriftVision(model, n_channels=50, pooling='cls')

    if quick:
        print("[!] Quick mode: Using synthetic data.")
        # Synthetic "Normal" data for calibration
        calib_data = [torch.randn(8, 3, 224, 224).to(device) for _ in range(5)]
        # Synthetic "OOD" data
        test_data = torch.randn(1, 3, 224, 224).to(device)
    else:
        print("[!] Full mode: Loading CIFAR-100 (Normal) and SVHN (OOD).")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Calibration data (CIFAR-100)
        calib_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        calib_subset = Subset(calib_set, range(100))
        calib_data = DataLoader(calib_subset, batch_size=16)

        # OOD data (SVHN)
        ood_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        ood_loader = DataLoader(Subset(ood_set, range(20)), batch_size=1)
        test_data_iter = iter(ood_loader)
        test_data, _ = next(test_data_iter)
        test_data = test_data.to(device)

    # 3. Calibration
    print("[*] Calibrating DeepDrift...")
    monitor.fit(calib_data, device=device)

    # 4. Detection
    print("[*] Testing on OOD sample...")
    diagnosis = monitor.predict(test_data)
    print(diagnosis)

    if diagnosis.is_anomaly:
        print("✅ SUCCESS: OOD Anomaly Detected!")
    else:
        print("❌ FAILURE: OOD Anomaly Not Detected (Check threshold/data)")

    # 5. Visualization (Simplified)
    plt.figure(figsize=(10, 5))
    plt.plot(diagnosis.layer_velocities, marker='o')
    plt.axhline(y=diagnosis.threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Semantic Velocity Profile (ViT)")
    plt.xlabel("Layer Index")
    plt.ylabel("Velocity")
    plt.legend()
    os.makedirs('experiments/plotting', exist_ok=True)
    plt.savefig('experiments/plotting/vit_ood_profile.png')
    print("[*] Plot saved to experiments/plotting/vit_ood_profile.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    run_experiment(quick=args.quick)
