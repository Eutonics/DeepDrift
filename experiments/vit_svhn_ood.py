#!/usr/bin/env python3
"""
DeepDrift ViT OOD Experiment (CIFAR-100 vs SVHN)
Supports --quick mode for smoke testing.
"""

import argparse
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from deepdrift import DeepDriftMonitor
from deepdrift.vision import DeepDriftVision

def run_quick():
    """Smoke test with synthetic data."""
    print("[!] Quick mode: Using synthetic data.")
    
    # Fake ViT-like model
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 768),
        torch.nn.ReLU(),
        torch.nn.Linear(768, 100)
    )
    
    # Create monitor
    monitor = DeepDriftVision(model, auto_hook=True, pooling='cls')
    
    # Fake calibration
    calib_data = torch.randn(10, 197, 768)
    for x in calib_data:
        monitor.predict(x.unsqueeze(0))
    
    # Test on fake OOD
    x_test = torch.randn(1, 197, 768)
    diagnosis = monitor.predict(x_test)
    
    print(f"  Vision Diagnosis: {diagnosis.status} | Peak Velocity: {diagnosis.peak_velocity:.4f}")
    if diagnosis.threshold is not None:
        print(f"  Threshold: {diagnosis.threshold:.4f}")
    else:
        print("  Threshold: None (quick mode)")
    
    print("[✓] Quick test passed")
    return 0

def run_full(args):
    """Full experiment (real data)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running ViT OOD Experiment. Device: {device}, Quick mode: False")

    # Data
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ID: CIFAR-100
    id_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    id_loader = torch.utils.data.DataLoader(id_set, batch_size=32, shuffle=False)

    # OOD: SVHN
    ood_set = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform
    )
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=32, shuffle=False)

    # Model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 100)
    model = model.to(device).eval()

    # Monitor
    monitor = DeepDriftVision(model, auto_hook=True, pooling='cls', n_channels=768)
    
    # Calibration (first 500 ID samples)
    calib_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(id_set, list(range(500))),
        batch_size=32
    )
    monitor.fit(calib_loader, device=device)
    
    # Evaluate on ID and OOD
    scores_id = []
    scores_ood = []
    
    with torch.no_grad():
        for images, _ in id_loader:
            images = images.to(device)
            diag = monitor.predict(images)
            scores_id.append(diag.peak_velocity)
        
        for images, _ in ood_loader:
            images = images.to(device)
            diag = monitor.predict(images)
            scores_ood.append(diag.peak_velocity)
    
    # Metrics
    y_true = [0] * len(scores_id) + [1] * len(scores_ood)
    y_score = scores_id + scores_ood
    auc = roc_auc_score(y_true, y_score)
    print(f"[*] AUROC: {auc:.4f}")
    
    # Plot ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'DeepDrift (AUC = {auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ViT OOD Detection (CIFAR-100 vs SVHN)')
    
    # ** ИСПРАВЛЕНИЕ: Проверяем, есть ли порог **
    if monitor.monitor.threshold is not None:
        plt.axhline(y=monitor.monitor.threshold, color='r', linestyle='--', 
                    label=f'Threshold ({monitor.monitor.threshold:.2f})')
    else:
        plt.text(0.5, 0.95, 'No threshold (calibration required)', 
                 transform=plt.gca().transAxes, ha='center', fontsize=10)
    
    plt.legend()
    plt.savefig('vit_ood_roc.png', dpi=150)
    print("[✓] ROC curve saved to vit_ood_roc.png")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    
    if args.quick:
        exit(run_quick())
    else:
        exit(run_full(args))
