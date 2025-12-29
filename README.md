# DeepDrift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18086612.svg)](https://doi.org/10.5281/zenodo.18086612)

**A Layer-Wise Diagnostic Framework for Neural Network Robustness.**


> "Stop guessing why your model failed. See exactly where it broke."

DeepDrift is an unsupervised diagnostic tool that analyzes how data representations evolve layer-by-layer. Instead of monitoring only output accuracy, it provides structural insights into model failures.

![Dashboard](stress_test_dashboard.png)

## Key Features

- **Architecture-agnostic**: Works with CNNs, Vision Transformers, and hybrid models.
- **Unsupervised**: No labeled OOD data required for deployment.
- **Lightweight**: <1% inference overhead, implemented via forward hooks.
- **Interpretable signatures**: Identifies failure modes by analyzing drift patterns across network depth.
- **Real-time capable**: Suitable for production monitoring and safety-critical systems.

## Installation

```bash
git clone https://github.com/Eutonics/DeepDrift.git  
cd DeepDrift
pip install .
```

## Quick Start

```python
import torch
import torchvision.models as models
from deepdrift import DeepDriftMonitor

# Load your model
model = models.resnet18(pretrained=True)
model.eval()

# Initialize monitor
monitor = DeepDriftMonitor(model, arch_name='ResNet-18')

# Calibrate on clean data (train_loader)
# monitor.calibrate(train_loader)

# Measure drift on new data (ood_batch is a tensor [B, C, H, W])
# drift_profile = monitor.scan(ood_batch)  # Returns [UV, Mid, Deep, IR] drift scores

# Visualize results
from deepdrift.visualization import plot_drift_profile
plot_drift_profile(drift_profile, title="Drift Analysis")
```

## Technical Overview

DeepDrift treats network depth as a scale dimension ($z$), where $z=0$ corresponds to shallow layers (UV scale) and $z=1$ to deep layers (IR scale). For each layer, it calculates:

1. A baseline mean representation ($\mu$) and standard deviation ($\sigma$) from calibration data.
2. Drift score: $D(z) = ||\mu_{test} - \mu|| / \sigma$

This approach reveals characteristic failure patterns:
- **Mid-Layer Bulge**: Spurious correlations (e.g., color shortcuts in Colored MNIST).
- **Avalanche Effect**: Error accumulation in CNNs under geometric stress.
- **Global Collapse**: Immediate instability in ViTs due to positional encoding mismatch.
- **Shadow Signature**: Persistent drift through deep layers for adversarial examples.

## Usage Examples

### Architecture Comparison

```python
# Compare different architectures under rotation stress
# for arch in ['ResNet-18', 'ViT-B/16', 'ConvNeXt-T']:
#     model = load_model(arch)
#     monitor = DeepDriftMonitor(model, arch)
#     monitor.calibrate(clean_loader)
#     drift_profile = monitor.scan(rotation_loader(angle=30))
    # Analyze results to select most robust architecture
```

### Adversarial Detection

```python
# Distinguish adversarial examples from benign noise
# monitor.calibrate(clean_loader)
# adv_drift = monitor.scan(adversarial_loader)   # Shows persistent drift
# noise_drift = monitor.scan(noise_loader)       # Shows attenuation with depth
```

## Citation

If you use DeepDrift in your research, please cite:

```bibtex
@article{evtushenko2025deepdrift,
  title={DeepDrift: A Layer-Wise Diagnostic Framework for Neural Network Robustness},
  author={Alexey Evtushenko},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

---

*DeepDrift: Making neural networks transparent, one layer at a time.*
