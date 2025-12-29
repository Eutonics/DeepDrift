# 🌌 DeepDrift

**A Layer-Wise Diagnostic Framework for Neural Network Robustness.**

> "Stop guessing *why* your model failed. See exactly *where* it broke."

DeepDrift is an unsupervised diagnostic tool that acts like an **MRI scan for your neural network**. Instead of just monitoring output accuracy, DeepDrift analyzes how data representations evolve layer-by-layer.

It allows you to distinguish between:
*   📷 **Sensor Failure** (High drift at input layers)
*   🧠 **Geometric Collapse** (Drift accumulation in deep layers)
*   👻 **Spurious Correlations** (Anomalies in mid-level features)

![Dashboard](dashboard.png)

## 🚀 Key Features

*   **Unsupervised:** No labeled OOD data required.
*   **Lightweight:** < 1% inference overhead.
*   **Interpretability:** Maps drift to network depth ($z$-axis).
*   **AI Doctor:** Built-in heuristics to classify failure modes.

## 📦 Installation

```bash
git clone https://github.com/Eutonics/DeepDrift.git
cd DeepDrift
pip install .
##⚡ Quick Start

```
import torchvision.models as models
from deepdrift import DeepDriftMonitor, diagnose_drift

# 1. Load your model
model = models.resnet18(pretrained=True)
monitor = DeepDriftMonitor(model, arch_name='ResNet-18')

# 2. Calibrate on clean data (establish baseline)
# monitor.calibrate(train_loader)

# 3. Diagnose OOD data
# drift_profile = monitor.scan(batch)
# status = diagnose_drift(drift_profile)
# print(status)
📄 Citation
If you use DeepDrift, please cite our paper:
code
Bibtex
@article{evtushenko2025deepdrift,
  title={DeepDrift: A Layer-Wise Diagnostic Framework for Neural Network Robustness},
  author={Evtushenko, Alexey},
  journal={arXiv preprint},
  year={2025}
}
