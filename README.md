# DeepDrift

[![PyPI](https://img.shields.io/pypi/v/deepdrift)](https://pypi.org/project/deepdrift/)
[![Python](https://img.shields.io/pypi/pyversions/deepdrift)](https://pypi.org/project/deepdrift/)
[![License](https://img.shields.io/github/license/Eutonics/DeepDrift)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Zenodo-blue)](https://zenodo.org/records/18622319)

**DeepDrift** is a runtime neural network monitoring library based on
analyzing the rate of change of hidden states (*Semantic Velocity*). It
requires no additional training, calibrates only on normal data, and
introduces \<1.5% computational overhead.

![Graphical Abstract](figures/graphical_abstract.png)

------------------------------------------------------------------------

## Installation

``` bash
# Install from source
git clone https://github.com/Eutonics/DeepDrift.git
cd DeepDrift
pip install .

# Or via pip (after publication)
# pip install deepdrift
```

------------------------------------------------------------------------

## Quick Start (ViT OOD)

``` python
from deepdrift import DeepDriftMonitor
from torchvision.models import vit_b_16
import torch

model = vit_b_16(pretrained=True)
monitor = DeepDriftMonitor(
    model,
    layer_names=['encoder.layers.5', 'encoder.layers.11'],
    pooling='cls'
)

x = torch.randn(1, 3, 224, 224)
_ = model(x)

velocities = monitor.get_spatial_velocity()
print(f"Peak velocity: {max(velocities):.4f}")
```

------------------------------------------------------------------------

## Use Cases

### 1. OOD Detection in Vision Transformers

Use `DeepDriftVision` for automatic calibration and Out-of-Distribution
detection.

![ViT OOD Profile](figures/vit_ood_profile.png)

  Dataset          Metric          AUC (DeepDrift)
  ---------------- --------------- -----------------
  CIFAR-100 (In)   \-              \-
  SVHN (OOD)       Peak Velocity   **0.982**
  LSUN (OOD)       Peak Velocity   **0.975**

``` python
from deepdrift import DeepDriftVision

monitor = DeepDriftVision(model)
monitor.fit(train_dataloader)

diagnosis = monitor.predict(test_batch)
if diagnosis.is_anomaly:
    print("OOD detected!")
```

------------------------------------------------------------------------

### 2. Predicting RL Agent Collapse

``` python
from deepdrift.rl import DeepDriftRL

monitor = DeepDriftRL(agent.policy, threshold=0.15)

for obs in episode:
    diag = monitor.step(obs)
    if diag.is_anomaly:
        print(f"Warning: High instability detected! Velocity: {diag.peak_velocity}")
```

------------------------------------------------------------------------

### 3. Hallucination Detection in LLMs

``` python
from deepdrift import DeepDriftGuard

guard = DeepDriftGuard(llm_model)

for token in generation_loop:
    diag = guard(token)
    if diag.is_anomaly:
        print("Possible hallucination detected.")
```

------------------------------------------------------------------------

### 4. Early Memorization Detection in Diffusion Models

See `experiments/diffusion_memorization.py`.

------------------------------------------------------------------------

## API Reference

### DeepDriftMonitor

-   `__init__(model, layer_names, pooling, n_channels, ...)`
-   `get_spatial_velocity()`
-   `get_temporal_velocity()`
-   `calibrate(dataloader)`

### DeepDriftVision / DeepDriftGuard

Specialized wrappers with simplified API.

------------------------------------------------------------------------

## Reproducing Experiments

``` bash
git clone https://github.com/Eutonics/DeepDrift
cd DeepDrift
bash scripts/reproduce_all.sh
```

------------------------------------------------------------------------

## Citation

``` bibtex
@article{evtushenko2026deepdrift,
  title={DeepDrift: Zero-Training Hidden-State Monitoring for Robustness in Vision, Language, and Generative Models},
  author={Alexey Evtushenko},
  year={2026},
  journal={arXiv preprint},
  url={https://github.com/Eutonics/DeepDrift}
}
```

------------------------------------------------------------------------

## License

MIT

------------------------------------------------------------------------

Generated using K-Dense Web (https://k-dense.ai)
