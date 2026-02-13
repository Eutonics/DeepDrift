# DeepDrift

[![PyPI version](https://img.shields.io/pypi/v/deepdrift)](https://pypi.org/project/deepdrift/)
[![Python versions](https://img.shields.io/pypi/pyversions/deepdrift)](https://pypi.org/project/deepdrift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/deepdrift)](https://pypi.org/project/deepdrift/)
[![GitHub stars](https://img.shields.io/github/stars/Eutonics/DeepDrift?style=social)](https://github.com/Eutonics/DeepDrift)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18622319.svg)](https://doi.org/10.5281/zenodo.18622319)

---

**DeepDrift** is a runtime neural network monitoring library based on analyzing the rate of change of hidden states — **Semantic Velocity**.

It requires **no additional training**, calibrates only on normal data, and introduces **<1.5% computational overhead**.

✅ **Zero‑training** — no gradient updates, no labeled failures  
✅ **Architecture‑agnostic** — works with PyTorch models (ViT, CNN, MLP, Transformers)  
✅ **Cross‑domain** — validated on LLMs, RL, vision, and diffusion models  
✅ **Production‑ready** — sparse sampling, IQR thresholding, <1.5% overhead  
✅ **Open source** — MIT license, pip installable

---

## 📊 Graphical Abstract

![DeepDrift Overview](https://raw.githubusercontent.com/Eutonics/DeepDrift/main/figures/graphical_abstract.png)
*Semantic Velocity monitors internal hidden‑state dynamics across four domains: LLM hallucination detection, RL failure prediction, Vision Transformer OOD detection, and diffusion memorization detection.*

---

## 🚀 Installation

### From PyPI (recommended)
```bash
pip install deepdrift
```

### From source (latest development version)
```bash
git clone https://github.com/Eutonics/DeepDrift.git
cd DeepDrift
pip install -e .
```

### For experiments and development (extra dependencies)
```bash
pip install -e ".[all]"
```

---

## 🔧 Quick Start

### Vision Transformer OOD Detection
```python
from deepdrift import DeepDriftMonitor
from torchvision.models import vit_b_16

model = vit_b_16(pretrained=True)
monitor = DeepDriftMonitor(
    model,
    layer_names=['encoder.layers.11'],  # monitor last layer
    pooling='cls'
)

x = torch.randn(1, 3, 224, 224)
_ = model(x)
vel = monitor.get_spatial_velocity()
print(f"Peak velocity: {max(vel):.4f}")
```

### RL Agent Failure Prediction
```python
from deepdrift import DeepDriftMonitor
from stable_baselines3 import PPO

model = PPO.load("ppo_lunarlander")
monitor = DeepDriftMonitor(
    model.policy,
    layer_names=None,  # auto-detect
    pooling='flatten'
)

obs = env.reset()
while True:
    vel = monitor.get_temporal_velocity()
    if vel > threshold:  # anomaly detected!
        print("⚠️ Agent panic detected!")
    # ... take action
```

---

## 📚 Documentation

- **Zenodo**: [10.5281/zenodo.18622319](https://doi.org/10.5281/zenodo.18622319)
- **GitHub**: [https://github.com/Eutonics/DeepDrift](https://github.com/Eutonics/DeepDrift)
- **PyPI**: [https://pypi.org/project/deepdrift/](https://pypi.org/project/deepdrift/)
- **Hugging Face Demo**: [DeepDrift Explorer](https://huggingface.co/spaces/Eutonics/DeepDrift-Explorer)

---

## 📈 Key Results

| Domain | Model / Algo | AUC | Cohen's d | Lead Time |
|--------|--------------|-----|-----------|-----------|
| LLM Hallucination | Qwen-2.5-7B | 0.891 | 3.12 | 7.2 tokens |
| RL (LunarLander) | PPO | 0.968 | 2.47 | 12.3 steps |
| RL (CartPole) | DQN | 0.985 | 2.79 | 168 steps |
| RL (CartPole+noise) | PPO | 1.000 | 3.37 | 97 steps |
| ViT OOD | ViT-B/16 | 0.817 | — | — |
| Diffusion Memorization | U-Net | 3× earlier | — | — |

---

## 🧪 Reproducibility

All experiments are fully reproducible:

```bash
git clone https://github.com/Eutonics/DeepDrift.git
cd DeepDrift
pip install -e ".[all]"
bash scripts/reproduce_all.sh
```

Results will be saved in `results/` and `paper/figures/`.

---

## 📖 Citation

If you use DeepDrift in your research, please cite:

```bibtex
@article{evtushenko2026deepdrift,
  title={DeepDrift: Zero-Training Hidden-State Monitoring for Robustness in Vision, Language, and Generative Models},
  author={Alexey Evtushenko},
  year={2026},
  journal={arXiv preprint},
  url={https://github.com/Eutonics/DeepDrift}
}
```

---

## 📄 License

MIT License © 2026 Alexey Evtushenko

---

*DeepDrift was developed as independent research. No affiliation with Yandex, K‑Dense, or any other organization.*
