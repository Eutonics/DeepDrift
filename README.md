# 🧠 DeepDrift  
### Neural MRI for AI Robustness

> **Detect hallucinations, model collapse, and policy panic *before* they happen.**  
> A universal thermodynamic framework for monitoring internal neural stability across Vision, Language, and Control.

[![PyPI Version](https://img.shields.io/pypi/v/deepdrift?color=blue)](https://pypi.org/project/deepdrift/)
[![Downloads](https://static.pepy.tech/badge/deepdrift)](https://pepy.tech/project/deepdrift)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Eutonics/DeepDrift-Explorer)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.18086612.svg)](https://doi.org/10.5281/zenodo.18086612)

---

## ⚡ What is DeepDrift?

Traditional AI monitoring looks at **inputs** (data drift) or **outputs** (confidence, perplexity).  
**DeepDrift looks inside the model itself.**

Think of it as an **MRI for neural networks**.

DeepDrift measures **Semantic Velocity** — the rate of change of hidden representations — and uses it as a real-time stability signal.  
It functions like a **“Check Engine” light** for AI systems.

| Domain | Failure Mode | DeepDrift Diagnosis |
|------|-------------|--------------------|
| 👁️ Vision | OOD / geometric stress | **Global Collapse**, **Avalanche Effect** |
| 🗣️ LLMs | Confident hallucinations | **Semantic Tremor** (7–8 tokens early) |
| 🤖 RL / Robotics | Silent policy failure | **Panic Zone** (seconds before crash) |

> *Softmax tells you where the model ends up.  
> Semantic Velocity tells you whether it is losing control on the way.*

---

## 🚀 Quick Start

### Installation
```bash
pip install deepdrift
```

---

## 🧪 Usage Examples

### 1️⃣ Vision — Detect Architectural Collapse

```python
from deepdrift import DeepDriftMonitor
import torchvision.models as models

model = models.resnet50(pretrained=True)
monitor = DeepDriftMonitor(model, arch_name="ResNet")

monitor.calibrate(clean_loader)

status, _ = monitor.step(ood_image)
print(f"Drift Score: {status['IR']['drift']:.2f}")
```

---

### 2️⃣ LLM — Real-Time Hallucination Detector

```python
from transformers import AutoModelForCausalLM
from deepdrift import DeepDriftMonitor

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
monitor = DeepDriftMonitor(model, arch_name="Qwen", strategy="last_token")

status, _ = monitor.step(input_ids)
velocity = status["IR"]["velocity"]

if velocity > 300:
    print("⚠️ High Semantic Tremor — possible hallucination")
```

---

## 🔬 The Science: Optical Depth Dynamics (ODD)

DeepDrift implements **Optical Depth Dynamics (ODD)** — a thermodynamic diagnostic framework for neural networks.

- **Laminar Flow** → stable reasoning  
- **Turbulent Flow** → hallucination, panic, collapse

📄 Full paper: https://doi.org/10.5281/zenodo.18086612

---

## ⚡ Performance (v0.4.0)

| Method | Inference | Monitor | Overhead |
|------|-----------|---------|----------|
| Full monitor | 12.8 ms | 16.2 ms | 126% |
| **DeepDrift v0.4** | **12.8 ms** | **0.03 ms** | **0.2%** |

---

## 🛠️ Features

- Plug & Play (PyTorch, Transformers, SB3)
- Auto-detect architectures
- <1% inference overhead
- Unsupervised
- Interpretable diagnostics

---

## 👤 Author

**Alexey Evtushenko**  
Independent Researcher & Engineer

GitHub: https://github.com/Eutonics  
X: https://x.com/axelgravitone  

---

> **Stop guessing why your model failed.  
> See exactly where it broke.**
