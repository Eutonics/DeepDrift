# 🧠 DeepDrift: Neural MRI for AI Robustness

> **Detect hallucinations, model collapse, and policy panic before they happen.**  
> A universal thermodynamic framework for monitoring internal neural stability across Vision, Language, and Control.

[![PyPI Version](https://img.shields.io/pypi/v/deepdrift?color=blue)](https://pypi.org/project/deepdrift/)
[![Downloads](https://static.pepy.tech/badge/deepdrift)](https://pepy.tech/project/deepdrift)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Eutonics/DeepDrift-Explorer)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.18086612.svg)](https://doi.org/10.5281/zenodo.18086612)

---

## ⚡ What is DeepDrift?

Traditional AI monitoring looks at inputs (data drift) or outputs (confidence, perplexity).  
**DeepDrift scans the model’s internal physics — like an MRI for neural networks.**

By measuring **Semantic Velocity** — the rate of change in hidden states — DeepDrift acts as a **"Check Engine" light** for AI systems:

| Domain | Problem | DeepDrift Diagnosis |
| :--- | :--- | :--- |
| **👁️ Vision** | Geometric stress, OOD data | Detects **Global Collapse** (ViT) or **Avalanche Effect** (CNN) at input layers. |
| **🗣️ LLM** | Confident hallucinations | Detects **Semantic Tremor** — high-frequency velocity spikes **7–8 tokens before** the lie finishes. |
| **🤖 RL / Robotics** | Silent policy failure | Identifies the **Panic Zone** — internal instability **seconds before** crash (`p < 0.001`, Cohen’s *d* > 2.0). |

> *“Softmax measures the final destination. Semantic Velocity measures the stability of the journey.”*

---

## 🚀 Quick Start

### Installation
```bash
pip install deepdrift
```
---

## 1. Vision — Detect Architectural Collapse

```bash

from deepdrift import DeepDriftMonitor
import torchvision.models as models

model = models.resnet50(pretrained=True)
monitor = DeepDriftMonitor(model, arch_name="ResNet")

# Calibrate on clean data
monitor.calibrate(clean_loader)

# Monitor new batch
status, _ = monitor.step(ood_image)
print(f"Drift Score: {status['IR']['drift']:.2f}")  # > 3.0 → anomaly
```

## 2. LLM — Real-Time Lie Detector

```bash

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
monitor = DeepDriftMonitor(model, arch_name="Qwen", strategy="last_token")

# During generation
status, _ = monitor.step(input_ids)
velocity = status['IR']['velocity']

if velocity > 300:
    print("⚠️ WARNING: High Semantic Tremor! Possible hallucination.")

```

## 🔬 The Science: Optical Depth Dynamics (ODD)
DeepDrift implements the Optical Depth Dynamics (ODD) framework — a unified diagnostic lens for neural networks.

We treat depth (spatial in vision, temporal in language) as a diagnostic dimension:

Laminar Flow: Low velocity → stable, factual, grounded processing.
Turbulent Flow: High velocity → confabulation, panic, structural failure.
This isn’t just theory — it’s a production-ready diagnostic tool with <1% overhead.

## 📄 Read the full work:
[Confidently Wrong: ODD as a Universal Thermodynamic Framework (Zenodo)](https://doi.org/10.5281/zenodo.18086612)

## ⚡ Production Performance (v0.4.0)

DeepDrift employs **Sparse Channel Sampling** and **Global Average Pooling**, making it suitable for high-load environments.

**Benchmark (ResNet-20 on CPU):**

| Method | Inference Time | Monitor Time | **Overhead** |
| :--- | :--- | :--- | :--- |
| Full Monitor | 12.8 ms | 16.2 ms | 126% |
| **DeepDrift v0.4** | **12.8 ms** | **0.03 ms** | **0.2%** |

*Note: With 50 sampled channels per layer, overhead becomes negligible (<1%) while maintaining >93% anomaly detection accuracy.*


## 🛠️ Features
Plug & Play: Works out-of-the-box with torch, transformers, stable-baselines3.
Auto-Detect: Supports ResNet, ViT, ConvNeXt, Llama, Qwen, GPT, and more.
Lightweight: <1% inference overhead via PyTorch forward hooks.
Unsupervised: No labels needed — only a small calibration set from nominal operation.
Interpretable: Outputs human-readable diagnostics: “Global Collapse”, “Mid-Layer Bulge”, “Policy Panic”.

## 👤 Author
Alexey Evtushenko — Independent Researcher & Engineer
Built this to bring reliability-first engineering to the world of neural networks.

- GitHub: [@Eutonics](https://github.com/Eutonics)
- X (Twitter): [@axelgravitone](https://x.com/axelgravitone)  
- Hugging Face: [DeepDrift-Explorer](https://huggingface.co/spaces/Eutonics/DeepDrift-Explorer)

## 📊 Benchmarks & Case Studies

### 💰 Financial Fraud Detection (PaySim)
We evaluated DeepDrift on the PaySim dataset (1.5M transactions) against a **"Shuffle Attack"** scenario (contextual fraud that preserves statistical moments like mean/std but breaks causality).

| Method | Feature Type | ROC-AUC | Result |
| :--- | :--- | :--- | :--- |
| Random Forest (Baseline) | Static (Mean/Std) | 0.16 | **Failed** (Blind) |
| **Random Forest + ODD** | **Static + Kinetic** | **0.40** | **+147% Improvement** |

**Why it works:** The model automatically selected Semantic Velocity as the **#1 most important feature** (Importance ~0.29), proving that kinetic dynamics contain signal invisible to static statistics.

![PaySim Results](figures/paysim_results.png)
*(See full reproduction in `examples/fintech.ipynb`)*

## 🚀 Live Demo

Try the tool yourself on Hugging Face Spaces:  
[![Open in HF Spaces](https://img.shields.io/badge/HF%20Spaces-DeepDrift-red)](https://huggingface.co/spaces/Eutonics/DeepDrift-Hallucination-Detector)

Watch how Semantic Velocity predicts hallucinations before they happen:


“Stop guessing why your model failed. See exactly where it broke.”
