# 🧠 DeepDrift: Neural MRI for AI Robustness

> **Detect hallucinations, model collapse, and policy panic before they happen.**
> A universal thermodynamic framework for monitoring internal neural stability across Vision, Language, and Control.

[![PyPI Version](https://img.shields.io/pypi/v/deepdrift?color=blue)](https://pypi.org/project/deepdrift/)
[![Downloads](https://static.pepy.tech/badge/deepdrift)](https://pepy.tech/project/deepdrift)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Eutonics/DeepDrift-Hallucination-Detector)
[![arXiv](https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg)](https://arxiv.org)

---

## ⚡ What is DeepDrift?

Traditional AI monitoring looks at inputs (drift) or outputs (confidence/perplexity). **DeepDrift scans the model's internal physics.**

By measuring **Semantic Velocity** (the rate of change in hidden states), DeepDrift acts as a "Check Engine" light for neural networks:

| Domain | Problem | DeepDrift Diagnosis |
| :--- | :--- | :--- |
| **👁️ Vision** | **OOD / Corruption** | Detects **Global Collapse** (ViT) or **Avalanche Effect** (CNN) at layer $z=0$. |
| **🗣️ LLM** | **Hallucinations** | Detects **Semantic Tremor** (high-frequency velocity spikes) 7-8 tokens before the lie is finished. |
| **🤖 RL/Robotics** | **Policy Collapse** | Identifies the **Panic Zone** (internal instability) *seconds before* the agent crashes ($p < 0.001$). |

> *"Softmax measures the final destination. Semantic Velocity measures the stability of the journey."*

---

## 🚀 Quick Start

### Installation
```bash
pip install deepdrift
