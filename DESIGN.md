# DeepDrift v1.1.0: Technical Design & Architecture

## Overview
DeepDrift 1.1.0 introduces a unified core architecture for Semantic Velocity monitoring. The library is designed to be plug-and-play, requiring zero retraining and minimal overhead (<1.5%).

## Package Structure
- `deepdrift/core.py`: The heart of the library. Contains `DeepDriftMonitor` which manages hooks and velocity calculations.
- `deepdrift/vision.py`, `llm.py`, `rl.py`: Domain-specific wrappers providing high-level APIs (Backward compatible).
- `deepdrift/utils/`:
    - `hooks.py`: Logic for automated layer discovery and registration.
    - `pooling.py`: Handlers for different tensor types (spatial, sequential, flat).
    - `stats.py`: Statistical robust estimation (IQR, Bootstrap).
- `deepdrift/diagnostics.py`: Structured dataclasses for results.

## Key Design Decisions

### 1. Unified Hook Management
Instead of duplicating hook logic across Vision and LLM modules, `DeepDriftMonitor` centralizes registration. It uses `forward_hooks` to capture activations without modifying the original model's forward pass.

### 2. Semantic Velocity Calculation
- **Spatial Velocity**: Measures the change in representation between subsequent layers $L_i$ and $L_{i+1}$. A spike in this velocity often indicates that the model is struggling to process the input (OOD).
- **Temporal Velocity**: Measures the change in the *same* layer's state between time steps $T$ and $T+1$. Critical for sequential models (RL, LLM).

### 3. Sparse Sampling & Pooling
To minimize overhead, activations are first pooled (Global Avg Pool for CNN, CLS token for ViT) and then optionally sub-sampled via `n_channels`. This reduces the dimensionality of the velocity calculation while preserving enough signal for drift detection.

### 4. Robust IQR Thresholding
Anomaly thresholds are calculated as $Q_{75} + 1.5 \times IQR$ on a calibration dataset. This is more robust to outliers in the "normal" data compared to traditional Z-scores.

## Extension Guide
To add support for a new architecture:
1. Define a custom pooling function in `utils/pooling.py` if needed.
2. Update the heuristic in `utils/hooks.py` to recognize the new layers.
3. Use `DeepDriftMonitor` with the new layer names.

## Trade-offs
- **Precision vs. Performance**: We use sparse sampling which may lose subtle signals but ensures near-zero latency impact.
- **Auto-Hook Heuristics**: While convenient, heuristics may fail on highly customized architectures, requiring users to manually specify `layer_names`.

---
Generated using K-Dense Web ([k-dense.ai](https://k-dense.ai))
