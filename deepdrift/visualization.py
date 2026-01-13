import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def plot_drift_profile(drift_profile, title="DeepDrift Profile", save_path=None):
    """
    Visualizes the layer-wise drift profile.
    Returns a PIL Image object if no save_path is provided.
    """
    layers = ["UV (Input)", "Mid-Level", "Deep-Level", "IR (Output)"]
    
    # Fallback if profile length differs
    if len(drift_profile) != len(layers):
        layers = [f"L{i}" for i in range(len(drift_profile))]
    
    x = np.arange(len(layers))
    
    fig = plt.figure(figsize=(8, 5))
    
    # Dynamic coloring based on threshold
    colors = ['green' if d < 3.0 else 'red' for d in drift_profile]
    
    plt.bar(x, drift_profile, color=colors, alpha=0.8, edgecolor='black')
    
    # Threshold line
    plt.axhline(3.0, color='red', linestyle='--', label='Alert Threshold (3σ)')
    
    # Zones background
    plt.axvspan(-0.5, 0.5, color='red', alpha=0.05) # UV
    plt.axvspan(len(layers)-1.5, len(layers)-0.5, color='blue', alpha=0.05) # IR
    
    plt.xticks(x, layers)
    plt.ylabel("Drift Score (Z)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
    else:
        # Convert plot to image for Gradio/Display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
