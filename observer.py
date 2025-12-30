import numpy as np
from collections import deque
from enum import Enum
from dataclasses import dataclass

class MonitorState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"  # High Trend (Predictive)
    ALERT = "ALERT"      # Threshold Breach (Critical)

@dataclass
class ObserverConfig:
    """
    Configuration for the layer observer.
    """
    theta_high: float = 3.0   # Alert threshold (Z-score)
    theta_low: float = 1.5    # Recovery threshold (Hysteresis)
    theta_slope: float = 0.05 # Trend threshold (Drift per step) - Sensitivity
    window_size: int = 20     # History window for trend calculation

class LayerObserver:
    """
    Stateful observer for a single neural layer. 
    Implements Hysteresis (Signal Debouncing) and Trend Analysis (Early Warning).
    """
    def __init__(self, layer_name, config=ObserverConfig()):
        self.name = layer_name
        self.cfg = config
        self.history = deque(maxlen=config.window_size)
        self.state = MonitorState.NORMAL
        self.current_beta = 0.0
        
    def update(self, value, step):
        """
        Updates state based on new drift value.
        Returns: (State, EventString or None)
        """
        self.history.append((step, value))
        event = None
        
        # 1. Calculate Trend (Beta / Slope)
        self.current_beta = 0.0
        if len(self.history) >= 5:
            # Simple linear regression: y = beta*x + alpha
            x = np.array([h[0] for h in self.history])
            y = np.array([h[1] for h in self.history])
            if np.std(x) > 0:
                self.current_beta = np.polyfit(x, y, 1)[0]

        # 2. State Machine with Hysteresis
        
        # TRANSITION FROM NORMAL
        if self.state == MonitorState.NORMAL:
            if value >= self.cfg.theta_high:
                self.state = MonitorState.ALERT
                event = f"🔴 ALERT [{self.name}]: Threshold Breach ({value:.2f} >= {self.cfg.theta_high})"
            elif self.current_beta > self.cfg.theta_slope:
                self.state = MonitorState.WARNING
                event = f"⚠️ WARNING [{self.name}]: Rapid Drift Detected (Slope {self.current_beta:.3f})"
        
        # TRANSITION FROM WARNING
        elif self.state == MonitorState.WARNING:
            if value >= self.cfg.theta_high:
                self.state = MonitorState.ALERT
                event = f"🔴 ALERT [{self.name}]: Escalated to Threshold Breach"
            elif self.current_beta <= 0:
                self.state = MonitorState.NORMAL
                event = f"✅ INFO [{self.name}]: Trend Stabilized"

        # TRANSITION FROM ALERT (Hysteresis Applied)
        elif self.state == MonitorState.ALERT:
            if value <= self.cfg.theta_low:
                self.state = MonitorState.NORMAL
                event = f"🟢 RECOVERY [{self.name}]: Signal returned to normal ({value:.2f})"
                
        return self.state, event
