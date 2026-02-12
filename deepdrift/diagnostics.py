from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class VelocityDiagnosis:
    peak_velocity: float
    layer_velocities: List[float]
    is_anomaly: bool
    threshold: Optional[float] = None
    drift_score: Optional[float] = None # For backward compatibility
    status: str = "NORMAL"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        icon = "🟢" if not self.is_anomaly else "🔴"
        return f"{icon} Velocity Diagnosis: {self.status} | Peak Velocity: {self.peak_velocity:.4f}"

@dataclass
class VisionDiagnosis:
    peak_velocity: float
    layer_velocities: List[float]
    is_anomaly: bool
    threshold: Optional[float] = None
    drift_score: Optional[float] = None
    status: str = "NORMAL"
    layer_drifts: Dict[str, float] = field(default_factory=dict) # For backward compatibility

    def __repr__(self):
        icon = "🟢" if not self.is_anomaly else "🔴"
        return f"{icon} Vision Diagnosis: {self.status} | Peak Velocity: {self.peak_velocity:.4f}"

@dataclass
class LLMDiagnosis:
    velocity: float
    is_anomaly: bool
    threshold: Optional[float] = None
    status: str = "NORMAL"

    def __repr__(self):
        icon = "🟢" if not self.is_anomaly else "🔴"
        return f"{icon} LLM Diagnosis: {self.status} | Velocity: {self.velocity:.4f}"
