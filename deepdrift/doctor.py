import numpy as np

def diagnose_drift(drift_profile, threshold=3.0):
    """
    Classifies neural pathology based on the DeepDrift profile.
    """
    if not drift_profile or len(drift_profile) < 4:
        return "Unknown (Profile too short)"
        
    profile = np.array(drift_profile)
    max_drift = np.max(profile)
    mean_drift = np.mean(profile)
    
    # Assuming profile order is [UV, Mid, Deep, IR]
    uv, mid, deep, ir = profile[0], profile[1], profile[2], profile[-1]
    
    if max_drift < threshold:
        return "Stable"
    
    if mean_drift > threshold * 1.5:
        return "CRITICAL: Global Collapse (Disorientation)"
        
    if (ir > deep or deep > mid) and ir > threshold:
        return "WARNING: Avalanche Effect (Geometric Failure)"
        
    if uv > threshold and uv > mid:
        return "ALERT: Sensor Shock (Input Noise)"
        
    if mid == max_drift and mid > threshold:
        return "WARNING: Spurious Correlation (Feature Mismatch)"
        
    return "Anomaly Detected (Unclassified)"
