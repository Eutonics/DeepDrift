import numpy as np

def diagnose_drift(drift_profile, threshold=3.0):
    """
    Holistic diagnosis based on layer coupling.
    Interprets the FLOW of drift, not just isolated spikes.
    """
    if not drift_profile or len(drift_profile) < 4:
        return "Unknown (Profile too short)"
        
    profile = np.array(drift_profile)
    max_drift = np.max(profile)
    
    # Map layers (Assuming [UV, Mid, Deep, IR] order)
    uv, mid, deep, ir = profile[0], profile[1], profile[2], profile[-1]
    
    # 1. Healthy State
    if max_drift < threshold:
        return "✅ Stable"
    
    # 2. Benign Shift (Validation of Robustness)
    # UV горит, но IR спокоен. Модель фильтрует шум.
    if uv > threshold and ir < threshold and ir < uv * 0.6:
        return "ℹ️ INFO: Benign Sensor Shift (Filtered)"

    # 3. Avalanche (Accumulation) - CHECK FIRST for specificity
    # Ошибка растет к выходу (характерно для CNN)
    if ir > deep and deep > mid and ir > threshold:
        return "⚠️ WARNING: Avalanche Effect (Geometric Instability)"

    # 4. Internal Rot (Spurious Correlation)
    # Вход ок, Выход ок, но Середина горит.
    if mid > threshold and mid > uv and mid > ir:
        return "🔴 ALERT: Internal Feature Mismatch (Spurious Correlation)"

    # 5. Critical Failure (Global Collapse)
    # Если горит всё, или среднее очень высокое (характерно для ViT)
    if np.mean(profile) > threshold * 1.2:
        return "⛔ CRITICAL: Global Collapse (Model Disoriented)"
        
    # Fallback
    return f"Anomaly Detected (Max Z={max_drift:.1f})"
