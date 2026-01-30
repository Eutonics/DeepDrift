import time
from functools import wraps

class KineticRouter:
    """
    Middleware for production inference.
    Implements 'Fail-Fast' logic: rejects requests if DeepDrift detects instability.
    """
    def __init__(self, monitor=None):
        self.monitor = monitor
        self.stats = {"processed": 0, "rejected": 0, "avg_latency": 0.0}

    def guard(self, func):
        """Decorator for inference functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                # Run inference
                result = func(*args, **kwargs)
                self.stats["processed"] += 1
                return result
                
            except Exception as e:
                # Catch DeepDrift specific stop signals
                error_msg = str(e)
                if "STOP" in error_msg or "Velocity" in error_msg or "CRITICAL" in error_msg:
                    self.stats["rejected"] += 1
                    print(f"🛡 KineticRouter: BLOCKED -> {error_msg}")
                    return {
                        "error": "Request rejected: Model Instability Detected", 
                        "code": 422,
                        "details": error_msg
                    }
                raise e
                
            finally:
                duration = time.time() - start_time
                self.stats["avg_latency"] = (self.stats["avg_latency"] * 0.9) + (duration * 0.1)
                
        return wrapper
