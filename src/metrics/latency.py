import time
import numpy as np
from typing import Any, Dict


def measure_latency(detector: Any, X: np.ndarray, n_repeats: int = 10, max_samples: int = 200) -> Dict[str, Any]:
    """Measure inference latency for a detector's `predict_proba` method.

    Parameters
    - detector: object with `predict_proba(X)` method
    - X: feature matrix (n_samples, n_features)
    - n_repeats: number of repeated full-batch calls to measure
    - max_samples: max number of samples to use from X for measurement

    Returns a dict with aggregated latency stats (ms).
    """
    if X is None or getattr(X, 'shape', None) is None:
        return {"error": "no_data"}

    n_samples = min(int(X.shape[0]), int(max_samples))
    if n_samples <= 0:
        return {"error": "no_samples"}

    X_sub = X[:n_samples]

    # Warm-up
    try:
        _ = detector.predict_proba(X_sub)
    except Exception:
        # If detector doesn't implement predict_proba, try predict
        try:
            _ = detector.predict(X_sub)
        except Exception:
            return {"error": "detector_no_predict"}

    call_times = []
    for i in range(max(1, int(n_repeats))):
        t0 = time.perf_counter()
        try:
            _ = detector.predict_proba(X_sub)
        except Exception:
            _ = detector.predict(X_sub)
        t1 = time.perf_counter()
        call_times.append((t1 - t0))

    call_times = np.array(call_times)
    per_call_ms = call_times * 1000.0
    per_sample_ms = per_call_ms / float(n_samples)

    stats = {
        "n_samples_measured": int(n_samples),
        "n_calls": int(len(call_times)),
        "per_call_ms": {
            "mean": float(per_call_ms.mean()),
            "median": float(np.median(per_call_ms)),
            "p95": float(np.percentile(per_call_ms, 95)),
        },
        "per_sample_ms": {
            "mean": float(per_sample_ms.mean()),
            "median": float(np.median(per_sample_ms)),
            "p95": float(np.percentile(per_sample_ms, 95)),
        }
    }
    return stats
