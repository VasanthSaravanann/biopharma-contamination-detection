"""Drift monitoring helpers and alerting utilities."""
from typing import Dict, Any
from .drift import compute_drift_acceptance


def check_and_alert_drift(X_ref, X_target, feature_names=None, alpha=0.05, alert_threshold=0.25):
    """Compute drift acceptance and return alert status.

    alert_threshold: fraction of accepted features below which alert should trigger.
    Returns dict with monitor results and `alert` boolean.
    """
    res = compute_drift_acceptance(X_ref, X_target, alpha=alpha, feature_names=feature_names)
    acceptance = res.get('acceptance_ratio', 0.0)
    alert = acceptance < alert_threshold
    res['alert'] = alert
    res['alert_threshold'] = alert_threshold
    return res
