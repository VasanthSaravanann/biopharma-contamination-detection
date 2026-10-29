import numpy as np
from typing import Dict, Any, Optional, List
from scipy.stats import ks_2samp


def compute_drift_acceptance(
    X_ref: np.ndarray,
    X_target: np.ndarray,
    alpha: float = 0.05,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute per-feature KS test between reference and target and return acceptance metrics.

    Returns a dict with:
    - n_features
    - n_accepted (p-value > alpha)
    - acceptance_ratio
    - per_feature: list of {feature, stat, pvalue, accepted}
    - summary stats: mean_stat, median_stat
    """
    if X_ref is None or X_target is None:
        return {"error": "missing_data"}

    X_ref = np.asarray(X_ref)
    X_target = np.asarray(X_target)

    if X_ref.ndim != 2 or X_target.ndim != 2:
        return {"error": "invalid_shape"}

    n_features = X_ref.shape[1]
    if X_target.shape[1] != n_features:
        return {"error": "feature_dim_mismatch"}

    per_feature = []
    stats = []
    pvalues = []

    for i in range(n_features):
        a = X_ref[:, i]
        b = X_target[:, i]
        try:
            res = ks_2samp(a, b, alternative='two-sided', mode='auto')
            stat = float(res.statistic)
            pval = float(res.pvalue)
        except Exception:
            stat = float('nan')
            pval = float('nan')

        accepted = bool((not np.isnan(pval)) and (pval > alpha))
        fname = feature_names[i] if feature_names and i < len(feature_names) else f"f{i}"
        per_feature.append({"feature": fname, "stat": stat, "pvalue": pval, "accepted": accepted})
        stats.append(stat if not np.isnan(stat) else 0.0)
        pvalues.append(pval if not np.isnan(pval) else 0.0)

    n_accepted = sum(1 for p in per_feature if p.get('accepted', False))
    acceptance_ratio = float(n_accepted) / float(n_features) if n_features > 0 else 0.0

    out = {
        "n_features": int(n_features),
        "n_accepted": int(n_accepted),
        "acceptance_ratio": acceptance_ratio,
        "mean_stat": float(np.mean(stats)),
        "median_stat": float(np.median(stats)),
        "per_feature": per_feature,
    }
    return out
