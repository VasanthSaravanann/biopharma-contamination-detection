import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Optional


def compute_pca_baseline_auc(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute PCA reconstruction-error baseline and ROC-AUC on test set.

    - Fits PCA on `X_ref` (reference/clean data).
    - Reconstructs `X_test` using `n_components` and computes per-sample MSE.
    - Computes ROC-AUC using `y_test` (1 indicates contaminated/anomalous).

    Returns dict with `auc`, `n_components`, `explained_variance_ratio`, and summary MSEs.
    """
    X_ref = np.asarray(X_ref)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if X_ref.ndim != 2 or X_test.ndim != 2:
        return {"error": "invalid_shape"}
    if X_ref.shape[1] != X_test.shape[1]:
        return {"error": "feature_dim_mismatch"}

    # Choose n_components
    max_comp = min(X_ref.shape[0], X_ref.shape[1])
    if n_components is None:
        n_components = min(10, max_comp)
    else:
        n_components = min(int(n_components), max_comp)

    pca = PCA(n_components=n_components)
    pca.fit(X_ref)
    X_recon = pca.inverse_transform(pca.transform(X_test))
    mse = np.mean((X_test - X_recon) ** 2, axis=1)

    try:
        auc = float(roc_auc_score(y_test, mse))
    except Exception:
        auc = float('nan')

    out = {
        "pca_n_components": int(n_components),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "pca_reconstruction_auc": auc,
        "mse_mean_clean": float(np.mean(mse[y_test == 0])) if np.any(y_test == 0) else None,
        "mse_mean_contaminated": float(np.mean(mse[y_test == 1])) if np.any(y_test == 1) else None,
    }
    return out
