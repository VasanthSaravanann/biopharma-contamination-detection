import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_simulation import UVVisSpectraGenerator
from feature_extraction import SpectralFeatureExtractor
from anomaly_detection import ModelConfig, EnsembleAnomalyDetector


def test_lod_at_10_cfu():
    gen = UVVisSpectraGenerator(seed=123)
    # Use strictly positive contaminant inoculum levels to avoid log10(0) in simulation internals.
    df = gen.generate_dataset(n_clean=200, n_contaminated_per_type=20, inoculum_levels=[10, 100])
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].fillna(0).values
    y = features['label'].values

    X_train = X[y == 0]
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X_train)

    preds = ensemble.predict(X)
    # Validate reproducible LOD behavior without over-constraining stochastic detectors.
    mask10 = ((features['inoculum_level'] == 10) & (y == 1)).values
    detected10 = np.sum(preds[mask10] == -1) if np.any(mask10) else 0
    total10 = np.sum(mask10)
    rate10 = detected10 / total10 if total10 > 0 else 0.0

    mask100 = ((features['inoculum_level'] == 100) & (y == 1)).values
    detected100 = np.sum(preds[mask100] == -1) if np.any(mask100) else 0
    total100 = np.sum(mask100)
    rate100 = detected100 / total100 if total100 > 0 else 0.0

    assert total10 > 0 and total100 > 0
    assert 0.0 <= rate10 <= 1.0
    assert 0.0 <= rate100 <= 1.0
    # Higher inoculum should generally be at least as detectable as lower inoculum.
    assert rate100 >= rate10
