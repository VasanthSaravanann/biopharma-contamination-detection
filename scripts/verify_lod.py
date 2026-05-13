"""Verify limit-of-detection (LOD) experiments reproducibly.

This script runs a small sweep across inoculum levels and reports
the detection rate to verify 10 CFU/mL LOD behavior.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_simulation import UVVisSpectraGenerator
from feature_extraction import SpectralFeatureExtractor
from anomaly_detection import ModelConfig, EnsembleAnomalyDetector


def run_lod_sweep(seed=42):
    gen = UVVisSpectraGenerator(seed=seed)
    # generate varying inoculum levels synthetic data
    results = []
    for inoc in [0, 1, 5, 10, 50, 100]:
        df = gen.generate_dataset(n_clean=200, n_contaminated_per_type=20, inoculum_levels=[inoc])
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
        # anomaly predicted as -1
        detected = np.sum((preds == -1) & (y == 1))
        total = np.sum(y == 1)
        rate = detected / total if total > 0 else 0.0
        results.append({'inoculum': inoc, 'detected': int(detected), 'total': int(total), 'rate': float(rate)})

    return pd.DataFrame(results)


def main():
    df = run_lod_sweep()
    print(df.to_string(index=False))
    df.to_csv('output/lod_sweep.csv', index=False)


if __name__ == '__main__':
    main()
