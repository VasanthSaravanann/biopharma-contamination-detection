"""Benchmark inference latency for the ensemble detector.

Usage:
  python scripts/benchmark_latency.py --n 100 --warmup 10
"""
import time
import platform
import argparse
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_simulation import UVVisSpectraGenerator
from anomaly_detection import ModelConfig, EnsembleAnomalyDetector


def measure_latency(detector, X_sample, n=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        detector.predict(X_sample)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        detector.predict(X_sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(times)
    return arr.mean(), arr.std(), arr.min(), arr.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    gen = UVVisSpectraGenerator(seed=42)
    df = gen.generate_dataset(n_clean=200, n_contaminated_per_type=0)
    from feature_extraction import SpectralFeatureExtractor
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].fillna(0).values

    # Train a small ensemble quickly
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X)

    # Single sample
    X_sample = X[:args.batch]

    mean, std, mn, mx = measure_latency(ensemble, X_sample, n=args.n, warmup=args.warmup)

    print(f"Host: {platform.platform()}")
    print(f"Samples: {X_sample.shape}, mean={mean:.2f} ms, std={std:.2f} ms, min={mn:.2f}, max={mx:.2f}")


if __name__ == '__main__':
    main()
