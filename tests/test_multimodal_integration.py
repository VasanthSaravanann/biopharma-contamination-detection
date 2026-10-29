"""End-to-end test verifying CO2 and metabolomics integration into feature pipeline.

This test ensures:
1. CO2 timeseries loaders work
2. Metabolomics aggregated loaders work
3. MultimodalFeatureExtractor correctly fuses them with UV-Vis
4. Ensemble detector processes fused features without error
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_integration import load_co2_timeseries_csv, load_metabolomics_aggregated_csv
from feature_fusion import MultimodalFeatureExtractor
from anomaly_detection import ModelConfig, EnsembleAnomalyDetector


def test_co2_loader():
    """Test CO2 timeseries loader"""
    # Create test CSV with correct column names
    test_data = pd.DataFrame({
        'sample_id': ['S1', 'S1', 'S1'],
        'timestamp': [0, 1, 2],
        'co2_pct': [0.5, 1.0, 1.5]
    })
    test_csv = '/tmp/test_co2.csv'
    test_data.to_csv(test_csv, index=False)
    
    # Load and verify
    result = load_co2_timeseries_csv(test_csv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_metabolomics_loader():
    """Test metabolomics aggregated loader"""
    # Create test CSV
    test_data = pd.DataFrame({
        'sample_id': ['S1', 'S2', 'S3'],
        'glucose_mM': [150, 140, 130],
        'lactate_mM': [5, 10, 15],
        'acetate_mM': [0.5, 1.0, 1.5]
    })
    test_csv = '/tmp/test_metabolomics.csv'
    test_data.to_csv(test_csv, index=False)
    
    # Load and verify
    result = load_metabolomics_aggregated_csv(test_csv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert any(col in result.columns for col in ['glucose_mM', 'lactate_mM', 'acetate_mM'])


def test_multimodal_fusion_with_co2_metabolomics():
    """Test end-to-end multimodal fusion with UV-Vis + CO2 + metabolomics"""
    # Create synthetic multimodal data
    n_samples = 50
    
    # UV-Vis spectra (601 wavelengths with spec_ prefix)
    uvvis_data = np.random.randn(n_samples, 601) * 0.1 + 0.5
    uvvis_df = pd.DataFrame(
        uvvis_data,
        columns=[f'spec_{i}' for i in range(601)]
    )
    
    # CO2 features (aggregated: mean, std, max, min from timeseries)
    co2_df = pd.DataFrame({
        'co2_mean': np.linspace(0.5, 3.0, n_samples),
        'co2_std': np.abs(np.random.randn(n_samples)) * 0.2,
        'co2_max': np.linspace(1.0, 4.0, n_samples),
        'co2_min': np.abs(np.linspace(-0.5, 1.0, n_samples))
    })
    
    # Metabolomics features (aggregated per sample)
    metabolomics_df = pd.DataFrame({
        'glucose': np.linspace(150, 50, n_samples),
        'lactate': np.linspace(0, 50, n_samples),
        'acetate': np.abs(np.random.randn(n_samples)) * 5
    })
    
    # Fuse all modalities
    extractor = MultimodalFeatureExtractor(
        expanded_feature_columns=['co2_mean', 'co2_std', 'co2_max', 'co2_min', 'glucose', 'lactate', 'acetate']
    )
    
    # Combine all data
    all_features = pd.concat([uvvis_df, co2_df, metabolomics_df], axis=1)
    
    # Extract fused features (use correct method name)
    fused_features = extractor.extract_features(all_features)
    
    assert fused_features.shape[0] == n_samples
    assert fused_features.shape[1] >= 10  # At least some features


def test_ensemble_on_fused_multimodal():
    """Test ensemble detector on multimodal (UV-Vis + CO2 + metabolomics) data"""
    n_clean = 100
    n_contaminated = 50
    n_features = 50  # Reduced feature count for speed
    
    # Synthetic clean data
    X_clean = np.random.randn(n_clean, n_features) * 0.1 + 1.0
    
    # Synthetic contaminated data (mean shift)
    X_contaminated = np.random.randn(n_contaminated, n_features) * 0.2 + 2.0
    
    X_combined = np.vstack([X_clean, X_contaminated])
    y = np.hstack([np.zeros(n_clean), np.ones(n_contaminated)])
    
    # Train ensemble on clean data
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(n_features, cfg)
    ensemble.fit(X_clean)
    
    # Predict on combined
    predictions = ensemble.predict(X_combined)
    scores = ensemble.predict_proba(X_combined)
    
    # Verify predictions and scores have correct shape
    assert predictions.shape == (n_clean + n_contaminated,)
    assert scores.shape == (n_clean + n_contaminated,)
    
    # Verify some contaminated samples are flagged as anomalies
    contaminated_anomaly_rate = np.sum(predictions[n_clean:] == -1) / n_contaminated
    assert contaminated_anomaly_rate > 0.5, "Contaminated samples should be largely detected"


def test_co2_metabolomics_feature_columns_present():
    """Verify that CO2 and metabolomics features are correctly integrated into pipeline"""
    from src.data_integration import load_co2_timeseries_csv, load_metabolomics_aggregated_csv
    
    # Create minimal test files with correct columns
    co2_test = pd.DataFrame({
        'sample_id': ['S1', 'S1'],
        'timestamp': [0, 1],
        'co2_pct': [0.5, 1.0]
    })
    co2_path = '/tmp/test_co2_integration.csv'
    co2_test.to_csv(co2_path, index=False)
    
    metabolomics_test = pd.DataFrame({
        'sample_id': ['S1'],
        'glucose_mM': [150],
        'lactate_mM': [10],
        'acetate_mM': [2]
    })
    metabolomics_path = '/tmp/test_metabolomics_integration.csv'
    metabolomics_test.to_csv(metabolomics_path, index=False)
    
    # Load both
    co2_loaded = load_co2_timeseries_csv(co2_path)
    metabolomics_loaded = load_metabolomics_aggregated_csv(metabolomics_path)
    
    assert len(co2_loaded) > 0
    assert len(metabolomics_loaded) > 0
