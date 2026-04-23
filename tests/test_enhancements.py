"""
Tests for Enhancements in Biopharmaceutical Contamination Detection System
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_simulation import UVVisSpectraGenerator, ContaminantType, ProcessConditions
from feature_extraction import SpectralFeatureExtractor
from anomaly_detection import ModelConfig, EnsembleAnomalyDetector, IsolationForestDetector
from validation import ValidationPipeline, ValidationConfig
from mh_ddpm import DDPMConfig, FeatureDiffusionModel

class TestEnsembleWeighting:
    """Tests for ensemble weighting logic"""
    
    def test_ensemble_weight_calculation(self):
        generator = UVVisSpectraGenerator(seed=42)
        df = generator.generate_dataset(n_clean=100, seed=42)
        
        extractor = SpectralFeatureExtractor()
        features_df = extractor.extract_all_features(df)
        feature_cols = [c for c in features_df.columns if c.startswith(('abs_'))]
        X_nominal = features_df[feature_cols].values
        
        config = ModelConfig(ae_epochs=10)
        ensemble = EnsembleAnomalyDetector(X_nominal.shape[1], config)
        ensemble.fit(X_nominal)
        
        # Verify weights are calculated and normalized
        assert len(ensemble.weights) == 3
        assert pytest.approx(sum(ensemble.weights.values())) == 1.0
        assert all(w > 0 for w in ensemble.weights.values())
        assert len(ensemble.score_stats) == 3

class TestBenignDrift:
    """Tests for benign process drift simulation"""
    
    def test_benign_drift_prediction(self):
        generator = UVVisSpectraGenerator(seed=42)
        df_train = generator.generate_dataset(n_clean=100, seed=42)
        
        extractor = SpectralFeatureExtractor()
        features_df = extractor.extract_all_features(df_train)
        feature_cols = [c for c in features_df.columns if c.startswith(('abs_'))]
        X_train = features_df[feature_cols].values
        
        config = ModelConfig()
        detector = IsolationForestDetector(config)
        detector.fit(X_train)
        
        # Evaluate benign drift
        results = generator.evaluate_benign_drift(detector, extractor=extractor, n_samples=20)
        
        assert results['n_samples'] == 20
        assert 'predicted_normal_rate' in results
        # Ideally, normal rate should be high for benign drift
        assert results['predicted_normal_rate'] >= 0.5 # Conservative check

class TestLODLogic:
    """Tests for LOD and 10 CFU metrics"""
    
    def test_lod_and_10cfu_metrics(self):
        generator = UVVisSpectraGenerator(seed=42)
        df = generator.generate_dataset(n_clean=50, n_contaminated_per_type=10, 
                                       inoculum_levels=[10, 50, 100], seed=42)
        
        extractor = SpectralFeatureExtractor()
        features_df = extractor.extract_all_features(df)
        feature_cols = [c for c in features_df.columns if c.startswith(('abs_'))]
        
        X = features_df[feature_cols].values
        y = features_df['label'].values
        inoculum = features_df['inoculum_level'].values
        
        X_train = X[y == 0]
        
        config = ModelConfig()
        detector = IsolationForestDetector(config)
        detector.fit(X_train)
        
        pipeline = ValidationPipeline(ValidationConfig(target_detection_limit=10))
        results = pipeline.validate_detector(detector, X, y, inoculum)
        
        assert 'detection_limit' in results
        assert 'detection_limit_90' in results['detection_limit']
        assert 'confusion_10cfu' in results
        
        conf = results['confusion_10cfu']
        if conf: # Might be empty if no 10 CFU samples in dataset (but we added them)
            assert 'tp' in conf
            assert 'sensitivity' in conf
            assert 'fpr' in conf

class TestFeatureDiffusion:
    """Tests for FeatureDiffusionModel"""
    
    def test_feature_diffusion_flow(self):
        n_features = 10
        n_samples = 50
        X = np.random.randn(n_samples, n_features)
        c_types = np.random.randint(0, 6, n_samples)
        i_levels = np.random.randint(0, 7, n_samples)
        
        config = DDPMConfig(input_dim=n_features, n_timesteps=10, epochs=5, batch_size=16)
        model = FeatureDiffusionModel(config)
        
        # Fit
        model.fit_features(X, c_types, i_levels, verbose=False)
        
        # Sample
        gen_features = model.sample_features(n_samples=5, contaminant_type=1, inoculum_level=2)
        
        assert gen_features.shape == (5, n_features)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
