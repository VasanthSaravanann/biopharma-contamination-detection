"""
Tests for Biopharmaceutical Contamination Detection System
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_simulation import UVVisSpectraGenerator, ContaminantType, ProcessConditions
from feature_extraction import SpectralFeatureExtractor, FeatureConfig
from anomaly_detection import ModelConfig, IsolationForestDetector


class TestDataSimulation:
    """Tests for data simulation module"""
    
    def test_generator_initialization(self):
        """Test generator initializes correctly"""
        generator = UVVisSpectraGenerator(seed=42)
        assert generator is not None
        assert len(generator.spectral_params.wavelengths) == 601
    
    def test_clean_spectrum_generation(self):
        """Test clean spectrum generation"""
        generator = UVVisSpectraGenerator(seed=42)
        wavelengths, absorbance = generator.generate_spectrum(
            ContaminantType.CLEAN
        )
        
        assert len(wavelengths) == 601
        assert len(absorbance) == 601
        assert np.all(absorbance >= 0)
        assert wavelengths[0] == 200
        assert wavelengths[-1] == 800
    
    def test_contaminated_spectrum_generation(self):
        """Test contaminated spectrum generation"""
        generator = UVVisSpectraGenerator(seed=42)
        wavelengths, absorbance = generator.generate_spectrum(
            ContaminantType.BACTERIA_E_COLI,
            inoculum_level=100
        )
        
        assert len(wavelengths) == 601
        assert len(absorbance) == 601
        assert np.all(absorbance >= 0)
    
    def test_process_conditions_variation(self):
        """Test process condition variation"""
        proc_cond = ProcessConditions()
        varied = proc_cond.add_variation()
        
        assert varied.temperature != proc_cond.temperature or \
               varied.ph != proc_cond.ph or \
               varied.dissolved_oxygen != proc_cond.dissolved_oxygen
    
    def test_dataset_generation(self):
        """Test full dataset generation"""
        generator = UVVisSpectraGenerator(seed=42)
        df = generator.generate_dataset(
            n_clean=50,
            n_contaminated_per_type=10,
            seed=42
        )
        
        expected_samples = 50 + 10 * 6  # clean + 6 contaminant types
        assert len(df) == expected_samples
        assert 'label' in df.columns
        assert 'contaminant_type' in df.columns


class TestFeatureExtraction:
    """Tests for feature extraction module"""
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly"""
        extractor = SpectralFeatureExtractor()
        assert extractor is not None
        assert len(extractor.wavelengths) == 601
    
    def test_single_spectrum_extraction(self):
        """Test feature extraction from single spectrum"""
        generator = UVVisSpectraGenerator(seed=42)
        _, absorbance = generator.generate_spectrum(ContaminantType.CLEAN)
        
        # Create mock row
        row = pd.Series({f'abs_{int(w)}': a for w, a in zip(
            np.arange(200, 801, 1), absorbance
        )})
        
        extractor = SpectralFeatureExtractor()
        features = extractor.extract_single_spectrum(row)
        
        assert 'abs_260' in features
        assert 'abs_280' in features
        assert 'n_peaks' in features
        assert 'scattering_exponent' in features
    
    def test_dataset_feature_extraction(self):
        """Test feature extraction from dataset"""
        generator = UVVisSpectraGenerator(seed=42)
        df = generator.generate_dataset(n_clean=20, seed=42)
        
        extractor = SpectralFeatureExtractor()
        features_df = extractor.extract_all_features(df)
        
        assert len(features_df) == len(df)
        assert 'spectrum_id' in features_df.columns


class TestAnomalyDetection:
    """Tests for anomaly detection module"""
    
    def test_isolation_forest_training(self):
        """Test Isolation Forest training"""
        # Generate clean training data
        generator = UVVisSpectraGenerator(seed=42)
        df = generator.generate_dataset(n_clean=100, seed=42)
        
        # Extract simple features
        wavelength_cols = [f'abs_{int(w)}' for w in range(200, 801, 10)]
        X = df[wavelength_cols].values
        
        # Train detector
        config = ModelConfig(iforest_n_estimators=10)
        detector = IsolationForestDetector(config)
        detector.fit(X)
        
        assert detector.is_fitted
        assert detector.threshold is not None
    
    def test_isolation_forest_prediction(self):
        """Test Isolation Forest prediction"""
        generator = UVVisSpectraGenerator(seed=42)
        df_train = generator.generate_dataset(n_clean=100, seed=42)
        df_test = generator.generate_dataset(n_clean=20, n_contaminated_per_type=10, seed=43)
        
        wavelength_cols = [f'abs_{int(w)}' for w in range(200, 801, 10)]
        X_train = df_train[wavelength_cols].values
        X_test = df_test[wavelength_cols].values
        y_test = df_test['label'].values
        
        config = ModelConfig(iforest_n_estimators=10)
        detector = IsolationForestDetector(config)
        detector.fit(X_train)
        
        predictions = detector.predict(X_test)
        scores = detector.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert set(predictions).issubset({-1, 1})


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to prediction"""
        # Generate data
        generator = UVVisSpectraGenerator(seed=42)
        df = generator.generate_dataset(n_clean=100, n_contaminated_per_type=20, seed=42)
        
        # Extract features
        extractor = SpectralFeatureExtractor()
        features_df = extractor.extract_all_features(df)
        
        # Prepare data
        feature_cols = [c for c in features_df.columns 
                       if c.startswith(('abs_', 'a260', 'uv_', 'peak_', 'scattering'))]
        X = features_df[feature_cols].fillna(0).values
        y = features_df['label'].values
        
        X_clean = X[y == 0]
        X_test = X
        y_test = y
        
        # Train and evaluate
        config = ModelConfig(iforest_n_estimators=10)
        detector = IsolationForestDetector(config)
        detector.fit(X_clean)
        
        scores = detector.predict_proba(X_test)
        
        # Basic sanity checks
        assert len(scores) == len(y_test)
        # Scores can be negative (raw anomaly scores from decision function)
        # Check that scores are finite
        assert np.all(np.isfinite(scores))

        # Check that contaminated samples have higher scores on average
        clean_scores = scores[y_test == 0]
        cont_scores = scores[y_test == 1]

        # Contaminated should have higher anomaly scores (not always, but on average)
        assert np.mean(cont_scores) > np.mean(clean_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
