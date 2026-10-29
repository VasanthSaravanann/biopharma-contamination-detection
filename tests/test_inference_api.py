"""Unit tests for model serialization and inference API"""
import numpy as np
import pytest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_simulation import UVVisSpectraGenerator
from feature_extraction import SpectralFeatureExtractor
from anomaly_detection import ModelConfig, EnsembleAnomalyDetector
from inference_api import ModelSerializationAPI, InferenceAPI, ModelMetadata


def test_model_serialization_save_load():
    """Test save/load cycle for ensemble model"""
    # Train a small ensemble
    gen = UVVisSpectraGenerator(seed=42)
    df = gen.generate_dataset(n_clean=100, seed=42)
    
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].values
    
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X)
    
    # Serialize
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        
        api = ModelSerializationAPI(ensemble)
        api.save(str(model_path))
        
        # Verify files were created
        assert model_path.exists()
        assert (model_path.parent / "test_model_metadata.json").exists()
        
        # Load and verify
        loaded_model = ModelSerializationAPI.load(str(model_path))
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')


def test_inference_api_predict_with_latency():
    """Test inference API returns both predictions and latency"""
    # Train small ensemble
    gen = UVVisSpectraGenerator(seed=42)
    df = gen.generate_dataset(n_clean=100, seed=42)
    
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].values
    
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X)
    
    # Create inference API
    inference_api = InferenceAPI(ensemble)
    
    # Test predict
    predictions, latency_ms = inference_api.predict(X[:10])
    assert predictions.shape == (10,)
    assert isinstance(latency_ms, float)
    assert latency_ms > 0


def test_inference_api_predict_proba_with_latency():
    """Test inference API scores with latency"""
    gen = UVVisSpectraGenerator(seed=42)
    df = gen.generate_dataset(n_clean=100, seed=42)
    
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].values
    
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X)
    
    inference_api = InferenceAPI(ensemble)
    scores, latency_ms = inference_api.predict_proba(X[:10])
    
    assert scores.shape == (10,)
    assert isinstance(latency_ms, float)


def test_latency_contract_enforcement():
    """Test that latency contract is enforced when enabled"""
    gen = UVVisSpectraGenerator(seed=42)
    df = gen.generate_dataset(n_clean=100, seed=42)
    
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].values
    
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X)
    
    # Create inference API with very strict latency target (should fail)
    inference_api = InferenceAPI(ensemble, latency_target_ms=0.001, latency_tolerance_ms=0.001)
    
    # This should raise RuntimeError due to latency contract violation
    with pytest.raises(RuntimeError, match="Latency.*exceeds contract"):
        inference_api.predict(X[:10], enforce_latency_contract=True)


def test_latency_stats_collection():
    """Test latency statistics accumulation"""
    gen = UVVisSpectraGenerator(seed=42)
    df = gen.generate_dataset(n_clean=100, seed=42)
    
    extractor = SpectralFeatureExtractor()
    features = extractor.extract_all_features(df)
    feature_cols = [c for c in features.columns if c.startswith('abs_')]
    X = features[feature_cols].values
    
    cfg = ModelConfig(ae_epochs=1)
    ensemble = EnsembleAnomalyDetector(X.shape[1], cfg)
    ensemble.fit(X)
    
    inference_api = InferenceAPI(ensemble)
    
    # Run multiple inferences
    for _ in range(5):
        inference_api.predict(X[:10])
    
    stats = inference_api.get_latency_stats()
    assert 'mean_ms' in stats
    assert 'std_ms' in stats
    assert 'min_ms' in stats
    assert 'max_ms' in stats
    assert stats['n_inferences'] == 5


def test_metadata_initialization():
    """Test model metadata creation"""
    metadata = ModelMetadata(
        input_dim=601,
        ensemble_weights={'iforest': 0.3, 'ocsvm': 0.5, 'autoencoder': 0.2},
        training_timestamp="2026-05-13T10:00:00Z"
    )
    
    metadata_dict = metadata.to_dict()
    assert metadata_dict['input_dim'] == 601
    assert metadata_dict['model_type'] == 'EnsembleAnomalyDetector'
    assert 'ensemble_weights' in metadata_dict
