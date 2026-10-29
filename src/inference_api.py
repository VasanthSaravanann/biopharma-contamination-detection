"""Model serialization and inference API for deployment.

Provides:
- Serialization of trained ensemble to disk (.pkl or .pt)
- Inference API with optional latency contract checks
- Model versioning and metadata
"""
import numpy as np
import pickle
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from anomaly_detection import EnsembleAnomalyDetector, ModelConfig


@dataclass
class ModelMetadata:
    """Metadata for serialized model"""
    model_type: str = "EnsembleAnomalyDetector"
    version: str = "1.0.0"
    input_dim: int = 601
    contamination: float = 0.01
    ensemble_weights: Dict[str, float] = None
    training_timestamp: str = ""
    latency_target_ms: float = 42.0
    latency_tolerance_ms: float = 6.0
    
    def to_dict(self):
        return asdict(self)


class ModelSerializationAPI:
    """API for serializing and loading ensemble models"""
    
    def __init__(self, model: EnsembleAnomalyDetector, metadata: Optional[ModelMetadata] = None):
        self.model = model
        self.metadata = metadata or ModelMetadata(
            input_dim=model.input_dim,
            ensemble_weights=dict(model.weights)
        )
    
    def save(self, path: str, save_metadata: bool = True):
        """Save model to disk.
        
        Args:
            path: Filepath (e.g., 'models/ensemble_v1.pkl')
            save_metadata: Also save metadata as JSON
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if save_metadata:
            metadata_path = path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
        
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(path: str) -> EnsembleAnomalyDetector:
        """Load model from disk.
        
        Args:
            path: Filepath
            
        Returns:
            Loaded EnsembleAnomalyDetector
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {path}")
        return model


class InferenceAPI:
    """High-level inference API with latency monitoring"""
    
    def __init__(self, model: EnsembleAnomalyDetector, 
                 latency_target_ms: float = 42.0,
                 latency_tolerance_ms: float = 6.0):
        self.model = model
        self.latency_target_ms = latency_target_ms
        self.latency_tolerance_ms = latency_tolerance_ms
        self.latencies = []
    
    def predict(self, X: np.ndarray, enforce_latency_contract: bool = False) -> Tuple[np.ndarray, float]:
        """Inference with optional latency contract enforcement.
        
        Args:
            X: Input data (n_samples, n_features)
            enforce_latency_contract: If True, raise error if latency exceeds target + tolerance
            
        Returns:
            (predictions, latency_ms)
            
        Raises:
            RuntimeError: If latency contract violated
        """
        t0 = time.perf_counter()
        predictions = self.model.predict(X)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        self.latencies.append(latency_ms)
        
        # Check latency contract
        if enforce_latency_contract:
            max_allowed = self.latency_target_ms + self.latency_tolerance_ms
            if latency_ms > max_allowed:
                raise RuntimeError(
                    f"Latency {latency_ms:.2f} ms exceeds contract "
                    f"({self.latency_target_ms}±{self.latency_tolerance_ms} ms)"
                )
        
        return predictions, latency_ms
    
    def predict_proba(self, X: np.ndarray, enforce_latency_contract: bool = False) -> Tuple[np.ndarray, float]:
        """Anomaly scores with optional latency contract enforcement.
        
        Args:
            X: Input data (n_samples, n_features)
            enforce_latency_contract: If True, raise error if latency exceeds target + tolerance
            
        Returns:
            (scores, latency_ms)
        """
        t0 = time.perf_counter()
        scores = self.model.predict_proba(X)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        self.latencies.append(latency_ms)
        
        if enforce_latency_contract:
            max_allowed = self.latency_target_ms + self.latency_tolerance_ms
            if latency_ms > max_allowed:
                raise RuntimeError(
                    f"Latency {latency_ms:.2f} ms exceeds contract "
                    f"({self.latency_target_ms}±{self.latency_tolerance_ms} ms)"
                )
        
        return scores, latency_ms
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Return latency statistics from inference calls"""
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'n_inferences': len(self.latencies)
        }
    
    def reset_latency_stats(self):
        """Clear latency history"""
        self.latencies = []
