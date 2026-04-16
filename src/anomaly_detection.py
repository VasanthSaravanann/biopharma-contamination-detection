"""
Anomaly Detection Models for Contamination Detection

Implements unsupervised anomaly detection models trained ONLY on nominal (clean) data:
- Isolation Forest
- Deep Autoencoder
- One-Class SVM

All models learn the distribution of clean spectra and flag deviations as anomalies.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for anomaly detection models"""
    # Common
    random_state: int = 42
    contamination: float = 0.01  # Expected proportion of anomalies
    
    # Isolation Forest
    iforest_n_estimators: int = 200
    iforest_max_samples: Union[int, float] = 'auto'
    iforest_max_features: float = 1.0
    
    # One-Class SVM
    ocsvm_kernel: str = 'rbf'
    ocsvm_gamma: str = 'auto'
    ocsvm_nu: float = 0.05
    
    # Autoencoder
    ae_hidden_layers: List[int] = None
    ae_latent_dim: int = 32
    ae_activation: str = 'relu'
    ae_dropout: float = 0.2
    ae_learning_rate: float = 0.001
    ae_batch_size: int = 64
    ae_epochs: int = 100
    ae_early_stopping_patience: int = 10
    ae_weight_decay: float = 1e-5
    
    def __post_init__(self):
        if self.ae_hidden_layers is None:
            self.ae_hidden_layers = [256, 128, 64]


class AnomalyDetectionBase:
    """Base class for anomaly detection models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.threshold = None
    
    def preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess input data"""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit model on nominal data only"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 for anomaly, 1 for normal)"""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (higher = more anomalous)"""
        raise NotImplementedError
    
    def set_threshold(self, X: np.ndarray, contamination: Optional[float] = None):
        """Set decision threshold based on training data"""
        scores = self.predict_proba(X)
        if contamination is None:
            contamination = self.config.contamination
        self.threshold = np.percentile(scores, 100 * (1 - contamination))
    
    def save(self, path: str):
        """Save model to disk"""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load model from disk"""
        raise NotImplementedError


class IsolationForestDetector(AnomalyDetectionBase):
    """
    Isolation Forest for anomaly detection.
    
    Efficient algorithm that isolates anomalies by randomly selecting features
    and split values. Anomalies are easier to isolate (shorter path length).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = IsolationForest(
            n_estimators=config.iforest_n_estimators,
            max_samples=config.iforest_max_samples,
            max_features=config.iforest_max_features,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit Isolation Forest on nominal data"""
        X_scaled = self.preprocess(X, fit=True)
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        self.set_threshold(X_scaled); print("DEBUG: is_fitted is", self.is_fitted)
        print("DEBUG: OCSVM fitted")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Returns negative score (more negative = more anomalous)
        Converted to positive scores for consistency.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.preprocess(X)
        # Decision function: negative for anomalies, positive for normal
        scores = -self.model.decision_function(X_scaled)
        return scores
    
    def get_path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Get average path lengths for samples"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.preprocess(X)
        return self.model.estimators_[0].path_length(X_scaled)
    
    def save(self, path: str):
        """Save model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'config': self.config
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str):
        """Load model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.config = model_data['config']
        self.is_fitted = True
        self.set_threshold(X_scaled); print("DEBUG: is_fitted is", self.is_fitted)
        print("DEBUG: OCSVM fitted")


class OneClassSVMDetector(AnomalyDetectionBase):
    """
    One-Class SVM for anomaly detection.
    
    Learns a decision boundary that separates nominal data from the origin
    in feature space. Uses RBF kernel for non-linear boundaries.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = OneClassSVM(
            kernel=config.ocsvm_kernel,
            gamma=config.ocsvm_gamma,
            nu=config.ocsvm_nu,
            cache_size=1000
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit One-Class SVM on nominal data"""
        X_scaled = self.preprocess(X, fit=True)
        
        # Subsample for large datasets (OCSVM is O(n^2) to O(n^3))
        if len(X_scaled) > 5000:
            indices = np.random.choice(len(X_scaled), 5000, replace=False)
            X_subset = X_scaled[indices]
        else:
            X_subset = X_scaled
        
        self.model.fit(X_subset)
        
        self.is_fitted = True
        self.set_threshold(X_scaled); print("DEBUG: is_fitted is", self.is_fitted)
        print("DEBUG: OCSVM fitted")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Distance from decision boundary (more negative = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.preprocess(X)
        scores = -self.model.decision_function(X_scaled)
        return scores
    
    def save(self, path: str):
        """Save model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'config': self.config
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str):
        """Load model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.config = model_data['config']
        self.is_fitted = True
        self.set_threshold(X_scaled); print("DEBUG: is_fitted is", self.is_fitted)
        print("DEBUG: OCSVM fitted")


class Conv1DAutoencoder(nn.Module):
    """
    Convolutional 1D Autoencoder for anomaly detection.
    
    Learns to reconstruct nominal spectra using 1D convolutions
    to capture spatial (wavelength) correlations.
    """
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = config.ae_latent_dim
        
        # Build encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dummy pass to compute dynamic flatten size
        dummy_input = torch.zeros(1, 1, input_dim)
        flatten_size = self.encoder_conv(dummy_input).shape[1]
        
        self.encoder_linear = nn.Linear(flatten_size, self.latent_dim)
        
        # Build decoder
        self.decoder_input = nn.Linear(self.latent_dim, flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = x.view(-1, 1, self.input_dim)
        z = self.encoder_linear(self.encoder_conv(x))
        
        h = self.decoder_input(z)
        h = h.view(-1, 64, h.shape[1] // 64)
        
        x_recon = self.decoder(h)
        
        # Final resizing to handle rounding drift
        if x_recon.shape[2] != self.input_dim:
            x_recon = nn.functional.interpolate(x_recon, size=self.input_dim, mode='linear', align_corners=False)
            
        return x_recon.view(-1, self.input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        x = x.view(-1, 1, self.input_dim)
        return self.encoder(x)


class AutoencoderDetector(AnomalyDetectionBase):
    """
    Deep Autoencoder-based anomaly detector.
    
    Trains on nominal data only. Anomalies are detected based on
    reconstruction error. Supports both MLP and Conv1D architectures.
    """
    
    def __init__(self, input_dim: int, config: ModelConfig,
                 use_conv: bool = True, device: Optional[str] = None):
        super().__init__(config)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_dim = input_dim
        self.use_conv = use_conv
        
        if use_conv:
            self.model = Conv1DAutoencoder(input_dim, config).to(self.device)
        else:
            # Fallback or keep MLP version if needed, but we prefer Conv1D now
            self.model = DeepAutoencoder(input_dim, config).to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.ae_learning_rate,
            weight_decay=self.config.ae_weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    
    def _reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss (MSE + L1)"""
        mse_loss = nn.MSELoss()(x_recon, x)
        l1_loss = nn.L1Loss()(x_recon, x) * 0.1  # Small L1 component
        return mse_loss + l1_loss
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            val_split: float = 0.1, verbose: bool = True):
        """
        Fit autoencoder on nominal data.
        
        Args:
            X: Training data (nominal only)
            y: Not used (unsupervised)
            val_split: Validation split fraction
            verbose: Print training progress
        """
        # Preprocess
        X_scaled = self.preprocess(X, fit=True)
        
        # Split data
        X_train, X_val = train_test_split(
            X_scaled, test_size=val_split, random_state=self.config.random_state
        )
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.ae_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.ae_batch_size, shuffle=False
        )
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.ae_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                self.optimizer.zero_grad()
                x_recon = self.model(x)
                loss = self._reconstruction_loss(x, x_recon)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item() * len(x)
            
            train_loss /= len(X_train)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    x_recon = self.model(x)
                    loss = self._reconstruction_loss(x, x_recon)
                    val_loss += loss.item() * len(x)
            
            val_loss /= len(X_val)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.ae_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if patience_counter >= self.config.ae_early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Set threshold based on training reconstruction errors
        
        self.is_fitted = True
        self.set_threshold(X_scaled); print("DEBUG: is_fitted is", self.is_fitted)
        print("DEBUG: OCSVM fitted")
        
        return self
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict anomaly labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        scores = self.predict_proba(X)
        if threshold is None:
            threshold = self.threshold
        
        # Higher score = more anomalous
        predictions = np.where(scores > threshold, -1, 1)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (reconstruction error).
        
        Higher error = more anomalous
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.preprocess(X)
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(X_scaled).to(self.device)
            x_recon = self.model(x)
            
            # Per-sample reconstruction error (MSE)
            errors = torch.mean((x - x_recon) ** 2, dim=1)
        
        return errors.cpu().numpy()
    
    def get_reconstruction(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get original and reconstructed spectra"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.preprocess(X)
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(X_scaled).to(self.device)
            x_recon = self.model(x)
        
        # Inverse transform
        x_recon_np = x_recon.cpu().numpy()
        x_orig = self.scaler.inverse_transform(X_scaled)
        x_recon_orig = self.scaler.inverse_transform(x_recon_np)
        
        return x_orig, x_recon_orig
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.preprocess(X)
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(X_scaled).to(self.device)
            z = self.model.encode(x)
        
        return z.cpu().numpy()
    
    def save(self, path: str):
        """Save model"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'config': self.config,
            'input_dim': self.input_dim,
            'training_history': self.training_history
        }
        torch.save(model_data, path)
    
    def load(self, path: str):
        """Load model"""
        model_data = torch.load(path, map_location=self.device)
        
        self.input_dim = model_data['input_dim']
        self.model = DeepAutoencoder(self.input_dim, model_data['config']).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.config = model_data['config']
        self.training_history = model_data['training_history']
        
        self._init_optimizer()
        self.is_fitted = True
        self.set_threshold(X_scaled); print("DEBUG: is_fitted is", self.is_fitted)
        print("DEBUG: OCSVM fitted")


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detectors.
    
    Combines predictions from Isolation Forest, One-Class SVM, and Autoencoder
    for improved robustness and accuracy.
    """
    
    def __init__(self, input_dim: int, config: ModelConfig,
                 use_conv: bool = True, device: Optional[str] = None):
        self.config = config
        self.device = device
        self.use_conv = use_conv
        
        self.detectors = {
            'iforest': IsolationForestDetector(config),
            'ocsvm': OneClassSVMDetector(config),
            'autoencoder': AutoencoderDetector(input_dim, config, use_conv, device)
        }
        
        self.weights = {'iforest': 1.0, 'ocsvm': 1.0, 'autoencoder': 2.0}
        self.ensemble_threshold = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all detectors on nominal data"""
        for name, detector in self.detectors.items():
            print(f"\nTraining {name}...")
            detector.fit(X, y)
        
        # Set ensemble threshold
        self._set_ensemble_threshold(X)
        
        return self
    
    def _set_ensemble_threshold(self, X: np.ndarray):
        """Set ensemble decision threshold"""
        scores = self.predict_proba(X)
        self.ensemble_threshold = np.percentile(
            scores, 100 * (1 - self.config.contamination)
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble anomaly scores"""
        all_scores = []
        total_weight = 0
        
        for name, detector in self.detectors.items():
            scores = detector.predict_proba(X)
            
            # Normalize scores to [0, 1] range
            scores_min = scores.min()
            scores_max = scores.max()
            if scores_max > scores_min:
                scores_norm = (scores - scores_min) / (scores_max - scores_min)
            else:
                scores_norm = np.zeros_like(scores)
            
            all_scores.append(scores_norm * self.weights[name])
            total_weight += self.weights[name]
        
        # Weighted average
        ensemble_scores = np.sum(all_scores, axis=0) / total_weight
        return ensemble_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels"""
        scores = self.predict_proba(X)
        return np.where(scores > self.ensemble_threshold, -1, 1)
    
    def save(self, base_path: str):
        """Save all detectors"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for name, detector in self.detectors.items():
            if name == 'autoencoder':
                path = base_path / f"{name}.pt"
            else:
                path = base_path / f"{name}.pkl"
            detector.save(path)
        
        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'threshold': self.ensemble_threshold,
            'config': self.config,
            'use_conv': self.use_conv
        }
        joblib.dump(metadata, base_path / "ensemble_metadata.pkl")
    
    def load(self, base_path: str):
        """Load all detectors"""
        base_path = Path(base_path)
        
        metadata = joblib.load(base_path / "ensemble_metadata.pkl")
        self.weights = metadata['weights']
        self.ensemble_threshold = metadata['threshold']
        self.config = metadata['config']
        self.use_conv = metadata.get('use_conv', True)
        
        for name, detector in self.detectors.items():
            if name == 'autoencoder':
                path = base_path / f"{name}.pt"
            else:
                path = base_path / f"{name}.pkl"
            detector.load(path)


def evaluate_detector(detector, X_test: np.ndarray, y_test: np.ndarray,
                      detector_name: str = "Detector") -> Dict:
    """
    Evaluate anomaly detector performance.
    
    Args:
        detector: Fitted anomaly detector
        X_test: Test data
        y_test: True labels (0 = normal, 1 = anomaly)
        detector_name: Name for reporting
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = detector.predict(X_test)
    scores = detector.predict_proba(X_test)
    
    # Convert labels: 0/1 to 1/-1 for sklearn
    y_true_sklearn = np.where(y_test == 1, -1, 1)
    
    # Calculate metrics
    metrics = {
        'detector': detector_name,
        'roc_auc': roc_auc_score(y_test, scores),
        'accuracy': np.mean(y_pred == y_true_sklearn),
        'precision': f1_score(y_true_sklearn, y_pred, pos_label=-1, average='binary'),
        'recall': None,  # Will be calculated below
        'f1': f1_score(y_true_sklearn, y_pred, pos_label=-1, average='binary'),
    }
    
    # Precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, scores)
    
    # Find optimal threshold
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    metrics['optimal_threshold'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0
    metrics['best_f1'] = f1_scores[optimal_idx]
    
    return metrics


if __name__ == "__main__":
    # Test anomaly detection models
    from data_simulation import UVVisSpectraGenerator, ContaminantType
    from feature_extraction import SpectralFeatureExtractor
    
    # Generate test data
    print("Generating test data...")
    generator = UVVisSpectraGenerator(seed=42)
    df = generator.generate_dataset(n_clean=500, n_contaminated_per_type=100, seed=42)
    
    # Extract features
    print("Extracting features...")
    extractor = SpectralFeatureExtractor()
    features_df = extractor.extract_all_features(df)
    
    # Prepare data
    feature_cols = [c for c in features_df.columns if c.startswith(('abs_', 'a260', 'uv_', 'vis_', 
                     'peak_', 'scattering', 'biomass', 'mean_', 'std_', 'integral_'))]
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Split data (train on clean only)
    X_clean = X[y == 0]
    X_test = X
    y_test = y
    
    print(f"\nTraining data: {len(X_clean)} clean samples")
    print(f"Test data: {len(X_test)} samples ({sum(y_test)} contaminated)")
    
    # Test Isolation Forest
    print("\n" + "="*50)
    print("Testing Isolation Forest...")
    config = ModelConfig()
    iforest = IsolationForestDetector(config)
    iforest.fit(X_clean)
    metrics = evaluate_detector(iforest, X_test, y_test, "Isolation Forest")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Test Autoencoder
    print("\n" + "="*50)
    print("Testing Deep Autoencoder...")
    ae = AutoencoderDetector(X_clean.shape[1], config)
    ae.fit(X_clean, verbose=True)
    metrics = evaluate_detector(ae, X_test, y_test, "Autoencoder")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
