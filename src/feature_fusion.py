import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import logging

class MultimodalFeatureExtractor:
    """Extracts and scales features from UV-Vis spectral and AMBR sensor process variables."""
    
    def __init__(self, mode: str = 'fused'):
        self.mode = mode
        self.spectral_scaler = StandardScaler()
        self.process_scaler = RobustScaler()
        
    def extract_features(self, df: pd.DataFrame):
        spec_cols = [c for c in df.columns if c.startswith('spec_')]
        # Exclude label, metadata, and timestamp-related columns
        exclude_patterns = ['label', 'timestamp', 'time', 'organism', 'cfu', 'injection', 'contamination', 'start', 'end', 'duration']
        process_cols = [c for c in df.columns 
                       if not c.startswith('spec_') 
                       and not any(pattern.lower() in c.lower() for pattern in exclude_patterns)]
        
        logging.info(f"Preflight: Found {len(spec_cols)} spectral, {len(process_cols)} process channels.")
        
        if self.mode == 'fused':
            if len(spec_cols) == 0 or len(process_cols) == 0:
                raise ValueError(f"Schema Error: Fused mode requires spectra and process features. Found {len(spec_cols)} spectra, {len(process_cols)} process.")
        
        features = []
        if spec_cols:
            X_spec = self.spectral_scaler.fit_transform(df[spec_cols].fillna(0))
            features.append(X_spec)
        
        # Process columns: handle inf/nan values
        X_proc_raw = df[process_cols].fillna(0).copy()
        # Replace inf values with large finite values
        X_proc_raw = X_proc_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Clip extreme values
        X_proc_raw = X_proc_raw.clip(-1e6, 1e6)
        
        X_proc = self.process_scaler.fit_transform(X_proc_raw)
        features.append(X_proc)
        
        return np.hstack(features)

def evaluate_roc_gate(detector, X_test, y_test):
    """Compute ROC-AUC for the detector on test data."""
    scores = detector.predict_proba(X_test)
    auc = roc_auc_score(y_test, scores)
    logging.info(f"Computed ROC-AUC: {auc:.4f}")
    return auc
