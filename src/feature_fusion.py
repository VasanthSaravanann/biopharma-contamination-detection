import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import logging

class MultimodalFeatureExtractor:
    """Extracts and scales features from UV-Vis spectral and AMBR sensor process variables."""
    
    def __init__(self, mode: str = 'fused', expanded_feature_columns=None):
        self.mode = mode
        self.spectral_scaler = StandardScaler()
        self.process_scaler = RobustScaler()
        self.expanded_feature_columns = list(expanded_feature_columns or [])
        self.expanded_scaler = RobustScaler()

    @staticmethod
    def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce to numeric and remove invalid values before scaling."""
        clean = df.apply(pd.to_numeric, errors='coerce')
        clean = clean.replace([np.inf, -np.inf], np.nan)
        # Clip extreme magnitudes to keep robust scaling numerically stable.
        clean = clean.clip(lower=-1e12, upper=1e12)
        return clean.fillna(0.0)
        
    def extract_features(self, df: pd.DataFrame):
        spec_cols = [c for c in df.columns if c.startswith('spec_')]
        # Exclude label, metadata, and timestamp-related columns
        exclude_patterns = ['label', 'timestamp', 'time', 'organism', 'cfu', 'injection', 'contamination', 'start', 'end', 'duration']
        process_cols = [c for c in df.columns 
                       if not c.startswith('spec_') 
                       and not any(pattern.lower() in c.lower() for pattern in exclude_patterns)]
        expanded_cols = [
            c for c in self.expanded_feature_columns
            if c in df.columns and c not in spec_cols and c not in process_cols
        ]
        
        logging.info(
            f"Preflight: Found {len(spec_cols)} spectral, {len(process_cols)} process, {len(expanded_cols)} expanded channels."
        )
        
        if self.mode == 'fused':
            if len(spec_cols) == 0 or (len(process_cols) == 0 and len(expanded_cols) == 0):
                raise ValueError(
                    f"Schema Error: Fused mode requires spectra and process or expanded features. Found {len(spec_cols)} spectra, {len(process_cols)} process, {len(expanded_cols)} expanded."
                )
        
        features = []
        if spec_cols:
            X_spec_df = self._sanitize_frame(df[spec_cols])
            X_spec = self.spectral_scaler.fit_transform(X_spec_df)
            features.append(X_spec)
        
        # Process columns: handle inf/nan values
        X_proc_raw = df[process_cols].fillna(0).copy()
        # Replace inf values with large finite values
        X_proc_raw = X_proc_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Clip extreme values
        X_proc_raw = X_proc_raw.clip(-1e6, 1e6)
        
        X_proc = self.process_scaler.fit_transform(X_proc_raw)
        features.append(X_proc)

        if expanded_cols:
            X_expanded_raw = self._sanitize_frame(df[expanded_cols])
            X_expanded = self.expanded_scaler.fit_transform(X_expanded_raw)
            features.append(X_expanded)
        
        return np.hstack(features)

def evaluate_roc_gate(detector, X_test, y_test):
    """Compute ROC-AUC for the detector on test data."""
    scores = detector.predict_proba(X_test)
    auc = roc_auc_score(y_test, scores)
    logging.info(f"Computed ROC-AUC: {auc:.4f}")
    return auc
