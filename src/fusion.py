import pandas as pd
import numpy as np
import logging
from pathlib import Path
import glob

class DataFuser:
    """Fuses surrogate spectral contamination signatures into real process baselines."""
    
    def __init__(self, ambr_path: str, bacteria_path: str, seed: int = 42):
        self.ambr_path = Path(ambr_path)
        self.bacteria_path = Path(bacteria_path)
        self.rng = np.random.default_rng(seed)
        
    def get_surrogate_signature(self, contamination_type: str, cfu: str):
        """Loads spectral signature files."""
        # Robust pattern matching: try case-insensitive or partial matches
        patterns = [
            f"*{contamination_type}*{cfu}*",
            f"*{contamination_type.replace('_', '')}*{cfu}*",
            f"*{contamination_type}*"
        ]
        
        files = []
        for p in patterns:
            files = glob.glob(str(self.bacteria_path / "**" / p), recursive=True)
            if files: break
            
        if not files:
            raise FileNotFoundError(f"No surrogate found for {contamination_type} {cfu}")
        
        df = pd.read_csv(files[0])
        # Apply numeric conversion and drop non-numeric columns
        return df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    def fuse(self, ambr_df: pd.DataFrame, contamination_type: str, cfu: str, n_injections: int = 3):
        signature = self.get_surrogate_signature(contamination_type, cfu)
        # Explicit spectral channel list (must match training schema)
        sig_cols = [c for c in signature.columns if 'abs' in c.lower()][:5]
        
        fused_df = ambr_df.copy()
        for col in sig_cols:
            fused_df[f"spec_{col}"] = 0.0 # Deterministic init
        
        fused_df['label'] = 0 
        n_samples = len(fused_df)
        
        for _ in range(n_injections):
            duration = self.rng.integers(30, 90)
            if n_samples <= duration: continue
            
            start_idx = self.rng.integers(0, n_samples - duration)
            jitter = self.rng.normal(1.0, 0.05, duration)
            
            for col in sig_cols:
                sig_val = signature[col].mean()
                t = np.linspace(0, 1, duration)
                decay_profile = np.sin(t * np.pi) * jitter
                
                indices = fused_df.index[start_idx:start_idx+duration]
                fused_df.loc[indices, f"spec_{col}"] += (sig_val * decay_profile)
            
            fused_df.loc[fused_df.index[start_idx:start_idx+duration], 'label'] = 1
            
        return fused_df
