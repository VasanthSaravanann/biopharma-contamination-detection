import pandas as pd
import numpy as np
import logging
from pathlib import Path
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DataFuser:
    """
    Fuses surrogate spectral contamination signatures into real process baselines.
    
    Preserves provenance:
    - Tracks injection start/end times
    - Records organism, inoculum, CFU level
    - Maintains contamination label alignment to time windows
    - Records decay profile used for injection
    """
    
    def __init__(self, ambr_path: str, bacteria_path: str, seed: int = 42):
        self.ambr_path = Path(ambr_path)
        self.bacteria_path = Path(bacteria_path)
        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.provenance_log = []
        
    def get_surrogate_signature(self, contamination_type: str, cfu_level):
        """
        Load spectral signature files for a contamination organism at given inoculum level.
        
        Args:
            contamination_type: Organism name (e.g., 'EColi')
            cfu_level: CFU/mL level (10, 100, etc.) or string like '10CFU'
        
        Returns:
            DataFrame with spectral features
        
        Raises:
            FileNotFoundError: If no surrogate file found for organism/level
        """
        # Normalize CFU level format
        if isinstance(cfu_level, int):
            cfu_str = f"{cfu_level}CFU"
        else:
            cfu_str = str(cfu_level)
        
        # Robust pattern matching: try case-insensitive or partial matches
        patterns = [
            f"*{contamination_type}*{cfu_str}*",
            f"*{contamination_type.replace('_', '')}*{cfu_str}*",
            f"*{contamination_type}*"
        ]
        
        files = []
        for p in patterns:
            files = glob.glob(str(self.bacteria_path / "**" / p.replace(' ', '_')), recursive=True)
            if files: break
            
        if not files:
            raise FileNotFoundError(
                f"No surrogate spectral signature found for {contamination_type} at {cfu_str}. "
                f"Searched in {self.bacteria_path}"
            )
        
        self.logger.info(f"Loading surrogate from: {files[0]}")
        df = pd.read_csv(files[0])
        # Apply numeric conversion and drop non-numeric columns
        return df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    def fuse(self, ambr_df: pd.DataFrame, contamination_type: str, cfu_level, n_injections: int = 3):
        """
        Fuse contamination spectral signature into AMBR process data.
        
        Args:
            ambr_df: Real AMBR sensor data (datetime index)
            contamination_type: Organism name
            cfu_level: CFU/mL inoculum level
            n_injections: Number of contamination events to inject
        
        Returns:
            Fused DataFrame with:
            - Original AMBR process columns
            - Injected spectral features (spec_*)
            - Label column (0=clean, 1=contaminated)
            - Provenance metadata columns
        """
        signature = self.get_surrogate_signature(contamination_type, cfu_level)
        
        # Explicit spectral channel list (must match training schema)
        sig_cols = [c for c in signature.columns if 'abs' in c.lower()][:5]
        if not sig_cols:
            sig_cols = [c for c in signature.columns if 'spec' in c.lower()][:5]
        if not sig_cols:
            sig_cols = signature.columns[:5]  # Fallback: first 5 columns
        
        self.logger.info(f"Using {len(sig_cols)} spectral channels: {sig_cols}")
        
        fused_df = ambr_df.copy()
        
        # Initialize spectral columns
        for col in sig_cols:
            fused_df[f"spec_{col}"] = 0.0
        
        # Initialize provenance columns
        fused_df['label'] = 0 
        fused_df['injection_id'] = -1  # -1 = no injection
        fused_df['injection_start_time'] = pd.NaT
        fused_df['injection_end_time'] = pd.NaT
        fused_df['contamination_organism'] = None
        fused_df['contamination_cfu_ml'] = None
        
        n_samples = len(fused_df)
        
        # Inject multiple contamination events with provenance tracking
        for inj_id in range(n_injections):
            duration = self.rng.integers(30, 90)
            if n_samples <= duration: continue
            
            start_idx = self.rng.integers(0, n_samples - duration)
            end_idx = start_idx + duration
            jitter = self.rng.normal(1.0, 0.05, duration)
            
            # Get time bounds for this injection
            start_time = fused_df.index[start_idx]
            end_time = fused_df.index[end_idx - 1] if end_idx < n_samples else fused_df.index[-1]
            
            # Record provenance
            prov_entry = {
                "injection_id": inj_id,
                "organism": contamination_type,
                "cfu_level": cfu_level,
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": duration,
                "start_index": start_idx,
                "end_index": end_idx,
                "n_samples_contaminated": duration
            }
            self.provenance_log.append(prov_entry)
            
            for col in sig_cols:
                sig_val = signature[col].mean()
                t = np.linspace(0, 1, duration)
                # Decay profile: smooth rise and fall (sinusoidal)
                decay_profile = np.sin(t * np.pi) * jitter
                
                indices = fused_df.index[start_idx:start_idx+duration]
                fused_df.loc[indices, f"spec_{col}"] += (sig_val * decay_profile)
            
            # Mark contaminated time window
            injection_slice = fused_df.index[start_idx:end_idx]
            fused_df.loc[injection_slice, 'label'] = 1
            fused_df.loc[injection_slice, 'injection_id'] = inj_id
            fused_df.loc[injection_slice, 'injection_start_time'] = start_time
            fused_df.loc[injection_slice, 'injection_end_time'] = end_time
            fused_df.loc[injection_slice, 'contamination_organism'] = contamination_type
            fused_df.loc[injection_slice, 'contamination_cfu_ml'] = cfu_level
            
        return fused_df
    
    def get_provenance_log(self):
        """Return log of all contamination injections for auditing."""
        return self.provenance_log.copy()
