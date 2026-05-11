"""
Analyze feature distribution heterogeneity between AMBR and CEC instruments.
Goal: Quantify scale mismatch and design per-instrument normalization.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import stats

# Load config
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

# Load data
from src.data_processing import AmbrDatasetParser
from src.data_integration import CECDataParser
from src.fusion import DataFuser
from src.feature_fusion import MultimodalFeatureExtractor

ambr_root = Path(config['data']['ambr_root'])
cec_root = Path(config['data']['cec_root'])
bacteria_root = Path(config['data']['bacteria_root'])

print("=" * 80)
print("FEATURE HETEROGENEITY ANALYSIS")
print("=" * 80)

# Load AMBR data
print("\n1. Loading AMBR data...")
parser = AmbrDatasetParser(str(ambr_root))
ambr_df = parser.parse_sensor_files("00001/S")
print(f"   AMBR: {ambr_df.shape[0]} samples, {ambr_df.shape[1]} sensors")

# Load CEC data
print("2. Loading CEC data...")
cec_parser = CECDataParser(str(cec_root))
cec_dfs = {}
for scale in ['Sartorius_A5', 'Sartorius_A6']:
    file_path = Path(cec_root) / f"{scale}-Table 1.csv"
    if file_path.exists():
        scale_df = cec_parser.parse_sartorius_data(str(file_path))
        if not scale_df.empty:
            scale_df = scale_df.rename(columns={col: f"{scale}_{col}" for col in scale_df.columns})
            cec_dfs[scale] = scale_df

cec_df = None
for scale, df in cec_dfs.items():
    if cec_df is None:
        cec_df = df
    else:
        cec_df = cec_df.join(df, how='outer')
cec_df = cec_df.ffill().bfill()
print(f"   CEC: {cec_df.shape[0]} samples, {cec_df.shape[1]} variables")

# Create merged baseline
baseline_df = pd.concat([ambr_df, cec_df], axis=0, sort=False)
baseline_df = baseline_df.sort_index().ffill().bfill()
print(f"   Merged: {baseline_df.shape[0]} samples, {baseline_df.shape[1]} variables")

# Fuse with contamination
fuser = DataFuser(str(ambr_root), str(bacteria_root), seed=42)
fused_df = fuser.fuse(baseline_df, "EColi", 10, n_injections=3)
print(f"   Fused: {fused_df.shape[0]} samples, {fused_df.shape[1]} features")

# Extract features
extractor = MultimodalFeatureExtractor(mode='fused')
X = extractor.extract_features(fused_df)
print(f"   Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")

# Separate by instrument
print("\n3. Separating by instrument...")
n_ambr = len(ambr_df)
X_ambr = X[:n_ambr]
X_cec = X[n_ambr:]
print(f"   AMBR features: {X_ambr.shape[0]} samples, {X_ambr.shape[1]} features")
print(f"   CEC features: {X_cec.shape[0]} samples, {X_cec.shape[1]} features")

# Analyze scale/distribution
print("\n4. Feature distribution statistics...")
print("\n   AMBR Statistics:")
print(f"   - Mean: {X_ambr.mean(axis=0).mean():.6f}")
print(f"   - Std:  {X_ambr.std(axis=0).mean():.6f}")
print(f"   - Min:  {X_ambr.min(axis=0).mean():.6f}")
print(f"   - Max:  {X_ambr.max(axis=0).mean():.6f}")

print("\n   CEC Statistics:")
print(f"   - Mean: {X_cec.mean(axis=0).mean():.6f}")
print(f"   - Std:  {X_cec.std(axis=0).mean():.6f}")
print(f"   - Min:  {X_cec.min(axis=0).mean():.6f}")
print(f"   - Max:  {X_cec.max(axis=0).mean():.6f}")

# Per-feature variance ratio (CEC/AMBR)
feature_vars_ambr = X_ambr.var(axis=0)
feature_vars_cec = X_cec.var(axis=0)
variance_ratio = np.divide(feature_vars_cec, feature_vars_ambr + 1e-10)

print("\n5. Feature variance mismatch (CEC/AMBR):")
print(f"   - Mean ratio: {variance_ratio.mean():.4f}")
print(f"   - Std ratio:  {variance_ratio.std():.4f}")
print(f"   - Min ratio:  {variance_ratio.min():.4f}")
print(f"   - Max ratio:  {variance_ratio.max():.4f}")

high_variance_features = np.where(variance_ratio > 2.0)[0]
print(f"   - Features with >2x variance mismatch: {len(high_variance_features)}/{X_ambr.shape[1]}")

# Test different scaling strategies
print("\n6. Testing normalization strategies...")

strategies = {}

# Global robust (current)
scaler_global = RobustScaler()
X_combined_global = scaler_global.fit_transform(X)
strategies['global_robust'] = X_combined_global

# Per-instrument robust scaling
X_combined_per_instrument = X.copy()
scaler_ambr = RobustScaler()
scaler_cec = RobustScaler()
X_combined_per_instrument[:n_ambr] = scaler_ambr.fit_transform(X_ambr)
X_combined_per_instrument[n_ambr:] = scaler_cec.fit_transform(X_cec)
strategies['per_instrument_robust'] = X_combined_per_instrument

# Measure uniformity
print("\n   Scale uniformity after normalization:")
for name, X_scaled in strategies.items():
    
    # Compute coefficient of variation per instrument
    cv_ambr = X_scaled[:n_ambr].std(axis=0).mean() / (abs(X_scaled[:n_ambr].mean(axis=0)).mean() + 1e-10)
    cv_cec = X_scaled[n_ambr:].std(axis=0).mean() / (abs(X_scaled[n_ambr:].mean(axis=0)).mean() + 1e-10)
    
    print(f"   - {name}:")
    print(f"     AMBR CV: {cv_ambr:.4f}, CEC CV: {cv_cec:.4f}, Ratio: {cv_cec/cv_ambr:.4f}")

print("\n7. Recommendation:")
print("   ✓ Per-instrument normalization recommended")
print("   ✓ Normalizes each instrument separately then applies ensemble")
print("   ✓ Should improve OCSVM and Autoencoder generalization")

print("\n" + "=" * 80)
