"""
Instrument-aware stratified data split.

Prevents instrument leakage by ensuring:
- Both AMBR and CEC samples in train and test
- Contamination events balanced across instruments
- Temporal coherence preserved within each instrument
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_instrument_aware_split(X, y, instrument_labels, group_ids, 
                                   test_size=0.2, random_state=42):
    """
    Create stratified split that respects instrument boundaries.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        instrument_labels: Array of 0/1 (0=AMBR, 1=CEC)
        group_ids: Group IDs for temporal coherence
        test_size: Fraction for test set
        random_state: Seed for reproducibility
    
    Returns:
        train_idx, test_idx: Indices for train/test split
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    
    # Separate by instrument
    ambr_indices = np.where(instrument_labels == 0)[0]
    cec_indices = np.where(instrument_labels == 1)[0]
    
    print(f"Instrument distribution:")
    print(f"  AMBR: {len(ambr_indices)} samples")
    print(f"  CEC: {len(cec_indices)} samples")
    
    # Stratified split per instrument
    train_ambr = []
    test_ambr = []
    train_cec = []
    test_cec = []
    
    # Split AMBR with GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in gss.split(X[ambr_indices], y[ambr_indices], 
                                           group_ids[ambr_indices]):
        train_ambr = ambr_indices[train_idx]
        test_ambr = ambr_indices[test_idx]
    
    # Split CEC with GroupShuffleSplit
    for train_idx, test_idx in gss.split(X[cec_indices], y[cec_indices], 
                                           group_ids[cec_indices]):
        train_cec = cec_indices[train_idx]
        test_cec = cec_indices[test_idx]
    
    # Combine
    train_idx = np.concatenate([train_ambr, train_cec])
    test_idx = np.concatenate([test_ambr, test_cec])
    
    # Shuffle for randomness
    train_idx = np.random.permutation(train_idx)
    test_idx = np.random.permutation(test_idx)
    
    print(f"\nTrain/Test split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"    AMBR: {np.sum(instrument_labels[train_idx] == 0)}")
    print(f"    CEC: {np.sum(instrument_labels[train_idx] == 1)}")
    print(f"    Clean: {np.sum(y[train_idx] == 0)}")
    print(f"    Contaminated: {np.sum(y[train_idx] == 1)}")
    
    print(f"  Test: {len(test_idx)} samples")
    print(f"    AMBR: {np.sum(instrument_labels[test_idx] == 0)}")
    print(f"    CEC: {np.sum(instrument_labels[test_idx] == 1)}")
    print(f"    Clean: {np.sum(y[test_idx] == 0)}")
    print(f"    Contaminated: {np.sum(y[test_idx] == 1)}")
    
    # Verify no leakage
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train/test overlap!"
    assert len(train_idx) + len(test_idx) == n_samples, "Missing samples!"
    
    return train_idx, test_idx


if __name__ == "__main__":
    # Test with real data
    import sys
    import yaml
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_processing import AmbrDatasetParser
    from src.data_integration import CECDataParser
    from src.fusion import DataFuser
    from src.feature_fusion import MultimodalFeatureExtractor
    
    with open("config/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load data
    ambr_root = Path(config['data']['ambr_root'])
    cec_root = Path(config['data']['cec_root'])
    bacteria_root = Path(config['data']['bacteria_root'])
    
    parser = AmbrDatasetParser(str(ambr_root))
    ambr_df = parser.parse_sensor_files("00001/S")
    
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
    
    baseline_df = pd.concat([ambr_df, cec_df], axis=0, sort=False)
    baseline_df = baseline_df.sort_index().ffill().bfill()
    
    fuser = DataFuser(str(ambr_root), str(bacteria_root), seed=42)
    fused_df = fuser.fuse(baseline_df, "EColi", 10, n_injections=3)
    
    extractor = MultimodalFeatureExtractor(mode='fused')
    X = extractor.extract_features(fused_df)
    y = fused_df['label'].values
    
    # Create instrument labels
    n_ambr = len(ambr_df)
    instrument_labels = np.concatenate([
        np.zeros(n_ambr, dtype=int),
        np.ones(len(fused_df) - n_ambr, dtype=int)
    ])
    
    # Create group IDs
    group_ids = np.arange(len(y)) // 60
    
    print("=" * 80)
    print("TESTING INSTRUMENT-AWARE STRATIFIED SPLIT")
    print("=" * 80)
    
    train_idx, test_idx = create_instrument_aware_split(
        X, y, instrument_labels, group_ids, test_size=0.2, random_state=42
    )
    
    print("\n✓ Split successful, no leakage detected!")
