import pandas as pd
import numpy as np
from src.feature_fusion import MultimodalFeatureExtractor


def test_multimodal_feature_extractor_basic():
    # create dummy spectral columns and process columns
    df = pd.DataFrame({
        'spec_200': [0.1, 0.2, 0.15],
        'spec_201': [0.05, 0.07, 0.06],
        'pH': [7.2, 7.3, 7.1],
        'TEMP': [37, 36.8, 37.1],
        'co2_mean': [0.4, 0.41, 0.39]
    })

    mfe = MultimodalFeatureExtractor(mode='fused', expanded_feature_columns=['co2_mean'])
    X = mfe.extract_features(df)
    # Expect a 3 x N feature matrix
    assert X.shape[0] == 3
    assert X.ndim == 2
