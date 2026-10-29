import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from src import data_integration as di


def test_load_co2_timeseries_csv(tmp_path):
    df = pd.DataFrame({
        'sample_id': ['s1'] * 5 + ['s2'] * 5,
        'timestamp': list(range(10)),
        'co2_pct': [0.4,0.42,0.41,0.39,0.4, 0.5,0.51,0.49,0.48,0.5]
    })
    p = tmp_path / 'co2.csv'
    df.to_csv(p, index=False)

    out = di.load_co2_timeseries_csv(str(p))
    assert 'co2_mean' in out.columns
    assert out.shape[0] == 2


def test_load_metabolomics_aggregated_csv(tmp_path):
    df = pd.DataFrame({
        'sample_id': ['s1','s2','s3'],
        'met_a': [1.0,2.0,3.0],
        'met_b': [10.0,20.0,30.0]
    })
    p = tmp_path / 'met.csv'
    df.to_csv(p, index=False)

    out = di.load_metabolomics_aggregated_csv(str(p))
    # z-scored, mean approx 0
    assert abs(out['met_a'].mean()) < 1e-6
    assert out.shape[0] == 3
