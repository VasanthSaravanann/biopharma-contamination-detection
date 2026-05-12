#!/usr/bin/env python3
"""
Simple script to compute per-sample latency statistics from an inputs/outputs timing log.
Usage: python scripts/reconcile_latency.py path/to/latency_log.csv
Expected CSV columns: sample_id, start_ts, end_ts
"""
import sys
import pandas as pd

def main(path):
    df = pd.read_csv(path)
    # Mixed timestamp precision is expected (with/without fractional seconds).
    df['start_ts'] = pd.to_datetime(df['start_ts'], format='mixed', errors='coerce')
    df['end_ts'] = pd.to_datetime(df['end_ts'], format='mixed', errors='coerce')
    df = df.dropna(subset=['start_ts', 'end_ts']).copy()
    df['latency_ms'] = (df['end_ts'] - df['start_ts']).dt.total_seconds() * 1000
    print('Count:', len(df))
    print('Mean latency (ms):', df['latency_ms'].mean())
    print('Std latency (ms):', df['latency_ms'].std())
    print('Latency median (ms):', df['latency_ms'].median())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/reconcile_latency.py path/to/latency_log.csv')
        sys.exit(1)
    main(sys.argv[1])
