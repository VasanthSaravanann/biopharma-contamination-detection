import pandas as pd
import glob
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AmbrDatasetParser:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def parse_sensor_files(self, folder_path: str):
        """Parses Pegasus .all.csv files into a unified dataframe."""
        search_path = self.base_path / folder_path / "**" / "*.all.csv"
        files = glob.glob(str(search_path), recursive=True)
        
        logging.info(f"Auditing {len(files)} files in {folder_path}...")
        
        all_frames = []
        for file_path in files:
            try:
                # 1. Load data
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    header_row = -1
                    sensor_name = "unknown"
                    for i, line in enumerate(lines):
                        if 'ObjectKey' in line:
                            sensor_name = line.split(',')[1].strip()
                        if 'timestamp' in line.lower() or 'ObjectKey' in line:
                            if len(line.split(',')) > 1:
                                header_row = i
                                break
                    if header_row == -1: continue
                
                df = pd.read_csv(file_path, skiprows=header_row)
                df = df.iloc[:, [0, 1]]
                df.columns = ['timestamp', 'value']
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['timestamp', 'value'])
                
                df = df.set_index('timestamp')[['value']].rename(columns={'value': sensor_name})
                
                # Corrected deduplication: keep first occurrence
                df = df[~df.index.duplicated(keep='first')]
                all_frames.append(df)
            except Exception as e:
                logging.debug(f"Skipping {file_path}: {e}")
        
        if not all_frames: return pd.DataFrame()
        
        # Concat all sensors and resample
        df_full = pd.concat(all_frames, axis=1)
        # Fix: ensure index is sorted and unique before resampling
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        return df_full.sort_index().resample('1min').mean().interpolate(method='linear')
