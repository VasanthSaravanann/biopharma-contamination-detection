import pandas as pd
import glob
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AmbrDatasetParser:
    """
    Parses AMBR (AMBR 250 bioreactor) *.all.csv sensor files into unified dataframe.
    
    Handles:
    - Multiple sensor files with different timestamps
    - Malformed headers and missing values
    - Sensor dropouts (NaN handling)
    - Provenance tracking (batch ID, organism, inoculum)
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parser_stats = {
            "total_files_found": 0,
            "successfully_parsed": 0,
            "parsing_failures": 0,
            "sensor_count": 0,
            "total_samples": 0
        }
        self.provenance = {}

    def parse_sensor_files(self, folder_path: str, resample_freq: str = '1min', fill_method: str = 'linear'):
        """
        Parse all sensor files in a folder path into a unified time-indexed dataframe.
        
        Args:
            folder_path: Path relative to base_path (e.g., "00001/S")
            resample_freq: Resampling frequency (default '1min')
            fill_method: Missing value interpolation method ('linear', 'ffill', 'bfill')
        
        Returns:
            DataFrame with datetime index and sensor columns
        
        Raises:
            FileNotFoundError: If no sensor files found
            ValueError: If all files failed to parse
        """
        search_path = self.base_path / folder_path / "**" / "*.all.csv"
        files = glob.glob(str(search_path), recursive=True)
        self.parser_stats["total_files_found"] = len(files)
        
        if not files:
            raise FileNotFoundError(f"No .all.csv files found in {search_path}")
        
        self.logger.info(f"Found {len(files)} sensor files in {folder_path}")
        
        all_frames = []
        sensors_by_type = {}  # Track sensor types found
        
        for file_path in files:
            try:
                # Extract sensor name and metadata from filename
                filename = os.path.basename(file_path)
                sensor_name = self._extract_sensor_name(filename)
                
                # Parse metadata from CSV header (ObjectKey line)
                metadata = self._extract_metadata_from_file(file_path)
                sensor_type = metadata.get('sensor_type', 'unknown')
                
                # Track sensor types
                if sensor_type not in sensors_by_type:
                    sensors_by_type[sensor_type] = []
                sensors_by_type[sensor_type].append(sensor_name)
                
                # Parse sensor data
                df = self._parse_sensor_csv(file_path)
                if df.empty:
                    self.logger.warning(f"Parsed file is empty: {filename}")
                    continue
                
                # Rename column with sensor name
                df = df.rename(columns={'value': sensor_name})
                
                # Deduplicate timestamps (keep first occurrence)
                df = df[~df.index.duplicated(keep='first')]
                all_frames.append(df)
                self.parser_stats["successfully_parsed"] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {filename}: {e}")
                self.parser_stats["parsing_failures"] += 1
        
        if not all_frames:
            raise ValueError(f"No sensor files successfully parsed from {search_path}")
        
        # Concatenate all sensors (outer join to preserve all timestamps)
        df_full = pd.concat(all_frames, axis=1)
        
        # Ensure index is sorted and unique before resampling
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        df_full = df_full.sort_index()
        
        # Resample and fill missing values
        df_resampled = df_full.resample(resample_freq).mean()
        
        if fill_method == 'linear':
            df_resampled = df_resampled.interpolate(method='linear')
        elif fill_method == 'ffill':
            df_resampled = df_resampled.fillna(method='ffill')
        elif fill_method == 'bfill':
            df_resampled = df_resampled.fillna(method='bfill')
        
        # Track final statistics
        self.parser_stats["sensor_count"] = df_resampled.shape[1]
        self.parser_stats["total_samples"] = df_resampled.shape[0]
        
        self.logger.info(
            f"Parsed {self.parser_stats['successfully_parsed']} files, "
            f"{self.parser_stats['sensor_count']} sensors, "
            f"{self.parser_stats['total_samples']} time points"
        )
        
        return df_resampled
    
    def _extract_sensor_name(self, filename: str) -> str:
        """Extract sensor name from filename (e.g., 'DO.all.csv' -> 'DO')"""
        name = os.path.splitext(filename)[0]  # Remove .csv
        name = name.replace('.all', '')  # Remove .all
        return name if name else 'sensor_unknown'
    
    def _extract_metadata_from_file(self, file_path: str) -> dict:
        """
        Extract metadata from CSV header (ObjectKey line).
        
        ObjectKey line typically contains sensor description.
        """
        metadata = {'sensor_type': 'unknown'}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if 'ObjectKey' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            metadata['sensor_type'] = parts[1].strip()
                        break
        except Exception as e:
            self.logger.debug(f"Could not extract metadata from {file_path}: {e}")
        
        return metadata
    
    def _parse_sensor_csv(self, file_path: str) -> pd.DataFrame:
        """
        Parse a single sensor CSV file.
        
        Handles different header formats and malformed data.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return pd.DataFrame()
            
            # Find the data header row
            header_row = -1
            for i, line in enumerate(lines):
                if 'timestamp' in line.lower() or 'ObjectKey' in line:
                    if len(line.split(',')) > 1:
                        header_row = i
                        break
            
            if header_row == -1:
                return pd.DataFrame()
        
        # Parse CSV starting from header row
        df = pd.read_csv(file_path, skiprows=header_row, on_bad_lines='skip')
        
        # Assume first two columns are timestamp and value
        if df.shape[1] < 2:
            return pd.DataFrame()
        
        df = df.iloc[:, [0, 1]]
        df.columns = ['timestamp', 'value']
        
        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop rows with NaT or NaN in key columns
        df = df.dropna(subset=['timestamp', 'value'])
        
        # Set timestamp as index
        df = df.set_index('timestamp')[['value']]
        
        return df
    
    def get_parser_stats(self) -> dict:
        """Return parsing statistics for auditing."""
        return self.parser_stats.copy()
