"""
Comprehensive Data Integration Module

Merges multiple data sources:
1. AMBR process data (FCIC_AMBR_05)
2. CEC_04_2L fermentation data (Sartorius scale data)
3. Bacteria contamination spectra
4. HPLC metabolite data for clean reference

Produces unified dataset for training.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_co2_timeseries_csv(path: str) -> pd.DataFrame:
    """Load CO2 off-gas timeseries CSV and return per-sample aggregated features.

    Expected columns: sample_id, timestamp, co2_pct
    Returns DataFrame indexed by sample_id with aggregated columns:
    - co2_mean, co2_std, co2_p95, co2_p05
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CO2 file not found: {path}")
    df = pd.read_csv(p)
    if 'sample_id' not in df.columns or 'co2_pct' not in df.columns:
        raise ValueError('CO2 CSV must contain sample_id and co2_pct columns')

    agg = df.groupby('sample_id')['co2_pct'].agg([
        ('co2_mean', 'mean'),
        ('co2_std', 'std'),
        ('co2_p95', lambda x: np.percentile(x, 95)),
        ('co2_p05', lambda x: np.percentile(x, 5)),
    ])
    return agg.reset_index().set_index('sample_id')


def load_metabolomics_aggregated_csv(path: str) -> pd.DataFrame:
    """Load aggregated metabolomics CSV (per-sample) and return normalized features.

    Expected columns: sample_id, metabolite_XXX (many columns)
    Returns DataFrame indexed by sample_id with z-scored metabolite intensities.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metabolomics file not found: {path}")
    df = pd.read_csv(p)
    if 'sample_id' not in df.columns:
        raise ValueError('Metabolomics CSV missing sample_id column')

    meta = df.set_index('sample_id')
    # Simple z-score normalization per metabolite
    meta = meta.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9), axis=0)
    return meta



class CECDataParser:
    """Parse CEC_04_2L fermentation bioreactor data (Sartorius A5/A6)"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_sartorius_data(self, file_path: str) -> pd.DataFrame:
        """
        Parse Sartorius bioreactor scale data.
        
        Returns time-indexed DataFrame with process variables:
        - pH, temperature, DO, agitation
        - Feed rates, CO2/N2/O2 setpoints
        - Batch age, vessel weight
        """
        try:
            # Skip metadata rows (row 2 with "Value" descriptors and row 3 with units)
            df = pd.read_csv(file_path, skiprows=[1, 2])
            
            # Standardize timestamp column
            if 'PDatTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['PDatTime'], errors='coerce')
            elif 'Time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Time'], errors='coerce')
            else:
                self.logger.warning(f"Could not find timestamp column in {file_path}")
                return pd.DataFrame()
            
            # Select relevant process columns
            process_cols = [
                'pH', 'TEMP', 'PO2', 'STIRR', 'FEED_FLOW',
                'Air_SP', 'CO2_SP', 'N2_SP', 'JTEMP'
            ]
            
            available_cols = [c for c in process_cols if c in df.columns]
            result = df[['timestamp'] + available_cols].copy()
            result = result.set_index('timestamp')
            result = result.sort_index()
            
            self.logger.info(f"Parsed Sartorius data: {len(result)} samples, {len(available_cols)} variables")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse Sartorius data: {e}")
            return pd.DataFrame()
    
    def parse_hplc_results(self, file_path: str) -> Dict:
        """
        Parse HPLC results for metabolite concentrations.
        
        Returns dict with clean reference values for metabolites:
        - Glucose, xylose, ethanol, lactic acid, acetic acid, cellobiose
        """
        try:
            df = pd.read_csv(file_path, skiprows=2)
            
            # Extract standard curves and sample results
            metabolite_data = {
                'glucose': [],
                'xylose': [],
                'ethanol': [],
                'acetic_acid': [],
                'lactic_acid': [],
                'cellobiose': []
            }
            
            # Parse injection data
            if 'Injection Name' in df.columns:
                for idx, row in df.iterrows():
                    if 'STD' in str(row['Injection Name']):
                        # Standard curve point
                        for metabolite in metabolite_data.keys():
                            col_name = metabolite.replace('_', ' ').title()
                            if col_name in df.columns:
                                val = row[col_name]
                                if pd.notna(val) and val != 'n.a.':
                                    metabolite_data[metabolite].append(float(val))
            
            self.logger.info(f"Parsed HPLC results: {len(metabolite_data)} metabolites quantified")
            return metabolite_data
            
        except Exception as e:
            self.logger.error(f"Failed to parse HPLC results: {e}")
            return {}
    
    def parse_fermentation_plan(self, file_path: str) -> Dict:
        """
        Parse CEC_04 fermentation plan for experiment metadata.
        
        Returns dict with:
        - Strain, substrate, temperature, pH, agitation
        - Antibiotic supplementation
        - Run date and conditions
        """
        try:
            df = pd.read_csv(file_path, header=None)
            
            metadata = {}
            for idx, row in df.iterrows():
                row_text = str(row[0]).strip()
                if 'Strain' in row_text:
                    metadata['strain'] = str(row[1]).strip() if len(row) > 1 else "Unknown"
                elif 'Substrate' in row_text:
                    metadata['substrate'] = str(row[1]).strip() if len(row) > 1 else "Unknown"
                elif 'Temp' in row_text:
                    metadata['temperature_celsius'] = str(row[1]).strip() if len(row) > 1 else "30"
                elif 'pH' in row_text and '~' in str(row[1]):
                    metadata['ph'] = str(row[1]).strip() if len(row) > 1 else "5"
                elif 'agitation' in row_text.lower():
                    metadata['agitation_rpm'] = str(row[1]).strip() if len(row) > 1 else "300"
            
            self.logger.info(f"Parsed fermentation plan: {metadata}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to parse fermentation plan: {e}")
            return {}


class UnifiedDataset:
    """Combine AMBR + CEC + spectra into unified training dataset"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_metadata = {
            "sources": [],
            "n_samples": 0,
            "timestamp": datetime.now().isoformat(),
            "feature_sets": {}
        }
    
    def load_ambr_data(self, ambr_root: str, folder_path: str = "00001/S") -> pd.DataFrame:
        """Load AMBR sensor data"""
        from src.data_processing import AmbrDatasetParser
        
        try:
            parser = AmbrDatasetParser(ambr_root)
            df = parser.parse_sensor_files(folder_path)
            
            self.logger.info(f"Loaded AMBR data: {df.shape[0]} samples, {df.shape[1]} sensors")
            self.dataset_metadata["sources"].append({
                "source": "AMBR_FCIC_05",
                "folder": folder_path,
                "n_samples": len(df),
                "n_sensors": df.shape[1]
            })
            
            return df
        except Exception as e:
            self.logger.error(f"Failed to load AMBR data: {e}")
            return pd.DataFrame()
    
    def load_cec_data(self, cec_root: str) -> pd.DataFrame:
        """Load CEC fermentation bioreactor data"""
        try:
            parser = CECDataParser(cec_root)
            
            # Parse both Sartorius scales
            dfs_by_scale = {}
            for scale in ['Sartorius_A5', 'Sartorius_A6']:
                file_path = Path(cec_root) / f"{scale}-Table 1.csv"
                if file_path.exists():
                    df = parser.parse_sartorius_data(str(file_path))
                    if not df.empty:
                        # Add scale prefix to column names to avoid conflicts
                        df = df.rename(columns={col: f"{scale}_{col}" for col in df.columns})
                        dfs_by_scale[scale] = df
            
            if dfs_by_scale:
                # Merge scales by joining on index (timestamps)
                combined = None
                for scale, df in dfs_by_scale.items():
                    if combined is None:
                        combined = df
                    else:
                        # Join on index, handling potential duplicate timestamps
                        combined = combined.join(df, how='outer')
                
                # Fill forward any missing values
                combined = combined.ffill().bfill()
                
                self.logger.info(f"Loaded CEC data: {combined.shape[0]} samples, {combined.shape[1]} variables")
                self.dataset_metadata["sources"].append({
                    "source": "CEC_04_2L_Fermentation",
                    "scales": list(dfs_by_scale.keys()),
                    "n_samples": len(combined),
                    "n_variables": combined.shape[1]
                })
                
                return combined
        except Exception as e:
            self.logger.error(f"Failed to load CEC data: {e}")
        
        return pd.DataFrame()
    
    def create_merged_baseline(self, ambr_df: pd.DataFrame, cec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge AMBR and CEC process data into unified baseline.
        
        Both are process-only (no contamination labels), representing
        clean fermentation conditions for unsupervised training.
        """
        try:
            # Standardize column names
            ambr_renamed = ambr_df.copy()
            cec_renamed = cec_df.copy()
            
            # Create merged dataframe
            merged = pd.concat([ambr_renamed, cec_renamed], axis=0)
            merged = merged.sort_index()
            
            # Forward fill missing values within each time group
            merged = merged.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Merged process data: {merged.shape[0]} total samples, {merged.shape[1]} variables")
            self.dataset_metadata["sources"].append({
                "source": "Merged_Baseline",
                "n_clean_samples": len(merged)
            })
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Failed to merge data: {e}")
            return pd.DataFrame()
    
    def load_contamination_labels(self, bacteria_root: str) -> pd.DataFrame:
        """
        Load contamination events from spectral data.
        
        Returns DataFrame with:
        - Timestamp of contamination event
        - Organism, CFU level
        - Duration of event
        - Spectral signature hash
        """
        from src.fusion import DataFuser
        
        try:
            contamination_events = []
            
            fuser = DataFuser(bacteria_root, bacteria_root)
            
            # Scan for all contamination files
            search_path = Path(bacteria_root) / "Contaminated samples" / "*.csv"
            files = glob.glob(str(search_path))
            
            for file_path in files:
                filename = Path(file_path).name
                
                # Parse filename: 20230323_D5_PAeruginosa9027_100CFU_1.csv
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 4:
                    try:
                        organism = parts[2]  # e.g., PAeruginosa9027
                        cfu = parts[3]  # e.g., 100CFU
                        
                        contamination_events.append({
                            'filename': filename,
                            'organism': organism,
                            'inoculum_level': cfu,
                            'label': 1  # Contaminated
                        })
                    except:
                        pass
            
            if contamination_events:
                df = pd.DataFrame(contamination_events)
                self.logger.info(f"Found {len(df)} contamination event records")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to load contamination labels: {e}")
            return pd.DataFrame()
    
    def save_unified_dataset(self, output_file: str = "unified_dataset.csv"):
        """Save unified dataset metadata for pipeline reference"""
        output_path = self.output_dir / output_file
        
        metadata_df = pd.DataFrame([self.dataset_metadata])
        metadata_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Saved dataset metadata to {output_path}")
        
        # Also save as JSON for structured access
        import json
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.dataset_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved dataset metadata to {json_path}")
    
    def generate_report(self) -> str:
        """Generate integration report"""
        report = []
        report.append("=" * 80)
        report.append("UNIFIED DATASET INTEGRATION REPORT")
        report.append("=" * 80)
        report.append(f"\nTimestamp: {self.dataset_metadata['timestamp']}")
        report.append(f"\nData Sources Integrated ({len(self.dataset_metadata['sources'])}):")
        
        total_samples = 0
        for i, source in enumerate(self.dataset_metadata["sources"], 1):
            report.append(f"\n{i}. {source.get('source', 'Unknown')}")
            for key, value in source.items():
                if key != 'source':
                    report.append(f"   - {key}: {value}")
                    if 'samples' in key.lower():
                        total_samples += value if isinstance(value, int) else 0
        
        report.append(f"\n{'=' * 80}")
        report.append(f"Total Training Samples: {total_samples}")
        report.append(f"{'=' * 80}\n")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test integration
    print("Testing unified data integration...")
    
    dataset = UnifiedDataset(output_dir="data/processed")
    
    # Load AMBR data
    ambr_df = dataset.load_ambr_data(
        "/run/media/sham/AI_/ai-stack/projects/biopharma-contamination-detection/FCIC_AMBR_05/Data"
    )
    
    # Load CEC data
    cec_df = dataset.load_cec_data(
        "/Users/sham/Desktop/IEEE_IES/biopharma-contamination-detection/data/CEC_04_2L-fermentation"
    )
    
    # Merge baselines
    if not ambr_df.empty and not cec_df.empty:
        merged = dataset.create_merged_baseline(ambr_df, cec_df)
    
    # Save metadata
    dataset.save_unified_dataset()
    
    # Print report
    print(dataset.generate_report())
