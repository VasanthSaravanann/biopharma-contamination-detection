"""Script to add modality flags to data/processed/dataset_metadata.csv

Scans the data folder for known modality subfolders and adds boolean columns
`has_co2` and `has_metabolomics` to the metadata CSV. If the CSV does not
exist, it creates a minimal manifest based on files present.
"""
import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path('data/processed')
METADATA_CSV = DATA_PROCESSED / 'dataset_metadata.csv'


def detect_modalities(root: Path) -> dict:
    has_co2 = (root / 'FCIC_AMBR_05').exists() or any(root.glob('**/*co2*.csv'))
    has_met = (root / 'ST001316').exists() or any(root.glob('**/*metabolomics*.csv'))
    return {'has_co2': bool(has_co2), 'has_metabolomics': bool(has_met)}


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    flags = detect_modalities(Path('data'))

    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV)
        df['has_co2'] = df.get('has_co2', flags['has_co2'])
        df['has_metabolomics'] = df.get('has_metabolomics', flags['has_metabolomics'])
    else:
        # Create a minimal metadata file
        df = pd.DataFrame({
            'sample_id': [],
            'has_co2': [],
            'has_metabolomics': []
        })
        df['has_co2'] = flags['has_co2']
        df['has_metabolomics'] = flags['has_metabolomics']

    df.to_csv(METADATA_CSV, index=False)
    print(f"Updated metadata at {METADATA_CSV} with flags: {flags}")


if __name__ == '__main__':
    main()
