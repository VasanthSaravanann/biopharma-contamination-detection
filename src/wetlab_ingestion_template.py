"""Wet-lab ingestion template

Lightweight helper to ingest per-sample wet-lab CSVs (spectra + metadata)
and produce a processed manifest row for the pipeline.

Usage:
    python -m src.wetlab_ingestion_template /path/to/wetlab/csvs --out data/processed/wetlab_manifest.csv
"""
import argparse
from pathlib import Path
import pandas as pd


def ingest_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob('*.csv')):
        try:
            df = pd.read_csv(p)
            # Assume first row contains metadata columns like sample_id, organism, cfu
            meta = {c: df[c].iloc[0] if c in df.columns else None for c in ['sample_id','organism','cfu','timestamp']}
            meta['file'] = str(p)
            rows.append(meta)
        except Exception:
            continue
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('--out', type=str, default='data/processed/wetlab_manifest.csv')
    args = parser.parse_args()

    folder = Path(args.folder)
    manifest = ingest_folder(folder)
    manifest.to_csv(args.out, index=False)
    print(f"Wrote manifest to {args.out} with {len(manifest)} entries")


if __name__ == '__main__':
    main()
