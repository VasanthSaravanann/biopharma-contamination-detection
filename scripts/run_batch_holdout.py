#!/usr/bin/env python3
"""Run batch-level holdout experiments from metadata and split definitions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


def load_split_definitions(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header_seen = False
        for raw in reader:
            if not raw:
                continue
            first = (raw[0] or "").strip()
            if first.startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                if first.lower() == "sample_id":
                    continue
            if len(raw) < 4:
                continue
            rows.append(
                {
                    "sample_id": raw[0].strip(),
                    "purpose": raw[1].strip(),
                    "modality": raw[2].strip(),
                    "reason": ",".join(raw[3:]).strip(),
                }
            )
    return rows


def resolve_holdout_mask(df: pd.DataFrame, split_id: str) -> pd.Series:
    dates = df["experiment_date"].fillna("").astype(str)
    if split_id.endswith("_all_batches_holdout") and split_id[:8].isdigit():
        cutoff = split_id[:8]
        return dates >= cutoff
    if "_batch_subset" in split_id:
        prefix = split_id.split("_", 1)[0]
        if prefix.isdigit() and len(prefix) == 6:
            return dates.str.startswith(prefix)
    return df["filename"].astype(str).eq(split_id)


def run_one_split(df: pd.DataFrame, split: Dict[str, str]) -> Dict[str, object]:
    split_id = split["sample_id"]
    holdout_mask = resolve_holdout_mask(df, split_id)
    train_mask = ~holdout_mask

    n_holdout = int(holdout_mask.sum())
    n_train = int(train_mask.sum())
    if n_holdout == 0 or n_train == 0:
        return {
            "split_id": split_id,
            "purpose": split["purpose"],
            "modality": split["modality"],
            "reason": split["reason"],
            "n_train": n_train,
            "n_holdout": n_holdout,
            "status": "skipped_empty_partition",
        }

    feature_cols = [
        c
        for c in ["cfu", "timepoint_hours", "n_wl", "wl_min", "wl_max", "abs_min", "abs_max", "abs_mean"]
        if c in df.columns
    ]
    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X_df.to_numpy(dtype=float)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()

    y_series = pd.Series(y, index=df.index)
    X_train_clean = X[train_mask & (y_series == 0)]
    if len(X_train_clean) < 10:
        return {
            "split_id": split_id,
            "purpose": split["purpose"],
            "modality": split["modality"],
            "reason": split["reason"],
            "n_train": n_train,
            "n_holdout": n_holdout,
            "status": "skipped_insufficient_clean_train",
        }

    model = IsolationForest(n_estimators=200, random_state=42, contamination=0.1)
    model.fit(X_train_clean)

    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]
    scores = -model.decision_function(X_holdout)

    if len(np.unique(y_holdout)) < 2:
        auc = None
        status = "evaluated_single_class_holdout"
    else:
        auc = float(roc_auc_score(y_holdout, scores))
        status = "evaluated"

    return {
        "split_id": split_id,
        "purpose": split["purpose"],
        "modality": split["modality"],
        "reason": split["reason"],
        "n_train": n_train,
        "n_holdout": n_holdout,
        "holdout_clean": int((y_holdout == 0).sum()),
        "holdout_contaminated": int((y_holdout == 1).sum()),
        "roc_auc": auc,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch-level holdout experiments")
    parser.add_argument("--metadata", default="data/processed/dataset_metadata.csv")
    parser.add_argument("--splits", default="data/splits/holdout_definition.csv")
    parser.add_argument("--out", default="output/holdout_results")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    split_path = Path(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    splits = load_split_definitions(split_path)

    results: List[Dict[str, object]] = []
    for split in splits:
        results.append(run_one_split(df, split))

    summary_path = out_dir / "holdout_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)

    json_path = out_dir / "holdout_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
