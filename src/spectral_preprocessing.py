"""Shared preprocessing utilities for raw UV-Vis spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _parse_absorbance_column(column_name: str) -> int:
    if not column_name.startswith("abs_"):
        raise ValueError(f"Not an absorbance column: {column_name}")
    return int(column_name.split("_", 1)[1])


def get_spectral_columns(df: pd.DataFrame) -> list[str]:
    """Return absorbance columns ordered by wavelength."""
    spectral_columns = [column for column in df.columns if column.startswith("abs_")]
    spectral_columns.sort(key=_parse_absorbance_column)
    return spectral_columns


def extract_raw_spectra(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract the raw absorbance matrix and wavelength axis from a dataframe."""
    spectral_columns = get_spectral_columns(df)
    if not spectral_columns:
        raise ValueError("No absorbance columns found; expected columns named abs_<wavelength>.")

    numeric = df[spectral_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    X = numeric.to_numpy(dtype=np.float32, copy=True)
    wavelengths = np.asarray([_parse_absorbance_column(column) for column in spectral_columns], dtype=np.float32)
    return X, wavelengths, spectral_columns


def load_spectral_file(
    file_path: str | Path,
    target_wavelengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a single wavelength/absorbance CSV and interpolate it onto a common grid."""
    path = Path(file_path)
    if target_wavelengths is None:
        target_wavelengths = np.arange(200.0, 801.0, 1.0, dtype=np.float32)
    else:
        target_wavelengths = np.asarray(target_wavelengths, dtype=np.float32)

    frame = pd.read_csv(path, skiprows=2, header=None)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in spectral file: {path}")

    wavelengths = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float32)
    absorbance = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(wavelengths) & np.isfinite(absorbance)
    wavelengths = wavelengths[valid_mask]
    absorbance = absorbance[valid_mask]

    if len(wavelengths) == 0:
        raise ValueError(f"No numeric wavelength data found in {path}")

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    absorbance = absorbance[order]

    interpolated = np.interp(
        target_wavelengths,
        wavelengths,
        absorbance,
        left=float(absorbance[0]),
        right=float(absorbance[-1]),
    ).astype(np.float32)

    metadata = {
        "file_path": str(path),
        "raw_points": int(len(wavelengths)),
        "wavelength_min": float(wavelengths.min()),
        "wavelength_max": float(wavelengths.max()),
    }
    if frame.shape[1] > 2:
        metadata["label"] = int(pd.to_numeric(frame.iloc[0, 2], errors="coerce")) if pd.notna(frame.iloc[0, 2]) else None
    if frame.shape[1] > 3:
        metadata["donor"] = int(pd.to_numeric(frame.iloc[0, 3], errors="coerce")) if pd.notna(frame.iloc[0, 3]) else None
    if frame.shape[1] > 4:
        metadata["cfu"] = int(pd.to_numeric(frame.iloc[0, 4], errors="coerce")) if pd.notna(frame.iloc[0, 4]) else None

    return interpolated, target_wavelengths, metadata


def load_spectral_directory(
    directory: str | Path,
    target_wavelengths: np.ndarray | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load every spectrum CSV in a directory into a dense raw-spectra matrix."""
    directory = Path(directory)
    file_paths = sorted(directory.glob("*.csv"))
    if limit is not None:
        file_paths = file_paths[:limit]
    if not file_paths:
        raise FileNotFoundError(f"No spectral CSV files found in {directory}")

    spectra = []
    metadata_rows = []
    wavelengths = None
    for file_path in file_paths:
        spectrum, wavelengths, metadata = load_spectral_file(file_path, target_wavelengths)
        spectra.append(spectrum)
        metadata_rows.append(metadata)

    return np.vstack(spectra), wavelengths, pd.DataFrame(metadata_rows)


def load_bacteria_spectra(
    directory: str | Path,
    target_wavelengths: np.ndarray | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load sterile absorbance spectra and interpolate them onto the 200-800 nm grid."""
    return load_spectral_directory(directory, target_wavelengths=target_wavelengths, limit=limit)


def apply_savgol_derivative(X: np.ndarray) -> np.ndarray:
    """Apply a first derivative Savitzky-Golay filter across each spectrum."""
    return savgol_filter(X, window_length=15, polyorder=2, deriv=1, axis=1)


def isolate_bandpass(X: np.ndarray, wavelengths: np.ndarray, band: Tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the wavelength channels inside the requested bandpass."""
    lower, upper = band
    mask = (wavelengths >= lower) & (wavelengths <= upper)
    if not np.any(mask):
        raise ValueError(f"No wavelengths found in band {band}")
    return X[:, mask], wavelengths[mask]


# PCA-based helpers intentionally removed to ensure downstream pipelines operate
# directly on raw spectral matrices.
# If PCA is required in future, reintroduce a dedicated preprocessing step
# that fits on clean-only baselines and persists the transform explicitly.