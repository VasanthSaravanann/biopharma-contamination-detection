# API Documentation

## Overview

This document provides detailed API documentation for the Biopharmaceutical Contamination Detection System.

---

## Data Simulation (`src/data_simulation.py`)

### `UVVisSpectraGenerator`

Main class for generating UV-Vis spectra.

#### `__init__(seed: Optional[int] = None)`

Initialize the spectra generator.

**Args:**
- `seed`: Random seed for reproducibility

**Example:**
```python
generator = UVVisSpectraGenerator(seed=42)
```

#### `generate_spectrum(contaminant_type, inoculum_level=None, ...)`

Generate a single UV-Vis spectrum.

**Args:**
- `contaminant_type`: ContaminantType enum value
- `inoculum_level`: CFU/mL (required for contaminated)
- `process_conditions`: ProcessConditions object
- `add_noise`: bool, add instrument noise
- `noise_level`: float, noise standard deviation

**Returns:**
- Tuple of (wavelengths, absorbance) arrays

**Example:**
```python
wavelengths, absorbance = generator.generate_spectrum(
    ContaminantType.BACTERIA_E_COLI,
    inoculum_level=100
)
```

#### `generate_dataset(n_clean, n_contaminated_per_type, ...)`

Generate complete dataset.

**Args:**
- `n_clean`: Number of clean spectra
- `n_contaminated_per_type`: Spectra per contaminant type
- `inoculum_levels`: List of CFU/mL levels
- `include_process_variation`: Add process variation
- `seed`: Random seed

**Returns:**
- pandas DataFrame with spectra and metadata

**Example:**
```python
df = generator.generate_dataset(
    n_clean=1000,
    n_contaminated_per_type=200,
    inoculum_levels=[10, 25, 50, 100, 250, 500, 1000]
)
```

### `ContaminantType` (Enum)

Available contaminant types:
- `BACTERIA_E_COLI`
- `BACTERIA_B_SUBTILIS`
- `BACTERIA_P_AERUGINOSA`
- `FUNGI_C_ALBICANS`
- `FUNGI_A_NIGER`
- `MYCOPLASMA`
- `CLEAN`

### `ProcessConditions`

Dataclass for process parameters.

**Fields:**
- `temperature`: float (°C)
- `ph`: float
- `dissolved_oxygen`: float (% saturation)
- `batch_age`: float (hours)
- `media_type`: str

---

## Feature Extraction (`src/feature_extraction.py`)

### `SpectralFeatureExtractor`

Extract features from UV-Vis spectra.

#### `__init__(config=None, wavelengths=None)`

**Args:**
- `config`: FeatureConfig object
- `wavelengths`: Wavelength array

#### `extract_all_features(df, include_derivatives=True, ...)`

Extract features from DataFrame.

**Args:**
- `df`: DataFrame with absorbance columns
- `include_derivatives`: Include derivative features
- `include_statistical`: Include statistical features

**Returns:**
- DataFrame with extracted features

**Example:**
```python
extractor = SpectralFeatureExtractor()
features_df = extractor.extract_all_features(df)
```

#### `extract_single_spectrum(row, ...)`

Extract features from single spectrum.

**Returns:**
- Dictionary of features

### Extracted Features

| Feature | Description |
|---------|-------------|
| `abs_260`, `abs_280`, `abs_430`, `abs_600` | Key wavelength absorbances |
| `a260_a280_ratio` | Nucleic acid / protein ratio |
| `uv_vis_ratio` | UV to visible ratio |
| `n_peaks` | Number of detected peaks |
| `max_peak_height` | Maximum peak height |
| `total_peak_area` | Integrated peak area |
| `scattering_exponent` | Light scattering exponent |
| `biomass_index` | Biomass indicator |
| `mean_abs`, `std_abs` | Statistical features |
| `first_deriv_max`, `second_deriv_std` | Derivative features |
| `integral_uv_250_300`, etc. | Region integrals |

---

## Anomaly Detection (`src/anomaly_detection.py`)

### `ModelConfig`

Configuration for all anomaly detection models.

**Key Fields:**
- `random_state`: int
- `contamination`: float (expected anomaly rate)
- `iforest_n_estimators`: int
- `ocsvm_nu`: float
- `ae_hidden_layers`: List[int]
- `ae_latent_dim`: int
- `ae_epochs`: int
- `ae_learning_rate`: float

### `IsolationForestDetector`

Isolation Forest anomaly detector.

#### `fit(X, y=None)`

Train on nominal data.

#### `predict(X)`

Predict anomaly labels (-1 = anomaly, 1 = normal).

#### `predict_proba(X)`

Get anomaly scores (higher = more anomalous).

**Example:**
```python
config = ModelConfig()
detector = IsolationForestDetector(config)
detector.fit(X_clean)
scores = detector.predict_proba(X_test)
```

### `AutoencoderDetector`

Deep Autoencoder anomaly detector.

#### `fit(X, y=None, val_split=0.1, verbose=True)`

Train autoencoder on nominal data.

#### `get_reconstruction(X)`

Get original and reconstructed spectra.

#### `get_latent_representation(X)`

Get latent space encoding.

### `EnsembleAnomalyDetector`

Ensemble of all detectors.

#### `fit(X, y=None, verbose=True)`

Train all detectors.

#### `predict_proba(X)`

Get weighted ensemble scores.

### `evaluate_detector(detector, X_test, y_test, ...)`

Evaluate detector performance.

**Returns:**
- Dictionary with ROC-AUC, F1, optimal threshold, etc.

---

## MH-DDPM (`src/mh_ddpm.py`)

### `DDPMConfig`

Configuration for MH-DDPM model.

**Key Fields:**
- `input_dim`: int (number of wavelengths)
- `hidden_dim`: int
- `n_timesteps`: int
- `beta_start`, `beta_end`: float
- `n_contaminant_types`: int
- `inoculum_levels`: int
- `epochs`: int
- `batch_size`: int

### `MHDDPM`

Multimodal Hierarchical DDPM model.

#### `fit(spectra, contaminant_types, inoculum_levels, ...)`

Train the diffusion model.

**Args:**
- `spectra`: (n_samples, n_wavelengths) array
- `contaminant_types`: contaminant type indices
- `inoculum_levels`: inoculum level indices

**Returns:**
- Training history dictionary

#### `sample(n_samples, contaminant_type, inoculum_level, ...)`

Generate synthetic spectra.

**Returns:**
- Generated spectra tensor

### `SyntheticDataGenerator`

High-level interface for synthetic data generation.

#### `generate(n_samples, contaminant_type, inoculum_level, ...)`

Generate synthetic spectra.

#### `generate_dataset(n_per_class, progress=False)`

Generate balanced synthetic dataset.

**Returns:**
- Tuple of (spectra, contaminant_types, inoculum_levels)

---

## Validation (`src/validation.py`)

### `ValidationConfig`

Validation configuration.

**Key Fields:**
- `target_detection_limit`: float (CFU/mL)
- `target_sensitivity`: float
- `target_specificity`: float
- `n_bootstrap_iterations`: int

### `ValidationPipeline`

Complete validation pipeline.

#### `run_full_validation(detector, X_train, X_test, y_test, ...)`

Run complete validation.

**Returns:**
- Dictionary with all validation results

**Example:**
```python
config = ValidationConfig(
    target_detection_limit=10,
    target_sensitivity=0.90
)
pipeline = ValidationPipeline(config)
results = pipeline.run_full_validation(
    detector, X_clean, X_test, y_test, inoculum_levels
)
```

### `generate_validation_report(results, output_path)`

Generate human-readable validation report.

---

## Visualization (`src/visualization.py`)

### `SpectraVisualizer`

#### `plot_spectra_by_contaminant(df, wavelengths, save_path)`

Plot spectra grouped by contaminant type.

#### `plot_multiple_spectra(wavelengths, spectra_dict, ...)`

Compare multiple spectra.

### `AnomalyVisualizer`

#### `plot_anomaly_scores(y_true, scores, threshold, ...)`

Plot anomaly score distribution.

#### `plot_roc_curve(y_true, scores, ...)`

Plot ROC curve with AUC.

#### `plot_precision_recall(y_true, scores, ...)`

Plot precision-recall curve.

#### `plot_confusion_matrix(y_true, y_pred, ...)`

Plot confusion matrix.

### `DetectionLimitVisualizer`

#### `plot_detection_by_level(detection_results, ...)`

Plot detection rate by inoculum level.

#### `plot_time_dependent_detection(metrics_over_time, ...)`

Plot detection performance over time.

### `create_all_visualizations(df, wavelengths, results, ...)`

Create all standard visualizations.

**Returns:**
- Dictionary of saved figure paths

---

## Pipeline (`run_pipeline.py`)

### `ContaminationDetectionPipeline`

Main pipeline orchestrator.

#### `run()`

Execute complete pipeline.

**Returns:**
- Complete results dictionary

### Command Line Interface

```bash
# Run demo
python run_pipeline.py --demo

# Run full pipeline
python run_pipeline.py --config config/pipeline_config.yaml

# Specify output directory
python run_pipeline.py --output my_output
```

---

## Data Structures

### Input DataFrame Format

| Column | Type | Description |
|--------|------|-------------|
| `spectrum_id` | str | Unique identifier |
| `contaminant_type` | str | Species name |
| `inoculum_level` | int | CFU/mL |
| `label` | int | 0=clean, 1=contaminated |
| `abs_200` to `abs_800` | float | Absorbance values |
| `temperature` | float | Process temperature |
| `ph` | float | Process pH |
| `dissolved_oxygen` | float | DO % |
| `batch_age` | float | Hours |
| `batch_id` | int | Batch identifier |
| `instrument_id` | int | Instrument identifier |

### Results Dictionary Structure

```python
{
    'data_generation': {
        'n_samples': int,
        'n_clean': int,
        'n_contaminated': int,
        'raw_data_path': str
    },
    'feature_extraction': {
        'n_features': int,
        'feature_columns': List[str],
        'features_path': str
    },
    'model_training': {
        'models_trained': List[str],
        'evaluation_results': Dict,
        'best_model': str
    },
    'synthetic_generation': {
        'n_synthetic': int,
        'synthetic_path': str
    },
    'validation': {
        'summary': Dict,
        'report_path': str
    },
    'visualization': {
        'figures': Dict[str, str],
        'figures_dir': str
    }
}
```

---

## Error Handling

All modules raise appropriate exceptions:

- `ValueError`: Invalid parameters
- `RuntimeError`: Model not fitted
- `FileNotFoundError`: Missing data files
- `ImportError`: Missing optional dependencies

---

## Performance Notes

- **Memory**: Dataset size scales with n_samples × n_wavelengths
- **Training**: Autoencoder requires GPU for large datasets
- **Inference**: All models support batch prediction
- **DDPM**: Generation is sequential; use fewer timesteps for speed
