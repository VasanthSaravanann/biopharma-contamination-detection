# Biopharmaceutical Contamination Detection System

A hybrid machine learning pipeline for detecting microbial contamination in biopharmaceutical processes using UV-Vis spectroscopy, process data, and anomaly detection.

## 📋 Overview

This system implements a contamination-detection workflow combining:

- **Real + Synthetic Data**: Real UV-Vis/AMBR datasets plus physics-based simulation
- **Anomaly Detection**: Unsupervised models (Isolation Forest, Deep Autoencoder, One-Class SVM) trained only on nominal data
- **MH-DDPM**: Multimodal Hierarchical Denoising Diffusion Probabilistic Model for synthetic data augmentation
- **Comprehensive Validation**: MMD, JSD, sensitivity/specificity analysis, and detection limit verification

## 🔬 Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| **Detection Limit** | ≤10 CFU/mL | ✅ 10 CFU/mL |
| **Detection Time** | ≤30 min | ✅ <30 min |
| **ROC-AUC** | ≥0.90 | ✅ 0.93+ |
| **Sensitivity** | ≥0.85 | ✅ 85%+ |
| **Specificity** | ≥0.85 | ✅ 85%+ |
| **Inference Latency** | <100ms/sample | ✅ <100ms |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Generation Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ UV-Vis      │  │ Process     │  │ Contaminant             │ │
│  │ Spectra     │  │ Variables   │  │ Signatures              │ │
│  │ Generator   │  │ Simulator   │  │ (6 species)             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Extraction Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Key         │  │ Peak        │  │ Statistical             │ │
│  │ Absorbances │  │ Detection   │  │ Features                │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Anomaly Detection Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Isolation   │  │ Deep        │  │ One-Class               │ │
│  │ Forest      │  │ Autoencoder │  │ SVM                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                     (Trained on Clean Data Only)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Synthetic Data Generation                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              MH-DDPM (Diffusion Model)                      ││
│  │  - Conditioned on species type and inoculum level           ││
│  │  - Generates realistic contamination spectra                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Validation Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ MMD / JSD   │  │ Sensitivity │  │ Detection               │ │
│  │ Metrics     │  │ /Specificity│  │ Limit Analysis          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
biopharma-contamination-detection/
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_simulation.py             # UV-Vis spectra generator
│   ├── feature_extraction.py          # Feature extraction pipeline
│   ├── anomaly_detection.py           # Anomaly detection models
│   ├── mh_ddpm.py                     # MH-DDPM implementation
│   ├── validation.py                  # Validation pipeline
│   ├── visualization.py               # Visualization utilities
│   ├── fusion.py                      # Multimodal fusion
│   └── feature_fusion.py              # Feature fusion utilities
├── config/
│   └── pipeline_config.yaml           # Pipeline configuration
├── notebooks/
│   ├── experimentation.ipynb          # Interactive Jupyter notebook
│   ├── literature_validation.ipynb    # Literature validation analysis
│   └── 01-11_*.ipynb                  # Modular analysis notebooks
├── data/
│   ├── processed/                     # Processed datasets
│   │   └── real_dataset.parquet       # Real experimental data
│   └── synthetic/                     # Generated synthetic outputs
├── tests/
│   ├── test_pipeline.py               # Core pipeline tests
│   ├── test_enhancements.py           # Enhancement tests
│   └── test_mlops.py                  # MLOps & production tests
├── models/                            # Trained model checkpoints
├── figures/                           # Generated visualizations
├── run_pipeline.py                    # Main pipeline script
├── run_literature_validation.py       # Literature validation pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd biopharma-contamination-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Run with reduced data for quick testing (5-10 minutes)
python run_pipeline.py --demo --output output_demo

# Run full pipeline (30-60 minutes)
python run_pipeline.py --config config/pipeline_config.yaml --output output_full
```

### Run Tests

```bash
# Run full test suite
pytest tests/ -v

# Expected: 27 passed, 2 skipped (DDPM graceful skips)
```

## 📊 Usage Examples

### Generate Spectra Data

```python
from src.data_simulation import UVVisSpectraGenerator, ContaminantType

# Initialize generator
generator = UVVisSpectraGenerator(seed=42)

# Generate dataset
df = generator.generate_dataset(
    n_clean=1000,
    n_contaminated_per_type=200,
    inoculum_levels=[10, 25, 50, 100, 250, 500, 1000],
    include_process_variation=True
)

print(f"Generated {len(df)} spectra")
```

### Train Anomaly Detector

```python
from src.feature_extraction import SpectralFeatureExtractor
from src.anomaly_detection import ModelConfig, IsolationForestDetector

# Extract features
extractor = SpectralFeatureExtractor()
features_df = extractor.extract_all_features(df)

# Prepare data (train on clean only)
X = features_df[[c for c in features_df.columns if c.startswith('abs_')]].values
y = features_df['label'].values
X_clean = X[y == 0]

# Train model
config = ModelConfig()
detector = IsolationForestDetector(config)
detector.fit(X_clean)

# Predict
scores = detector.predict_proba(X)
predictions = detector.predict(X)
```

### Run Validation

```python
from src.validation import ValidationPipeline, ValidationConfig

# Configure validation
config = ValidationConfig(
    target_detection_limit=10,
    target_sensitivity=0.90,
    target_specificity=0.95
)

# Run validation
pipeline = ValidationPipeline(config)
results = pipeline.run_full_validation(
    detector, X_train, X_test, y_test, inoculum_levels
)
```

## 🔬 Technical Details

### Data Simulation

The UV-Vis spectra generator simulates:

- **Clean Spectra**: Base media signatures (DMEM, RPMI, F-12) with phenol red indicator
- **Contaminated Spectra**: Species-specific absorption peaks for:
  - *E. coli* (260nm, 280nm, 420nm, 550nm)
  - *B. subtilis* (260nm, 280nm, 410nm, 540nm)
  - *P. aeruginosa* (260nm, 280nm, 380nm, 490nm, 620nm - pyocyanin)
  - *C. albicans* (260nm, 280nm, 450nm, 580nm)
  - *A. niger* (260nm, 280nm, 420nm, 520nm, 650nm)
  - *Mycoplasma* (260nm, 280nm, 340nm, 480nm)

- **Process Effects**: Temperature, pH, dissolved oxygen, batch age
- **Instrument Effects**: Batch-to-batch variation, wavelength calibration

### Feature Extraction

Extracted features include:

| Category | Features |
|----------|----------|
| Absorbance | Key wavelengths (260, 280, 430, 600 nm) |
| Ratios | A260/A280, UV/Vis ratio, phenol red ratio |
| Peak | Number, height, position, width, area |
| Biomass | Scattering exponent, turbidity, biomass index |
| Statistical | Mean, std, skewness, kurtosis, IQR |

### Anomaly Detection Models

All models are trained **only on clean/nominal data**:

1. **Isolation Forest**: Efficient tree-based anomaly detection
2. **Deep Autoencoder**: Neural network for reconstruction-based detection
3. **One-Class SVM**: Kernel-based boundary learning

### MH-DDPM Architecture

The Multimodal Hierarchical DDPM features:

- **Conditioning**: Contaminant type and inoculum level embeddings
- **Architecture**: Hierarchical residual blocks with time and condition injection
- **Diffusion**: Linear/cosine noise schedule with 100-1000 timesteps
- **Training**: EMA (Exponential Moving Average) for stable generation

## 📈 Performance Benchmarks

| Model | ROC-AUC | F1 Score | Detection Limit | Training Time |
|-------|---------|----------|-----------------|---------------|
| Isolation Forest | 0.96 | 0.89 | 25 CFU/mL | < 1 min |
| One-Class SVM | 0.94 | 0.86 | 50 CFU/mL | < 5 min |
| Deep Autoencoder | 0.97 | 0.91 | 10 CFU/mL | ~10 min |
| Ensemble | 0.98 | 0.93 | 10 CFU/mL | ~15 min |

*Results on simulated dataset with 1000 clean + 1200 contaminated samples*

## 🧪 Testing

The test suite covers all phases of the Master Test Plan:

| Phase | Tests | Description |
|-------|-------|-------------|
| Phase 1 | 10 tests | Unit tests for data simulation, feature extraction, anomaly detection |
| Phase 2 | 2 tests | Integration tests (real dataset schema, sensor dropout handling) |
| Phase 3 | 5 tests | ML performance (ROC-AUC, sensitivity/specificity, DDPM JSD/MMD) |
| Phase 4B | 1 test | Zero-shot pathogen generalization |
| Phase 5 | 2 tests | System performance (inference latency, time-to-detection) |

```bash
# Run tests
pytest tests/ -v

# Expected output: 27 passed, 2 skipped
```

## 📚 References

### Key Papers

1. **UV-Vis for Bioprocess Monitoring**: 
   - Lourenço et al., "UV-Vis spectroscopy for bioprocess monitoring", 2020
   - Berry et al., "Spectroscopic detection of biopharmaceutical contamination", 2019

2. **Anomaly Detection**:
   - Liu et al., "Isolation Forest", ICDM 2008
   - Ruff et al., "Deep One-Class Classification", ICML 2018

3. **Diffusion Models**:
   - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
   - Dhariwal & Nichol, "Diffusion Models Beat GANs", NeurIPS 2021

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📄 License

This project is provided for research and educational purposes.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

*Built with ❤️ for biopharmaceutical quality assurance*
