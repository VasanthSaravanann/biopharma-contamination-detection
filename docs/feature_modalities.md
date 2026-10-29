**Feature Modalities Inventory & Expansion Plan**

Current modalities
------------------
- UV-Vis spectra (per-sample CSVs)
- Scalar process variables (pH, DO, Temperature, pump rates, gas flows)
- Metabolomics (mzData XML files)
- Chromatography (`.d` runs)

Recommended expansions
----------------------
1. Dissolved CO2 (off-gas CO2% and headspace CO2 trends) — available in `data/FCIC_AMBR_05` as off-gas variables; integrate as time-series features.
2. Metabolite profiles (LC-MS/GC-MS derived features) — add aggregated metabolite intensities (AUC) per sample and link by sample_id.
3. Conductivity / Osmolality — where available from AMBR logs, include as process scalar.
4. Optical density / turbidity traces (from spectrophotometer) — to cross-check UV-Vis features.
5. Sensor health signals (calibration flags, sensor temperature) — use for filtering and drift detection.

Implementation steps
--------------------
1. Create `src/data_integration.py` loaders for CO2 and LC-MS aggregated features.
2. Add modality mapping in `data/processed/dataset_metadata.csv` with `has_metabolomics,has_co2` boolean columns.
3. Update the feature-engineering pipeline to produce per-sample fused feature vectors.
