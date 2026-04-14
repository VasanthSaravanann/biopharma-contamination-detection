# Gap Analysis: Paper Claims vs Code Reality

**Date:** 2026-04-14
**Paper:** "Multimodal Sensor Fusion and Generative Diffusion Models for Real-Time Biomanufacturing Anomaly Detection"
**Authors:** Vasanth S, Varsha E, Keerthana S, Swathi R B (IIT Madras et al.)
**Repository:** `/run/media/sham/AI_/ai-stack/projects/biopharma-contamination-detection/`

---

## Executive Summary

The codebase is a **computational proof-of-concept with physics-based simulation only**. No real experimental data has been integrated. The core ML pipeline (data generation → feature extraction → anomaly detection → validation → visualization) is structurally complete and runnable, but **every quantitative performance claim rests on simulated data**. The paper's title claims "Multimodal Sensor Fusion" but the code only processes simulated UV-Vis spectra — no actual sensor fusion occurs. The paper claims a "Deep Convolutional Autoencoder" but the code implements a standard fully-connected (MLP) autoencoder.

---

## Claim-by-Claim Verification Table

| # | Paper Claim | Code Status | Actual Result | Gap |
|---|-------------|-------------|---------------|-----|
| 1 | **ROC-AUC 0.98 ensemble** | ⚠️ Simulated | `run_minimal_experiment.py` produces ROC-AUC **1.0** for both IF and OCSVM (not 0.98). The ensemble is configured but **never trained** (`train_ensemble: false` in config). The 0.98 figure appears only in PROPOSAL_V2 as a target, not as measured output. | 🔴 The 0.98 ensemble AUC is **not reproduced by any script**. Individual models score 1.0 on simulated data (expected — simulated contamination is perfectly separable). No ensemble actually runs end-to-end. |
| 2 | **Sensitivity 98%, Specificity 98.5%** | ⚠️ Simulated | Actual run: Sensitivity **100%**, Specificity **98.7%** (IF) and **93%** (OCSVM). These are on **simulated data only**. The proposal claims 98%/98.5% as ensemble targets. | 🔴 Numbers differ from actual run output. The specific 98%/98.5% pair is not produced by any current execution. |
| 3 | **Detection at 10 CFU/mL** | ❌ Not achieved | Detection rate at 10 CFU/mL: **0.00%** in actual run. All CFU levels show 0% detection rate due to a bug in the threshold calculation (`np.percentile(if_scores[:1000], 95)` uses wrong index range). The proposal claims 10 CFU/mL detection but code does not achieve it. | 🔴 **Critical gap.** The detection limit analysis is broken — returns 0% for all levels. Even if fixed, the simulated contamination signal at 10 CFU/mL is extremely weak (log-linear scaling makes low-CFU spectra nearly identical to clean). |
| 4 | **MH-DDPM JSD < 0.1, MMD p > 0.05** | ⚠️ Code exists, never validated | `validation.py` implements both MMD and JSD calculators — **complete implementations**. However, `run_pipeline.py` fails before reaching validation (seaborn import error). `run_minimal_experiment.py` does not call MMD/JSD at all. The paper claims these values were achieved, but **no script actually computes them**. | 🔴 The metrics are implemented but **never executed against any data**. The claimed values (JSD < 0.1, MMD p > 0.05) are aspirational, not measured. |
| 5 | **30-minute detection window** | ⚠️ Simulated only | `validation.py` has `DetectionWindowSimulator` class with exponential growth modeling. It's a **simulation of time**, not real temporal data. The 3.7-second runtime of the minimal experiment is computation time, not detection latency. | 🟡 The 30-minute window is modeled via growth kinetics (exponential growth equation), but no real time-course experiment validates it. The code simulates what *would* happen over 30 minutes — it doesn't measure it. |
| 6 | **Weighted ensemble scoring** | ⚠️ Implemented, not trained | `EnsembleAnomalyDetector` class in `anomaly_detection.py` is **fully implemented** with weights `{iforest: 1.0, ocsvm: 1.0, autoencoder: 2.0}` and normalized scoring. But `train_ensemble: false` in pipeline config, and the minimal experiment doesn't use it at all. | 🟡 Code is complete but never exercised. The weighting mechanism uses fixed defaults, not "learned via cross-validation" as the proposal claims. |
| 7 | **6 compendial species modeled** | ✅ Complete | `data_simulation.py`: E_coli, B_subtilis, P_aeruginosa, C_albicans, A_niger, Mycoplasma. `literature_based_simulation.py`: E_coli, S_aureus, B_subtilis, P_aeruginosa, C_albicans, A_brasiliensis. Both have 6 species with literature-derived parameters. | ✅ **Verified.** All 6 species have distinct spectral signatures, growth parameters, and scattering coefficients. |
| 8 | **Deep Convolutional Autoencoder** | ❌ FALSE — MLP Autoencoder | `anomaly_detection.py` line 239: `DeepAutoencoder` uses `nn.Linear` layers only (fully-connected MLP). **Zero Conv1d/Conv2d layers anywhere in the codebase.** The paper says "Deep Convolutional Autoencoder" — this is factually incorrect. | 🔴 **Major discrepancy.** The paper claims convolutional architecture; the code uses a standard feedforward autoencoder. This is not a minor naming issue — convolutions and MLPs have fundamentally different inductive biases for spectral data. |
| 9 | **Multimodal sensor fusion** | ❌ Not implemented | The paper title says "Multimodal Sensor Fusion." The code processes **only UV-Vis spectra**. There is no code that fuses spectroscopy data with pH, DO, temperature, or any other sensor modality. `feature_extraction.py` extracts features from spectra alone. The `ProcessConditions` class exists but is only used as metadata, not as fused input features. | 🔴 **Major gap.** The term "multimodal" in the paper title is unsupported by code. No sensor fusion occurs. Process variables (pH, DO, temp) are stored as columns but never combined with spectral features in a fused model. |
| 10 | **Real ambr bioreactor data processing** | ❌ Not present | The proposal (PROPOSAL_V2) states "Real bioreactor process data acquired — FCIC_AMBR_05.zip (440MB)." **The FCIC_AMBR_05 data is not in this repository.** The `data/raw/`, `data/processed/`, and `data/synthetic/` directories are all empty. No Python file reads, parses, or processes ambr format data (.pegpro, .csv from ambr 250). | 🔴 **Critical gap.** The real bioreactor data exists (per proposal) but has not been integrated into the codebase. Zero ambr processing code exists. |

---

## Stubs / Incomplete Implementations

### 🔴 Critical

1. **Ensemble scoring not executed** — `EnsembleAnomalyDetector` is fully coded but `train_ensemble: false` in all configs. The paper's primary claim (ROC-AUC 0.98 ensemble) is never validated by any script.

2. **Detection limit analysis broken** — `run_minimal_experiment.py` line ~275: threshold calculated as `np.percentile(if_scores[:1000], 95)` but `if_scores[:1000]` contains mixed clean+contaminated scores, producing a threshold that classifies everything as anomalous, yielding 0% detection at all CFU levels.

3. **No real data integration** — All three data directories (`data/raw/`, `data/processed/`, `data/synthetic/`) are empty. Every result comes from `UVVisSpectraGenerator` which is a physics-based simulator, not real measurements.

4. **ambr bioreactor data processing absent** — Despite the proposal claiming 440MB of FCIC_AMBR_05 data has been acquired, there is zero code to read, parse, normalize, or analyze this data.

5. **run_pipeline.py crashes** — Fails with `ModuleNotFoundError: No module named 'seaborn'` at import time. Even after installing seaborn, the full pipeline has not been successfully executed end-to-end.

### 🟡 Partially Complete

6. **MH-DDPM never trained on real data** — `mh_ddpm.py` is a complete DDPM implementation (~776 lines) with conditioning, EMA, and synthetic generation. But it has only ever been tested on simulated spectra from `data_simulation.py`. The claimed JSD < 0.1 and MMD p > 0.05 have never been computed.

7. **Detection window simulation** — `DetectionWindowSimulator` in `validation.py` models exponential growth and spectral evolution over 30 minutes, but uses a simplified additive noise model (`spectral_change = contamination_factor * np.log10(grown_cfus + 1)`) rather than actual time-series data.

8. **Deep Autoencoder** — `AutoencoderDetector` is a complete PyTorch implementation with early stopping, gradient clipping, and learning rate scheduling. It works on simulated data. But it's an MLP, not convolutional.

9. **Virtual spike-in experiments** — `virtual_spike_in.py` (682 lines) is a complete simulation of spike-in methodology with growth modeling, time-course sampling, and detection analysis. But it uses simulated spectra, not real measurements.

10. **Literature-based simulation** — `literature_based_simulation.py` (739 lines) is the most thorough module, with literature-derived parameters for 6 compendial organisms, Beer-Lambert law modeling, and Rayleigh-Mie scattering. This is well-implemented but remains a simulation.

### 💭 Minor

11. **`AnomalyDetectionBase` has abstract methods** — `fit()`, `predict()`, `predict_proba()`, `save()`, `load()` all raise `NotImplementedError`. This is fine — they're meant to be abstract — but there's no `abc.ABC` decorator or `@abstractmethod` usage.

12. **No config YAML loading in minimal experiment** — `run_minimal_experiment.py` hardcodes all parameters and doesn't use `config/pipeline_config.yaml`.

13. **Visualization module not reachable** — `run_pipeline.py` imports `seaborn` (missing) and `visualization.py` which requires seaborn. Figures are only generated by `run_minimal_experiment.py`.

---

## What Actually Works

| Component | Status | Evidence |
|-----------|--------|----------|
| `data_simulation.py` — UV-Vis spectra generator | ✅ Working | Generates 601-wavelength spectra for 6 species with process variation, batch effects, instrument variation |
| `feature_extraction.py` — Spectral features | ✅ Working | Extracts 50+ features across 7 categories (absorbances, ratios, peaks, biomass, derivatives, statistics, integrals) |
| `anomaly_detection.py` — Isolation Forest | ✅ Working | Trains and predicts; ROC-AUC 1.0 on simulated data |
| `anomaly_detection.py` — One-Class SVM | ✅ Working | Trains and predicts; ROC-AUC 1.0 on simulated data |
| `anomaly_detection.py` — Deep Autoencoder (MLP) | ✅ Code complete | Full PyTorch implementation with training loop, early stopping, gradient clipping |
| `anomaly_detection.py` — Ensemble detector | ✅ Code complete | Weighted scoring with normalization, but never executed |
| `literature_based_simulation.py` — Literature params | ✅ Working | 6 organisms with literature-derived absorbance values, growth rates, scattering parameters |
| `virtual_spike_in.py` — Virtual experiments | ✅ Working | Complete spike-in simulation with growth modeling, time-course, detection analysis |
| `validation.py` — MMD calculator | ✅ Code complete | RBF kernel, median heuristic gamma, permutation test for p-value |
| `validation.py` — JSD calculator | ✅ Code complete | Histogram-based 1D + random projection multivariate |
| `validation.py` — Detection metrics | ✅ Code complete | Sensitivity, specificity, ROC-AUC, detection limit analysis, bootstrap CIs |
| `validation.py` — Detection window simulator | ✅ Code complete | Exponential growth model, time-dependent detection evaluation |
| `visualization.py` — Plotting utilities | ✅ Code complete | Spectra plots, ROC curves, confusion matrices, composite figures |
| `run_minimal_experiment.py` | ✅ Runs successfully | Completes in ~4 seconds, generates 4 figures + report |
| `tests/test_pipeline.py` | ✅ 10 tests exist | Covers data simulation, feature extraction, anomaly detection, integration |
| `config/pipeline_config.yaml` | ✅ Complete | All hyperparameters documented |

---

## What Needs Real Data

| Component | Why Real Data Is Required |
|-----------|--------------------------|
| **ROC-AUC, Sensitivity, Specificity claims** | Current metrics are on simulated data where contamination signatures are perfectly separable from clean. Real UV-Vis spectra will have noise, matrix effects, and overlapping signatures that will reduce performance. |
| **Detection limit (10 CFU/mL)** | The simulated Beer-Lambert model produces clean log-linear scaling. Real instruments have detection limits, stray light, and baseline drift that will obscure low-CFU signals. |
| **MH-DDPM validation (JSD, MMD)** | These metrics compare synthetic vs real distributions. Without real spectra, there's nothing to compare against. |
| **30-minute detection window** | The growth model assumes ideal exponential growth. Real microbial lag phases, media effects, and instrument sampling rates will differ. |
| **Multimodal sensor fusion** | Requires actual co-registered pH, DO, temperature, CER, OUR, gas flow data from bioreactors. |
| **Ensemble weighting** | Cross-validation weight optimization requires real data with realistic class overlap. |
| **ambr bioreactor integration** | 440MB of FCIC_AMBR_05 data needs parsing, normalization, batch alignment, and feature extraction. |
| **MIT UV-Vis dataset** | Email sent to Rajeev J. Ram (MIT) — pending response. Critical for real spectral validation. |

---

## Publishable Now vs Needs Work

### ✅ Publishable Now (with honest framing)

These components are solid and can be published **if framed as computational proof-of-concept**:

1. **Physics-based UV-Vis simulation methodology** — The Beer-Lambert + Rayleigh-Mie scattering approach with literature-derived parameters is novel and well-implemented.
2. **Virtual spike-in experimental design** — The computational framework for simulating spike-in experiments with growth modeling is complete and reproducible.
3. **Literature-based hybrid validation approach** — The methodology of deriving spectral parameters from published papers and validating against literature detection limits is a legitimate contribution.
4. **Feature extraction pipeline** — 50+ features across 7 categories from UV-Vis spectra is thorough and well-documented.
5. **Anomaly detection model implementations** — IF, OCSVM, and AE are all complete with proper training/evaluation pipelines.

**Framing required:** "This is a computational proof-of-concept. All results are from simulated data validated against literature parameters. Wet-lab experiments are needed to confirm performance claims."

### 🔴 Needs Significant Work Before Publication

These claims **cannot** be published as-is:

1. **"ROC-AUC 0.98"** — No script produces this number. Individual models score 1.0 on simulated data. The ensemble never runs. On real data, performance will be lower.
2. **"Sensitivity 98%, Specificity 98.5%"** — Not reproduced by any script. Actual run shows 100%/98.7% (IF) and 100%/93% (OCSVM).
3. **"Detection at 10 CFU/mL"** — Detection rate is 0% at all CFU levels in actual execution. The analysis code has a bug.
4. **"Deep Convolutional Autoencoder"** — The code implements an MLP autoencoder. Either change the paper or change the code to use Conv1d layers.
5. **"Multimodal Sensor Fusion"** — No sensor fusion exists in code. Either implement it or remove the claim from the title.
6. **"MH-DDPM JSD < 0.1, MMD p > 0.05"** — These values have never been computed. The validation code exists but has never been run.
7. **"Real ambr bioreactor data"** — The data exists (per proposal) but zero processing code has been written.
8. **"30-minute detection window"** — This is a growth model simulation, not an empirical measurement.

### 🟡 Needs Moderate Work

9. **Ensemble training** — Code is complete; just needs to be enabled in config and run. But the weighting should use CV-learned weights, not fixed defaults.
10. **Detection limit bug** — The threshold calculation in `run_minimal_experiment.py` needs fixing. Should use clean-only scores for threshold.
11. **run_pipeline.py dependency fix** — Install seaborn or make it optional. Then the full pipeline can run end-to-end.
12. **Autoencoder architecture** — If the paper claims "convolutional," add Conv1d layers to the encoder/decoder. For 1D spectral data, Conv1d is actually more appropriate than MLP.

---

## Recommended Priority Order

1. **Fix the paper-code mismatch** — Either update the paper to match what the code actually does (MLP autoencoder, no sensor fusion, simulated data only), or update the code to match the paper (add Conv1d, implement sensor fusion, integrate real data).
2. **Fix detection limit analysis bug** — The threshold calculation is wrong. Fix it, re-run, and report actual detection rates.
3. **Enable and run ensemble** — Set `train_ensemble: true`, run it, and report actual ensemble metrics.
4. **Run MH-DDPM validation** — Train the DDPM, generate synthetic data, compute JSD and MMD. Report actual values.
5. **Integrate ambr data** — Write parser for FCIC_AMBR_05 .pegpro/.csv files. Extract time-series features.
6. **Follow up on MIT dataset** — The email to Rajeev J. Ram is the highest-leverage remaining task. Real UV-Vis spectra will validate or invalidate every claim.

---

*Analysis based on reading all 9 source files in `src/`, 3 runner scripts, proposal, README, experiment summary, test suite, config, and actual execution output. Paper text extracted via pdftotext.*
