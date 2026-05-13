# Simulation Provenance and Parameter Audit Trail

## Overview
This document provides transparency about the synthetic data generation used in all computational validation experiments. The entire dataset is generated using physics-based Beer-Lambert and Rayleigh-Mie models, NOT empirical wet-lab measurements.

## Critical Limitation
**All evaluation results are computational evidence only.** Results are based on synthetic UV-Vis spectra derived from theoretical optical models. Before any industrial deployment, validation on real fermentation data across multiple organisms and contamination types is mandatory.

---

## Beer-Lambert Law Implementation

### Formula
```
A(λ) = ε(λ) × c × b
```
Where:
- **A(λ)** = Absorbance at wavelength λ
- **ε(λ)** = Molar absorptivity (extinction coefficient)
- **c** = Concentration of absorbing species (M)
- **b** = Path length (cm), fixed at 1 cm in model

### Parameters Used

#### Organism-Specific Extinction Coefficients
Mapped to contaminant types in the simulation:

| Organism | ε₂₆₀ (M⁻¹cm⁻¹) | ε₂₈₀ (M⁻¹cm⁻¹) | ε₄₀₅ (M⁻¹cm⁻¹) | Source |
|----------|------|------|------|--------|
| E. coli | 0.65 | 0.70 | 0.12 | Encoded in `ContaminantType.E_COLI` |
| B. subtilis | 0.68 | 0.72 | 0.14 | Encoded in `ContaminantType.B_SUBTILIS` |
| P. aeruginosa | 0.70 | 0.75 | 0.16 | Encoded in `ContaminantType.P_AERUGINOSA` |
| C. albicans | 0.72 | 0.76 | 0.18 | Encoded in `ContaminantType.C_ALBICANS` |
| A. niger | 0.75 | 0.80 | 0.20 | Encoded in `ContaminantType.A_NIGER` |
| Mycoplasma | 0.60 | 0.65 | 0.10 | Encoded in `ContaminantType.MYCOPLASMA` |

**Note:** These values are illustrative and derived from typical literature values for nucleic acid and protein absorption. Actual values vary with specific media, pH, temperature, and growth phase.

#### CFU-to-Concentration Mapping
```python
# src/data_simulation.py, line ~450
# Assumed relationship: inoculum CFU/mL → approximate intracellular protein/nucleic acid concentration
concentration_M = (inoculum_level / 1e7) * 1e-5  # M
# This assumes ~10⁷ CFU/mL yields ~10⁻⁵ M equivalent absorbing species
```

**Justification:** Typical fermentation contamination occurs at 10–10⁶ CFU/mL. At 10 CFU/mL, the concentration of nucleic acids and proteins released/present is assumed to produce measurable absorbance in the Beer-Lambert model.

---

## Rayleigh-Mie Scattering Implementation

### Formula (Gaussian Approximation)
```
S(λ) = amplitude × exp(-(λ - center)² / (2 × width²))
```
Where:
- **S(λ)** = Scattering intensity at wavelength λ
- **amplitude** = Peak scattering magnitude (organism-specific)
- **center** = Peak wavelength (organism-specific, typically 400–600 nm)
- **width** = Spectral width (σ, Gaussian spread)

### Scattering Profile Parameters (by organism)

| Organism | Peak λ (nm) | Amplitude | Width (nm) | Justification |
|----------|------|---------|-------|------|
| E. coli | 450 | 0.15 | 80 | Rod-shaped, ~0.5 μm; strong Rayleigh |
| B. subtilis | 480 | 0.18 | 85 | Slightly larger rods; stronger scattering |
| P. aeruginosa | 470 | 0.16 | 82 | Motile rods; intermediate scattering |
| C. albicans | 520 | 0.22 | 100 | Larger cells (~3–5 μm); strong Mie |
| A. niger | 550 | 0.25 | 120 | Filamentous; extended scattering signature |
| Mycoplasma | 420 | 0.08 | 70 | Very small (~0.1–0.3 μm); weak Rayleigh |

**Note:** Rayleigh scattering dominates for small particles (d << λ); Mie scattering for larger organisms. The model uses simplified Gaussian approximations rather than full Lorenz-Mie theory.

---

## Inoculum Level Encoding

### Inoculum Levels (0–6 scale)
```python
# src/data_simulation.py, lines ~245–260
# Discrete levels map to CFU/mL ranges:
inoculum_level_to_cfu_range = {
    0: (0, "Sterile/Background"),
    1: (1, "1 CFU/mL"),
    2: (10, "10 CFU/mL (LOD target)"),
    3: (50, "50 CFU/mL"),
    4: (100, "100 CFU/mL"),
    5: (1000, "1000 CFU/mL"),
    6: (10000, "10K CFU/mL")
}
```

### Contamination Event Simulation
- **Contamination rate per sample:** 60% (only when `contamination=True`)
- **Random organism selection** from 6 types with uniform probability
- **Random inoculum level** from 1–6 (non-zero CFU, level 0 = clean control)
- **Noise added:** Gaussian, σ = 0.005 A (~0.5% relative absorbance noise)

---

## Process Condition Parameters

### Temperature & pH Variations
```python
# Encoded in ProcessConditions class (data_simulation.py, lines ~570–590)
# Optional: scale baseline absorption/scattering by ±5% per degree C or ±0.5 pH unit
# Currently NOT applied in baseline (flag: include_process_effects=False)
```

### Buffer & Media Effects
- **Media:** Fixed phosphate-buffered saline (PBS) assumed
- **Path length:** Fixed at 1 cm (cuvette standard)
- **Temperature:** Fixed at 25 °C (room temperature)
- **pH:** Fixed at 7.0 (neutral)

**Future Enhancement:** Parameterize media osmolarity, temperature, pH to test robustness.

---

## Spectral Range & Wavelength Grid

### Wavelength Coverage
- **Range:** 200–800 nm (UV-Vis region)
- **Resolution:** 1 nm increments → 601 wavelength bins
- **Total features:** 601 × 1 = 601-dimensional input to models

### Physical Rationale
- **200–280 nm:** Protein (aromatic amino acids: Trp, Tyr) and nucleic acid absorption (peak ~260 nm)
- **280–400 nm:** Secondary absorption; scattering begins to dominate
- **400–800 nm:** Primarily scattering and background; reduced absorption

---

## Validation Against Synthetic Benchmarks

### ROC-AUC & Performance Targets
- **Ensemble model:** Targets ROC-AUC ≥ 0.98 (threshold for synthetic validation)
- **PCA baseline:** Expected ROC-AUC ~0.82 (comparison baseline)
- **Limit-of-detection:** 10 CFU/mL (inoculum_level=2)

**Caveat:** These targets assume ideal noise levels and Beer-Lambert/Mie models accurately capture real contamination spectra. Real fermentation data will introduce **model mismatch** (e.g., spectral interference from media, biofilm formation, metabolite accumulation) that synthetic data cannot capture.

---

## Parameters Locked for Reproducibility

All experiments use:
- **Seed:** 42 (set via `np.random.seed(42)` and `torch.manual_seed(42)`)
- **Noise level:** σ = 0.005 A (1.0% relative on baseline)
- **Path length:** 1.0 cm
- **Temperature:** 25 °C (no per-sample variation)
- **pH:** 7.0 (no per-sample variation)
- **Sample count:** 200 clean, 1200 contaminated (6 types × 200 per type) in standard generation

### Hyperparameters Locked in Config
See `config/pipeline_config.yaml`:
```yaml
ae_epochs: 100
ae_latent_dim: 32
ocsvm_nu: 0.05
iforest_n_estimators: 200
```

---

## Audit Trail & Versioning

### Git Provenance
All simulation parameters are versioned in `src/data_simulation.py`. Changes to organism extinction coefficients, scattering profiles, or CFU mappings must be documented in commit messages with rationale.

### Reproducibility Checks
- Run `python scripts/verify_lod.py` to confirm LOD behavior is stable
- Run `python scripts/benchmark_latency.py` to confirm latency is within 42±6 ms
- Run `pytest tests/test_lod.py -v` to verify reproducible LOD trends

---

## Future Work: Real-Lab Validation

Before deployment:

1. **Collect wet-lab spectra** using the same UV-Vis instrument and path length (1 cm)
2. **Measure extinction coefficients** for real media (not PBS) at actual fermentation conditions
3. **Quantify model mismatch** by comparing synthetic spectra to real contamination spectra
4. **Retrain ensemble** on real data (or fine-tune pre-trained synthetic ensemble)
5. **Validate LOD** empirically at 10 CFU/mL across multiple organisms and media formulations

See `docs/wet_lab_validation_plan.md` for full wet-lab protocol.

---

## References & Sources

1. **Beer-Lambert Law:** Derived from first principles of radiative transfer
2. **Extinction Coefficients:** Typical values from literature; organism-specific profiles TBD via wet-lab measurement
3. **Rayleigh-Mie Approximation:** Simplified Gaussian model; full Lorenz-Mie solution available via `PyMieScatt` library (not currently integrated)
4. **Contamination Microbiology:** Standard biocontamination levels from USP <71> and EU GMP guidelines

---

**Document Version:** 1.0  
**Last Updated:** May 13, 2026  
**Maintainer:** Biopharmaceutical Contamination Detection Team
