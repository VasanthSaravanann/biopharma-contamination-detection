# Wet-Lab Sampling SOPs and Data Templates

## Overview

This document provides standard operating procedures (SOPs) for wet-lab validation of the ensemble contamination detector. The goal is to collect real fermentation data and validate computational results against empirical measurements.

---

## SOP 1: UV-Vis Spectroscopy Data Collection

### Objective
Capture UV-Vis absorbance spectra (200–800 nm) from fermentation samples containing known contamination levels.

### Equipment Required
- UV-Vis spectrophotometer (1 cm path length cuvettes)
- Calibrated cuvettes (matched pair for baseline)
- Temperature-controlled cell holder (set to 25 °C ± 1 °C)
- Sterile serological pipettes (1, 5, 10 mL)
- Sterile 1.5 mL microcentrifuge tubes

### Materials
- Phosphate-buffered saline (PBS) pH 7.0 (for blanks and sample dilution if needed)
- Growth media for fermentation
- Contaminant strains: E. coli, B. subtilis, P. aeruginosa, C. albicans, A. niger, Mycoplasma

### Procedure

#### Pre-Experiment Setup
1. **Baseline Calibration**
   - Fill matched cuvette pair with PBS pH 7.0
   - Insert one into reference compartment, one into sample compartment
   - Record baseline absorbance at 260, 280, 340, 405, 450, 600 nm
   - Note: Full spectrum (200–800 nm) preferred if instrument allows

2. **Temperature Equilibration**
   - Allow spectrophotometer to warm up for 15 minutes
   - Set temperature to 25 °C ± 1 °C
   - Allow 5 minutes for cell holder to equilibrate

#### Sample Preparation
3. **Inoculum Preparation**
   - Prepare serial dilutions of each contaminant strain:
     - Target levels: 0, 1, 10, 50, 100, 1K, 10K CFU/mL
   - Verify inoculum concentration by plate count (gold standard) or optical density (OD₆₀₀)
   - Record actual CFU/mL values (not approximate)

4. **Fermentation Sample Fortification**
   - Collect sterile fermentation sample (50 mL) from reactor at steady-state
   - Divide into 7 aliquots (one per inoculum level)
   - Aseptically spike each aliquot with contaminant inoculum
   - Incubate at 37 °C for 2 hours (allow contamination to release intracellular material)
   - Allow to cool to 25 °C (5 minutes)

#### Spectroscopy Measurement
5. **Absorbance Measurement**
   - Transfer sample to 1 cm cuvette (pre-warmed to 25 °C)
   - Insert into spectrophotometer
   - Record full spectrum (200–800 nm) at 1 nm resolution OR record at fixed wavelengths: 260, 280, 340, 405, 450, 600 nm
   - Wait 30 seconds after insertion for temperature equilibration before recording
   - Take 3 replicate measurements per sample (same cuvette)
   - Calculate mean and standard deviation

6. **Data Recording**
   - Record spectrum as CSV: `sample_id, wavelength (nm), absorbance`
   - Include metadata: date, time, organism, CFU/mL (verified), media composition, temperature, operator

#### Quality Control
7. **QC Checks**
   - **Blank Check:** Verify PBS baseline A < 0.05 at 260 nm
   - **Duplicate Precision:** Relative standard deviation between replicates < 2%
   - **Wavelength Accuracy:** Verify spectrophotometer wavelength calibration (hydrogen lamp or holmium filter)
   - **Path Length Verification:** Run cuvette holder calibration

---

## SOP 2: Metabolite Quantification (HPLC or Enzymatic Assays)

### Objective
Measure key metabolites (glucose, lactate, acetate) to augment UV-Vis data.

### Equipment
- HPLC system with UV/Refractive Index (RI) detector OR
- Enzymatic assay kits (e.g., YSI Bioprofile or equivalent)

### Procedure (Enzymatic Assay as Simpler Alternative)
1. Prepare sample aliquots (0.5 mL) and freeze at −20 °C until analysis
2. Thaw and centrifuge at 10,000 g for 5 minutes
3. Run enzymatic assay per kit instructions
4. Record: glucose (mg/dL), lactate (mg/dL), acetate (mM)
5. Calculate delta per time-point

---

## SOP 3: CFU Enumeration (Plate Counting)

### Objective
Verify actual CFU/mL in each sample (ground truth for inoculum level validation).

### Procedure
1. Prepare 10-fold serial dilutions (10⁻¹ to 10⁻⁸) in sterile PBS
2. Plate 100 µL on selective/differential media
3. Incubate at 37 °C for 24–48 hours
4. Count colonies (30–300 per plate, if possible)
5. Calculate CFU/mL = (count × dilution factor) / volume plated
6. Record for each organism

---

## Data Template: UV-Vis Spectrum CSV

```csv
experiment_id,date,organism,inoculum_cfu_per_ml,replicate,wavelength_nm,absorbance,media_composition,temperature_C,ph
WLV_001,2026-05-14,E_coli,10,1,200,0.234,LB_broth,25.0,7.0
WLV_001,2026-05-14,E_coli,10,1,201,0.231,LB_broth,25.0,7.0
...
WLV_001,2026-05-14,E_coli,10,1,800,0.012,LB_broth,25.0,7.0
```

**Columns:**
- `experiment_id`: Unique identifier (e.g., WLV_001)
- `date`: ISO 8601 format (YYYY-MM-DD)
- `organism`: Contaminant type (E_coli, B_subtilis, P_aeruginosa, C_albicans, A_niger, Mycoplasma)
- `inoculum_cfu_per_ml`: Actual CFU/mL (from plate count)
- `replicate`: 1, 2, 3 (biological replicates)
- `wavelength_nm`: 200–800 (1 nm resolution)
- `absorbance`: Measured A(λ)
- `media_composition`: Media name/code
- `temperature_C`: Temperature at measurement
- `ph`: pH at measurement

---

## Data Template: Metabolite Profile CSV

```csv
experiment_id,date,organism,inoculum_cfu_per_ml,glucose_mg_dl,lactate_mg_dl,acetate_mM
WLV_001,2026-05-14,E_coli,10,180,15,0.5
WLV_001,2026-05-14,E_coli,50,175,25,1.2
...
```

---

## Data Template: CFU Enumeration CSV

```csv
experiment_id,date,organism,target_dilution,plate_count,cfu_per_ml
WLV_001,2026-05-14,E_coli,1e-6,125,1.25e8
WLV_001,2026-05-14,E_coli,1e-7,15,1.5e7
```

---

## Safety and Regulatory Compliance

### Biosafety
- **Level of Work:** BSL-1 or BSL-2 depending on organism (C. albicans may require BSL-2)
- **Containment:** Use biosafety cabinet for sample preparation
- **PPE:** Lab coat, gloves, safety glasses
- **Waste:** Autoclave all contaminated media and samples

### Data Integrity
- **Record Retention:** Retain raw spectra, calculations, and plate counts for ≥3 years
- **Chain of Custody:** Document sample handling from collection to analysis
- **Traceability:** Link all results to batch records and SOP versions

### Quality Assurance
- **Traceability:** All calibrations tied to reference standards
- **Validation:** Two independent operators validate SOP procedure
- **Documentation:** Maintain deviation log for any protocol deviations

---

## Expected Outcomes & Success Criteria

### ROC-AUC Target Validation
- Train ensemble on computational (synthetic) data
- Test ensemble on real wet-lab spectra
- **Target:** ROC-AUC ≥ 0.85 (relaxed from synthetic 0.98 to account for model mismatch)

### LOD Verification
- Measure detection rate at each inoculum level
- **Target:** ≥50% detection at 10 CFU/mL

### Data Quality Checks
- Replicate CV < 5%
- Blank baseline A₂₆₀ < 0.05
- CFU enumeration within 1 log of inoculum spike

---

## Multi-Site Data Sharing

### Pre-Registration
Before conducting experiments, sites must pre-register on the project hub:
1. Submit proposal (organism, media, number of samples)
2. Get approval and unique `site_id`
3. Follow SOP exactly; report any deviations
4. Submit data in standardized CSV format

### Data Format for Multi-Site Submission
Include additional column in spectra CSV:
```csv
site_id,experiment_id,date,...
SITE_A,WLV_001,2026-05-14,...
SITE_B,WLV_002,2026-05-15,...
```

### Anonymization
- Remove operator names (use role ID)
- Aggregate replicates to mean ± SD if individual replicates sensitive

---

## References

1. **USP <71> Sterility Tests** — Regulatory standard for contamination detection
2. **EPA Method 9213A** — Spectrophotometry guidelines
3. **ISO 11135:2014** — Sterilization validation
4. **Beer-Lambert Law** — Physical basis of absorbance measurement

---

**SOP Version:** 1.0  
**Effective Date:** May 13, 2026  
**Next Review:** May 13, 2027
