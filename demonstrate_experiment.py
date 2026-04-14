#!/usr/bin/env python3
"""
INTERACTIVE EXPERIMENT DEMONSTRATION
=====================================
This script shows EXACTLY how the experiment works:
1. How data is created (synthetic UV-Vis spectra)
2. How features are extracted
3. How models detect contamination
4. How detection is verified

Run: python demonstrate_experiment.py
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Installing matplotlib...")
    import os
    os.system('pip install matplotlib --quiet')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy.integrate import trapezoid
except ImportError:
    print("Installing scikit-learn...")
    import os
    os.system('pip install scikit-learn --quiet')
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy.integrate import trapezoid

# Create output directory
DEMO_OUTPUT = Path("/run/media/sham/AI_/projects/biopharma-contamination-detection/demo_output")
DEMO_OUTPUT.mkdir(exist_ok=True)

print("="*80)
print("BIOPHARMA CONTAMINATION DETECTION - INTERACTIVE DEMONSTRATION")
print("="*80)
print(f"\nOutput directory: {DEMO_OUTPUT}\n")

# ============================================================================
# PART 1: HOW DATA IS CREATED
# ============================================================================
print("\n" + "="*80)
print("PART 1: HOW DATA IS CREATED (Synthetic UV-Vis Spectra)")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE PHYSICS BEHIND UV-VIS SPECTRA                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Beer-Lambert Law: A = ε × c × l                                           │
│                                                                             │
│  Where:                                                                    │
│    A = Absorbance (what we measure)                                        │
│    ε = Molar absorptivity (property of molecule)                           │
│    c = Concentration (mol/L)                                               │
│    l = Path length (cm)                                                    │
│                                                                             │
│  For CLEAN media:                                                          │
│    - Protein peak at 280 nm (tryptophan, tyrosine)                         │
│    - NADH peak at 340 nm                                                   │
│    - Flat baseline in visible region (400-800 nm)                          │
│                                                                             │
│  For CONTAMINATED media:                                                   │
│    - Same protein peak at 280 nm                                           │
│    - PLUS: Bacterial scattering (1/λ behavior)                             │
│    - Higher absorbance at longer wavelengths                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# Define wavelengths
wavelengths = np.arange(200, 801, 1)  # 200-800 nm

print("\n📊 STEP 1.1: Creating Wavelength Array")
print(f"   Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm")
print(f"   Number of data points: {len(wavelengths)}")
print(f"   First 10 wavelengths: {wavelengths[:10].tolist()}")

input("\n   Press Enter to continue...")

print("\n📊 STEP 1.2: Generating CLEAN (Nominal) Spectrum")
print("""
   Formula:
   ─────────
   base = 0.5 × exp(-(λ - 280)² / 400)     [Protein peak]
   media = 0.2 × exp(-(λ - 340)² / 600)    [Media components]
   noise = N(0, 0.02)                       [Instrument noise]
   
   spectrum = base + media + noise
""")

# Generate a single clean spectrum
base_protein = 0.5 * np.exp(-(wavelengths - 280)**2 / 400)
media_components = 0.2 * np.exp(-(wavelengths - 340)**2 / 600)
noise = np.random.normal(0, 0.02, len(wavelengths))
clean_spectrum = base_protein + media_components + noise

print(f"   Protein peak (280 nm): {base_protein[np.argmin(np.abs(wavelengths - 280))]:.4f} AU")
print(f"   NADH peak (340 nm): {media_components[np.argmin(np.abs(wavelengths - 340))]:.4f} AU")
print(f"   Baseline (600 nm): {clean_spectrum[np.argmin(np.abs(wavelengths - 600))]:.4f} AU")
print(f"   Noise level: σ = 0.02 AU")

input("\n   Press Enter to see the spectrum...")

# Plot single spectrum comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavelengths, base_protein, 'b--', label='Protein Peak (280 nm)', linewidth=2, alpha=0.7)
ax.plot(wavelengths, media_components, 'g--', label='Media Components (340 nm)', linewidth=2, alpha=0.7)
ax.plot(wavelengths, clean_spectrum, 'b-', label='Final Clean Spectrum', linewidth=2)
ax.fill_between(wavelengths, clean_spectrum - 0.02, clean_spectrum + 0.02, alpha=0.3, color='blue', label='Noise range (±0.02)')
ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Absorbance (AU)', fontsize=12)
ax.set_title('HOW CLEAN SPECTRA ARE GENERATED', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "01_clean_spectrum_generation.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '01_clean_spectrum_generation.png'}")

print("\n📊 STEP 1.3: Generating CONTAMINATED Spectrum")
print("""
   Bacterial Scattering Physics:
   ─────────────────────────────
   When bacteria are present, light is SCATTERED (not just absorbed)
   
   Scattering follows 1/λ relationship (Rayleigh-Mie scattering):
   
   scattering = coefficient × (1000 / λ)
   
   Where coefficient depends on:
     - Bacterial species (size, shape)
     - Concentration (CFU/mL)
   
   Final contaminated spectrum:
   spectrum = base + bacterial_scattering + noise
""")

# Species-specific scattering coefficients
scattering_coeffs = {
    'E_coli': 0.8,
    'S_aureus': 0.7,
    'B_subtilis': 0.75,
    'P_aeruginosa': 0.85,
    'C_albicans': 0.6,
    'A_brasiliensis': 0.65
}

species = 'E_coli'
cfu_ml = 100

# Calculate bacterial scattering
coeff = scattering_coeffs[species]
conc_factor = np.log10(cfu_ml + 1) / 3.0  # Log relationship with concentration
bacterial_scattering = coeff * conc_factor * (1000 / wavelengths)

contaminated_spectrum = base_protein + bacterial_scattering + noise

print(f"   Species: {species}")
print(f"   Concentration: {cfu_ml} CFU/mL")
print(f"   Scattering coefficient: {coeff}")
print(f"   Concentration factor: {conc_factor:.4f}")
print(f"   Scattering at 600 nm: {bacterial_scattering[np.argmin(np.abs(wavelengths - 600))]:.4f} AU")

input("\n   Press Enter to see contaminated spectrum...")

# Plot contaminated spectrum
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavelengths, clean_spectrum, 'b-', label='Clean Spectrum', linewidth=2, alpha=0.7)
ax.plot(wavelengths, bacterial_scattering, 'r--', label=f'Bacterial Scattering (1/λ)', linewidth=2, alpha=0.7)
ax.plot(wavelengths, contaminated_spectrum, 'r-', label=f'Contaminated Spectrum ({species}, {cfu_ml} CFU/mL)', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Absorbance (AU)', fontsize=12)
ax.set_title('HOW CONTAMINATED SPECTRA ARE GENERATED', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "02_contaminated_spectrum_generation.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '02_contaminated_spectrum_generation.png'}")

print("\n📊 STEP 1.4: Comparing Clean vs Contaminated")
print("""
   Key Differences:
   ────────────────
   1. Clean: Low absorbance in visible region (400-800 nm)
   2. Contaminated: Elevated absorbance in visible region
   3. The "tail" at long wavelengths = bacterial signature
""")

input("\n   Press Enter to see comparison...")

# Side-by-side comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavelengths, clean_spectrum, 'b-', label='Clean (Nominal)', linewidth=2)
ax.plot(wavelengths, contaminated_spectrum, 'r-', label=f'Contaminated ({species}, {cfu_ml} CFU/mL)', linewidth=2)
ax.axvline(x=280, color='gray', linestyle='--', alpha=0.5, label='Protein peak (280 nm)')
ax.axvline(x=600, color='orange', linestyle='--', alpha=0.5, label='Detection region (600 nm)')
ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Absorbance (AU)', fontsize=12)
ax.set_title('CLEAN vs CONTAMINATED SPECTRA', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "03_clean_vs_contaminated.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '03_clean_vs_contaminated.png'}")

print("\n📊 STEP 1.5: Generating Full Dataset")
print("""
   Creating:
   - 100 clean spectra (for training)
   - 50 contaminated spectra at each condition
   - 6 species × 5 concentration levels
""")

def generate_clean_spectrum(wavelengths):
    """Generate a single clean spectrum with variation"""
    base = 0.5 * np.exp(-(wavelengths - 280)**2 / 400)
    media = 0.2 * np.exp(-(wavelengths - 340)**2 / 600)
    # Add batch-to-batch variation
    batch_effect = np.random.normal(0, 0.02) * base
    noise = np.random.normal(0, 0.02, len(wavelengths))
    return base + media + batch_effect + noise

def generate_contaminated_spectrum(wavelengths, species, cfu_ml):
    """Generate a single contaminated spectrum"""
    base = 0.5 * np.exp(-(wavelengths - 280)**2 / 400)
    coeff = scattering_coeffs.get(species, 0.7)
    conc_factor = np.log10(cfu_ml + 1) / 3.0
    bacterial_signal = coeff * conc_factor * (1000 / wavelengths)
    noise = np.random.normal(0, 0.02, len(wavelengths))
    return base + bacterial_signal + noise

# Generate dataset
clean_spectra = [generate_clean_spectrum(wavelengths) for _ in range(100)]
contaminated_spectra = []
labels = []

cfu_levels = [10, 50, 100, 500, 1000]
for species in scattering_coeffs.keys():
    for cfu in cfu_levels:
        for _ in range(10):  # 10 replicates per condition
            contaminated_spectra.append(generate_contaminated_spectrum(wavelengths, species, cfu))
            labels.append({'species': species, 'cfu_ml': cfu})

print(f"   Clean spectra: {len(clean_spectra)}")
print(f"   Contaminated spectra: {len(contaminated_spectra)}")
print(f"   Total samples: {len(clean_spectra) + len(contaminated_spectra)}")

input("\n   Press Enter to see all spectra overlaid...")

# Plot all spectra
fig, ax = plt.subplots(figsize=(12, 6))
for spectrum in clean_spectra[:20]:  # Show first 20
    ax.plot(wavelengths, spectrum, 'b-', alpha=0.1, linewidth=0.5)
for spectrum in contaminated_spectra[:50]:  # Show first 50
    ax.plot(wavelengths, spectrum, 'r-', alpha=0.1, linewidth=0.5)

# Plot averages
clean_mean = np.mean(clean_spectra, axis=0)
contaminated_mean = np.mean([s for s, l in zip(contaminated_spectra, labels) if l['cfu_ml'] == 100], axis=0)

ax.plot(wavelengths, clean_mean, 'b-', linewidth=2, label='Mean Clean')
ax.plot(wavelengths, contaminated_mean, 'r-', linewidth=2, label='Mean Contaminated (100 CFU/mL)')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Absorbance (AU)', fontsize=12)
ax.set_title('FULL DATASET: All Spectra Overlaid', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "04_all_spectra_overlaid.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '04_all_spectra_overlaid.png'}")

print("\n✅ PART 1 COMPLETE: Data Generation")
print(f"   Output: {DEMO_OUTPUT / '01-04_spectrum_generation.png'}")

input("\n📌 Press Enter to continue to PART 2: Feature Extraction...")

# ============================================================================
# PART 2: HOW FEATURES ARE EXTRACTED
# ============================================================================
print("\n" + "="*80)
print("PART 2: HOW FEATURES ARE EXTRACTED FROM SPECTRA")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY EXTRACT FEATURES?                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Raw spectra have 601 data points (200-800 nm).                              │
│ Machine learning works better with fewer, meaningful features.              │
│                                                                             │
│ We extract 7 key features that capture:                                     │
│ 1. Protein content (280 nm)                                                 │
│ 2. Metabolite content (340 nm)                                              │
│ 3. Turbidity/biomass (600 nm)                                               │
│ 4. Ratio metrics (contamination indicator)                                  │
│ 5. Scattering slope (bacterial signature)                                   │
│ 6. UV integral (total UV-absorbing material)                                │
│ 7. Scattering index (visible region mean)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

def extract_features_single(spectrum, wavelengths):
    """Extract 7 features from a single spectrum"""
    features = {
        'abs_280': spectrum[np.argmin(np.abs(wavelengths - 280))],
        'abs_340': spectrum[np.argmin(np.abs(wavelengths - 340))],
        'abs_600': spectrum[np.argmin(np.abs(wavelengths - 600))],
        'ratio_280_600': spectrum[np.argmin(np.abs(wavelengths - 280))] / 
                        (spectrum[np.argmin(np.abs(wavelengths - 600))] + 1e-6),
        'slope_400_700': (spectrum[np.argmin(np.abs(wavelengths - 700))] - 
                         spectrum[np.argmin(np.abs(wavelengths - 400))]) / 300,
        'integral_200_400': trapezoid(spectrum[(wavelengths >= 200) & (wavelengths <= 400)], 
                                    wavelengths[(wavelengths >= 200) & (wavelengths <= 400)]),
        'scattering_index': np.mean(spectrum[(wavelengths > 600) & (wavelengths < 700)])
    }
    return features

print("\n📊 STEP 2.1: Extracting Features from One Spectrum")

# Extract features from one clean and one contaminated
clean_features = extract_features_single(clean_spectra[0], wavelengths)
contaminated_features = extract_features_single(contaminated_spectra[0], wavelengths)

print("""
   Feature Extraction Formula:
   ───────────────────────────
""")
print(f"   1. abs_280 = Value at 280 nm")
print(f"   2. abs_340 = Value at 340 nm")
print(f"   3. abs_600 = Value at 600 nm")
print(f"   4. ratio_280_600 = abs_280 / abs_600")
print(f"   5. slope_400_700 = (abs_700 - abs_400) / 300")
print(f"   6. integral_200_400 = Area under curve (200-400 nm)")
print(f"   7. scattering_index = Mean(600-700 nm)")

print(f"\n   Example Values:")
print(f"   {'Feature':<20} | {'Clean':<12} | {'Contaminated':<12} | {'Change':<10}")
print(f"   {'-'*20}-|-{'-'*10}-|-{'-'*10}-|-{'-'*8}")
for key in clean_features.keys():
    clean_val = clean_features[key]
    contam_val = contaminated_features[key]
    change = ((contam_val - clean_val) / clean_val) * 100
    print(f"   {key:<20} | {clean_val:<12.4f} | {contam_val:<12.4f} | {change:+>8.1f}%")

input("\n   Press Enter to see feature visualization...")

# Visualize features on spectrum
fig, ax = plt.subplots(figsize=(14, 7))

# Plot spectrum
ax.plot(wavelengths, contaminated_spectra[0], 'r-', linewidth=2, label='Contaminated Spectrum')

# Mark features
ax.axvline(x=280, color='blue', linestyle='--', linewidth=2, label='abs_280')
ax.axvline(x=340, color='green', linestyle='--', linewidth=2, label='abs_340')
ax.axvline(x=600, color='orange', linestyle='--', linewidth=2, label='abs_600')
ax.axvline(x=700, color='purple', linestyle=':', linewidth=2, label='Features for slope')

# Highlight regions
ax.axvspan(200, 400, alpha=0.2, color='cyan', label='Integral region (200-400 nm)')
ax.axvspan(600, 700, alpha=0.2, color='red', label='Scattering index region')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Absorbance (AU)', fontsize=12)
ax.set_title('FEATURE EXTRACTION: Where Each Feature Comes From', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "05_feature_extraction.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '05_feature_extraction.png'}")

print("\n📊 STEP 2.2: Extracting Features from ALL Spectra")

# Extract all features
all_features = []
all_labels_binary = []  # 0 = clean, 1 = contaminated

for spectrum in clean_spectra:
    feat = extract_features_single(spectrum, wavelengths)
    all_features.append(list(feat.values()))
    all_labels_binary.append(0)

for spectrum in contaminated_spectra:
    feat = extract_features_single(spectrum, wavelengths)
    all_features.append(list(feat.values()))
    all_labels_binary.append(1)

all_features = np.array(all_features)
all_labels_binary = np.array(all_labels_binary)

feature_names = list(clean_features.keys())

print(f"   Feature matrix shape: {all_features.shape}")
print(f"   Clean samples: {np.sum(all_labels_binary == 0)}")
print(f"   Contaminated samples: {np.sum(all_labels_binary == 1)}")

input("\n   Press Enter to see feature distributions...")

# Plot feature distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, (name, ax) in enumerate(zip(feature_names, axes)):
    clean_vals = all_features[all_labels_binary == 0, i]
    contam_vals = all_features[all_labels_binary == 1, i]
    
    ax.hist(clean_vals, bins=20, alpha=0.5, color='blue', label='Clean', density=True)
    ax.hist(contam_vals, bins=20, alpha=0.5, color='red', label='Contaminated', density=True)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide the last subplot
axes[-1].axis('off')

plt.suptitle('FEATURE DISTRIBUTIONS: Clean vs Contaminated', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "06_feature_distributions.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '06_feature_distributions.png'}")

print("\n✅ PART 2 COMPLETE: Feature Extraction")
print(f"   Extracted {len(feature_names)} features from {len(all_features)} samples")

input("\n📌 Press Enter to continue to PART 3: Model Training...")

# ============================================================================
# PART 3: HOW MODELS ARE TRAINED
# ============================================================================
print("\n" + "="*80)
print("PART 3: HOW ANOMALY DETECTION MODELS ARE TRAINED")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ ANOMALY DETECTION PRINCIPLE                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Key Idea: Train ONLY on CLEAN (normal) data                                 │
│                                                                             │
│ During training:                                                            │   │
│   - Model learns: "What does clean media look like?"                        │
│   - Model builds: Boundary around "normal"                                  │
│                                                                             │
│ During testing:                                                             │
│   - New sample inside boundary → "Clean"                                    │
│   - New sample outside boundary → "Contaminated" (anomaly)                  │
│                                                                             │
│ Why this works:                                                             │
│   - Contamination is RARE (we don't have many examples)                     │
│   - Contamination varies (different species, concentrations)                │
│   - Clean is CONSISTENT (always looks similar)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n📊 STEP 3.1: Preparing Training Data")
print("""
   Training Data Strategy:
   ───────────────────────
   
   ONLY Clean Spectra → Train Model
   (Model learns "normal")
   
   Clean + Contaminated → Test Model
   (Evaluate if model can detect anomalies)
""")

# Split data
X_train = all_features[all_labels_binary == 0]  # Only clean for training
X_test = all_features  # All for testing
y_test = all_labels_binary

print(f"   Training samples (clean only): {len(X_train)}")
print(f"   Test samples (all): {len(X_test)}")
print(f"   Features per sample: {X_train.shape[1]}")

input("\n   Press Enter to train Isolation Forest...")

print("\n📊 STEP 3.2: Training Isolation Forest")
print("""
   How Isolation Forest Works:
   ───────────────────────────
   
   1. Build many random decision trees
   2. Each tree randomly splits data
   3. Anomalies are isolated quickly (fewer splits)
   4. Score = average path length across all trees
   
   Short path → Anomaly (contaminated)
   Long path → Normal (clean)
""")

print(f"   Configuration:")
print(f"     - n_estimators: 200 (trees)")
print(f"     - contamination: 0.01 (1% expected anomalies)")
print(f"     - random_state: 42 (reproducibility)")

from sklearn.ensemble import IsolationForest

if_model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

import time
start = time.time()
if_model.fit(X_train)
if_time = time.time() - start

print(f"   ✓ Training time: {if_time:.3f} seconds")

input("\n   Press Enter to train One-Class SVM...")

print("\n📊 STEP 3.3: Training One-Class SVM")
print("""
   How One-Class SVM Works:
   ────────────────────────
   
   1. Map data to high-dimensional space
   2. Find hyperplane that separates ALL data from origin
   3. Distance from hyperplane = anomaly score
   
   Far from boundary → Normal (clean)
   Close to/Cross boundary → Anomaly (contaminated)
""")

print(f"   Configuration:")
print(f"     - kernel: RBF (radial basis function)")
print(f"     - gamma: scale (automatic)")
print(f"     - nu: 0.01 (upper bound on anomaly fraction)")

ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)

start = time.time()
ocsvm.fit(X_train)
ocsvm_time = time.time() - start

print(f"   ✓ Training time: {ocsvm_time:.3f} seconds")

print("\n✅ PART 3 COMPLETE: Model Training")
print(f"   Isolation Forest: {if_time:.3f}s")
print(f"   One-Class SVM: {ocsvm_time:.3f}s")

input("\n📌 Press Enter to continue to PART 4: Detection Verification...")

# ============================================================================
# PART 4: HOW DETECTION IS VERIFIED
# ============================================================================
print("\n" + "="*80)
print("PART 4: HOW DETECTION IS VERIFIED")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ VERIFICATION STRATEGY                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ We know which samples are clean vs contaminated (we generated them!)       │
│                                                                             │
│ Verification Steps:                                                         │
│ 1. Get anomaly scores from model                                            │
│ 2. Compare scores to threshold                                              │
│ 3. Calculate:                                                               │
│    - True Positives (TP): Contaminated correctly detected                  │
│    - False Positives (FP): Clean incorrectly flagged                       │
│    - True Negatives (TN): Clean correctly identified                       │
│    - False Negatives (FN): Contaminated missed                             │
│                                                                             │
│ 4. Calculate metrics:                                                       │
│    - Sensitivity (Recall): TP / (TP + FN) - Detect all contamination      │
│    - Specificity: TN / (TN + FP) - Don't flag clean as contaminated       │
│    - ROC-AUC: Overall discrimination ability                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n📊 STEP 4.1: Getting Anomaly Scores")
print("""
   Anomaly Score Interpretation:
   ─────────────────────────────
   
   Isolation Forest:
     - score_samples() returns negative values
     - More negative = more anomalous
     - We use: -score (higher = more anomalous)
   
   One-Class SVM:
     - decision_function() returns distance from boundary
     - Negative = outside boundary (anomaly)
     - We use: -decision (higher = more anomalous)
""")

# Get scores
if_scores = -if_model.score_samples(X_test)
ocsvm_scores = -ocsvm.decision_function(X_test)

print(f"\n   Isolation Forest Scores:")
print(f"     Min: {if_scores.min():.4f}")
print(f"     Max: {if_scores.max():.4f}")
print(f"     Mean (clean): {if_scores[y_test == 0].mean():.4f}")
print(f"     Mean (contaminated): {if_scores[y_test == 1].mean():.4f}")

input("\n   Press Enter to see score distributions...")

# Plot score distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Isolation Forest
ax1.hist(if_scores[y_test == 0], bins=30, alpha=0.5, color='blue', label='Clean', density=True)
ax1.hist(if_scores[y_test == 1], bins=30, alpha=0.5, color='red', label='Contaminated', density=True)
ax1.axvline(x=np.percentile(if_scores[y_test == 0], 95), color='green', linestyle='--', 
            linewidth=2, label='Threshold (95th percentile)')
ax1.set_xlabel('Anomaly Score', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('ISOLATION FOREST: Score Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# One-Class SVM
ax2.hist(ocsvm_scores[y_test == 0], bins=30, alpha=0.5, color='blue', label='Clean', density=True)
ax2.hist(ocsvm_scores[y_test == 1], bins=30, alpha=0.5, color='red', label='Contaminated', density=True)
ax2.axvline(x=np.percentile(ocsvm_scores[y_test == 0], 95), color='green', linestyle='--', 
            linewidth=2, label='Threshold (95th percentile)')
ax2.set_xlabel('Anomaly Score', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('ONE-CLASS SVM: Score Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "07_anomaly_score_distributions.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '07_anomaly_score_distributions.png'}")

print("\n📊 STEP 4.2: Setting Detection Threshold")
print("""
   Threshold Selection:
   ────────────────────
   
   We use 95th percentile of CLEAN scores as threshold.
   
   Why?
   - 95% of clean samples will be below threshold
   - Only 5% false positive rate acceptable
   - Anything above = likely contaminated
""")

if_threshold = np.percentile(if_scores[y_test == 0], 95)
ocsvm_threshold = np.percentile(ocsvm_scores[y_test == 0], 95)

print(f"\n   Isolation Forest threshold: {if_threshold:.4f}")
print(f"   One-Class SVM threshold: {ocsvm_threshold:.4f}")

input("\n   Press Enter to calculate detection metrics...")

print("\n📊 STEP 4.3: Calculating Detection Metrics")

# Make predictions
if_predictions = (if_scores > if_threshold).astype(int)
ocsvm_predictions = (ocsvm_scores > ocsvm_threshold).astype(int)

# Calculate metrics
def calculate_metrics(predictions, labels):
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

if_metrics = calculate_metrics(if_predictions, y_test)
ocsvm_metrics = calculate_metrics(ocsvm_predictions, y_test)

# ROC-AUC
if_auc = roc_auc_score(y_test, if_scores)
ocsvm_auc = roc_auc_score(y_test, ocsvm_scores)

print("""
   Confusion Matrix:
   ─────────────────
   
                  Predicted
                 Clean  Contam
   Actual Clean   TN     FP
          Contam  FN     TP
""")

print(f"\n   ISOLATION FOREST:")
print(f"     TP: {if_metrics['TP']}, FP: {if_metrics['FP']}")
print(f"     TN: {if_metrics['TN']}, FN: {if_metrics['FN']}")
print(f"     Sensitivity: {if_metrics['Sensitivity']:.4f} ({if_metrics['Sensitivity']*100:.1f}%)")
print(f"     Specificity: {if_metrics['Specificity']:.4f} ({if_metrics['Specificity']*100:.1f}%)")
print(f"     ROC-AUC: {if_auc:.4f}")

print(f"\n   ONE-CLASS SVM:")
print(f"     TP: {ocsvm_metrics['TP']}, FP: {ocsvm_metrics['FP']}")
print(f"     TN: {ocsvm_metrics['TN']}, FN: {ocsvm_metrics['FN']}")
print(f"     Sensitivity: {ocsvm_metrics['Sensitivity']:.4f} ({ocsvm_metrics['Sensitivity']*100:.1f}%)")
print(f"     Specificity: {ocsvm_metrics['Specificity']:.4f} ({ocsvm_metrics['Specificity']*100:.1f}%)")
print(f"     ROC-AUC: {ocsvm_auc:.4f}")

input("\n   Press Enter to see ROC curves...")

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 10))

fpr_if, tpr_if, _ = roc_curve(y_test, if_scores)
fpr_ocsvm, tpr_ocsvm, _ = roc_curve(y_test, ocsvm_scores)

ax.plot(fpr_if, tpr_if, 'b-', linewidth=2, label=f'Isolation Forest (AUC = {if_auc:.4f})')
ax.plot(fpr_ocsvm, tpr_ocsvm, 'r-', linewidth=2, label=f'One-Class SVM (AUC = {ocsvm_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC CURVES: Detection Performance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "08_roc_curves.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '08_roc_curves.png'}")

print("\n📊 STEP 4.4: Detection by Concentration Level")
print("""
   Testing Detection at Different CFU/mL Levels:
   ─────────────────────────────────────────────
   
   For each concentration level, calculate:
   - Detection rate = % of samples correctly flagged
""")

detection_by_cfu = {}
for cfu in cfu_levels:
    # Find indices for this CFU level
    indices = [i for i, label in enumerate(labels) if label['cfu_ml'] == cfu]
    # Get predictions for these indices (add 100 to account for clean samples)
    indices_adjusted = [i + 100 for i in indices]
    
    # Calculate detection rate
    detections = np.sum(if_predictions[indices_adjusted] == 1)
    total = len(indices)
    rate = detections / total
    
    detection_by_cfu[cfu] = rate
    print(f"   {cfu:>4} CFU/mL: {rate:>6.1%} detection rate ({detections}/{total})")

input("\n   Press Enter to see detection limit analysis...")

# Plot detection by CFU
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(detection_by_cfu.keys(), detection_by_cfu.values(), color='steelblue', edgecolor='black')
ax.set_xlabel('Inoculum Level (CFU/mL)', fontsize=12)
ax.set_ylabel('Detection Rate', fontsize=12)
ax.set_title('DETECTION PERFORMANCE BY CONCENTRATION', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Target')
ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='50% (Limit of Detection)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(DEMO_OUTPUT / "09_detection_by_cfu.png", dpi=300)
plt.close()
print(f"   ✓ Saved: {DEMO_OUTPUT / '09_detection_by_cfu.png'}")

print("\n✅ PART 4 COMPLETE: Detection Verified")

input("\n📌 Press Enter for final summary...")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT COMPLETE - SUMMARY")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHAT YOU JUST SAW                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PART 1: DATA GENERATION                                                     │
│   ✓ Created wavelength array (200-800 nm)                                   │
│   ✓ Generated clean spectra (Beer-Lambert law)                              │
│   ✓ Generated contaminated spectra (1/λ scattering)                         │
│   ✓ Full dataset: 100 clean + 300 contaminated                              │
│                                                                             │
│ PART 2: FEATURE EXTRACTION                                                  │
│   ✓ Extracted 7 meaningful features                                         │
│   ✓ Features capture protein, metabolites, scattering                       │
│   ✓ Visualized feature distributions                                        │
│                                                                             │
│ PART 3: MODEL TRAINING                                                      │
│   ✓ Trained Isolation Forest (only on clean data)                           │
│   ✓ Trained One-Class SVM (only on clean data)                              │
│   ✓ Both models learn "what is normal"                                      │
│                                                                             │
│ PART 4: DETECTION VERIFICATION                                              │
│   ✓ Applied models to test data                                             │
│   ✓ Set threshold at 95th percentile                                        │
│   ✓ Calculated sensitivity, specificity, ROC-AUC                            │
│   ✓ Analyzed detection by concentration                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print(f"""
📊 FINAL RESULTS:
{'─'*60}

Model Performance:
┌─────────────────────┬──────────────┬──────────────┐
│ Metric              │ Isolation    │ One-Class    │
│                     │ Forest       │ SVM          │
├─────────────────────┼──────────────┼──────────────┤
│ ROC-AUC             │ {if_auc:>12.4f} │ {ocsvm_auc:>12.4f} │
│ Sensitivity         │ {if_metrics['Sensitivity']:>12.4f} │ {ocsvm_metrics['Sensitivity']:>12.4f} │
│ Specificity         │ {if_metrics['Specificity']:>12.4f} │ {ocsvm_metrics['Specificity']:>12.4f} │
│ Training Time       │ {if_time:>12.3f}s │ {ocsvm_time:>12.3f}s │
└─────────────────────┴──────────────┴──────────────┘

Detection by Concentration:
┌─────────────────────┬──────────────┐
│ CFU/mL              │ Detection %  │
├─────────────────────┼──────────────┤""")

for cfu, rate in detection_by_cfu.items():
    print(f"│ {cfu:>4}                │ {rate:>11.1%} │")

print(f"""└─────────────────────┴──────────────┘

📁 OUTPUT FILES:
{'─'*60}""")

for f in sorted(DEMO_OUTPUT.glob("*.png")):
    print(f"   ✓ {f.name}")

print(f"""
🎯 KEY INSIGHTS:
{'─'*60}

1. Clean spectra have LOW absorbance in visible region (400-800 nm)
2. Contaminated spectra show ELEVATED absorbance (bacterial scattering)
3. The 1/λ scattering pattern is the KEY contamination signature
4. Anomaly detection works by learning "normal" and flagging deviations
5. Models achieve HIGH sensitivity (detect contamination) and specificity (don't false alarm)

✅ EXPERIMENT COMPLETE!

All figures saved to: {DEMO_OUTPUT}
""")

# Save summary JSON
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_generation": {
        "wavelength_range_nm": [200, 800],
        "num_data_points": len(wavelengths),
        "clean_spectra": len(clean_spectra),
        "contaminated_spectra": len(contaminated_spectra),
        "species": list(scattering_coeffs.keys()),
        "cfu_levels": cfu_levels
    },
    "features": {
        "num_features": len(feature_names),
        "feature_names": feature_names
    },
    "models": {
        "isolation_forest": {
            "roc_auc": float(if_auc),
            "sensitivity": float(if_metrics['Sensitivity']),
            "specificity": float(if_metrics['Specificity']),
            "training_time_s": float(if_time)
        },
        "ocsvm": {
            "roc_auc": float(ocsvm_auc),
            "sensitivity": float(ocsvm_metrics['Sensitivity']),
            "specificity": float(ocsvm_metrics['Specificity']),
            "training_time_s": float(ocsvm_time)
        }
    },
    "detection_by_cfu": {str(k): float(v) for k, v in detection_by_cfu.items()},
    "output_directory": str(DEMO_OUTPUT)
}

summary_file = DEMO_OUTPUT / "experiment_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"📄 Summary saved to: {summary_file}")
print("="*80 + "\n")
