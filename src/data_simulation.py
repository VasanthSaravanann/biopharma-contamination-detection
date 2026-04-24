"""
UV-Vis Spectroscopy Data Simulation Module

Generates realistic UV-Vis absorbance spectra for biopharmaceutical contamination detection.
Simulates clean/nominal process data and contaminated data across multiple microbial species.

References:
- UV-Vis spectroscopy principles for bioprocess monitoring
- Microbial contamination spectral signatures
- Beer-Lambert law for absorbance modeling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from scipy import signal
from scipy.stats import norm, lognorm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ContaminantType(Enum):
    """Types of microbial contaminants"""
    BACTERIA_E_COLI = "E_coli"
    BACTERIA_B_SUBTILIS = "B_subtilis"
    BACTERIA_P_AERUGINOSA = "P_aeruginosa"
    FUNGI_C_ALBICANS = "C_albicans"
    FUNGI_A_NIGER = "A_niger"
    MYCOPLASMA = "Mycoplasma"
    CLEAN = "Clean"


@dataclass
class ProcessConditions:
    """Process conditions affecting UV-Vis spectra"""
    temperature: float = 37.0  # °C
    ph: float = 7.2  # pH units
    dissolved_oxygen: float = 40.0  # % saturation
    batch_age: float = 24.0  # hours
    media_type: str = "DMEM"  # Culture media type
    
    def add_variation(self, temp_std: float = 0.5, ph_std: float = 0.1, 
                      do_std: float = 5.0, age_std: float = 2.0) -> 'ProcessConditions':
        """Add realistic process variation"""
        return ProcessConditions(
            temperature=max(20, min(45, np.random.normal(self.temperature, temp_std))),
            ph=max(6.0, min(8.0, np.random.normal(self.ph, ph_std))),
            dissolved_oxygen=max(0, min(100, np.random.normal(self.dissolved_oxygen, do_std))),
            batch_age=max(0, np.random.normal(self.batch_age, age_std)),
            media_type=self.media_type
        )


@dataclass
class SpectralParameters:
    """Parameters for UV-Vis spectrum generation"""
    wavelength_range: Tuple[float, float] = (200.0, 800.0)  # nm
    resolution: float = 1.0  # nm
    absorbance_range: Tuple[float, float] = (0.0, 3.0)  # AU
    
    @property
    def wavelengths(self) -> np.ndarray:
        """Generate wavelength array"""
        return np.arange(self.wavelength_range[0], 
                        self.wavelength_range[1] + self.resolution, 
                        self.resolution)


@dataclass
class ContaminantSignature:
    """Spectral signature parameters for a contaminant"""
    name: str
    peak_positions: List[float]  # nm
    peak_widths: List[float]  # nm (standard deviation)
    peak_ratios: List[float]  # Relative peak intensities
    base_absorbance: float  # Background absorbance level
    scattering_exponent: float  # For light scattering effects
    
    # pH-dependent shifts
    ph_sensitivity: float = 0.5  # nm/pH unit
    temp_sensitivity: float = 0.1  # nm/°C


class UVVisSpectraGenerator:
    """
    Generate realistic UV-Vis absorbance spectra for biopharmaceutical processes.
    
    Simulates:
    - Clean/nominal process spectra (sterile media, process fluids)
    - Contaminated spectra with microbial signatures
    - Process variable effects (temperature, pH, dissolved oxygen)
    - Instrument noise and batch effects
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the spectra generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.spectral_params = SpectralParameters()
        self.contaminant_signatures = self._load_contaminant_signatures()
        self.media_signatures = self._load_media_signatures()
        
    def _load_contaminant_signatures(self) -> Dict[ContaminantType, ContaminantSignature]:
        """
        Load spectral signatures for different contaminants.
        
        Based on literature values for microbial UV-Vis absorption:
        - Bacteria: Protein peaks at 280nm, nucleic acids at 260nm
        - Fungi: Additional pigment absorption in visible range
        - Mycoplasma: Distinctive membrane lipid signatures
        """
        return {
            ContaminantType.BACTERIA_E_COLI: ContaminantSignature(
                name="E. coli",
                peak_positions=[260, 280, 420, 550],
                peak_widths=[15, 20, 25, 30],
                peak_ratios=[1.0, 0.8, 0.3, 0.2],
                base_absorbance=0.02,
                scattering_exponent=2.5,
                ph_sensitivity=0.3,
                temp_sensitivity=0.05
            ),
            ContaminantType.BACTERIA_B_SUBTILIS: ContaminantSignature(
                name="B. subtilis",
                peak_positions=[260, 280, 410, 540],
                peak_widths=[15, 18, 20, 25],
                peak_ratios=[1.0, 0.85, 0.25, 0.15],
                base_absorbance=0.018,
                scattering_exponent=2.4,
                ph_sensitivity=0.35,
                temp_sensitivity=0.06
            ),
            ContaminantType.BACTERIA_P_AERUGINOSA: ContaminantSignature(
                name="P. aeruginosa",
                peak_positions=[260, 280, 380, 490, 620],  # Pyocyanin pigment
                peak_widths=[15, 20, 30, 35, 40],
                peak_ratios=[1.0, 0.75, 0.4, 0.5, 0.3],
                base_absorbance=0.025,
                scattering_exponent=2.6,
                ph_sensitivity=0.4,
                temp_sensitivity=0.08
            ),
            ContaminantType.FUNGI_C_ALBICANS: ContaminantSignature(
                name="C. albicans",
                peak_positions=[260, 280, 450, 580],
                peak_widths=[18, 22, 35, 40],
                peak_ratios=[1.0, 0.7, 0.35, 0.25],
                base_absorbance=0.03,
                scattering_exponent=2.8,
                ph_sensitivity=0.5,
                temp_sensitivity=0.1
            ),
            ContaminantType.FUNGI_A_NIGER: ContaminantSignature(
                name="A. niger",
                peak_positions=[260, 280, 420, 520, 650],
                peak_widths=[18, 22, 40, 45, 50],
                peak_ratios=[1.0, 0.65, 0.3, 0.4, 0.2],
                base_absorbance=0.035,
                scattering_exponent=3.0,
                ph_sensitivity=0.6,
                temp_sensitivity=0.12
            ),
            ContaminantType.MYCOPLASMA: ContaminantSignature(
                name="Mycoplasma",
                peak_positions=[260, 280, 340, 480],
                peak_widths=[12, 15, 20, 30],
                peak_ratios=[1.0, 0.9, 0.2, 0.15],
                base_absorbance=0.015,
                scattering_exponent=2.2,
                ph_sensitivity=0.25,
                temp_sensitivity=0.04
            )
        }
    
    def _load_media_signatures(self) -> Dict[str, np.ndarray]:
        """
        Load baseline media absorbance signatures.
        
        Different culture media have characteristic UV-Vis profiles.
        """
        wavelengths = self.spectral_params.wavelengths
        media_sigs = {}
        
        # DMEM media signature
        dmem = self._gaussian_peak(wavelengths, 230, 25, 0.15)  # Phenol red
        dmem += self._gaussian_peak(wavelengths, 280, 15, 0.08)  # Amino acids
        dmem += self._gaussian_peak(wavelengths, 430, 30, 0.05)  # Phenol red visible
        dmem += self._gaussian_peak(wavelengths, 560, 35, 0.03)  # Phenol red
        media_sigs["DMEM"] = np.maximum(0, dmem)
        
        # RPMI media signature
        rpmi = self._gaussian_peak(wavelengths, 230, 25, 0.12)
        rpmi += self._gaussian_peak(wavelengths, 280, 15, 0.06)
        rpmi += self._gaussian_peak(wavelengths, 430, 30, 0.04)
        media_sigs["RPMI"] = np.maximum(0, rpmi)
        
        # F-12 media signature
        f12 = self._gaussian_peak(wavelengths, 230, 25, 0.10)
        f12 += self._gaussian_peak(wavelengths, 280, 15, 0.05)
        f12 += self._gaussian_peak(wavelengths, 350, 20, 0.02)
        media_sigs["F12"] = np.maximum(0, f12)
        
        return media_sigs
    
    def _gaussian_peak(self, wavelengths: np.ndarray, center: float, 
                       width: float, amplitude: float) -> np.ndarray:
        """Generate a Gaussian absorption peak"""
        return amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
    
    def _lorentzian_peak(self, wavelengths: np.ndarray, center: float, 
                         width: float, amplitude: float) -> np.ndarray:
        """Generate a Lorentzian absorption peak"""
        return amplitude * (width ** 2) / ((wavelengths - center) ** 2 + width ** 2)
    
    def _generate_clean_spectrum(self, process_conditions: ProcessConditions,
                                  add_noise: bool = True,
                                  noise_level: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a clean/nominal spectrum (no contamination).
        
        Args:
            process_conditions: Process parameters
            add_noise: Whether to add instrument noise
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (wavelengths, absorbance)
        """
        wavelengths = self.spectral_params.wavelengths
        
        # Get base media signature
        media_type = process_conditions.media_type
        if media_type in self.media_signatures:
            absorbance = self.media_signatures[media_type].copy()
        else:
            absorbance = self.media_signatures["DMEM"].copy()
        
        # Add pH-dependent shifts (phenol red indicator)
        ph_shift = (process_conditions.ph - 7.2) * 2.0  # nm shift
        absorbance = np.roll(absorbance, int(ph_shift))
        
        # Add temperature effects (slight baseline shift)
        temp_factor = 1.0 + (process_conditions.temperature - 37.0) * 0.002
        absorbance *= temp_factor
        
        # Add dissolved oxygen effects (minor)
        do_factor = 1.0 + (process_conditions.dissolved_oxygen - 40.0) * 0.001
        absorbance *= do_factor
        
        # Add batch age effects (metabolite accumulation)
        age_factor = 1.0 + min(process_conditions.batch_age / 100.0, 0.3)
        absorbance *= age_factor
        
        # Add Rayleigh scattering baseline
        scattering = 0.01 * (wavelengths / 500.0) ** (-2.5)
        absorbance += scattering
        
        # Add instrument noise
        if add_noise:
            absorbance += np.random.normal(0, noise_level, len(absorbance))
        
        # Ensure non-negative absorbance
        absorbance = np.maximum(0, absorbance)
        
        return wavelengths, absorbance

    def _generate_benign_drift_spectrum(self, process_conditions: ProcessConditions,
                                        drift_magnitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a spectrum with benign process drift (e.g., metabolic pH drop).
        Crucially, this contains NO bacterial scattering or microbial peaks.
        """
        # Simulate significant pH drop due to cell metabolism (benign)
        drift_conditions = ProcessConditions(
            temperature=process_conditions.temperature,
            ph=max(6.0, process_conditions.ph - (0.5 * drift_magnitude)), 
            dissolved_oxygen=max(0, process_conditions.dissolved_oxygen * (1.0 - 0.2 * drift_magnitude)),
            batch_age=process_conditions.batch_age + (12 * drift_magnitude),
            media_type=process_conditions.media_type
        )
        
        # Generate as clean but with drifted conditions
        return self._generate_clean_spectrum(drift_conditions)

    def _generate_benign_scenario_spectrum(self, process_conditions: ProcessConditions,
                                           scenario: str,
                                           drift_magnitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate benign non-bacterial process-variation spectra.

        Supported scenarios:
        - ph_drift
        - temperature_drift
        - combined_drift
        """
        if scenario == 'ph_drift':
            drift_conditions = ProcessConditions(
                temperature=process_conditions.temperature,
                ph=max(6.0, min(8.0, process_conditions.ph - (0.7 * drift_magnitude))),
                dissolved_oxygen=process_conditions.dissolved_oxygen,
                batch_age=process_conditions.batch_age + (8 * drift_magnitude),
                media_type=process_conditions.media_type,
            )
        elif scenario == 'temperature_drift':
            drift_conditions = ProcessConditions(
                temperature=max(20.0, min(45.0, process_conditions.temperature + (3.0 * drift_magnitude))),
                ph=process_conditions.ph,
                dissolved_oxygen=max(0, process_conditions.dissolved_oxygen * (1.0 - 0.1 * drift_magnitude)),
                batch_age=process_conditions.batch_age + (6 * drift_magnitude),
                media_type=process_conditions.media_type,
            )
        elif scenario == 'combined_drift':
            drift_conditions = ProcessConditions(
                temperature=max(20.0, min(45.0, process_conditions.temperature + (2.0 * drift_magnitude))),
                ph=max(6.0, min(8.0, process_conditions.ph - (0.5 * drift_magnitude))),
                dissolved_oxygen=max(0, process_conditions.dissolved_oxygen * (1.0 - 0.2 * drift_magnitude)),
                batch_age=process_conditions.batch_age + (12 * drift_magnitude),
                media_type=process_conditions.media_type,
            )
        else:
            raise ValueError(f"Unknown benign scenario: {scenario}")

        return self._generate_clean_spectrum(drift_conditions)
    
    def evaluate_benign_drift(self, detector, extractor=None, n_samples: int = 100) -> Dict:
        """
        Evaluation helper: asserts that benign drift is predicted as normal.
        """
        records = []
        wavelengths = self.spectral_params.wavelengths
        
        for i in range(n_samples):
            proc_cond = ProcessConditions().add_variation()
            _, absorbance = self._generate_benign_drift_spectrum(proc_cond)
            records.append(absorbance)
            
        X_drift_raw = np.vstack(records)
        
        if extractor is not None:
            df_drift = pd.DataFrame(X_drift_raw, columns=[f'abs_{int(w)}' for w in wavelengths])
            X_test_all = extractor.extract_all_features(df_drift)
            # Try to match the number of features the detector expects
            if hasattr(detector, 'scaler') and hasattr(detector.scaler, 'n_features_in_'):
                n_expected = detector.scaler.n_features_in_
                # Standard feature subsets
                abs_cols = [c for c in X_test_all.columns if c.startswith('abs_')]
                if len(abs_cols) == n_expected:
                    X_test = X_test_all[abs_cols].values
                else:
                    # Just take the first N features if we can't match
                    X_test = X_test_all.iloc[:, :n_expected].values
            else:
                X_test = X_test_all.values
        else:
            X_test = X_drift_raw
            
        y_pred = detector.predict(X_test)
        normal_rate = np.mean(y_pred == 1)
        
        return {
            'n_samples': n_samples,
            'predicted_normal_rate': float(normal_rate),
            'false_positive_rate': float(1.0 - normal_rate),
            'status': "PASS" if normal_rate >= 0.90 else "FAIL"
        }

    def evaluate_benign_controls(self, detector, extractor=None, n_samples: int = 100,
                                 scenarios: Optional[List[str]] = None) -> Dict:
        """
        Evaluate non-bacterial controls and report false-positive behavior.

        Args:
            detector: Trained anomaly detector with predict()
            extractor: Optional SpectralFeatureExtractor
            n_samples: Samples per scenario
            scenarios: Optional scenario list

        Returns:
            Scenario-wise normal/anomaly rates and overall summary
        """
        if scenarios is None:
            scenarios = ['ph_drift', 'temperature_drift']

        wavelengths = self.spectral_params.wavelengths
        scenario_results = {}

        for scenario in scenarios:
            records = []
            for _ in range(n_samples):
                proc_cond = ProcessConditions().add_variation()
                _, absorbance = self._generate_benign_scenario_spectrum(
                    proc_cond,
                    scenario=scenario,
                    drift_magnitude=np.random.uniform(0.6, 1.2),
                )
                records.append(absorbance)

            X_raw = np.vstack(records)

            if extractor is not None:
                df_drift = pd.DataFrame(X_raw, columns=[f'abs_{int(w)}' for w in wavelengths])
                X_test_all = extractor.extract_all_features(df_drift)

                if hasattr(detector, 'scaler') and hasattr(detector.scaler, 'n_features_in_'):
                    n_expected = detector.scaler.n_features_in_
                    abs_cols = [c for c in X_test_all.columns if c.startswith('abs_')]
                    if len(abs_cols) == n_expected:
                        X_test = X_test_all[abs_cols].values
                    else:
                        X_test = X_test_all.iloc[:, :n_expected].values
                else:
                    X_test = X_test_all.values
            else:
                X_test = X_raw

            y_pred = detector.predict(X_test)
            normal_rate = float(np.mean(y_pred == 1))
            scenario_results[scenario] = {
                'n_samples': int(n_samples),
                'predicted_normal_rate': normal_rate,
                'predicted_anomaly_rate': float(1.0 - normal_rate),
                'false_positive_rate': float(1.0 - normal_rate),
                'status': 'PASS' if normal_rate >= 0.90 else 'FAIL',
            }

        avg_normal = np.mean([v['predicted_normal_rate'] for v in scenario_results.values()])
        return {
            'scenarios': scenario_results,
            'average_predicted_normal_rate': float(avg_normal),
            'average_false_positive_rate': float(1.0 - avg_normal),
            'overall_status': 'PASS' if avg_normal >= 0.90 else 'FAIL',
        }

    def _generate_contaminant_spectrum(self, contaminant_type: ContaminantType,
                                        inoculum_level: float,
                                        process_conditions: ProcessConditions,
                                        add_noise: bool = True,
                                        noise_level: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a contaminated spectrum.
        
        Args:
            contaminant_type: Type of contaminant
            inoculum_level: CFU/mL
            process_conditions: Process parameters
            add_noise: Whether to add instrument noise
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (wavelengths, absorbance)
        """
        wavelengths = self.spectral_params.wavelengths
        
        # Start with clean spectrum
        _, absorbance = self._generate_clean_spectrum(process_conditions, add_noise=False)
        
        # Get contaminant signature
        signature = self.contaminant_signatures[contaminant_type]
        
        # Calculate concentration-dependent absorbance (Beer-Lambert law)
        # Log-linear relationship between CFU/mL and absorbance
        log_cfu = np.log10(inoculum_level)
        concentration_factor = (log_cfu - 1) / 2.0  # Normalize to 10-1000 CFU/mL range
        
        # Add contaminant peaks with pH/temperature shifts
        ph_shift = (process_conditions.ph - 7.2) * signature.ph_sensitivity
        temp_shift = (process_conditions.temperature - 37.0) * signature.temp_sensitivity
        total_shift = int(ph_shift + temp_shift)
        
        for peak_pos, peak_width, peak_ratio in zip(
            signature.peak_positions, 
            signature.peak_widths, 
            signature.peak_ratios
        ):
            shifted_pos = peak_pos + total_shift
            peak_amplitude = peak_ratio * concentration_factor * signature.base_absorbance
            absorbance += self._gaussian_peak(wavelengths, shifted_pos, peak_width, peak_amplitude)
        
        # Add light scattering from biomass
        scattering_amplitude = concentration_factor * 0.02 * signature.scattering_exponent
        scattering = scattering_amplitude * (wavelengths / 500.0) ** (-signature.scattering_exponent)
        absorbance += scattering
        
        # Add process condition effects
        temp_factor = 1.0 + (process_conditions.temperature - 37.0) * 0.003
        absorbance *= temp_factor
        
        # Add instrument noise
        if add_noise:
            absorbance += np.random.normal(0, noise_level, len(absorbance))
        
        # Ensure non-negative absorbance
        absorbance = np.maximum(0, absorbance)
        
        return wavelengths, absorbance
    
    def generate_spectrum(self, contaminant_type: ContaminantType,
                          inoculum_level: Optional[float] = None,
                          process_conditions: Optional[ProcessConditions] = None,
                          add_noise: bool = True,
                          noise_level: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single UV-Vis spectrum.
        
        Args:
            contaminant_type: Type of contaminant (use CLEAN for nominal data)
            inoculum_level: CFU/mL (required for contaminated, ignored for clean)
            process_conditions: Process parameters (uses defaults if None)
            add_noise: Whether to add instrument noise
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (wavelengths, absorbance)
        """
        if process_conditions is None:
            process_conditions = ProcessConditions()
        
        if contaminant_type == ContaminantType.CLEAN:
            return self._generate_clean_spectrum(process_conditions, add_noise, noise_level)
        else:
            if inoculum_level is None:
                raise ValueError("inoculum_level required for contaminated spectra")
            return self._generate_contaminant_spectrum(
                contaminant_type, inoculum_level, process_conditions, add_noise, noise_level
            )
    
    def generate_dataset(self, n_clean: int = 1000,
                         n_contaminated_per_type: int = 200,
                         inoculum_levels: Optional[List[float]] = None,
                         include_process_variation: bool = True,
                         seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a complete dataset for training and validation.
        
        Args:
            n_clean: Number of clean/nominal spectra
            n_contaminated_per_type: Number of contaminated spectra per contaminant type
            inoculum_levels: List of CFU/mL levels to simulate
            include_process_variation: Add realistic process variation
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with spectra and metadata
        """
        if seed is not None:
            np.random.seed(seed)
        
        if inoculum_levels is None:
            inoculum_levels = [10, 25, 50, 100, 250, 500, 1000]
        
        wavelengths = self.spectral_params.wavelengths
        records = []
        
        # Generate clean spectra
        print(f"Generating {n_clean} clean spectra...")
        for i in range(n_clean):
            if include_process_variation:
                proc_cond = ProcessConditions().add_variation()
            else:
                proc_cond = ProcessConditions()
            
            _, absorbance = self._generate_clean_spectrum(proc_cond)
            
            records.append({
                'spectrum_id': f'clean_{i:05d}',
                'contaminant_type': 'Clean',
                'inoculum_level': 0,
                'temperature': proc_cond.temperature,
                'ph': proc_cond.ph,
                'dissolved_oxygen': proc_cond.dissolved_oxygen,
                'batch_age': proc_cond.batch_age,
                'media_type': proc_cond.media_type,
                'label': 0,  # 0 = clean
                **{f'abs_{int(w)}': a for w, a in zip(wavelengths, absorbance)}
            })
        
        # Generate contaminated spectra
        contaminant_types = [ct for ct in ContaminantType if ct != ContaminantType.CLEAN]
        
        for contaminant_type in contaminant_types:
            print(f"Generating {n_contaminated_per_type} {contaminant_type.value} spectra...")
            
            for i in range(n_contaminated_per_type):
                # Random inoculum level
                inoculum = np.random.choice(inoculum_levels)
                
                if include_process_variation:
                    proc_cond = ProcessConditions().add_variation()
                else:
                    proc_cond = ProcessConditions()
                
                _, absorbance = self._generate_contaminant_spectrum(
                    contaminant_type, inoculum, proc_cond
                )
                
                records.append({
                    'spectrum_id': f'{contaminant_type.value}_{i:05d}',
                    'contaminant_type': contaminant_type.value,
                    'inoculum_level': inoculum,
                    'temperature': proc_cond.temperature,
                    'ph': proc_cond.ph,
                    'dissolved_oxygen': proc_cond.dissolved_oxygen,
                    'batch_age': proc_cond.batch_age,
                    'media_type': proc_cond.media_type,
                    'label': 1,  # 1 = contaminated
                    **{f'abs_{int(w)}': a for w, a in zip(wavelengths, absorbance)}
                })
        
        df = pd.DataFrame(records)
        print(f"Generated dataset with {len(df)} spectra")
        print(f"  Clean: {n_clean}")
        print(f"  Contaminated: {len(contaminant_types) * n_contaminated_per_type}")
        
        return df
    
    def add_batch_effect(self, df: pd.DataFrame, n_batches: int = 5,
                         batch_key: str = 'batch_id') -> pd.DataFrame:
        """
        Add realistic batch effects to a dataset.
        
        Args:
            df: Input DataFrame with spectra
            n_batches: Number of batches to simulate
            batch_key: Column name for batch ID
            
        Returns:
            DataFrame with batch effects added
        """
        df = df.copy()
        wavelengths = self.spectral_params.wavelengths
        abs_cols = [f'abs_{int(w)}' for w in wavelengths]
        
        # Assign batches
        df[batch_key] = np.random.choice(range(n_batches), len(df))
        
        # Add batch-specific shifts
        for batch_id in range(n_batches):
            batch_mask = df[batch_key] == batch_id
            
            # Random baseline shift
            baseline_shift = np.random.normal(0, 0.02)
            
            # Random scaling
            scale_factor = np.random.normal(1.0, 0.05)
            
            # Random wavelength-dependent shift
            wavelength_shift = np.random.normal(0, 0.005, len(wavelengths))
            wavelength_shift = signal.savgol_filter(wavelength_shift, 51, 3)
            
            # Apply batch effects
            for i, col in enumerate(abs_cols):
                df.loc[batch_mask, col] = (
                    df.loc[batch_mask, col] * scale_factor + 
                    baseline_shift + 
                    wavelength_shift[i]
                )
            
            # Ensure non-negative
            df.loc[batch_mask, abs_cols] = df.loc[batch_mask, abs_cols].clip(lower=0)
        
        return df
    
    def add_instrument_variation(self, df: pd.DataFrame, n_instruments: int = 3,
                                  instrument_key: str = 'instrument_id') -> pd.DataFrame:
        """
        Add instrument-to-instrument variation.
        
        Args:
            df: Input DataFrame with spectra
            n_instruments: Number of instruments to simulate
            instrument_key: Column name for instrument ID
            
        Returns:
            DataFrame with instrument variation added
        """
        df = df.copy()
        wavelengths = self.spectral_params.wavelengths
        abs_cols = [f'abs_{int(w)}' for w in wavelengths]
        
        # Assign instruments
        df[instrument_key] = np.random.choice(range(n_instruments), len(df))
        
        # Add instrument-specific calibration differences
        for inst_id in range(n_instruments):
            inst_mask = df[instrument_key] == inst_id
            
            # Wavelength calibration shift
            wl_shift = np.random.randint(-2, 3)  # ±2 nm
            
            # Photometric accuracy difference
            photo_factor = np.random.normal(1.0, 0.03)
            
            # Stray light effect (more at low wavelengths)
            stray_light = 0.01 * np.exp(-(wavelengths - 200) / 100)
            
            # Apply instrument effects
            for i, col in enumerate(abs_cols):
                df.loc[inst_mask, col] = (
                    df.loc[inst_mask, col] * photo_factor + stray_light[i]
                )
            
            # Ensure non-negative
            df.loc[inst_mask, abs_cols] = df.loc[inst_mask, abs_cols].clip(lower=0)
        
        return df


def generate_sample_dataset(output_path: str = "data/raw/spectra_dataset.csv",
                            seed: int = 42) -> pd.DataFrame:
    """
    Generate a sample dataset for demonstration.
    
    Args:
        output_path: Path to save the dataset
        seed: Random seed
        
    Returns:
        Generated DataFrame
    """
    generator = UVVisSpectraGenerator(seed=seed)
    
    df = generator.generate_dataset(
        n_clean=1000,
        n_contaminated_per_type=150,
        inoculum_levels=[10, 25, 50, 100, 250, 500, 1000],
        include_process_variation=True,
        seed=seed
    )
    
    # Add batch and instrument effects
    df = generator.add_batch_effect(df, n_batches=5)
    df = generator.add_instrument_variation(df, n_instruments=3)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample dataset
    df = generate_sample_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn count: {len(df.columns)}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nContaminant type distribution:\n{df['contaminant_type'].value_counts()}")
