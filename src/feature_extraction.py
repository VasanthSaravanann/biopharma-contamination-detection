"""
Feature Extraction Pipeline for UV-Vis Spectra

Extracts meaningful features from UV-Vis absorbance spectra for contamination detection.
Includes metabolite-linked features, pH-related shifts, biomass patterns, and peak analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import mahalanobis
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    # Spectral regions of interest (nm)
    uv_region: Tuple[float, float] = (200, 400)
    visible_region: Tuple[float, float] = (400, 800)
    
    # Key wavelengths for specific analytes
    protein_wavelength: float = 280  # nm (aromatic amino acids)
    nucleic_acid_wavelength: float = 260  # nm
    phenol_red_wavelength: float = 430  # nm (pH indicator)
    
    # Peak detection parameters
    peak_prominence: float = 0.01  # Minimum peak prominence
    peak_width_range: Tuple[float, float] = (5, 50)  # nm
    
    # Smoothing parameters
    smoothing_window: int = 11  # Savitzky-Golay window
    smoothing_order: int = 3  # Polynomial order


class SpectralFeatureExtractor:
    """
    Extract features from UV-Vis spectra for contamination detection.
    
    Features include:
    - Direct absorbance values at key wavelengths
    - Peak characteristics (position, height, width, area)
    - Spectral ratios and derivatives
    - pH-sensitive features
    - Biomass indicators
    - Statistical features
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None,
                 wavelengths: Optional[np.ndarray] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Feature extraction configuration
            wavelengths: Wavelength array (200-800 nm by default)
        """
        self.config = config or FeatureConfig()
        
        if wavelengths is None:
            self.wavelengths = np.arange(200, 801, 1)
        else:
            self.wavelengths = wavelengths
        
        self.abs_cols = [f'abs_{int(w)}' for w in self.wavelengths]
        
        # Pre-compute wavelength indices for key regions
        self.uv_indices = self._get_indices(self.config.uv_region)
        self.vis_indices = self._get_indices(self.config.visible_region)
        self.protein_idx = self._find_closest_index(self.config.protein_wavelength)
        self.nucleic_idx = self._find_closest_index(self.config.nucleic_acid_wavelength)
        self.phenol_red_idx = self._find_closest_index(self.config.phenol_red_wavelength)
    
    def _get_indices(self, wavelength_range: Tuple[float, float]) -> np.ndarray:
        """Get indices for a wavelength range"""
        mask = (self.wavelengths >= wavelength_range[0]) & \
               (self.wavelengths <= wavelength_range[1])
        return np.where(mask)[0]
    
    def _find_closest_index(self, wavelength: float) -> int:
        """Find index closest to a specific wavelength"""
        return int(np.argmin(np.abs(self.wavelengths - wavelength)))
    
    def _get_spectrum(self, row: pd.Series) -> np.ndarray:
        """Extract spectrum array from a DataFrame row"""
        return row[self.abs_cols].values
    
    def extract_all_features(self, df: pd.DataFrame,
                             include_derivatives: bool = True,
                             include_statistical: bool = True) -> pd.DataFrame:
        """
        Extract all features from a DataFrame of spectra.
        
        Args:
            df: Input DataFrame with absorbance columns
            include_derivatives: Include spectral derivatives
            include_statistical: Include statistical features
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for idx, row in df.iterrows():
            features = self.extract_single_spectrum(row, include_derivatives, include_statistical)
            features['spectrum_id'] = row.get('spectrum_id', f'sample_{idx}')
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Preserve original metadata columns
        metadata_cols = ['spectrum_id', 'contaminant_type', 'inoculum_level', 
                         'label', 'temperature', 'ph', 'dissolved_oxygen', 
                         'batch_age', 'media_type', 'batch_id', 'instrument_id']
        
        for col in metadata_cols:
            if col in df.columns:
                features_df[col] = df[col].values
        
        return features_df
    
    def extract_single_spectrum(self, row: pd.Series,
                                 include_derivatives: bool = True,
                                 include_statistical: bool = True) -> Dict:
        """
        Extract features from a single spectrum.
        
        Args:
            row: DataFrame row with absorbance values
            include_derivatives: Include spectral derivatives
            include_statistical: Include statistical features
            
        Returns:
            Dictionary of extracted features
        """
        spectrum = self._get_spectrum(row)
        features = {}
        
        # 1. Key wavelength absorbances
        features.update(self._extract_key_absorbances(spectrum))
        
        # 2. Spectral ratios
        features.update(self._extract_spectral_ratios(spectrum))
        
        # 3. Peak features
        features.update(self._extract_peak_features(spectrum))
        
        # 4. pH-sensitive features
        features.update(self._extract_ph_features(spectrum, row.get('ph', 7.2)))
        
        # 5. Biomass indicators
        features.update(self._extract_biomass_features(spectrum))
        
        # 6. Derivative features
        if include_derivatives:
            features.update(self._extract_derivative_features(spectrum))
        
        # 7. Statistical features
        if include_statistical:
            features.update(self._extract_statistical_features(spectrum))
        
        # 8. Region integrals
        features.update(self._extract_region_integrals(spectrum))
        
        return features
    
    def _extract_key_absorbances(self, spectrum: np.ndarray) -> Dict:
        """Extract absorbance at key wavelengths"""
        return {
            'abs_260': spectrum[self.nucleic_idx],
            'abs_280': spectrum[self.protein_idx],
            'abs_430': spectrum[self.phenol_red_idx],
            'abs_600': spectrum[self._find_closest_index(600)],  # Turbidity
            'a260_a280_ratio': spectrum[self.nucleic_idx] / max(spectrum[self.protein_idx], 0.001),
        }
    
    def _extract_spectral_ratios(self, spectrum: np.ndarray) -> Dict:
        """Extract diagnostic spectral ratios"""
        uv_mean = np.mean(spectrum[self.uv_indices])
        vis_mean = np.mean(spectrum[self.vis_indices])
        
        # Specific ratios for contamination detection
        return {
            'uv_vis_ratio': uv_mean / max(vis_mean, 0.001),
            'uv_mean': uv_mean,
            'vis_mean': vis_mean,
            'slope_350_400': (spectrum[self._find_closest_index(400)] - 
                             spectrum[self._find_closest_index(350)]) / 50,
            'slope_500_600': (spectrum[self._find_closest_index(600)] - 
                             spectrum[self._find_closest_index(500)]) / 100,
        }
    
    def _extract_peak_features(self, spectrum: np.ndarray) -> Dict:
        """Extract peak characteristics"""
        # Smooth spectrum for peak detection
        smoothed = signal.savgol_filter(spectrum, 
                                        self.config.smoothing_window,
                                        self.config.smoothing_order)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            smoothed,
            prominence=self.config.peak_prominence,
            width=self.config.peak_width_range
        )
        
        features = {
            'n_peaks': len(peaks),
            'max_peak_height': np.max(smoothed) if len(smoothed) > 0 else 0,
            'max_peak_position': self.wavelengths[np.argmax(smoothed)] if len(smoothed) > 0 else 0,
        }
        
        # Peak areas (integration)
        if len(peaks) > 0:
            # Total peak area
            total_area = 0
            for i, peak in enumerate(peaks):
                left = properties['left_bases'][i]
                right = properties['right_bases'][i]
                if left is not None and right is not None:
                    peak_area = np.trapz(smoothed[int(left):int(right)],
                                        self.wavelengths[int(left):int(right)])
                    total_area += peak_area
            
            features['total_peak_area'] = total_area
            features['avg_peak_width'] = np.mean(properties['widths'])
        else:
            features['total_peak_area'] = 0
            features['avg_peak_width'] = 0
        
        # Specific peak heights
        features['peak_260_height'] = self._get_local_max(spectrum, 260, 20)
        features['peak_280_height'] = self._get_local_max(spectrum, 280, 20)
        features['peak_420_height'] = self._get_local_max(spectrum, 420, 30)
        
        return features
    
    def _get_local_max(self, spectrum: np.ndarray, center_wl: float, 
                       window: float) -> float:
        """Get local maximum around a wavelength"""
        center_idx = self._find_closest_index(center_wl)
        window_idx = int(window / 2)
        
        start_idx = max(0, center_idx - window_idx)
        end_idx = min(len(spectrum), center_idx + window_idx)
        
        return np.max(spectrum[start_idx:end_idx])
    
    def _extract_ph_features(self, spectrum: np.ndarray, ph: float) -> Dict:
        """Extract pH-sensitive features"""
        # Phenol red ratio (pH indicator)
        abs_430 = spectrum[self.phenol_red_idx]
        abs_560 = spectrum[self._find_closest_index(560)]
        
        # pH-sensitive peak shift
        uv_region = spectrum[self.uv_indices]
        if len(uv_region) > 10:
            # Find UV peak position
            uv_peak_idx = np.argmax(uv_region)
            uv_peak_wl = self.wavelengths[self.uv_indices[0] + uv_peak_idx]
        else:
            uv_peak_wl = 280
        
        return {
            'phenol_red_ratio': abs_430 / max(abs_560, 0.001),
            'ph_indicator_diff': abs_430 - abs_560,
            'uv_peak_position': uv_peak_wl,
            'estimated_ph_shift': (uv_peak_wl - 280) / 0.5,  # Approximate
        }
    
    def _extract_biomass_features(self, spectrum: np.ndarray) -> Dict:
        """Extract biomass-related features"""
        # Scattering coefficient (turbidity indicator)
        # Fit power law: A = k * λ^(-n)
        log_wl = np.log(self.wavelengths[self.vis_indices])
        log_abs = np.log(spectrum[self.vis_indices] + 0.001)  # Avoid log(0)
        
        if len(log_wl) > 2:
            try:
                slope, intercept, _, _, _ = stats.linregress(log_wl, log_abs)
                scattering_exponent = -slope
                scattering_coefficient = np.exp(intercept)
            except:
                scattering_exponent = 2.5
                scattering_coefficient = 0.01
        else:
            scattering_exponent = 2.5
            scattering_coefficient = 0.01
        
        # Biomass indicators
        return {
            'scattering_exponent': scattering_exponent,
            'scattering_coefficient': scattering_coefficient,
            'turbidity_600': spectrum[self._find_closest_index(600)],
            'turbidity_700': spectrum[self._find_closest_index(700)],
            'biomass_index': (spectrum[self.protein_idx] + spectrum[self.nucleic_idx]) / 2,
        }
    
    def _extract_derivative_features(self, spectrum: np.ndarray) -> Dict:
        """Extract spectral derivative features"""
        # First derivative
        first_deriv = np.gradient(spectrum, self.wavelengths)
        
        # Second derivative
        second_deriv = np.gradient(first_deriv, self.wavelengths)
        
        # Smoothed derivatives
        first_deriv_smooth = signal.savgol_filter(spectrum, 
                                                   self.config.smoothing_window,
                                                   self.config.smoothing_order,
                                                   deriv=1)
        
        return {
            'first_deriv_max': np.max(first_deriv),
            'first_deriv_min': np.min(first_deriv),
            'first_deriv_std': np.std(first_deriv),
            'second_deriv_max': np.max(second_deriv),
            'second_deriv_min': np.min(second_deriv),
            'second_deriv_std': np.std(second_deriv),
            'deriv_zero_crossings': np.sum(np.abs(np.diff(np.sign(first_deriv))) > 0),
        }
    
    def _extract_statistical_features(self, spectrum: np.ndarray) -> Dict:
        """Extract statistical features from spectrum"""
        return {
            'mean_abs': np.mean(spectrum),
            'std_abs': np.std(spectrum),
            'min_abs': np.min(spectrum),
            'max_abs': np.max(spectrum),
            'range_abs': np.max(spectrum) - np.min(spectrum),
            'skewness': stats.skew(spectrum),
            'kurtosis': stats.kurtosis(spectrum),
            'median_abs': np.median(spectrum),
            'iqr_abs': np.percentile(spectrum, 75) - np.percentile(spectrum, 25),
        }
    
    def _extract_region_integrals(self, spectrum: np.ndarray) -> Dict:
        """Extract integrated absorbance over key regions"""
        regions = {
            'uv_200_250': (200, 250),
            'uv_250_300': (250, 300),
            'uv_300_400': (300, 400),
            'vis_400_500': (400, 500),
            'vis_500_600': (500, 600),
            'vis_600_700': (600, 700),
            'vis_700_800': (700, 800),
        }
        
        integrals = {}
        for name, (wl_start, wl_end) in regions.items():
            start_idx = self._find_closest_index(wl_start)
            end_idx = self._find_closest_index(wl_end)
            
            integral = np.trapz(spectrum[start_idx:end_idx],
                               self.wavelengths[start_idx:end_idx])
            integrals[f'integral_{name}'] = integral
        
        return integrals


class MetaboliteFeatureExtractor:
    """
    Extract metabolite-specific features from UV-Vis spectra.
    
    Identifies and quantifies metabolites based on their characteristic
    absorption patterns.
    """
    
    def __init__(self, wavelengths: Optional[np.ndarray] = None):
        """
        Initialize metabolite extractor.
        
        Args:
            wavelengths: Wavelength array
        """
        if wavelengths is None:
            self.wavelengths = np.arange(200, 801, 1)
        else:
            self.wavelengths = wavelengths
        
        # Metabolite absorption signatures (wavelength, width, relative intensity)
        self.metabolite_signatures = {
            'aromatic_amino_acids': [(280, 15, 1.0), (260, 20, 0.5)],
            'nucleotides': [(260, 12, 1.0), (280, 15, 0.3)],
            'phenol_red': [(430, 25, 1.0), (560, 30, 0.5)],
            'pyocyanin': [(380, 20, 0.8), (490, 25, 1.0), (620, 30, 0.6)],
            'flavins': [(370, 20, 0.7), (450, 25, 1.0)],
            'cytochromes': [(420, 15, 1.0), (530, 20, 0.6), (560, 20, 0.5)],
        }
    
    def extract_metabolite_features(self, spectrum: np.ndarray) -> Dict:
        """
        Extract metabolite-specific features.
        
        Args:
            spectrum: Absorbance spectrum
            
        Returns:
            Dictionary of metabolite features
        """
        features = {}
        
        for metabolite, peaks in self.metabolite_signatures.items():
            metabolite_score = 0
            
            for wl, width, intensity in peaks:
                # Find closest wavelength index
                idx = int(np.argmin(np.abs(self.wavelengths - wl)))
                
                # Get local absorbance
                window = int(width / 2)
                start_idx = max(0, idx - window)
                end_idx = min(len(spectrum), idx + window)
                
                local_abs = spectrum[start_idx:end_idx]
                
                if len(local_abs) > 0:
                    metabolite_score += np.mean(local_abs) * intensity
            
            # Normalize by number of peaks
            metabolite_score /= len(peaks)
            features[f'metabolite_{metabolite}'] = metabolite_score
        
        # Calculate metabolite ratios
        if features.get('metabolite_aromatic_amino_acids', 0) > 0:
            features['nucleotide_protein_ratio'] = (
                features.get('metabolite_nucleotides', 0) / 
                features['metabolite_aromatic_amino_acids']
            )
        else:
            features['nucleotide_protein_ratio'] = 0
        
        return features


def extract_features_from_dataset(input_path: str,
                                   output_path: str,
                                   include_derivatives: bool = True,
                                   include_statistical: bool = True) -> pd.DataFrame:
    """
    Extract features from a saved dataset.
    
    Args:
        input_path: Path to input CSV with spectra
        output_path: Path to save features
        include_derivatives: Include derivative features
        include_statistical: Include statistical features
        
    Returns:
        DataFrame with extracted features
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {len(df)} spectra")
    
    # Initialize extractor
    extractor = SpectralFeatureExtractor()
    
    # Extract features
    print("Extracting features...")
    features_df = extractor.extract_all_features(
        df,
        include_derivatives=include_derivatives,
        include_statistical=include_statistical
    )
    
    # Save features
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(f"Extracted {len(features_df.columns)} features")
    
    return features_df


if __name__ == "__main__":
    # Test feature extraction
    from data_simulation import UVVisSpectraGenerator, ContaminantType
    
    # Generate test data
    generator = UVVisSpectraGenerator(seed=42)
    test_df = generator.generate_dataset(n_clean=100, n_contaminated_per_type=50, seed=42)
    
    # Extract features
    extractor = SpectralFeatureExtractor()
    features_df = extractor.extract_all_features(test_df)
    
    print(f"\nExtracted {len(features_df.columns)} features")
    print(f"\nFeature columns:")
    for col in features_df.columns:
        print(f"  - {col}")
    
    # Show sample features
    print(f"\nSample feature values:")
    print(features_df.iloc[0][['abs_260', 'abs_280', 'a260_a280_ratio', 
                               'n_peaks', 'scattering_exponent', 'biomass_index']])
