"""
Validation Pipeline for Contamination Detection System

Comprehensive validation including:
- Maximum Mean Discrepancy (MMD) for distribution comparison
- Jensen-Shannon Divergence (JSD)
- Sensitivity/Specificity analysis
- Detection limit verification
- 30-minute detection window simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline"""
    # MMD parameters
    mmd_kernel: str = 'rbf'
    mmd_gamma: Optional[float] = None  # Auto-calculated if None
    
    # Detection thresholds
    target_detection_limit: float = 10.0  # CFU/mL
    detection_window_minutes: int = 30
    
    # Statistical tests
    confidence_level: float = 0.95
    n_bootstrap_iterations: int = 1000
    
    # Performance targets
    target_sensitivity: float = 0.90
    target_specificity: float = 0.95
    target_auc: float = 0.95


class MaximumMeanDiscrepancy:
    """
    Calculate Maximum Mean Discrepancy between two distributions.
    
    MMD measures the distance between two probability distributions
    in a reproducing kernel Hilbert space (RKHS).
    """
    
    def __init__(self, kernel: str = 'rbf', gamma: Optional[float] = None):
        """
        Initialize MMD calculator.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            gamma: Kernel bandwidth (auto-calculated if None)
        """
        self.kernel = kernel
        self.gamma = gamma
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, 
                    gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix"""
        # Compute squared Euclidean distances
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
        distances = X_sq + Y_sq - 2 * np.dot(X, Y.T)
        distances = np.maximum(distances, 0)  # Numerical stability
        
        return np.exp(-gamma * distances)
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute linear kernel matrix"""
        return np.dot(X, Y.T)
    
    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray,
                           degree: int = 3) -> np.ndarray:
        """Compute polynomial kernel matrix"""
        return (np.dot(X, Y.T) + 1) ** degree
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray,
                        gamma: float) -> np.ndarray:
        """Compute kernel matrix based on kernel type"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y, gamma)
        elif self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")
    
    def _median_heuristic(self, X: np.ndarray) -> float:
        """Calculate gamma using median heuristic"""
        # Compute pairwise distances
        n = len(X)
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        distances = X_sq + X_sq.T - 2 * np.dot(X, X.T)
        distances = np.maximum(distances, 0)
        
        # Get median of upper triangle
        upper_tri = distances[np.triu_indices(n, k=1)]
        median_dist = np.median(upper_tri)
        
        # Gamma = 1 / (2 * median_distance^2)
        return 1.0 / (2 * median_dist ** 2 + 1e-10)
    
    def calculate(self, X: np.ndarray, Y: np.ndarray,
                  n_permutations: int = 100) -> Tuple[float, float]:
        """
        Calculate MMD between two distributions.
        
        Args:
            X: Samples from first distribution (n_samples, n_features)
            Y: Samples from second distribution (n_samples, n_features)
            n_permutations: Number of permutations for p-value
            
        Returns:
            Tuple of (MMD value, p-value)
        """
        # Auto-calculate gamma if not provided
        if self.gamma is None:
            combined = np.vstack([X, Y])
            self.gamma = self._median_heuristic(combined)
        
        n_x, n_y = len(X), len(Y)
        
        # Compute kernel matrices
        K_XX = self._compute_kernel(X, X, self.gamma)
        K_YY = self._compute_kernel(Y, Y, self.gamma)
        K_XY = self._compute_kernel(X, Y, self.gamma)
        
        # Calculate MMD^2
        mmd_squared = (
            np.sum(K_XX) / (n_x * n_x) +
            np.sum(K_YY) / (n_y * n_y) -
            2 * np.sum(K_XY) / (n_x * n_y)
        )
        
        mmd_squared = max(0, mmd_squared)  # Numerical stability
        mmd = np.sqrt(mmd_squared)
        
        # Calculate p-value using permutation test
        p_value = self._permutation_test(X, Y, mmd, n_permutations)
        
        return mmd, p_value
    
    def _permutation_test(self, X: np.ndarray, Y: np.ndarray,
                          observed_mmd: float, n_permutations: int) -> float:
        """Calculate p-value using permutation test"""
        combined = np.vstack([X, Y])
        n_x = len(X)
        
        permuted_mmds = []
        for _ in range(n_permutations):
            # Shuffle combined data
            indices = np.random.permutation(len(combined))
            X_perm = combined[indices[:n_x]]
            Y_perm = combined[indices[n_x:]]
            
            # Calculate MMD for permuted data
            K_XX = self._compute_kernel(X_perm, X_perm, self.gamma)
            K_YY = self._compute_kernel(Y_perm, Y_perm, self.gamma)
            K_XY = self._compute_kernel(X_perm, Y_perm, self.gamma)
            
            mmd_squared = (
                np.sum(K_XX) / (n_x * n_x) +
                np.sum(K_YY) / (len(Y_perm) * len(Y_perm)) -
                2 * np.sum(K_XY) / (n_x * len(Y_perm))
            )
            
            permuted_mmds.append(max(0, mmd_squared))
        
        # P-value = proportion of permuted MMDs >= observed MMD
        p_value = np.mean([m >= observed_mmd ** 2 for m in permuted_mmds])
        
        return p_value


class JensenShannonDivergence:
    """
    Calculate Jensen-Shannon Divergence between distributions.
    
    JSD is a symmetric, smoothed version of KL divergence.
    Range: [0, 1] where 0 = identical distributions.
    """
    
    def __init__(self, n_bins: int = 50):
        """
        Initialize JSD calculator.
        
        Args:
            n_bins: Number of bins for histogram estimation
        """
        self.n_bins = n_bins
    
    def _estimate_pdf(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate probability density using histogram"""
        hist, bin_edges = np.histogram(data, bins=self.n_bins, density=True)
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        # Normalize
        hist = hist / np.sum(hist)
        return hist, bin_edges
    
    def calculate(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate JSD between two 1D distributions.
        
        Args:
            X: Samples from first distribution
            Y: Samples from second distribution
            
        Returns:
            JSD value (0 to 1)
        """
        # Use common bin edges
        all_data = np.concatenate([X, Y])
        bin_edges = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        # Estimate PDFs
        hist_x, _ = np.histogram(X, bins=bin_edges, density=True)
        hist_y, _ = np.histogram(Y, bins=bin_edges, density=True)
        
        # Normalize
        hist_x = hist_x / np.sum(hist_x) + 1e-10
        hist_y = hist_y / np.sum(hist_y) + 1e-10
        
        # Calculate M = (P + Q) / 2
        m = 0.5 * (hist_x + hist_y)
        
        # Calculate JSD
        jsd = 0.5 * (
            np.sum(hist_x * np.log(hist_x / m)) +
            np.sum(hist_y * np.log(hist_y / m))
        )
        
        return np.sqrt(max(0, jsd))  # Return Jensen-Shannon distance
    
    def calculate_multivariate(self, X: np.ndarray, Y: np.ndarray,
                               n_projections: int = 100) -> float:
        """
        Calculate JSD for multivariate distributions using random projections.
        
        Args:
            X: Samples from first distribution (n, d)
            Y: Samples from second distribution (n, d)
            n_projections: Number of random projections
            
        Returns:
            Average JSD across projections
        """
        n_features = X.shape[1]
        jsd_values = []
        
        for _ in range(n_projections):
            # Random projection
            projection = np.random.randn(n_features)
            projection = projection / np.linalg.norm(projection)
            
            # Project data
            X_proj = X @ projection
            Y_proj = Y @ projection
            
            # Calculate JSD
            jsd = self.calculate(X_proj, Y_proj)
            jsd_values.append(jsd)
        
        return np.mean(jsd_values)


class DetectionMetrics:
    """
    Calculate detection performance metrics.
    
    Includes sensitivity, specificity, ROC analysis, and detection limits.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def calculate_basic_metrics(self, y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 scores: np.ndarray) -> Dict:
        """
        Calculate basic detection metrics.
        
        Args:
            y_true: True labels (0 = clean, 1 = contaminated)
            y_pred: Predicted labels
            scores: Anomaly scores
            
        Returns:
            Dictionary of metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # ROC-AUC
        auc = roc_auc_score(y_true, scores)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'auc_roc': auc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
        }
    
    def calculate_roc_metrics(self, y_true: np.ndarray,
                               scores: np.ndarray) -> Dict:
        """
        Calculate ROC curve metrics.
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            
        Returns:
            Dictionary of ROC metrics
        """
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'optimal_threshold': optimal_threshold,
            'youden_j': j_scores[optimal_idx],
        }
    
    def calculate_detection_limit(self, scores: np.ndarray,
                                   inoculum_levels: np.ndarray,
                                   y_true: np.ndarray) -> Dict:
        """
        Calculate detection limit (minimum detectable CFU/mL).
        
        Args:
            scores: Anomaly scores
            inoculum_levels: CFU/mL for each sample
            y_true: True labels
            
        Returns:
            Detection limit analysis
        """
        # Group by inoculum level
        levels = np.unique(inoculum_levels[y_true == 1])
        
        detection_results = []
        for level in levels:
            mask = (inoculum_levels == level) & (y_true == 1)
            level_scores = scores[mask]
            
            # Calculate detection rate at this level
            # Using threshold from clean data (95th percentile)
            clean_scores = scores[y_true == 0]
            threshold = np.percentile(clean_scores, 95)
            
            detection_rate = np.mean(level_scores > threshold)
            
            detection_results.append({
                'inoculum_level': level,
                'detection_rate': detection_rate,
                'n_samples': len(level_scores),
            })
        
        # Find detection limit (level with >= 90% detection rate)
        detection_limit = None
        for result in sorted(detection_results, key=lambda x: x['inoculum_level']):
            if result['detection_rate'] >= 0.90:
                detection_limit = result['inoculum_level']
                break
        
        return {
            'by_level': detection_results,
            'detection_limit_90': detection_limit,
            'target_detection_limit': self.config.target_detection_limit,
            'meets_target': detection_limit is not None and 
                           detection_limit <= self.config.target_detection_limit
        }
    
    def bootstrap_confidence_intervals(self, y_true: np.ndarray,
                                        scores: np.ndarray,
                                        n_iterations: int = 1000) -> Dict:
        """
        Calculate confidence intervals using bootstrapping.
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Confidence intervals for metrics
        """
        auc_scores = []
        sensitivity_scores = []
        specificity_scores = []
        
        n = len(y_true)
        
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            y_boot = y_true[indices]
            s_boot = scores[indices]
            
            # Skip if only one class
            if len(np.unique(y_boot)) < 2:
                continue
            
            # Calculate metrics
            try:
                auc = roc_auc_score(y_boot, s_boot)
                auc_scores.append(auc)
                
                # Find optimal threshold
                fpr, tpr, thresholds = roc_curve(y_boot, s_boot)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                threshold = thresholds[optimal_idx]
                
                y_pred = (s_boot >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_boot, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                sensitivity_scores.append(sensitivity)
                specificity_scores.append(specificity)
            except:
                continue
        
        return {
            'auc': {
                'mean': np.mean(auc_scores),
                'std': np.std(auc_scores),
                'ci_lower': np.percentile(auc_scores, 2.5),
                'ci_upper': np.percentile(auc_scores, 97.5),
            },
            'sensitivity': {
                'mean': np.mean(sensitivity_scores),
                'std': np.std(sensitivity_scores),
                'ci_lower': np.percentile(sensitivity_scores, 2.5),
                'ci_upper': np.percentile(sensitivity_scores, 97.5),
            },
            'specificity': {
                'mean': np.mean(specificity_scores),
                'std': np.std(specificity_scores),
                'ci_lower': np.percentile(specificity_scores, 2.5),
                'ci_upper': np.percentile(specificity_scores, 97.5),
            },
        }


class DetectionWindowSimulator:
    """
    Simulate 30-minute detection window for contamination detection.
    
    Models the time-dependent accumulation of contamination signatures
    and evaluates detection performance over time.
    """
    
    def __init__(self, growth_rate: float = 0.02,
                 detection_window: int = 30):
        """
        Initialize simulator.
        
        Args:
            growth_rate: Microbial growth rate (per minute)
            detection_window: Detection window in minutes
        """
        self.growth_rate = growth_rate
        self.detection_window = detection_window
    
    def simulate_growth(self, initial_cfus: np.ndarray,
                        time_minutes: int) -> np.ndarray:
        """
        Simulate microbial growth over time.
        
        Args:
            initial_cfus: Initial CFU/mL
            time_minutes: Time in minutes
            
        Returns:
            CFU/mL after growth
        """
        # Exponential growth: N(t) = N0 * e^(rt)
        return initial_cfus * np.exp(self.growth_rate * time_minutes)
    
    def simulate_spectral_evolution(self, base_spectra: np.ndarray,
                                     initial_cfus: np.ndarray,
                                     contamination_factor: float = 0.001) -> List[np.ndarray]:
        """
        Simulate spectral evolution over detection window.
        
        Args:
            base_spectra: Clean spectra
            initial_cfus: Initial contamination levels
            contamination_factor: Factor for contamination effect
            
        Returns:
            List of spectra at each time point
        """
        time_points = list(range(0, self.detection_window + 1, 5))  # Every 5 minutes
        spectra_over_time = []
        
        for t in time_points:
            # Calculate grown CFUs
            grown_cfus = self.simulate_growth(initial_cfus, t)
            
            # Simulate spectral changes
            # Contamination adds absorbance proportional to log(CFU)
            spectral_change = contamination_factor * np.log10(grown_cfus + 1)
            
            # Add to base spectrum (simplified model)
            evolved_spectra = base_spectra + spectral_change[:, np.newaxis] * np.random.randn(
                len(base_spectra), base_spectra.shape[1]
            ) * 0.01
            
            spectra_over_time.append(evolved_spectra)
        
        return spectra_over_time
    
    def evaluate_time_dependent_detection(self, detector,
                                           base_spectra: np.ndarray,
                                           initial_cfus: np.ndarray,
                                           y_true: np.ndarray) -> Dict:
        """
        Evaluate detection performance over time.
        
        Args:
            detector: Anomaly detector
            base_spectra: Clean base spectra
            initial_cfus: Initial contamination levels
            y_true: True labels
            
        Returns:
            Time-dependent detection metrics
        """
        time_points = list(range(0, self.detection_window + 1, 5))
        metrics_over_time = []
        
        spectra_evolution = self.simulate_spectral_evolution(base_spectra, initial_cfus)
        
        for t, spectra in zip(time_points, spectra_evolution):
            # Get predictions
            scores = detector.predict_proba(spectra)
            
            # Calculate metrics
            threshold = np.percentile(scores[y_true == 0], 95)
            y_pred = (scores >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_over_time.append({
                'time_minutes': t,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'detection_rate': np.mean(scores[y_true == 1] >= threshold),
            })
        
        # Find time to 90% detection
        time_to_90 = None
        for m in metrics_over_time:
            if m['detection_rate'] >= 0.90:
                time_to_90 = m['time_minutes']
                break
        
        return {
            'metrics_over_time': metrics_over_time,
            'time_to_90_detection': time_to_90,
            'meets_30min_target': time_to_90 is not None and time_to_90 <= 30,
        }


class ValidationPipeline:
    """
    Complete validation pipeline for contamination detection system.
    
    Integrates all validation metrics and generates comprehensive reports.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        self.mmd_calculator = MaximumMeanDiscrepancy(
            kernel=self.config.mmd_kernel,
            gamma=self.config.mmd_gamma
        )
        self.jsd_calculator = JensenShannonDivergence()
        self.metrics_calculator = DetectionMetrics(self.config)
        self.window_simulator = DetectionWindowSimulator(
            detection_window=self.config.detection_window_minutes
        )
    
    def validate_synthetic_data(self, real_spectra: np.ndarray,
                                 synthetic_spectra: np.ndarray) -> Dict:
        """
        Validate synthetic data quality.
        
        Args:
            real_spectra: Real spectra
            synthetic_spectra: Generated synthetic spectra
            
        Returns:
            Validation results
        """
        print("Validating synthetic data quality...")
        
        # Calculate MMD
        mmd, mmd_pvalue = self.mmd_calculator.calculate(real_spectra, synthetic_spectra)
        
        # Calculate JSD (multivariate)
        jsd = self.jsd_calculator.calculate_multivariate(real_spectra, synthetic_spectra)
        
        # Compare marginal distributions
        marginal_jsds = []
        for i in range(min(real_spectra.shape[1], 50)):  # Sample wavelengths
            jsd_marginal = self.jsd_calculator.calculate(
                real_spectra[:, i], synthetic_spectra[:, i]
            )
            marginal_jsds.append(jsd_marginal)
        
        results = {
            'mmd': mmd,
            'mmd_pvalue': mmd_pvalue,
            'jsd': jsd,
            'marginal_jsd_mean': np.mean(marginal_jsds),
            'marginal_jsd_std': np.std(marginal_jsds),
            'distributions_similar': mmd_pvalue > 0.05,  # Fail to reject null
        }
        
        return results
    
    def validate_detector(self, detector, X_test: np.ndarray,
                          y_test: np.ndarray,
                          inoculum_levels: np.ndarray) -> Dict:
        """
        Validate anomaly detector performance.
        
        Args:
            detector: Trained anomaly detector
            X_test: Test spectra
            y_test: True labels
            inoculum_levels: CFU/mL for each sample
            
        Returns:
            Validation results
        """
        print("Validating detector performance...")
        
        # Get predictions
        y_pred = detector.predict(X_test)
        scores = detector.predict_proba(X_test)
        
        # Basic metrics
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(y_test, y_pred, scores)
        
        # ROC metrics
        roc_metrics = self.metrics_calculator.calculate_roc_metrics(y_test, scores)
        
        # Detection limit
        detection_limit = self.metrics_calculator.calculate_detection_limit(
            scores, inoculum_levels, y_test
        )
        
        # Bootstrap confidence intervals
        ci = self.metrics_calculator.bootstrap_confidence_intervals(
            y_test, scores, self.config.n_bootstrap_iterations
        )
        
        # Check against targets
        meets_targets = {
            'sensitivity': basic_metrics['sensitivity'] >= self.config.target_sensitivity,
            'specificity': basic_metrics['specificity'] >= self.config.target_specificity,
            'auc': basic_metrics['auc_roc'] >= self.config.target_auc,
            'detection_limit': detection_limit['meets_target'],
        }
        
        results = {
            'basic_metrics': basic_metrics,
            'roc_metrics': roc_metrics,
            'detection_limit': detection_limit,
            'confidence_intervals': ci,
            'meets_targets': meets_targets,
            'all_targets_met': all(meets_targets.values()),
        }
        
        return results
    
    def validate_detection_window(self, detector, X_clean: np.ndarray,
                                   X_contaminated: np.ndarray,
                                   inoculum_levels: np.ndarray,
                                   y_test: np.ndarray) -> Dict:
        """
        Validate 30-minute detection window.
        
        Args:
            detector: Trained anomaly detector
            X_clean: Clean spectra
            X_contaminated: Contaminated spectra
            inoculum_levels: CFU/mL for each sample
            y_test: True labels
            
        Returns:
            Detection window validation results
        """
        print("Validating detection window...")
        
        # Simulate time-dependent detection
        results = self.window_simulator.evaluate_time_dependent_detection(
            detector, X_clean, inoculum_levels, y_test
        )
        
        return results
    
    def run_full_validation(self, detector, X_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             inoculum_levels: np.ndarray,
                             synthetic_spectra: Optional[np.ndarray] = None) -> Dict:
        """
        Run complete validation pipeline.
        
        Args:
            detector: Trained anomaly detector
            X_train: Training data (for synthetic validation)
            X_test: Test data
            y_test: True labels
            inoculum_levels: CFU/mL for each sample
            synthetic_spectra: Optional synthetic spectra
            
        Returns:
            Complete validation results
        """
        results = {}
        
        # Validate synthetic data if provided
        if synthetic_spectra is not None:
            results['synthetic_validation'] = self.validate_synthetic_data(
                X_train, synthetic_spectra
            )
        
        # Validate detector
        results['detector_validation'] = self.validate_detector(
            detector, X_test, y_test, inoculum_levels
        )
        
        # Validate detection window
        results['detection_window'] = self.validate_detection_window(
            detector, X_test[y_test == 0], X_test[y_test == 1],
            inoculum_levels[y_test == 1], y_test[y_test == 1]
        )
        
        # Summary
        results['summary'] = {
            'synthetic_data_valid': results.get('synthetic_validation', {}).get('distributions_similar', True),
            'detector_valid': results['detector_validation']['all_targets_met'],
            'detection_window_valid': results['detection_window']['meets_30min_target'],
            'overall_valid': (
                results.get('synthetic_validation', {}).get('distributions_similar', True) and
                results['detector_validation']['all_targets_met'] and
                results['detection_window']['meets_30min_target']
            ),
        }
        
        return results


def generate_validation_report(results: Dict, output_path: str = "validation_report.txt"):
    """
    Generate human-readable validation report.
    
    Args:
        results: Validation results dictionary
        output_path: Path to save report
    """
    report = []
    report.append("=" * 60)
    report.append("CONTAMINATION DETECTION SYSTEM - VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Synthetic Data Validation
    if 'synthetic_validation' in results:
        sv = results['synthetic_validation']
        report.append("SYNTHETIC DATA VALIDATION")
        report.append("-" * 40)
        report.append(f"MMD: {sv['mmd']:.6f} (p-value: {sv['mmd_pvalue']:.4f})")
        report.append(f"JSD: {sv['jsd']:.6f}")
        report.append(f"Distributions Similar: {sv['distributions_similar']}")
        report.append("")
    
    # Detector Validation
    dv = results['detector_validation']
    bm = dv['basic_metrics']
    report.append("DETECTOR PERFORMANCE")
    report.append("-" * 40)
    report.append(f"Sensitivity: {bm['sensitivity']:.4f} ({dv['confidence_intervals']['sensitivity']['ci_lower']:.4f} - {dv['confidence_intervals']['sensitivity']['ci_upper']:.4f})")
    report.append(f"Specificity: {bm['specificity']:.4f} ({dv['confidence_intervals']['specificity']['ci_lower']:.4f} - {dv['confidence_intervals']['specificity']['ci_upper']:.4f})")
    report.append(f"AUC-ROC: {bm['auc_roc']:.4f} ({dv['confidence_intervals']['auc']['ci_lower']:.4f} - {dv['confidence_intervals']['auc']['ci_upper']:.4f})")
    report.append(f"F1 Score: {bm['f1_score']:.4f}")
    report.append("")
    
    # Detection Limit
    dl = dv['detection_limit']
    report.append("DETECTION LIMIT")
    report.append("-" * 40)
    report.append(f"Target: {dl['target_detection_limit']} CFU/mL")
    report.append(f"Achieved: {dl['detection_limit_90']} CFU/mL" if dl['detection_limit_90'] else "Not achieved")
    report.append(f"Meets Target: {dl['meets_target']}")
    report.append("")
    
    # Detection Window
    dw = results['detection_window']
    report.append("DETECTION WINDOW")
    report.append("-" * 40)
    report.append(f"Time to 90% Detection: {dw['time_to_90_detection']} minutes" if dw['time_to_90_detection'] else ">30 minutes")
    report.append(f"Meets 30-min Target: {dw['meets_30min_target']}")
    report.append("")
    
    # Summary
    summary = results['summary']
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"Synthetic Data Valid: {summary['synthetic_data_valid']}")
    report.append(f"Detector Valid: {summary['detector_valid']}")
    report.append(f"Detection Window Valid: {summary['detection_window_valid']}")
    report.append(f"OVERALL VALID: {summary['overall_valid']}")
    report.append("")
    report.append("=" * 60)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {output_path}")
    
    return report_text


if __name__ == "__main__":
    # Test validation pipeline
    from data_simulation import UVVisSpectraGenerator
    from anomaly_detection import IsolationForestDetector, ModelConfig
    
    # Generate test data
    print("Generating test data...")
    generator = UVVisSpectraGenerator(seed=42)
    df = generator.generate_dataset(n_clean=500, n_contaminated_per_type=100, seed=42)
    
    # Prepare data
    wavelength_cols = [c for c in df.columns if c.startswith('abs_')]
    X = df[wavelength_cols].values
    y = df['label'].values
    inoculum = df['inoculum_level'].values
    
    # Split
    X_clean = X[y == 0]
    X_test = X
    y_test = y
    
    # Train detector
    print("Training detector...")
    config = ModelConfig()
    detector = IsolationForestDetector(config)
    detector.fit(X_clean)
    
    # Run validation
    print("\nRunning validation...")
    pipeline = ValidationPipeline()
    results = pipeline.run_full_validation(
        detector, X_clean, X_test, y_test, inoculum
    )
    
    # Generate report
    print("\nGenerating report...")
    generate_validation_report(results)
