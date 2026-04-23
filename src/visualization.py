"""
Visualization Utilities for Contamination Detection System

Publication-ready figures for:
- UV-Vis spectra visualization
- Anomaly detection results
- Model performance metrics
- Synthetic data validation
- Detection limit analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import seaborn as sns
from scipy import stats

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
COLOR_PALETTE = {
    'clean': '#2E86AB',
    'contaminated': '#A23B72',
    'e_coli': '#F18F01',
    'b_subtilis': '#C73E1D',
    'p_aeruginosa': '#6A994E',
    'c_albicans': '#BC4749',
    'a_niger': '#7209B7',
    'mycoplasma': '#3A0CA3',
    'anomaly': '#EF476F',
    'normal': '#06D6A0',
}


class SpectraVisualizer:
    """Visualize UV-Vis spectra"""
    
    @staticmethod
    def plot_single_spectrum(wavelengths: np.ndarray, absorbance: np.ndarray,
                             title: str = "UV-Vis Spectrum",
                             highlight_regions: Optional[List[Tuple[float, float]]] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a single UV-Vis spectrum.
        
        Args:
            wavelengths: Wavelength array (nm)
            absorbance: Absorbance values
            title: Plot title
            highlight_regions: List of (start, end) wavelength regions to highlight
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(wavelengths, absorbance, linewidth=2, color=COLOR_PALETTE['clean'])
        
        # Highlight regions
        if highlight_regions:
            for start, end in highlight_regions:
                ax.axvspan(start, end, alpha=0.2, color='yellow', 
                          label=f'{start}-{end} nm')
        
        # Key wavelength markers
        key_wavelengths = [260, 280, 430, 600]
        for wl in key_wavelengths:
            ax.axvline(wl, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(wl, ax.get_ylim()[1] * 0.95, f'{wl} nm', 
                   rotation=90, va='top', ha='center', fontsize=9)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance (AU)')
        ax.set_title(title)
        ax.set_xlim(200, 800)
        
        if highlight_regions:
            ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_multiple_spectra(wavelengths: np.ndarray,
                               spectra_dict: Dict[str, np.ndarray],
                               title: str = "UV-Vis Spectra Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple spectra for comparison.
        
        Args:
            wavelengths: Wavelength array
            spectra_dict: Dictionary of {label: spectra_array}
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = list(COLOR_PALETTE.values())
        
        for (label, spectra), color in zip(spectra_dict.items(), colors):
            if len(spectra.shape) == 1:
                # Single spectrum
                ax.plot(wavelengths, spectra, label=label, color=color, linewidth=2)
            else:
                # Multiple spectra - plot mean and std
                mean_spec = np.mean(spectra, axis=0)
                std_spec = np.std(spectra, axis=0)
                
                ax.plot(wavelengths, mean_spec, label=label, color=color, linewidth=2)
                ax.fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec,
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance (AU)')
        ax.set_title(title)
        ax.set_xlim(200, 800)
        ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_spectra_by_contaminant(df: pd.DataFrame,
                                     wavelengths: np.ndarray,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spectra grouped by contaminant type.
        
        Args:
            df: DataFrame with spectra and contaminant_type column
            wavelengths: Wavelength array
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        contaminant_types = df['contaminant_type'].unique()
        
        for i, cont_type in enumerate(contaminant_types):
            mask = df['contaminant_type'] == cont_type
            spectra = df.loc[mask, [f'abs_{int(w)}' for w in wavelengths]].values
            
            mean_spec = np.mean(spectra, axis=0)
            std_spec = np.std(spectra, axis=0)
            
            color = COLOR_PALETTE.get(cont_type.lower(), COLOR_PALETTE['contaminated'])
            
            ax.plot(wavelengths, mean_spec, label=cont_type, color=color, linewidth=2)
            ax.fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec,
                           alpha=0.15, color=color)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Absorbance (AU)', fontsize=14)
        ax.set_title('UV-Vis Spectra by Contaminant Type', fontsize=16)
        ax.set_xlim(200, 800)
        ax.legend(loc='upper right', fontsize=11)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class AnomalyVisualizer:
    """Visualize anomaly detection results"""
    
    @staticmethod
    def plot_anomaly_scores(y_true: np.ndarray, scores: np.ndarray,
                            threshold: float,
                            title: str = "Anomaly Scores",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot anomaly scores with threshold.
        
        Args:
            y_true: True labels (0=clean, 1=contaminated)
            scores: Anomaly scores
            threshold: Decision threshold
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separate clean and contaminated
        clean_scores = scores[y_true == 0]
        cont_scores = scores[y_true == 1]
        
        # Plot distributions
        ax.hist(clean_scores, bins=50, alpha=0.6, label='Clean',
               color=COLOR_PALETTE['normal'], density=True)
        ax.hist(cont_scores, bins=50, alpha=0.6, label='Contaminated',
               color=COLOR_PALETTE['anomaly'], density=True)
        
        # Plot threshold
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold: {threshold:.4f}')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray,
                       title: str = "ROC Curve",
                       save_path: Optional[str] = None) -> Tuple[plt.Figure, Dict]:
        """
        Plot ROC curve with AUC.
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Tuple of (figure, metrics dict)
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc_value = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, color=COLOR_PALETTE['contaminated'], linewidth=2,
               label=f'ROC Curve (AUC = {auc_value:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        # Fill area under curve
        ax.fill_between(fpr, tpr, alpha=0.3, color=COLOR_PALETTE['contaminated'])
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_aspect('equal')
        
        metrics = {'fpr': fpr, 'tpr': tpr, 'auc': auc_value}
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, metrics
    
    @staticmethod
    def plot_precision_recall(y_true: np.ndarray, scores: np.ndarray,
                               title: str = "Precision-Recall Curve",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            title: Plot title
            save_path: Path to save figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        ap_score = average_precision_score(y_true, scores)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(recall, precision, color=COLOR_PALETTE['contaminated'], linewidth=2,
               label=f'PR Curve (AP = {ap_score:.4f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='lower left')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for classes
            save_path: Path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = ['Clean', 'Contaminated']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        
        # Add text
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=16)
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class DetectionLimitVisualizer:
    """Visualize detection limit analysis"""
    
    @staticmethod
    def plot_detection_by_level(detection_results: List[Dict],
                                 target_limit: float = 10,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detection rate by inoculum level.
        
        Args:
            detection_results: List of {inoculum_level, detection_rate, n_samples}
            target_limit: Target detection limit (CFU/mL)
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        levels = [r['inoculum_level'] for r in detection_results]
        rates = [r['detection_rate'] for r in detection_results]
        n_samples = [r['n_samples'] for r in detection_results]
        
        # Color by detection rate
        colors = [COLOR_PALETTE['normal'] if r >= 0.9 else COLOR_PALETTE['anomaly'] 
                 for r in rates]
        
        bars = ax.bar(levels, rates, color=colors, alpha=0.7, edgecolor='black')
        
        # Add target line
        ax.axhline(0.9, color='red', linestyle='--', linewidth=2,
                  label='90% Detection Threshold')
        ax.axvline(target_limit, color='green', linestyle=':', linewidth=2,
                  label=f'Target Limit ({target_limit} CFU/mL)')
        
        # Add sample counts
        for bar, n in zip(bars, n_samples):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'n={n}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Inoculum Level (CFU/mL)')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rate by Contamination Level')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_xscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_time_dependent_detection(metrics_over_time: List[Dict],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detection performance over time.
        
        Args:
            metrics_over_time: List of {time_minutes, sensitivity, specificity, detection_rate}
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        times = [m['time_minutes'] for m in metrics_over_time]
        sensitivity = [m['sensitivity'] for m in metrics_over_time]
        specificity = [m['specificity'] for m in metrics_over_time]
        detection_rate = [m['detection_rate'] for m in metrics_over_time]
        
        # Plot 1: Sensitivity and Specificity over time
        ax1 = axes[0]
        ax1.plot(times, sensitivity, 'o-', label='Sensitivity', 
                color=COLOR_PALETTE['contaminated'], linewidth=2)
        ax1.plot(times, specificity, 's-', label='Specificity',
                color=COLOR_PALETTE['clean'], linewidth=2)
        ax1.axhline(0.9, color='gray', linestyle='--', alpha=0.5,
                   label='90% Threshold')
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Performance')
        ax1.set_title('Detection Performance Over Time')
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        
        # Plot 2: Detection rate over time
        ax2 = axes[1]
        ax2.plot(times, detection_rate, 'o-', linewidth=2,
                color=COLOR_PALETTE['anomaly'])
        ax2.axhline(0.9, color='red', linestyle='--', linewidth=2,
                   label='90% Detection')
        ax2.axvline(30, color='green', linestyle=':', linewidth=2,
                   label='30-min Target')
        
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Detection Rate')
        ax2.set_title('Detection Rate Over Time')
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class SyntheticDataVisualizer:
    """Visualize synthetic data validation"""
    
    @staticmethod
    def plot_feature_overlap(real_data: np.ndarray, 
                             physics_data: np.ndarray,
                             ddpm_data: np.ndarray,
                             method: str = 'tsne',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize feature space overlap between real, physics-based, and DDPM data.
        
        Args:
            real_data: Real spectra
            physics_data: Physics-based synthetic spectra
            ddpm_data: DDPM generated spectra
            method: 'tsne' or 'pca'
            save_path: Path to save figure
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Combine data
        combined = np.vstack([real_data, physics_data, ddpm_data])
        labels = (['Real'] * len(real_data) + 
                  ['Physics-based'] * len(physics_data) + 
                  ['MH-DDPM'] * len(ddpm_data))
        
        # Dim reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        proj = reducer.fit_transform(combined)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'Real': '#2E86AB', 'Physics-based': '#A23B72', 'MH-DDPM': '#F18F01'}
        
        for label in ['Physics-based', 'MH-DDPM', 'Real']:
            mask = [l == label for l in labels]
            ax.scatter(proj[mask, 0], proj[mask, 1], label=label, 
                       color=colors[label], alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
            
        ax.set_title(f'Feature Space Overlap ({method.upper()})')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_wavelength_mean_difference(real_data: np.ndarray,
                                       synthetic_data: np.ndarray,
                                       wavelengths: np.ndarray,
                                       title: str = "Mean Absolute Difference by Wavelength",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot wavelength-wise mean-difference between real and synthetic data.
        
        Args:
            real_data: Real spectra
            synthetic_data: Synthetic spectra
            wavelengths: Wavelength array
            save_path: Path to save figure
        """
        mean_real = np.mean(real_data, axis=0)
        mean_synth = np.mean(synthetic_data, axis=0)
        
        diff = mean_real - mean_synth
        abs_diff = np.abs(diff)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top: Mean spectra
        ax1.plot(wavelengths, mean_real, label='Real (Mean)', color='#2E86AB', linewidth=2)
        ax1.plot(wavelengths, mean_synth, label='Synthetic (Mean)', color='#F18F01', linestyle='--', linewidth=2)
        ax1.set_ylabel('Absorbance (AU)')
        ax1.set_title('Mean Spectra Comparison')
        ax1.legend()
        
        # Bottom: Difference
        ax2.fill_between(wavelengths, 0, diff, where=(diff >= 0), color='green', alpha=0.3, label='Real > Synth')
        ax2.fill_between(wavelengths, 0, diff, where=(diff < 0), color='red', alpha=0.3, label='Synth > Real')
        ax2.plot(wavelengths, diff, color='black', linewidth=1, alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Difference (Real - Synth)')
        ax2.set_title(title)
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_distribution_comparison(real_data: np.ndarray,
                                      synthetic_data: np.ndarray,
                                      feature_names: Optional[List[str]] = None,
                                      n_features: int = 9,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare real and synthetic data distributions.
        
        Args:
            real_data: Real spectra
            synthetic_data: Synthetic spectra
            feature_names: Feature names
            n_features: Number of features to show
            save_path: Path to save figure
        """
        n_features = min(n_features, real_data.shape[1])
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Select features to show (evenly spaced)
        indices = np.linspace(0, real_data.shape[1] - 1, n_features, dtype=int)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Plot distributions
            ax.hist(real_data[:, idx], bins=30, alpha=0.6, label='Real',
                   color=COLOR_PALETTE['clean'], density=True)
            ax.hist(synthetic_data[:, idx], bins=30, alpha=0.6, label='Synthetic',
                   color=COLOR_PALETTE['contaminated'], density=True)
            
            wl = 200 + idx if feature_names is None else feature_names[idx]
            ax.set_title(f'Feature {wl}')
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_pca_comparison(real_data: np.ndarray,
                            synthetic_data: np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare real and synthetic data in PCA space.
        
        Args:
            real_data: Real spectra
            synthetic_data: Synthetic spectra
            save_path: Path to save figure
        """
        from sklearn.decomposition import PCA
        
        # Combine data for PCA
        combined = np.vstack([real_data, synthetic_data])
        labels = ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot
        ax.scatter(pca_result[:len(real_data), 0], pca_result[:len(real_data), 1],
                  alpha=0.5, label='Real', color=COLOR_PALETTE['clean'], s=50)
        ax.scatter(pca_result[len(real_data):, 0], pca_result[len(real_data):, 1],
                  alpha=0.5, label='Synthetic', color=COLOR_PALETTE['contaminated'], s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA Comparison: Real vs Synthetic Data')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CompositeVisualizer:
    """Create composite figures for publications"""
    
    @staticmethod
    def create_publication_figure(df: pd.DataFrame,
                                   wavelengths: np.ndarray,
                                   validation_results: Dict,
                                   save_path: str = "figures/publication_figure.png") -> plt.Figure:
        """
        Create a multi-panel publication figure.
        
        Args:
            df: DataFrame with spectra
            wavelengths: Wavelength array
            validation_results: Validation results dictionary
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Spectra by contaminant type (top left, 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        contaminant_types = df['contaminant_type'].unique()
        for cont_type in contaminant_types:
            mask = df['contaminant_type'] == cont_type
            spectra = df.loc[mask, [f'abs_{int(w)}' for w in wavelengths]].values
            mean_spec = np.mean(spectra, axis=0)
            
            color = COLOR_PALETTE.get(cont_type.lower(), 'gray')
            ax1.plot(wavelengths, mean_spec, label=cont_type, color=color, linewidth=2)
        
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Absorbance (AU)')
        ax1.set_title('(A) UV-Vis Spectra by Contaminant Type')
        ax1.set_xlim(200, 800)
        ax1.legend(loc='upper right', fontsize=9)
        
        # Panel B: Anomaly score distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        if 'anomaly_scores' in df.columns:
            clean_scores = df.loc[df['label'] == 0, 'anomaly_scores']
            cont_scores = df.loc[df['label'] == 1, 'anomaly_scores']
            
            ax2.hist(clean_scores, bins=30, alpha=0.6, label='Clean',
                    color=COLOR_PALETTE['normal'], density=True)
            ax2.hist(cont_scores, bins=30, alpha=0.6, label='Contaminated',
                    color=COLOR_PALETTE['anomaly'], density=True)
            ax2.set_xlabel('Anomaly Score')
            ax2.set_ylabel('Density')
            ax2.set_title('(B) Anomaly Score Distribution')
            ax2.legend()
        
        # Panel C: ROC Curve (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if 'roc_fpr' in validation_results and 'roc_tpr' in validation_results:
            ax3.plot(validation_results['roc_fpr'], validation_results['roc_tpr'],
                    color=COLOR_PALETTE['contaminated'], linewidth=2)
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title(f'(C) ROC Curve (AUC={validation_results.get("auc", 0):.3f})')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        # Panel D: Detection by level (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'detection_by_level' in validation_results:
            det_results = validation_results['detection_by_level']
            levels = [r['inoculum_level'] for r in det_results]
            rates = [r['detection_rate'] for r in det_results]
            
            ax4.bar(levels, rates, color=COLOR_PALETTE['contaminated'], alpha=0.7)
            ax4.axhline(0.9, color='red', linestyle='--')
            ax4.set_xlabel('Inoculum Level (CFU/mL)')
            ax4.set_ylabel('Detection Rate')
            ax4.set_title('(D) Detection Rate by Level')
            ax4.set_xscale('log')
            ax4.set_ylim(0, 1.05)
        
        # Panel E: Time-dependent detection (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        if 'time_dependent' in validation_results:
            time_metrics = validation_results['time_dependent']
            times = [m['time_minutes'] for m in time_metrics]
            det_rates = [m['detection_rate'] for m in time_metrics]
            
            ax5.plot(times, det_rates, 'o-', linewidth=2)
            ax5.axhline(0.9, color='red', linestyle='--')
            ax5.axvline(30, color='green', linestyle=':')
            ax5.set_xlabel('Time (minutes)')
            ax5.set_ylabel('Detection Rate')
            ax5.set_title('(E) Detection Over Time')
            ax5.set_ylim(0, 1.05)
        
        # Panel F: Confusion Matrix (bottom center)
        ax6 = fig.add_subplot(gs[2, 1])
        
        if 'confusion_matrix' in validation_results:
            cm = validation_results['confusion_matrix']
            im = ax6.imshow(cm, cmap=plt.cm.Blues)
            plt.colorbar(im, ax=ax6)
            
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax6.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            ax6.set_xticks([0, 1])
            ax6.set_yticks([0, 1])
            ax6.set_xticklabels(['Clean', 'Contaminated'])
            ax6.set_yticklabels(['Clean', 'Contaminated'])
            ax6.set_xlabel('Predicted')
            ax6.set_ylabel('True')
            ax6.set_title('(F) Confusion Matrix')
        
        plt.suptitle('Biopharmaceutical Contamination Detection System',
                    fontsize=16, y=1.02)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_all_visualizations(df: pd.DataFrame,
                               wavelengths: np.ndarray,
                               validation_results: Dict,
                               output_dir: str = "figures") -> Dict[str, str]:
    """
    Create all standard visualizations.
    
    Args:
        df: DataFrame with spectra
        wavelengths: Wavelength array
        validation_results: Validation results
        output_dir: Output directory
        
    Returns:
        Dictionary of saved figure paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Spectra visualizations
    print("Creating spectra visualizations...")
    
    fig = SpectraVisualizer.plot_spectra_by_contaminant(
        df, wavelengths, str(output_dir / "spectra_by_contaminant.png")
    )
    saved_paths['spectra_by_contaminant'] = str(output_dir / "spectra_by_contaminant.png")
    plt.close(fig)
    
    # Anomaly detection visualizations
    print("Creating anomaly detection visualizations...")
    
    if 'anomaly_scores' in df.columns and 'threshold' in validation_results:
        fig = AnomalyVisualizer.plot_anomaly_scores(
            df['label'].values,
            df['anomaly_scores'].values,
            validation_results['threshold'],
            str(output_dir / "anomaly_scores.png")
        )
        saved_paths['anomaly_scores'] = str(output_dir / "anomaly_scores.png")
        plt.close(fig)
    
    # ROC curve
    if 'anomaly_scores' in df.columns:
        fig, _ = AnomalyVisualizer.plot_roc_curve(
            df['label'].values,
            df['anomaly_scores'].values,
            str(output_dir / "roc_curve.png")
        )
        saved_paths['roc_curve'] = str(output_dir / "roc_curve.png")
        plt.close(fig)
    
    # Detection limit
    print("Creating detection limit visualizations...")
    
    if 'detection_limit' in validation_results:
        fig = DetectionLimitVisualizer.plot_detection_by_level(
            validation_results['detection_limit']['by_level'],
            save_path=str(output_dir / "detection_by_level.png")
        )
        saved_paths['detection_by_level'] = str(output_dir / "detection_by_level.png")
        plt.close(fig)
    
    # Time-dependent detection
    if 'detection_window' in validation_results:
        fig = DetectionLimitVisualizer.plot_time_dependent_detection(
            validation_results['detection_window']['metrics_over_time'],
            save_path=str(output_dir / "detection_over_time.png")
        )
        saved_paths['detection_over_time'] = str(output_dir / "detection_over_time.png")
        plt.close(fig)
    
    # Publication figure
    print("Creating publication figure...")
    
    fig = CompositeVisualizer.create_publication_figure(
        df, wavelengths, validation_results,
        str(output_dir / "publication_figure.png")
    )
    saved_paths['publication_figure'] = str(output_dir / "publication_figure.png")
    plt.close(fig)
    
    print(f"\nSaved {len(saved_paths)} figures to {output_dir}")
    
    return saved_paths


if __name__ == "__main__":
    # Test visualizations
    from data_simulation import UVVisSpectraGenerator
    
    print("Generating test data for visualization...")
    generator = UVVisSpectraGenerator(seed=42)
    df = generator.generate_dataset(n_clean=200, n_contaminated_per_type=50, seed=42)
    
    wavelengths = np.arange(200, 801, 1)
    
    # Create output directory
    output_dir = Path("figures/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test spectra visualization
    print("Testing spectra visualization...")
    fig = SpectraVisualizer.plot_spectra_by_contaminant(
        df, wavelengths, str(output_dir / "test_spectra.png")
    )
    plt.close(fig)
    
    print(f"Test figures saved to {output_dir}")
