#!/usr/bin/env python3
"""
Main Pipeline for Biopharmaceutical Contamination Detection System

Complete end-to-end pipeline for:
1. Data generation and simulation
2. Feature extraction
3. Anomaly detection model training
4. MH-DDPM synthetic data generation
5. Validation and metrics calculation
6. Visualization and reporting

Usage:
    python run_pipeline.py --config config/pipeline_config.yaml
    python run_pipeline.py --demo  # Run with default settings
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_simulation import UVVisSpectraGenerator, ContaminantType, generate_sample_dataset
from feature_extraction import SpectralFeatureExtractor, extract_features_from_dataset
from anomaly_detection import (
    ModelConfig, IsolationForestDetector, OneClassSVMDetector,
    AutoencoderDetector, EnsembleAnomalyDetector, evaluate_detector
)
from mh_ddpm import DDPMConfig, MHDDPM, SyntheticDataGenerator
from validation import (
    ValidationConfig, ValidationPipeline, generate_validation_report
)
from visualization import create_all_visualizations


class ContaminationDetectionPipeline:
    """
    Main pipeline class for contamination detection system.
    
    Orchestrates all components from data generation to validation.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "output"):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration dictionary
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize components
        self.wavelengths = np.arange(
            config['data']['wavelength_start'],
            config['data']['wavelength_end'] + 1,
            config['data']['wavelength_step']
        )
        
        self.results = {}
    
    def _setup_logging(self):
        """Configure logging"""
        log_path = self.output_dir / "pipeline.log"
        logger.remove()
        logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {message}")
        logger.add(log_path, level="DEBUG", rotation="10 MB")
        self.logger = logger
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config.get('seed', 42)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def run_data_generation(self) -> pd.DataFrame:
        """
        Run data generation step.
        
        Returns:
            Generated DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Data Generation")
        logger.info("=" * 60)
        
        config = self.config['data']
        
        # Initialize generator
        generator = UVVisSpectraGenerator(seed=self.config['seed'])
        
        # Generate dataset
        logger.info(f"Generating {config['n_clean']} clean spectra...")
        logger.info(f"Generating {config['n_contaminated_per_type']} contaminated spectra per type...")
        
        df = generator.generate_dataset(
            n_clean=config['n_clean'],
            n_contaminated_per_type=config['n_contaminated_per_type'],
            inoculum_levels=config['inoculum_levels'],
            include_process_variation=config['include_process_variation'],
            seed=self.config['seed']
        )
        
        # Add batch and instrument effects
        if config.get('add_batch_effects', True):
            logger.info("Adding batch effects...")
            df = generator.add_batch_effect(df, n_batches=config.get('n_batches', 5))
        
        if config.get('add_instrument_variation', True):
            logger.info("Adding instrument variation...")
            df = generator.add_instrument_variation(df, n_instruments=config.get('n_instruments', 3))
        
        # Save raw data
        raw_path = self.output_dir / "data" / "raw_spectra.csv"
        df.to_csv(raw_path, index=False)
        logger.info(f"Raw data saved to {raw_path}")
        
        # Log statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts().to_dict()}")
        
        self.results['data_generation'] = {
            'n_samples': len(df),
            'n_clean': (df['label'] == 0).sum(),
            'n_contaminated': (df['label'] == 1).sum(),
            'n_features': len(self.wavelengths),
            'raw_data_path': str(raw_path),
        }
        
        return df
    
    def run_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run feature extraction step.
        
        Args:
            df: Raw spectra DataFrame
            
        Returns:
            Features DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Extraction")
        logger.info("=" * 60)
        
        # Initialize extractor
        extractor = SpectralFeatureExtractor(wavelengths=self.wavelengths)
        
        # Extract features
        logger.info("Extracting spectral features...")
        features_df = extractor.extract_all_features(
            df,
            include_derivatives=self.config['features'].get('include_derivatives', True),
            include_statistical=self.config['features'].get('include_statistical', True)
        )
        
        # Save features
        features_path = self.output_dir / "data" / "features.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Features saved to {features_path}")
        
        # Log feature statistics
        feature_cols = [c for c in features_df.columns if not c.startswith('spectrum_id')]
        logger.info(f"Extracted {len(feature_cols)} features")
        
        self.results['feature_extraction'] = {
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'features_path': str(features_path),
        }
        
        return features_df
    
    def run_model_training(self, features_df: pd.DataFrame) -> Dict:
        """
        Run anomaly detection model training.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Anomaly Detection Model Training")
        logger.info("=" * 60)
        
        # Prepare data
        feature_cols = self.results['feature_extraction']['feature_columns']
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Split: train on clean only
        X_clean = X[y == 0]
        X_test = X
        y_test = y
        
        logger.info(f"Training data: {len(X_clean)} clean samples")
        logger.info(f"Test data: {len(X_test)} samples ({sum(y_test)} contaminated)")
        
        # Initialize model config
        model_config = ModelConfig(
            random_state=self.config['seed'],
            contamination=self.config['models'].get('contamination', 0.01)
        )
        
        trained_models = {}
        evaluation_results = {}
        
        # Train Isolation Forest
        if self.config['models'].get('train_iforest', True):
            logger.info("\nTraining Isolation Forest...")
            iforest = IsolationForestDetector(model_config)
            iforest.fit(X_clean)
            trained_models['iforest'] = iforest
            
            # Evaluate
            metrics = evaluate_detector(iforest, X_test, y_test, "Isolation Forest")
            evaluation_results['iforest'] = metrics
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            
            # Save model
            model_path = self.output_dir / "models" / "iforest.pkl"
            iforest.save(model_path)
        
        # Train One-Class SVM
        if self.config['models'].get('train_ocsvm', True):
            logger.info("\nTraining One-Class SVM...")
            ocsvm = OneClassSVMDetector(model_config)
            ocsvm.fit(X_clean)
            trained_models['ocsvm'] = ocsvm
            
            # Evaluate
            metrics = evaluate_detector(ocsvm, X_test, y_test, "One-Class SVM")
            evaluation_results['ocsvm'] = metrics
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            
            # Save model
            model_path = self.output_dir / "models" / "ocsvm.pkl"
            ocsvm.save(model_path)
        
        # Train Deep Autoencoder
        if self.config['models'].get('train_autoencoder', True):
            logger.info("\nTraining Deep Autoencoder...")
            ae_config = ModelConfig(
                ae_epochs=self.config['models'].get('ae_epochs', 100),
                ae_batch_size=self.config['models'].get('ae_batch_size', 64),
                ae_latent_dim=self.config['models'].get('ae_latent_dim', 32),
            )
            ae = AutoencoderDetector(X_clean.shape[1], ae_config)
            ae.fit(X_clean, verbose=False)
            trained_models['autoencoder'] = ae
            
            # Evaluate
            metrics = evaluate_detector(ae, X_test, y_test, "Autoencoder")
            evaluation_results['autoencoder'] = metrics
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            
            # Save model
            model_path = self.output_dir / "models" / "autoencoder.pt"
            ae.save(model_path)
        
        # Train Ensemble (optional)
        if self.config['models'].get('train_ensemble', False):
            logger.info("\nTraining Ensemble Detector...")
            ensemble = EnsembleAnomalyDetector(X_clean.shape[1], model_config)
            ensemble.fit(X_clean, verbose=False)
            trained_models['ensemble'] = ensemble
            
            # Evaluate
            ensemble_scores = ensemble.predict_proba(X_test)
            ensemble_preds = ensemble.predict(X_test)
            metrics = evaluate_detector(ensemble, X_test, y_test, "Ensemble")
            evaluation_results['ensemble'] = metrics
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            
            # Save model
            model_path = self.output_dir / "models" / "ensemble"
            ensemble.save(model_path)
        
        # Select best model
        best_model_name = max(evaluation_results.keys(), 
                             key=lambda k: evaluation_results[k]['roc_auc'])
        best_model = trained_models[best_model_name]
        
        logger.info(f"\nBest model: {best_model_name} (AUC: {evaluation_results[best_model_name]['roc_auc']:.4f})")
        
        self.results['model_training'] = {
            'models_trained': list(trained_models.keys()),
            'evaluation_results': {k: {'roc_auc': v['roc_auc'], 'f1': v['f1']} 
                                  for k, v in evaluation_results.items()},
            'best_model': best_model_name,
            'models_dir': str(self.output_dir / "models"),
        }
        
        # Store test data for validation
        self._test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'inoculum_levels': features_df['inoculum_level'].values,
            'features_df': features_df,
        }
        
        return {
            'models': trained_models,
            'best_model': best_model,
            'evaluation': evaluation_results,
        }
    
    def run_synthetic_data_generation(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Run MH-DDPM synthetic data generation.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Generated synthetic spectra
        """
        logger.info("=" * 60)
        logger.info("STEP 4: MH-DDPM Synthetic Data Generation")
        logger.info("=" * 60)
        
        config = self.config['mh_ddpm']
        
        # Prepare contaminated data for training
        df = features_df
        contaminated_mask = df['label'] == 1
        
        wavelength_cols = [f'abs_{int(w)}' for w in self.wavelengths]
        spectra = df.loc[contaminated_mask, wavelength_cols].values
        
        # Map contaminant types to indices
        contaminant_map = {
            'E_coli': 0, 'B_subtilis': 1, 'P_aeruginosa': 2,
            'C_albicans': 3, 'A_niger': 4, 'Mycoplasma': 5
        }
        contaminant_types = df.loc[contaminated_mask, 'contaminant_type'].map(
            lambda x: contaminant_map.get(x, 0)
        ).values
        
        # Map inoculum levels to indices
        inoculum_map = {10: 0, 25: 1, 50: 2, 100: 3, 250: 4, 500: 5, 1000: 6}
        inoculum_levels = df.loc[contaminated_mask, 'inoculum_level'].map(
            lambda x: inoculum_map.get(x, 0)
        ).values
        
        logger.info(f"Training on {len(spectra)} contaminated spectra")
        
        # Initialize DDPM config
        ddpm_config = DDPMConfig(
            input_dim=spectra.shape[1],
            n_timesteps=config.get('n_timesteps', 100),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 64),
            hidden_dim=config.get('hidden_dim', 256),
        )
        
        # Initialize and train model
        logger.info("Training MH-DDPM model...")
        model = MHDDPM(ddpm_config)
        history = model.fit(spectra, contaminant_types, inoculum_levels, verbose=False)
        
        # Save model
        model_path = self.output_dir / "models" / "mh_ddpm.pt"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Generate synthetic data
        logger.info("Generating synthetic spectra...")
        synth_generator = SyntheticDataGenerator(model)
        
        n_synthetic_per_class = config.get('n_synthetic_per_class', 50)
        synthetic_spectra, synth_contaminants, synth_inoculum = synth_generator.generate_dataset(
            n_per_class=n_synthetic_per_class,
            progress=False
        )
        
        logger.info(f"Generated {len(synthetic_spectra)} synthetic spectra")
        
        # Save synthetic data
        synthetic_path = self.output_dir / "data" / "synthetic_spectra.csv"
        synthetic_df = pd.DataFrame(
            synthetic_spectra,
            columns=wavelength_cols
        )
        synthetic_df['contaminant_type'] = [
            list(contaminant_map.keys())[i] for i in synth_contaminants
        ]
        synthetic_df['inoculum_level'] = [
            list(inoculum_map.keys())[i] for i in synth_inoculum
        ]
        synthetic_df['label'] = 1
        synthetic_df.to_csv(synthetic_path, index=False)
        logger.info(f"Synthetic data saved to {synthetic_path}")
        
        self.results['synthetic_generation'] = {
            'n_synthetic': len(synthetic_spectra),
            'synthetic_path': str(synthetic_path),
            'training_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'validation_loss': history['val_loss'][-1] if history['val_loss'] else None,
        }
        
        return synthetic_spectra
    
    def run_validation(self, best_model, synthetic_spectra: np.ndarray) -> Dict:
        """
        Run validation pipeline.
        
        Args:
            best_model: Best trained anomaly detector
            synthetic_spectra: Generated synthetic spectra
            
        Returns:
            Validation results
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Validation")
        logger.info("=" * 60)
        
        # Get test data
        X_test = self._test_data['X_test']
        y_test = self._test_data['y_test']
        inoculum_levels = self._test_data['inoculum_levels']
        
        # Initialize validation config
        val_config = ValidationConfig(
            target_detection_limit=self.config['validation'].get('target_detection_limit', 10),
            target_sensitivity=self.config['validation'].get('target_sensitivity', 0.90),
            target_specificity=self.config['validation'].get('target_specificity', 0.95),
            n_bootstrap_iterations=self.config['validation'].get('n_bootstrap_iterations', 100),
        )
        
        # Run validation pipeline
        pipeline = ValidationPipeline(val_config)
        
        validation_results = pipeline.run_full_validation(
            best_model,
            X_test[y_test == 0],  # Clean training data
            X_test,
            y_test,
            inoculum_levels,
            synthetic_spectra
        )
        
        # Generate report
        report_path = self.output_dir / "reports" / "validation_report.txt"
        generate_validation_report(validation_results, str(report_path))
        
        # Store results for visualization
        self._validation_results = validation_results
        
        self.results['validation'] = {
            'summary': validation_results['summary'],
            'report_path': str(report_path),
        }
        
        return validation_results
    
    def run_visualization(self, features_df: pd.DataFrame) -> Dict[str, str]:
        """
        Run visualization generation.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Dictionary of saved figure paths
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Visualization")
        logger.info("=" * 60)
        
        # Add anomaly scores to DataFrame
        best_model = self.results['model_training']['best_model']
        # Note: In a real scenario, we'd load the best model and get scores
        
        # Prepare validation results for visualization
        val_results = self._validation_results
        
        # Add additional metrics to validation results
        if 'detector_validation' in val_results:
            dv = val_results['detector_validation']
            val_results['auc'] = dv['basic_metrics']['auc_roc']
            val_results['threshold'] = dv['roc_metrics']['optimal_threshold']
            val_results['roc_fpr'] = dv['roc_metrics']['fpr']
            val_results['roc_tpr'] = dv['roc_metrics']['tpr']
            val_results['confusion_matrix'] = None  # Would calculate from predictions
        
        # Create visualizations
        figure_paths = create_all_visualizations(
            features_df,
            self.wavelengths,
            val_results,
            str(self.output_dir / "figures")
        )
        
        self.results['visualization'] = {
            'figures': figure_paths,
            'figures_dir': str(self.output_dir / "figures"),
        }
        
        return figure_paths
    
    def run(self) -> Dict:
        """
        Run complete pipeline.
        
        Returns:
            Complete results dictionary
        """
        start_time = datetime.now()
        logger.info(f"Pipeline started at {start_time}")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        try:
            # Step 1: Data Generation
            df = self.run_data_generation()
            
            # Step 2: Feature Extraction
            features_df = self.run_feature_extraction(df)
            
            # Step 3: Model Training
            training_results = self.run_model_training(features_df)
            
            # Step 4: Synthetic Data Generation
            synthetic_spectra = self.run_synthetic_data_generation(features_df)
            
            # Step 5: Validation
            validation_results = self.run_validation(
                training_results['best_model'],
                synthetic_spectra
            )
            
            # Step 6: Visualization
            figure_paths = self.run_visualization(features_df)
            
            # Save complete results
            end_time = datetime.now()
            self.results['pipeline_info'] = {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'config': self.config,
            }
            
            results_path = self.output_dir / "reports" / "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total duration: {(end_time - start_time).total_seconds():.2f} seconds")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"Validation summary: {self.results['validation']['summary']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def get_default_config() -> Dict:
    """Get default pipeline configuration"""
    return {
        'seed': 42,
        'data': {
            'wavelength_start': 200,
            'wavelength_end': 800,
            'wavelength_step': 1,
            'n_clean': 1000,
            'n_contaminated_per_type': 200,
            'inoculum_levels': [10, 25, 50, 100, 250, 500, 1000],
            'include_process_variation': True,
            'add_batch_effects': True,
            'add_instrument_variation': True,
            'n_batches': 5,
            'n_instruments': 3,
        },
        'features': {
            'include_derivatives': True,
            'include_statistical': True,
        },
        'models': {
            'contamination': 0.01,
            'train_iforest': True,
            'train_ocsvm': True,
            'train_autoencoder': True,
            'train_ensemble': False,
            'ae_epochs': 100,
            'ae_batch_size': 64,
            'ae_latent_dim': 32,
        },
        'mh_ddpm': {
            'n_timesteps': 100,
            'epochs': 100,
            'batch_size': 64,
            'hidden_dim': 256,
            'n_synthetic_per_class': 50,
        },
        'validation': {
            'target_detection_limit': 10,
            'target_sensitivity': 0.90,
            'target_specificity': 0.95,
            'n_bootstrap_iterations': 100,
        },
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Biopharmaceutical Contamination Detection Pipeline"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with reduced data for testing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif args.demo:
        config = get_default_config()
        # Reduce data for demo
        config['data']['n_clean'] = 200
        config['data']['n_contaminated_per_type'] = 50
        config['mh_ddpm']['epochs'] = 20
        config['mh_ddpm']['n_timesteps'] = 50
        config['models']['ae_epochs'] = 20
        config['validation']['n_bootstrap_iterations'] = 50
    else:
        config = get_default_config()
    
    # Run pipeline
    pipeline = ContaminationDetectionPipeline(config, args.output)
    results = pipeline.run()
    
    return results


if __name__ == "__main__":
    main()
