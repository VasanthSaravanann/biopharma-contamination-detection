import sys
import argparse
import json
import yaml
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from src.data_processing import AmbrDatasetParser
from src.data_integration import UnifiedDataset, CECDataParser
from src.fusion import DataFuser
from src.feature_fusion import MultimodalFeatureExtractor, evaluate_roc_gate
from src.anomaly_detection import EnsembleAnomalyDetector, OneClassSVMDetector, ModelConfig, evaluate_detector
from src.metrics.latency import measure_latency
from src.metrics.drift import compute_drift_acceptance
from src.metrics.pca_baseline import compute_pca_baseline_auc

class ContaminationDetectionPipeline:
    def __init__(self, config: Dict[str, Any], output_dir: str = "output"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)

    def _save_artifact(self, name: str, data: Any):
        # Ensure path is always relative to output/results/
        path = self.output_dir / "results" / name
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)

    def run_fused_pipeline(self, args):
        """
        Manifest-driven real-data pipeline.
        Loads config, runs specified experiment, validates results, and saves artifacts.
        """
        manifest = {}
        metrics = {"ensemble_auc": None, "base_auc": None, "shuffled_auc": None}
        
        try:
            # 1. Preflight: Load config and resolve paths
            config_path = Path(args.config) if args.config else Path("config/pipeline_config.yaml")
            with open(config_path, 'r') as f:
                pipeline_config = yaml.safe_load(f)
            
            # Resolve data paths from config
            ambr_root = Path(pipeline_config['data']['ambr_root'])
            bacteria_root = Path(pipeline_config['data']['bacteria_root'])
            cec_root = Path(pipeline_config['data'].get('cec_root', 'data/CEC_04_2L-fermentation'))
            
            if not ambr_root.exists():
                raise FileNotFoundError(f"AMBR data root not found: {ambr_root}")
            if not bacteria_root.exists():
                raise FileNotFoundError(f"Bacteria data root not found: {bacteria_root}")
            # CEC root is optional - warn if not found but continue
            if not cec_root.exists():
                print(f"WARNING: CEC data not found at {cec_root} - proceeding with AMBR only", flush=True)
                cec_root = None
            
            # Get experiment from manifest
            experiment = args.experiment or "EColi_10CFU"
            organism, inoculum = self._parse_experiment_name(experiment, pipeline_config)
            
            print(f"Pipeline Configuration: {config_path}", flush=True)
            print(f"Running experiment: {experiment}", flush=True)
            print(f"  Organism: {organism}, Inoculum: {inoculum} CFU/mL", flush=True)
            
            # 2. Ingest real AMBR data
            parser = AmbrDatasetParser(str(ambr_root))
            ambr_df = parser.parse_sensor_files("00001/S")
            if ambr_df.empty:
                raise ValueError("No AMBR sensor data ingested")
            
            print(f"Ingested AMBR data: {ambr_df.shape[0]} samples, {ambr_df.shape[1]} sensors", flush=True)
            
            # 2a. Ingest CEC fermentation data if available and enabled
            baseline_df = ambr_df.copy()
            if cec_root and pipeline_config['data'].get('sources', {}).get('cec', {}).get('enabled', True):
                try:
                    print(f"Loading CEC fermentation data from {cec_root}...", flush=True)
                    cec_parser = CECDataParser(str(cec_root))
                    cec_dfs = {}
                    
                    # Try to parse Sartorius scale data
                    for scale in ['Sartorius_A5', 'Sartorius_A6']:
                        file_path = Path(cec_root) / f"{scale}-Table 1.csv"
                        if file_path.exists():
                            scale_df = cec_parser.parse_sartorius_data(str(file_path))
                            if not scale_df.empty:
                                # Rename columns to include scale prefix
                                scale_df = scale_df.rename(columns={col: f"{scale}_{col}" for col in scale_df.columns})
                                cec_dfs[scale] = scale_df
                    
                    if cec_dfs:
                        # Merge scales by joining on index (timestamps)
                        cec_df = None
                        for scale, df in cec_dfs.items():
                            if cec_df is None:
                                cec_df = df
                            else:
                                # Join on index, handling duplicate timestamps
                                cec_df = cec_df.join(df, how='outer')
                        
                        if cec_df is not None and not cec_df.empty:
                            # Forward fill any missing values
                            cec_df = cec_df.ffill().bfill()
                            
                            print(f"Ingested CEC data: {cec_df.shape[0]} samples, {cec_df.shape[1]} variables", flush=True)
                            
                            # Merge AMBR and CEC data as combined baseline
                            # Use concat with axis=0 to append rows (AMBR then CEC)
                            baseline_df = pd.concat([baseline_df, cec_df], axis=0, sort=False)
                            baseline_df = baseline_df.sort_index()
                            baseline_df = baseline_df.ffill().bfill()
                            
                            print(f"Merged baseline data: {baseline_df.shape[0]} total samples, {baseline_df.shape[1]} variables", flush=True)
                        else:
                            print(f"No CEC data successfully parsed, proceeding with AMBR only", flush=True)
                    else:
                        print(f"No CEC scale data found, proceeding with AMBR only", flush=True)
                except Exception as e:
                    print(f"WARNING: CEC data integration failed: {e}. Proceeding with AMBR only.", flush=True)
            
            # 3. Fuse with bacterial contamination spectra
            fuser = DataFuser(str(ambr_root), str(bacteria_root), seed=pipeline_config['reproducibility']['random_state'])
            fused_df = fuser.fuse(baseline_df, organism, inoculum, n_injections=3)
            
            if fused_df.empty:
                raise ValueError("Fusion produced empty dataframe")
            
            print(f"Fused data: {fused_df.shape[0]} samples, {fused_df.shape[1]} features", flush=True)
            print(f"Contaminated samples: {(fused_df['label'] == 1).sum()}", flush=True)
            
            # Save provenance metadata
            provenance = {
                "experiment": experiment,
                "organism": organism,
                "inoculum_cfu_ml": inoculum,
                "timestamp": datetime.now().isoformat(),
                "ambr_root": str(ambr_root),
                "bacteria_root": str(bacteria_root),
                "cec_root": str(cec_root) if cec_root else None,
                "baseline_sources": ["AMBR"],
                "n_samples": len(fused_df),
                "contaminated_count": int((fused_df['label'] == 1).sum()),
                "data_integration": "AMBR+CEC" if baseline_df.shape[0] > ambr_df.shape[0] else "AMBR_only"
            }
            
            # Update baseline sources if CEC was included
            if cec_root and baseline_df.shape[0] > ambr_df.shape[0]:
                provenance["baseline_sources"] = ["AMBR", "CEC_04_2L_Fermentation"]
            
            # 4. Feature extraction
            extractor = MultimodalFeatureExtractor(mode='fused')
            X = extractor.extract_features(fused_df)
            y = fused_df['label'].values
            
            if X.shape[0] < 50: raise ValueError("Preflight Failed: Insufficient samples")
            if X.shape[0] < pipeline_config['split']['min_samples_train']:
                raise ValueError(f"Insufficient samples: {X.shape[0]} < {pipeline_config['split']['min_samples_train']} required")
            
            print(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
            
            # 5. Data split using configured strategy
            split_cfg = pipeline_config['split']
            groups = np.arange(len(y)) // split_cfg['group_window_minutes']
            gss = GroupShuffleSplit(
                n_splits=split_cfg['n_splits'],
                test_size=split_cfg['test_size'],
                random_state=split_cfg['random_state']
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            manifest = {
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "seed": split_cfg['random_state'],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_clean_train": int((y_train == 0).sum()),
                "n_contaminated_train": int((y_train == 1).sum()),
                "n_clean_test": int((y_test == 0).sum()),
                "n_contaminated_test": int((y_test == 1).sum()),
                "provenance": provenance
            }
            self._save_artifact("split_manifest.json", manifest)
            print(f"Split: {len(train_idx)} train, {len(test_idx)} test samples", flush=True)
            
            # 6. Model training with ensemble configuration
            ensemble_cfg = pipeline_config['ensemble']
            iforest_params = ensemble_cfg['detectors']['isolation_forest']['params']
            ocsvm_params = ensemble_cfg['detectors']['one_class_svm']['params']
            ae_params = ensemble_cfg['detectors']['deep_autoencoder']['params']
            model_cfg = ModelConfig(
                contamination=iforest_params.get('contamination', 0.01),
                iforest_n_estimators=iforest_params.get('n_estimators', 200),
                iforest_max_samples=iforest_params.get('max_samples', 'auto'),
                iforest_max_features=iforest_params.get('max_features', 1.0),
                ocsvm_kernel=ocsvm_params.get('kernel', 'rbf'),
                ocsvm_gamma=ocsvm_params.get('gamma', 'auto'),
                ocsvm_nu=ocsvm_params.get('nu', 0.05),
                ae_epochs=ae_params.get('epochs', 100),
                ae_hidden_layers=ae_params.get('hidden_layers', [256, 128, 64]),
                ae_latent_dim=ae_params.get('latent_dim', 32),
                ae_dropout=ae_params.get('dropout', 0.2),
                ae_learning_rate=ae_params.get('learning_rate', 0.001),
                ae_batch_size=ae_params.get('batch_size', 64),
                ae_early_stopping_patience=ae_params.get('early_stopping_patience', 10),
                ae_weight_decay=ae_params.get('weight_decay', 1e-5)
            )
            
            # Train baseline OCSVM
            base_detector = OneClassSVMDetector(model_cfg)
            base_detector.fit(X_train[y_train == 0])
            base_auc = evaluate_detector(base_detector, X_test, y_test, "OCSVM_Baseline")['roc_auc']
            print(f"Baseline (OCSVM) AUC: {base_auc:.4f}", flush=True)

            # PCA reconstruction baseline
            try:
                pca_cfg = pipeline_config.get('baseline', {}).get('pca', {})
                pca_res = compute_pca_baseline_auc(
                    X_train[y_train == 0],
                    X_test,
                    y_test,
                    n_components=pca_cfg.get('n_components', None)
                )
                metrics['pca_baseline'] = pca_res
                print(f"PCA baseline AUC: {pca_res.get('pca_reconstruction_auc', 'NA')}", flush=True)
            except Exception as e:
                metrics['pca_baseline'] = {"error": str(e)}
                print(f"PCA baseline computation failed: {e}", flush=True)
            
            # Train ensemble detector
            detector = EnsembleAnomalyDetector(X.shape[1], model_cfg, use_conv=True)
            detector.fit(X_train[y_train == 0])
            
            # 7. Validation
            print("Evaluating ensemble ROC gate...", flush=True)
            ensemble_auc = evaluate_roc_gate(detector, X_test, y_test)
            print(f"Ensemble AUC: {ensemble_auc:.4f}", flush=True)
            
            print("Running anti-shortcut validation...", flush=True)
            X_test_shuffled = X_test.copy()
            np.random.shuffle(X_test_shuffled)
            ensemble_shuffled_auc = roc_auc_score(y_test, detector.predict_proba(X_test_shuffled))
            base_shuffled_auc = roc_auc_score(y_test, base_detector.predict_proba(X_test_shuffled))
            print(f"Shuffled AUC (Ensemble): {ensemble_shuffled_auc:.4f}", flush=True)
            print(f"Shuffled AUC (Baseline): {base_shuffled_auc:.4f}", flush=True)
            metrics = {
                "ensemble_auc": ensemble_auc,
                "base_auc": base_auc,
                "shuffled_auc": ensemble_shuffled_auc,
                "base_shuffled_auc": base_shuffled_auc,
            }

            # 7a. Measure inference latency (batch-based, aggregated per-sample)
            perf_cfg = pipeline_config.get('performance', {})
            try:
                lat_metrics = measure_latency(
                    detector,
                    X_test,
                    n_repeats=perf_cfg.get('latency_repeats', 10),
                    max_samples=perf_cfg.get('latency_max_samples', 200)
                )
                metrics['latency_ms'] = lat_metrics
                print(f"Latency (ms/sample) mean: {lat_metrics['per_sample_ms']['mean']:.4f}", flush=True)
            except Exception as e:
                metrics['latency_ms'] = {"error": str(e)}
                print(f"Latency measurement failed: {e}", flush=True)

            # 7b. Compute drift acceptance metric between training baseline and test
            try:
                drift_cfg = pipeline_config.get('validation', {}).get('drift', {})
                # Use training clean data as reference if available
                X_ref = X_train
                X_tgt = X_test
                drift_res = compute_drift_acceptance(
                    X_ref,
                    X_tgt,
                    alpha=drift_cfg.get('alpha', 0.05),
                )
                metrics['drift_acceptance'] = drift_res
                print(f"Drift acceptance ratio: {drift_res.get('acceptance_ratio', 'NA'):.3f}", flush=True)
            except Exception as e:
                metrics['drift_acceptance'] = {"error": str(e)}
                print(f"Drift computation failed: {e}", flush=True)
            
            print("Checking gates...", flush=True)
            val_cfg = pipeline_config['validation']
            require_ensemble_superiority = val_cfg.get('require_ensemble_superiority', False)
            if require_ensemble_superiority:
                selected_detector = "ensemble"
                selected_auc = ensemble_auc
                selected_shuffled_auc = ensemble_shuffled_auc
            else:
                if ensemble_auc >= base_auc:
                    selected_detector = "ensemble"
                    selected_auc = ensemble_auc
                    selected_shuffled_auc = ensemble_shuffled_auc
                else:
                    selected_detector = "baseline"
                    selected_auc = base_auc
                    selected_shuffled_auc = base_shuffled_auc

            if selected_shuffled_auc > (selected_auc * val_cfg['shuffled_auc_ratio_threshold']):
                raise ValueError(
                    f"Anti-Shortcut Failed ({selected_detector}): Shuffled {selected_shuffled_auc:.4f}"
                )

            gate_passed = selected_auc >= val_cfg['targets']['ensemble_auc_min']
            reason = "Pass" if gate_passed else (
                f"Fail: {selected_detector} AUC {selected_auc:.4f} < {val_cfg['targets']['ensemble_auc_min']:.2f}"
            )
            if not gate_passed: raise ValueError(reason)
            
            # 8. Save success bundle
            self._save_artifact("run_bundle.json", {
                "metrics": metrics,
                "gate": {"passed": True, "reason": "Pass"},
                "manifest": manifest,
                "timestamp": time.time(),
                "config_snapshot": {
                    "ensemble_strategy": ensemble_cfg['weighting_strategy'],
                    "split_strategy": split_cfg['strategy'],
                    "feature_mode": pipeline_config['features']['mode'],
                    "selected_detector": selected_detector,
                    "require_ensemble_superiority": require_ensemble_superiority,
                }
            })
            print("Audit-grade pipeline complete.", flush=True)
            
            # 9. Save models
            print("\nSynchronizing root models...", flush=True)
            root_model_dir = Path(pipeline_config['output']['models_dir'])
            root_model_dir.mkdir(parents=True, exist_ok=True)
            detector.save(str(root_model_dir / "ensemble_audit"))
            print(f"Audit models saved to {root_model_dir / 'ensemble_audit'}", flush=True)
            
        except Exception as e:
            # Atomic failure bundle with full context
            self._save_artifact("run_bundle.json", {
                "metrics": metrics,
                "gate": {"passed": False, "reason": str(e)},
                "manifest": manifest,
                "timestamp": time.time(),
                "error_type": type(e).__name__
            })
            print(f"Pipeline failed: {e}", flush=True)
            raise e
    
    def _parse_experiment_name(self, experiment: str, config: Dict[str, Any]):
        """
        Parse experiment name like 'EColi_10CFU' into organism and inoculum.
        """
        parts = experiment.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid experiment name: {experiment}. Expected format: Organism_InoculumCFU")
        
        organism_name = parts[0]
        inoculum_str = parts[1].replace('CFU', '')
        
        try:
            inoculum = int(inoculum_str)
        except ValueError:
            raise ValueError(f"Invalid inoculum level: {inoculum_str}")
        
        # Validate organism exists in config
        org_cfg = config['experiments']['organisms']
        if organism_name not in org_cfg:
            valid_organisms = list(org_cfg.keys())
            raise ValueError(f"Unknown organism: {organism_name}. Valid: {valid_organisms}")
        
        # Validate inoculum level is supported
        if inoculum not in config['experiments']['inoculum_levels']:
            valid_levels = config['experiments']['inoculum_levels']
            raise ValueError(f"Unsupported inoculum: {inoculum}. Valid: {valid_levels}")
        
        return organism_name, inoculum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="EColi_10CFU",
        help="Experiment name, e.g., 'EColi_10CFU', 'PAeruginosa_100CFU'"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Load config to get output directory if not overridden
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    output_dir = args.output if args.output != "output" else cfg['output']['root_dir']
    
    pipeline = ContaminationDetectionPipeline(cfg, output_dir)
    pipeline.run_fused_pipeline(args)

if __name__ == "__main__":
    main()
