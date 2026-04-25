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
from src.fusion import DataFuser
from src.feature_fusion import MultimodalFeatureExtractor, evaluate_roc_gate
from src.anomaly_detection import EnsembleAnomalyDetector, OneClassSVMDetector, ModelConfig, evaluate_detector

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
        # Metadata context to persist on failure
        manifest = {}
        metrics = {"ensemble_auc": None, "base_auc": None, "shuffled_auc": None}
        
        try:
            # 1. Preflight
            root = Path("/run/media/sham/AI_/ai-stack/projects/biopharma-contamination-detection")
            if not root.exists():
                root = Path(__file__).resolve().parent

            ambr_candidate = root / "FCIC_AMBR_05" / "Data"
            if not ambr_candidate.exists():
                ambr_candidate = root / "data" / "FCIC_AMBR_05" / "Data"

            bacteria_candidate = root / "Bacteria Contamination Work"
            if not bacteria_candidate.exists():
                bacteria_candidate = root / "data" / "Bacteria Contamination Work"

            parser = AmbrDatasetParser(str(ambr_candidate))
            ambr_df = parser.parse_sensor_files("00001/S")
            fuser = DataFuser(str(ambr_candidate), str(bacteria_candidate))
            fused_df = fuser.fuse(ambr_df, "EColi", "10CFU")
            extractor = MultimodalFeatureExtractor(mode='fused')
            X = extractor.extract_features(fused_df)
            y = fused_df['label'].values
            if X.shape[0] < 50: raise ValueError("Preflight Failed: Insufficient samples")
            
            # 2. Data Build (Versioned)
            groups = np.arange(len(y)) // 60
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            manifest = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(), "seed": 42}
            self._save_artifact("split_manifest.json", manifest)
            
            # 3. Baseline & Model Training
            config = ModelConfig(ae_epochs=50)
            base_detector = OneClassSVMDetector(config)
            base_detector.fit(X_train[y_train == 0])
            base_auc = evaluate_detector(base_detector, X_test, y_test, "OCSVM_Baseline")['roc_auc']
            
            detector = EnsembleAnomalyDetector(X.shape[1], config, use_conv=True)
            detector.fit(X_train[y_train == 0])
            
            print("\nEvaluating ROC gate...", flush=True)
            ensemble_auc = evaluate_roc_gate(detector, X_test, y_test)
            print(f"Ensemble AUC: {ensemble_auc:.4f}", flush=True)
            
            print("Running anti-shortcut validation...", flush=True)
            X_test_shuffled = X_test.copy()
            np.random.shuffle(X_test_shuffled)
            shuffled_auc = roc_auc_score(y_test, detector.predict_proba(X_test_shuffled))
            print(f"Shuffled AUC: {shuffled_auc:.4f}", flush=True)
            metrics = {"ensemble_auc": ensemble_auc, "base_auc": base_auc, "shuffled_auc": shuffled_auc}
            
            # 4. Gates
            print("Checking gates...", flush=True)
            if shuffled_auc > (ensemble_auc * 0.8): raise ValueError(f"Anti-Shortcut Failed: Shuffled {shuffled_auc:.4f}")
            gate_passed = (ensemble_auc >= 0.95 and ensemble_auc > base_auc)
            reason = "Pass" if gate_passed else f"Fail: Ensemble AUC {ensemble_auc:.4f} <= Baseline/Threshold"
            if not gate_passed: raise ValueError(reason)
            
            # 5. Success Bundle
            self._save_artifact("run_bundle.json", {"metrics": metrics, "gate": {"passed": True, "reason": "Pass"}, "manifest": manifest, "timestamp": time.time()})
            print("Audit-grade pipeline complete.", flush=True)
            
            # 6. Synchronize root models
            print("\nSynchronizing root models...", flush=True)
            root_model_dir = Path("/run/media/sham/AI_/ai-stack/projects/biopharma-contamination-detection/models")
            root_model_dir.mkdir(parents=True, exist_ok=True)
            detector.save(str(root_model_dir / "ensemble_audit"))
            print(f"Audit models saved to {root_model_dir / 'ensemble_audit'}", flush=True)
            
        except Exception as e:
            # Atomic failure bundle with full context
            self._save_artifact("run_bundle.json", {
                "metrics": metrics, "gate": {"passed": False, "reason": str(e)}, 
                "manifest": manifest, "timestamp": time.time()
            })
            raise e
        print("Standard pipeline.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standard", "fused"], default="standard")
    args = parser.parse_args()
    
    c = ContaminationDetectionPipeline({}, "output")
    if args.mode == "fused":
        c.run_fused_pipeline(args)
    else:
        c.run()

if __name__ == "__main__":
    main()
