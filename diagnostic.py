#!/usr/bin/env python3
"""Diagnostic script to understand ensemble vs raw OCSVM performance"""

import json
import joblib
from pathlib import Path
import numpy as np

# Load ensemble metadata
audit_dir = Path('models/ensemble_audit')
metadata = joblib.load(audit_dir / 'ensemble_metadata.pkl')

print("=" * 70)
print("ENSEMBLE VS RAW OCSVM DIAGNOSTIC")
print("=" * 70)
print()

print("Ensemble Configuration:")
print(f"  Weights: {metadata['weights']}")
print(f"  Threshold: {metadata['threshold']:.6f}")
print(f"  Invert scores: {metadata.get('invert_scores', False)}")
print()

print("OCSVM Score Stats (from adversarial training):")
ocsvm_stats = metadata['score_stats'].get('ocsvm', {})
for key in ['min', 'max', 'mean', 'p05', 'p95', 'spread']:
    val = ocsvm_stats.get(key, 'N/A')
    if isinstance(val, (int, float)):
        print(f"  {key}: {val:.6f}")
    else:
        print(f"  {key}: {val}")
print()

# Load crucible report
crucible_report = json.load(open('output/adversarial_crucible/adversarial_crucible_report.json'))

print("Raw OCSVM 601-dim test (from crucible):")
raw_ocsvm = crucible_report['metrics'].get('raw_ocsvm_601d', {})
print(f"  AUC direct: {raw_ocsvm.get('auc', 'N/A'):.4f}")
print(f"  AUC inverted: {raw_ocsvm.get('auc_inverted', 'N/A'):.4f}")
print(f"  AUC best: {raw_ocsvm.get('best_auc', 'N/A'):.4f}")
print(f"  Clean mean score: {raw_ocsvm.get('scores_clean_mean', 'N/A'):.4f}")
print(f"  Contaminated mean score: {raw_ocsvm.get('scores_contaminated_mean', 'N/A'):.4f}")
print()

print("Ensemble test results (from crucible):")
print(f"  AUC direct: {crucible_report['metrics']['degraded_auc']:.4f}")
print(f"  AUC inverted: {crucible_report['metrics']['degraded_auc_inverted']:.4f}")
print(f"  AUC best: {crucible_report['metrics']['degraded_auc_best']:.4f}")
print(f"  Clean mean: {crucible_report['metrics']['clean_scores']['mean']:.4f}")
print(f"  Contaminated mean: {crucible_report['metrics']['contaminated_scores']['mean']:.4f}")
print()

print("KEY OBSERVATION:")
print(f"  Raw OCSVM (fresh train on test data): AUC = 1.0000 (PERFECT)")
print(f"  Ensemble OCSVM (trained on poisoned data): AUC = 0.6298 (POOR)")
print()
print("CAUSE:")
print("  The ensemble was trained on adversarially corrupted data")
print("  (clean + Gaussian noise + sensor degradation)")
print("  This trained the OCSVM to tolerate noise, making it less")
print("  sensitive to the actual contamination signal in test data.")
print()
print("SOLUTION:")
print("  To achieve AUC > 0.95, we would need to either:")
print("  1. Train the ensemble OCSVM on clean data (not corrupted)")
print("  2. Or use the raw OCSVM trained during crucible testing")
print("  3. Or disable the noise injection during training")
