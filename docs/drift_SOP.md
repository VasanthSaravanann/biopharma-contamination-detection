**Drift Detection SOP & Gates**

Purpose
-------
Provide a standard operating procedure for detecting, quantifying, and reacting to data distribution drift across spectra and sensor features.

Method
------
1. On each model run, compute two-sample Kolmogorov–Smirnov (KS) test p-values for each feature comparing current batch to training baseline.
2. Compute `drift_acceptance_ratio = (#features with p>0.05) / (total_features)`.
3. Monitor rolling window of `drift_acceptance_ratio` (window = 7 runs).

Gates and actions
-----------------
- Green (ratio ≥ 0.75): Continue normal operation. No action.
- Amber (0.4 ≤ ratio < 0.75): Trigger investigation: run feature importance, check sensors health, increase manual sampling frequency.
- Red (ratio < 0.4): Suspend automated alerts (fail-open), escalate to on-call engineer, start recalibration procedure and schedule shadow-mode re-training.

Recalibration procedure
----------------------
1. Collect representative samples from affected batches (≥3) and run plating/qPCR for ground truth.
2. Retrain detectors using combined baseline + verified new samples, keeping a strict time-based holdout for validation.
3. Re-run anti-shortcut tests and sanity checks; only promote updated model if performance metrics meet gating thresholds.

Logging & Audit
---------------
- Record all drift metrics and actions in `output/drift_logs/` with timestamps and operator IDs.
- Archive model version and input snapshot for each retrain.
