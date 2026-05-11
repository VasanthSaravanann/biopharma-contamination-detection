# Measurement Protocol

This project records computational validation metrics in `output/results/run_bundle.json` so each run has an auditable artifact.

## What is measured

- ROC-AUC for the inverse-variance ensemble.
- ROC-AUC for the One-Class SVM baseline.
- PCA reconstruction-error ROC-AUC baseline.
- Anti-shortcut shuffled-score AUC checks.
- Latency in milliseconds per call and per sample.
- Drift acceptance ratio computed with a two-sample KS test over feature columns.

## How to reproduce

Run the pipeline with the same config and experiment:

```bash
python run_pipeline.py --experiment EColi_10CFU --config config/pipeline_config.yaml --output output
```

The resulting artifact is written to `output/results/run_bundle.json` and should be cited together with the commit SHA.

## Reporting conventions

- Report the exact config and commit used.
- Report means and variability when a metric is repeated across runs.
- Do not present synthetic-only metrics as wet-lab evidence.
- Keep ablation-only components, such as MH-DDPM, labeled as non-deployed.
