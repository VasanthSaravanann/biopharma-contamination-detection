# Biopharmaceutical Contamination Detection — README

This repository provides code, analysis, and artifacts for an academic proof-of-concept: an inverse-variance weighted ensemble (Isolation Forest, Deep Autoencoder, One-Class SVM) for contamination detection using UV–Vis spectra and bioreactor process data. A physics-aware MH-DDPM is provided for synthetic augmentation used only in ablation experiments.

**Status:** Computational validation complete (real and physics-derived synthetic data). Not deployment-certified — prospective wet-lab validation required.

**Scope note:** The results in this repository derive from computational analyses on curated instrument datasets and physics-derived synthetic augmentations. Wet-lab experiments across multiple organisms and instruments are required for deployment qualification.

**Quick facts:**
- **Primary ensemble AUC (held-out test):** 0.94006 (see [output/results/run_bundle.json](output/results/run_bundle.json))
- **Baseline (OCSVM) AUC:** 0.93403 (see [output/results/run_bundle.json](output/results/run_bundle.json))
- **Latency (per-call mean):** 38.79 ms (per-sample mean 0.1939 ms) (see [output/results/run_bundle.json](output/results/run_bundle.json))
- **Drift acceptance ratio:** 0.423 (see [output/results/run_bundle.json](output/results/run_bundle.json))
- **Detection limit reported:** 10 CFU/mL (validated on synthetic + controlled real-data subsets; see [src/validation.py](src/validation.py#L50))

**Ablation note:** Using MH-DDPM for feature augmentation improved ablation AUC to ~0.984 (see [output/ablation/ablation_results.csv](output/ablation/ablation_results.csv)). MH-DDPM outputs are in [output/augmented/mh_ddpm/README.txt](output/augmented/mh_ddpm/README.txt). Crucially, MH-DDPM is _not_ used in the deployed scoring path — it's ablation-only (see [docs/mh_ddpm_provenance.md](docs/mh_ddpm_provenance.md)).

**What is included (local files):**
- **Code:** `src/` (all pipeline code)
- **Config:** `config/pipeline_config.yaml`
- **Processed metadata:** [data/processed/dataset_metadata.csv](data/processed/dataset_metadata.csv) (948 lines)
- **Run artifacts:** [output/results/run_bundle.json](output/results/run_bundle.json), [output/results/ensemble_report.md](output/results/ensemble_report.md)
- **Ablation artifacts:** [output/ablation/ablation_results.csv](output/ablation/ablation_results.csv), [output/ablation/ablation_metrics.png](output/ablation/ablation_metrics.png)
- **Holdout summary:** [output/holdout_results/holdout_summary.csv](output/holdout_results/holdout_summary.csv)

**What is NOT included in the repo (must be provided):**
- Raw instrument files and large datasets (UV-Vis spectra, full AMBR logs, raw LC-MS) are not tracked here by design. Expected locations in the project tree:
  - `data/Bacteria Contamination Work/` (UV–Vis files)
  - `data/FCIC_AMBR_05/` (AMBR sensor logs)
  - `data/ST001316/` etc. (metabolomics)

If you cannot include those files due to size or IP, place them in the above paths following the filename conventions in `data/processed/dataset_metadata.csv`.

**Reproducibility — exact, minimal steps**
1. Create Python venv and install:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Smoke test (quick):
   - `python run_smoke.py`
3. Full pipeline (produces `output/results/run_bundle.json`):
   - `python run_pipeline.py --config config/pipeline_config.yaml --experiment EColi_10CFU --output output`
4. Ablation (MH-DDPM experiments):
   - `python run_ablation.py`
   - or the placeholder ablation script: `bash scripts/run_mh_ddpm_ablation.sh 42 100 output/augmented/mh_ddpm`
5. Batch-level holdouts (uses `data/processed/dataset_metadata.csv` and `data/splits/holdout_definition.csv`):
   - `python scripts/run_batch_holdout.py --splits data/splits/holdout_definition.csv --out output/holdout_results`
6. Test suite:
   - `pytest tests/ -q` (expected: 24 passed, 3 skipped, 2 xfailed on this run)

Use `output/results/split_manifest.json` and `output/results/run_bundle.json` as provenance for any reported metric (seed, split indices, config snapshot).

**Key files to cite when reproducing a run**
- `output/results/run_bundle.json` — canonical metrics & config snapshot
- `output/results/split_manifest.json` — training/test indices used
- `data/processed/dataset_metadata.csv` — metadata linking spectra ↔ samples

**Design decisions & limitations (short)**
- Ensemble weighting strategy: inverse-variance weighting calibrated on clean data (see `config/pipeline_config.yaml`).
- MH-DDPM: physics-conditioned diffusion (Beer–Lambert + Rayleigh–Mie priors) used only for synthetic ablation; not part of the deployed scoring path.
- Primary limitation: although many experiments use real instrument baseline data, several core validation claims rely on physics-derived synthetic contamination — prospective wet-lab validation across instruments and organisms is required before production or clinical use.

**If you want to publish or present results**
- Reference `output/results/run_bundle.json` for the exact numeric claims and `output/results/split_manifest.json` for the exact split used. Attach `output/ablation/ablation_results.csv` for DDPM ablation claims.

**Contact & contribution**
- To reproduce, fork and open a PR. For data access questions, open an issue.

---
*This README is intentionally concise and reproducibility-focused. For more detail see `DEPLOYMENT_VALIDATION_REPORT.md`, `docs/mh_ddpm_provenance.md`, and `src/validation.py`.*
